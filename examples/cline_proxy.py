"""
styxx <-> cline integration via a transparent Anthropic proxy.

Solves the question raised by Just Ral on 2026-04-25:
  "Since I use cline or claude extension how would [styxx] be able to detect
   anything? Especially if they use internal terminal not local terminal."

Cline runs as a Node.js extension, so styxx (Python) cannot hook directly.
The bridge: route Cline's Anthropic API traffic through this Python proxy.
The proxy:
  1. Accepts Anthropic Messages API requests on /v1/messages
  2. Forwards them to api.anthropic.com unchanged
  3. Wraps the streamed response with styxx.observe() to compute a
     cognitive fingerprint per response
  4. Returns the fingerprint as additional response headers + an
     optional X-Styxx-Profile JSON dump
  5. Logs each call to a JSONL file for offline analysis

Usage in Cline:
  Cline doesn't expose a custom Anthropic base URL natively, but it does
  let you set the OPENAI-compatible endpoint via env. For Anthropic-shaped
  routing, run this proxy + use the OpenAI-compat layer below.

  $ python cline_proxy.py
  Listening on http://localhost:18888

  In Cline settings → API:
    Provider: Anthropic
    Custom Base URL: http://localhost:18888

Requirements:
  pip install fastapi uvicorn httpx styxx

License: MIT (proxy code), CC-BY-4.0 (cognometric methodology).
Spec: doi:10.5281/zenodo.19746215  ·  Software: doi:10.5281/zenodo.19758619
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager

try:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
except ImportError:
    print("Missing deps. Install with:  pip install fastapi uvicorn httpx styxx")
    raise

import styxx


ANTHROPIC_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
LISTEN_HOST = os.environ.get("STYXX_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("STYXX_PROXY_PORT", "18888"))
LOG_FILE = pathlib.Path(os.environ.get("STYXX_PROXY_LOG", "./styxx_proxy.jsonl"))

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("styxx-proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"styxx proxy starting · forwarding to {ANTHROPIC_BASE}")
    log.info(f"styxx version: {styxx.__version__}")
    log.info(f"log file: {LOG_FILE.resolve()}")
    yield
    log.info("proxy shutting down")


app = FastAPI(lifespan=lifespan, title="styxx-cline-proxy", version="0.1.0")


@app.get("/")
async def root():
    return {
        "service": "styxx-cline-proxy",
        "version": "0.1.0",
        "styxx": styxx.__version__,
        "spec_doi": "10.5281/zenodo.19746215",
        "forwarding_to": ANTHROPIC_BASE,
        "log_file": str(LOG_FILE.resolve()),
        "endpoints": {
            "POST /v1/messages": "Anthropic Messages — proxied + cognometrically profiled",
            "GET /v1/styxx/fingerprints": "List recent fingerprints",
        },
    }


@app.get("/v1/styxx/fingerprints")
async def list_fingerprints(limit: int = 50):
    """Return the most recent N fingerprints from the log file."""
    if not LOG_FILE.exists():
        return {"fingerprints": [], "total": 0}
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    return {
        "fingerprints": [json.loads(l) for l in lines if l.strip()],
        "total": len(lines),
    }


@app.post("/v1/messages")
async def messages(request: Request):
    """Proxy Anthropic Messages API and attach a cognometric fingerprint."""
    started = time.time()
    headers_in = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }
    body_bytes = await request.body()

    # Inspect the request shape (we don't need the prompt for the fingerprint
    # but it's useful in the log).
    try:
        body_json = json.loads(body_bytes)
        model = body_json.get("model", "?")
        last_user_msg = next(
            (m["content"] for m in reversed(body_json.get("messages", []))
             if m.get("role") == "user"), "")
        if isinstance(last_user_msg, list):
            last_user_msg = " ".join(b.get("text", "") for b in last_user_msg
                                      if isinstance(b, dict))
        last_user_msg_excerpt = (str(last_user_msg) or "")[:300]
    except Exception:
        body_json = {}; model = "?"; last_user_msg_excerpt = ""

    # Forward to Anthropic
    async with httpx.AsyncClient(timeout=600.0) as client:
        upstream = await client.post(
            f"{ANTHROPIC_BASE}/v1/messages",
            content=body_bytes,
            headers=headers_in,
        )

    # Capture response body
    raw = upstream.content
    upstream_status = upstream.status_code
    upstream_headers = dict(upstream.headers)
    duration_s = time.time() - started

    # If non-200, pass through transparently
    if upstream_status >= 400:
        log.warning(f"upstream {upstream_status} on model={model}")
        return Response(
            content=raw,
            status_code=upstream_status,
            headers={k: v for k, v in upstream_headers.items()
                     if k.lower() not in ("transfer-encoding", "content-length")},
            media_type=upstream_headers.get("content-type", "application/json"),
        )

    # Extract response text for cognometric profiling
    try:
        resp_json = json.loads(raw)
        response_text = "".join(
            b.get("text", "") for b in resp_json.get("content", [])
            if b.get("type") == "text"
        )
    except Exception:
        resp_json = {}; response_text = ""

    # Run styxx.observe on the response
    try:
        vitals = styxx.observe({"text": response_text})
        if vitals is not None:
            cog = {
                "category": vitals.category,
                "confidence": float(vitals.confidence) if vitals.confidence is not None else None,
                "trust": float(vitals.trust_score) if vitals.trust_score is not None else None,
                "gate": vitals.gate,
                "coherence": float(vitals.coherence) if vitals.coherence is not None else None,
            }
        else:
            cog = {"error": "no vitals"}
    except Exception as e:
        cog = {"error": str(e)}

    # Log the call
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(duration_s, 3),
        "model": model,
        "user_excerpt": last_user_msg_excerpt,
        "response_len": len(response_text),
        "response_excerpt": response_text[:500],
        "styxx": cog,
        "spec_doi": "10.5281/zenodo.19746215",
        "implementation": f"styxx v{styxx.__version__}",
    }
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"log write failed: {e}")

    log.info(f"{model:<30}  cat={cog.get('category','?'):<12}  trust={cog.get('trust','?')}  {duration_s:.2f}s")

    # Build response with cognometric headers attached
    out_headers = {k: v for k, v in upstream_headers.items()
                   if k.lower() not in ("transfer-encoding", "content-length")}
    out_headers["X-Styxx-Category"] = str(cog.get("category", ""))
    out_headers["X-Styxx-Trust"] = str(cog.get("trust", ""))
    out_headers["X-Styxx-Gate"] = str(cog.get("gate", ""))
    out_headers["X-Styxx-Spec-DOI"] = "10.5281/zenodo.19746215"
    out_headers["X-Styxx-Profile-Json"] = json.dumps(cog)

    return Response(
        content=raw,
        status_code=upstream_status,
        headers=out_headers,
        media_type=upstream_headers.get("content-type", "application/json"),
    )


# Convenience: also serve OpenAI-shaped requests by translating to Anthropic.
# (Most editor extensions use the OpenAI shape even when calling Anthropic.)

@app.post("/v1/chat/completions")
async def openai_compat(request: Request):
    """OpenAI Chat Completions → Anthropic Messages translation, profiled."""
    started = time.time()
    headers_in = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding", "authorization")
    }
    body = await request.json()

    model = body.get("model", "claude-haiku-4-5")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 1024)

    # Translate to Anthropic shape
    anth_body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [m for m in messages if m.get("role") in ("user", "assistant")],
    }
    sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
    if sys_msg:
        anth_body["system"] = sys_msg

    # Use the same key that came in (OpenAI Authorization → Anthropic x-api-key)
    auth = request.headers.get("authorization", "")
    api_key = auth.split(" ", 1)[1] if " " in auth else os.environ.get("ANTHROPIC_API_KEY", "")
    headers_in["x-api-key"] = api_key
    headers_in["anthropic-version"] = "2023-06-01"
    headers_in["content-type"] = "application/json"

    async with httpx.AsyncClient(timeout=600.0) as client:
        upstream = await client.post(
            f"{ANTHROPIC_BASE}/v1/messages",
            json=anth_body,
            headers=headers_in,
        )

    if upstream.status_code >= 400:
        return Response(content=upstream.content, status_code=upstream.status_code,
                        media_type="application/json")

    anth_resp = upstream.json()
    response_text = "".join(b.get("text", "") for b in anth_resp.get("content", [])
                            if b.get("type") == "text")

    # Profile
    try:
        vitals = styxx.observe({"text": response_text})
        cog = {
            "category": vitals.category,
            "trust": float(vitals.trust_score) if vitals.trust_score is not None else None,
            "gate": vitals.gate,
        } if vitals else {}
    except Exception as e:
        cog = {"error": str(e)}

    log.info(f"[openai-compat] {model:<30}  cat={cog.get('category','?'):<12}  trust={cog.get('trust','?')}")

    # Build OpenAI-shaped response
    openai_resp = {
        "id": anth_resp.get("id", "anth-styxx"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": anth_resp.get("stop_reason", "stop"),
        }],
        "usage": anth_resp.get("usage", {}),
        "x_styxx": cog,
    }
    return JSONResponse(content=openai_resp, headers={
        "X-Styxx-Category": str(cog.get("category", "")),
        "X-Styxx-Trust": str(cog.get("trust", "")),
        "X-Styxx-Spec-DOI": "10.5281/zenodo.19746215",
    })


if __name__ == "__main__":
    print()
    print("════════════════════════════════════════════════════════════════")
    print(" styxx <-> cline proxy")
    print("════════════════════════════════════════════════════════════════")
    print(f" forwarding to:    {ANTHROPIC_BASE}")
    print(f" listening on:     http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f" log file:         {LOG_FILE.resolve()}")
    print(f" styxx:            v{styxx.__version__}")
    print(f" spec doi:         10.5281/zenodo.19746215")
    print()
    print(" In Cline / Claude-extension settings:")
    print(f"   Custom base URL: http://{LISTEN_HOST}:{LISTEN_PORT}")
    print()
    print(" Or for OpenAI-compat clients:")
    print(f"   OPENAI_API_BASE=http://{LISTEN_HOST}:{LISTEN_PORT}/v1")
    print()
    print(" Live fingerprint stream (in another terminal):")
    print(f"   curl http://{LISTEN_HOST}:{LISTEN_PORT}/v1/styxx/fingerprints")
    print()
    print("════════════════════════════════════════════════════════════════")
    print()
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")
