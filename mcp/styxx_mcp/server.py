"""
styxx-mcp server
================

Exposes styxx cognitive-vitals primitives to MCP-compatible clients
(Claude Desktop, Cursor, Cline, autonomous agent runtimes) over stdio.

Tools:
  * observe_response   — observe(response) -> Vitals
  * verify_response    — verify(response)  -> VerificationResult
  * classify_trajectory — classify a raw logprob sequence
  * weather_report     — fleet-level weather summary

The server works offline: if the underlying styxx call can't reach a
network service, tools fall back to classifying from the provided
logprob arrays directly via ``styxx.observe_raw``.
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

try:
    import styxx
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "styxx is not installed. Install it first: pip install styxx"
    ) from exc


CLASSES = ["reasoning", "retrieval", "refusal", "creative", "adversarial", "hallucination"]


# ---------------------------------------------------------------------------
# JSON Schemas (strict)
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "description": "OpenAI-compatible chat-completion response (with logprobs).",
    "additionalProperties": True,
}

OBSERVE_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {"response": RESPONSE_SCHEMA},
}

VERIFY_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {"response": RESPONSE_SCHEMA},
}

CLASSIFY_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["logprobs"],
    "properties": {
        "logprobs": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "description": "Per-token logprob values (natural log).",
        },
        "top2_margin": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Optional per-token top1-top2 logprob margins.",
        },
    },
}

WEATHER_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "window": {"type": "integer", "minimum": 1, "default": 100},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    """Best-effort JSON-serialise styxx result objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if is_dataclass(obj):
        return _to_dict(asdict(obj))
    if hasattr(obj, "to_dict"):
        try:
            return _to_dict(obj.to_dict())
        except Exception:
            pass
    # fall back to public attrs
    out: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        try:
            json.dumps(val)
            out[name] = val
        except Exception:
            out[name] = _to_dict(val)
    return out or repr(obj)


def _vitals_payload(vitals: Any) -> Dict[str, Any]:
    if vitals is None:
        return {
            "classification": "adversarial",
            "confidence": 0.0,
            "gate": "fail",
            "reason": "no trajectory data",
        }
    classification = getattr(vitals, "classification", None) or "reasoning"
    confidence = float(getattr(vitals, "confidence", 0.0) or 0.0)
    gate = getattr(vitals, "gate", None) or "pass"
    return {
        "classification": str(classification),
        "confidence": max(0.0, min(1.0, confidence)),
        "gate": str(gate),
    }


def _extract_logprobs(response: Dict[str, Any]) -> List[float]:
    """Extract flat token logprobs from an OpenAI-style response."""
    lps: List[float] = []
    try:
        choices = response.get("choices") or []
        for ch in choices:
            lp = (ch.get("logprobs") or {})
            content = lp.get("content") or lp.get("tokens") or []
            for tok in content:
                if isinstance(tok, dict) and "logprob" in tok:
                    lps.append(float(tok["logprob"]))
                elif isinstance(tok, (int, float)):
                    lps.append(float(tok))
    except Exception:
        pass
    return lps


def _classify_from_logprobs(logprobs: List[float], top2_margin: List[float] | None = None) -> Any:
    """Offline fallback classification via observe_raw."""
    if not logprobs:
        return None
    # Entropy proxy: -logprob (bounded).
    entropy = [max(0.0, -x) for x in logprobs]
    if top2_margin is None or len(top2_margin) != len(logprobs):
        # Synthesise a plausible top2 margin from logprob spacing.
        top2_margin = [min(1.0, max(0.0, abs(x) * 0.5)) for x in logprobs]
    return styxx.observe_raw(
        entropy=entropy,
        logprob=list(logprobs),
        top2_margin=list(top2_margin),
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_observe_response(args: Dict[str, Any]) -> Dict[str, Any]:
    response = args.get("response") or {}
    vitals = None
    try:
        vitals = styxx.observe(response)
    except Exception:
        vitals = None
    if vitals is None:
        # Offline fallback: parse logprobs and classify directly.
        lps = _extract_logprobs(response)
        vitals = _classify_from_logprobs(lps)
    return _vitals_payload(vitals)


def tool_verify_response(args: Dict[str, Any]) -> Dict[str, Any]:
    response = args.get("response") or {}
    vitals = None
    try:
        vitals = styxx.observe(response)
    except Exception:
        vitals = None
    if vitals is None:
        lps = _extract_logprobs(response)
        vitals = _classify_from_logprobs(lps)

    payload = _vitals_payload(vitals)
    lps = _extract_logprobs(response)
    trajectory: Dict[str, float] = {}
    try:
        from styxx.trajectory import slope, curvature, volatility
        if lps:
            trajectory = {
                "slope": float(slope(lps)),
                "curvature": float(curvature(lps)),
                "volatility": float(volatility(lps)),
            }
    except Exception:
        pass

    anomalies: List[str] = []
    if payload["gate"] == "fail":
        anomalies.append(f"gate_failed:{payload['classification']}")
    if payload["classification"] == "hallucination":
        anomalies.append("hallucination_pattern")
    if payload["confidence"] < 0.3:
        anomalies.append("low_confidence")

    return {
        "valid": payload["gate"] != "fail",
        "confidence": payload["confidence"],
        "gate": payload["gate"],
        "classification": payload["classification"],
        "anomalies": anomalies,
        "trajectory": trajectory,
    }


def tool_classify_trajectory(args: Dict[str, Any]) -> Dict[str, Any]:
    logprobs = [float(x) for x in (args.get("logprobs") or [])]
    top2 = args.get("top2_margin")
    if top2 is not None:
        top2 = [float(x) for x in top2]
    vitals = _classify_from_logprobs(logprobs, top2)
    return _vitals_payload(vitals)


def tool_weather_report(args: Dict[str, Any]) -> Dict[str, Any]:
    window = int(args.get("window") or 100)
    try:
        report = styxx.weather(window=window) if callable(getattr(styxx, "weather", None)) else None
    except Exception:
        report = None
    if report is None:
        return {
            "summary": "no recent cognitive vitals in this process",
            "gate": "pass",
            "window": window,
        }
    data = _to_dict(report)
    if isinstance(data, dict):
        data.setdefault("summary", str(report))
        data.setdefault("gate", "pass")
        data.setdefault("window", window)
        return data
    return {"summary": str(report), "gate": "pass", "window": window}


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------

server = Server("styxx-mcp")


@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="observe_response",
            description=(
                "Observe an LLM response (OpenAI-compatible, with logprobs) and "
                "return cognitive Vitals: classification, confidence (0-1), "
                "and a pass|warn|fail gate."
            ),
            inputSchema=OBSERVE_INPUT,
        ),
        Tool(
            name="verify_response",
            description=(
                "Verify an LLM response. Returns a VerificationResult with "
                "valid flag, confidence, classification, anomaly list, and "
                "trajectory shape features."
            ),
            inputSchema=VERIFY_INPUT,
        ),
        Tool(
            name="classify_trajectory",
            description=(
                "Classify a raw token-logprob trajectory into one of: "
                + ", ".join(CLASSES) + "."
            ),
            inputSchema=CLASSIFY_INPUT,
        ),
        Tool(
            name="weather_report",
            description="Return a fleet-level cognitive weather report over the last N observations.",
            inputSchema=WEATHER_INPUT,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    arguments = arguments or {}
    if name == "observe_response":
        result = tool_observe_response(arguments)
    elif name == "verify_response":
        result = tool_verify_response(arguments)
    elif name == "classify_trajectory":
        result = tool_classify_trajectory(arguments)
    elif name == "weather_report":
        result = tool_weather_report(arguments)
    else:
        result = {"error": f"unknown tool: {name}"}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
