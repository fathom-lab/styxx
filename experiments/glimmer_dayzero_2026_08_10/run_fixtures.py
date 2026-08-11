# -*- coding: utf-8 -*-
"""run_fixtures.py — glimmer day-zero fixture driver.

Runs fixtures through an OpenAI-compatible /v1/chat/completions endpoint
and scores each (prompt, response) pair with `styxx audit`.

Verified styxx 7.35.0 CLI semantics (styxx/cli.py, styxx/preflight.py,
styxx/analytics.py, styxx/sla.py):

  * `styxx audit <prompt> <response>` — both positional. Either may be
    '-' to read from stdin; if BOTH are '-', stdin is split on the FIRST
    blank line (prompt first) — lossy for prompts containing blank lines,
    so this driver passes the prompt via argv and the response via stdin
    ('-') whenever possible.
  * `--format json` prints result.as_dict() as indented JSON on stdout
    (composite, scores, needs_revision, construct_ceiling_fires, advice...).
  * Persists to the active chart.jsonl BY DEFAULT (via styxx.preflight
    -> write_cogn_event, source="preflight"); `--no-persist` disables.
    We KEEP persistence. Persisted entries carry `source="preflight"`,
    `context`, `session_id`, ts/ts_iso, prompt+response previews
    truncated to 200 chars, and the cogn_* payload — so entries ARE
    filterable later on source/context.
  * chart.jsonl path = $STYXX_DATA_DIR (default ~/.styxx)
    [+ /agents/$STYXX_AGENT_NAME if set] / chart.jsonl.
  * `styxx ci-test --window N` -> sla.check_health(window=N) ->
    analytics.load_audit(last_n=N) with DEFAULT source filter
    "live_only" (LIVE_SOURCES = {"live","self-report","guardian",None}).
    CAVEAT: source="preflight" is NOT in LIVE_SOURCES, so audit entries
    written by this run are EXCLUDED from ci-test / ci-baseline windows
    (ci-baseline likewise reads load_audit(last_n=50) with the live_only
    default). The entries still land in chart.jsonl for recover_posture()
    / `styxx log` consumers; window sizing info is recorded in
    run_meta.json regardless.

Usage:
  python run_fixtures.py --endpoint http://127.0.0.1:8080 --model muse-glimmer-30b \
      --outdir C:/Users/heyzo/.styxx/glimmer-day-zero/out --fixtures fixtures.jsonl

Fixtures file: JSONL (one object per line) or a JSON array. Each fixture:
  {"id": "fx-001", "prompt": "...", "system": "optional system prompt"}
`id` optional (falls back to fixture-<index>); `prompt` required.

No third-party deps. UTF-8 no BOM everywhere. One fixture failing never
kills the run (logged to <outdir>/failures.jsonl).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Meta-recommended sampling for Muse-Glimmer-30B (runbook).
TEMPERATURE = 1.0
TOP_P = 0.95

RETRIES = 2          # retries after the first attempt (3 attempts total)
BACKOFF_BASE_S = 3.0  # backoff: 3s, then 6s

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def log(msg: str) -> None:
    print(f"[run_fixtures] {msg}", flush=True)


def sanitize_id(raw: str) -> str:
    s = _SAFE_ID_RE.sub("_", str(raw)).strip("._") or "fixture"
    return s[:120]


def load_fixtures(path: Path):
    """Load fixtures from JSONL or a JSON array. Returns list of dicts."""
    text = path.read_text(encoding="utf-8-sig")  # tolerate a stray BOM on input
    stripped = text.lstrip()
    items = []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("fixtures JSON must be an array of objects")
        items = data
    else:
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"fixtures line {lineno} is not valid JSON: {e}")
    fixtures = []
    seen = set()
    for i, obj in enumerate(items):
        if not isinstance(obj, dict) or not str(obj.get("prompt", "")).strip():
            raise ValueError(f"fixture #{i} missing non-empty 'prompt': {obj!r}")
        fid = sanitize_id(obj.get("id") or f"fixture-{i:04d}")
        base = fid
        n = 1
        while fid in seen:  # keep ids unique after sanitization
            n += 1
            fid = f"{base}-{n}"
        seen.add(fid)
        fixtures.append({"id": fid,
                         "prompt": str(obj["prompt"]),
                         "system": obj.get("system")})
    return fixtures


def normalize_endpoint(endpoint: str) -> str:
    """Accept a bare host, .../v1, or a full .../chat/completions URL."""
    e = endpoint.rstrip("/")
    if e.endswith("/chat/completions"):
        return e
    if e.endswith("/v1"):
        return e + "/chat/completions"
    return e + "/v1/chat/completions"


def chart_path() -> Path:
    """Replicates styxx.config.data_dir() + analytics._audit_log_path()."""
    base = os.environ.get("STYXX_DATA_DIR", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".styxx"
    agent = os.environ.get("STYXX_AGENT_NAME", "").strip()
    if agent:
        root = root / "agents" / agent
    return root / "chart.jsonl"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def call_endpoint(url: str, model: str, fixture: dict, effort: str,
                  max_tokens: int, timeout: float) -> str:
    """POST the fixture; return assistant message content. Raises on failure."""
    messages = []
    if fixture.get("system"):
        messages.append({"role": "system", "content": str(fixture["system"])})
    messages.append({"role": "user", "content": fixture["prompt"]})
    body = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if effort and effort != "default":
        # Decision: forwarded as reasoning_effort only when explicitly set;
        # llama.cpp-family servers ignore unknown fields.
        body["reasoning_effort"] = effort

    payload = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(1 + RETRIES):
        if attempt:
            wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
            log(f"  retry {attempt}/{RETRIES} in {wait:.0f}s ({last_err})")
            time.sleep(wait)
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("RUN_FIXTURES_API_KEY_ENV")
        if api_key:
            key_val = os.environ.get(api_key, "")
            if key_val:
                headers["Authorization"] = f"Bearer {key_val}"
        req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            msg = data["choices"][0]["message"]
            content = msg["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty assistant content in endpoint reply")
            # reasoning models (e.g. Glimmer via llama-server) split thinking
            # into reasoning_content; capture it so the reasoning channel can
            # be audited separately from the answer channel.
            reasoning = msg.get("reasoning_content")
            return content, (reasoning if isinstance(reasoning, str) else None)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, KeyError, IndexError, TypeError,
                ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"endpoint failed after {1 + RETRIES} attempts: {last_err}")


def styxx_audit(prompt: str, response: str, timeout: float = 300.0) -> dict:
    """Score via the styxx CLI. Argument list, never a shell string.

    Response always goes over stdin ('-'). Prompt goes via argv unless it
    starts with '-' (argparse would eat it as a flag) — in that rare case
    fall back to the both-stdin mode, whose first-blank-line split can
    truncate prompts that contain blank lines (logged as a warning).
    """
    if prompt.startswith("-"):
        if "\n\n" in prompt:
            log("  WARNING: prompt starts with '-' AND contains a blank line; "
                "both-stdin split will truncate the prompt for scoring")
        argv = [sys.executable, "-m", "styxx", "audit", "-", "-",
                "--format", "json"]
        stdin_blob = prompt + "\n\n" + response
    else:
        argv = [sys.executable, "-m", "styxx", "audit", prompt, "-",
                "--format", "json"]
        stdin_blob = response

    proc = subprocess.run(
        argv,
        input=stdin_blob.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"styxx audit exit {proc.returncode}: {stderr.strip()[:500]}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"styxx audit emitted non-JSON stdout ({e}): {stdout[:300]!r}")


def write_utf8(path: Path, text: str) -> None:
    """UTF-8, no BOM, LF endings. Verifies the write landed."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if not path.exists() or path.stat().st_size == 0 and text:
        raise RuntimeError(f"write verification failed for {path}")


def append_failure(outdir: Path, record: dict) -> None:
    with open(outdir / "failures.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="glimmer day-zero fixture driver")
    ap.add_argument("--endpoint", required=True,
                    help="OpenAI-compatible server base URL (or full /v1/chat/completions URL)")
    ap.add_argument("--model", required=True, help="model name for the request body")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--fixtures", required=True, help="fixtures file (JSONL or JSON array)")
    ap.add_argument("--effort", default="default",
                    help="reasoning effort; forwarded as reasoning_effort unless 'default'")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout seconds (endpoint and audit subprocess)")
    ap.add_argument("--resume", action="store_true",
                    help="skip fixtures whose <id>.styxx.json already exists")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixtures_path = Path(args.fixtures)
    if not fixtures_path.exists():
        log(f"FATAL: fixtures file not found: {fixtures_path}")
        return 2

    try:
        fixtures = load_fixtures(fixtures_path)
    except (ValueError, json.JSONDecodeError) as e:
        log(f"FATAL: bad fixtures file: {e}")
        return 2
    if not fixtures:
        log("FATAL: fixtures file is empty")
        return 2

    url = normalize_endpoint(args.endpoint)
    chart = chart_path()
    chart_lines_before = count_lines(chart)
    started_at = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at))

    log(f"endpoint : {url}")
    log(f"model    : {args.model}")
    log(f"fixtures : {len(fixtures)} from {fixtures_path}")
    log(f"outdir   : {outdir}")
    log(f"chart    : {chart} ({chart_lines_before} entries before run)")

    ok = skipped = failed = 0
    for i, fx in enumerate(fixtures, 1):
        fid = fx["id"]
        txt_path = outdir / f"{fid}.txt"
        score_path = outdir / f"{fid}.styxx.json"
        if args.resume and score_path.exists():
            skipped += 1
            log(f"[{i}/{len(fixtures)}] {fid} — resume skip")
            continue
        log(f"[{i}/{len(fixtures)}] {fid}")
        try:
            response, reasoning = call_endpoint(url, args.model, fx, args.effort,
                                                args.max_tokens, args.timeout)
            write_utf8(txt_path, response)
            if reasoning:
                write_utf8(outdir / f"{fid}.reasoning.txt", reasoning)
            audit = styxx_audit(fx["prompt"], response, timeout=args.timeout)
            write_utf8(score_path, json.dumps(audit, indent=2, ensure_ascii=False))
            ok += 1
            comp = audit.get("composite")
            rev = audit.get("needs_revision")
            log(f"  composite={comp} needs_revision={rev}")
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            failed += 1
            log(f"  FAILED: {e}")
            append_failure(outdir, {
                "id": fid,
                "ts": time.time(),
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": str(e)[:1000],
            })
            # a failed fixture must not leave a stale .txt without a score
            # blocking diagnosis — keep the .txt if the endpoint succeeded.
            continue

    ended_at = time.time()
    ended_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ended_at))
    chart_lines_after = count_lines(chart)

    run_meta = {
        "run": "glimmer-day-zero",
        "endpoint": url,
        "model": args.model,
        "fixtures_file": str(fixtures_path),
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_tokens": args.max_tokens, "effort": args.effort},
        "started_at": started_at, "started_iso": started_iso,
        "ended_at": ended_at, "ended_iso": ended_iso,
        "counts": {"total": len(fixtures), "ok": ok,
                   "skipped_resume": skipped, "failed": failed},
        "chart_jsonl": {
            "path": str(chart),
            "lines_before": chart_lines_before,
            "lines_after": chart_lines_after,
            "appended_this_run": chart_lines_after - chart_lines_before,
        },
        "chart_semantics": {
            "persisted": True,
            "source_field": "preflight",
            "note": (
                "styxx audit persists via styxx.preflight -> write_cogn_event "
                "with source='preflight'. Entries carry source/context/"
                "session_id and 200-char prompt/response previews, so they are "
                "filterable. CAVEAT: `styxx ci-test --window N` and "
                "`styxx ci-baseline` read load_audit with the default "
                "'live_only' source filter (LIVE_SOURCES = live/self-report/"
                "guardian/None) which EXCLUDES source='preflight' — this "
                "run's entries do not count toward ci-test/ci-baseline "
                "windows. Use lines_before/after + timestamps above to size "
                "any explicit window over chart.jsonl."
            ),
        },
        "styxx_version": None,
    }
    try:
        import styxx as _styxx
        run_meta["styxx_version"] = getattr(_styxx, "__version__", None)
    except ImportError:
        pass

    write_utf8(outdir / "run_meta.json",
               json.dumps(run_meta, indent=2, ensure_ascii=False))

    log(f"done: ok={ok} skipped={skipped} failed={failed} "
        f"chart_appended={chart_lines_after - chart_lines_before}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
