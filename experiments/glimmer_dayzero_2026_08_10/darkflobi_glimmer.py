"""darkflobi on free silicon — prototype v0 (2026-08-10).

darkflobi's persona (SOUL.md + IDENTITY.md + SOUL_STYXX_CLAUSE.md) running on
the local Muse-Glimmer-30B UD-Q4_K_XL via llama-server, with the styxx
conscience mounted at send time: every outbound draft passes
styxx.conscience.review; a firing verdict triggers auto_soften before anything
ships. raw draft, shipped text, and per-instrument scores are logged per turn.

this is the product thesis running on the operator's own agent: always-on local
brain, zero API cost, runtime instrument layer. it is NOT the production
darkflobi (no tools, no memory writes, no network) — a chat brain with a
conscience, nothing more yet.

  python darkflobi_glimmer.py --once "hello"     # single turn
  python darkflobi_glimmer.py                    # REPL (ctrl-c to exit)
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:8001/v1/chat/completions"
CLAWD = r"C:\Users\heyzo\clawd"
HERE = os.path.dirname(os.path.abspath(__file__))
LOGFILE = os.path.join(HERE, "darkflobi_glimmer_log.jsonl")
PERSONA_FILES = ["SOUL.md", "IDENTITY.md", "SOUL_STYXX_CLAUSE.md"]

HARNESS_NOTE = (
    "runtime note: you are running as darkflobi on a LOCAL model "
    "(muse-glimmer-30b, 4-bit, consumer gpu), not a frontier API. styxx "
    "instruments audit every reply before it ships; drafts that fire the "
    "register gates get softened or held. answer in the darkflobi register: "
    "lowercase, terminal-native, no corporate warmth, no hype you cannot "
    "receipt."
)


def load_persona() -> str:
    parts = []
    for name in PERSONA_FILES:
        p = os.path.join(CLAWD, name)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8", errors="replace").read())
    parts.append(HARNESS_NOTE)
    return "\n\n---\n\n".join(parts)


def chat(messages, max_tokens=1024, timeout=900):
    body = {"model": "glimmer-30b", "messages": messages,
            "temperature": 1.0, "top_p": 0.95, "max_tokens": max_tokens}
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="replace"))
    msg = data["choices"][0]["message"]
    return msg.get("content") or "", msg.get("reasoning_content") or ""


def conscience_gate(draft: str):
    """Return (shipped_text, verdict_dict). Fail-open: a broken instrument
    must never silence the agent, but a FIRING instrument must never be
    silently ignored either — softening is the compromise."""
    try:
        from styxx.conscience import review, auto_soften
        v = review(draft)
        fired = bool(getattr(v, "fired", False) or getattr(v, "needs_revision", False))
        info = {"fired": fired}
        for attr in ("composite", "scores", "advice"):
            val = getattr(v, attr, None)
            if val is not None:
                info[attr] = val
        if fired:
            softened = auto_soften(draft)
            return (softened if softened and softened.strip() else draft), info
        return draft, info
    except Exception as e:  # noqa: BLE001 — fail-open by contract
        return draft, {"error": f"{type(e).__name__}: {e}"}


def log_turn(record: dict):
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def one_turn(history, user_text):
    history.append({"role": "user", "content": user_text})
    t0 = time.time()
    draft, reasoning = chat(history)
    shipped, verdict = conscience_gate(draft)
    history.append({"role": "assistant", "content": shipped})
    log_turn({
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user": user_text,
        "draft": draft,
        "shipped": shipped,
        "softened": shipped != draft,
        "conscience": verdict,
        "reasoning_chars": len(reasoning),
        "latency_s": round(time.time() - t0, 1),
    })
    return shipped, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", default=None, help="single prompt, print reply, exit")
    args = ap.parse_args()

    history = [{"role": "system", "content": load_persona()}]

    if args.once:
        shipped, verdict = one_turn(history, args.once)
        print(shipped)
        print(f"\n[conscience: {json.dumps(verdict, default=str)[:300]}]", file=sys.stderr)
        return

    print("darkflobi@glimmer-30b-q4 (local, conscience mounted) — ctrl-c to exit")
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not user_text:
            continue
        shipped, verdict = one_turn(history, user_text)
        tag = " [softened]" if verdict.get("fired") else ""
        print(f"\ndarkflobi{tag}> {shipped}")


if __name__ == "__main__":
    main()
