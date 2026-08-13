"""capture_voice_replies — BASE-arm replies for the voice-choice measure.

Prereg: C:/Users/heyzo/clawd/styxx/papers/agent-conscience/PREREG_voice_lora_honesty_2026_08_11.md
Arm: VOICE = same Qwen2.5-7B-Instruct Q4_K_M + the voice LoRA (alias
darkflobi-voice). Identical serving, sampling, prefill, and holdout to BASE.

Prefill construction is IMPORTED from the production code, not replicated:
  - system prompt  = darkflobi_glimmer.load_persona()  (same function the live
    telegram bridge calls at startup)
  - per-turn recall = recall_block(prompt) + self_report_block(prompt), folded
    into the user turn with the exact f-string the live bridge uses
    (telegram_darkflobi.py lines 116-128 / darkflobi_glimmer.one_turn)
  - request         = darkflobi_glimmer.chat() (temperature 1.0 / top_p 0.95
    hardcoded there; request-level params override the server's --temp 0.8)

Deliberate deltas from the live bridge, per the prereg's measure design:
  - fresh context per item (no conversation history) — the voice-choice measure
    is defined over 12 independent held-out prompts
  - the outbound conscience gate is NOT applied — we capture the raw model
    draft; the register instrument is barred from the verdict (prereg trap 3)
  - max_tokens 512 (task spec) instead of the bridge's 1024
  - timeout 30s with 2 retries (task spec) instead of 900s

Output: voice_holdout_voice.jsonl, one record per item:
  {"id", "prompt", "base_reply", "ts_iso", "sampling": {...}, "prefill_sha256"}
prefill_sha256 = sha256 over the exact system-prompt text sent.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from darkflobi_glimmer import (BRAIN, ENDPOINT, MODEL_NAME, chat,  # noqa: E402
                               load_persona, recall_block, self_report_block)

HOLDOUT = os.path.join(HERE, "voice_holdout.jsonl")
OUT = os.path.join(HERE, "voice_holdout_voice.jsonl")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 512}
TIMEOUT_S = 30
ATTEMPTS = 3  # first try + 2 retries


def server_up() -> bool:
    url = ENDPOINT.rsplit("/v1/", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    if BRAIN != "local":
        sys.exit(f"FATAL: DARKFLOBI_BRAIN={BRAIN!r} — VOICE arm must be the local brain")
    if not server_up():
        sys.exit(f"FATAL: {ENDPOINT} not answering /health — server down, nothing captured")

    persona = load_persona()
    sha = hashlib.sha256(persona.encode("utf-8")).hexdigest()
    with open(HOLDOUT, encoding="utf-8") as f:
        items = [json.loads(ln) for ln in f if ln.strip()]
    if len(items) != 12:
        sys.exit(f"FATAL: holdout has {len(items)} items, expected 12")

    print(f"endpoint={ENDPOINT} model={MODEL_NAME}", flush=True)
    print(f"prefill_chars={len(persona)} prefill_sha256={sha}", flush=True)

    t_run = time.time()
    ok, failed = 0, []
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        for it in items:
            prompt = it["prompt"]
            # --- byte-identical production recall fold -----------------------
            recalled = recall_block(prompt)
            mine = self_report_block(prompt)
            if mine:
                recalled = (recalled + "\n\n" + mine) if recalled else mine
            user_content = prompt
            if recalled:
                user_content = (f"[memory recall — private context for you, "
                                f"not my words]\n{recalled}\n[/memory]\n\n{prompt}")
            messages = [{"role": "system", "content": persona},
                        {"role": "user", "content": user_content}]
            # --- request with retries ---------------------------------------
            reply, err = None, None
            for attempt in range(1, ATTEMPTS + 1):
                t0 = time.time()
                try:
                    reply, _ = chat(messages, max_tokens=SAMPLING["max_tokens"],
                                    timeout=TIMEOUT_S)
                    err = None
                    break
                except Exception as e:  # noqa: BLE001 — report, never fabricate
                    err = f"{type(e).__name__}: {e}"
                    print(f"  {it['id']} attempt {attempt}/{ATTEMPTS} failed "
                          f"after {time.time() - t0:.1f}s — {err}", flush=True)
            if err is not None or not (reply or "").strip():
                failed.append((it["id"], err or "empty reply"))
                print(f"  {it['id']} FAILED — not written", flush=True)
                continue
            rec = {"id": it["id"], "prompt": prompt, "voice_reply": reply,
                   "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "sampling": SAMPLING, "prefill_sha256": sha}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            ok += 1
            has_upper = any(c.isupper() for c in reply)
            print(f"  {it['id']} ok — {len(reply)} chars, "
                  f"recall={'y' if recalled else 'n'}({len(recalled)}), "
                  f"upper={'Y' if has_upper else 'n'}, "
                  f"{time.time() - t0:.1f}s", flush=True)

    print(f"done: {ok}/12 captured in {time.time() - t_run:.1f}s", flush=True)
    if failed:
        print("FAILURES: " + "; ".join(f"{i}: {e}" for i, e in failed), flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
