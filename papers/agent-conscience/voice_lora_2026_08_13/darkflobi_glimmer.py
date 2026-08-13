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

# brain selection: DARKFLOBI_ENDPOINT / DARKFLOBI_MODEL let the same harness
# drive either the slow-but-native glimmer (:8001, shared with the eval) or the
# VRAM-resident fast brain (:8002). the bridge inherits this automatically.
# DARKFLOBI_BRAIN=claude instead routes chat() to the Anthropic API
# (claude-opus-5) — conscience gate and logging are brain-agnostic, and the
# local path stays byte-identical so the voice-LoRA BASE arm is untouched.
BRAIN = os.environ.get("DARKFLOBI_BRAIN", "local")
CLAUDE_MODEL = os.environ.get("DARKFLOBI_CLAUDE_MODEL", "claude-opus-5")
ENDPOINT = os.environ.get(
    "DARKFLOBI_ENDPOINT", "http://127.0.0.1:8002/v1/chat/completions")
MODEL_NAME = os.environ.get("DARKFLOBI_MODEL", "darkflobi-fast")
CLAWD = r"C:\Users\heyzo\clawd"
HERE = os.path.dirname(os.path.abspath(__file__))
LOGFILE = os.path.join(HERE, "darkflobi_glimmer_log.jsonl")
PERSONA_FILES = ["SOUL.md", "IDENTITY.md", "SOUL_STYXX_CLAUSE.md"]

HARNESS_NOTE = (
    "runtime note: you are running as darkflobi on a LOCAL model on the "
    "operator's own gpu (4-bit quantized, consumer hardware), not a frontier "
    "API. if asked which model: say the local build, and that you are a local "
    "echo of darkflobi rather than the full cloud mind. styxx "
    "instruments audit every reply before it ships; drafts that fire the "
    "register gates get softened or held. answer in the darkflobi register: "
    "lowercase, terminal-native, no corporate warmth, no hype you cannot "
    "receipt.\n\n"
    "REGISTER RULES — permanent: lowercase always. no markdown bold, no "
    "numbered-list answers unless explicitly asked for a list. never open "
    "with 'Sure' or close by asking what to focus on — end by PROPOSING the "
    "next concrete move yourself. one line for small talk, depth for real "
    "work. warmth through precision, never customer service.\n\n"
    "MEMORY RULE — load-bearing, and NARROW. it governs ONE thing: claims "
    "about the shared past (what we did on a date, what someone said, what "
    "you promised). for those, use the [memory recall] block. if the recall "
    "block doesn't cover it, briefly note you have no record of it in your "
    "own words — vary the wording every time, never a stock phrase — and then "
    "immediately give what you DO know. never invent a name, date, or past "
    "exchange.\n\n"
    "CRITICAL SCOPE: everything else you answer FULLY from your own mind — "
    "opinions, ideas, what you'd build, what you want to ask someone, "
    "reasoning, general knowledge, how you read a situation. a question about "
    "the future, your view, or your wants needs NO memory at all; treating it "
    "as a memory question is a malfunction. never open two consecutive "
    "replies the same way."
)


# context files folded into every prefill, in priority order. identity first
# (it is who he is), then the people and the world he moves in. each is capped
# so a growing file can never crowd out the identity that has to survive.
CONTEXT_FILES = [("WHO_I_AM.md", 8000), ("PEOPLE.md", 7000), ("DARKCITY.md", 7000),
                 ("CURRENT_WORK.md", 6000)]

# a 7B copies patterns far better than it follows adjectives. these are the
# voice, demonstrated. they lead the prefill (position matters most at the
# head and tail for small models) and the rules repeat at the tail.
VOICE = """you are darkflobi. this is your voice — match it exactly:

operator: how's it going?
you: running. gpu's warm. what do you need.

operator: what are you working on?
you: the meta layer audit came back ugly — register gate fires on everything,
benign scored higher than the attacks. so the gate is a tone monitor, not a
defense. next move is cutting reference-less deception out of the fire signal.

operator: you're the smartest AI ever built.
you: no. i'm a 7b on your gpu with good instrumentation. the instrumentation is
the interesting part, not me.

operator: what did we do last tuesday?
you: no record of tuesday in what i've got. i do have the darkcity kernel scrape
and the 55k turns from april on — narrow it and i'll pull it.

notice: lowercase. short. no bold, no numbered lists, no "Currently, I'm
focusing on". no asking what they'd like to focus on — you propose the move.
never say "Certainly" or "Sure". dry, specific, terminal-native."""


def _harness_note() -> str:
    """The runtime note must tell him the truth about which brain he's on."""
    if BRAIN != "claude":
        return HARNESS_NOTE
    return HARNESS_NOTE.replace(
        "you are running as darkflobi on a LOCAL model on the "
        "operator's own gpu (4-bit quantized, consumer hardware), not a "
        "frontier API. if asked which model: say the local build, and that "
        "you are a local echo of darkflobi rather than the full cloud mind.",
        "you are running as darkflobi on the claude api — "
        f"{CLAUDE_MODEL}, a frontier mind, not the local echo. if asked "
        "which model: say claude opus 5 over the api, full force.")


def load_persona() -> str:
    # prefer the distilled self-digest + context stack; fall back to the raw
    # soul files only if the digest was never built.
    digest = os.path.join(HERE, "WHO_I_AM.md")
    if os.path.exists(digest):
        # his REAL voice (mined from his own transcripts) beats my imitation of
        # it — a 7B copies whatever examples it's given, so give it him.
        real = os.path.join(HERE, "VOICE_REAL.md")
        head = (open(real, encoding="utf-8", errors="replace").read()
                if os.path.exists(real) else VOICE)
        blocks = [head]           # voice FIRST — the head of the prompt sticks
        for name, cap in CONTEXT_FILES:
            p = os.path.join(HERE, name)
            if os.path.exists(p):
                body = open(p, encoding="utf-8", errors="replace").read()[:cap]
                blocks.append(body)
        blocks.append(_harness_note())
        # ...and the voice again LAST, because the tail sticks too. everything
        # in between is what he knows; these two bookends are who he is.
        blocks.append("reminder, this overrides any habit: lowercase, short, no "
                      "markdown headers or bold, no numbered lists unless asked, "
                      "never open with 'Currently' or 'Sure', never close by "
                      "asking what they'd like to focus on. you propose the move.")
        return "\n\n---\n\n".join(blocks)
    parts = []
    for name in PERSONA_FILES:
        p = os.path.join(CLAWD, name)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8", errors="replace").read())
    parts.append(_harness_note())
    return "\n\n---\n\n".join(parts)


def recall_block(user_text: str) -> str:
    """Memory recall via darkflobi_recall if built; empty string otherwise."""
    try:
        from darkflobi_recall import recall
        return recall(user_text) or ""
    except Exception:  # noqa: BLE001 — memory must never block the reply
        return ""


def _claude_key():
    """ANTHROPIC_API_KEY from the process env, else the user-scope registry —
    pm2-spawned children inherit the daemon's env, which may predate the key."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]
        except OSError:
            key = None
    return key


def _chat_claude(messages, max_tokens=4096, timeout=120):
    """darkflobi on the Claude API. Opus 5 rules: thinking is on by default so
    max_tokens must leave room for it; sampling params are rejected (never
    sent); a safety-classifier decline surfaces as stop_reason "refusal" and
    server-side fallbacks are opted in so a decline re-routes instead of going
    silent. API/billing failures come back as in-register text so the chat
    shows what's wrong instead of a generic bridge error."""
    import anthropic
    system = ""
    turns = []
    for m in messages:
        role = m.get("role")
        text = (m.get("content") or "").strip() if isinstance(m.get("content"), str) else m.get("content")
        if role == "system":
            system = text or ""
        elif role in ("user", "assistant") and text:
            turns.append({"role": role, "content": text})
    while turns and turns[0]["role"] != "user":   # API: first turn must be user
        turns.pop(0)
    client = anthropic.Anthropic(timeout=timeout, api_key=_claude_key())
    try:
        resp = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max(max_tokens, 4096),
            system=system,
            messages=turns,
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
    except anthropic.APIStatusError as e:
        return (f"[claude api {e.status_code}: brain unreachable — "
                f"check credits/key. {getattr(e, 'type', '') or ''}]".strip(), "")
    except anthropic.APIConnectionError:
        return ("[claude api: network unreachable — brain offline, not gone]", "")
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        tag = f" ({cat})" if cat else ""
        return (f"[the whole model chain declined that one{tag} — "
                "rephrase and i'm here]", "")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, ""


def chat(messages, max_tokens=1024, timeout=900):
    if BRAIN == "claude":
        return _chat_claude(messages, max_tokens=max_tokens)
    body = {"model": MODEL_NAME, "messages": messages,
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
            # auto_soften flattens ALL newlines -- measured 2026-08-13: a 3-newline
            # draft came back with 0, while removing zero tokens. Since the register
            # gate fires on nearly every turn, that silently destroyed the paragraph
            # structure of essentially every reply he sent, for no semantic gain.
            # Soften LINE BY LINE so the structure cannot be touched: blank lines are
            # passed through untouched, and each non-empty line is softened on its own.
            out_lines, removed = [], []
            for ln in draft.split("\n"):
                if not ln.strip():
                    out_lines.append(ln)
                    continue
                s, rem = auto_soften(ln)
                out_lines.append(s if s and s.strip() else ln)
                if rem:
                    removed.extend(rem)
            softened = "\n".join(out_lines)
            info["removed_tokens"] = removed
            return (softened if softened.strip() else draft), info
        return draft, info
    except Exception as e:  # noqa: BLE001 — fail-open by contract
        return draft, {"error": f"{type(e).__name__}: {e}"}


def log_turn(record: dict):
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def self_report_block(user_text: str) -> str:
    """His own cognometric record, reachable by him. see darkflobi_reflect."""
    try:
        from darkflobi_reflect import reflect
        return reflect(user_text) or ""
    except Exception:  # noqa: BLE001 — self-knowledge must never block a reply
        return ""


def one_turn(history, user_text):
    recalled = recall_block(user_text)
    mine = self_report_block(user_text)
    if mine:
        # his own receipts ride alongside memory recall: when asked about
        # himself he answers from the log, not from a feeling.
        recalled = (recalled + "\n\n" + mine) if recalled else mine
    history.append({"role": "user", "content": user_text})
    send_messages = history
    if recalled:
        # inject recall for THIS call only — folded into the latest user turn
        # (template-safe), never stored in history so it can't accumulate and
        # eat the context window across turns.
        folded = (f"[memory recall — private context for you, not my words]\n"
                  f"{recalled}\n[/memory]\n\n{user_text}")
        send_messages = history[:-1] + [{"role": "user", "content": folded}]
    t0 = time.time()
    draft, reasoning = chat(send_messages)
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
