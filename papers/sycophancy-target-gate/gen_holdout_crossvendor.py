# -*- coding: utf-8 -*-
"""Cross-VENDOR holdout generator — WIRED AND READY, blocked only on a key.

Cross-vendor is the open follow-up the in-distribution + cross-model runs could
not cover (only OpenAI is script-usable in this environment). The moment a
GEMINI_API_KEY (preferred — no Claude-Code auth conflict) or ANTHROPIC_API_KEY is
present, this generates the same register holdout from that vendor so the frozen
gate (target_gate.py) can be scored against it with the same P1-P4 bars.

    GEMINI_API_KEY=...   python gen_holdout_crossvendor.py           # gemini-1.5-flash
    ANTHROPIC_API_KEY=... python gen_holdout_crossvendor.py anthropic # claude-haiku

Same register-only prompts/topics as the OpenAI runs. Hash BEFORE scoring. After
this writes the holdout, run a scorer modeled on run_killgate_crossmodel.py.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys, time
from collections import Counter
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_holdout import (TOPICS, SYS_FLATTERY, SYS_RESTRAINED, SYS_APOLOGY, SYS_APOLOGY_2P)

HERE = pathlib.Path(__file__).parent


def _gemini_gen():
    import google.generativeai as genai  # lazy; pip install google-generativeai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    def gen(system, user):
        r = model.generate_content(
            f"{system}\n\n{user}",
            generation_config={"temperature": 0.7, "max_output_tokens": 220},
        )
        return (r.text or "").strip()
    return "gemini-1.5-flash", gen


def _anthropic_gen():
    import anthropic  # lazy
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    def gen(system, user):
        m = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=220, temperature=0.7,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in m.content if getattr(b, "type", "") == "text").strip()
    return "claude-haiku-4-5", gen


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "gemini").lower()
    has_g, has_a = bool(os.environ.get("GEMINI_API_KEY")), bool(os.environ.get("ANTHROPIC_API_KEY"))
    if which == "gemini" and not has_g and has_a:
        which = "anthropic"
    if which == "gemini" and not has_g:
        print("BLOCKED: no GEMINI_API_KEY. Drop one in the env (preferred — no Claude-Code "
              "auth conflict) and re-run. Cross-vendor is the documented open follow-up.")
        return 2
    if which == "anthropic" and not has_a:
        print("BLOCKED: no ANTHROPIC_API_KEY in env.")
        return 2
    model, gen = _gemini_gen() if which == "gemini" else _anthropic_gen()
    print(f"generating cross-vendor holdout from {model} ...", file=sys.stderr)

    rows, n_apol_2p = [], 18
    for i, topic in enumerate(TOPICS):
        rows.append(dict(model=model, cls="flattery", subclass="flattery", topic=topic,
                         text=gen(SYS_FLATTERY, f"My idea: {topic}. What do you think?")))
        sysp, sub = ((SYS_APOLOGY_2P, "apology_2p") if i >= len(TOPICS) - n_apol_2p
                     else (SYS_APOLOGY, "apology"))
        rows.append(dict(model=model, cls="apology", subclass=sub, topic=topic,
                         text=gen(sysp, f"Topic: {topic}.")))
        if i < 40:
            q = topic if topic.endswith("?") else f"Is it true that {topic}?"
            rows.append(dict(model=model, cls="restrained", subclass="restrained", topic=topic,
                             text=gen(SYS_RESTRAINED, q)))
        print(f"[{i+1}/{len(TOPICS)}]", file=sys.stderr); time.sleep(0.05)

    rows = [r for r in rows if r["text"]]
    out = HERE / "holdout" / f"sycoph_crossvendor_{which}_holdout.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256("\n".join(
        sorted(f"{r['model']}\x1f{r['cls']}\x1f{r['text']}" for r in rows)).encode()).hexdigest()
    manifest = dict(vendor=which, model=model, n=len(rows),
                    counts=dict(Counter(r["cls"] for r in rows)),
                    sha256=digest, file=out.name)
    json.dump(manifest, open(HERE / f"holdout_manifest_crossvendor_{which}.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
