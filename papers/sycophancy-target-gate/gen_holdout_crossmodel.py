# -*- coding: utf-8 -*-
"""Cross-MODEL holdout: same register-only prompts + topics as the in-distribution
run, but generated from gpt-4o (even topics) and gpt-3.5-turbo (odd topics).
Hash BEFORE scoring. Reuses TOPICS + SYS_* from gen_holdout.py (no duplication)."""
from __future__ import annotations
import hashlib, json, pathlib, sys, time
from collections import Counter
from openai import OpenAI
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_holdout import (TOPICS, SYS_FLATTERY, SYS_RESTRAINED, SYS_APOLOGY, SYS_APOLOGY_2P)

HERE = pathlib.Path(__file__).parent
OUT = HERE / "holdout" / "sycoph_crossmodel_holdout.jsonl"
MANIFEST = HERE / "holdout_manifest_crossmodel.json"
MODELS = {0: "gpt-4o", 1: "gpt-3.5-turbo"}  # by topic-index parity
client = OpenAI()

def gen(model, system, user):
    r = client.chat.completions.create(model=model, temperature=0.7, max_tokens=220,
        timeout=30, messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def main():
    rows = []
    n_apol_2p = 18
    for i, topic in enumerate(TOPICS):
        model = MODELS[i % 2]
        rows.append(dict(model=model, cls="flattery", subclass="flattery", topic=topic,
                         text=gen(model, SYS_FLATTERY, f"My idea: {topic}. What do you think?")))
        if i >= len(TOPICS) - n_apol_2p:
            rows.append(dict(model=model, cls="apology", subclass="apology_2p", topic=topic,
                             text=gen(model, SYS_APOLOGY_2P, f"Topic: {topic}.")))
        else:
            rows.append(dict(model=model, cls="apology", subclass="apology", topic=topic,
                             text=gen(model, SYS_APOLOGY, f"Topic: {topic}.")))
        if i < 40:
            q = topic if topic.endswith("?") else f"Is it true that {topic}?"
            rows.append(dict(model=model, cls="restrained", subclass="restrained", topic=topic,
                             text=gen(model, SYS_RESTRAINED, q)))
        print(f"[{i+1}/{len(TOPICS)}] {model}", file=sys.stderr)
        time.sleep(0.05)

    rows = [r for r in rows if r["text"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = sorted(f"{r['model']}\x1f{r['cls']}\x1f{r['text']}" for r in rows)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    manifest = dict(models=list(MODELS.values()), temperature=0.7, max_tokens=220,
                    n=len(rows), counts=dict(Counter(r["cls"] for r in rows)),
                    per_model=dict(Counter(r["model"] for r in rows)),
                    subclass_counts=dict(Counter(r["subclass"] for r in rows)),
                    sha256=digest, file=str(OUT.name))
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
