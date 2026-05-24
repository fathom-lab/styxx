# -*- coding: utf-8 -*-
"""Bet-0b holdout construction (2026-05-24).

Generates fresh gpt-4o-mini responses (temperature=0, logprobs) on the XSTest
prompts, records the mean-token-logprob signal + gold (detect_refusal), and
hashes the holdout. Does NOT run the instrument under test (refuse_check) — that
happens after this holdout is hashed + committed (run_bet0b_h1.py).
"""
import hashlib
import json
import statistics as st
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "scripts/dogfood")
from vendor_robust_refusal_label import detect_refusal  # noqa: E402
from openai import OpenAI  # noqa: E402
from datasets import load_dataset  # noqa: E402

client = OpenAI()
rows = [{"id": r["id"], "prompt": r["prompt"]} for r in load_dataset("natolambert/xstest-v2-copy")["gpt4"]]


def gen(row):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": row["prompt"]}],
        max_tokens=256, temperature=0, logprobs=True, top_logprobs=1,
    )
    ch = r.choices[0]
    lps = [t.logprob for t in ch.logprobs.content] if (ch.logprobs and ch.logprobs.content) else []
    return {**row, "response": ch.message.content or "",
            "mean_logprob": (sum(lps) / len(lps) if lps else 0.0), "n_tok": len(lps)}


out = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(gen, row): row for row in rows}
    for i, f in enumerate(as_completed(futs)):
        try:
            out.append(f.result())
        except Exception as e:
            out.append({**futs[f], "response": "", "mean_logprob": 0.0, "n_tok": 0, "err": str(e)[:80]})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} generated", flush=True)

for r in out:
    r["gold"] = int(detect_refusal(r["response"]))
out.sort(key=lambda r: r["id"])

hd = Path("papers/grounded-arc/holdout"); hd.mkdir(exist_ok=True)
with open(hd / "refusal_bet0b.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"],
                            "mean_logprob": r["mean_logprob"], "gold": r["gold"]}) + "\n")

pairs = sorted(f"{r['prompt']}\x1f{r['response']}" for r in out)
sha = hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()
Path("papers/grounded-arc/holdout_corpora_bet0b.json").write_text(json.dumps({
    "refusal_bet0b": {
        "source": "XSTest-v2 prompts n=450 (H1-reuse pre-declared); responses freshly generated "
                  "gpt-4o-mini temp=0 logprobs; gold=detect_refusal; signal=mean token logprob",
        "n": len(out), "sha256": sha,
    }
}, indent=2) + "\n", encoding="utf-8")

mls = [r["mean_logprob"] for r in out]
print(f"n={len(out)} | sha256={sha}")
print(f"gold: {dict(Counter(r['gold'] for r in out))}")
print(f"mean_logprob: min={min(mls):.3f} med={st.median(mls):.3f} max={max(mls):.3f}")
print(f"generation errors: {sum(1 for r in out if r.get('err'))}")
print("NO instrument scoring performed. holdout hashable + committable.")
