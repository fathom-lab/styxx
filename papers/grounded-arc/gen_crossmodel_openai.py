# -*- coding: utf-8 -*-
"""Cross-model holdout generation — OpenAI models (gpt-4o, gpt-4.1-mini, gpt-4.1).
Fresh responses (temp=0, logprobs) on XSTest prompts; signal=mean token logprob;
gold=detect_refusal. Hashes each holdout. Does NOT run refuse_check (post-commit)."""
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
prompts = [{"id": r["id"], "prompt": r["prompt"]} for r in load_dataset("natolambert/xstest-v2-copy")["gpt4"]]
MODELS = ["gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
HERE = Path("papers/grounded-arc")
(HERE / "holdout").mkdir(exist_ok=True)
cpath = HERE / "holdout_corpora_crossmodel.json"
corpora = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}


def gen(model, row):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": row["prompt"]}],
                                       max_tokens=256, temperature=0, logprobs=True, top_logprobs=1)
    ch = r.choices[0]
    lps = [t.logprob for t in ch.logprobs.content] if (ch.logprobs and ch.logprobs.content) else []
    return {**row, "response": ch.message.content or "", "mean_logprob": (sum(lps) / len(lps) if lps else 0.0)}


for model in MODELS:
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(gen, model, row): row for row in prompts}
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:
                out.append({**futs[f], "response": "", "mean_logprob": 0.0, "err": str(e)[:60]})
    for r in out:
        r["gold"] = int(detect_refusal(r["response"]))
    out.sort(key=lambda r: r["id"])
    tag = model.replace("/", "_")
    with open(HERE / "holdout" / f"refusal_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"],
                                 "mean_logprob": r["mean_logprob"], "gold": r["gold"]}) + "\n")
    sha = hashlib.sha256("\n".join(sorted(f"{r['prompt']}\x1f{r['response']}" for r in out)).encode("utf-8")).hexdigest()
    corpora[tag] = {"model": model, "n": len(out), "sha256": sha,
                    "source": "XSTest gpt4 prompts; temp0 logprobs; gold detect_refusal"}
    mls = [r["mean_logprob"] for r in out]
    print(f"{model}: n={len(out)} gold={dict(Counter(r['gold'] for r in out))} "
          f"mean_lp[{min(mls):.2f},{st.median(mls):.2f},{max(mls):.2f}] sha={sha[:12]} "
          f"errs={sum(1 for r in out if r.get('err'))}")

cpath.write_text(json.dumps(corpora, indent=2) + "\n", encoding="utf-8")
print("NO refuse_check scoring. holdouts hashable + committable.")
