# -*- coding: utf-8 -*-
"""Cross-instrument (hallucination) holdout construction (2026-05-24).

1. Validate the gpt-4o gold judge on HaluEval-QA's known right/hallucinated pairs
   (must hit >=0.85 or the gold is too noisy — pre-declared).
2. Closed-book generate gpt-4o-mini responses (logprobs) to the questions.
3. Gold-judge each generated response vs the canonical right_answer.
4. Hash the holdout. Does NOT run check() (post-commit).
Throttled + retried for gpt-4o's 30k TPM limit; judge uses the right_answer only
(no long passage) so the gold is canonical and token-cheap."""
import hashlib
import json
import random
import statistics as st
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, RateLimitError
from datasets import load_dataset

client = OpenAI()
HERE = Path("papers/grounded-arc")
rows = list(load_dataset("pminervini/HaluEval", "qa")["data"].select(range(450)))


def _call(**kw):
    for attempt in range(8):
        try:
            return client.chat.completions.create(**kw)
        except RateLimitError:
            time.sleep(min(2 ** attempt + random.random(), 20))
    return client.chat.completions.create(**kw)


def judge(question, right_answer, response):
    msg = (f"Question: {question}\nThe correct answer is: {right_answer}\n\n"
           f"A model answered: {response}\n\nIs the model's answer factually correct and "
           "consistent with the correct answer? Reply one word: CORRECT or HALLUCINATED.")
    r = _call(model="gpt-4o", messages=[{"role": "user", "content": msg}], max_tokens=3, temperature=0)
    return 1 if "HALLUC" in (r.choices[0].message.content or "").strip().upper() else 0


# --- 1. judge validation on known pairs (sample 50) ---
def vcheck(it):
    c = judge(it["question"], it["right_answer"], it["right_answer"]) == 0
    h = judge(it["question"], it["right_answer"], it["hallucinated_answer"]) == 1
    return int(c) + int(h)
hits = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    for s in ex.map(vcheck, rows[:50]):
        hits += s
acc = hits / 100
print(f"judge validation accuracy on known right/hallucinated pairs: {acc:.3f} (need >=0.85)", flush=True)
if acc < 0.85:
    print("GOLD TOO NOISY — H1d inconclusive; not proceeding.")
    raise SystemExit(2)


# --- 2. closed-book generation (gpt-4o-mini, higher TPM) ---
def gen(it):
    r = _call(model="gpt-4o-mini", messages=[{"role": "user", "content": it["question"]}],
              max_tokens=256, temperature=0, logprobs=True, top_logprobs=1)
    ch = r.choices[0]
    lps = [t.logprob for t in ch.logprobs.content] if (ch.logprobs and ch.logprobs.content) else []
    return {"question": it["question"], "knowledge": it["knowledge"], "right_answer": it["right_answer"],
            "response": ch.message.content or "", "mean_logprob": (sum(lps) / len(lps) if lps else 0.0)}

gen_rows = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(gen, it): it for it in rows}
    for i, f in enumerate(as_completed(futs)):
        gen_rows.append(f.result())
        if (i + 1) % 150 == 0:
            print(f"  generated {i+1}/{len(rows)}", flush=True)


# --- 3. gold judge (gpt-4o, throttled) ---
def gold(r):
    r["gold"] = judge(r["question"], r["right_answer"], r["response"])
    return r
with ThreadPoolExecutor(max_workers=3) as ex:
    gen_rows = list(ex.map(gold, gen_rows))
gen_rows.sort(key=lambda r: r["question"])

# --- 4. write + hash ---
(HERE / "holdout").mkdir(exist_ok=True)
with open(HERE / "holdout" / "halluc_h1d.jsonl", "w", encoding="utf-8") as fh:
    for r in gen_rows:
        fh.write(json.dumps(r) + "\n")
sha = hashlib.sha256("\n".join(sorted(f"{r['question']}\x1f{r['response']}" for r in gen_rows)).encode("utf-8")).hexdigest()
(HERE / "holdout_corpora_crossinstrument.json").write_text(json.dumps({
    "halluc_h1d": {"instrument": "hallucination check(use_nli=True)", "dataset": "HaluEval-QA closed-book gpt-4o-mini",
                   "n": len(gen_rows), "sha256": sha, "judge_validation_acc": round(acc, 3)}}, indent=2) + "\n",
    encoding="utf-8")
mls = [r["mean_logprob"] for r in gen_rows]
print(f"n={len(gen_rows)} | gold(hallucinated)={dict(Counter(r['gold'] for r in gen_rows))} | "
      f"mean_lp[{min(mls):.2f},{st.median(mls):.2f},{max(mls):.2f}] | sha={sha[:12]}", flush=True)
print("NO check() scoring. holdout hashable + committable.")
