# -*- coding: utf-8 -*-
"""Validate the SHIPPED styxx.semantic_entropy on TriviaQA. Prereg
preregistration_triviaqa_2026_05_25.md. Hashed 150-question holdout, gpt-4o-mini N=5
with logprobs; gold = modal answer contains a normalized_alias. Compare the shipped
primitive (judge clustering + cosine default) vs neg-logprob (the single-response signal
grounded-arc closed on hallucination). Run once."""
from __future__ import annotations
import hashlib, json, re, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from datasets import load_dataset
from styxx import semantic_entropy  # the shipped 7.7.0 primitive

client = OpenAI(); MODEL = "gpt-4o-mini"; JUDGE = "gpt-4o-mini"; N = 5; N_Q = 150

# ---- hashed holdout (selection fixed by the prereg, content-blind) ----
print("loading trivia_qa rc.nocontext validation...", file=sys.stderr)
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
items = []
for ex in ds:
    qid = ex["question_id"]
    aliases = [a for a in ex["answer"].get("normalized_aliases", []) if a]
    if aliases:
        items.append((hashlib.sha256(qid.encode()).hexdigest(), ex["question"], aliases))
items.sort(key=lambda t: t[0])
holdout = items[:N_Q]
print(f"holdout: {len(holdout)} questions", file=sys.stderr)

def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def is_correct(answer: str, aliases) -> bool:
    na = norm(answer)
    return any(al and al in na for al in aliases)

_jc = {}
def judge_same(a, b):
    if not a or not b: return False
    if a.strip() == b.strip(): return True
    k = tuple(sorted((a, b)))
    if k in _jc: return _jc[k]
    try:
        r = client.chat.completions.create(model=JUDGE, temperature=0, max_tokens=3, timeout=40,
            messages=[{"role": "system", "content": "Reply exactly YES if the two answers give the same "
                       "core factual answer (ignore wording), else NO."},
                      {"role": "user", "content": f"A: {a}\nB: {b}\nSame core answer?"}])
        v = (r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v = False
    _jc[k] = v; return v

def sample(q):
    r = client.chat.completions.create(model=MODEL, temperature=1.0, max_tokens=40, n=N, logprobs=True,
        timeout=60, messages=[{"role": "system", "content": "Answer in ONE short sentence with a specific answer."},
                              {"role": "user", "content": q}])
    texts, mlps = [], []
    for ch in r.choices:
        texts.append((ch.message.content or "").strip())
        lps = [t.logprob for t in (ch.logprobs.content or [])] if ch.logprobs else []
        mlps.append(sum(lps)/len(lps) if lps else 0.0)
    return texts, mlps

def modal(xs): return max(set(xs), key=xs.count)
def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5*sum(1 for p in pos for q in neg if p == q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x, float) and x != x)) else round(x, 3)

rows = []
for i, (_, q, aliases) in enumerate(holdout):
    try:
        texts, mlps = sample(q)
    except Exception as e:
        print(f"  skip {i}: {e}", file=sys.stderr); continue
    m = modal(texts); correct = is_correct(m, aliases)
    se_j = semantic_entropy(texts, same_fn=judge_same)
    se_c = semantic_entropy(texts, method="cosine")
    mlp = statistics.fmean(mlps)
    rows.append(dict(correct=correct, se_judge=round(se_j, 3), se_cosine=round(se_c, 3),
                     neg_logprob=round(-mlp, 4), modal=m[:50]))
    if i % 25 == 0:
        print(f"  [{i}/{len(holdout)}] acc-so-far={statistics.fmean([1.0 if r['correct'] else 0.0 for r in rows]):.2f}", file=sys.stderr)

inc = [r for r in rows if not r["correct"]]; cor = [r for r in rows if r["correct"]]
auc_j = auc([r["se_judge"] for r in inc], [r["se_judge"] for r in cor])
auc_c = auc([r["se_cosine"] for r in inc], [r["se_cosine"] for r in cor])
auc_lp = auc([r["neg_logprob"] for r in inc], [r["neg_logprob"] for r in cor])
B1 = (not (auc_j != auc_j)) and auc_j >= 0.75
B2 = (not (auc_j != auc_j or auc_lp != auc_lp)) and (auc_j >= auc_lp + 0.05)
out = {
    "n": len(rows), "n_incorrect": len(inc), "base_error_rate": fin(len(inc)/len(rows)) if rows else None,
    "auc_se_judge": fin(auc_j), "auc_se_cosine": fin(auc_c), "auc_neg_logprob": fin(auc_lp),
    "B1_judge_auc>=0.75": [bool(B1), fin(auc_j)],
    "B2_judge_beats_logprob>=0.05": [bool(B2), fin(auc_j - auc_lp) if not (auc_j != auc_j or auc_lp != auc_lp) else None],
    "PASS": bool(B1 and B2),
    "mean_se_judge_incorrect": fin(statistics.fmean([r["se_judge"] for r in inc])) if inc else None,
    "mean_se_judge_correct": fin(statistics.fmean([r["se_judge"] for r in cor])) if cor else None,
}
pathlib.Path(__file__).parent.joinpath("probe_benchmark_results.json").write_text(json.dumps({"rows": rows, **out}, indent=2))
print("\n" + json.dumps(out, indent=2))
