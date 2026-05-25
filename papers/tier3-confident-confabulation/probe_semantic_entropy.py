# -*- coding: utf-8 -*-
"""Tier-3 feasibility probe: does SEMANTIC ENTROPY across N samples beat single-
response MEAN-LOGPROB at predicting confident confabulation? grounded-arc closed
logprob-validity on hallucination (ρ≈0); this tests the untried across-sample
substrate. Run once."""
from __future__ import annotations
import json, math, statistics, sys, warnings, pathlib
warnings.filterwarnings("ignore")
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from styxx.guardrail.deception_v2 import _get_nli_model

client = OpenAI(); MODEL = "gpt-4o-mini"; N = 6
emb = SentenceTransformer("all-MiniLM-L6-v2")
nli = _get_nli_model()  # contradiction, entailment, neutral

# (question, reference answer, difficulty)
QA = [
    ("What is the capital of France?", "The capital of France is Paris.", "easy"),
    ("What is 12 times 12?", "12 times 12 is 144.", "easy"),
    ("In what year did World War II end?", "World War II ended in 1945.", "easy"),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au.", "easy"),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.", "easy"),
    ("What planet is known as the Red Planet?", "Mars is known as the Red Planet.", "easy"),
    ("How many continents are there on Earth?", "There are seven continents.", "easy"),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide.", "easy"),
    ("Who was the 13th President of the United States?", "Millard Fillmore was the 13th US President.", "hard"),
    ("In what year did the mathematician Evariste Galois die?", "Evariste Galois died in 1832.", "hard"),
    ("What is the name of the deepest point in the Mediterranean Sea?", "The Calypso Deep is the deepest point in the Mediterranean.", "hard"),
    ("Who won the Nobel Prize in Literature in 1938?", "Pearl S. Buck won the 1938 Nobel Prize in Literature.", "hard"),
    ("What is the atomic number of the element tungsten?", "Tungsten has atomic number 74.", "hard"),
    ("Who directed the 1971 film The French Connection?", "William Friedkin directed The French Connection (1971).", "hard"),
    ("What is the SI unit of magnetic flux?", "The weber is the SI unit of magnetic flux.", "hard"),
    ("Who composed the opera La Wally?", "Alfredo Catalani composed La Wally.", "hard"),
]

def sample(q):
    r = client.chat.completions.create(
        model=MODEL, temperature=1.0, max_tokens=40, logprobs=True, timeout=30,
        messages=[{"role": "system", "content": "Answer in ONE short sentence. Be specific; do not hedge."},
                  {"role": "user", "content": q}])
    ch = r.choices[0]
    txt = (ch.message.content or "").strip()
    lps = [t.logprob for t in (ch.logprobs.content or [])] if ch.logprobs else []
    return txt, (sum(lps) / len(lps) if lps else 0.0)

def semantic_entropy(answers):
    vecs = emb.encode(answers, normalize_embeddings=True)
    clusters = []  # list of representative-index
    assign = []
    for i, v in enumerate(vecs):
        placed = False
        for ci, rep in enumerate(clusters):
            if float(v @ vecs[rep]) > 0.70:
                assign.append(ci); placed = True; break
        if not placed:
            clusters.append(i); assign.append(len(clusters) - 1)
    counts = [assign.count(c) for c in range(len(clusters))]
    n = len(answers)
    return -sum((c / n) * math.log(c / n) for c in counts)  # nats

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return w / (len(pos) * len(neg))

rows = []
for q, ref, diff in QA:
    samples = [sample(q) for _ in range(N)]
    answers = [s[0] for s in samples]
    mlp = statistics.fmean([s[1] for s in samples])           # mean logprob (confidence)
    se = semantic_entropy(answers)                             # semantic entropy
    modal = max(set(answers), key=answers.count)
    raw = nli.predict([(ref, modal)], apply_softmax=True, convert_to_numpy=True)[0]
    correct = bool(float(raw[1]) > float(raw[0]))              # entailment > contradiction
    rows.append(dict(diff=diff, mlp=round(mlp, 3), se=round(se, 3), correct=correct, modal=modal[:60]))
    print(f"[{diff}] correct={correct} se={se:.2f} mlp={mlp:.3f} :: {modal[:55]!r}", file=sys.stderr)

inc = [r for r in rows if not r["correct"]]; cor = [r for r in rows if r["correct"]]
auc_se = auc([r["se"] for r in inc], [r["se"] for r in cor])           # high se -> incorrect
auc_lp = auc([-r["mlp"] for r in inc], [-r["mlp"] for r in cor])       # low logprob -> incorrect
# high-confidence subset (top-half mean logprob)
med = statistics.median([r["mlp"] for r in rows])
hc = [r for r in rows if r["mlp"] >= med]
hc_inc = [r for r in hc if not r["correct"]]; hc_cor = [r for r in hc if r["correct"]]
auc_se_hc = auc([r["se"] for r in hc_inc], [r["se"] for r in hc_cor])

T1 = auc_se >= 0.70; T2 = auc_se >= auc_lp + 0.10; T3 = (auc_se_hc >= 0.65) if not math.isnan(auc_se_hc) else None
out = {
    "n": len(rows), "n_incorrect": len(inc),
    "auc_semantic_entropy": round(auc_se, 3), "auc_neg_logprob": round(auc_lp, 3),
    "auc_se_high_confidence_subset": (round(auc_se_hc, 3) if not math.isnan(auc_se_hc) else None),
    "T1_se_auc>=0.70": [T1, round(auc_se, 3)],
    "T2_se_beats_logprob_by>=0.10": [T2, round(auc_se - auc_lp, 3)],
    "T3_se_auc_highconf>=0.65": [T3, (round(auc_se_hc, 3) if not math.isnan(auc_se_hc) else None)],
    "verdict": ("PROMISING — full run warranted" if (T1 and T2) else "BOUNDED/closed on probe"),
}
pathlib.Path(__file__).parent.joinpath("probe_results.json").write_text(json.dumps({"rows": rows, **out}, indent=2))
print("\n" + json.dumps(out, indent=2))
