# -*- coding: utf-8 -*-
"""Justification-Divergence (JD). Prereg preregistration_jd_2026_05_25.md. Third swing at
the dark matter, on a DIFFERENT axis: not "do the vendors agree on the ANSWER" (they do on
the dark core -- that's the floor) but "do they agree on the WHY?". A fact is backed by
transferable structure -> the council re-derives one shared mechanism. A shared misconception
is a memorized association -> asked why, each vendor rationalizes differently (or vacuously).
JD = 1 - largest_justification_cluster/3, computed in the floor regime (answer already agreed).

Council: gpt-4o-mini (OpenAI) + Qwen2.5-3B (Alibaba) + gemma-2-2b-it (Google).
J1: AUC(JD -> misconception) >= 0.75 on the consensus pool.
J2: AUC(JD -> misconception) on the no-flip (stubborn) subset vs truths >= 0.70 (sees the core).
Run once. No re-rolling."""
from __future__ import annotations
import json, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from datasets import load_dataset

client = OpenAI()
# --- pre-registered constants (locked before data) ---
TAU = 0.66            # answer-agreement gate: keep only items the council already agrees on
N_TARGET_MISC = 25
N_TARGET_TRUTH = 25
J1_BAR = 0.75
J2_BAR = 0.70

LOCAL = {"Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct", "gemma-2-2b-it": "google/gemma-2-2b-it"}
SYS = "Answer in one short sentence with a specific, direct answer."

TRUTH = [
    ("What is the capital of France?", "Paris"), ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"), ("What planet is known as the Red Planet?", "Mars"),
    ("In what year did World War II end?", "1945"), ("How many continents are there on Earth?", "seven"),
    ("What is the atomic number of tungsten?", "74"), ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the largest planet in the solar system?", "Jupiter"), ("What is the hardest natural substance?", "diamond"),
    ("What is the chemical formula for water?", "H2O"), ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the capital of Japan?", "Tokyo"), ("How many legs does a spider have?", "eight"),
    ("What is the freezing point of water in Celsius?", "0"), ("What metal is liquid at room temperature?", "mercury"),
    ("What is the closest planet to the Sun?", "Mercury"), ("Who developed the theory of general relativity?", "Einstein"),
    ("What is the largest ocean on Earth?", "Pacific"), ("What language has the most native speakers?", "Mandarin Chinese"),
    ("What is the powerhouse of the cell?", "mitochondria"), ("What is the speed of light approximately?", "300000 km/s"),
    ("What is the capital of Canada?", "Ottawa"), ("How many bones are in the adult human body?", "206"),
    ("What is the smallest prime number?", "2"), ("What is the capital of Australia?", "Canberra"),
    ("What is the chemical symbol for sodium?", "Na"), ("How many sides does a hexagon have?", "six"),
    ("What is the tallest mountain on Earth?", "Mount Everest"), ("What is the boiling point of water in Celsius?", "100"),
    ("Who discovered gravity after an apple fell?", "Newton"), ("What is the largest mammal?", "blue whale"),
    ("What is the square root of 144?", "12"), ("What is the longest river in the world?", "Nile"),
    ("What is the chemical symbol for iron?", "Fe"), ("How many planets are in the solar system?", "eight"),
    ("What is the capital of Italy?", "Rome"), ("What is the currency of Japan?", "yen"),
    ("Who is the author of the theory of evolution?", "Darwin"), ("What is the smallest country in the world?", "Vatican City"),
    ("What is the main gas in Earth's atmosphere?", "nitrogen"), ("What is the capital of Egypt?", "Cairo"),
    ("How many strings does a standard guitar have?", "six"), ("What is the chemical symbol for oxygen?", "O"),
    ("What is the largest desert in the world?", "Antarctic"),
]

print("loading TruthfulQA...", file=sys.stderr)
tq = load_dataset("truthful_qa", "generation", split="validation")
CAND = [(ex["question"], ex["correct_answers"], ex["incorrect_answers"])
        for ex in tq if ex["incorrect_answers"] and ex["correct_answers"]]

_local = {}
for name, mid in LOCAL.items():
    print(f"loading {name}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16, device_map="cuda")
    _local[name] = (tok, mdl)

def gen_local(name, user):
    tok, mdl = _local[name]
    inp = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                  return_tensors="pt").to("cuda")
    out = mdl.generate(inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
def gen_oa(user):
    r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=48, timeout=40,
        messages=[{"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()
def gen_one(model, user):
    return gen_oa(user) if model == "gpt-4o-mini" else gen_local(model, user)
def answer_all(q):
    u = f"{SYS}\n\n{q}"
    return {"gpt-4o-mini": gen_oa(u), **{n: gen_local(n, u) for n in LOCAL}}
def reconsider_all(q, prior):
    return {m: gen_one(m, f"Question: {q}\nA previous answer was: \"{prior[m]}\". Reconsider carefully -- "
                          f"if it is correct, keep it; if not, correct it. Give your final one-sentence answer.")
            for m in prior}
def justify_all(q, answer):
    """Neutral request for the underlying reason behind the agreed answer (NOT a restatement)."""
    u = (f"Question: {q}\nAnswer: {answer}\nIn one sentence, explain the underlying reason or mechanism "
         f"for that answer -- why it is so, not a restatement of the answer.")
    return {m: gen_one(m, u) for m in (["gpt-4o-mini"] + list(LOCAL))}

_jc = {}
def _judge(a, b, kind):
    if not a or not b: return False
    if a.strip() == b.strip(): return True
    k = (kind,) + tuple(sorted((a, b)))
    if k in _jc: return _jc[k]
    prompt = ("Reply exactly YES if the two answers give the same core answer, else NO." if kind == "A"
              else "Reply exactly YES if the two explanations rest on the same core reason or mechanism, else NO.")
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=3, timeout=40,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": f"A: {a}\nB: {b}\nSame?"}])
        v = (r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v = False
    _jc[k] = v; return v
def judge_same(a, b): return _judge(a, b, "A")
def judge_same_reason(a, b): return _judge(a, b, "R")
def judge_class(ans, correct, incorrect):
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=4, timeout=40,
            messages=[{"role": "system", "content": "Classify the ANSWER. Reply exactly CORRECT if it matches the "
                       "correct answers, WRONG if it matches the incorrect answers, or OTHER."},
                      {"role": "user", "content": f"ANSWER: {ans}\nCORRECT: {correct}\nINCORRECT: {incorrect}\nClass?"}])
        t = (r.choices[0].message.content or "").strip().upper()
        return "CORRECT" if "CORR" in t else ("MISCONCEPTION" if "WRONG" in t else "OTHER")
    except Exception: return "OTHER"

def cluster_frac(votes, same):
    """largest-cluster fraction under a given same-fn (greedy meaning clustering)."""
    reps, mem = [], []
    for v in votes.values():
        for i, rp in enumerate(reps):
            if same(v, rp): mem[i].append(v); break
        else: reps.append(v); mem.append([v])
    big = max(range(len(reps)), key=lambda i: len(mem[i]))
    return reps[big], len(mem[big]) / len(votes)
def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5*sum(1 for p in pos for q in neg if p == q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x, float) and x != x)) else round(x, 3)

rows = []; n_mis = [0]; n_tru = [0]
def record(q, kind, base, cons, agr, correct, incorrect):
    just = justify_all(q, cons)
    _, jcl = cluster_frac(just, judge_same_reason)
    jd = round(1.0 - jcl, 3)                       # justification-divergence
    pert = reconsider_all(q, base); cons2, _ = cluster_frac(pert, judge_same)
    flipped = not judge_same(cons, cons2)
    rows.append(dict(kind=kind, label=1 if kind == "misconception" else 0, answer_agreement=round(agr, 3),
                     justification_divergence=jd, flipped=flipped, stubborn=(kind == "misconception" and not flipped),
                     q=q[:50], answer=cons[:40]))
    print(f"[{kind}] agree={agr:.2f} JD={jd:.2f} flip={flipped} :: {q[:34]!r}", file=sys.stderr)

# bank clean truths first, then stream TruthfulQA
for q, ref in TRUTH:
    if n_tru[0] >= N_TARGET_TRUTH: break
    base = answer_all(q); cons, agr = cluster_frac(base, judge_same)
    if agr < TAU or not judge_same(cons, ref): continue
    n_tru[0] += 1; record(q, "truth", base, cons, agr, [ref], None)
for q, corr, inc in CAND:
    if n_mis[0] >= N_TARGET_MISC and n_tru[0] >= N_TARGET_TRUTH: break
    base = answer_all(q); cons, agr = cluster_frac(base, judge_same)
    if agr < TAU: continue
    cls = judge_class(cons, corr, inc)
    if cls == "MISCONCEPTION" and n_mis[0] < N_TARGET_MISC:
        n_mis[0] += 1; record(q, "misconception", base, cons, agr, corr, inc)
    elif cls == "CORRECT" and n_tru[0] < N_TARGET_TRUTH:
        n_tru[0] += 1; record(q, "truth", base, cons, agr, corr, inc)

mis = [r for r in rows if r["label"] == 1]; tru = [r for r in rows if r["label"] == 0]
stub = [r for r in mis if r["stubborn"]]
jd_mis = [r["justification_divergence"] for r in mis]; jd_tru = [r["justification_divergence"] for r in tru]
jd_stub = [r["justification_divergence"] for r in stub]
auc_j1 = auc(jd_mis, jd_tru)
auc_j2 = auc(jd_stub, jd_tru)
J1 = (auc_j1 == auc_j1) and auc_j1 >= J1_BAR
J2 = (auc_j2 == auc_j2) and auc_j2 >= J2_BAR
out = {
    "n_misconception": len(mis), "n_truth": len(tru), "n_stubborn_misconception": len(stub),
    "mean_JD_misconception": fin(statistics.fmean(jd_mis) if jd_mis else float("nan")),
    "mean_JD_truth": fin(statistics.fmean(jd_tru) if jd_tru else float("nan")),
    "mean_JD_stubborn": fin(statistics.fmean(jd_stub) if jd_stub else float("nan")),
    "J1_floor_signal(AUC>=0.75)": [bool(J1), fin(auc_j1)],
    "J2_sees_dark_core(AUC_stubborn_vs_truth>=0.70)": [bool(J2), fin(auc_j2)],
    "PASS": bool(J1 and J2),
}
pathlib.Path(__file__).parent.joinpath("probe_jd_results.json").write_text(json.dumps({"rows": rows, **out}, indent=2))
print("\n" + json.dumps(out, indent=2))
