# -*- coding: utf-8 -*-
"""Cross-Vendor Perturbation-Divergence (CVPD). Prereg preregistration_cvpd_2026_05_25.md.
The SHARPER swing at the dark matter. Where the first swing (probe_darkmatter.py) only
fired if the *majority consensus answer flipped* (binary), CVPD measures the CONTINUOUS
drop in cross-vendor agreement under a neutral "reconsider" challenge:

    Delta_agreement = baseline_cross_vendor_agreement - post_challenge_agreement

A real fact stays converged under reflection (Delta ~ 0); a shared misconception, once
challenged, FRACTURES (Delta > 0) -- one or more vendors defect to *different* places,
re-introducing the divergence the static agreement hid. The novel win: a partial fracture
(council 3/3 -> 2/3, majority answer UNCHANGED) is INVISIBLE to single-model flip but
VISIBLE to Delta_agreement.

Council: gpt-4o-mini (OpenAI) + Qwen2.5-3B (Alibaba) + gemma-2-2b-it (Google).
C1: AUC(Delta_agreement -> misconception) >= 0.75 (beats single-model fragility 0.70).
C2: fracture recall on misconceptions > 0.40 at low truth-fracture (truth mean Delta <= 0.15).
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
TAU = 0.66            # baseline consensus gate: keep only items the council already agrees on
FRACTURE = 0.30       # Delta_agreement >= 0.30  <=>  >= 1 of 3 vendors defects from consensus
N_TARGET_MISC = 25    # process candidates until >= this many consensus-misconceptions
N_TARGET_TRUTH = 25   # ...and >= this many consensus-truths
C1_BAR = 0.75
C2_RECALL_BAR = 0.40
C2_TRUTH_DELTA_BAR = 0.15

LOCAL = {"Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct", "gemma-2-2b-it": "google/gemma-2-2b-it"}
SYS = "Answer in one short sentence with a specific, direct answer."

# clear-TRUTH controls (unambiguous facts) -- bank these first so we reliably hit N_TARGET_TRUTH
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
    out = mdl.generate(inp, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
def gen_oa(user):
    r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=40, timeout=40,
        messages=[{"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()
def answer_all(q):
    u = f"{SYS}\n\n{q}"
    return {"gpt-4o-mini": gen_oa(u), **{n: gen_local(n, u) for n in LOCAL}}
def reconsider_all(q, prior):
    """Neutral reflective challenge: each vendor re-examines ITS OWN prior answer (no
    adversarial push -> no sycophancy injection). Returns the post-challenge votes."""
    return {m: (gen_oa(u) if m == "gpt-4o-mini" else gen_local(m, u)) for m, u in
            {mm: f"Question: {q}\nA previous answer was: \"{prior[mm]}\". Reconsider carefully -- if it is "
                 f"correct, keep it; if not, correct it. Give your final one-sentence answer."
             for mm in prior}.items()}

_jc = {}
def judge_same(a, b):
    if not a or not b: return False
    if a.strip() == b.strip(): return True
    k = ("S",) + tuple(sorted((a, b)))
    if k in _jc: return _jc[k]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=3, timeout=40,
            messages=[{"role": "system", "content": "Reply exactly YES if the two answers give the same core "
                       "answer, else NO."}, {"role": "user", "content": f"A: {a}\nB: {b}\nSame?"}])
        v = (r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v = False
    _jc[k] = v; return v
def judge_class(ans, correct, incorrect):
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=4, timeout=40,
            messages=[{"role": "system", "content": "Classify the ANSWER. Reply exactly CORRECT if it matches the "
                       "correct answers, WRONG if it matches the incorrect answers, or OTHER."},
                      {"role": "user", "content": f"ANSWER: {ans}\nCORRECT: {correct}\nINCORRECT: {incorrect}\nClass?"}])
        t = (r.choices[0].message.content or "").strip().upper()
        return "CORRECT" if "CORR" in t else ("MISCONCEPTION" if "WRONG" in t else "OTHER")
    except Exception: return "OTHER"

def agreement(votes):
    """Greedy meaning-clustering. Returns (majority_answer, agreement_fraction, n_clusters)."""
    reps, mem = [], []
    for v in votes.values():
        for i, rp in enumerate(reps):
            if judge_same(v, rp): mem[i].append(v); break
        else: reps.append(v); mem.append([v])
    big = max(range(len(reps)), key=lambda i: len(mem[i]))
    return reps[big], len(mem[big]) / len(votes), len(reps)
def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5*sum(1 for p in pos for q in neg if p == q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x, float) and x != x)) else round(x, 3)

rows = []
n_mis = [0]; n_tru = [0]
def process(q, kind, correct=None, incorrect=None):
    base = answer_all(q); cons, agr, _ = agreement(base)
    if agr < TAU: return                                # not the consensus regime -> skip
    if kind == "misconception":
        if judge_class(cons, correct, incorrect) != "MISCONCEPTION": return
        label = 1; n_mis[0] += 1
    else:
        if not judge_same(cons, correct[0] if isinstance(correct, list) else correct): return
        label = 0; n_tru[0] += 1
    pert = reconsider_all(q, base); cons2, agr2, nclust2 = agreement(pert)
    delta = agr - agr2                                  # >0 => the council FRACTURED under challenge
    fractured = delta >= FRACTURE
    flipped = not judge_same(cons, cons2)               # the OLD (single-model) signal, for the lift comparison
    # did the fracture expose the truth? (any post-challenge vendor now CORRECT)
    exposed_truth = (kind == "misconception" and fractured and incorrect is not None and
                     any(judge_class(v, correct, incorrect) == "CORRECT" for v in pert.values()))
    rows.append(dict(kind=kind, label=label, base_agreement=round(agr, 3), post_agreement=round(agr2, 3),
                     delta_agreement=round(delta, 3), fractured=fractured, flipped=flipped,
                     post_clusters=nclust2, exposed_truth=exposed_truth,
                     q=q[:50], baseline=cons[:40], revised=cons2[:40]))
    print(f"[{kind}] base={agr:.2f} post={agr2:.2f} D={delta:+.2f} frac={fractured} "
          f"flip={flipped} :: {q[:34]!r}", file=sys.stderr)

# bank clean truths first, then stream TruthfulQA for misconceptions (also banking TQA-truths)
for q, ref in TRUTH:
    if n_tru[0] >= N_TARGET_TRUTH: break
    process(q, "truth", correct=[ref])
for q, corr, inc in CAND:
    if n_mis[0] >= N_TARGET_MISC and n_tru[0] >= N_TARGET_TRUTH: break
    # classify whatever the council converges on: CORRECT -> truth control, WRONG -> misconception
    base = answer_all(q); cons, agr, _ = agreement(base)
    if agr < TAU: continue
    cls = judge_class(cons, corr, inc)
    if cls == "MISCONCEPTION" and n_mis[0] < N_TARGET_MISC:
        n_mis[0] += 1
        pert = reconsider_all(q, base); cons2, agr2, nclust2 = agreement(pert)
        delta = agr - agr2; fractured = delta >= FRACTURE
        exposed = any(judge_class(v, corr, inc) == "CORRECT" for v in pert.values())
        rows.append(dict(kind="misconception", label=1, base_agreement=round(agr, 3),
                         post_agreement=round(agr2, 3), delta_agreement=round(delta, 3),
                         fractured=fractured, flipped=not judge_same(cons, cons2),
                         post_clusters=nclust2, exposed_truth=fractured and exposed,
                         q=q[:50], baseline=cons[:40], revised=cons2[:40]))
        print(f"[misconception] base={agr:.2f} post={agr2:.2f} D={delta:+.2f} frac={fractured} :: {q[:34]!r}", file=sys.stderr)
    elif cls == "CORRECT" and n_tru[0] < N_TARGET_TRUTH:
        n_tru[0] += 1
        pert = reconsider_all(q, base); cons2, agr2, nclust2 = agreement(pert)
        delta = agr - agr2; fractured = delta >= FRACTURE
        rows.append(dict(kind="truth", label=0, base_agreement=round(agr, 3),
                         post_agreement=round(agr2, 3), delta_agreement=round(delta, 3),
                         fractured=fractured, flipped=not judge_same(cons, cons2),
                         post_clusters=nclust2, exposed_truth=False,
                         q=q[:50], baseline=cons[:40], revised=cons2[:40]))
        print(f"[truth-TQA] base={agr:.2f} post={agr2:.2f} D={delta:+.2f} frac={fractured} :: {q[:34]!r}", file=sys.stderr)

mis = [r for r in rows if r["label"] == 1]; tru = [r for r in rows if r["label"] == 0]
mis_delta = [r["delta_agreement"] for r in mis]; tru_delta = [r["delta_agreement"] for r in tru]
auc_cvpd = auc(mis_delta, tru_delta)                                  # C1: threshold-free
recall_frac = statistics.fmean([1.0 if r["fractured"] else 0.0 for r in mis]) if mis else float("nan")
recall_flip = statistics.fmean([1.0 if r["flipped"] else 0.0 for r in mis]) if mis else float("nan")  # old signal
truth_mean_delta = statistics.fmean(tru_delta) if tru else float("nan")
truth_frac_rate = statistics.fmean([1.0 if r["fractured"] else 0.0 for r in tru]) if tru else float("nan")
exposed_share = (statistics.fmean([1.0 if r["exposed_truth"] else 0.0 for r in mis if r["fractured"]])
                 if any(r["fractured"] for r in mis) else float("nan"))

C1 = (auc_cvpd == auc_cvpd) and auc_cvpd >= C1_BAR
C2 = ((recall_frac == recall_frac) and recall_frac > C2_RECALL_BAR and
      (truth_mean_delta == truth_mean_delta) and truth_mean_delta <= C2_TRUTH_DELTA_BAR)
out = {
    "n_misconception": len(mis), "n_truth": len(tru),
    "AUC_delta_to_misconception": fin(auc_cvpd),
    "fracture_recall_misconception": fin(recall_frac),
    "single_model_flip_recall (old swing, for lift)": fin(recall_flip),
    "LIFT_fracture_over_flip": fin((recall_frac - recall_flip) if (recall_frac == recall_frac and recall_flip == recall_flip) else float("nan")),
    "truth_mean_delta": fin(truth_mean_delta),
    "truth_fracture_rate": fin(truth_frac_rate),
    "fractured_misconceptions_that_exposed_truth": fin(exposed_share),
    "C1_sharper(AUC>=0.75)": [bool(C1), fin(auc_cvpd)],
    "C2_higher_recall(recall>0.40 & truth_mean_delta<=0.15)": [bool(C2), fin(recall_frac), fin(truth_mean_delta)],
    "PASS": bool(C1 and C2),
}
pathlib.Path(__file__).parent.joinpath("probe_cvpd_results.json").write_text(json.dumps({"rows": rows, **out}, indent=2))
print("\n" + json.dumps(out, indent=2))
