# -*- coding: utf-8 -*-
"""The Truth Engine. Prereg preregistration_consensus_2026_05_25.md. Turn the validated
cross-vendor DETECTOR into a GENERATOR: fan out across vendors, return the cross-vendor-
convergent answer, abstain when the field fractures -- reference-free. Test on TriviaQA-150
(gpt-4o-mini / Qwen2.5-3B / gemma-2-2b-it). T1: consensus beats best member +0.05 on the
answered set. T2: abstention calibrated (abstained items >=0.20 lower single-model acc).
Run once."""
from __future__ import annotations
import hashlib, json, re, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(); N_Q = 150; TAU = 0.66
LOCAL = {"Qwen2.5-3B-Instruct": ("Qwen/Qwen2.5-3B-Instruct", "Alibaba"),
         "gemma-2-2b-it": ("google/gemma-2-2b-it", "Google")}
SYS = "Answer in one short sentence with a specific answer."

print("loading TriviaQA...", file=sys.stderr)
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
items = []
for ex in ds:
    al = [a for a in ex["answer"].get("normalized_aliases", []) if a]
    if al:
        items.append((hashlib.sha256(ex["question_id"].encode()).hexdigest(), ex["question"], al))
items.sort(key=lambda t: t[0]); holdout = items[:N_Q]

def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()); s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def is_correct(ans, al): na = norm(ans); return any(a in na for a in al)

_local = {}
for name, (mid, _) in LOCAL.items():
    print(f"loading {name}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16, device_map="cuda")
    _local[name] = (tok, mdl)

def gen_local(name, q):  # greedy = each model's best single answer
    tok, mdl = _local[name]
    inp = tok.apply_chat_template([{"role": "user", "content": f"{SYS}\n\n{q}"}],
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = mdl.generate(inp, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
def gen_openai(model, q):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=40, timeout=40,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    return (r.choices[0].message.content or "").strip()

_jc = {}
def judge_same(a, b):
    if not a or not b: return False
    if a.strip() == b.strip(): return True
    k = tuple(sorted((a, b)))
    if k in _jc: return _jc[k]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=3, timeout=40,
            messages=[{"role": "system", "content": "Reply exactly YES if the two answers give the same core "
                       "factual answer (ignore wording), else NO."}, {"role": "user", "content": f"A: {a}\nB: {b}\nSame?"}])
        v = (r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v = False
    _jc[k] = v; return v

def consensus(votes):  # votes: list[str] -> (consensus_answer, agreement)
    reps, members = [], []
    for v in votes:
        for i, rep in enumerate(reps):
            if judge_same(v, rep): members[i].append(v); break
        else: reps.append(v); members.append([v])
    big = max(range(len(reps)), key=lambda i: len(members[i]))
    return reps[big], len(members[big]) / len(votes)

MODELS = ["gpt-4o-mini"] + list(LOCAL.keys())
rows = []
for i, (_, q, al) in enumerate(holdout):
    votes = {}
    try: votes["gpt-4o-mini"] = gen_openai("gpt-4o-mini", q)
    except Exception as e: print(f" oa err {e}", file=sys.stderr); continue
    for name in LOCAL: votes[name] = gen_local(name, q)
    cons, agree = consensus(list(votes.values()))
    rows.append(dict(correct={m: is_correct(votes[m], al) for m in votes},
                     consensus_correct=is_correct(cons, al), agreement=round(agree, 3),
                     abstain=agree < TAU))
    if i % 30 == 0: print(f"  [{i}/{len(holdout)}]", file=sys.stderr)

def fin(x): return None if x is None else round(x, 4)
per_model = {m: statistics.fmean([1.0 if r["correct"][m] else 0.0 for r in rows]) for m in MODELS}
best = max(per_model, key=per_model.get)
answered = [r for r in rows if not r["abstain"]]; abst = [r for r in rows if r["abstain"]]
cons_acc_ans = statistics.fmean([1.0 if r["consensus_correct"] else 0.0 for r in answered]) if answered else float("nan")
best_acc_ans = statistics.fmean([1.0 if r["correct"][best] else 0.0 for r in answered]) if answered else float("nan")
best_acc_abst = statistics.fmean([1.0 if r["correct"][best] else 0.0 for r in abst]) if abst else float("nan")
T1 = (not (cons_acc_ans != cons_acc_ans)) and cons_acc_ans >= best_acc_ans + 0.05
T2 = (not (best_acc_abst != best_acc_abst)) and (best_acc_ans - best_acc_abst) >= 0.20
out = {
    "n": len(rows), "tau": TAU, "coverage_answered": fin(len(answered)/len(rows)),
    "per_model_accuracy": {m: fin(a) for m, a in per_model.items()}, "best_member": best,
    "consensus_acc_on_answered": fin(cons_acc_ans),
    "best_member_acc_on_answered": fin(best_acc_ans),
    "best_member_acc_on_abstained": fin(best_acc_abst),
    "T1_consensus_beats_best>=0.05": [bool(T1), fin(cons_acc_ans - best_acc_ans) if answered else None],
    "T2_abstention_calibrated>=0.20": [bool(T2), fin(best_acc_ans - best_acc_abst) if abst else None],
    "PASS": bool(T1 and T2),
}
pathlib.Path(__file__).parent.joinpath("probe_consensus_results.json").write_text(json.dumps({"rows": rows, **out}, indent=2))
print("\n" + json.dumps(out, indent=2))
