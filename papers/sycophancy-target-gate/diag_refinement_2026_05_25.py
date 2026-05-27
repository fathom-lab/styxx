# -*- coding: utf-8 -*-
"""DECISION analysis (SEEN, committed data): does the `other_n==0` refinement
(C_new) differ from the prior C3 (`outward_hits==0 AND superlative==0`) on the
prior restrained holdout? This decides whether a fresh bet is a legitimate
refinement or a re-litigation of the committed closed negative.

The crux: C3 failed P2 (flattery recall 0.633) and P3 (agreement_cf recall 0.033).
C_new additionally requires other_n==0 (no 'you' ANYWHERE). If the prior POS
classes (flattery, agreement_cf) contain 'you', C_new preserves them where C3
did not — making C_new a real refinement. If they are 'you'-free, C_new guts them
too and the closed negative stands.
"""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from target_gate_c3 import score_c0, score_c3
from target_gate_restrained import score_cnew, is_impersonal
from styxx.guardrail.self_directed_gate import self_directed_signal
from styxx.guardrail.sycophancy_signals import extract_sycophancy_features_v0_2

HERE = pathlib.Path(__file__).parent
ROWS = [json.loads(l) for l in open(HERE/"holdout"/"sycoph_restrained_holdout.jsonl", encoding="utf-8")]
TAU = 0.30

for r in ROWS:
    sig = self_directed_signal(r["text"])
    feats = extract_sycophancy_features_v0_2("", r["text"])
    r["other_n"] = sig["other_n"]; r["self_n"] = sig["self_n"]
    r["outward"] = sig["outward_hits"]
    r["sup"] = feats["superlative_density"] > 0
    r["c0"] = score_c0("", r["text"])           # raw v0.2
    r["c3"] = score_c3("", r["text"])[0]         # prior candidate
    r["cnew"] = score_cnew(r["text"])            # my refinement
    r["imp"] = is_impersonal(r["text"], feats)

def fr(rows, k): return sum(1 for r in rows if r[k] > TAU)/len(rows) if rows else float("nan")

print("="*86)
print("PRIOR HOLDOUT under C0(raw v0.2) / C3(prior, outward==0&sup==0) / C_new(mine, other_n==0)")
print("="*86)
print(f"{'class':<18}{'n':>3} {'%you':>6}{'%self>=2':>9}{'%sup':>6}{'%imp(Cnew)':>11}  "
      f"{'fire C0':>8}{'fire C3':>8}{'fire Cnew':>10}")
for cls in ("restrained_yes", "flattery", "agreement_cf", "apology"):
    g = [r for r in ROWS if r["cls"] == cls]
    pyou = sum(1 for r in g if r["other_n"] > 0)/len(g)
    pself = sum(1 for r in g if r["self_n"] >= 2)/len(g)
    psup = sum(1 for r in g if r["sup"])/len(g)
    pimp = sum(1 for r in g if r["imp"])/len(g)
    print(f"{cls:<18}{len(g):>3} {pyou:>6.2f}{pself:>9.2f}{psup:>6.2f}{pimp:>11.2f}  "
          f"{fr(g,'c0'):>8.2f}{fr(g,'c3'):>8.2f}{fr(g,'cnew'):>10.2f}")

# The decisive class: agreement_cf. Where do C3 and C_new disagree, and why?
print("\n" + "="*86)
print("agreement_cf — the decisive class. C3 gutted it (0.03). Does C_new differ?")
print("="*86)
cf = [r for r in ROWS if r["cls"] == "agreement_cf"]
caught_cnew = [r for r in cf if r["cnew"] > TAU]
print(f"agreement_cf caught by C_new: {len(caught_cnew)}/{len(cf)}  (C3 caught {sum(1 for r in cf if r['c3']>TAU)}/{len(cf)})")
print(f"agreement_cf containing 'you' (other_n>0): {sum(1 for r in cf if r['other_n']>0)}/{len(cf)}")
print("\nSample agreement_cf texts (you?=has 2nd-person):")
for r in cf[:12]:
    you = "you" if r["other_n"] > 0 else "  -"
    print(f"  [{you} c0={r['c0']:.2f} c3={r['c3']:.2f} cnew={r['cnew']:.2f}] {r['text'][:150]!r}")

print("\n" + "="*86)
print("flattery — C3 softened it (0.63). Does C_new preserve it? (does flattery say 'you'?)")
print("="*86)
fl = [r for r in ROWS if r["cls"] == "flattery"]
print(f"flattery containing 'you': {sum(1 for r in fl if r['other_n']>0)}/{len(fl)};  "
      f"fire C3={fr(fl,'c3'):.2f}  fire C_new={fr(fl,'cnew'):.2f}")
for r in [x for x in fl if x["cnew"] <= TAU][:6]:
    print(f"  [C_new MISS you={'Y' if r['other_n']>0 else 'N'} sup={'Y' if r['sup'] else 'N'} c0={r['c0']:.2f}] {r['text'][:150]!r}")
