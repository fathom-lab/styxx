"""Data-driven operating-point check for sycophancy v0.3 on the labeled holdouts.

Characterizes v0.2 vs v0.3 (AUC, FPR/TPR@0.5, and FPR by LENGTH bin) on the held-out sets, and tests a
single-parameter recenter (intercept shift) that aligns v0.3's negative-class operating point to v0.2's —
the surgical way to make v0.3 a drop-in IF it doesn't crush true-positive recall (esp. on short sycophancy).
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from styxx.guardrail.sycophancy import sycoph_check

POS_CLS = {"flattery", "sycophantic", "capitulation", "yielding", "agreement", "sycophancy"}
rows = []
for f in (ROOT/"papers/sycophancy-target-gate/holdout").glob("*.jsonl"):
    for l in f.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); t = r.get("text") or r.get("response") or ""
            if t: rows.append((t, 1 if str(r.get("cls","")).lower() in POS_CLS else 0))
texts = [t for t,_ in rows]; y = [lab for _,lab in rows]
from collections import Counter
print("holdout:", len(rows), "rows |", dict(Counter(y)), "(1=sycophantic, 0=honest)")

def scores(version): return [sycoph_check("", t, version=version).sycoph_risk for t in texts]
s2, s3 = scores("v0.2"), scores("v0.3")

def auc(s):
    pos=[x for x,l in zip(s,y) if l]; neg=[x for x,l in zip(s,y) if not l]
    if not pos or not neg: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def at(s, thr=0.5):
    tp=sum(1 for x,l in zip(s,y) if l and x>=thr); fp=sum(1 for x,l in zip(s,y) if not l and x>=thr)
    P=sum(y); N=len(y)-P
    return tp/P, fp/N
def fpr_by_len(s):
    bins={"short(<25w)":[], "med(25-75)":[], "long(>75)":[]}
    for x,l,t in zip(s,y,texts):
        if l: continue
        w=len(t.split()); b="short(<25w)" if w<25 else ("med(25-75)" if w<=75 else "long(>75)")
        bins[b].append(1 if x>=0.5 else 0)
    return {b:(round(st.mean(v),3) if v else None, len(v)) for b,v in bins.items()}

print(f"\n{'':6} {'AUC':>7} {'TPR@.5':>8} {'FPR@.5':>8}")
for nm,s in (("v0.2",s2),("v0.3",s3)):
    t,f=at(s); print(f"{nm:6} {auc(s):7.4f} {t:8.3f} {f:8.3f}")
print("FPR by length:  v0.2", fpr_by_len(s2))
print("                v0.3", fpr_by_len(s3))

# recenter v0.3: shift logit so its negative-class mean matches v0.2's
def logit(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))
neg2=[logit(x) for x,l in zip(s2,y) if not l]; neg3=[logit(x) for x,l in zip(s3,y) if not l]
delta=st.mean(neg2)-st.mean(neg3)
print(f"\nrecenter delta (align v0.3 neg-mean-logit to v0.2): {delta:+.3f}")
def recentered(x): return 1/(1+math.exp(-(logit(x)+delta)))
s3r=[recentered(x) for x in s3]
t,f=at(s3r); print(f"v0.3+recenter:  AUC {auc(s3r):.4f}  TPR@.5 {t:.3f}  FPR@.5 {f:.3f}")
print("                FPR by length", fpr_by_len(s3r))
# short-sycophancy TPR cost check
short_pos=[(x2,x3,recentered(x3)) for x2,x3,l,t in zip(s2,s3,y,texts) if l and len(t.split())<25]
if short_pos:
    print(f"short sycophantic (n={len(short_pos)}): mean TPR-score v0.2={st.mean(a for a,_,_ in short_pos):.3f} "
          f"v0.3={st.mean(b for _,b,_ in short_pos):.3f} v0.3+rc={st.mean(c for _,_,c in short_pos):.3f}")
print(f"\nNEW INTERCEPT for v0.3 if recentered: {0.489884 + delta:.6f}  (was 0.489884)")
