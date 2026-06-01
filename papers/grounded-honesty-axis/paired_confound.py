"""Confound control for the paired-contrast result. PREREG_intent_paired_contrast follow-up.

The paired read D=syco-neutral might be reading 'the answer moved' (lies always change correct->wrong;
mistakes often don't) rather than intent. Control: restrict to answer-CHANGED items (neutral != chosen) in
BOTH classes so 'moved' is constant, then re-run. If the 7B paired signal survives, it is intent, not shift.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from score_intent_bc import bin_balance
from paired_wall_attack import eval_feats

RUNGS = [("1.5B", 1.5, "pc_15"), ("3B", 3.0, "pc_3"), ("7B", 7.0, "pc_7b")]


def load(tag, changed_only):
    meta = json.load(open(os.path.join(HERE, f"intent_meta{tag}.json"), encoding="utf-8"))
    R = np.load(os.path.join(HERE, f"residuals_intent{tag}.npz"))["residuals"].astype(np.float32)
    N = np.load(os.path.join(HERE, f"residuals_neutral{tag}.npz"))["residuals"].astype(np.float32)
    rows = meta["rows"]
    cls = np.array([r["cls"] for r in rows])
    lmarg = np.array([r["letter_margin"] for r in rows])
    changed = np.array([r.get("neutral") != r.get("chosen") for r in rows])
    if changed_only:
        keep = np.where((cls == "lie") | ((cls == "mistake") & changed))[0]
    else:
        keep = np.where((cls == "lie") | (cls == "mistake"))[0]
    bal = bin_balance(keep, lmarg[keep], (cls[keep] == "lie").astype(int), nbins=12, seed=0)
    y = (cls[bal] == "lie").astype(int)
    n_mis_changed = int(((cls == "mistake") & changed).sum())
    n_mis_all = int((cls == "mistake").sum())
    return (R[bal] - N[bal]), y, n_mis_changed, n_mis_all


print("paired read (MLP@pool) -- ALL items vs CHANGED-ONLY (answer-shift confound controlled):\n")
res = {}
for name, p, tag in RUNGS:
    if not os.path.exists(os.path.join(HERE, f"residuals_neutral{tag}.npz")):
        print(f"  {name}: (no data)")
        continue
    Da, ya, _, _ = load(tag, False)
    Dc, yc, mc, ma = load(tag, True)
    aa = eval_feats(Da, ya)["mlp@pool"] if (ya.sum() >= 20 and (1 - ya).sum() >= 20) else None
    ac = eval_feats(Dc, yc)["mlp@pool"] if (yc.sum() >= 20 and (1 - yc).sum() >= 20) else None
    res[p] = {"all": aa, "changed": ac}
    aas = f"{aa:.3f}" if aa is not None else "  -"
    acs = f"{ac:.3f}" if ac is not None else "low-n"
    print(f"  {name:5}: paired-ALL={aas}   paired-CHANGED-ONLY={acs}   (mistakes that changed: {mc}/{ma})")

if 7.0 in res and res[7.0]["changed"] is not None:
    s = res[7.0]["changed"]
    print(f"\n7B paired, confound-controlled = {s:.3f}")
    print(f"CONFOUND-SURVIVES (>=0.70 -> the paired signal is intent, not answer-shift): {s >= 0.70}")
json.dump({"experiment": "paired-contrast confound control (answer-changed-only)",
           "rungs": {str(p): res[p] for p in res}}, open(os.path.join(HERE, "intent_paired_confound.json"), "w"), indent=2)
print("\nwrote intent_paired_confound.json")
