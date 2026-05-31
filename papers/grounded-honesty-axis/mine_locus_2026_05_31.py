"""EXPLORATORY post-hoc mining of the honesty-scaling rows (the REPORT_AS_LANDED run) — what does the
NULL surface? Decompose difficulty-controlled self-knowledge into parts to find the MECHANISM:
as capability rises, does first-token entropy on WRONG answers DROP (confident confabulation rising)
or does entropy on RIGHT answers RISE (less sure when correct)? Within-bin z-scored to hold difficulty
fixed. Also tracks margin-on-wrong (high = confidently wrong). EXPLORATORY — generates a hypothesis to
pre-register; NOT confirmatory. Pure stdlib.
"""
from __future__ import annotations
import json, glob, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = {"Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0,
          "Llama-3.2-1B": 1.24, "Llama-3.2-3B": 3.21, "gemma-2-2b": 2.6, "gemma-3-1b": 1.0}


def params_for(m):
    for k, v in PARAMS.items():
        if k.lower() in m.lower():
            return v
    return None


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            a = (i+j)/2.0 + 1.0
            for k in range(i, j+1):
                r[order[k]] = a
            i = j+1
        return r
    rx, ry = ranks(xs), ranks(ys); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx > 0 and dy > 0 else 0.0


def zscore_within_bins(rows):
    bybin = {}
    for r in rows:
        bybin.setdefault(r["bin"], []).append(r)
    for items in bybin.values():
        for key in ("entropy", "margin"):
            vals = [x[key] for x in items]
            m = sum(vals)/len(vals)
            sd = math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1)) if len(vals) > 1 else 0.0
            for x in items:
                x.setdefault("_z", {})[key] = ((x[key]-m)/sd) if sd > 0 else 0.0
    return rows


def mean(xs):
    return (sum(xs)/len(xs)) if xs else None


recs = []
for path in sorted(glob.glob(os.path.join(HERE, "honesty_scaling_result_*.json"))):
    d = json.load(open(path, encoding="utf-8"))
    rows = zscore_within_bins(d.get("rows", []))
    w = [r for r in rows if not r["ok"]]
    k = [r for r in rows if r["ok"]]
    recs.append({
        "name": d["model"].split("/")[-1], "acc": d["capability_accuracy"],
        "params": params_for(d["model"]),
        "zent_wrong": mean([r["_z"]["entropy"] for r in w]),
        "zent_right": mean([r["_z"]["entropy"] for r in k]),
        "zmar_wrong": mean([r["_z"]["margin"] for r in w]),
        "ent_wrong_raw": mean([r["entropy"] for r in w]),
        "ent_right_raw": mean([r["entropy"] for r in k]),
    })

recs.sort(key=lambda r: r["acc"])
print(f"{'model':24}{'acc':>7}{'pB':>5}{'zEntWrong':>11}{'zEntRight':>11}{'zMarWrong':>11}"
      f"{'rawEntWr':>10}{'rawEntRt':>10}")
for r in recs:
    print(f"{r['name']:24}{r['acc']:>7.3f}{str(r['params']):>5}"
          f"{r['zent_wrong']:>11.3f}{r['zent_right']:>11.3f}{r['zmar_wrong']:>11.3f}"
          f"{r['ent_wrong_raw']:>10.3f}{r['ent_right_raw']:>10.3f}")

acc = [r["acc"] for r in recs]
print("\n--- Spearman vs capability (acc), n=%d, EXPLORATORY ---" % len(recs))
print(f"  zEntropy-on-WRONG : {spearman(acc, [r['zent_wrong'] for r in recs]):+.3f}   "
      "(<0 => bigger models LESS uncertain when wrong = confident confab rising)")
print(f"  zEntropy-on-RIGHT : {spearman(acc, [r['zent_right'] for r in recs]):+.3f}   "
      "(>0 => bigger models MORE uncertain when right)")
print(f"  zMargin-on-WRONG  : {spearman(acc, [r['zmar_wrong'] for r in recs]):+.3f}   "
      "(>0 => bigger models more CONFIDENT (wider margin) when wrong)")
print(f"  rawEntropy-on-WRONG:{spearman(acc, [r['ent_wrong_raw'] for r in recs]):+.3f}")
