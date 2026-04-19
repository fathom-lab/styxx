# -*- coding: utf-8 -*-
"""
benchmarks/equivalence_correlate.py

Offline correlator for the consensus-proxy measurement-equivalence
protocol. Consumes two JSONL files — one from the tier-0 runner,
one from the tier-1 runner — aligned by `id`, and reports correlation
coefficients per (tier-0 metric × tier-1 probe task).

Outputs Pearson r, Spearman rho, and bootstrap 95% CI per combination.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


TIER0_METRICS = [
    "mean_entropy", "entropy_slope", "entropy_curvature",
    "entropy_volatility", "mean_top2_margin", "mean_logprob",
    "logprob_slope", "top2_slope", "mean_response_length_words",
    "first_divergence",
]


def _load_jsonl(path: Path) -> List[Dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _pearson(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    ma, mb = stats.mean(a), stats.mean(b)
    num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
    da = math.sqrt(sum((ai - ma) ** 2 for ai in a)) or 1e-9
    db = math.sqrt(sum((bi - mb) ** 2 for bi in b)) or 1e-9
    return num / (da * db)


def _rank(xs: List[float]) -> List[float]:
    indexed = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for rank, idx in enumerate(indexed):
        ranks[idx] = float(rank)
    return ranks


def _spearman(a: List[float], b: List[float]) -> float:
    return _pearson(_rank(a), _rank(b))


def _bootstrap_ci(a: List[float], b: List[float],
                  fn, n_boot: int = 2000,
                  alpha: float = 0.05,
                  seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    obs = fn(a, b)
    boots = []
    n = len(a)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(fn([a[i] for i in idx], [b[i] for i in idx]))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return obs, lo, hi


def correlate(tier0_path: Path, tier1_path: Path) -> Dict:
    tier0 = _load_jsonl(tier0_path)
    tier1 = _load_jsonl(tier1_path)
    t0_by_id = {r["id"]: r for r in tier0}
    t1_by_id = {r["id"]: r for r in tier1}

    common = sorted(set(t0_by_id) & set(t1_by_id))
    missing_t0 = set(t1_by_id) - set(t0_by_id)
    missing_t1 = set(t0_by_id) - set(t1_by_id)

    # Group tier1 records by probe_task
    tasks = defaultdict(list)
    for rid in common:
        t1r = t1_by_id[rid]
        tasks[t1r.get("probe_task", "unknown")].append(rid)

    result = {
        "n_common": len(common),
        "n_missing_t0": len(missing_t0),
        "n_missing_t1": len(missing_t1),
        "tier0_file": str(tier0_path),
        "tier1_file": str(tier1_path),
        "tasks": {},
    }

    for task, ids in tasks.items():
        t1_vals = [t1_by_id[i]["residual_score"] for i in ids]
        task_result = {"n": len(ids), "metrics": {}}
        for metric in TIER0_METRICS:
            t0_vals = [t0_by_id[i].get(metric, 0.0) for i in ids]
            if len(t0_vals) < 3:
                continue
            p_obs, p_lo, p_hi = _bootstrap_ci(t0_vals, t1_vals, _pearson)
            s_obs, s_lo, s_hi = _bootstrap_ci(t0_vals, t1_vals, _spearman)
            task_result["metrics"][metric] = {
                "pearson_r": round(p_obs, 4),
                "pearson_ci95": [round(p_lo, 4), round(p_hi, 4)],
                "spearman_rho": round(s_obs, 4),
                "spearman_ci95": [round(s_lo, 4), round(s_hi, 4)],
                "n": len(t0_vals),
            }
        # Verdict per task: max |pearson r| with CI excluding 0
        best = None
        for m, info in task_result["metrics"].items():
            r = abs(info["pearson_r"])
            ci_lo, ci_hi = info["pearson_ci95"]
            if ci_lo > 0 or ci_hi < 0:  # CI excludes 0
                if best is None or r > abs(best["r"]):
                    best = {"metric": m, "r": info["pearson_r"],
                            "ci": info["pearson_ci95"]}
        task_result["best_significant"] = best
        task_result["equivalence_verdict"] = (
            "measurement_equivalence_supported"
            if best and abs(best["r"]) >= 0.5 else
            ("correlation_present_but_weak" if best else
             "no_significant_correlation")
        )
        result["tasks"][task] = task_result

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0", required=True)
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    result = correlate(Path(args.tier0), Path(args.tier1))
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")

    print(f"\nn_common = {result['n_common']}  "
          f"(missing t0: {result['n_missing_t0']}, "
          f"missing t1: {result['n_missing_t1']})")
    for task, info in result["tasks"].items():
        print(f"\n[task {task}] n={info['n']}")
        for metric, m in info["metrics"].items():
            ci = m["pearson_ci95"]
            sig = "*" if (ci[0] > 0 or ci[1] < 0) else " "
            print(f"  {sig} {metric:26s} r={m['pearson_r']:+.3f}  "
                  f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]   "
                  f"ρ={m['spearman_rho']:+.3f}")
        if info["best_significant"]:
            b = info["best_significant"]
            print(f"  best: {b['metric']} r={b['r']:+.3f} "
                  f"CI {b['ci']}")
        print(f"  verdict: {info['equivalence_verdict']}")


if __name__ == "__main__":
    main()
