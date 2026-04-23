# -*- coding: utf-8 -*-
"""Multi-seed wrapper around cross_dataset_8bench.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEEDS = [31, 47, 83]
N = 150


def run(seed: int):
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "hallucination_test" /
            "cross_dataset_8bench.py"),
        "--n", str(N), "--seed", str(seed), "--no_entity", "--nli",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, cwd=str(ROOT), env=env,
                   capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    result_file = (ROOT / "benchmarks" / "hallucination_test" /
                   "results" / "cross_dataset_8bench.json")
    with open(result_file, encoding="utf-8") as f:
        return json.load(f)


def main():
    aucs_per_ds: dict[str, list[float]] = {}
    coefs_per_seed = []
    intercepts = []
    per_seed_results = []

    print(f"=== 8-benchmark multi-seed ({len(SEEDS)} seeds, n={N}) ===")
    for seed in SEEDS:
        print(f"\nseed {seed}...")
        r = run(seed)
        per_seed_results.append({
            "seed": seed,
            "coefs": r["coefs"],
            "intercept": r["intercept"],
            "test_per_dataset": r["test_per_dataset"],
        })
        coefs_per_seed.append(r["coefs"])
        intercepts.append(r["intercept"])
        for ds, m in r["test_per_dataset"].items():
            aucs_per_ds.setdefault(ds, []).append(m["auc"])
        for ds, m in r["test_per_dataset"].items():
            print(f"  {ds:<26s} AUC {m['auc']:.4f}")

    print(f"\n=== mean +/- std across {len(SEEDS)} seeds ===")
    per_ds_stats = {}
    for ds, vals in aucs_per_ds.items():
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        per_ds_stats[ds] = {
            "mean": round(m, 4),
            "std": round(std, 4),
            "seeds": [round(v, 4) for v in vals],
        }
        print(f"  {ds:<26s} {m:.4f} +/- {std:.4f}")
    overall_mean = (sum(v for vs in aucs_per_ds.values() for v in vs)
                    / sum(len(vs) for vs in aucs_per_ds.values()))

    # Averaged coefficients
    keys = list(coefs_per_seed[0].keys())
    avg_coefs = {}
    for k in keys:
        vs = [c[k] for c in coefs_per_seed]
        avg_coefs[k] = round(sum(vs) / len(vs), 4)
    avg_intercept = round(sum(intercepts) / len(intercepts), 4)

    print(f"\n=== averaged coefficients ===")
    for k, v in avg_coefs.items():
        print(f"  {k:<26s}: {v:+.4f}")
    print(f"  intercept              : {avg_intercept:+.4f}")
    print(f"\noverall mean AUC: {overall_mean:.4f}")

    out = {
        "n_per_dataset": N,
        "seeds": SEEDS,
        "per_dataset": per_ds_stats,
        "overall_mean": round(overall_mean, 4),
        "averaged_coefs": avg_coefs,
        "averaged_intercept": avg_intercept,
        "per_seed": per_seed_results,
    }
    out_path = (ROOT / "benchmarks" / "hallucination_test" /
                "results" / "cross_dataset_8bench_multiseed.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
