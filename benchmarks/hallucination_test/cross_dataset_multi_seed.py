# -*- coding: utf-8 -*-
"""Run cross_dataset_calibrate with multiple seeds, report mean+std."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEEDS = [31, 47, 83]
N = 200


def run(seed, with_nli):
    import os
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "hallucination_test" /
            "cross_dataset_calibrate.py"),
        "--n", str(N), "--seed", str(seed), "--no_entity",
    ]
    if with_nli:
        cmd.append("--nli")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, capture_output=True, text=True,
                        cwd=str(ROOT),
                        encoding="utf-8", errors="replace",
                        env=env)
    # Parse result JSON
    result_file = (ROOT / "benchmarks" / "hallucination_test" /
                   "results" / "cross_dataset_calibration.json")
    with open(result_file, encoding="utf-8") as f:
        return json.load(f)


def main():
    out = {}
    for label, nli in [("no_nli", False), ("with_nli", True)]:
        print(f"\n=== {label} (n={N}, seeds={SEEDS}) ===")
        aucs_per_ds = {}
        coefs_per_ds = []
        per_seed = []
        for seed in SEEDS:
            print(f"  seed {seed}...")
            res = run(seed, nli)
            coefs_per_ds.append(res.get("coefs", {}))
            per_seed.append({
                "seed": seed,
                "coefs": res.get("coefs", {}),
                "test_per_dataset": res["test_per_dataset"],
            })
            for ds, m in res["test_per_dataset"].items():
                aucs_per_ds.setdefault(ds, []).append(m["auc"])

        per_ds_stats = {}
        print(f"  dataset              mean (std) over {len(SEEDS)} seeds")
        print(f"  --------------------------------------")
        for ds, vals in aucs_per_ds.items():
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            per_ds_stats[ds] = {"mean": m, "std": std, "seeds": vals}
            print(f"  {ds:<22s} {m:.4f} +/- {std:.4f}")
        all_vals = [v for vs in aucs_per_ds.values() for v in vs]
        mean_overall = sum(all_vals) / len(all_vals)
        print(f"  --------------------------------------")
        print(f"  overall mean         {mean_overall:.4f}")

        # Average coefficients
        avg_coefs = {}
        if coefs_per_ds:
            coef_agg = {}
            for c in coefs_per_ds:
                for k, v in c.items():
                    coef_agg.setdefault(k, []).append(v)
            print(f"  --- averaged LR coefs ---")
            for k, vs in coef_agg.items():
                m = sum(vs) / len(vs)
                avg_coefs[k] = round(m, 4)
                print(f"    {k:<24s}: {m:+.4f}")

        out[label] = {
            "n": N, "seeds": SEEDS,
            "per_dataset": per_ds_stats,
            "overall_mean": mean_overall,
            "averaged_coefs": avg_coefs,
            "per_seed": per_seed,
        }

    out_path = (ROOT / "benchmarks" / "hallucination_test" /
                "results" / "multi_seed_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
