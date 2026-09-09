"""Analysis of the MMLU batch-nuisance run, fixed by the preregistration plus one labelled addition.

The preregistration (papers/v8/batch_nuisance_mmlu/PREREG_batch_nuisance_scores_2026_09_08.md)
fixed H1-H4, the kill gates, and the reported quantities. It did NOT fix the comparison against the
benchmark's own sampling error; a prior-art report received after the run recommended it, so it is
computed here and labelled POST-HOC throughout. Nothing preregistered is recomputed or reinterpreted.

The three quantities are kept separate and never collapsed, because they say different things:
  per-item flip rate  - how often the procedure changes an answer
  signed net delta    - how much the headline moves after flips cancel
  net delta in SE     - whether that movement is small next to the benchmark's own precision
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "receipt.json").read_text(encoding="utf-8"))

NUISANCE = ["repeat_bs1", "bs8", "bs32", "bs8_perm11", "bs32_perm12"]
print(f"benchmark: {r['benchmark']['name']} {r['benchmark']['split']}  "
      f"n={r['benchmark']['total_items']}  excluded={len(r['benchmark']['excluded'])}")
print(f"subjects: {', '.join(r['benchmark']['subjects'])}")
print(f"scoring: {r['benchmark']['scoring']}")
print(f"runtime: torch {r['runtime']['torch']}, transformers {r['runtime']['transformers']}, "
      f"{r['runtime']['gpu']}\n")

for label, m in r["models"].items():
    pv = m["per_variant"]
    n = pv["ref_bs1"]["n"]
    ref_acc = pv["ref_bs1"]["accuracy"]
    se_pp = math.sqrt(ref_acc * (1 - ref_acc) / n) * 100.0     # POST-HOC

    print("=" * 78)
    print(f"{label}   rev {m['model']['revision'][:12]}")
    print(f"kill gates: {m['kill_gates']}")
    if not all(m["kill_gates"].values()):
        print("  *** A KILL GATE FAILED: the preregistration voids this run. ***")
    print()
    print(f"  {'variant':13s} {'acc':>7s} {'correct':>8s} {'delta_pp':>9s} "
          f"{'flips':>6s} {'flip_rate':>10s} {'identical_margins':>18s}")
    for name, v in pv.items():
        tag = "" if name in NUISANCE or name == "ref_bs1" else "   <- precision arm, not a nuisance"
        d_pp = (v["accuracy"] - ref_acc) * 100.0
        fl = v.get("flips_vs_ref", 0)
        print(f"  {name:13s} {v['accuracy']:7.4f} {v['correct']:8d} {d_pp:+9.3f} "
              f"{fl:6d} {fl / n:10.4f} {v.get('n_identical_margin', 0):18d}{tag}")

    accs = [pv[x]["accuracy"] for x in (["ref_bs1"] + NUISANCE) if x in pv]
    spread_pp = (max(accs) - min(accs)) * 100.0
    union = m["n_union_nuisance_flips"]
    net_max_pp = max(abs(pv[x]["accuracy"] - ref_acc) for x in NUISANCE if x in pv) * 100.0

    print()
    print(f"  PREREGISTERED QUANTITIES")
    print(f"    H1  any nuisance variant differs in accuracy : "
          f"{'YES' if any(pv[x]['accuracy'] != ref_acc for x in NUISANCE if x in pv) else 'NO'}")
    print(f"    H2  items whose prediction changed (union)   : {union} of {n} "
          f"= {union / n:.4f}  (H2 wanted >0 and <10%: "
          f"{'PASS' if 0 < union < 0.10 * n else 'FAIL'})")
    print(f"    H3  mean reference margin, flipped items     : {m['mean_ref_margin_flipped']}")
    print(f"        mean reference margin, unflipped items   : {m['mean_ref_margin_unflipped']}")
    if m["mean_ref_margin_flipped"] is not None:
        print(f"        H3 direction (flipped < unflipped)       : "
              f"{'PASS' if m['mean_ref_margin_flipped'] < m['mean_ref_margin_unflipped'] else 'FAIL'}")
    print(f"    H4  accuracy spread across nuisance variants : {spread_pp:.3f} pp "
          f"(threshold 0.5 pp: {'PASS' if spread_pp >= 0.5 else 'FAIL'})")
    print()
    print(f"  POST-HOC, NOT PREREGISTERED (added on a prior-art recommendation received after the run)")
    print(f"    binomial standard error of this metric at n={n}, p={ref_acc:.4f} : {se_pp:.3f} pp")
    print(f"    largest net nuisance delta                                   : {net_max_pp:.3f} pp "
          f"= {net_max_pp / se_pp:.2f} SE")
    print(f"    per-item flips vs net change: {union} items changed answer, "
          f"net headline moved {net_max_pp:.3f} pp = {net_max_pp / 100 * n:.1f} items")
    if union > 0:
        print(f"    cancellation ratio (flips per net item)                      : "
              f"{union / max(1e-9, net_max_pp / 100 * n):.2f}x")
    print()
    print(f"  PRECISION ARM (declared change, reported separately, never in the spread)")
    print(f"    {m['precision_arm']}")
    print()

print("=" * 78)
print("Reminder of what the preregistration forbids claiming: not that published results are")
print("wrong; not that this transfers to other models, harnesses, kernels, hardware or")
print("benchmarks; not novelty (the prior-art report names Yuan et al. arXiv 2506.09501,")
print("Thinking Machines 2025, Hochlehnert et al. COLM 2025 and Pape et al. arXiv 2605.19537).")
