# -*- coding: utf-8 -*-
"""
run_anchor_power.py -- ship the ANCHOR THRESHOLD as a first-class styxx instrument.

The paper's section 7 established, by an exploratory simulation, that a shared blind spot which
every CONSENSUS estimator is provably blind to becomes visible from a handful of known-negative
anchors: one unanimous-wrong known-negative is a ~150x likelihood ratio, and ~20-30 anchors give
>90% power. That simulation (anchor_threshold_result.json) used a CONSERVATIVE one-sided test --
reject iff X > c rather than the most-powerful X >= c -- so its power column is a valid LOWER
BOUND. styxx.anchors now exposes the standard, tight version as design-time functions:

  anchor_lr(...)                -- single-anchor likelihood ratio (unchanged by the convention)
  blindspot_power(K, ...)       -- exact power from K known-negative anchors
  min_anchors_for_power(p, ...) -- smallest anchor budget reaching target power

This script (a) reproduces the frozen single-anchor LR to prove continuity with section 7, (b)
reports the corrected (tight) power table, (c) shows the standard test only ever raises power vs
the conservative receipt (never lowers it -- the correction is favourable, so section 7 does not
overclaim), and (d) writes anchor_power_result.json so every shipped number is receipt-bound.
"""
import json
import pathlib

from styxx import anchors

HERE = pathlib.Path(__file__).parent

# --- the section-7 design point (J=3 correlated weak panel) ---------------------------------
J, FP = 3, 0.10           # 3 judges, 10% per-judge false-positive on a known-negative (benign)
TRAP = 0.15               # 15% of true negatives are shared traps (all judges wrong)
FP_ALT = 0.0961           # non-trap negatives' fp in the frozen malign world (matches the receipt)
ALPHA = 0.05
KS = [1, 3, 5, 10, 20, 30, 50]

print("=" * 84)
print("ANCHOR THRESHOLD -- shipped instrument vs the frozen section-7 receipt")
print("=" * 84)

# (a) single-anchor LR -- must match the frozen 150.8 to 1 dp
lr = anchors.anchor_lr(J=J, fp_rate=FP, trap_rate=TRAP, fp_rate_alt=FP_ALT)
print(f"\nsingle-anchor likelihood ratio (blind-spot vs benign): {lr:.1f}x")

# (b) corrected (tight, most-powerful) power table
print("\n  K   reject_at   alpha_actual   power(tight)")
tight = {}
for K in KS:
    r = anchors.blindspot_power(K, J=J, fp_rate=FP, trap_rate=TRAP, fp_rate_alt=FP_ALT, alpha=ALPHA)
    tight[K] = round(r["power"], 4)
    print(f"{K:>3}   {str(r['reject_at']):>7}   {r['alpha_actual']:>10.4f}   {r['power']:>10.4f}")

# (c) compare to the frozen conservative receipt -- the correction must be >= at every K
frozen_path = HERE / "anchor_threshold_result.json"
frozen = json.loads(frozen_path.read_text())
frozen_pow = {int(k): v for k, v in frozen["power_by_K"].items()}
print("\n  K   power(section-7, conservative)   power(shipped, tight)   delta")
all_favourable = True
for K in KS:
    d = tight[K] - frozen_pow[K]
    all_favourable &= (d >= -1e-9)
    print(f"{K:>3}   {frozen_pow[K]:>28.4f}   {tight[K]:>21.4f}   {d:>+7.4f}")
print(f"\ncorrection favourable at every K (tight >= conservative): {all_favourable}")
lr_matches = abs(lr - frozen['single_anchor_LR']) < 0.05
print(f"single-anchor LR matches frozen receipt ({frozen['single_anchor_LR']}): {lr_matches}")

# (d) the design-time answer the datasheet raises: minimum anchor budget
min90 = anchors.min_anchors_for_power(0.90, J=J, fp_rate=FP, trap_rate=TRAP, fp_rate_alt=FP_ALT, alpha=ALPHA)
min95 = anchors.min_anchors_for_power(0.95, J=J, fp_rate=FP, trap_rate=TRAP, fp_rate_alt=FP_ALT, alpha=ALPHA)
print(f"\nminimum known-negative anchors for 0.90 power: {min90['K']}  (power {min90['power']:.4f})")
print(f"minimum known-negative anchors for 0.95 power: {min95['K']}  (power {min95['power']:.4f})")

# guardrails: the whole point is that section 7's claim holds and is in fact conservative
assert lr_matches, "single-anchor LR drifted from the frozen receipt"
assert all_favourable, "tight power fell below the conservative receipt somewhere -- investigate"
assert min90["K"] <= 30, "shipped threshold should be <= the paper's ~20-30 (it is tighter)"
assert anchors.blindspot_power(1, J=J, fp_rate=FP, trap_rate=TRAP, fp_rate_alt=FP_ALT)["power"] > 0.0, \
    "single anchor must have nonzero power -- consistent with the 150x LR headline"

receipt = {
    "design_point": {"J": J, "fp_rate": FP, "trap_rate": TRAP, "fp_rate_alt": FP_ALT, "alpha": ALPHA},
    "single_anchor_lr": round(lr, 1),
    "power_tight_by_K": tight,
    "power_conservative_by_K_section7": {str(K): frozen_pow[K] for K in KS},
    "correction_favourable_all_K": bool(all_favourable),
    "single_anchor_lr_matches_section7": bool(lr_matches),
    "min_anchors_power_0.90": min90["K"],
    "min_anchors_power_0.95": min95["K"],
    "single_anchor_power": round(anchors.blindspot_power(1, J=J, fp_rate=FP, trap_rate=TRAP,
                                                         fp_rate_alt=FP_ALT)["power"], 4),
}
(HERE / "anchor_power_result.json").write_text(json.dumps(receipt, indent=2))
print("\nwrote anchor_power_result.json (receipt)")
