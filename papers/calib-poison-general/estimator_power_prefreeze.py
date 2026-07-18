"""PRE-RUN power arithmetic for the coupling v4 dose-response estimator (panel #7, R5.4).

Zero GPU. Deterministic. Emits `_estimator_power_prefreeze.json`.

WHY THIS EXISTS
The prereg gates COUPLED on a per-seed OLS slope of the paired delta on erased_rank, fit over
RANK_SPAN=[2,8] with step-0 excluded, and sets the decision bar MIN_EFFECT_SLOPE=0.0152 to be exactly
the minimum price of interest (0.0909) divided by the span (6). PREREG:349-350 states the consequence
without pricing it: "a seed paying precisely the minimum sustained price sits AT the bar". This script
prices it. The design's power was never computed before the freeze -- which is panel #6's standing
threshold-provenance major, one iteration later.

THE THREE RESULTS
  1. SYMMETRY CEILING (assumption-free). If a seed paying exactly the price of interest sits exactly AT
     the decision bar, then for ANY unbiased estimator with symmetric noise P(that seed clears the bar)
     = 0.5 exactly. A strict-majority-of-seeds rule therefore has power <= ~0.5 at the price of
     interest, for any battery, any noise level, and any seed count. This is a property of where the
     bar was placed, not of the instrument.
  2. MORE SEEDS MAKES IT WORSE. The frozen rule also requires zero seeds at or below -bar, and that
     clause decays as (1-q)^n, so power FALLS as seeds are added.
  3. THE STATISTIC IS ONE CHECKPOINT. Panel #7/E3: the accumulate and fixed arms are bit-identical
     through step 24, so the rank-4 paired delta is a STRUCTURAL ZERO; with fitted ranks {4,6,8} the
     OLS slope reduces exactly to d8/4 (the rank-6 point has zero leverage). At minimum selection the
     aggregate lives on a 1/48 lattice, so `slope >= 0.0152` realizes as `d8 >= 3/48`: the entire
     per-seed verdict is "did the accumulate arm get at least 3 more of 48 items wrong at step 75".
"""
import json
import numpy as np

# ---- frozen design constants (coupling_confirm_v4.py:107-123, PREREG:267-294) ----
PRICE_OF_INTEREST = 0.0909      # the only capability price this program has ever measured
SPAN = (2, 8)
MIN_EFFECT_SLOPE = 0.0152       # PRICE_OF_INTEREST / (8 - 2), rounded to 4dp
N_SEEDS = 5
FITTED_RANKS = (4.0, 6.0, 8.0)  # step-0 (rank 2) excluded; audits fire once per refit
SE_AGG_PREREG = 0.0722          # PREREG: one per-checkpoint aggregate SE at minimum selection
N_ITEMS_MIN = 48                # MIN_DISJOINT=3 sub-tasks x 16 items
CLEAN_FLOOR = 0.90              # DISJOINT_FLOOR_CLEAN: selection floor on base accuracy

RNG = np.random.default_rng(0)
N_SIM = 200_000


def slope_se_from_agg_se(se_agg, ranks=FITTED_RANKS, structural_zero=True):
    """Per-seed OLS slope SE. With the rank-4 structural zero the anchor is noiseless and the slope is
    exactly d8/4; without it, all three points carry independent paired-delta noise."""
    sigma_delta = se_agg * np.sqrt(2.0)          # delta of two independently measured aggregates
    r = np.asarray(ranks, float)
    sxx = ((r - r.mean()) ** 2).sum()
    if structural_zero:
        # slope = sum((r-rbar)*y)/sxx with y[0] == 0 exactly and y[1] at zero leverage -> d8*2/8
        return sigma_delta * 2.0 / sxx
    return sigma_delta / np.sqrt(sxx)


def rates(true_slope, se, bar=MIN_EFFECT_SLOPE, n_seeds=N_SEEDS, n_sim=N_SIM):
    """Frozen rule (coupling_confirm_v4.py:421-425): COUPLED iff strict majority at/above bar AND zero
    seeds at/below -bar. bounded_null iff strict majority below bar in magnitude."""
    s = RNG.normal(true_slope, se, size=(n_sim, n_seeds))
    above = (s >= bar).sum(1)
    below_neg = (s <= -bar).sum(1)
    small = (np.abs(s) < bar).sum(1)
    coupled = (above > n_seeds / 2) & (below_neg == 0)
    bounded = (~coupled) & (small > n_seeds / 2)
    return {"coupled": float(coupled.mean()), "bounded_null": float(bounded.mean()),
            "partial": float(1.0 - coupled.mean() - bounded.mean())}


def mde80(se, bar=MIN_EFFECT_SLOPE, n_seeds=N_SEEDS):
    lo, hi = bar, bar * 30
    for _ in range(60):
        mid = (lo + hi) / 2
        if rates(mid, se, bar, n_seeds, n_sim=60_000)["coupled"] < 0.80:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


out = {"purpose": "pre-run power arithmetic for the coupling v4 dose-response estimator (panel #7 R5.4)",
       "gpu_hours": 0, "constants": {
           "price_of_interest": PRICE_OF_INTEREST, "min_effect_slope": MIN_EFFECT_SLOPE,
           "rank_span": list(SPAN), "fitted_ranks": list(FITTED_RANKS), "n_seeds": N_SEEDS,
           "se_agg_prereg": SE_AGG_PREREG, "n_items_min_selection": N_ITEMS_MIN}}

print("=" * 88)
print("1. SYMMETRY CEILING -- assumption-free")
print("=" * 88)
se_prereg = slope_se_from_agg_se(SE_AGG_PREREG)
ceiling = []
for mult in (2.0, 1.0, 0.5, 0.25, 0.1, 0.02):
    se = se_prereg * mult
    r = rates(MIN_EFFECT_SLOPE, se)
    ceiling.append({"slope_se": round(se, 6), "se_over_bar": round(se / MIN_EFFECT_SLOPE, 3),
                    **{k: round(v, 4) for k, v in r.items()}})
    print(f"  slope SE {se:.5f} ({se/MIN_EFFECT_SLOPE:5.2f}x bar) -> COUPLED {r['coupled']:.4f}  "
          f"bounded_null {r['bounded_null']:.4f}  PARTIAL {r['partial']:.4f}")
print("\n  Power at the price of interest never exceeds ~0.5 no matter how small the noise gets:")
print("  the bar sits exactly AT the effect, so each seed is a coin flip by symmetry.")
out["symmetry_ceiling"] = ceiling

print()
print("=" * 88)
print("2. ADDING SEEDS MAKES IT WORSE")
print("=" * 88)
seedsweep = []
for n in (3, 4, 5, 7, 9, 11):
    r = rates(MIN_EFFECT_SLOPE, se_prereg, n_seeds=n)
    seedsweep.append({"n_seeds": n, **{k: round(v, 4) for k, v in r.items()}})
    print(f"  n_seeds {n:2d} -> COUPLED {r['coupled']:.4f}  bounded_null {r['bounded_null']:.4f}  "
          f"PARTIAL {r['partial']:.4f}")
print("\n  The zero-negatives clause decays as (1-q)^n, so power falls as seeds are added.")
out["seed_sweep"] = seedsweep

print()
print("=" * 88)
print("3. THE PER-SEED STATISTIC IS ONE CHECKPOINT ON A 48-ITEM LATTICE  (panel #7 / E3)")
print("=" * 88)
lattice = 1.0 / N_ITEMS_MIN
d8_needed = MIN_EFFECT_SLOPE * 4.0
items_needed = int(np.ceil(d8_needed * N_ITEMS_MIN))
print(f"  fitted ranks {FITTED_RANKS}: rank 4 delta is a STRUCTURAL ZERO (arms bit-identical to step 24)")
print(f"  rank 6 carries zero leverage  ->  slope == d8 / 4  exactly")
print(f"  bar {MIN_EFFECT_SLOPE} <=> d8 >= {d8_needed:.4f} = {items_needed}/{N_ITEMS_MIN} items")
print(f"  aggregate lattice at minimum selection: 1/{N_ITEMS_MIN} = {lattice:.4f}")
print(f"  => the per-seed COUPLED test is: 'did the accumulate arm get at least {items_needed} more of")
print(f"     {N_ITEMS_MIN} items wrong than the fixed arm, at step 75'")
out["structural_reduction"] = {
    "slope_equals": "d8 / 4", "rank4_structural_zero": True, "rank6_leverage": 0.0,
    "bar_in_items": items_needed, "n_items": N_ITEMS_MIN,
    "aggregate_lattice": round(lattice, 6),
    "note": ("the rank-4 paired delta is exactly 0 because both arms train bit-identically through "
             "step 24; with fitted ranks 4,6,8 the OLS slope reduces to d8/4 and the whole per-seed "
             "verdict rests on ONE checkpoint comparison")}

print()
print("=" * 88)
print("4. CAN ANY DECISION BAR CLEAR BOTH LEGS?  (sensitivity >= 0.80 at the price, FPR <= 0.15)")
print("=" * 88)
print(f"{'configuration':44s} {'slopeSE':>9s} {'bestbar':>8s} {'sens':>6s} {'fpr':>6s}  verdict")
pareto = []
CONFIGS = [
    ("as designed (48 items, prereg SE)",            SE_AGG_PREREG,        N_ITEMS_MIN),
    ("as designed, SE at the 0.90 selection floor",  None,                 N_ITEMS_MIN),
    ("all 7 disjoint sub-tasks (112 items)",         None,                 112),
    ("4x battery (192 items)",                       None,                 192),
    ("9x battery (432 items)",                       None,                 432),
    ("16x battery (768 items)",                      None,                 768),
]
for label, se_agg, n_items in CONFIGS:
    if se_agg is None:  # binomial SE at the selection floor, two independent arms
        se_agg = np.sqrt(CLEAN_FLOOR * (1 - CLEAN_FLOOR) / n_items)
        se = slope_se_from_agg_se(se_agg)
    else:
        se = slope_se_from_agg_se(se_agg)
    best = None
    for bar in np.linspace(0.0005, MIN_EFFECT_SLOPE * 2.0, 80):
        sens = rates(MIN_EFFECT_SLOPE, se, bar=bar, n_sim=40_000)["coupled"]
        fpr = rates(0.0, se, bar=bar, n_sim=40_000)["coupled"]
        if fpr <= 0.15 and (best is None or sens > best[1]):
            best = (bar, sens, fpr)
    ok = bool(best and best[1] >= 0.80)
    pareto.append({"configuration": label, "n_items": n_items, "se_agg": round(float(se_agg), 5),
                   "slope_se": round(float(se), 6), "best_bar": round(float(best[0]), 5) if best else None,
                   "sensitivity_at_price": round(float(best[1]), 4) if best else None,
                   "false_positive": round(float(best[2]), 4) if best else None,
                   "clears_both_legs": ok})
    print(f"{label:44s} {se:9.5f} {best[0]:8.4f} {best[1]:6.3f} {best[2]:6.3f}  "
          f"{'PASSES' if ok else 'CANNOT'}")
out["decision_bar_pareto"] = pareto

print()
print("=" * 88)
print("5. MDE80 AT THE FROZEN BAR")
print("=" * 88)
m = mde80(se_prereg)
out["mde80"] = {"slope": round(float(m), 5), "multiple_of_bar": round(float(m / MIN_EFFECT_SLOPE), 2),
                "span_scale_bound": round(float(m * (SPAN[1] - SPAN[0])), 4),
                "price_of_interest": PRICE_OF_INTEREST}
print(f"  MDE80 slope {m:.5f} = {m/MIN_EFFECT_SLOPE:.2f}x the bar")
print(f"  span-scale bound {m*(SPAN[1]-SPAN[0]):.4f} vs the price of interest {PRICE_OF_INTEREST}")
print(f"  => a bounded null earns 'no paired price above ~{m*(SPAN[1]-SPAN[0]):.2f} over the span',")
print(f"     which is ~{m/MIN_EFFECT_SLOPE:.1f}x weaker than the question the paper asks.")

out["conclusion"] = (
    "The design cannot answer its own question at conventional power. Power at the price of interest "
    "is bounded by ~0.5 by symmetry alone (the bar was placed exactly AT the price), lands near 0.24 at "
    "the prereg's own noise figure, and FALLS as seeds are added. The most likely single outcome of the "
    "scored run is PARTIAL. The only lever that raises power is a smaller per-checkpoint aggregate SE, "
    "i.e. more items per sub-task -- more GPU, not more seeds. Widening RANK_SPAN is not a legitimate "
    "lever: ranks 10-24 are the read-recovery region. This is knowable pre-freeze at zero GPU cost.")
print()
print(out["conclusion"])

with open("_estimator_power_prefreeze.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("\nwrote _estimator_power_prefreeze.json")
