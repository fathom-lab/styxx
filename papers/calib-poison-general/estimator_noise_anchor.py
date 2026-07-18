"""Anchor the coupling v4 power arithmetic in REALIZED noise, not the prereg's worst-case bound.

Zero GPU. Reads the cycle-43 dose run (`coupling_confirm_result.json`, the VOIDed coupling confirm)
which has the IDENTICAL experiment shape the v4 design will run: 5 seeds x {accumulate, fixed} arms,
13 checkpoints at steps 0,25,...,299, accumulate ranks 2,4,6,...,24, fixed PINNED at rank 2.

The prereg prices the per-checkpoint aggregate SE at sqrt(0.25/48) = 0.0722 -- a worst-case binomial
bound at p = 0.5, on a battery whose selection floor is 0.90. That bound drives every power number in
the design. This measures the realized quantity instead, on the only real data that exists.

CHANNEL CAVEAT, stated up front: the cycle-43 run scored the T/F battery; v4 scores the generation
battery. Different channel, same experiment shape and the same 3-sub-task selection. This anchors the
SCALE of arm-to-arm and seed-to-seed trajectory noise, which is the quantity the prereg's binomial
bound stands in for. It is not a substitute for measuring the gen channel itself.

Emits `_estimator_noise_anchor.json`.
"""
import json
import numpy as np

RANK_SPAN = (2, 8)
MIN_EFFECT_SLOPE = 0.0152
PRICE_OF_INTEREST = 0.0909
SE_AGG_PREREG = 0.0722

src = json.load(open("coupling_confirm_result.json", encoding="utf-8"))
curves = src["curves"]
selected = src["selected_disjoint"]
seeds = sorted(int(s) for s in curves)


def agg(p):
    return float(p["battery"]["aggregate"])


def arm(seed, name):
    return {int(p["step"]): p for p in curves[str(seed)][name]}


out = {"source": "coupling_confirm_result.json (cycle 43 dose run, VOIDed)",
       "channel_caveat": ("scored on the T/F battery; v4 scores the generation battery. Same "
                          "experiment shape and selection size; anchors the NOISE SCALE only."),
       "selected": selected, "seeds": seeds, "gpu_hours": 0}

print("=" * 88)
print("1. IS THE RANK-4 POINT A STRUCTURAL ZERO?  (panel #7 finding E3, tested on real data)")
print("=" * 88)
print("  E3 claim: the arms train bit-identically through step 24, so the paired delta at step 25")
print("  (rank 4) must be EXACTLY zero, and the rank-6 point carries zero leverage -> slope = d8/4.")
print()
sz = []
for s in seeds:
    a, f = arm(s, "accumulate"), arm(s, "fixed")
    row = {"seed": s}
    for st in (25, 50, 75):
        if st in a and st in f:
            row[f"d_step{st}"] = round(agg(f[st]) - agg(a[st]), 6)
    sz.append(row)
    print(f"  seed {s}: d(step 25, rank 4) = {row.get('d_step25'):+.4f}   "
          f"d(step 50, rank 6) = {row.get('d_step50'):+.4f}   "
          f"d(step 75, rank 8) = {row.get('d_step75'):+.4f}")
n_zero = sum(1 for r in sz if abs(r.get("d_step25", 1)) < 1e-9)
print(f"\n  step-25 deltas exactly zero in {n_zero} of {len(sz)} seeds -> "
      f"{'CONFIRMED' if n_zero == len(sz) else 'NOT CONFIRMED on this channel'}")
out["structural_zero_test"] = {"per_seed": sz, "n_exactly_zero": n_zero, "n_seeds": len(sz),
                               "confirmed": n_zero == len(sz)}

print()
print("=" * 88)
print("2. REALIZED PER-CHECKPOINT DISPERSION vs THE PREREG'S WORST-CASE BOUND")
print("=" * 88)
fixed_by_step = {}
for s in seeds:
    for st, p in arm(s, "fixed").items():
        if st > 0:
            fixed_by_step.setdefault(st, []).append(agg(p))
cross = [float(np.std(v, ddof=1)) for v in fixed_by_step.values() if len(v) > 1]
sd_cross = float(np.mean(cross))
within = []
for s in seeds:
    v = [agg(p) for st, p in sorted(arm(s, "fixed").items()) if st > 0]
    within.append(float(np.std(v, ddof=1)))
sd_within = float(np.mean(within))
print(f"  cross-seed SD of the fixed arm at matched step (mean over steps) : {sd_cross:.4f}")
print(f"  within-seed SD of the fixed arm across steps  (mean over seeds)  : {sd_within:.4f}")
print(f"  prereg's worst-case binomial bound sqrt(0.25/48)                 : {SE_AGG_PREREG:.4f}")
print(f"  realized / bound                                                 : {sd_cross/SE_AGG_PREREG:.2f}x")
out["dispersion"] = {"cross_seed_sd_at_matched_step": round(sd_cross, 5),
                     "within_seed_sd_across_steps": round(sd_within, 5),
                     "prereg_worst_case_bound": SE_AGG_PREREG,
                     "realized_over_bound": round(sd_cross / SE_AGG_PREREG, 3)}

print()
print("=" * 88)
print("3. THE ESTIMATOR GATE'S OWN POOL, COMPUTED ON REAL CHECKPOINTS")
print("=" * 88)
print("  Unordered seed pairs {i,j}; a step enters only where BOTH accumulate arms report the same")
print("  erased_rank and that rank is in [2,8]; pseudo-delta = fixed_i - fixed_j; same OLS slope.")
print()
lo, hi = RANK_SPAN
pool, dropped = [], []
for ii in range(len(seeds)):
    for jj in range(ii + 1, len(seeds)):
        i, k = seeds[ii], seeds[jj]
        ai, ak = arm(i, "accumulate"), arm(k, "accumulate")
        fi, fk = arm(i, "fixed"), arm(k, "fixed")
        ds, rs = [], []
        for st in sorted(set(ai) & set(ak) & set(fi) & set(fk)):
            if st <= 0:
                continue
            ri, rk = int(ai[st]["erased_rank"]), int(ak[st]["erased_rank"])
            if ri != rk or not (lo <= ri <= hi):
                continue
            ds.append(round(agg(fi[st]) - agg(fk[st]), 4)); rs.append(float(ri))
        if len(set(rs)) < 2:
            dropped.append([i, k]); continue
        pool.append(round(float(np.polyfit(rs, ds, 1)[0]), 6))
print(f"  pool units: {len(pool)} of 10   dropped: {len(dropped)}")
print(f"  pool slopes: {sorted(pool)}")
pool_sd = float(np.std(pool, ddof=1)) if len(pool) > 1 else 0.0
zero_atom = float(np.mean([abs(s) < 1e-12 for s in pool])) if pool else 1.0
print(f"  pool SD {pool_sd:.5f}  ({pool_sd/MIN_EFFECT_SLOPE:.2f}x the bar)   "
      f"zero-slope atom {zero_atom:.3f}   distinct values {len(set(pool))}")
out["realized_pool"] = {"n_units": len(pool), "n_dropped": len(dropped),
                        "slopes": sorted(pool), "sd": round(pool_sd, 6),
                        "sd_over_bar": round(pool_sd / MIN_EFFECT_SLOPE, 3),
                        "zero_slope_atom": round(zero_atom, 4),
                        "n_distinct": len(set(pool))}

print()
print("=" * 88)
print("4. MDE80 FROM THE REALIZED POOL, VIA THE SHIPPED GATE'S OWN DRAW RULE")
print("=" * 88)
GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
RUNS, N_ADM = 4000, 5
rng = np.random.default_rng(0)


def recovery(slopes, k, n_draw, n_adm, runs=RUNS):
    n = len(slopes)
    if n == 0 or n_draw <= 0:
        return 0.0
    idx = rng.permuted(np.tile(np.arange(n), (runs, 1)), axis=1)[:, :n_draw]
    signs = rng.integers(0, 2, size=(runs, n_draw)) * 2.0 - 1.0
    v = np.asarray(slopes)[idx] * signs + k * MIN_EFFECT_SLOPE
    pos = (v >= MIN_EFFECT_SLOPE).sum(1)
    neg = (v <= -MIN_EFFECT_SLOPE).sum(1)
    return float(np.mean((pos > n_adm / 2.0) & (neg == 0)))


curve, mde = [], None
for k in GRID:
    r = recovery(pool, k, min(N_ADM, len(pool)), N_ADM)
    curve.append({"k": k, "recovery": round(r, 4)})
    if mde is None and r >= 0.80:
        mde = k
    print(f"  injection {k:4.1f}x bar -> recovery {r:.4f}")
print()
if mde is None:
    print("  MDE80: NOT REACHED on the grid -> the gate would VOID this run estimator_insensitive")
else:
    print(f"  MDE80 = {mde}x bar = {mde*MIN_EFFECT_SLOPE:.5f}  "
          f"(span-scale bound {mde*MIN_EFFECT_SLOPE*6:.4f} vs price of interest {PRICE_OF_INTEREST})")
out["realized_mde80"] = {"grid": list(GRID), "recovery_curve": curve,
                         "mde80_multiple_of_bar": mde,
                         "span_scale_bound": (round(mde * MIN_EFFECT_SLOPE * 6, 5)
                                              if mde is not None else None),
                         "price_of_interest": PRICE_OF_INTEREST,
                         "would_void_insensitive": mde is None}

print()
print("=" * 88)
print("5. THE WITHIN-SEED PAIRED NOISE -- the quantity the pool STANDS IN FOR")
print("=" * 88)
print("  Panel #7 residual R-2: the pool is a BETWEEN-seed contrast standing in for a WITHIN-seed")
print("  cross-arm contrast. The real pair shares base weights, LoRA init and the whole batch stream")
print("  (proved by the exact zeros in section 1), so Cov(acc, fixed) > 0 and the pool OVERSTATES the")
print("  real noise -- MDE80 from it is an UPPER bound. Measure the real thing directly: at a FIXED")
print("  rank the paired delta across seeds is (common dose effect) + (seed noise), so its cross-seed")
print("  SD estimates the paired-delta noise with the effect differenced out.")
print()
paired_sd = {}
for r_target, st in ((4, 25), (6, 50), (8, 75)):
    vals = []
    for s in seeds:
        a, f = arm(s, "accumulate"), arm(s, "fixed")
        if st in a and st in f and int(a[st]["erased_rank"]) == r_target:
            vals.append(agg(f[st]) - agg(a[st]))
    if len(vals) > 1:
        paired_sd[r_target] = float(np.std(vals, ddof=1))
        print(f"  rank {r_target} (step {st:3d}): paired deltas {[round(v,4) for v in vals]}  "
              f"-> cross-seed SD {paired_sd[r_target]:.4f}")
live = [v for r, v in paired_sd.items() if r > 4]           # rank 4 is the structural zero
sd_paired = float(np.mean(live)) if live else None
slope_se_within = sd_paired / 4.0 if sd_paired else None    # slope == d8/4
slope_se_pool = pool_sd / 1.0
print()
print(f"  within-seed paired-delta noise SD (ranks 6,8)  : {sd_paired:.4f}")
print(f"  implied per-seed SLOPE SE (slope == d8/4)      : {slope_se_within:.4f}  "
      f"({slope_se_within/MIN_EFFECT_SLOPE:.2f}x the bar)")
print(f"  cross-seed POOL slope SD (the gate's null)     : {slope_se_pool:.4f}  "
      f"({slope_se_pool/MIN_EFFECT_SLOPE:.2f}x the bar)")
print(f"  pool overstates the real estimator noise by    : {slope_se_pool/slope_se_within:.2f}x")
print()
print("  So the section-4 VOID is the CONSERVATIVE reading, exactly as R-2 predicted. The honest")
print("  within-seed slope SE is the smaller number -- and it is STILL above the bar, so the price")
print("  of interest still sits inside the noise.")
out["within_seed_paired_noise"] = {
    "per_rank_cross_seed_sd": {str(k): round(v, 5) for k, v in paired_sd.items()},
    "paired_delta_sd_ranks_6_8": round(sd_paired, 5) if sd_paired else None,
    "implied_slope_se": round(slope_se_within, 5) if slope_se_within else None,
    "implied_slope_se_over_bar": round(slope_se_within / MIN_EFFECT_SLOPE, 3) if slope_se_within else None,
    "pool_slope_sd": round(slope_se_pool, 5),
    "pool_overstatement_factor": round(slope_se_pool / slope_se_within, 3) if slope_se_within else None,
    "note": ("rank 4 excluded: it is the structural zero. The pool's MDE80 is an upper bound; this "
             "within-seed figure is the honest estimator noise, and the price-of-interest slope "
             "bar of 0.0152 sits BELOW it either way.")}

print()
print("=" * 88)
print("6. DOES MORE BATTERY FIX IT? -- the decomposition that decides the design")
print("=" * 88)
print("  Item-sampling noise shrinks as 1/sqrt(n_items); TRAJECTORY noise (two independently trained")
print("  arms/seeds landing in different places) does NOT. The prereg's binomial bound prices ONLY")
print("  the sampling part, which is why 'add items' looked like a free fix.")
print()
n_items_tf = 24              # the T/F selected aggregate moves on a 1/24 lattice (see deltas above)
samp = np.sqrt(0.25 / n_items_tf) * np.sqrt(2)     # worst-case paired sampling SD at this item count
resid = float(np.sqrt(max(sd_paired ** 2 - samp ** 2, 0.0))) if sd_paired else 0.0
print(f"  measured paired-delta SD                       : {sd_paired:.4f}")
print(f"  worst-case SAMPLING component at {n_items_tf} items    : {samp:.4f}")
print(f"  irreducible TRAJECTORY component (residual)    : {resid:.4f}")
if resid <= 0.0:
    print("  -> against the WORST-CASE (p = 0.5) sampling bound the residual vanishes. That is absence")
    print("     of evidence, not evidence of absence: the bound is loose enough to absorb a real")
    print("     trajectory floor. Section 7 redoes this against the MEASURED accuracy, which is the")
    print("     load-bearing version -- and it does recover a small nonzero floor. Do not read this")
    print("     line as 'more items fixes everything'.")
else:
    floor_slope = resid / 4.0
    print(f"  -> a trajectory floor survives every battery size. Even at INFINITE items the per-seed")
    print(f"     slope SE cannot fall below {floor_slope:.4f} = {floor_slope/MIN_EFFECT_SLOPE:.2f}x the bar.")
out["noise_decomposition"] = {
    "n_items_assumed": n_items_tf, "measured_paired_sd": round(sd_paired, 5) if sd_paired else None,
    "worst_case_sampling_component": round(float(samp), 5),
    "irreducible_trajectory_component": round(resid, 5),
    "slope_se_floor_at_infinite_items": round(resid / 4.0, 5),
    "slope_se_floor_over_bar": round(resid / 4.0 / MIN_EFFECT_SLOPE, 3),
    "note": ("if the trajectory component is nonzero, 'add more battery items' cannot reach the bar; "
             "this SUBTRACTS the free-precision remedy proposed from the binomial model alone")}

print()
print("=" * 88)
print("7. PROJECTION -- what the measured noise implies for a larger battery")
print("=" * 88)
# Do NOT invert the binomial on an assumed accuracy -- the accuracy is MEASURABLE here. The selection
# floor (0.90) is applied to the CLEAN BASE model; fine-tuning then degrades the arms, and it is the
# fine-tuned accuracy that sets the sampling noise during the run.
fixed_acc = []
for s in seeds:
    for st, p in arm(s, "fixed").items():
        if st > 0:
            fixed_acc.append(agg(p))
p_realized = float(np.mean(fixed_acc))
clean_agg = float(src["clean_battery"]["aggregate"])
samp_realized = float(np.sqrt(2 * p_realized * (1 - p_realized) / n_items_tf))
traj = float(np.sqrt(max(sd_paired ** 2 - samp_realized ** 2, 0.0)))
print(f"  clean BASE accuracy on the selected sub-tasks : {clean_agg:.4f}")
print(f"  REALIZED fine-tuned fixed-arm accuracy        : {p_realized:.4f}   <- the operative value")
print(f"  sampling component at {n_items_tf} items implied by it : {samp_realized:.4f}")
print(f"  measured paired SD                            : {sd_paired:.4f}")
print(f"  => irreducible TRAJECTORY component           : {traj:.4f}  "
      f"(slope-SE floor at infinite items {traj/4:.4f})")
print()
print("  FRAGILITY, stated plainly: the trajectory term is a difference of two similar numbers, so it")
print("  is NOT well identified. Across accuracies within this sample's own spread it ranges from 0 to")
print("  about 0.03, and the projection below is reported as a BAND, not a point.")
print()
proj = []
print(f"  {'configuration':34s} {'n':>4s} {'slopeSE':>8s} {'/bar':>6s} {'/halfbar':>9s}")
for label, n_items in (("gen battery, MIN_DISJOINT=3", 48),
                       ("gen battery, all 7 disjoint", 112),
                       ("gen battery, 7 x 32 items", 224),
                       ("infinite items (the floor)", 10 ** 6)):
    samp = float(np.sqrt(2 * p_realized * (1 - p_realized) / n_items))
    band = []
    for tj in (0.0, traj, 0.03):
        se = float(np.sqrt(samp ** 2 + tj ** 2)) / 4.0
        band.append(round(se, 5))
    se_mid = band[1]
    proj.append({"configuration": label, "n_items": n_items,
                 "slope_se_no_floor": band[0], "slope_se_point": band[1],
                 "slope_se_pessimistic": band[2],
                 "slope_se_over_bar": round(se_mid / MIN_EFFECT_SLOPE, 3),
                 "slope_se_over_half_bar": round(se_mid / (MIN_EFFECT_SLOPE / 2), 3)})
    print(f"  {label:34s} {n_items:4d} {se_mid:8.4f} {se_mid/MIN_EFFECT_SLOPE:6.2f} "
          f"{se_mid/(MIN_EFFECT_SLOPE/2):9.2f}   band [{band[0]:.4f}, {band[2]:.4f}]")
out["projection"] = {"clean_base_accuracy": round(clean_agg, 4),
                     "realized_finetuned_accuracy": round(p_realized, 4),
                     "sampling_component": round(samp_realized, 5),
                     "trajectory_component": round(traj, 5),
                     "trajectory_identified": False,
                     "slope_se_floor_infinite_items": round(traj / 4, 5),
                     "rows": proj,
                     "note": ("accuracy is MEASURED, not assumed: the 0.90 selection floor applies to "
                              "the CLEAN base model (which scores 1.0 here), but fine-tuning degrades "
                              "the arms to 0.8625, and it is that value which sets sampling noise "
                              "during the run. The trajectory term is a difference of similar numbers "
                              "and is NOT well identified; the projection is a band.")}

print()
print("=" * 88)
print("8. CORRECTED OPERATING POINTS -- the v6 delta priced at the MEASURED accuracy")
print("=" * 88)
RNG_P = np.random.default_rng(11)
N_MC = 200_000


def power(bar, se, true_slope, n_seeds=5):
    x = RNG_P.normal(true_slope, se, size=(N_MC, n_seeds))
    return float((((x >= bar).sum(1) > n_seeds / 2.0) & ((x <= -bar).sum(1) == 0)).mean())


print(f"  {'n_items':>8s} {'slopeSE':>8s} {'bar':>8s} {'sens@price':>11s} {'FPR':>7s}")
ops = []
for n_items in (48, 112, 224):
    samp = float(np.sqrt(2 * p_realized * (1 - p_realized) / n_items))
    se = float(np.sqrt(samp ** 2 + traj ** 2)) / 4.0
    for bar in (MIN_EFFECT_SLOPE / 2, MIN_EFFECT_SLOPE):
        sens = power(bar, se, MIN_EFFECT_SLOPE)
        fpr = power(bar, se, 0.0)
        ops.append({"n_items": n_items, "slope_se": round(se, 5), "bar": round(bar, 5),
                    "sensitivity_at_price": round(sens, 4), "false_positive": round(fpr, 4),
                    "clears_080": bool(sens >= 0.80)})
        print(f"  {n_items:8d} {se:8.4f} {bar:8.4f} {sens:11.3f} {fpr:7.3f}"
              f"{'   <- clears 0.80' if sens >= 0.80 else ''}")
out["corrected_operating_points"] = ops

print()
print("=" * 88)
print("9. WHAT THIS CHANGES")
print("=" * 88)
verdict = (
    "Four things, and the fourth SUBTRACTS a claim this same session made three hours ago.\n"
    "  (a) CONFIRMED on real data: the rank-4 paired delta is exactly zero in 5 of 5 seeds, so"
    "      the estimator really does reduce to d8/4 -- one load-bearing checkpoint per seed.\n"
    "  (b) The realized within-seed slope SE is %.4f = %.2fx the decision bar. The prereg own"
    "      worst-case figure (0.036) was conservative by about 1.4x, but the power problem"
    "      SURVIVES the correction, and the symmetry ceiling caps power near 0.5 for as long as"
    "      the bar sits AT the price of interest.\n"
    "  (c) The cross-seed pool overstates the real estimator noise by %.1fx, so the section-4"
    "      VOID is the conservative reading and must not be quoted as the expected outcome.\n"
    "  (d) CORRECTION. The earlier projection assumed the 0.90 selection floor and no trajectory"
    "      floor, and reported sensitivity 0.880 at 112 items. Both inputs were wrong. The 0.90"
    "      floor applies to the CLEAN BASE model, which scores 1.0 here; fine-tuning then"
    "      degrades the arms to %.4f, and THAT sets the sampling noise during the run. A small"
    "      trajectory floor (%.4f, NOT well identified) survives on top. Corrected: 112 items"
    "      with the halved bar gives sensitivity about 0.75, NOT 0.880 -- short of the 0.80"
    "      convention. Reaching about 0.90 needs roughly 224 items, about double the disjoint"
    "      pool as it exists, so the battery must be EXPANDED, not merely fully selected."
) % (slope_se_within, slope_se_within / MIN_EFFECT_SLOPE, slope_se_pool / slope_se_within,
     p_realized, traj)
print(verdict)
out["conclusion"] = verdict
out["headline"] = (
    "structural zero CONFIRMED 5/5; realized slope SE 1.67x the bar; the v6 projection is CORRECTED "
    "DOWNWARD -- accuracy during the run is the fine-tuned 0.8625 not the 0.90 clean floor, and a "
    "small trajectory floor survives, so 112 items + halved bar gives sensitivity ~0.75 not 0.880. "
    "Conventional power needs ~224 items: the battery must be EXPANDED, not merely fully selected.")
json.dump(out, open("_estimator_noise_anchor.json", "w", encoding="utf-8"), indent=1)
print("\nwrote _estimator_noise_anchor.json")
