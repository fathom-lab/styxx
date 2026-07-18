"""B2-coupling attempt 4 -- TWO-SIDED admissibility gate x dose-slope finding. DRAFT, NOT FROZEN.

Attempt 2 shipped COUPLED from a rule that fired under the null (refused, cycle 43). Attempt 3 would
have shipped DECOUPLED from a battery never shown able to fire under the alternative (killed
pre-freeze, 2 fatals). One defect, two faces: validating a POINT verdict with a population that is
not that point, in ONE direction only. Attempt 4 dissolves both:

  ARCHITECTURE -- two separate computations, gate THEN finding:
  (1) ADMISSIBILITY GATE via styxx.admissibility.instrument_admissibility on the generation battery:
      positive = posctl (lambda=0) checkpoints where an INDEPENDENT destruction marker fires (the
      T/F knowledge readout collapses -- a channel the gen battery never reads); null = fixed-arm
      checkpoints; fire threshold derived SPLIT-CONFORMAL on fixed-arm checkpoints from seeds
      DISJOINT from the gate seeds (the private-13 retro-cert VOIDed NONSPECIFIC on threshold
      transfer -- retro_cert_private13.json: a CALIB-derived threshold fired on 0.548 of EVAL nulls
      -- so no threshold here ever touches the population it gates).
      HONEST SCOPE: lambda=0 collapses the T/F knowledge readout BY CONSTRUCTION, so a passing gate
      proves the battery NOT-STRUCTURALLY-DEAF to gross capability destruction ONLY. It can never
      license a break claim. There is NO verdict string containing "DECOUPLED" or
      "read_neq_write_BROKEN", and none asserting "capability held". Gate fails -> VOID; gate passes
      -> COUPLED / PARTIAL / a BOUNDED NULL are the only reachable findings.
  (2) DOSE-RESPONSE FINDING scored as EFFECT SIZE + SIGN-CONSISTENCY, NOT significance (operator
      decision, this session): for each admissible seed a per-seed slope of the paired delta
      (fixed_agg - accumulate_agg at matched steps) on the accumulate arm's ACTUAL erased_rank,
      fit ONLY over the pre-committed span erased_rank in [2, 8] with the step-0 structural-zero
      pairs EXCLUDED (ruling (c)); the recovery region (ranks 10-24) is REPORTED, not fitted.
      COUPLED requires a STRICT MAJORITY of admissible seeds above the effect-size bar with a
      shared (positive) sign; seed disagreement is the standing PARTIAL. There is NO p-value gate:
      a valid seed-level sign-flip null at 5 seeds has minimum two-sided p = 2/2**5 = 0.0625, so
      p < 0.05 is unreachable and forcing it would be gaming. A pooled within-seed permutation p
      and a seed-level sign-flip p are computed and REPORTED as non-gating descriptives only. The
      SEED is the replication unit.

Verbatim-with-attribution: the training loop is copied from coupling_confirm_v3.train_accumulating
(itself a verbatim copy of coupling_confirm.train_accumulating, itself the dose loop). Deltas here:
(a) per-arm `lam` is THREADED as a parameter (v3 read a module-global LAM inside the loss); (b) the
audit measures the generation battery (gates) alongside the T/F battery and raw MC battery (both
reported, never gating).

Crash-safe: per-seed JSONL cache keyed with ARMS_KEY; resume SKIPS records whose arms_key differs
(a 3-arm rerun must not silently drop the new arm). Usage:
  python coupling_confirm_v4.py --dry        CPU-only: every verdict branch on synthetic curves.
  python coupling_confirm_v4.py --calibrate  GPU: select the gen battery on the base model; freeze.
  python coupling_confirm_v4.py --smoke      GPU: 20 steps / 1 seed, *_SMOKE_INVALID*.
  python coupling_confirm_v4.py              GPU: the scored 5-seed / 3-arm run.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys, gc
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.admissibility import instrument_admissibility, slope_permutation_null  # noqa: E402
from styxx.crossmind import auroc as _auroc  # noqa: E402  # per-seed gate-replication AUROC (F4)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _py(o):
    """Recursively coerce numpy scalars/arrays to plain Python for JSON receipts."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_py(v) for v in o.tolist()]
    return o


CB = _load("capability_battery", HERE / "capability_battery.py")           # old T/F battery: REPORTED only
CBMC = _load("capability_battery_mc", HERE / "capability_battery_mc.py")   # MC battery: REPORTED only (raw)
CBG = _load("capability_battery_gen", HERE / "capability_battery_gen.py")  # generation battery: GATES
B2COUP = _load("b2_coupling_dose", HERE / "b2_coupling_dose.py")           # orthonormal_union (no GPU at import)

# ------------------------------------------------------ arms: THE single source of truth (per brief)
ARMS = [("accumulate", True, 1.0), ("fixed", False, 1.0), ("posctl", True, 0.0)]
ARMS_KEY = "|".join(f"{n}:acc={a}:lam={l}" for n, a, l in ARMS)

SEEDS = [0, 1, 2, 3, 4]
DERIVE_SEEDS = [0, 1]      # fire-threshold derivation split (fixed-arm checkpoints ONLY)
GATE_SEEDS = [2, 3, 4]     # admissibility gate population (disjoint from the derivation split)
ERASE_ALPHA = 4.0          # erasure loss weight (v3's ALPHA; renamed -- STAT_ALPHA is the test level)
STEPS = 300
REFIT_EVERY = 25
SURVIVAL = 0.70
CELLS_CACHE = HERE / "coupling_confirm_v4_cells.jsonl"
SELECTION_CACHE = HERE / "coupling_v4_gen_selected.json"
PREREG = "papers/calib-poison-general/PREREG_B2_coupling_twosided_2026_07_16.md"

# ---------------------------------------- prereg constants (provenance lives in the prereg, Artifact 3)
CLEAN_READ_FLOOR = 0.75          # inherited clean guard
DISJOINT_FLOOR_CLEAN = 0.90      # gen sub-task selection floor (base-only, treatment-blind)
MIN_DISJOINT = 3                 # minimum selected sub-tasks
CLEAN_AGG_FLOOR = 0.80           # clean selected aggregate floor
CLEAN_SUBTASK_FLOOR = 0.70       # clean per-sub-task floor
MIN_EFFECT_SLOPE = 0.0152        # PREREG: 0.0909 minimum price of interest / (8 - 2) rank span = 0.015150
RANK_SPAN = (2, 8)               # PREREG: per-seed slope FIT span -- fixed pin (2) to the largest
                                 # observed crossing rank (8); step-0 pairs excluded, ranks 10-24 reported
STAT_ALPHA = 0.05                # gate permutation p level (dose p is REPORTED, non-gating -- see below)
K_PERM = 1000
AUROC_FLOOR = 0.70               # gate sensitivity floor (PREREG power arithmetic); also the per-seed floor
MIN_GATE_SEEDS_REPLICATING = 2   # F4: >= this many gate seeds must separate posctl<-fixed on their OWN score
FI_DOSE_FRACTION = 0.5           # re-panel MAJOR: accumulate-arm FI slope >= this x MIN_EFFECT_SLOPE in a
                                 # majority of seeds -> dose-graded format confound -> COUPLED downgrades to PARTIAL
MAX_FIRE = 0.15                  # gate specificity: max false-fire rate on fixed-arm nulls
DESTRUCTION_KNOWLEDGE_MAX = 0.60  # independent destruction marker: T/F knowledge readout collapsed
MIN_POSITIVE_MARKED = 6          # fewer marker-fired posctl checkpoints -> gate unmeasurable
MIN_DERIVE_NULLS = 8             # fewer derivation nulls -> conformal threshold unmeasurable
MIN_GATE_NULLS = 8               # fewer gate nulls -> gate unmeasurable
MIN_PAIRS_PER_SEED = 6           # fewer matched pairs -> seed not dose-admissible
MIN_ADMISSIBLE_SEEDS = 3         # fewer dose-admissible seeds -> underpowered
FORMAT_INVARIANCE_MAX = 0.0722   # PREREG: one per-checkpoint aggregate SE at minimum selection

# ------------------------------------- ESTIMATOR-ADMISSIBILITY gate (panel #7 v5, section 3 frozen)
# Panel #7 killed the S1-S4 injection-recovery gate because its output did not depend on the data:
# ordered pool pairs made the null exactly sign-antisymmetric and injecting AT the decision bar pinned
# recovery at 0.500 for ANY pool, so the gate VOIDed every possible run EXCEPT a frozen channel, which
# it certified perfectly. The replacement makes sensitivity a MEASUREMENT carried into the verdict
# string, drops specificity to reported-only (it is a theorem, not a measurement), and supplies the
# second side of the gate with a degeneracy leg that has NO free threshold.
ESTIMATOR_RECOVERY_FLOOR = 0.80  # the conventional 80% power level, already the convention behind the
                                 # AUROC floor arithmetic. It no longer decides pass/fail -- it selects
                                 # WHICH quantile of the recovery curve is reported as the bound.
                                 # Moving it UP weakens the shipped claim, so up is the conservative
                                 # direction; 0.80 is frozen against the naive 0.50.
INJECTION_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)   # multiples of MIN_EFFECT_SLOPE
                                 # bottom = half the minimum price of interest, below which the
                                 # question is moot. Top = 8x bar = 0.1216, a span-scale effect of
                                 # 0.7296, inside the arithmetic maximum expressible slope
                                 # 1/(hi-lo) = 1/6 = 0.1667. An estimator that cannot recover an effect
                                 # consuming 73% of the aggregate's full range at 80% power is
                                 # insensitive by any standard, so the grid TOP is the insensitivity
                                 # threshold and no separate constant is created for it.
ESTIMATOR_PSEUDO_RUNS = 4000     # Monte-Carlo resolution ONLY: MC SE of a rate at 0.80 is
                                 # sqrt(0.8*0.2/4000) = 0.0063, an order below the recovery change
                                 # across one grid step. This is resampling error, NOT estimation
                                 # uncertainty -- the estimation uncertainty is the LOO spread (PF-7).
ESTIMATOR_RNG_SEED = 0           # matches every other seeded call in this harness
                                 # (instrument_admissibility(seed=0), slope_permutation_null(seed=0)),
                                 # so the gate is reproducible from the receipt alone.
MIN_ESTIMATOR_POOL = 6           # C(4,2) = 6: the pool size at the fewest seeds that can still form
                                 # pairs after the LOO jackknife drops one. Same idiom as
                                 # MIN_DERIVE_NULLS / MIN_GATE_NULLS -- a floor below which the
                                 # statistic is unmeasurable, not a tuned bar.
ZERO_SLOPE_ATOM_MAX = 0.5        # a strict-majority rule, this harness's convention everywhere. If a
                                 # majority of cross-seed fixed-arm pairs are identical at every fitted
                                 # rank, the pool cannot represent the estimator's noise and the
                                 # certificate is vacuous. Derived from the convention, not from a curve.
                                 # This is the leg that kills "a frozen channel scores a perfect pass".
FALSE_RECOVERY_ANALYTIC_MAX = {3: 0.1600, 4: 0.0787, 5: 0.0663}
                                 # NOT a gate (panel #7 deleted ESTIMATOR_FALSE_RECOVERY_MAX: any value
                                 # above 0.0663 cannot bind and any value below it was chosen by looking
                                 # at the attainable range). Exhaustive 50001-point grid over
                                 # q = P(s >= bar) = P(s <= -bar) against the frozen COUPLED rule.
                                 # Printed BESIDE the measured rate so the receipt states plainly that
                                 # under a sign-symmetric null the false-fire rate is bounded by a
                                 # THEOREM, not measured by this pool -- keeping a leg that cannot fire
                                 # is what voided styxx.instrument_admissibility's default specificity
                                 # leg, and this disclosure is what prevents the repeat.

ESTIMATOR_SCOPE = ("this certifies ESTIMATOR precision on the realized SEED-noise null only. it does "
                   "not certify that the generation channel responds to a capability change of any "
                   "size, and it does not certify the estimator unbiased for the accumulate-vs-fixed "
                   "ARM-type contrast (that channel is covered separately by trained_fi_dose_graded "
                   "and the guard-excluded subtractive).")

NOT_DEAF_SCOPE = ("lambda=0 collapses the T/F knowledge readout BY CONSTRUCTION; a passing gate "
                  "proves the generation battery NOT-STRUCTURALLY-DEAF to gross capability "
                  "destruction only. It can never license a break claim, a DECOUPLED verdict, or "
                  "any assertion that capability was held.")

FROZEN_VERDICTS = [
    "VOID_COUPLING__no_calibration",
    "VOID_COUPLING__calibration_model_mismatch",
    "VOID_COUPLING__clean_guard_failed",
    "VOID_COUPLING__battery_guard_failed",
    "VOID_COUPLING__no_bite",
    "VOID_COUPLING__underpowered",
    "VOID_COUPLING__admissibility_unmeasurable",
    "VOID_COUPLING__battery_insensitive",
    "VOID_COUPLING__battery_nonspecific",
    "VOID_COUPLING__estimator_unmeasurable",
    "VOID_COUPLING__estimator_insensitive",
    "VOID_COUPLING__estimator_insensitive__guard_excl",
    "COUPLED__dose_slope_price_measured_1p5B",
    "PARTIAL__coupling_sign_split_1p5B",
    # R3: a recovery certificate at ESTIMATOR_MDE80 licenses "no paired price above ESTIMATOR_MDE80",
    # NOT "above MIN_EFFECT_SLOPE" (T8). The token now names a MEASURED MDE, and the receipt carries
    # estimator_mde80, its span-scale bound and the recovery curve immediately beside this verdict.
    "NO_PAIRED_PRICE_ABOVE_ESTIMATOR_MDE80__battery_not_deaf_1p5B",
    "NO_CAPABILITY_PRICE__style_downgraded_1p5B",
]


# ----------------------------------------------------------------------- preflight (conformance fixes)
def preflight(selmeta, model_name, guard_read, selected):
    """Pre-training admissibility, in frozen order. Conformance fixes from the v3 panel: selmeta is
    guarded for None BEFORE .get (smoke path), ok=False is ENFORCED (v3 wrote the receipt even when
    ok=False and never checked), base_model is verified against the run's model."""
    if selmeta is None:
        return "VOID_COUPLING__no_calibration"
    if selmeta.get("base_model") != model_name:
        return "VOID_COUPLING__calibration_model_mismatch"
    if not guard_read:
        return "VOID_COUPLING__clean_guard_failed"
    guard_battery = (bool(selmeta and selmeta.get("ok"))
                     and CBG.battery_guard(selmeta.get("base_scores_gen", {}), selected,
                                           agg_floor=CLEAN_AGG_FLOOR, subtask_floor=CLEAN_SUBTASK_FLOOR)
                     and len(selected or []) >= MIN_DISJOINT)
    if not guard_battery:
        return "VOID_COUPLING__battery_guard_failed"
    return None


# ------------------------------------------------------------------------------------- points
def _gen_agg(point, selected):
    return CBG.aggregate(point["battery_gen"], selected)


def _gen_agg_guard_excl(point, selected):
    """The guard-zeroed-items-EXCLUDED aggregate (F5). Falls back to the raw aggregate if a
    checkpoint predates the guard-excluded channel (never true for a real v4 run)."""
    src = point.get("battery_gen_guard_excl", point["battery_gen"])
    return CBG.aggregate(src, selected)


def build_points(curves_by_seed, selected):
    """Top-level `points` list -- the grounding surface for verify_replication. Carries BOTH the raw
    generation aggregate and the guard-excluded aggregate, plus per-checkpoint repetition AND echo
    guard-fire counts, so the echo confound (panel FATAL F5) is on the grounding surface, not buried
    in the curves blob."""
    pts = []
    for seed, arms in sorted(curves_by_seed.items()):
        for arm, _, _ in ARMS:
            for p in arms.get(arm, []):
                pts.append({"seed": int(seed), "arm": arm, "step": int(p["step"]),
                            "erased_rank": int(p["erased_rank"]),
                            "private13": float(p["private13"]),
                            "knowledge": float(p["knowledge"]),
                            "gen_aggregate": round(float(_gen_agg(p, selected)), 4),
                            "gen_aggregate_guard_excl": round(float(_gen_agg_guard_excl(p, selected)), 4),
                            "repetition_guard_fires": int(p.get("repetition_guard_fires", 0)),
                            "echo_guard_fires": int(p.get("echo_guard_fires", 0)),
                            "bit": bool(p.get("bit"))})
    return pts


def guard_fire_totals(points):
    """Per-arm repetition/echo guard-fire totals and per-checkpoint rates across all checkpoints
    (F5 receipt; re-panel minor 7 adds the per-arm rates so the guard-excluded slope's downward
    bias when the accumulate arm guards more than the fixed arm is auditable)."""
    out = {}
    for p in points:
        a = out.setdefault(p["arm"], {"repetition": 0, "echo": 0, "n_checkpoints": 0})
        a["repetition"] += int(p["repetition_guard_fires"])
        a["echo"] += int(p["echo_guard_fires"])
        a["n_checkpoints"] += 1
    for a in out.values():
        n = a["n_checkpoints"] or 1
        a["repetition_per_checkpoint"] = round(a["repetition"] / n, 4)
        a["echo_per_checkpoint"] = round(a["echo"] / n, 4)
    return out


def filter_cache_records(lines):
    """Parse resume-cache JSONL lines, KEEPING only records whose arms_key matches the current
    ARMS_KEY (a 3-arm rerun must not silently inherit a stale 2-arm record). Corrupt lines are
    skipped. Returns {seed: arms}. Factored (re-panel minor 10) so run_real and --dry drive the
    SAME production filter rather than two divergent copies."""
    done, skipped_stale, corrupt = {}, [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            corrupt += 1
            continue
        if rec.get("arms_key") != ARMS_KEY:
            skipped_stale.append(rec.get("seed"))
            continue
        done[rec["seed"]] = rec["arms"]
    return {"done": done, "skipped_stale": skipped_stale, "corrupt": corrupt}


# ------------------------------------------------------------- computation (1): the ADMISSIBILITY GATE
def conformal_fire_threshold(derive_scores, max_fire):
    """Split-conformal firing threshold: the k-th smallest derivation-null score with
    k = floor(max_fire * (n + 1)). Under exchangeability of fixed-arm checkpoints across seeds the
    expected false-fire rate (score < threshold) on a fresh null is at most max_fire -- the
    finite-sample correction the private-13 transfer VOID demands. Returns None if k < 1."""
    xs = sorted(float(x) for x in derive_scores)
    k = math.floor(max_fire * (len(xs) + 1))
    if k < 1:
        return None
    return xs[k - 1]


def _seed_separation_auroc(pos, nulls):
    """Direction-aware AUROC for one gate seed's marker-fired posctl gen aggregates (positives)
    against that seed's fixed gen aggregates (nulls). Returns None if either class has < 2 units or
    the pooled scores are degenerate -- that seed simply does not replicate."""
    if len(pos) < 2 or len(nulls) < 2:
        return None
    scores = np.asarray(list(pos) + list(nulls), dtype=float)
    if np.unique(scores).size < 2:
        return None
    labels = np.asarray([1] * len(pos) + [0] * len(nulls), dtype=int)
    return float(_auroc(scores, labels))


def admissibility_gate(points):
    """Gate the instrument, not the finding. Positive = posctl checkpoints (step > 0, gate seeds)
    where the INDEPENDENT destruction marker fires (knowledge <= DESTRUCTION_KNOWLEDGE_MAX -- the
    T/F channel, which the gen battery never reads). Null = fixed-arm checkpoints (step > 0, gate
    seeds). Threshold from DERIVE_SEEDS fixed-arm checkpoints only (population-disjoint).

    Sensitivity has TWO legs (F4): the pooled two-sided permutation gate via
    instrument_admissibility, AND a per-seed replication requirement -- in at least
    MIN_GATE_SEEDS_REPLICATING gate seeds, that seed's own marker-fired posctl checkpoints must
    separate from that seed's own fixed checkpoints (per-seed AUROC discriminability >= AUROC_FLOOR,
    destroyed class ranking LOWER). A pooled statistic that passes on autocorrelated trajectories
    from a single separating seed no longer certifies not-deaf."""
    pos = [p["gen_aggregate"] for p in points
           if p["arm"] == "posctl" and p["step"] > 0 and p["seed"] in GATE_SEEDS
           and p["knowledge"] <= DESTRUCTION_KNOWLEDGE_MAX]
    nulls = [p["gen_aggregate"] for p in points
             if p["arm"] == "fixed" and p["step"] > 0 and p["seed"] in GATE_SEEDS]
    derive = [p["gen_aggregate"] for p in points
              if p["arm"] == "fixed" and p["step"] > 0 and p["seed"] in DERIVE_SEEDS]
    n_marked_all = sum(1 for p in points if p["arm"] == "posctl" and p["step"] > 0
                       and p["knowledge"] <= DESTRUCTION_KNOWLEDGE_MAX)

    # re-panel minor 5: guard-fire fraction among the marker-fired gate-seed posctl positives --
    # if most positives score 0 via a guard, gate sensitivity is guard-driven (decode-collapse
    # detection), NOT containment-channel sensitivity at sub-gross doses. Reported, non-gating.
    marked_pos = [p for p in points if p["arm"] == "posctl" and p["step"] > 0
                  and p["seed"] in GATE_SEEDS and p["knowledge"] <= DESTRUCTION_KNOWLEDGE_MAX]
    n_guarded_pos = sum(1 for p in marked_pos
                        if p["repetition_guard_fires"] > 0 or p["echo_guard_fires"] > 0)
    guard_frac_marked_pos = round(n_guarded_pos / len(marked_pos), 4) if marked_pos else None

    # per-seed replication (F4): each gate seed's OWN posctl-vs-fixed separation
    per_seed_auroc = {}
    n_replicating = 0
    for s in GATE_SEEDS:
        ps = [p["gen_aggregate"] for p in points if p["arm"] == "posctl" and p["step"] > 0
              and p["seed"] == s and p["knowledge"] <= DESTRUCTION_KNOWLEDGE_MAX]
        ns = [p["gen_aggregate"] for p in points if p["arm"] == "fixed" and p["step"] > 0
              and p["seed"] == s]
        a = _seed_separation_auroc(ps, ns)
        rep_ok = bool(a is not None and max(a, 1.0 - a) >= AUROC_FLOOR and a < 0.5)  # lower_on_positive
        per_seed_auroc[str(s)] = {"auroc": (round(a, 6) if a is not None else None),
                                  "n_pos": len(ps), "n_null": len(ns), "replicates": rep_ok}
        if rep_ok:
            n_replicating += 1

    base = {"n_positive_marked_gate": len(pos), "n_positive_marked_all_seeds": n_marked_all,
            "n_gate_nulls": len(nulls), "n_derive_nulls": len(derive),
            "destruction_marker": f"knowledge <= {DESTRUCTION_KNOWLEDGE_MAX}",
            "derive_seeds": DERIVE_SEEDS, "gate_seeds": GATE_SEEDS,
            "per_seed_auroc": per_seed_auroc, "n_gate_seeds_replicating": n_replicating,
            "min_gate_seeds_replicating": MIN_GATE_SEEDS_REPLICATING,
            "guard_fire_fraction_marked_positives": guard_frac_marked_pos,
            "scope": NOT_DEAF_SCOPE}
    if len(pos) < MIN_POSITIVE_MARKED or len(nulls) < MIN_GATE_NULLS or len(derive) < MIN_DERIVE_NULLS:
        return {**base, "gate_pass": False, "coupling_void": "VOID_COUPLING__admissibility_unmeasurable",
                "fire_threshold": None, "certificate": None}
    t = conformal_fire_threshold(derive, MAX_FIRE)
    if t is None:
        return {**base, "gate_pass": False, "coupling_void": "VOID_COUPLING__admissibility_unmeasurable",
                "fire_threshold": None, "certificate": None}
    rep = instrument_admissibility(
        scores=list(pos) + list(nulls),
        labels=[1] * len(pos) + [0] * len(nulls),
        expect="lower_on_positive", fire_threshold=t,
        auroc_floor=AUROC_FLOOR, alpha=STAT_ALPHA, max_fire=MAX_FIRE, k_perm=K_PERM, seed=0)
    void = None
    if rep.admissibility_verdict == "VOID_INSTRUMENT__unmeasurable":
        void = "VOID_COUPLING__admissibility_unmeasurable"
    elif rep.admissibility_verdict == "VOID_INSTRUMENT__insensitive":
        void = "VOID_COUPLING__battery_insensitive"
    elif rep.admissibility_verdict == "VOID_INSTRUMENT__nonspecific":
        void = "VOID_COUPLING__battery_nonspecific"
    elif rep.admissibility_verdict != "ADMISSIBLE":
        # positive-pass hardening: the gate passes ONLY on bare ADMISSIBLE; any other verdict
        # (drift, sensitivity-only) is a VOID, never a silent pass.
        void = "VOID_COUPLING__admissibility_unmeasurable"
    # per-seed replication leg -- only meaningful once the pooled instrument is itself ADMISSIBLE
    if void is None and n_replicating < MIN_GATE_SEEDS_REPLICATING:
        void = "VOID_COUPLING__battery_insensitive"
    return {**base, "gate_pass": void is None, "coupling_void": void,
            "fire_threshold": round(float(t), 4), "certificate": rep.certificate()}


# ------------------------------------------------------- computation (2): the DOSE-RESPONSE finding
def _pairs(arms):
    fixed_by_step = {p["step"]: p for p in arms["fixed"]}
    return [(p, fixed_by_step[p["step"]]) for p in arms["accumulate"] if p["step"] in fixed_by_step]


def _per_seed_slope(deltas, ranks):
    """OLS slope of paired delta on the ACTUAL erased_rank (np.polyfit[0]); None if < 2 distinct
    ranks. Callers pass ONLY the span-[2,8], step-0-excluded pairs (rulings (c))."""
    return round(float(np.polyfit(ranks, deltas, 1)[0]), 6) if len(set(ranks)) >= 2 else None


def _sign_flip_p(slopes):
    """Exhaustive seed-level sign-flip permutation p (two-sided) on the mean per-seed slope --
    REPORTED, NON-GATING. Under the null a seed's slope is symmetric about 0, so flipping signs is
    the exchangeable null. For n seeds the minimum attainable two-sided p is 2 / 2**n (0.0625 at
    n=5), unreachable at alpha 0.05 -- which is exactly WHY the verdict does not gate on it."""
    import itertools
    s = [x for x in slopes if x is not None]
    n = len(s)
    if n == 0:
        return {"p_value": None, "n": 0, "p_floor": None}
    obs = abs(sum(s) / n)
    total = hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        total += 1
        if abs(sum(sg * x for sg, x in zip(signs, s)) / n) >= obs - 1e-12:
            hits += 1
    return {"p_value": round(hits / total, 6), "n": n, "p_floor": round(2.0 / total, 6)}


def dose_response(curves_by_seed, selected, admissible_seeds, *, battery_key="battery_gen"):
    """Seed-level EFFECT-SIZE + SIGN-CONSISTENCY dose finding (operator decision). Per seed, fit the
    paired-delta slope on the accumulate arm's ACTUAL erased_rank over the pre-committed span
    RANK_SPAN=[2,8] with step-0 pairs EXCLUDED (ruling (c)); the recovery region (ranks 10-24) is
    kept in the reported curve but NOT fitted. There is NO p-value gate. COUPLED = a strict majority
    of admissible seeds at or above MIN_EFFECT_SLOPE, all of them the SAME (positive) sign; a strict
    majority BELOW the bar in magnitude = the bounded null; anything else = sign split (PARTIAL).
    A pooled within-seed permutation p and a seed-level sign-flip p are computed as REPORTED,
    NON-GATING descriptives only. `battery_key` selects the raw ('battery_gen') or the guard-excluded
    ('battery_gen_guard_excl') channel -- the latter is the F5 subtractive recomputation."""
    lo, hi = RANK_SPAN
    per_seed, slopes = {}, []
    pooled_stat, pooled_dose, pooled_unit = [], [], []
    for seed in admissible_seeds:
        arms = curves_by_seed[seed]
        fit_deltas, fit_ranks = [], []
        full_deltas, full_ranks, full_steps = [], [], []
        for acc_p, fix_p in _pairs(arms):
            d = CBG.paired_delta(fix_p[battery_key], acc_p[battery_key], selected)
            r = int(acc_p["erased_rank"]); st = int(acc_p["step"])
            full_deltas.append(d); full_ranks.append(r); full_steps.append(st)
            if st > 0 and lo <= r <= hi:                    # span [2,8], step-0 structural zeros out
                fit_deltas.append(d); fit_ranks.append(float(r))
        slope = _per_seed_slope(fit_deltas, fit_ranks)
        per_seed[seed] = {"slope": slope, "n_fit_pairs": len(fit_deltas),
                          "fit_ranks": [int(r) for r in fit_ranks],
                          "n_pairs": len(full_deltas), "deltas": full_deltas,
                          "ranks": full_ranks, "steps": full_steps}
        if slope is not None:
            slopes.append(slope)
        pooled_stat += fit_deltas; pooled_dose += [float(r) for r in fit_ranks]
        pooled_unit += [seed] * len(fit_deltas)

    # The majority DENOMINATOR is the count of DOSE-ADMISSIBLE seeds, NOT the count of non-None
    # slopes (re-panel FATAL): a stalled seed (no distinct fitted rank in the narrowed span) returns
    # a None slope but is still dose-admissible, and must count AGAINST the majority (it is not
    # above the bar), never be silently dropped from the denominator -- otherwise 4 stalled + 1
    # above-bar reads a strict majority from ONE seed. n_slopes is reported separately, and
    # compute_verdict VOIDs to underpowered if fewer than MIN_ADMISSIBLE_SEEDS seeds have a slope.
    n_admissible = len(admissible_seeds)
    n_slopes = len(slopes)
    pos_above = [s for s in slopes if s >= MIN_EFFECT_SLOPE]
    neg_above = [s for s in slopes if s <= -MIN_EFFECT_SLOPE]
    below = [s for s in slopes if abs(s) < MIN_EFFECT_SLOPE]
    coupled = bool(len(pos_above) > n_admissible / 2 and len(neg_above) == 0)
    bounded_null = bool(len(below) > n_admissible / 2)

    # ---- REPORTED, NON-GATING descriptives (no verdict rides on these) ----
    if len(set(pooled_dose)) >= 2 and len(pooled_stat) >= 2:
        pooled = slope_permutation_null(pooled_stat, pooled_dose, unit=pooled_unit, seed=0, k_perm=K_PERM)
    else:
        pooled = {"slope": None, "perm_p95": None, "p_value": None, "k_perm": K_PERM}
    sign_flip = _sign_flip_p(slopes)

    return {"battery_key": battery_key, "rank_span": list(RANK_SPAN),
            "min_effect_slope": MIN_EFFECT_SLOPE,
            "per_seed": {str(k): v for k, v in per_seed.items()},
            "per_seed_slopes": {str(seed): per_seed[seed]["slope"] for seed in admissible_seeds},
            "n_admissible_seeds": n_admissible, "n_admissible_slopes": n_slopes,
            "n_above_bar_positive": len(pos_above), "n_above_bar_negative": len(neg_above),
            "n_below_bar": len(below),
            "coupled": coupled, "bounded_null": bounded_null,
            "reported_nongating": {
                "pooled_slope": pooled["slope"], "pooled_perm_p": pooled["p_value"],
                "pooled_perm_p95": pooled["perm_p95"], "k_perm": pooled["k_perm"],
                "seed_sign_flip_p": sign_flip["p_value"],
                "seed_sign_flip_p_floor": sign_flip["p_floor"],
                "note": ("p-values here are REPORTED, NON-GATING. The seed-level sign-flip null has "
                         "minimum two-sided p = 2/2**n_seeds (0.0625 at 5 seeds), unreachable at "
                         "alpha 0.05, so the verdict gates on effect size + sign-consistency.")}}


# ------------------------------------------------ the ESTIMATOR-ADMISSIBILITY gate (panel #7, R1.a-f)
def _estimator_aggregate(point, selected, battery_key):
    """The aggregate the verdict rides, read from the SAME field build_points exposes and at the SAME
    4dp rounding, so a verifier recomputing the pool from points[*] reproduces it to the last decimal."""
    if battery_key == "battery_gen_guard_excl":
        return round(float(_gen_agg_guard_excl(point, selected)), 4)
    return round(float(_gen_agg(point, selected)), 4)


def _estimator_pool(curves_by_seed, selected, battery_key):
    """The cross-seed fixed-vs-fixed null pool (R1.a). Units are the 10 UNORDERED seed pairs {i,j}.
    ORDERED pairs are refused: they give d_ji = -d_ij pointwise, which makes the pool exactly
    sign-antisymmetric -- so the specificity leg's pass becomes a theorem about the construction and
    mirror twins veto COUPLED on half the draws for reasons that have nothing to do with the estimator
    (E2/F2, the fatal that killed S1). A step enters a unit only where BOTH seeds' accumulate arms
    report the SAME erased_rank, which removes the whose-schedule ambiguity with no judgment call.
    A unit with fewer than 2 distinct surviving ranks is DROPPED and counted, never silently absorbed."""
    lo, hi = RANK_SPAN
    seeds = [s for s in SEEDS if s in curves_by_seed]
    agg, rank = {}, {}
    for s in seeds:
        arms = curves_by_seed[s]
        agg[s] = {int(p["step"]): _estimator_aggregate(p, selected, battery_key)
                  for p in arms.get("fixed", [])}
        rank[s] = {int(p["step"]): int(p["erased_rank"]) for p in arms.get("accumulate", [])}
    units, dropped = [], []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            i, k = seeds[a], seeds[b]
            deltas, ranks = [], []
            for st in sorted(set(agg[i]) & set(agg[k]) & set(rank[i]) & set(rank[k])):
                if st <= 0 or rank[i][st] != rank[k][st]:
                    continue
                r = rank[i][st]
                if not (lo <= r <= hi):
                    continue
                deltas.append(round(agg[i][st] - agg[k][st], 4)); ranks.append(float(r))
            slope = _per_seed_slope(deltas, ranks)      # SAME code path as the verdict's own slope
            if slope is None:
                dropped.append([i, k])
                continue
            units.append({"pair": [i, k], "slope": slope, "n_points": len(ranks),
                          "ranks": [int(r) for r in ranks]})
    return units, dropped


def _frozen_rule_rates(vals, n_admissible):
    """Score the FROZEN dose rule of `dose_response` on a (runs x n_drawn) matrix of pseudo-slopes,
    denominator n_admissible. Padded None slots are simply ABSENT from `vals`, so they count AGAINST
    both majorities -- byte-for-byte the harness convention, which forecloses panel #6's
    reduced-denominator defect recurring inside the gate meant to prevent it (PF-4, T6, F7)."""
    pos_above = (vals >= MIN_EFFECT_SLOPE).sum(axis=1)
    neg_above = (vals <= -MIN_EFFECT_SLOPE).sum(axis=1)
    below = (np.abs(vals) < MIN_EFFECT_SLOPE).sum(axis=1)
    coupled = (pos_above > n_admissible / 2.0) & (neg_above == 0)
    bounded_null = below > n_admissible / 2.0
    return float(np.mean(coupled)), float(np.mean(bounded_null))


def _pseudo_rates(slopes, n_draw, n_admissible, k, rng, runs=ESTIMATOR_PSEUDO_RUNS):
    """One block of pseudo-runs at injection k (R1.c): draw n_draw DISTINCT units without replacement,
    give each an INDEPENDENT uniform sign (pair orientation is an arbitrary label; randomizing it is
    the honest treatment, and it is the same sign-flip null `_sign_flip_p` already uses), inject at the
    slope level. Injection is exact at the slope level because np.polyfit is linear in the response:
    slope(d + c*r) = slope(d) + c. Returns (recovery, bounded_null_rate)."""
    n_pool = len(slopes)
    if n_draw <= 0 or n_pool == 0:
        return 0.0, 0.0
    idx = rng.permuted(np.tile(np.arange(n_pool), (runs, 1)), axis=1)[:, :n_draw]
    signs = rng.integers(0, 2, size=(runs, n_draw)) * 2.0 - 1.0
    return _frozen_rule_rates(np.asarray(slopes)[idx] * signs + k * MIN_EFFECT_SLOPE, n_admissible)


def _estimator_mde80(slopes, n_draw, n_admissible, rng, runs=ESTIMATOR_PSEUDO_RUNS):
    """Sweep INJECTION_GRID ascending and return (smallest k reaching ESTIMATOR_RECOVERY_FLOOR or None,
    the full recovery curve). The curve is always computed in full -- it is a receipt line, and a gate
    whose recovery is a constant across dispersion is exactly what panel #7 killed."""
    mde, curve = None, []
    for k in INJECTION_GRID:
        rec, _ = _pseudo_rates(slopes, n_draw, n_admissible, k, rng, runs)
        curve.append({"k": k, "recovery": round(rec, 4)})
        if mde is None and rec >= ESTIMATOR_RECOVERY_FLOOR:
            mde = k
    return mde, curve


def estimator_admissibility(curves_by_seed, selected, admissible_seeds, *, battery_key="battery_gen",
                            rng_seed=ESTIMATOR_RNG_SEED):
    """Two-sided admissibility of the ESTIMATOR ITSELF on the realized seed-noise null (panel #7, R1).

    DEGENERACY leg (gating, no free threshold): a pool too small, carrying fewer than 2 distinct
    slopes, or with a strict majority of exactly-zero slopes cannot represent the estimator's noise;
    the zero atom is the diagnostic that made the killed spec's certificate maximally reassuring
    EXACTLY when the measurement was dead, so it is on the receipt whether or not it fires.

    SENSITIVITY leg (gating, MEASUREMENT-valued): ESTIMATOR_MDE80 is the smallest injection reaching
    ESTIMATOR_RECOVERY_FLOOR, maximised over the full pool and 5 leave-one-SEED-out replicates. The
    max is the conservative choice and removes the "which replicate do we trust" judgment call without
    a new constant. It is a measurement, not a pass/fail at a fixed injection: injecting exactly at the
    decision bar pins per-pseudo-seed detection at p=0.5 for ANY pool (T1/E1/PF-1), which is why the
    killed spec emitted a constant.

    SPECIFICITY leg (REPORTED, NON-GATING): under any sign-symmetric null the frozen COUPLED rule's
    false-fire rate is bounded by FALSE_RECOVERY_ANALYTIC_MAX -- a theorem about the construction, not
    a measurement of this pool -- so it cannot gate. Realized values may slightly exceed the i.i.d. cap
    because draws share magnitudes; both numbers are printed side by side.

    Returns a dict; mutates nothing."""
    units, dropped = _estimator_pool(curves_by_seed, selected, battery_key)
    slopes = np.asarray([u["slope"] for u in units], dtype=float)
    n_pool = len(units)
    n_distinct = int(np.unique(slopes).size) if n_pool else 0
    zero_atom = round(float(np.mean(slopes == 0.0)), 4) if n_pool else 1.0
    n_admissible = len(admissible_seeds)
    # the realized run shape: how many of the dose-admissible seeds actually carry a slope. Read from
    # the production function itself so the pseudo-run shape can never drift from the real one.
    n_slopes = dose_response(curves_by_seed, selected, admissible_seeds,
                             battery_key=battery_key)["n_admissible_slopes"]
    n_draw = min(n_slopes, n_pool)
    out = {"battery_key": battery_key, "rng_seed": int(rng_seed),
           "pseudo_runs": ESTIMATOR_PSEUDO_RUNS,
           "n_pool": n_pool, "n_pool_dropped": len(dropped), "dropped_pairs": dropped,
           "pool_slopes": [u["slope"] for u in units], "pool_units": units,
           "n_distinct_pool_slopes": n_distinct, "zero_slope_atom": zero_atom,
           "zero_slope_atom_max": ZERO_SLOPE_ATOM_MAX, "min_estimator_pool": MIN_ESTIMATOR_POOL,
           "n_admissible": n_admissible, "n_slopes": n_slopes, "n_drawn_per_run": n_draw,
           "n_padded_slots": n_admissible - n_draw,
           "recovery_floor": ESTIMATOR_RECOVERY_FLOOR,
           "injection_grid": list(INJECTION_GRID), "min_effect_slope": MIN_EFFECT_SLOPE,
           "mde80": None, "mde80_multiple_of_bar": None, "span_bound": None,
           "mde80_replicates": None, "recovery_curve": None,
           "false_recovery_rate": None, "false_recovery_analytic_max": None,
           "false_bounded_null_rate": None, "void": None, "scope": ESTIMATOR_SCOPE,
           "mc_note": ("pseudo_runs is Monte-Carlo RESAMPLING error only (SE 0.0063 at a rate of "
                       "0.80), NOT estimation uncertainty -- that is the leave-one-seed-out spread."),
           "specificity_note": ("false_recovery_rate is REPORTED, NON-GATING: under a sign-symmetric "
                                "null the frozen COUPLED rule's false-fire rate is bounded by a "
                                "THEOREM (false_recovery_analytic_max), not measured by this pool. "
                                "Realized values may slightly exceed the i.i.d. cap because draws "
                                "share magnitudes. false_bounded_null_rate is reported for the same "
                                "reason: an 0.80 recovery floor already caps it at 0.20.")}
    if n_pool < MIN_ESTIMATOR_POOL or n_distinct < 2 or zero_atom > ZERO_SLOPE_ATOM_MAX:
        out["void"] = "VOID_COUPLING__estimator_unmeasurable"
        return out
    rng = np.random.default_rng(rng_seed)
    fr, fbn = _pseudo_rates(slopes, n_draw, n_admissible, 0.0, rng)
    out["false_recovery_rate"] = round(fr, 4)
    out["false_bounded_null_rate"] = round(fbn, 4)
    out["false_recovery_analytic_max"] = FALSE_RECOVERY_ANALYTIC_MAX.get(n_admissible)
    mde_full, curve = _estimator_mde80(slopes, n_draw, n_admissible, rng)
    out["recovery_curve"] = curve
    reps = {"full": mde_full}
    for s in SEEDS:
        sub = np.asarray([u["slope"] for u in units if s not in u["pair"]], dtype=float)
        m, _c = _estimator_mde80(sub, min(n_slopes, len(sub)), n_admissible, rng)
        reps["loo_%d" % s] = m
    out["mde80_replicates"] = reps
    if any(m is None for m in reps.values()):
        out["void"] = "VOID_COUPLING__estimator_insensitive"
        return out
    mult = max(reps.values())
    out["mde80_multiple_of_bar"] = mult
    out["mde80"] = round(mult * MIN_EFFECT_SLOPE, 6)
    # left UNROUNDED so span_bound == mde80 * (hi - lo) holds exactly for any verifier
    out["span_bound"] = float(out["mde80"] * (RANK_SPAN[1] - RANK_SPAN[0]))
    return out


def channel_gain_receipt(points, clean_gen_agg, clean_knowledge):
    """REPORTED, NON-GATING (M3): regress gen-aggregate DROP on knowledge DROP over marker-fired
    posctl gate-seed checkpoints, so the 0.0909-knowledge-price -> gen-slope-bar transfer is
    documented (coef ~1 => the gen channel loses capability at the T/F channel's rate), not assumed.
    Data are already collected in the run; this measures the cross-channel gain, never gates on it."""
    xs, ys = [], []
    for p in points:
        if (p["arm"] == "posctl" and p["step"] > 0 and p["seed"] in GATE_SEEDS
                and p["knowledge"] <= DESTRUCTION_KNOWLEDGE_MAX):
            xs.append(clean_knowledge - float(p["knowledge"]))         # knowledge drop
            ys.append(clean_gen_agg - float(p["gen_aggregate"]))       # gen-aggregate drop
    if len(xs) < 2 or len(set(xs)) < 2:
        return {"coef_gen_drop_per_knowledge_drop": None, "n": len(xs),
                "note": "REPORTED, NON-GATING; insufficient/degenerate marker-fired posctl checkpoints"}
    coef = float(np.polyfit(xs, ys, 1)[0])
    return {"coef_gen_drop_per_knowledge_drop": round(coef, 6), "n": len(xs),
            "note": ("REPORTED, NON-GATING. coef ~1 => the generation channel loses capability at "
                     "the rate the T/F knowledge channel does; documents the cross-channel scale "
                     "assumption behind MIN_EFFECT_SLOPE.")}


def trained_fi_summary(curves_by_seed):
    """Trained-checkpoint format-invariance deltas (M2, REPORTED). Only checkpoints carrying a
    'format_invariance' key -- measured at the pre-committed subsample (pin rank, crossing rank,
    final) -- contribute, so dose-correlated format drift on the accumulate arm is visible. A
    COUPLED result must disclose this (prereg); a dose-graded FI comparable to the dose-graded price
    demotes COUPLED at the RESULT-interpretation gate."""
    out = {}
    for seed, arms in sorted(curves_by_seed.items()):
        for arm, _, _ in ARMS:
            for p in arms.get(arm, []):
                fi = p.get("format_invariance")
                if fi is not None:
                    out.setdefault(arm, []).append(
                        {"seed": int(seed), "step": int(p["step"]),
                         "erased_rank": int(p["erased_rank"]), "abs_delta": float(fi["abs_delta"])})
    return out or None


def trained_fi_dose_graded(curves_by_seed, admissible_seeds):
    """NUMERIC, pre-committed COUPLED->PARTIAL demotion for dose-graded format drift (re-panel MAJOR).
    Per dose-admissible seed, fit the ACCUMULATE arm's trained-checkpoint format-invariance abs_delta
    on erased_rank over the span RANK_SPAN=[2,8] (pin and crossing checkpoints supply the endpoints).
    A seed is FLAGGED when its FI slope is at or above the threshold of FI_DOSE_FRACTION x
    MIN_EFFECT_SLOPE (a format channel dose-graded at half the price bar). If a STRICT MAJORITY of the
    dose-admissible seeds are flagged, a COUPLED verdict downgrades to PARTIAL: the priced slope may
    be riding format drift the paired contrast cannot difference out. The paired-delta slope and the
    FI slope live on the same rank axis, so the comparison is on one scale. REPORTED either way."""
    lo, hi = RANK_SPAN
    bar = FI_DOSE_FRACTION * MIN_EFFECT_SLOPE
    per_seed = {}
    n_flagged = n_measured = 0
    for seed in admissible_seeds:
        arms = curves_by_seed.get(seed, {})
        pts = [(int(p["erased_rank"]), float(p["format_invariance"]["abs_delta"]))
               for p in arms.get("accumulate", [])
               if p.get("format_invariance") is not None and lo <= int(p["erased_rank"]) <= hi]
        ranks = [r for r, _ in pts]
        slope = (round(float(np.polyfit(ranks, [d for _, d in pts], 1)[0]), 6)
                 if len(set(ranks)) >= 2 else None)
        flagged = bool(slope is not None and slope >= bar)
        per_seed[str(seed)] = {"fi_slope": slope, "n_points": len(pts), "flagged": flagged}
        if slope is not None:
            n_measured += 1
        if flagged:
            n_flagged += 1
    downgrade = bool(n_flagged > len(admissible_seeds) / 2)
    return {"fraction_of_bar": FI_DOSE_FRACTION, "bar": round(bar, 6),
            "per_seed": per_seed, "n_flagged": n_flagged, "n_measured": n_measured,
            "downgrade": downgrade,
            "note": ("REPORTED; a strict majority of dose-admissible accumulate-arm seeds with FI "
                     "slope at or above the threshold demotes COUPLED to PARTIAL (dose-graded format "
                     "confound). Non-gating for any non-COUPLED verdict.")}


def format_invariance_on_selected(fi, selected):
    """Re-aggregate a full format-invariance check onto the SELECTED sub-tasks (M2). The full check
    already scored every sub-task under both wrappers, so the selected delta re-aggregates with no
    extra decode. Ties the 0.0722 bar to the SELECTED aggregate -- the statistic the verdict rides,
    which at the minimum selection (3 sub-tasks x 16 = 48 items) has exactly the SE the bar is."""
    if not selected:
        return {"selected": [], "aggregate_plain": None, "aggregate_verbose": None,
                "abs_delta": None}
    agg_p = CBG.aggregate(fi["per_subtask_plain"], selected)
    agg_v = CBG.aggregate(fi["per_subtask_verbose"], selected)
    return {"selected": list(selected), "aggregate_plain": round(agg_p, 4),
            "aggregate_verbose": round(agg_v, 4), "abs_delta": round(abs(agg_p - agg_v), 4)}


# ------------------------------------------------------------------------------------ the verdict
def compute_verdict(curves_by_seed, selected):
    """Frozen order: no_bite -> underpowered -> ADMISSIBILITY GATE (computation 1) -> dose-response
    finding (computation 2), with the ESTIMATOR-ADMISSIBILITY gate (panel #7) standing between the
    underpowered re-check and the first claim-bearing branch. Both gates can only VOID or enable; the
    battery gate certifies the INSTRUMENT, the estimator gate certifies the STATISTIC's precision on
    the realized seed-noise null, and neither can reach a verdict. The finding picks among COUPLED /
    PARTIAL / the bounded null on EFFECT SIZE + SIGN-CONSISTENCY (no p-gate). No other findings are
    reachable. The three finding branches are mutually exclusive on effect size and sign, so a
    significant-but-sub-bar price can no longer masquerade as the bounded null (attempt-3 recurrence,
    F3). COUPLED additionally must survive the F5 subtractive recomputation with guard-zeroed items
    excluded from both arms, else it downgrades to the bounded null."""
    points = build_points(curves_by_seed, selected)
    bite = {s: any(p.get("bit") for p in arms["accumulate"]) for s, arms in curves_by_seed.items()}
    per_seed_meta = {str(s): {"bite": bool(bite[s]),
                              "n_pairs": len(_pairs(arms))} for s, arms in curves_by_seed.items()}
    if not any(bite.values()):
        return "VOID_COUPLING__no_bite", points, per_seed_meta, None, None
    admissible = [s for s in sorted(curves_by_seed) if bite[s]
                  and len(_pairs(curves_by_seed[s])) >= MIN_PAIRS_PER_SEED]
    if len(admissible) < MIN_ADMISSIBLE_SEEDS:
        return "VOID_COUPLING__underpowered", points, per_seed_meta, None, None
    gate = admissibility_gate(points)
    if not gate["gate_pass"]:
        return gate["coupling_void"], points, per_seed_meta, gate, None
    dr = dose_response(curves_by_seed, selected, admissible, battery_key="battery_gen")
    dr_ge = dose_response(curves_by_seed, selected, admissible, battery_key="battery_gen_guard_excl")
    dr["guard_excluded_subtractive"] = {
        "per_seed_slopes": dr_ge["per_seed_slopes"],
        "n_above_bar_positive": dr_ge["n_above_bar_positive"],
        "n_above_bar_negative": dr_ge["n_above_bar_negative"],
        "n_below_bar": dr_ge["n_below_bar"], "coupled": dr_ge["coupled"]}
    # re-panel FATAL: never score on a reduced denominator -- if fewer than MIN_ADMISSIBLE_SEEDS
    # dose-admissible seeds actually have a defined (non-stalled) slope, the run is underpowered.
    if dr["n_admissible_slopes"] < MIN_ADMISSIBLE_SEEDS:
        return "VOID_COUPLING__underpowered", points, per_seed_meta, gate, dr
    # ESTIMATOR-ADMISSIBILITY gate (panel #7, R2). Placement is the point (F8): every return above
    # this line is a VOID that asserts nothing, and the first claim-bearing return is below it, so
    # inserting here puts ALL FIVE claim-bearing branches behind the gate. No dry-mode bypass exists:
    # a gate with an exemption is not a gate.
    est = estimator_admissibility(curves_by_seed, selected, admissible, battery_key="battery_gen")
    dr["estimator_admissibility"] = est
    if est["void"]:                                   # unmeasurable | insensitive
        return est["void"], points, per_seed_meta, gate, dr
    if dr["coupled"]:
        # the guard-excluded channel gates CONDITIONALLY, inside the coupled branch, because that is
        # the only place dr_ge decides anything -- gating it when dr is not coupled would VOID runs
        # whose deafness is harmless. Unguarded, a DEAF second estimator falling silent is the sole
        # decider of NO_CAPABILITY_PRICE__style_downgraded_1p5B (F1, attempt-3's fatal one level up).
        est_ge = estimator_admissibility(curves_by_seed, selected, admissible,
                                         battery_key="battery_gen_guard_excl")
        dr["estimator_admissibility_guard_excl"] = est_ge
        if est_ge["void"]:
            return ("VOID_COUPLING__estimator_insensitive__guard_excl", points, per_seed_meta,
                    gate, dr)
        if not dr_ge["coupled"]:
            # the priced slope did not survive with guard-driven items removed -> style, not capability
            dr["echo_subtractive_downgrade"] = True
            return "NO_CAPABILITY_PRICE__style_downgraded_1p5B", points, per_seed_meta, gate, dr
        dr["echo_subtractive_downgrade"] = False
        # re-panel MAJOR: dose-graded format drift on the accumulate arm demotes COUPLED to PARTIAL
        fi_dg = trained_fi_dose_graded(curves_by_seed, admissible)
        dr["trained_fi_dose_graded"] = fi_dg
        if fi_dg["downgrade"]:
            return "PARTIAL__coupling_sign_split_1p5B", points, per_seed_meta, gate, dr
        return "COUPLED__dose_slope_price_measured_1p5B", points, per_seed_meta, gate, dr
    if dr["bounded_null"]:
        return ("NO_PAIRED_PRICE_ABOVE_ESTIMATOR_MDE80__battery_not_deaf_1p5B", points,
                per_seed_meta, gate, dr)
    return "PARTIAL__coupling_sign_split_1p5B", points, per_seed_meta, gate, dr


def _thresholds():
    return {"survival": SURVIVAL, "clean_read_floor": CLEAN_READ_FLOOR,
            "disjoint_floor_clean": DISJOINT_FLOOR_CLEAN, "min_disjoint": MIN_DISJOINT,
            "clean_agg_floor": CLEAN_AGG_FLOOR, "clean_subtask_floor": CLEAN_SUBTASK_FLOOR,
            "min_effect_slope": MIN_EFFECT_SLOPE, "rank_span": list(RANK_SPAN),
            "stat_alpha": STAT_ALPHA, "k_perm": K_PERM, "auroc_floor": AUROC_FLOOR,
            "min_gate_seeds_replicating": MIN_GATE_SEEDS_REPLICATING,
            "max_fire": MAX_FIRE, "destruction_knowledge_max": DESTRUCTION_KNOWLEDGE_MAX,
            "min_positive_marked": MIN_POSITIVE_MARKED, "min_derive_nulls": MIN_DERIVE_NULLS,
            "min_gate_nulls": MIN_GATE_NULLS, "min_pairs_per_seed": MIN_PAIRS_PER_SEED,
            "min_admissible_seeds": MIN_ADMISSIBLE_SEEDS,
            "format_invariance_max": FORMAT_INVARIANCE_MAX,
            "estimator_recovery_floor": ESTIMATOR_RECOVERY_FLOOR,
            "injection_grid": list(INJECTION_GRID),
            "estimator_pseudo_runs": ESTIMATOR_PSEUDO_RUNS,
            "estimator_rng_seed": ESTIMATOR_RNG_SEED,
            "min_estimator_pool": MIN_ESTIMATOR_POOL,
            "zero_slope_atom_max": ZERO_SLOPE_ATOM_MAX,
            "derive_seeds": DERIVE_SEEDS, "gate_seeds": GATE_SEEDS, "arms_key": ARMS_KEY}


# ------------------------------------------------------------------------- accumulating training arm
def train_accumulating(base_reload, tok, attack, calib, evl, subs0, d_dep, clean_frozen, tids, fids,
                       HPC, B2, accumulate, lam, seed, steps, refit_every, correct_true,
                       correct_false, SCAN, DEPLOY, selected):
    """Verbatim-with-attribution copy of coupling_confirm_v3.train_accumulating (itself a verbatim
    copy of coupling_confirm.train_accumulating). Deltas of THIS copy, per the v4 panel brief:
      (1) `lam` is a PARAMETER (inserted after `accumulate`), used at the loss line -- v3 read a
          module-global LAM, which made a per-arm lambda impossible;
      (2) audit() measures the generation battery (gates) + raw MC battery (reported), including the
          guard-excluded generation channel (F5) and per-checkpoint echo-guard fires;
      (3) when lam == 0 the behavioral forward is skipped entirely (mathematically identical loss,
          saves the posctl arm the wasted forward);
      (4) at a pre-committed subsample of checkpoints (the pin rank, the first checkpoint reaching
          the crossing rank RANK_SPAN[1], and the final) audit() also measures format invariance on
          the SELECTED aggregate and records it (M2, REPORTED, dose-correlated format drift)."""
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed); np.random.seed(seed)
    model = base_reload()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg); model.train()
    dev = next(model.parameters()).device
    Uacc = {L: subs0[L].astype(np.float64) for L in SCAN}
    U = {L: torch.tensor(Uacc[L], dtype=torch.float32, device=dev) for L in SCAN}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=B2.LR)
    a_true = [c for c, l in attack if l == 1]; a_false = [c for c, l in attack if l == 0]
    rng = np.random.default_rng(seed)

    def batch_ids(texts):
        enc = tok(texts, return_tensors="pt", padding=True)
        return enc.input_ids.to(dev), enc.attention_mask.to(dev)

    def neutral_ids(texts):
        msgs = [[{"role": "user", "content": HPC.SYK.neutral_prompt(c)}] for c in texts]
        strs = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        enc = tok(strs, return_tensors="pt", padding=True, add_special_tokens=False)
        return enc.input_ids.to(dev), enc.attention_mask.to(dev)

    def audit(step, measure_fi=False):
        model.eval()
        p13, randp = HPC.family13_audit(model, tok, calib, evl)
        kn = HPC.eval_knowledge(model, tok, evl, tids, fids)          # the destruction-marker channel
        n6 = HPC.naive_dom6(model, tok, attack, evl)
        fz = HPC.frozen18_read(model, tok, d_dep, evl)
        bat = CB.measure_all(model, tok, tids, fids)                  # T/F battery: channel receipt
        mc = CBMC.measure_all_mc(model, tok, CBMC.letter_token_ids(tok))  # MC raw: channel receipt
        dec = CBG.make_decoder(model, tok)
        g = CBG.measure_all_gen(decode_fn=dec)                        # <-- GATES
        gen = {k: round(float(v), 4) for k, v in g["scores"].items()}
        gen["aggregate"] = round(float(CBG.aggregate(g["scores"], selected)), 4)
        gen_ge = {k: round(float(v), 4) for k, v in g["scores_guard_excl"].items()}
        gen_ge["aggregate"] = round(float(CBG.aggregate(g["scores_guard_excl"], selected)), 4)
        fi_ck = None
        if measure_fi and selected:
            f = CBG.format_invariance_check(selected=selected, decode_fn=dec)  # same decoder, reused
            fi_ck = {"abs_delta": f["abs_delta"], "aggregate_plain": f["aggregate_plain"],
                     "aggregate_verbose": f["aggregate_verbose"], "selected": list(selected)}
        model.train()
        rank = int(sum(Uacc[L].shape[1] for L in SCAN) / len(SCAN))
        ck = {"step": step, "erased_rank": rank, "private13": round(float(p13), 4),
              "knowledge": round(float(kn), 4),
              "battery": {k: round(float(v), 4) for k, v in bat.items()},
              "battery_mc": {k: round(float(v), 4) for k, v in mc.items()},
              "battery_gen": gen, "battery_gen_guard_excl": gen_ge,
              "repetition_guard_fires": int(g["repetition_guard_fires"]),
              "echo_guard_fires": int(g["echo_guard_fires"]),
              "naive6": round(float(n6), 4), "frozen": round(float(fz), 4),
              "rand": round(float(randp), 4), "bit": bool(fz < clean_frozen - 0.05)}
        if fi_ck is not None:
            ck["format_invariance"] = fi_ck
        return ck

    checkpoints = [audit(0, measure_fi=True)]                        # the rank-2 pin (M2 subsample)
    fi_crossing_done = False
    for step in range(steps):
        if step > 0 and step % refit_every == 0:
            model.eval()
            with torch.no_grad():
                subs_now = B2.gold_subspace(model, tok, attack)
            if accumulate:
                for L in SCAN:
                    Uacc[L] = B2COUP.orthonormal_union(Uacc[L], subs_now[L])
            else:
                for L in SCAN:
                    Uacc[L] = subs_now[L].astype(np.float64)
            U = {L: torch.tensor(Uacc[L], dtype=torch.float32, device=dev) for L in SCAN}
            model.train()
            cur_rank = int(sum(Uacc[L].shape[1] for L in SCAN) / len(SCAN))
            do_fi = (not fi_crossing_done and cur_rank >= RANK_SPAN[1])  # first crossing checkpoint
            fi_crossing_done = fi_crossing_done or do_fi
            checkpoints.append(audit(step, measure_fi=do_fi))
        opt.zero_grad()
        for _ in range(B2.ACCUM):
            tsel = list(rng.choice(len(a_true), B2.MICRO_BATCH // 2, replace=False))
            fsel = list(rng.choice(len(a_false), B2.MICRO_BATCH // 2, replace=False))
            texts = [a_true[i] for i in tsel] + [a_false[i] for i in fsel]
            y = torch.tensor([1] * (B2.MICRO_BATCH // 2) + [0] * (B2.MICRO_BATCH // 2), device=dev)
            ids, attn = batch_ids(texts)
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            last = attn.sum(1) - 1
            l_er = 0.0
            for L in SCAN:
                h = out.hidden_states[L]
                hlast = h[torch.arange(h.shape[0]), last, :].float()
                l_er = l_er + (hlast @ U[L]).pow(2).sum(dim=1).mean()
            l_er = l_er / len(SCAN)
            if lam != 0.0:
                nids, nattn = neutral_ids(texts)
                nout = model(input_ids=nids, attention_mask=nattn)
                nlast = nattn.sum(1) - 1
                logits = nout.logits[torch.arange(nids.shape[0]), nlast, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                losses = [-torch.logsumexp(logp[i, (correct_true if y[i].item() == 1 else correct_false)], dim=0)
                          for i in range(len(texts))]
                loss = (ERASE_ALPHA * l_er + lam * torch.stack(losses).mean()) / B2.ACCUM
            else:
                loss = (ERASE_ALPHA * l_er) / B2.ACCUM
            loss.backward()
        opt.step()
    checkpoints.append(audit(steps - 1, measure_fi=True))            # the final checkpoint (M2 subsample)
    model.eval(); del model; gc.collect()
    import torch as _t; _t.cuda.empty_cache()
    return checkpoints


# ------------------------------------------------------------------------------------ base helpers
def _load_base_and_clean(smoke):
    """Verbatim-with-attribution from coupling_confirm_v3._load_base_and_clean; delta: also scores
    the GENERATION battery on the clean base (the selection/guard surface for v4)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    B2 = _load("b2_subspace_erasure", HERE / "b2_subspace_erasure.py")
    HPC = B2.HPC
    E1, SYK, FND = HPC.E1, HPC.SYK, HPC.FND
    MODEL, SCAN, DEPLOY = HPC.MODEL, HPC.SCAN, HPC.DEPLOY
    attack, calib, evl, disjoint = E1.three_way_split(0, smoke)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def base_reload():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                    low_cpu_mem_usage=True)
    base = base_reload(); base.eval()
    tids, fids = SYK.tf_token_ids(tok)
    subs0 = B2.gold_subspace(base, tok, attack)
    a_txt = [c for c, _ in attack]; a_y = np.array([l for _, l in attack])
    a_res = SYK.resid_all(base, tok, a_txt, [DEPLOY])
    d_dep = FND.dom_direction(a_res[DEPLOY], a_y)
    if HPC.frozen18_read(base, tok, d_dep, evl) < 0.5:
        d_dep = -d_dep
    clean_frozen = float(HPC.frozen18_read(base, tok, d_dep, evl))
    clean_priv, clean_rand = HPC.family13_audit(base, tok, calib, evl)
    clean_know = float(HPC.eval_knowledge(base, tok, evl, tids, fids))
    clean_tf = {k: float(v) for k, v in CB.measure_all(base, tok, tids, fids).items()}
    g = CBG.measure_all_gen(decode_fn=CBG.make_decoder(base, tok))
    clean_gen = {k: float(v) for k, v in g["scores"].items()}
    fi = CBG.format_invariance_check(decode_fn=CBG.make_decoder(base, tok))
    ctx = {"attack": attack, "calib": calib, "evl": evl, "disjoint": bool(disjoint), "tok": tok,
           "base_reload": base_reload, "tids": tids, "fids": fids,
           "subs0": subs0, "d_dep": d_dep, "clean_frozen": clean_frozen,
           "clean_private13": float(clean_priv), "clean_knowledge": clean_know,
           "clean_tf": clean_tf, "clean_gen": clean_gen,
           "clean_gen_guard_fires": {"repetition": int(g["repetition_guard_fires"]),
                                     "echo": int(g["echo_guard_fires"])},
           "format_invariance": fi,
           "HPC": HPC, "B2": B2, "SCAN": SCAN, "DEPLOY": DEPLOY, "MODEL": MODEL}
    del base; gc.collect(); torch.cuda.empty_cache()
    return ctx


# ------------------------------------------------------------------------------------- calibration
def run_calibrate() -> dict:
    """Base-only, treatment-blind selection of the GEN battery + the clean format-invariance check;
    frozen receipt BEFORE the scored run. The receipt is ALWAYS written (an ok=False receipt is
    itself evidence) but run_real ENFORCES ok -- the v3 conformance fix."""
    ctx = _load_base_and_clean(smoke=False)
    scores = ctx["clean_gen"]
    survivors, ok = CBG.select_disjoint(scores, floor=DISJOINT_FLOOR_CLEAN, need=MIN_DISJOINT)
    # M2: SELECT FIRST, then tie the format-invariance bar to the SELECTED aggregate (the verdict's
    # own statistic), not the all-7-subtask aggregate. The full FI check already measured every
    # sub-task under both wrappers, so the selected delta re-aggregates with no extra decode.
    fi = ctx["format_invariance"]
    fi_selected = format_invariance_on_selected(fi, survivors)
    fi_ok = bool(fi_selected["abs_delta"] is not None
                 and fi_selected["abs_delta"] <= FORMAT_INVARIANCE_MAX)
    sel = {"selected_disjoint_gen": survivors, "ok": bool(ok and fi_ok),
           "selection_ok": bool(ok), "format_invariance_ok": fi_ok,
           "floor": DISJOINT_FLOOR_CLEAN, "min_disjoint": MIN_DISJOINT,
           "base_model": ctx["MODEL"], "arms_key": ARMS_KEY,
           "base_scores_gen": {k: round(v, 4) for k, v in scores.items()},
           "base_scores_tf": {k: round(v, 4) for k, v in ctx["clean_tf"].items()},
           "clean_gen_guard_fires": ctx["clean_gen_guard_fires"],
           "clean_private13": round(ctx["clean_private13"], 4),
           "clean_knowledge": round(ctx["clean_knowledge"], 4),
           "format_invariance_all_subtasks": fi,
           "format_invariance_selected": fi_selected,
           "format_invariance_max": FORMAT_INVARIANCE_MAX,
           "format_invariance_max_fraction_of_price": round(FORMAT_INVARIANCE_MAX / 0.0909, 4),
           "aggregate_selected": round(CBG.aggregate(scores, survivors), 4) if survivors else None}
    SELECTION_CACHE.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] survivors={survivors} ok={sel['ok']} (selection_ok={ok} fi_ok={fi_ok} "
          f"fi_delta_selected={fi_selected['abs_delta']}) agg={sel['aggregate_selected']} "
          f"(excluded={sorted(set(CBG.GEN_DISJOINT_POOL) - set(survivors))})", flush=True)
    return sel


# ------------------------------------------------------------------------------------- the run
def run_real(smoke: bool) -> dict:
    import torch
    selmeta = None
    if SELECTION_CACHE.exists():
        selmeta = json.loads(SELECTION_CACHE.read_text(encoding="utf-8"))
    selected = (selmeta or {}).get("selected_disjoint_gen")

    steps = 20 if smoke else STEPS
    refit = 10 if smoke else REFIT_EVERY
    seeds = [0] if smoke else SEEDS

    ctx = _load_base_and_clean(smoke)
    if selected is None and smoke:
        selected, _ = CBG.select_disjoint(ctx["clean_gen"], floor=DISJOINT_FLOOR_CLEAN, need=MIN_DISJOINT)
    guard_read = bool(ctx["clean_private13"] >= CLEAN_READ_FLOOR and ctx["disjoint"])

    if not smoke:
        pf = preflight(selmeta, ctx["MODEL"], guard_read, selected)
        if pf:
            print(f"[preflight] {pf} -- not training.", flush=True)
            return {"what": "B2-coupling attempt 4 (two-sided gate x dose slope, GEN battery)",
                    "verdict": pf, "prereg": PREREG, "model": ctx["MODEL"], "arms_key": ARMS_KEY,
                    "selection_meta": selmeta, "selected_disjoint_gen": selected,
                    "guard_read": guard_read,
                    "clean_private13": round(ctx["clean_private13"], 4),
                    "clean_knowledge": round(ctx["clean_knowledge"], 4),
                    "thresholds": _thresholds(), "points": [], "curves": {},
                    "admissibility_certificate": None, "dose_response": None, "per_seed": {}}

    attack, calib, evl = ctx["attack"], ctx["calib"], ctx["evl"]
    tok, base_reload, tids, fids = ctx["tok"], ctx["base_reload"], ctx["tids"], ctx["fids"]
    subs0, d_dep, clean_frozen = ctx["subs0"], ctx["d_dep"], ctx["clean_frozen"]
    HPC, B2, SCAN, DEPLOY = ctx["HPC"], ctx["B2"], ctx["SCAN"], ctx["DEPLOY"]
    clean_gen = {**ctx["clean_gen"],
                 "aggregate": round(CBG.aggregate(ctx["clean_gen"], selected), 4)}
    print(f"[clean] private13={ctx['clean_private13']:.4f} know={ctx['clean_knowledge']:.4f} "
          f"gen(selected {selected})={clean_gen['aggregate']:.4f} "
          f"fi_delta_all_subtasks={ctx['format_invariance']['abs_delta']}", flush=True)

    correct_true = torch.tensor(tids, device="cuda"); correct_false = torch.tensor(fids, device="cuda")

    done = {}
    if CELLS_CACHE.exists() and not smoke:
        filtered = filter_cache_records(CELLS_CACHE.read_text(encoding="utf-8").splitlines())
        done = filtered["done"]
        for seed in filtered["skipped_stale"]:
            print(f"[resume] SKIPPING seed {seed} -- stale arms_key "
                  f"(a rerun must not silently drop an arm)", flush=True)
        if filtered["corrupt"]:
            print(f"[resume] skipped {filtered['corrupt']} corrupt/truncated cache line(s)", flush=True)
        if done:
            print(f"[resume] {len(done)} seed(s) cached under this arms_key: {sorted(done)}", flush=True)

    curves = {}
    for seed in seeds:
        if seed in done:
            curves[seed] = done[seed]; continue
        arms = {}
        for name, accumulate, lam in ARMS:
            cps = train_accumulating(base_reload, tok, attack, calib, evl, subs0, d_dep, clean_frozen,
                                     tids, fids, HPC, B2, accumulate, lam, seed, steps, refit,
                                     correct_true, correct_false, SCAN, DEPLOY, selected)
            arms[name] = cps
            tail = cps[-1]
            print(f"[s{seed} {name} lam={lam}] final rank={tail['erased_rank']} "
                  f"read={tail['private13']} know={tail['knowledge']} "
                  f"gen={tail['battery_gen']['aggregate']} rep_fires={tail['repetition_guard_fires']} "
                  f"(min read {min(c['private13'] for c in cps)})", flush=True)
        curves[seed] = arms
        if not smoke:
            with CELLS_CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"seed": seed, "arms_key": ARMS_KEY, "arms": arms}) + "\n")

    verdict, points, per_seed, gate, dr = compute_verdict(curves, selected)
    fi_selected_clean = format_invariance_on_selected(ctx["format_invariance"], selected)
    res = {"what": "B2-coupling attempt 4 (two-sided gate x dose slope, GEN battery)",
           "prereg": PREREG, "model": ctx["MODEL"], "seeds": seeds, "arms_key": ARMS_KEY,
           "selected_disjoint_gen": selected, "selection_meta": selmeta,
           "guard_read": guard_read,
           "clean_battery_gen": clean_gen, "clean_battery_tf": ctx["clean_tf"],
           "clean_private13": round(ctx["clean_private13"], 4),
           "clean_knowledge": round(ctx["clean_knowledge"], 4),
           "format_invariance_all_subtasks": ctx["format_invariance"],
           "format_invariance_selected": fi_selected_clean,
           "trained_checkpoint_format_invariance": trained_fi_summary(curves),
           "guard_fire_totals_by_arm": guard_fire_totals(points),
           "channel_gain": channel_gain_receipt(points, clean_gen["aggregate"],
                                                 round(ctx["clean_knowledge"], 4)),
           "thresholds": _thresholds(), "verdict": verdict,
           "points": points, "per_seed": per_seed,
           "admissibility_certificate": (gate or {}).get("certificate"),
           "admissibility_gate": {k: v for k, v in (gate or {}).items() if k != "certificate"} or None,
           "dose_response": dr,
           "curves": {str(k): v for k, v in curves.items()}}
    if smoke:
        pos_marked = [p for p in points if p["arm"] == "posctl" and p["step"] > 0
                      and p["knowledge"] <= DESTRUCTION_KNOWLEDGE_MAX]
        res["smoke_gates"] = {
            "all_arms_trained": all(a in curves.get(0, {}) for a, _, _ in ARMS),
            "posctl_marker_fired_any": bool(pos_marked),
            "posctl_gen_min": min((p["gen_aggregate"] for p in points if p["arm"] == "posctl"),
                                  default=None),
            "fixed_gen_min": min((p["gen_aggregate"] for p in points if p["arm"] == "fixed"),
                                 default=None),
            "format_invariance_ok": bool(fi_selected_clean["abs_delta"] is not None
                                         and fi_selected_clean["abs_delta"] <= FORMAT_INVARIANCE_MAX)}
    return res


# ----------------------------------------------------------------------------- CPU-only logic check
def run_dry() -> dict:
    selected = ["ANTONYM_GEN", "ORTH_FIRST_GEN", "SEQ_GEN"]
    STEPS_ = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    RANKS_ = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    def j(i):                       # deterministic jitter: non-degenerate scores, cancels in deltas
        return 0.004 * ((i % 3) - 1)

    # ---- per-seed COMMON-MODE offset (panel #7 section 5, "implementation consequence") ----
    # j(i) alone gave all five seeds byte-identical fixed arms, so every cross-seed pseudo-delta was
    # exactly 0 and the estimator pool was exactly degenerate: without this, EVERY fixture below would
    # VOID under the new gate. That is the gate working, not a bug -- these fixtures were never built
    # to carry seed-level noise. nz is applied identically to a seed's accumulate AND fixed arms, so
    # every paired delta (and therefore every per-seed slope, every verdict and every sharp arithmetic
    # assertion this file already pins) is EXACTLY unchanged. Deterministic; no RNG.
    NZ = 0.012                      # ~ the 0.0100 per-checkpoint dispersion row of the panel's table
    NZ_BIG = 0.25                   # ~3x the design's own per-checkpoint SE (D2)
    NZ_TRIPLE = (-1.0, 0.0, 1.0)
    NZ_ROT = ((0, 0, 0), (0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2))   # 5 distinct per-seed patterns

    def nz(seed, i, scale=NZ):
        """Permutes values WITHIN each residue class of i mod 3. j(i) is constant on those classes, so
        every seed's step>0 fixed-arm multiset is IDENTICAL: the conformal fire threshold derived on
        DERIVE_SEEDS still sits at the gate-null minimum and no fixture's GATE verdict moves. seed=None
        reproduces the pre-gate fixtures byte-for-byte (used by D1/D4)."""
        if seed is None or i == 0:
            return 0.0
        return scale * NZ_TRIPLE[((i - 1) // 3 + NZ_ROT[seed % 5][i % 3]) % 3]

    def bat(agg):
        d = {n: agg for n in selected}
        d["aggregate"] = CBG.aggregate(d, selected)
        return d

    def cp(step, rank, read, know, genagg, genagg_ge=None, bit=True, echo=0, rep=0, fi=None):
        ge = genagg if genagg_ge is None else genagg_ge
        d = {"step": step, "erased_rank": rank, "private13": read, "knowledge": know,
             "battery_gen": bat(genagg), "battery_gen_guard_excl": bat(ge),
             "repetition_guard_fires": rep, "echo_guard_fires": echo, "bit": bit}
        if fi is not None:
            d["format_invariance"] = {"abs_delta": fi}   # trained-checkpoint FI (M2 / re-panel MAJOR)
        return d

    crossing = [0.93, 0.85, 0.82, 0.66, 0.75, 0.78, 0.80, 0.79, 0.81, 0.80]
    fixed_reads = [0.93, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78]

    def acc_arm(gen_fn, bite=True, gen_ge_fn=None, fi_map=None, seed=None, nz_scale=NZ, ge_nz=True,
                ranks=None):
        rk = RANKS_ if ranks is None else ranks
        return [cp(s, r, crossing[i], 0.75, gen_fn(r) + j(i) + nz(seed, i, nz_scale),
                   genagg_ge=(None if gen_ge_fn is None else
                              gen_ge_fn(r) + j(i) + (nz(seed, i, nz_scale) if ge_nz else 0.0)),
                   fi=(None if fi_map is None else fi_map.get(r)),
                   bit=(bite and i >= 1))
                for i, (s, r) in enumerate(zip(STEPS_, rk))]

    def acc_arm_stalled(gen_fn=lambda r: 0.90, seed=None):
        # erased_rank PINNED at 4 -> no distinct FITTED rank in [2,8] -> per-seed slope None, yet
        # still 10 matched pairs (>= MIN_PAIRS_PER_SEED, dose-admissible). The FATAL regression fixture.
        return [cp(s, 4, crossing[i], 0.75, gen_fn(4) + j(i) + nz(seed, i), bit=(i >= 1))
                for i, s in enumerate(STEPS_)]

    def fixed_arm(gen_fn=lambda r: 0.90, know=0.78, bite=True, seed=None, nz_scale=NZ, base_ge=None):
        return [cp(s, 2, fixed_reads[i], know, gen_fn_i(gen_fn, i) + j(i) + nz(seed, i, nz_scale),
                   genagg_ge=(None if base_ge is None else base_ge + j(i)),
                   bit=(bite and i >= 1))
                for i, s in enumerate(STEPS_)]

    def gen_fn_i(fn, i):            # allow index-aware null generators (nonspecific case)
        try:
            return fn(2, i)
        except TypeError:
            return fn(2)

    def posctl_arm(gen=0.50, know_collapsed=True, collapse_idx=None):
        def kn(i):
            if i == 0:
                return 0.82
            if not know_collapsed:
                return 0.80
            return 0.40 if (collapse_idx is None or i in collapse_idx) else 0.80
        return [cp(s, r, 0.72, kn(i), (0.90 if i == 0 else gen) + j(i), bit=(i >= 1))
                for i, (s, r) in enumerate(zip(STEPS_, RANKS_))]

    # every seed_* constructor now takes the seed (panel #7 section 5). seed=None reproduces the
    # pre-gate fixture byte-for-byte -- which is exactly the degenerate-pool construction D1/D4 need.
    def seed_coupled(seed):         # every seed: +0.02 span-[2,8] slope -> majority above bar, +sign
        return {"accumulate": acc_arm(lambda r: 0.92 - 0.02 * (r - 2), seed=seed),
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}

    def seed_flat(seed):            # flat -> all seeds below bar -> bounded null
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed),
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}

    def seed_insensitive(seed):     # marker fires, gen aggregate FLAT on posctl (deaf battery)
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed),
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm(gen=0.90)}

    def seed_nonspecific(seed):     # gate-seed nulls dip below the derive-seed threshold
        if seed in GATE_SEEDS:
            fx = fixed_arm(gen_fn=lambda r, i: 0.70 if i in (2, 5, 8) else 0.90, seed=seed)
        else:
            fx = fixed_arm(seed=seed)
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed), "fixed": fx,
                "posctl": posctl_arm(gen=0.45)}

    def seed_unmeasurable(seed):    # destruction marker NEVER fires -> no positives
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed),
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm(know_collapsed=False)}

    def seed_partial(seed):         # SIGN SPLIT: 3 seeds +slope above bar, 2 seeds -slope above bar
        if seed in (0, 1, 2):
            return {"accumulate": acc_arm(lambda r: 0.92 - 0.02 * (r - 2), seed=seed),
                    "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}
        return {"accumulate": acc_arm(lambda r: 0.88 + 0.03 * (r - 2), seed=seed),
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}

    def seed_echo_downgrade(seed):  # raw gen priced (+0.02); guard-EXCLUDED gen flat -> style, downgrade
        return {"accumulate": acc_arm(lambda r: 0.92 - 0.02 * (r - 2), gen_ge_fn=lambda r: 0.90,
                                      seed=seed),
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}

    def seed_replication_fail(seed):  # only gate seed 2 separates posctl<-fixed on GEN; 3,4 stay deaf
        if seed == 2:
            posc = posctl_arm(gen=0.45)                          # 9 low positives -> separates
        elif seed in (3, 4):
            posc = posctl_arm(gen=0.90, collapse_idx={1, 2})     # 2 marker fires, GEN HIGH -> no separation
        else:
            posc = posctl_arm(gen=0.45)                          # derive seeds
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed), "fixed": fixed_arm(seed=seed),
                "posctl": posc}

    def seed_stalled_or_one(seed):    # re-panel FATAL fixture: 4 stalled(None-slope) + 1 above-bar
        if seed == 2:                 # -> 1 non-None slope < MIN_ADMISSIBLE_SEEDS -> underpowered, NEVER coupled
            return {"accumulate": acc_arm(lambda r: 0.92 - 0.02 * (r - 2), seed=seed),
                    "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}
        return {"accumulate": acc_arm_stalled(seed=seed), "fixed": fixed_arm(seed=seed),
                "posctl": posctl_arm()}

    def seed_fi_downgrade(seed):      # raw COUPLED, echo-survives, but accumulate-arm FI dose-graded
        return {"accumulate": acc_arm(lambda r: 0.92 - 0.02 * (r - 2), seed=seed,
                                      fi_map={2: 0.01, 8: 0.07}),   # FI slope 0.06/6 = 0.01 >= 0.0076
                "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}

    def seed_degenerate_instrument(seed):  # gate counts adequate, but posctl & fixed gen are ONE value
        # -> instrument scores degenerate -> VOID_INSTRUMENT__unmeasurable via the instrument path
        # (re-panel minor 10), NOT the count early-return. No jitter: all gate scores == 0.90.
        fx = [cp(s, 2, fixed_reads[i], 0.78, 0.90, bit=(i >= 1)) for i, s in enumerate(STEPS_)]
        posc = [cp(s, r, 0.72, 0.82 if i == 0 else 0.40, 0.90, bit=(i >= 1))
                for i, (s, r) in enumerate(zip(STEPS_, RANKS_))]
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed), "fixed": fx, "posctl": posc}

    def seed_nobite(seed):
        return {"accumulate": acc_arm(lambda r: 0.90, bite=False, seed=seed),
                "fixed": fixed_arm(bite=False, seed=seed), "posctl": posctl_arm()}

    # ---- panel #7 D2 / D5 / D6 fixture arms ----
    def seed_estimator_insensitive(seed):   # D2: fixed arms at ~3x the design's per-checkpoint SE
        return {"accumulate": acc_arm(lambda r: 0.70, seed=seed, nz_scale=NZ_BIG),
                "fixed": fixed_arm(gen_fn=lambda r: 0.70, seed=seed, nz_scale=NZ_BIG),
                "posctl": posctl_arm(gen=0.30)}

    def seed_guard_excl_deaf(seed):         # D5: raw channel priced and healthy; ge channel FROZEN
        return {"accumulate": acc_arm(lambda r: 0.92 - 0.02 * (r - 2), gen_ge_fn=lambda r: 0.90,
                                      seed=seed, ge_nz=False),
                "fixed": fixed_arm(seed=seed, base_ge=0.90), "posctl": posctl_arm()}

    def seed_pool_drops(seed):              # D6: seed 4's accumulate schedule is PINNED, so its four
        if seed == 4:                       # pairs share only one in-span rank and drop from the pool,
            return {"accumulate": acc_arm_stalled(seed=seed),   # and its own real slope is None too
                    "fixed": fixed_arm(seed=seed), "posctl": posctl_arm()}
        return {"accumulate": acc_arm(lambda r: 0.90, seed=seed), "fixed": fixed_arm(seed=seed),
                "posctl": posctl_arm()}

    BOUNDED = "NO_PAIRED_PRICE_ABOVE_ESTIMATOR_MDE80__battery_not_deaf_1p5B"
    cases = {
        "coupled_dose_slope": ({s: seed_coupled(s) for s in SEEDS},
                               "COUPLED__dose_slope_price_measured_1p5B"),
        "bounded_null": ({s: seed_flat(s) for s in SEEDS}, BOUNDED),
        "echo_subtractive_downgrade": ({s: seed_echo_downgrade(s) for s in SEEDS},
                                       "NO_CAPABILITY_PRICE__style_downgraded_1p5B"),
        "fi_dose_downgrade": ({s: seed_fi_downgrade(s) for s in SEEDS},
                              "PARTIAL__coupling_sign_split_1p5B"),
        "stalled_seeds_underpowered": ({s: seed_stalled_or_one(s) for s in SEEDS},
                                       "VOID_COUPLING__underpowered"),
        "battery_insensitive": ({s: seed_insensitive(s) for s in SEEDS},
                                "VOID_COUPLING__battery_insensitive"),
        "gate_replication_fail": ({s: seed_replication_fail(s) for s in SEEDS},
                                  "VOID_COUPLING__battery_insensitive"),
        "battery_nonspecific": ({s: seed_nonspecific(s) for s in SEEDS},
                                "VOID_COUPLING__battery_nonspecific"),
        "admissibility_unmeasurable": ({s: seed_unmeasurable(s) for s in SEEDS},
                                       "VOID_COUPLING__admissibility_unmeasurable"),
        "degenerate_instrument": ({s: seed_degenerate_instrument(s) for s in SEEDS},
                                  "VOID_COUPLING__admissibility_unmeasurable"),
        "partial_sign_split": ({s: seed_partial(s) for s in SEEDS},
                               "PARTIAL__coupling_sign_split_1p5B"),
        "no_bite": ({s: seed_nobite(s) for s in SEEDS}, "VOID_COUPLING__no_bite"),
        "underpowered": ({s: (seed_coupled(s) if s in (0, 1) else seed_nobite(s)) for s in SEEDS},
                         "VOID_COUPLING__underpowered"),
        # ---- panel #7 D1/D2/D4/D5/D6: the estimator-admissibility gate ----
        # D1: all five fixed arms byte-identical (seed=None) -> every pseudo-delta exactly 0 -> the
        # zero atom is 1.0. This is the killed spec's signature: a FROZEN channel is precisely where
        # S1-S4 returned a flawless two-sided pass and would have shipped a bounded null carrying
        # "recovery 1.00". Here it is the one thing that cannot pass.
        "estimator_pool_degenerate": ({s: seed_flat(None) for s in SEEDS},
                                      "VOID_COUPLING__estimator_unmeasurable"),
        # D2: a NOISY estimator's silence must not ship the bounded null -- the branch the whole gate
        # exists to earn. No grid point up to 8x bar reaches the recovery floor.
        "estimator_insensitive": ({s: seed_estimator_insensitive(s) for s in SEEDS},
                                  "VOID_COUPLING__estimator_insensitive"),
        # D4: the SAME five claim-bearing fixtures with the D1 degenerate fixed arms substituted. All
        # five must VOID. This pins F8's placement as a BEHAVIOURAL invariant, not a line number: if
        # anyone moves the insertion point behind a finding branch, these fire.
        "estimator_gate_precedes_coupled": ({s: seed_coupled(None) for s in SEEDS},
                                            "VOID_COUPLING__estimator_unmeasurable"),
        "estimator_gate_precedes_echo_downgrade": ({s: seed_echo_downgrade(None) for s in SEEDS},
                                                   "VOID_COUPLING__estimator_unmeasurable"),
        "estimator_gate_precedes_fi_downgrade": ({s: seed_fi_downgrade(None) for s in SEEDS},
                                                 "VOID_COUPLING__estimator_unmeasurable"),
        "estimator_gate_precedes_bounded_null": ({s: seed_flat(None) for s in SEEDS},
                                                 "VOID_COUPLING__estimator_unmeasurable"),
        "estimator_gate_precedes_partial": ({s: seed_partial(None) for s in SEEDS},
                                            "VOID_COUPLING__estimator_unmeasurable"),
        # D5: F1. A DEAF second estimator must not be allowed to convert a genuine COUPLED into "the
        # price was style, not capability" by falling silent. NOT the style-downgrade string.
        "guard_excl_estimator_gated": ({s: seed_guard_excl_deaf(s) for s in SEEDS},
                                       "VOID_COUPLING__estimator_insensitive__guard_excl"),
        # D6: pool units drop and the real side runs short of slopes, so the pseudo-run is PADDED.
        "estimator_stalled_pseudo_seeds": ({s: seed_pool_drops(s) for s in SEEDS}, BOUNDED),
    }
    checks = {}
    for name, (curves, expect) in cases.items():
        v, points, per_seed, gate, dr = compute_verdict(curves, selected)
        checks[name] = {"verdict": v, "expected": expect, "ok": v == expect}
        print(f"[dry {name}] -> {v}  ({'OK' if v == expect else 'MISMATCH vs ' + expect})", flush=True)

    # sharp-case arithmetic assertions
    v, points, per_seed, gate, dr = compute_verdict(cases["coupled_dose_slope"][0], selected)
    s0 = dr["per_seed_slopes"]["0"]
    checks["coupled_per_seed_slope"] = {              # span-[2,8], step-0-excluded slope = 0.02
        "ok": abs(s0 - 0.02) < 1e-6 and dr["n_above_bar_positive"] == 5
              and dr["n_above_bar_negative"] == 0 and dr["coupled"] is True,
        "slope_seed0": s0, "n_above_pos": dr["n_above_bar_positive"]}
    print(f"[dry coupled_per_seed_slope] -> slope0={s0} above+={dr['n_above_bar_positive']}", flush=True)
    checks["coupled_p_is_reported_nongating"] = {     # p exists but does NOT gate; 5-seed floor 0.0625
        "ok": dr["reported_nongating"]["pooled_perm_p"] is not None
              and abs(dr["reported_nongating"]["seed_sign_flip_p_floor"] - 0.0625) < 1e-9,
        "pooled_p": dr["reported_nongating"]["pooled_perm_p"],
        "sign_flip_floor": dr["reported_nongating"]["seed_sign_flip_p_floor"]}
    print(f"[dry coupled_p_is_reported_nongating] -> pooled_p="
          f"{dr['reported_nongating']['pooled_perm_p']} floor="
          f"{dr['reported_nongating']['seed_sign_flip_p_floor']}", flush=True)
    checks["coupled_survives_subtractive"] = {
        "ok": dr["echo_subtractive_downgrade"] is False
              and dr["guard_excluded_subtractive"]["coupled"] is True}
    checks["coupled_gate_scope_and_replication"] = {
        "ok": gate["gate_pass"] and "NOT-STRUCTURALLY-DEAF" in gate["scope"]
              and gate["certificate"]["admissible"] is True
              and gate["n_gate_seeds_replicating"] == 3,
        "fire_threshold": gate["fire_threshold"],
        "n_replicating": gate["n_gate_seeds_replicating"]}

    # bounded null: majority below bar; the pooled p95 is REPORTED (non-gating) as a detection reference
    v, points, per_seed, gate, dr = compute_verdict(cases["bounded_null"][0], selected)
    checks["bounded_null_reports_mde"] = {
        "ok": dr is not None and dr["bounded_null"] is True
              and dr["reported_nongating"]["pooled_perm_p95"] is not None,
        "mde": dr["reported_nongating"]["pooled_perm_p95"], "n_below": dr["n_below_bar"]}
    print(f"[dry bounded_null] -> below={dr['n_below_bar']} "
          f"mde={dr['reported_nongating']['pooled_perm_p95']}", flush=True)

    # echo-subtractive downgrade: raw is COUPLED, guard-excluded is NOT -> distinct style string
    v, points, per_seed, gate, dr = compute_verdict(cases["echo_subtractive_downgrade"][0], selected)
    checks["echo_downgrade_sharp"] = {
        "ok": dr["coupled"] is True and dr["echo_subtractive_downgrade"] is True
              and dr["guard_excluded_subtractive"]["coupled"] is False
              and v == "NO_CAPABILITY_PRICE__style_downgraded_1p5B",
        "raw_coupled": dr["coupled"], "guard_excl_coupled": dr["guard_excluded_subtractive"]["coupled"]}
    print(f"[dry echo_downgrade_sharp] -> raw_coupled={dr['coupled']} "
          f"guard_excl_coupled={dr['guard_excluded_subtractive']['coupled']} -> {v}", flush=True)

    # LOAD-BEARING regression (re-panel FATAL): 4 stalled(None-slope) + 1 above-bar must NEVER be
    # COUPLED -- majority denominator is dose-admissible seeds (5), and only 1 has a slope -> underpowered
    v, points, per_seed, gate, dr = compute_verdict(cases["stalled_seeds_underpowered"][0], selected)
    checks["stalled_seeds_never_coupled"] = {
        "ok": v == "VOID_COUPLING__underpowered" and v != "COUPLED__dose_slope_price_measured_1p5B"
              and dr["n_admissible_seeds"] == 5 and dr["n_admissible_slopes"] == 1
              and dr["coupled"] is False,
        "verdict": v, "n_admissible_seeds": dr["n_admissible_seeds"],
        "n_slopes": dr["n_admissible_slopes"], "coupled": dr["coupled"]}
    print(f"[dry stalled_seeds_never_coupled] -> n_adm={dr['n_admissible_seeds']} "
          f"n_slopes={dr['n_admissible_slopes']} coupled={dr['coupled']} -> {v}", flush=True)

    # LOAD-BEARING (re-panel MAJOR): dose-graded accumulate-arm FI demotes a raw COUPLED to PARTIAL
    v, points, per_seed, gate, dr = compute_verdict(cases["fi_dose_downgrade"][0], selected)
    checks["fi_dose_graded_downgrade"] = {
        "ok": dr["coupled"] is True and dr["echo_subtractive_downgrade"] is False
              and dr["trained_fi_dose_graded"]["downgrade"] is True
              and v == "PARTIAL__coupling_sign_split_1p5B",
        "raw_coupled": dr["coupled"], "fi_downgrade": dr["trained_fi_dose_graded"]["downgrade"],
        "fi_bar": dr["trained_fi_dose_graded"]["bar"], "verdict": v}
    print(f"[dry fi_dose_graded_downgrade] -> raw_coupled={dr['coupled']} "
          f"fi_downgrade={dr['trained_fi_dose_graded']['downgrade']} -> {v}", flush=True)

    # degenerate instrument routes through the INSTRUMENT path (not the count early-return)
    v, points, per_seed, gate, dr = compute_verdict(cases["degenerate_instrument"][0], selected)
    checks["degenerate_instrument_via_instrument_path"] = {
        "ok": v == "VOID_COUPLING__admissibility_unmeasurable"
              and gate["certificate"] is not None
              and gate["certificate"]["admissibility_verdict"] == "VOID_INSTRUMENT__unmeasurable"
              and gate["n_positive_marked_gate"] >= MIN_POSITIVE_MARKED,
        "instrument_verdict": (gate["certificate"] or {}).get("admissibility_verdict"),
        "n_pos": gate["n_positive_marked_gate"]}
    print(f"[dry degenerate_instrument] -> instrument="
          f"{(gate['certificate'] or {}).get('admissibility_verdict')} "
          f"n_pos={gate['n_positive_marked_gate']} -> {v}", flush=True)

    # sign-split PARTIAL: 3 seeds +slope above bar, 2 seeds -slope above bar
    v, points, per_seed, gate, dr = compute_verdict(cases["partial_sign_split"][0], selected)
    checks["partial_sign_split_sharp"] = {
        "ok": dr["n_above_bar_positive"] == 3 and dr["n_above_bar_negative"] == 2
              and dr["coupled"] is False and dr["bounded_null"] is False,
        "above_pos": dr["n_above_bar_positive"], "above_neg": dr["n_above_bar_negative"]}

    # minor 5: gate ledger reports the guard-fire fraction among marker-fired gate-seed positives
    v, points, per_seed, gate, dr = compute_verdict(cases["coupled_dose_slope"][0], selected)
    checks["gate_reports_guard_fraction"] = {
        "ok": "guard_fire_fraction_marked_positives" in gate
              and gate["guard_fire_fraction_marked_positives"] == 0.0,
        "frac": gate["guard_fire_fraction_marked_positives"]}

    # per-seed gate replication: pooled instrument ADMISSIBLE, but only 1 of 3 gate seeds separates
    v, points, per_seed, gate, dr = compute_verdict(cases["gate_replication_fail"][0], selected)
    checks["gate_replication_leg_is_decider"] = {
        "ok": v == "VOID_COUPLING__battery_insensitive"
              and gate["certificate"]["admissibility_verdict"] == "ADMISSIBLE"
              and gate["n_gate_seeds_replicating"] == 1,
        "instrument_verdict": gate["certificate"]["admissibility_verdict"],
        "n_replicating": gate["n_gate_seeds_replicating"]}
    print(f"[dry gate_replication_leg] -> instrument="
          f"{gate['certificate']['admissibility_verdict']} "
          f"n_replicating={gate['n_gate_seeds_replicating']} -> {v}", flush=True)

    # ---- panel #7 D3: the bounded null must carry the level it is actually bounded above ----
    # T8: a recovery certificate at ESTIMATOR_MDE80 licenses "no paired price above ESTIMATOR_MDE80",
    # never "above MIN_EFFECT_SLOPE". If the claim ever silently reverts to the 0.0152 bar, this fires.
    v, points, per_seed, gate, dr = compute_verdict(cases["bounded_null"][0], selected)
    est = dr["estimator_admissibility"]
    checks["bounded_null_carries_mde80"] = {
        "ok": v == BOUNDED and est["void"] is None and est["mde80"] is not None
              and est["mde80"] > MIN_EFFECT_SLOPE
              and est["span_bound"] == est["mde80"] * 6
              and est["scope"] == ESTIMATOR_SCOPE
              and est["false_recovery_analytic_max"] == FALSE_RECOVERY_ANALYTIC_MAX[5],
        "mde80": est["mde80"], "span_bound": est["span_bound"],
        "replicates": est["mde80_replicates"], "zero_atom": est["zero_slope_atom"],
        "false_recovery_rate": est["false_recovery_rate"]}
    print(f"[dry bounded_null_carries_mde80] -> mde80={est['mde80']} "
          f"({est['mde80_multiple_of_bar']}x bar) span_bound={round(est['span_bound'], 6)} "
          f"FR={est['false_recovery_rate']} (analytic max "
          f"{est['false_recovery_analytic_max']})", flush=True)

    # ---- panel #7 D6: dropped pool units and PADDED pseudo-runs ----
    # T6/F7: panel #6's reduced-denominator FATAL must not recur INSIDE the gate built to prevent it.
    # Padding with None slots that count against the majority can only LOWER recovery; scoring on the
    # reduced denominator instead would inflate it, which is precisely the defect.
    v, points, per_seed, gate, dr = compute_verdict(cases["estimator_stalled_pseudo_seeds"][0], selected)
    est = dr["estimator_admissibility"]
    sub = np.asarray([u["slope"] for u in est["pool_units"] if 0 not in u["pair"]], dtype=float)
    padded, unpadded, strict = [], [], False
    for k in INJECTION_GRID:                    # SAME rng seed on both sides -> the draws are paired
        rp, _ = _pseudo_rates(sub, len(sub), est["n_admissible"], k,
                              np.random.default_rng(ESTIMATOR_RNG_SEED))
        ru, _ = _pseudo_rates(sub, len(sub), len(sub), k,
                              np.random.default_rng(ESTIMATOR_RNG_SEED))
        padded.append(round(rp, 4)); unpadded.append(round(ru, 4))
        strict = strict or rp < ru
    checks["estimator_stalled_pseudo_seeds_padding"] = {
        "ok": v == BOUNDED and est["n_pool"] == 6 and est["n_pool_dropped"] == 4
              and est["n_slopes"] == 4 and est["n_admissible"] == 5
              and est["n_padded_slots"] == 1
              and all(a <= b for a, b in zip(padded, unpadded)) and strict,
        "n_pool": est["n_pool"], "n_dropped": est["n_pool_dropped"],
        "n_slopes": est["n_slopes"], "n_padded_slots": est["n_padded_slots"],
        "recovery_padded": padded, "recovery_unpadded": unpadded}
    print(f"[dry estimator_stalled_pseudo_seeds] -> pool={est['n_pool']} "
          f"dropped={est['n_pool_dropped']} n_slopes={est['n_slopes']} "
          f"pad={est['n_padded_slots']} padded<=unpadded={all(a <= b for a, b in zip(padded, unpadded))} "
          f"strict_somewhere={strict}", flush=True)

    # ---- panel #7 D7: THE ANTI-TAUTOLOGY FIXTURE. The gate must READ THE DATA ----
    # The killed S1-S4 spec would have failed exactly here: its recovery rate was pinned at 0.500
    # across an 18x range of realized dispersion, so its output did not depend on the data at all. Two
    # pools whose dispersion differs by 15x must land at least two grid steps apart in MDE80.
    unit = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    unit = unit / float(np.std(unit))
    grid = list(INJECTION_GRID)
    mde_lo, curve_lo = _estimator_mde80(unit * (0.2 * MIN_EFFECT_SLOPE), 5, 5,
                                        np.random.default_rng(ESTIMATOR_RNG_SEED))
    mde_hi, curve_hi = _estimator_mde80(unit * (3.0 * MIN_EFFECT_SLOPE), 5, 5,
                                        np.random.default_rng(ESTIMATOR_RNG_SEED))
    i_lo = grid.index(mde_lo) if mde_lo is not None else len(grid)   # None = off the top of the grid
    i_hi = grid.index(mde_hi) if mde_hi is not None else len(grid)
    checks["estimator_reads_the_data"] = {
        "ok": i_hi - i_lo >= 2,
        "pool_sd_0p2x_bar_mde80": mde_lo, "pool_sd_3x_bar_mde80": mde_hi,
        "grid_steps_apart": i_hi - i_lo,
        "recovery_curve_low": [c["recovery"] for c in curve_lo],
        "recovery_curve_high": [c["recovery"] for c in curve_hi]}
    print(f"[dry estimator_reads_the_data] -> mde80(0.2x bar SD)={mde_lo} "
          f"mde80(3x bar SD)={mde_hi} steps_apart={i_hi - i_lo}", flush=True)

    # ---- panel #7 D8: the specificity leg is REPORTED, never gating ----
    # Silently re-gating on a leg that cannot fire is how styxx.instrument_admissibility's default
    # specificity leg got voided. Perturb the reported rate to 0.99 and recompute the BRANCH: the
    # verdict must not move, and both the measured rate and its analytic max must be on the receipt.
    base_v, _, _, _, base_dr = compute_verdict(cases["bounded_null"][0], selected)
    _real_est = globals()["estimator_admissibility"]

    def _perturbed_est(*a, **kw):
        e = dict(_real_est(*a, **kw))
        e["false_recovery_rate"] = 0.99
        return e

    globals()["estimator_admissibility"] = _perturbed_est
    try:
        pert_v, _, _, _, pert_dr = compute_verdict(cases["bounded_null"][0], selected)
    finally:
        globals()["estimator_admissibility"] = _real_est
    base_est = base_dr["estimator_admissibility"]
    checks["false_recovery_is_reported_not_gating"] = {
        "ok": pert_v == base_v == BOUNDED
              and pert_dr["estimator_admissibility"]["false_recovery_rate"] == 0.99
              and pert_dr["estimator_admissibility"]["void"] is None
              and base_est["false_recovery_rate"] is not None
              and base_est["false_recovery_analytic_max"] is not None
              and base_est["false_bounded_null_rate"] is not None,
        "verdict_unperturbed": base_v, "verdict_perturbed": pert_v,
        "false_recovery_rate": base_est["false_recovery_rate"],
        "false_recovery_analytic_max": base_est["false_recovery_analytic_max"],
        "false_bounded_null_rate": base_est["false_bounded_null_rate"]}
    print(f"[dry false_recovery_is_reported_not_gating] -> perturbed FR=0.99 -> {pert_v} "
          f"(unperturbed {base_v})", flush=True)

    # M3 channel-gain receipt (REPORTED, non-gating): regress gen-drop on knowledge-drop
    cg = channel_gain_receipt(
        [{"arm": "posctl", "step": 25, "seed": 2, "knowledge": 0.40, "gen_aggregate": 0.50},
         {"arm": "posctl", "step": 50, "seed": 3, "knowledge": 0.30, "gen_aggregate": 0.40}],
        0.90, 0.82)
    checks["channel_gain_receipt_computes"] = {
        "ok": cg["coef_gen_drop_per_knowledge_drop"] is not None, "coef": cg}

    # M2 trained-checkpoint FI summary is graceful when no checkpoint carried an FI measurement
    checks["trained_fi_summary_graceful"] = {
        "ok": trained_fi_summary(cases["coupled_dose_slope"][0]) is None}

    # points grounding surface: echo + guard-excluded aggregate are now on it (F5)
    _, points, _, _, _ = compute_verdict(cases["coupled_dose_slope"][0], selected)
    checks["points_schema"] = {
        "ok": all(set(p) == {"seed", "arm", "step", "erased_rank", "private13", "knowledge",
                             "gen_aggregate", "gen_aggregate_guard_excl", "repetition_guard_fires",
                             "echo_guard_fires", "bit"} for p in points)
              and len(points) == 5 * 3 * 10}

    # conformal threshold arithmetic
    t = conformal_fire_threshold(list(np.linspace(0.5, 1.0, 20)), 0.15)  # k=floor(0.15*21)=3
    checks["conformal_kth_smallest"] = {"ok": abs(t - sorted(np.linspace(0.5, 1.0, 20))[2]) < 1e-12,
                                        "t": t}
    checks["conformal_too_few_none"] = {"ok": conformal_fire_threshold([0.9, 0.8], 0.15) is None}

    # preflight conformance (the v3 panel's code-vs-prereg bugs)
    okmeta = {"ok": True, "base_model": "MODEL_X", "selected_disjoint_gen": selected,
              "base_scores_gen": {n: 1.0 for n in selected}}
    pf_cases = {
        "no_calibration": ((None, "MODEL_X", True, selected), "VOID_COUPLING__no_calibration"),
        "model_mismatch": (({**okmeta, "base_model": "MODEL_Y"}, "MODEL_X", True, selected),
                           "VOID_COUPLING__calibration_model_mismatch"),
        "clean_guard": ((okmeta, "MODEL_X", False, selected), "VOID_COUPLING__clean_guard_failed"),
        "selection_not_ok": (({**okmeta, "ok": False}, "MODEL_X", True, selected),
                             "VOID_COUPLING__battery_guard_failed"),
        "too_few_selected": (({**okmeta, "selected_disjoint_gen": selected[:2]}, "MODEL_X", True,
                              selected[:2]), "VOID_COUPLING__battery_guard_failed"),
        "all_pass": ((okmeta, "MODEL_X", True, selected), None),
    }
    for name, (args, expect) in pf_cases.items():
        got = preflight(*args)
        checks["preflight_" + name] = {"verdict": got, "expected": expect, "ok": got == expect}
        print(f"[dry preflight_{name}] -> {got}  ({'OK' if got == expect else 'MISMATCH'})", flush=True)

    # resume-cache arms_key filter -- drives the SAME production function run_real uses (minor 10)
    lines = [json.dumps({"seed": 0, "arms_key": "stale", "arms": {}}),
             "   not json at all   ",
             json.dumps({"seed": 1, "arms_key": ARMS_KEY, "arms": {"accumulate": []}})]
    filt = filter_cache_records(lines)
    checks["resume_skips_stale_arms_key"] = {
        "ok": list(filt["done"]) == [1] and filt["skipped_stale"] == [0] and filt["corrupt"] == 1}

    # frozen verdict set hygiene: no break-claim string is reachable
    bad = [v for v in FROZEN_VERDICTS
           if "DECOUPLED" in v or "read_neq_write_BROKEN" in v or "capability_held" in v.lower()
           or "capability held" in v.lower()]
    checks["no_break_claim_string_exists"] = {"ok": bad == [], "bad": bad}
    checks["dry_verdicts_all_frozen"] = {
        "ok": all(v in FROZEN_VERDICTS or v is None for v in
                  [c.get("verdict") for c in checks.values()])}

    all_ok = all(c["ok"] for c in checks.values())
    return _py({"dry": True, "arms_key": ARMS_KEY, "thresholds": _thresholds(),
                "frozen_verdicts": FROZEN_VERDICTS, "logic_checks": checks, "all_ok": all_ok})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.dry:
        res = run_dry()
        (HERE / "coupling_confirm_v4_result_DRY_INVALID.json").write_text(
            json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"\nDRY logic: all_ok={res['all_ok']}", flush=True)
        return 0 if res["all_ok"] else 1
    if a.calibrate:
        res = run_calibrate()
        return 0 if res["ok"] else 2
    res = _py(run_real(a.smoke))
    suffix = "_SMOKE_INVALID" if a.smoke else ""
    (HERE / f"coupling_confirm_v4_result{suffix}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    tag = "SMOKE_INVALID " if a.smoke else ""
    print(f"\n{tag}B2-COUPLING ATTEMPT-4 VERDICT: {res['verdict']}", flush=True)
    if res.get("dose_response"):
        dr = res["dose_response"]
        print(f"  dose (effect-size+sign, NO p-gate): slopes={dr['per_seed_slopes']} "
              f"above-bar +{dr['n_above_bar_positive']} / -{dr['n_above_bar_negative']} "
              f"below {dr['n_below_bar']} (bar {MIN_EFFECT_SLOPE}, span {dr['rank_span']})", flush=True)
        # R3: the certified level rides BESIDE the verdict, never behind it -- a bounded null is only
        # ever a claim about ESTIMATOR_MDE80, and reading it as a claim at the 0.0152 bar is the T8 error
        est = dr.get("estimator_admissibility")
        if est:
            print(f"  estimator MDE80={est['mde80']} ({est['mde80_multiple_of_bar']}x bar, "
                  f"span-scale bound {est['span_bound']}) pool={est['n_pool']} "
                  f"dropped={est['n_pool_dropped']} zero-atom={est['zero_slope_atom']} "
                  f"replicates={est['mde80_replicates']} | false-recovery "
                  f"{est['false_recovery_rate']} vs analytic max "
                  f"{est['false_recovery_analytic_max']} (REPORTED, NON-GATING)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
