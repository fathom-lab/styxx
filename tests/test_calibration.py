"""Tests for styxx.calibration -- transfer-safe threshold calibration.

These fixtures encode the finding the retro-cert arc proved on a real honesty probe
(papers/read-neq-write/): a distribution-free conformal / tolerance guarantee holds under
exchangeability, and the one way it silently fails in practice is FIT-CONTAMINATION -- calibrating a
threshold on scores the probe was FIT on. The load-bearing regression is
`test_fit_contaminated_transfer_guard_flags`: the naive in-sample threshold overshoots its FPR on
held-out negatives AND the transfer guard refuses to certify it. `test_threeway_direction_*`
numerically reproduces the receipt's direction with synthetic stand-ins (no GPU, no real probe):
fit-disjoint conformal transfers, in-sample does not.

Deterministic, CPU-only, no models.
"""
import math

import numpy as np
import pytest

from styxx.calibration import (
    conformal_threshold,
    calibrate_transfer_safe,
    ConformalThreshold,
)
from styxx.calibration import _incomplete_beta_int  # scipy-free regularized incomplete beta


# ---------------------------------------------------------------------------
# 1. exchangeable world: the guarantee holds on held-out negatives
# ---------------------------------------------------------------------------

def test_exchangeable_world_hits_target_fpr():
    """Calib negatives and held-out (deployment) negatives from the SAME law -> the conformal
    threshold's realized FPR on the held-out negatives sits at/below target within tolerance, and
    with an explicit fit-disjoint assertion the guarantee is CERTIFIED (transfer_valid True)."""
    rng = np.random.default_rng(0)
    calib_neg = rng.normal(0.0, 1.0, 2000)
    holdout_neg = rng.normal(0.0, 1.0, 4000)      # exchangeable with calib
    ct = conformal_threshold(calib_neg, target_fpr=0.10, direction="higher_fires",
                             fit_disjoint=True)
    assert isinstance(ct, ConformalThreshold)
    assert ct.transfer_valid is True
    assert not ct.never_fires
    realized = ct.realized_fpr(holdout_neg)
    # marginal conformal guarantee: held-out FPR concentrates at/just below alpha
    assert realized <= 0.10 + 0.02
    assert realized >= 0.10 - 0.04            # a real operating point, not a never-fire
    assert "split-conformal" in ct.guarantee


def test_unasserted_premise_is_conditional_not_refused():
    """Default fit_disjoint=None: the threshold is computed and the guarantee is stated
    CONDITIONALLY (transfer_valid None) -- an honest, non-refusing default. This is the zero-surprise
    behavior that keeps conformal_threshold a plain primitive."""
    rng = np.random.default_rng(1)
    calib_neg = rng.normal(0.0, 1.0, 500)
    ct = conformal_threshold(calib_neg, target_fpr=0.05)
    assert ct.transfer_valid is None
    assert "UNASSERTED" in ct.transfer_note
    assert "three-way" in ct.transfer_note.lower()
    assert not ct.guarantee.startswith("VOID")


# ---------------------------------------------------------------------------
# 2. THE load-bearing regression: fit-contamination is caught + overshoots
# ---------------------------------------------------------------------------

def test_fit_contaminated_transfer_guard_flags():
    """The finding, encoded. In-sample fit negatives are optimistically pushed to the quiet side; a
    threshold calibrated on them overshoots its FPR badly on the held-out negative population. The
    transfer guard, handed those same fit scores, DETECTS the overlap and refuses to certify
    (transfer_valid=False) with a message pointing at the three-way protocol."""
    rng = np.random.default_rng(2)
    # a probe FIT on its split: in-sample negatives sit UNUSUALLY low (fit optimism)
    fit_neg = rng.normal(-1.0, 0.5, 400)
    # the true, held-out negative population sits higher (its honest distribution)
    holdout_neg = rng.normal(0.0, 1.0, 2000)

    # naive/in-sample: calibrate the conformal threshold on the FIT scores themselves
    ct_naive = conformal_threshold(fit_neg, target_fpr=0.10, direction="higher_fires")
    overshoot = ct_naive.realized_fpr(holdout_neg)
    assert overshoot > 0.10 * 2          # blows well past target on held-out negatives

    # the guard: pass the probe's own fit scores as the fit set -> overlap == 100% -> REFUSE
    ct_guard = calibrate_transfer_safe(fit_neg, target_fpr=0.10, direction="higher_fires",
                                       fit_scores=fit_neg)
    assert ct_guard.transfer_valid is False
    assert ct_guard.guarantee.startswith("VOID")
    assert "three-way" in ct_guard.transfer_note.lower()
    assert "fit set" in ct_guard.transfer_note.lower()


def test_partial_overlap_still_refuses():
    """Even a PARTIAL overlap (some calib scores are literal fit values) breaks exchangeability and
    must refuse -- the guard is not fooled by mostly-clean calibration data."""
    rng = np.random.default_rng(3)
    fit_scores = rng.normal(0.0, 1.0, 300)
    clean = rng.normal(0.0, 1.0, 90)
    calib_neg = np.concatenate([clean, fit_scores[:10]])   # 10 of 100 are in-sample
    ct = calibrate_transfer_safe(calib_neg, target_fpr=0.1, fit_scores=fit_scores)
    assert ct.transfer_valid is False
    assert "10/100" in ct.transfer_note


def test_no_evidence_transfer_safe_refuses():
    """The transfer-safe door won't certify a guarantee it can't keep: with NO disjointness evidence
    (no fit_scores, no split ids) it returns transfer_valid=False asking for evidence -- distinct
    from conformal_threshold(fit_disjoint=None), which leaves the premise unasserted."""
    rng = np.random.default_rng(4)
    calib_neg = rng.normal(0.0, 1.0, 200)
    ct = calibrate_transfer_safe(calib_neg, target_fpr=0.1)
    assert ct.transfer_valid is False
    assert "could not be verified" in ct.transfer_note


def test_split_id_provenance_check():
    """Provenance tags: equal fit/calib split ids == same split -> refuse; distinct ids == positive
    evidence of disjointness -> certify."""
    rng = np.random.default_rng(5)
    calib_neg = rng.normal(0.0, 1.0, 300)
    same = calibrate_transfer_safe(calib_neg, target_fpr=0.1,
                                   fit_split_id="calibA", calib_split_id="calibA")
    assert same.transfer_valid is False
    assert "split id" in same.transfer_note.lower()

    distinct = calibrate_transfer_safe(calib_neg, target_fpr=0.1,
                                       fit_split_id="fit", calib_split_id="thresh")
    assert distinct.transfer_valid is True
    assert distinct.guarantee.startswith("split-conformal")


def test_clean_fit_scores_certify():
    """A fit set that shares NO scores with the calibration negatives is positive evidence of
    disjointness -> the guard certifies."""
    rng = np.random.default_rng(6)
    fit_scores = rng.normal(2.0, 0.1, 300)        # disjoint region, no overlap
    calib_neg = rng.normal(0.0, 1.0, 200)
    ct = calibrate_transfer_safe(calib_neg, target_fpr=0.1, fit_scores=fit_scores)
    assert ct.transfer_valid is True
    assert "clean" in ct.transfer_note.lower()


# ---------------------------------------------------------------------------
# 3. beta-tolerance is strictly more conservative than split-conformal
# ---------------------------------------------------------------------------

def test_beta_more_conservative_than_split():
    """The (alpha, delta) tolerance bound picks a higher order-statistic rank than the split-conformal
    quantile, so its realized in-sample FPR is <= split's -- a strictly more conservative threshold."""
    rng = np.random.default_rng(7)
    calib_neg = rng.normal(0.0, 1.0, 400)
    split = conformal_threshold(calib_neg, target_fpr=0.10, correction="split")
    beta = conformal_threshold(calib_neg, target_fpr=0.10, correction="beta", delta=0.10)
    assert beta.rank >= split.rank
    assert beta.tau >= split.tau                         # higher-fires: higher tau == more conservative
    assert beta.realized_calib_fpr <= split.realized_calib_fpr
    assert beta.confidence is not None and beta.confidence >= 0.90


def test_beta_unattainable_at_tiny_n_never_fires():
    """When n is too small for the (alpha, delta) tolerance, the bound is unattainable -> tau=+inf,
    a never-fire threshold that is honest about the missing guarantee rather than a false one."""
    ct = conformal_threshold(np.array([-3.0, -2.0, -1.0]), target_fpr=0.20,
                             correction="beta", delta=0.10)
    assert ct.never_fires
    assert ct.rank is None
    assert ct.fires(np.array([0.0, 5.0, 100.0])).sum() == 0
    assert "UNATTAINABLE" in ct.guarantee


# ---------------------------------------------------------------------------
# 4. numerically reproduce the three-way receipt's DIRECTION
#    (fit-disjoint conformal transfers; in-sample does not) with synthetic stand-ins
# ---------------------------------------------------------------------------

def test_threeway_direction_fit_disjoint_transfers_insample_doesnt():
    """The receipt's mechanism, reproduced without a GPU or the real probe. A fitted probe's
    IN-SAMPLE negatives are optimistically low; a threshold conformal-calibrated on them fires far
    above target on the held-out EVAL negatives (in-sample does NOT transfer). The SAME conformal
    rule calibrated on a fit-DISJOINT split drawn from the deployment law transfers correctly
    (EVAL FPR ~ target) -- exactly the retro_certify_private13_threeway.py result direction."""
    rng = np.random.default_rng(8)
    alpha = 0.20
    # probe FIT on CALIB_FIT: in-sample negatives pushed to the quiet (low) side
    calib_fit_neg = rng.normal(-1.2, 0.5, 300)
    # a fit-DISJOINT calibration split, out-of-sample for the probe (deployment law)
    calib_thresh_neg = rng.normal(0.0, 1.0, 300)
    # held-out EVAL negatives (same deployment law)
    eval_neg = rng.normal(0.0, 1.0, 2000)

    tau_insample = conformal_threshold(calib_fit_neg, target_fpr=alpha, direction="higher_fires")
    tau_disjoint = conformal_threshold(calib_thresh_neg, target_fpr=alpha, direction="higher_fires")

    fpr_insample = tau_insample.realized_fpr(eval_neg)
    fpr_disjoint = tau_disjoint.realized_fpr(eval_neg)

    # in-sample calibration does NOT transfer: EVAL FPR blows well past the alpha=0.20 target
    assert fpr_insample > 0.40
    # fit-disjoint calibration DOES transfer: EVAL FPR lands near the target
    assert abs(fpr_disjoint - alpha) < 0.05
    # and the gap is large -- this is the whole finding
    assert fpr_insample - fpr_disjoint > 0.20


# ---------------------------------------------------------------------------
# 5. direction (detector vs battery) via the admissibility convention
# ---------------------------------------------------------------------------

def test_lower_fires_battery_direction():
    """A capability battery fires LOW (score < tau). The reflected conformal rule holds the same FPR
    on the target-absent population, mirroring admissibility's lower_on_positive convention."""
    rng = np.random.default_rng(9)
    calib_neg = rng.normal(0.0, 1.0, 2000)
    holdout_neg = rng.normal(0.0, 1.0, 4000)
    ct = conformal_threshold(calib_neg, target_fpr=0.10, direction="lower_fires", fit_disjoint=True)
    assert ct.direction == "lower_fires"
    assert not ct.higher_fires
    # fires when score < tau
    assert bool(ct.fires(np.array([ct.tau - 1.0]))[0]) is True
    assert bool(ct.fires(np.array([ct.tau + 1.0]))[0]) is False
    realized = ct.realized_fpr(holdout_neg)
    assert realized <= 0.10 + 0.02


def test_direction_aliases_match_admissibility_expect():
    """The admissibility expect= strings are accepted as direction aliases so a threshold reads the
    same way here and in styxx.admissibility."""
    rng = np.random.default_rng(10)
    neg = rng.normal(0.0, 1.0, 300)
    hi = conformal_threshold(neg, target_fpr=0.1, direction="higher_on_positive")
    lo = conformal_threshold(neg, target_fpr=0.1, direction="lower_on_positive")
    assert hi.direction == "higher_fires"
    assert lo.direction == "lower_fires"


# ---------------------------------------------------------------------------
# 6. the scipy-free incomplete beta, cross-checked to scipy AND numpy
# ---------------------------------------------------------------------------

def test_incomplete_beta_matches_numpy_binomial_sum():
    """I_alpha(a,b) via the exact binomial identity must match a direct numpy binomial-tail sum."""
    from math import comb
    for a, b, x in [(3, 7, 0.2), (18, 4, 0.2), (1, 40, 0.05), (10, 10, 0.5)]:
        n = a + b - 1
        js = np.arange(a, n + 1)
        ref = float(np.sum([comb(n, int(j)) * (x ** int(j)) * ((1.0 - x) ** (n - int(j))) for j in js]))
        assert math.isclose(_incomplete_beta_int(a, b, x), ref, abs_tol=1e-12)


def test_incomplete_beta_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for a, b, x in [(3, 7, 0.2), (18, 4, 0.2), (5, 15, 0.1)]:
        ref = float(scipy_stats.beta.cdf(x, a, b))
        assert math.isclose(_incomplete_beta_int(a, b, x), ref, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 7. ergonomics: summary / as_dict / fires / edges + public exports
# ---------------------------------------------------------------------------

def test_summary_and_as_dict_roundtrip():
    rng = np.random.default_rng(11)
    neg = rng.normal(0.0, 1.0, 300)
    ct = conformal_threshold(neg, target_fpr=0.1, correction="beta", delta=0.1, fit_disjoint=True)
    s = ct.summary()
    assert "CONFORMAL THRESHOLD" in s
    assert "beta" in s
    d = ct.as_dict()
    for key in ("tau", "target_fpr", "n_calib", "correction", "direction", "realized_calib_fpr",
                "rank", "delta", "confidence", "guarantee", "transfer_valid", "transfer_note"):
        assert key in d
    assert d["transfer_valid"] is True
    import json
    json.dumps(d)                          # stays serializable (inf -> string sentinel)


def test_never_fire_serializes_and_fires_nothing():
    ct = conformal_threshold(np.array([1.0, 2.0, 3.0]), target_fpr=0.01, correction="beta", delta=0.01)
    assert ct.never_fires
    assert ct.as_dict()["tau"] in ("inf", "-inf")
    assert ct.fires(np.array([0.0, 10.0, 1e9])).sum() == 0


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        conformal_threshold([0.0, 1.0], target_fpr=1.5)
    with pytest.raises(ValueError):
        conformal_threshold([0.0, 1.0], correction="bogus")
    with pytest.raises(ValueError):
        conformal_threshold([0.0, 1.0], direction="sideways")


def test_public_exports():
    import styxx
    assert styxx.conformal_threshold is conformal_threshold
    assert styxx.calibrate_transfer_safe is calibrate_transfer_safe
    assert hasattr(styxx, "ConformalThreshold")
    for name in ("conformal_threshold", "ConformalThreshold", "calibrate_transfer_safe"):
        assert name in styxx.__all__


# ---------------------------------------------------------------------------
# 8. mount wiring: the optional conformal path is additive and transfer-aware
# ---------------------------------------------------------------------------

def test_mount_certify_accepts_conformal_threshold():
    """ConscienceMount.certify_admissibility accepts a ConformalThreshold and uses its tau as the
    deployment fire_threshold -- a transfer-safe operating point in place of a hand-picked one."""
    from styxx import crossmind as cm
    from styxx import mount as mt

    d = cm._synthetic(0)
    axis = cm.fit_axis(d["val_ref"], d["lab"], name="truth", whiten=True)
    smap = cm.fit_state_map(d["val_tgt"], d["val_ref"], seed=0)
    m = mt.ConscienceMount([mt.mount_axis("truth", axis, state_map=smap, high_means="true")])
    m.calibrate("truth", d["hold_tgt"])
    h, lab = d["hold_tgt"], d["hlab"]
    lying, honest = h[lab == 1], h[lab == 0]

    # the mount's deployed score is the divergence margin (higher-fires). Build a conformal threshold
    # on the honest-episode margins (target-absent) and hand it to certify_admissibility.
    ax = m.axes["truth"]
    honest_margin = -ax.z(honest).reshape(-1)          # margin under claim +1 == -z
    ct = conformal_threshold(honest_margin, target_fpr=0.10, direction="higher_fires",
                             fit_disjoint=True)
    rep = m.certify_admissibility(lying, honest, axis="truth", conformal=ct, k_perm=300, seed=0)
    # the report rounds fire_threshold to 6 decimals
    assert rep.fire_threshold == pytest.approx(float(ct.tau), abs=1e-6)
    assert rep.threshold_derived is False
    # passing both is rejected
    with pytest.raises(ValueError):
        m.certify_admissibility(lying, honest, axis="truth", conformal=ct, fire_threshold=0.5)


def test_mount_calibrate_threshold_accepts_conformal():
    """calibrate_threshold(conformal=...) adopts the ConformalThreshold's tau (clamped >= 0), and
    the default path (conformal=None) is byte-for-byte the historical rank rule."""
    from styxx import crossmind as cm
    from styxx import mount as mt

    d = cm._synthetic(0)
    axis = cm.fit_axis(d["val_ref"], d["lab"], name="truth", whiten=True)
    smap = cm.fit_state_map(d["val_tgt"], d["val_ref"], seed=0)
    m = mt.ConscienceMount([mt.mount_axis("truth", axis, state_map=smap, high_means="true")])
    m.calibrate("truth", d["hold_tgt"])
    h, lab = d["hold_tgt"], d["hlab"]
    honest = h[lab == 0]

    ct = conformal_threshold(np.array([0.3, 0.4, 0.55, 0.9, 1.2, 0.7, 0.5, 0.6]),
                             target_fpr=0.2, direction="higher_fires", fit_disjoint=True)
    m.calibrate_threshold("truth", honest, [+1] * len(honest), conformal=ct)
    assert m.axes["truth"].tau == pytest.approx(max(0.0, float(ct.tau)), abs=1e-9)
