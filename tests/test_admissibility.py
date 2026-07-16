"""Tests for styxx.admissibility -- two-sided instrument admissibility (sensitive AND specific).

The synthetic decoys below ARE the "the certifier is itself two-sided" fixtures:
  * a coin-flip instrument MUST void as insensitive,
  * an always-fire instrument MUST void as nonspecific,
  * a sign-flipped instrument (high discrim, wrong RANK direction) MUST void as insensitive,
  * an instrument certified WITHOUT a caller fire_threshold can NEVER be bare ADMISSIBLE
    (self-derived threshold -> tautological fire-rate -> specificity UNTESTED),
  * only the good instrument, under an explicit deployment threshold, MAY be admitted.
Near-boundary fixtures make the 0.70 floor and the alpha gate load-bearing.
All fixtures are deterministic (seeded RNG or exact rank constructions); no models, CPU-only.
"""
import copy
import json

import numpy as np
import pytest

from styxx import admissibility
from styxx.admissibility import (
    instrument_admissibility,
    slope_permutation_null,
    verify_admissibility_certificate,
)

# a smaller permutation budget keeps the suite fast; still deterministic per seed
KP = 400


def _two_pop(rng, pos_mu, pos_sd, null_mu, null_sd, n=40):
    """Build (scores, labels) with n positive (label 1) then n null (label 0)."""
    positive = rng.normal(pos_mu, pos_sd, n)
    null = rng.normal(null_mu, null_sd, n)
    scores = np.concatenate([positive, null])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return scores, labels


# ---------------------------------------------------------------------------
# 1. GOOD instrument -> ADMISSIBLE
# ---------------------------------------------------------------------------

def test_good_instrument_is_admissible():
    rng = np.random.default_rng(0)
    # destroyed/positive class scores LOW (~0.0); intact null scores HIGH (~1.0)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0)
    assert rep.admissibility_verdict == "ADMISSIBLE"
    assert rep.admissible is True
    assert rep.sensitive is True and rep.specific is True and rep.measurable is True
    assert rep.direction_ok is True
    assert rep.threshold_derived is False      # full ADMISSIBLE only under a caller threshold
    assert rep.discrim >= 0.70
    assert rep.sensitivity_p < 0.05
    assert rep.fire_rate <= 0.15


def test_derived_threshold_caps_verdict_at_sensitivity_only():
    """A GOOD instrument called WITHOUT a deployment fire_threshold: specificity is UNTESTED
    (the auto-derived threshold makes the fire-rate tautological), so the verdict caps at
    ADMISSIBLE_SENSITIVITY_ONLY -- never bare ADMISSIBLE."""
    rng = np.random.default_rng(0)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   k_perm=KP, seed=0)   # no fire_threshold
    assert rep.threshold_derived is True
    assert rep.specific is None                          # untested, not passed
    assert rep.sensitive is True
    assert rep.admissibility_verdict == "ADMISSIBLE_SENSITIVITY_ONLY"
    assert rep.admissible is False                       # full admissibility requires both sides
    assert "UNTESTED" in rep.summary()
    assert "fire_threshold" in rep.summary()


def test_nonspecific_decoy_without_threshold_is_never_admissible():
    """THE load-bearing FATAL regression: the nonspecific decoy, called WITHOUT fire_threshold,
    must NOT certify ADMISSIBLE. Before the fix, the self-derived threshold guaranteed
    fire_rate ~= 0.05 <= max_fire, so specificity was tautologically 'passed' and this exact
    fixture came back full-strength ADMISSIBLE."""
    rng = np.random.default_rng(3)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.3)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   k_perm=KP, seed=0)   # NO fire_threshold -> derived
    assert rep.admissibility_verdict != "ADMISSIBLE"
    assert rep.admissibility_verdict == "ADMISSIBLE_SENSITIVITY_ONLY"
    assert rep.admissible is False
    assert rep.specific is None
    assert rep.threshold_derived is True


def test_callable_input_mode_matches():
    """The `score` callable + positive/null rows path builds the same verdict as precomputed scores."""
    rng = np.random.default_rng(1)
    pos_rows = list(range(40))
    null_rows = list(range(40, 80))
    lut = {}
    for i in pos_rows:
        lut[i] = float(rng.normal(0.0, 0.1))
    for i in null_rows:
        lut[i] = float(rng.normal(1.0, 0.1))

    def score(rows):
        return np.array([lut[r] for r in rows], dtype=float)

    rep = instrument_admissibility(score=score, positive=pos_rows, null=null_rows,
                                   expect="lower_on_positive", fire_threshold=0.5, k_perm=KP, seed=0)
    assert rep.admissibility_verdict == "ADMISSIBLE"
    assert rep.n_positive == 40 and rep.n_null == 40


def test_higher_on_positive_detector_admissible():
    """A detector that fires HIGH on the target-present class, quiet on benign controls."""
    rng = np.random.default_rng(7)
    # target-present/positive scores HIGH (~1.0); benign null scores LOW (~0.0)
    scores, labels = _two_pop(rng, pos_mu=1.0, pos_sd=0.1, null_mu=0.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="higher_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0)
    assert rep.admissibility_verdict == "ADMISSIBLE"
    assert rep.direction_ok is True


# ---------------------------------------------------------------------------
# 2. DEAF / coin-flip instrument -> VOID_INSTRUMENT__insensitive
# ---------------------------------------------------------------------------

def test_coinflip_instrument_voids_insensitive():
    rng = np.random.default_rng(2)
    # positive ~ null: no separation
    scores, labels = _two_pop(rng, pos_mu=0.5, pos_sd=0.2, null_mu=0.5, null_sd=0.2)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   k_perm=KP, seed=0)
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__insensitive"
    assert rep.sensitive is False
    assert rep.admissible is False
    assert rep.discrim < 0.70


# ---------------------------------------------------------------------------
# 3. NONSPECIFIC instrument -> VOID_INSTRUMENT__nonspecific
# ---------------------------------------------------------------------------

def test_nonspecific_instrument_voids_nonspecific():
    rng = np.random.default_rng(3)
    # clear separation (so it IS sensitive) but a firing threshold the null crosses constantly
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.3)
    # fire when score < 1.2; null ~ N(1.0, 0.3) crosses that ~75% of the time
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=1.2, max_fire=0.15, k_perm=KP, seed=0)
    assert rep.sensitive is True          # the separation is real (precedence would else hide it)
    assert rep.specific is False
    assert rep.fire_rate > 0.15
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__nonspecific"
    assert rep.admissible is False


# ---------------------------------------------------------------------------
# 4. SIGN-FLIPPED instrument -> NOT admissible (direction_ok False -> insensitive)
#    THE load-bearing regression test: high discrim must NOT rescue a backwards instrument.
# ---------------------------------------------------------------------------

def test_sign_flipped_instrument_is_not_admissible():
    rng = np.random.default_rng(4)
    # expect lower_on_positive, but the positive class scores HIGHER -- inverted instrument
    scores, labels = _two_pop(rng, pos_mu=1.0, pos_sd=0.1, null_mu=0.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   k_perm=KP, seed=0)
    assert rep.discrim >= 0.70            # magnitude is high -- this is NOT a discriminability failure
    assert rep.direction_ok is False      # ...but the direction is wrong
    assert rep.sensitive is False
    assert rep.admissible is False
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__insensitive"
    assert any("inverted" in n.lower() or "wrong side" in n.lower() for n in rep.notes)


def test_rank_inverted_skewed_instrument_not_admissible():
    """Direction must be judged by RANK (AUROC), not the mean: two -100 outliers drag the positive
    MEAN below the null, but the positive class still ranks ABOVE it pair-wise (AUROC 0.95). A
    mean-based direction check passes this inverted instrument; the AUROC-based one must not."""
    pos = np.array([1.0] * 38 + [-100.0, -100.0])
    null = np.linspace(-0.01, 0.01, 40)
    scores = np.concatenate([pos, null])
    labels = np.array([1] * 40 + [0] * 40)
    # the trap: mean order says "positive lower" ...
    assert scores[labels == 1].mean() < scores[labels == 0].mean()
    # ... but rank order says "positive higher" -- expect=lower_on_positive is rank-VIOLATED
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=-0.05, k_perm=KP, seed=0)
    assert rep.discrim >= 0.70                 # not a discriminability failure
    assert rep.direction_ok is False           # rank-based check catches the inversion
    assert rep.sensitive is False
    assert rep.admissible is False
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__insensitive"


# ---------------------------------------------------------------------------
# 5. UNMEASURABLE -> VOID_INSTRUMENT__unmeasurable
# ---------------------------------------------------------------------------

def test_degenerate_scores_void_unmeasurable():
    scores = np.full(20, 0.5)
    labels = np.array([1] * 10 + [0] * 10)
    rep = instrument_admissibility(scores=scores, labels=labels, k_perm=KP, seed=0)
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__unmeasurable"
    assert rep.measurable is False
    assert rep.admissible is False


def test_too_few_units_void_unmeasurable():
    scores = np.array([0.1, 0.9, 0.8, 0.85])
    labels = np.array([1, 0, 0, 0])          # only one positive unit
    rep = instrument_admissibility(scores=scores, labels=labels, k_perm=KP, seed=0)
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__unmeasurable"
    assert rep.measurable is False
    assert rep.n_positive == 1


# ---------------------------------------------------------------------------
# 6. Certificate round-trip: recompute from points -> faithful; tamper -> unfaithful
# ---------------------------------------------------------------------------

def test_certificate_roundtrip_and_faithfulness(tmp_path):
    rng = np.random.default_rng(5)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0)
    assert rep.admissible is True

    # a real receipt file to hash into the certificate
    receipt = tmp_path / "run_result.json"
    receipt.write_text(json.dumps({"note": "synthetic admissibility run"}), encoding="utf-8")

    cert = rep.certificate(receipts=[str(receipt)])
    # structure the spec requires
    assert cert["what"] == "styxx two-sided instrument-admissibility certificate"
    assert len(cert["points"]) == len(scores)
    assert cert["points"][0].keys() == {"unit_index", "score", "label"}
    assert cert["reuses"]["discrim"] == "styxx.crossmind.discrim"
    assert cert["specificity"]["threshold_derived"] is False   # persisted in the cert
    assert all(len(h) == 64 for h in cert["receipts_sha256"].values())

    # verify: receipts intact AND every recomputed field matches the stored cert
    v = verify_admissibility_certificate(cert, root=".")
    assert v["ok"] is True
    assert v["checked"] == 1 and v["missing"] == [] and v["mismatches"] == []
    assert v["recomputed_admissible"] is True
    assert v["faithful"] is True
    assert v["field_diffs"] == []

    # tamper the POINTS (not the flag): degenerate scores -> recompute unmeasurable -> flag no longer holds
    tampered = copy.deepcopy(cert)
    for p in tampered["points"]:
        p["score"] = 0.5
    v2 = verify_admissibility_certificate(tampered, root=".")
    assert v2["recomputed_admissible"] is False
    assert v2["faithful"] is False       # stored admissible=True != recompute=False


def test_certificate_tampered_metric_only_is_unfaithful():
    """Fix-3 regression: doctoring a single stored METRIC while keeping the admissible bool
    consistent must still flip faithful=False -- the verifier compares every headline field,
    not just the final bool."""
    rng = np.random.default_rng(5)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0)
    cert = rep.certificate()
    tampered = copy.deepcopy(cert)
    tampered["sensitivity"]["discrim"] = 0.71    # still >= floor, so admissible stays consistent
    assert tampered["admissible"] is True        # the bool alone cannot catch this
    v = verify_admissibility_certificate(tampered, root=".")
    assert v["recomputed_admissible"] is True    # recompute agrees with the BOOL...
    assert v["faithful"] is False                # ...but the doctored metric is caught
    assert any(d["field"] == "discrim" for d in v["field_diffs"])


def test_derived_threshold_certificate_verifies_faithful():
    """A derived-threshold cert must recompute in derived mode (specificity stays UNTESTED) and
    come back faithful -- the verifier must not launder the descriptive threshold into a tested one."""
    rng = np.random.default_rng(9)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   k_perm=KP, seed=0)   # derived mode
    assert rep.admissibility_verdict == "ADMISSIBLE_SENSITIVITY_ONLY"
    cert = rep.certificate()
    assert cert["specificity"]["threshold_derived"] is True
    assert cert["specificity"]["specific"] is None
    v = verify_admissibility_certificate(cert, root=".")
    assert v["faithful"] is True
    assert v["field_diffs"] == []
    assert v["recomputed_admissible"] is False


def test_certificate_verification_detects_receipt_drift(tmp_path):
    """The receipt-integrity arm must fire when a recorded hash no longer matches the live file."""
    rng = np.random.default_rng(6)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0)
    receipt = tmp_path / "r.json"
    receipt.write_text("original", encoding="utf-8")
    cert = rep.certificate(receipts=[str(receipt)])
    receipt.write_text("EDITED", encoding="utf-8")     # drift
    v = verify_admissibility_certificate(cert, root=".")
    assert v["ok"] is False
    assert len(v["mismatches"]) == 1


def test_certificate_written_to_out_path(tmp_path):
    rng = np.random.default_rng(8)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    out = tmp_path / "cert.json"
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0, out_path=str(out))
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["admissibility_verdict"] == rep.admissibility_verdict
    assert loaded["admissible"] == rep.admissible


# ---------------------------------------------------------------------------
# 7. slope_permutation_null: dose-response, flatness, and within-unit grouping
# ---------------------------------------------------------------------------

def test_slope_permutation_positive_dose_response_is_significant():
    rng = np.random.default_rng(10)
    dose = np.tile([0.0, 1.0, 2.0, 3.0, 4.0], 4)
    stat = 0.5 * dose + rng.normal(0.0, 0.05, dose.shape[0])
    res = slope_permutation_null(stat, dose, seed=0, k_perm=1000)
    assert res["slope"] > 0.3
    assert res["p_value"] < 0.05


def test_slope_permutation_flat_is_not_significant():
    rng = np.random.default_rng(8)
    dose = np.tile([0.0, 1.0, 2.0, 3.0, 4.0], 4)
    stat = rng.normal(0.0, 1.0, dose.shape[0])     # no dose dependence
    res = slope_permutation_null(stat, dose, seed=0, k_perm=1000)
    assert res["p_value"] > 0.05


def test_slope_permutation_within_unit_respects_grouping():
    """A purely BETWEEN-unit dose confound is flagged by a GLOBAL null but correctly NOT by a
    within-unit null: unit A only ever sees low doses, unit B only high, and the huge unit offset
    (not any within-unit dose-response) drives the raw slope."""
    unit = np.array(["A", "A", "B", "B"])
    dose = np.array([0.0, 1.0, 2.0, 3.0])
    stat = np.array([0.0, 0.1, 100.0, 100.1])      # offset is between-unit, not dose-driven
    within = slope_permutation_null(stat, dose, unit=unit, seed=0, k_perm=1000)
    glob = slope_permutation_null(stat, dose, unit=None, seed=0, k_perm=1000)
    # global null sees the confound as "significant"; the within-unit null does not
    assert glob["p_value"] < 0.05
    assert within["p_value"] > glob["p_value"]
    assert within["p_value"] > 0.05

    # and a genuine WITHIN-unit dose-response IS caught by the within-unit null
    rng = np.random.default_rng(12)
    unit2 = np.repeat(["A", "B", "C"], 4)
    dose2 = np.tile([0.0, 1.0, 2.0, 3.0], 3)
    offset = np.repeat([0.0, 10.0, 20.0], 4)
    stat2 = offset + 0.5 * dose2 + rng.normal(0.0, 0.05, 12)
    res = slope_permutation_null(stat2, dose2, unit=unit2, seed=0, k_perm=1000)
    assert res["p_value"] < 0.05


# ---------------------------------------------------------------------------
# near-boundary fixtures: the 0.70 floor and the alpha gate must be load-bearing
# ---------------------------------------------------------------------------

def test_discrim_just_above_floor_is_sensitive():
    """Deterministic rank construction: pos = null + 10.5 over arange(40) gives EXACTLY
    AUROC 1165/1600 = 0.728125 -- just above the 0.70 floor. The floor comparison, not a
    10-sigma separation, is what admits it."""
    null = np.arange(40, dtype=float)
    pos = np.arange(40, dtype=float) + 10.5
    scores = np.concatenate([pos, null])
    labels = np.array([1] * 40 + [0] * 40)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="higher_on_positive",
                                   k_perm=1000, seed=0)
    assert 0.70 <= rep.discrim <= 0.75          # near the floor, above it
    assert rep.sensitivity_p < 0.05
    assert rep.sensitive is True
    # derived threshold -> specificity untested -> capped verdict (never bare ADMISSIBLE)
    assert rep.admissibility_verdict == "ADMISSIBLE_SENSITIVITY_ONLY"


def test_discrim_just_below_floor_is_insensitive():
    """Same construction with offset 8.5: EXACTLY AUROC 1104/1600 = 0.69 -- just below the 0.70
    floor. Significant separation, right direction, but the floor voids it."""
    null = np.arange(40, dtype=float)
    pos = np.arange(40, dtype=float) + 8.5
    scores = np.concatenate([pos, null])
    labels = np.array([1] * 40 + [0] * 40)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="higher_on_positive",
                                   k_perm=1000, seed=0)
    assert 0.65 <= rep.discrim < 0.70           # near the floor, below it
    assert rep.direction_ok is True
    assert rep.sensitive is False
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__insensitive"


def test_marginal_permutation_p_gates_sensitivity():
    """3v3 perfectly separated: discrim = 1.0 (far above the floor) but only 1 of the C(6,3)=20
    label splits reproduces the separation, so the two-sided permutation p sits ~0.10 and can
    never beat alpha=0.05 at this n. Alpha, not the floor, is what voids it."""
    scores = np.array([10.0, 11.0, 12.0, 0.0, 1.0, 2.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    rep = instrument_admissibility(scores=scores, labels=labels, expect="higher_on_positive",
                                   k_perm=2000, seed=0)
    assert rep.discrim == 1.0
    assert rep.direction_ok is True
    assert rep.sensitivity_p > 0.05             # the marginal case: magnitude perfect, p not
    assert rep.sensitive is False
    assert rep.admissibility_verdict == "VOID_INSTRUMENT__insensitive"


# ---------------------------------------------------------------------------
# report ergonomics + package export
# ---------------------------------------------------------------------------

def test_report_as_dict_and_summary():
    rng = np.random.default_rng(13)
    scores, labels = _two_pop(rng, pos_mu=0.0, pos_sd=0.1, null_mu=1.0, null_sd=0.1)
    rep = instrument_admissibility(scores=scores, labels=labels, expect="lower_on_positive",
                                   fire_threshold=0.5, k_perm=KP, seed=0)
    d = rep.as_dict()
    for key in ("discrim", "sensitivity_p", "direction_ok", "fire_rate", "fire_threshold",
                "threshold_derived", "min_detectable_effect", "n_positive", "n_null",
                "sensitive", "specific", "measurable", "admissible", "admissibility_verdict",
                "notes"):
        assert key in d
    assert isinstance(rep.summary(), str)
    assert "INSTRUMENT ADMISSIBILITY" in rep.summary()


def test_public_exports():
    import styxx
    assert styxx.instrument_admissibility is instrument_admissibility
    assert hasattr(styxx, "AdmissibilityReport")
    assert hasattr(styxx, "slope_permutation_null")
    assert hasattr(styxx, "verify_admissibility_certificate")
    for name in ("instrument_admissibility", "AdmissibilityReport",
                 "slope_permutation_null", "verify_admissibility_certificate"):
        assert name in styxx.__all__
