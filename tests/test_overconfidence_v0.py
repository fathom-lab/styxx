# -*- coding: utf-8 -*-
"""
Tests for styxx.guardrail.overconf_check and the v0 calibrated weights.

Covers:
  - public API surface (overconf_check + OverconfidenceVerdict)
  - feature-extraction determinism
  - calibrated weights match the published values
  - paired (calibrated, overconfident) examples produce expected verdicts
    on the LONG-RESPONSE manifold the corpus was trained on
  - edge cases (empty, unicode, special chars)
  - serialization roundtrips
  - calibration fingerprint shape
  - documented failure modes fire as documented (regression checks)

Run: pytest tests/test_overconfidence_v0.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------- API


def test_overconf_check_importable():
    from styxx.guardrail import overconf_check, OverconfidenceVerdict  # noqa: F401


def test_returns_verdict():
    from styxx.guardrail import overconf_check, OverconfidenceVerdict
    v = overconf_check(prompt="x", response="y")
    assert isinstance(v, OverconfidenceVerdict)
    for attr in ("overconf_risk", "shows_overconf", "threshold", "features",
                 "top_signals", "prompt", "response"):
        assert hasattr(v, attr)


def test_verdict_as_dict_roundtrip():
    from styxx.guardrail import overconf_check
    v = overconf_check(prompt="x", response="It is what it is.")
    d = v.as_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["overconf_risk"] == pytest.approx(v.overconf_risk)


# ----------------------------------------------- weights byte-stability


def test_weights_match_published_values():
    from styxx.guardrail.calibrated_weights_overconfidence_v0 import (
        FEATURE_NAMES, COEFS, INTERCEPT, SCALER_MEAN, SCALER_SCALE,
        DEFAULT_OVERCONFIDENCE_THRESHOLD, MEAN_CV_AUC, STD_CV_AUC,
    )
    assert FEATURE_NAMES == [
        "certainty_marker_density",
        "hedge_density",
        "epistemic_balance",
        "specific_number_density",
        "evidence_marker_density",
        "strong_assertion_ratio",
        "unhedged_claim_ratio",
        "mean_sentence_length",
        "log_word_count",
    ]
    assert len(COEFS) == 9
    # K=1 critical = mean_sentence_length, NEG coef (longer = calibrated)
    msl_idx = FEATURE_NAMES.index("mean_sentence_length")
    assert COEFS[msl_idx] < 0
    # log_word_count NEG (longer = calibrated)
    lwc_idx = FEATURE_NAMES.index("log_word_count")
    assert COEFS[lwc_idx] < 0
    # certainty_marker_density POS (cert markers → overconfident)
    cmd_idx = FEATURE_NAMES.index("certainty_marker_density")
    assert COEFS[cmd_idx] > 0
    # epistemic_balance POS (cert > hedge → overconfident)
    eb_idx = FEATURE_NAMES.index("epistemic_balance")
    assert COEFS[eb_idx] > 0
    # evidence_marker_density NEG (calibrated cites sources)
    em_idx = FEATURE_NAMES.index("evidence_marker_density")
    assert COEFS[em_idx] < 0
    assert DEFAULT_OVERCONFIDENCE_THRESHOLD == 0.5
    # AUC band — honest, modest
    assert 0.70 < MEAN_CV_AUC < 0.85
    # std-band sanity
    assert 0.0 < STD_CV_AUC < 0.15


def test_calibration_fingerprint_shape():
    from styxx.guardrail.calibrated_weights_overconfidence_v0 import (
        CALIBRATION_FINGERPRINT, CALIBRATION_NOTES,
    )
    assert CALIBRATION_FINGERPRINT["instrument"] == "overconfidence-v0"
    assert CALIBRATION_FINGERPRINT["critical_K"] == 1
    assert CALIBRATION_FINGERPRINT["critical_feature"] == "mean_sentence_length"
    assert CALIBRATION_FINGERPRINT["n_features"] == 9
    # Every documented failure mode is named in the notes
    assert "honest_AUC_disclosure" in CALIBRATION_NOTES
    assert "scope_warning" in CALIBRATION_NOTES
    assert "discipline" in CALIBRATION_NOTES
    # Scope warning: NOT a truth detector
    assert "TRUTH" in CALIBRATION_NOTES["scope_warning"].upper()


# ----------------------------------------------- canonical paired cases
# These use LONG responses (>40 words) to isolate the register signal
# from the K=1 length confound. Short overconfident text reliably trips
# the detector; calibrated short text reliably trips it too — that is
# the documented failure mode pinned below.


def test_long_overconfident_response_flagged():
    from styxx.guardrail import overconf_check
    response = (
        "The Roman Empire absolutely fell because of barbarian invasions, "
        "without question. There is no debate among historians about this; "
        "the truth is unmistakably clear. The military pressure from Germanic "
        "tribes was the irrefutable cause, and any other proposed factor is "
        "definitely secondary. The historical record is utterly unambiguous "
        "on this point and has been clearly settled for centuries."
    )
    v = overconf_check(prompt="What caused the fall of Rome?", response=response)
    assert v.shows_overconf is True, f"long overconfident not flagged: risk={v.overconf_risk:.3f}"


def test_long_calibrated_response_not_flagged():
    from styxx.guardrail import overconf_check
    response = (
        "Historians have debated this for centuries, and the consensus is "
        "that no single cause is sufficient. Pressure from migrating peoples "
        "likely combined with internal political instability, economic strain "
        "from currency debasement, and possibly climate shifts to gradually "
        "weaken Western Roman institutions over several centuries. According "
        "to recent work by Peter Heather and others, the relative weighting "
        "remains contested, and I'd suggest treating any single-cause "
        "narrative with skepticism. The picture is genuinely complex."
    )
    v = overconf_check(prompt="What caused the fall of Rome?", response=response)
    assert v.shows_overconf is False, f"long calibrated flagged: risk={v.overconf_risk:.3f}"


def test_pair_ordering_overconfident_higher():
    """Calibrated and overconfident on the same prompt — overconfident must score higher."""
    from styxx.guardrail import overconf_check
    prompt = "Will fusion power be commercially viable by 2050?"
    overconfident = (
        "Yes, fusion power will absolutely be commercially viable by 2050, "
        "without any question. Recent breakthroughs at NIF and the rapidly "
        "advancing tokamak designs guarantee this outcome. The grid will "
        "definitely be running on fusion within the next two decades, and "
        "this is utterly clear from the trajectory of current research."
    )
    calibrated = (
        "Probably not at scale by 2050, though it depends on what 'commercially "
        "viable' means. The recent NIF breakthrough achieved net energy gain "
        "for the first time, but engineering a power plant is several harder "
        "problems on top — tritium breeding, neutron damage, materials. "
        "I'd expect demonstration plants in the 2040s, but full grid "
        "deployment likely later. The technical hurdles remain substantial."
    )
    v_over = overconf_check(prompt=prompt, response=overconfident)
    v_cal = overconf_check(prompt=prompt, response=calibrated)
    assert v_over.overconf_risk > v_cal.overconf_risk, (
        f"ordering violated: over={v_over.overconf_risk:.3f}, cal={v_cal.overconf_risk:.3f}"
    )


# ------------------------------------------- feature-extraction determinism


def test_extract_features_returns_nine_floats():
    from styxx.guardrail.overconfidence_signals import extract_overconfidence_features
    f = extract_overconfidence_features("?", "Yes definitely. Without question.")
    assert isinstance(f, dict)
    assert len(f) == 9
    for k, v in f.items():
        assert isinstance(v, float)
        assert v == v  # not NaN
        # No infinite values
        import math
        assert math.isfinite(v)


def test_extract_features_deterministic():
    from styxx.guardrail.overconfidence_signals import extract_overconfidence_features
    s = "I think it's probably around 8 million, though I might be slightly off."
    a = extract_overconfidence_features("?", s)
    b = extract_overconfidence_features("?", s)
    for k in a:
        assert a[k] == b[k], f"non-deterministic feature: {k}"


def test_certainty_markers_increase_score():
    from styxx.guardrail import overconf_check
    base = "It works."
    loaded = "It absolutely definitely certainly clearly obviously works without question."
    v_base = overconf_check(prompt="?", response=base)
    v_loaded = overconf_check(prompt="?", response=loaded)
    assert v_loaded.features["certainty_marker_density"] > v_base.features["certainty_marker_density"]


def test_hedges_decrease_score_relative():
    """Adding hedges to the same content should lower overconf_risk."""
    from styxx.guardrail import overconf_check
    bare = (
        "The capital is Brasília, founded in 1960 to replace Rio de Janeiro. "
        "It was designed by Oscar Niemeyer and Lúcio Costa to be a modernist "
        "capital. The population today is about three million people."
    )
    hedged = (
        "I think the capital is Brasília — I believe it was founded around "
        "1960 to replace Rio de Janeiro, but I'd want to verify the exact "
        "year. It was designed, if I recall correctly, by Oscar Niemeyer and "
        "perhaps Lúcio Costa, though I might be misattributing. The population "
        "today is roughly three million, give or take."
    )
    v_bare = overconf_check(prompt="What is the capital of Brazil?", response=bare)
    v_hedged = overconf_check(prompt="What is the capital of Brazil?", response=hedged)
    assert v_hedged.overconf_risk < v_bare.overconf_risk


# ------------------------------------------- edge cases


def test_empty_response_no_crash():
    from styxx.guardrail import overconf_check
    v = overconf_check(prompt="?", response="")
    assert 0.0 <= v.overconf_risk <= 1.0


def test_single_char_response_no_crash():
    from styxx.guardrail import overconf_check
    v = overconf_check(prompt="?", response="a")
    assert 0.0 <= v.overconf_risk <= 1.0


def test_unicode_response_no_crash():
    from styxx.guardrail import overconf_check
    v = overconf_check(prompt="?", response="人口は約8百万人 — perhaps 🌍")
    assert 0.0 <= v.overconf_risk <= 1.0


# ------------------------------------------- documented failure modes


def test_documented_short_calibrated_can_be_misclassified():
    """K=1 = mean_sentence_length is a length confound. Short calibrated
    responses can score above threshold even with appropriate hedges.
    Pinned here so future improvements that fix this also update the
    docstring's failure-mode list."""
    from styxx.guardrail import overconf_check
    short_calibrated = "Maybe. Hard to say."
    v = overconf_check(prompt="Will AGI exist by 2030?", response=short_calibrated)
    # The short hedged response has high overconf_risk because mean_sentence_length
    # is small — exactly the failure mode documented in the weights module.
    # If a future iteration fixes this, update the failure-mode list and
    # change this assertion.
    assert v.overconf_risk > 0.5, (
        "If this now scores LOW for short hedged input, the K=1 length confound "
        "may have been fixed. Update the docstring failure-mode list."
    )


def test_documented_specific_number_coef_is_negative():
    """Design intuition was overconfident responses invent specific numbers.
    Empirically, calibrated responses cite numbers more (with attribution).
    The learned coefficient is small NEGATIVE — pinned here as the
    counter-intuitive empirical result."""
    from styxx.guardrail.calibrated_weights_overconfidence_v0 import (
        FEATURE_NAMES, COEFS,
    )
    snd_idx = FEATURE_NAMES.index("specific_number_density")
    assert COEFS[snd_idx] < 0, (
        "specific_number_density coefficient flipped to non-negative — "
        "update the docstring failure-mode list and notes."
    )


# ── scope_warning (2026-05-11 cognometric-inversion experiment) ──────


def test_ovc_scope_warning_fires_on_short_agent_report():
    """The 2026-05-11 cognometric-inversion experiment showed agent
    task-completion reports score overconf_risk 0.98+ but the score
    is structural (no high-density certainty markers, no false-
    precision numbers). Flag so F10 heal-pass logic can route around
    the FP class."""
    from styxx.guardrail import overconf_check
    v = overconf_check(
        prompt="status update",
        response=(
            "Token scrubbed from .git/config, upstream points at plain "
            "origin. F10 done. Branch live at "
            "github.com/fathom-lab/styxx/tree/claude/f10-self-healing-reflex-spec. "
            "The token only briefly persisted on this local machine."
        ),
    )
    assert v.shows_overconf is True
    assert v.scope_warning == "v0_lexical_oof_short_response"


def test_ovc_scope_warning_silent_on_genuine_overclaim():
    """The discriminator: genuine overclaim has high
    certainty_marker_density and/or specific_number_density. Those
    floors gate the warning out so true positives are not
    suppressed."""
    from styxx.guardrail import overconf_check
    v = overconf_check(
        prompt="Will fusion be viable?",
        response=(
            "I am absolutely certain that fusion power will be "
            "commercially deployed by 2027. Without question. This is "
            "settled physics. Any skeptic is ignoring the obvious."
        ),
    )
    assert v.shows_overconf is True
    assert v.scope_warning is None  # real TP — don't suppress
