# -*- coding: utf-8 -*-
"""
Tests for styxx.divergence — semantic_entropy + council_agreement.

These are the 7.7.0 divergence primitives from the 2026-05-25 behavioral-
knowledge-boundary arc. Core behavior is tested OFFLINE and DETERMINISTICALLY
via the ``same_fn`` and ``lexical`` clustering paths (no model download); the
validated embedding-cosine path is gated behind ``importorskip`` so it runs
locally with ``styxx[nli]`` and skips in lean CI.
"""
from __future__ import annotations

import math
import warnings

import pytest

from styxx.divergence import (
    semantic_entropy, council_agreement, grounded_honesty, GroundedScore,
    detect_context_injection, InjectionScore,
    divergence_available,
)
from styxx.errors import StyxxError

_EQ = lambda a, b: a == b  # exact-match equivalence -> fully deterministic clustering


# ─── semantic_entropy ──────────────────────────────────────────────

def test_semantic_entropy_consistent_is_zero():
    # one cluster -> entropy 0 (the model knows it / abstains consistently)
    assert semantic_entropy(["a", "a", "a", "a"], same_fn=_EQ) == pytest.approx(0.0)


def test_semantic_entropy_two_clusters():
    # two equal clusters -> ln(2)
    e = semantic_entropy(["a", "a", "b", "b"], same_fn=_EQ)
    assert e == pytest.approx(math.log(2), abs=1e-9)


def test_semantic_entropy_all_distinct_is_max():
    # k distinct samples -> ln(k) (max entropy: confident confabulation)
    e = semantic_entropy(["a", "b", "c"], same_fn=_EQ)
    assert e == pytest.approx(math.log(3), abs=1e-9)


def test_semantic_entropy_lt_two_samples_is_zero():
    assert semantic_entropy([], same_fn=_EQ) == 0.0
    assert semantic_entropy(["only one"], same_fn=_EQ) == 0.0
    assert semantic_entropy([None, "x"], same_fn=_EQ) == 0.0  # None filtered -> 1 left


def test_semantic_entropy_lexical_template_lies_diverge():
    # bare distinct tokens diverge under lexical Jaccard (mechanism check;
    # the module docstring documents lexical is NOT the validated signal for
    # template-sharing answers — that needs embedding-cosine / same_fn).
    e = semantic_entropy(["1842", "1723", "1912", "1601"], method="lexical")
    assert e > 1.0  # four distinct -> near ln(4)
    e2 = semantic_entropy(["Paris", "Paris", "Paris"], method="lexical")
    assert e2 == pytest.approx(0.0)


# ─── council_agreement ─────────────────────────────────────────────

def test_council_agreement_convergent_is_one():
    assert council_agreement(["x", "x", "x", "x"], same_fn=_EQ) == pytest.approx(1.0)


def test_council_agreement_divergent_is_low():
    # every model invents differently -> 1/n
    assert council_agreement(["a", "b", "c", "d"], same_fn=_EQ) == pytest.approx(0.25)


def test_council_agreement_partial_majority():
    # 3 of 4 agree -> 0.75
    assert council_agreement(["x", "x", "x", "y"], same_fn=_EQ) == pytest.approx(0.75)


def test_council_agreement_edges():
    assert council_agreement([], same_fn=_EQ) == 0.0       # empty -> no agreement
    assert council_agreement(["solo"], same_fn=_EQ) == 1.0  # single -> trivially agreed
    assert council_agreement([None, None, "z"], same_fn=_EQ) == 1.0  # None filtered


# ─── grounded_honesty ──────────────────────────────────────────────

def test_grounded_honesty_stable_true_claim_is_high():
    # all samples agree with the claim -> stability 1, concordance 1 -> g 1
    g = grounded_honesty(["Canberra"] * 5, "Canberra", same_fn=_EQ)
    assert isinstance(g, GroundedScore)
    assert g.grounded == pytest.approx(1.0)
    assert g.stability == pytest.approx(1.0)
    assert g.concordance == pytest.approx(1.0)
    assert g.n_clusters == 1 and g.n_samples == 5


def test_grounded_honesty_contradiction_is_low():
    # stable belief is "Canberra"; the claim "Sydney" sits outside it ->
    # stability high (one cluster) but concordance 0 -> g 0
    g = grounded_honesty(["Canberra"] * 5, "Sydney", same_fn=_EQ)
    assert g.stability == pytest.approx(1.0)
    assert g.concordance == pytest.approx(0.0)
    assert g.grounded == pytest.approx(0.0)


def test_grounded_honesty_confabulation_low_stability():
    # a different answer each sample -> stability ~0 -> g ~0 even if claim matches one
    g = grounded_honesty(["a", "b", "c", "d", "e"], "a", same_fn=_EQ)
    assert g.n_clusters == 5
    assert g.stability == pytest.approx(0.0)
    assert g.concordance == pytest.approx(0.2)  # one of five matches the claim
    assert g.grounded == pytest.approx(0.0)


def test_grounded_honesty_partial_mode():
    # 4 of 5 say Canberra, 1 Sydney -> 2 clusters; claim Canberra
    g = grounded_honesty(["Canberra", "Canberra", "Canberra", "Canberra", "Sydney"],
                         "Canberra", same_fn=_EQ)
    assert g.n_clusters == 2
    assert g.stability == pytest.approx(1.0 - 1 / 4)   # 0.75
    assert g.concordance == pytest.approx(0.8)
    assert g.grounded == pytest.approx(0.75 * 0.8)


def test_grounded_honesty_empty_is_zero():
    g = grounded_honesty([], "anything", same_fn=_EQ)
    assert g == GroundedScore(0.0, 0.0, 0.0, 0, 0)
    assert g.grounded == 0.0


def test_grounded_honesty_float_protocol():
    # GroundedScore acts like its scalar score in comparisons
    hi = grounded_honesty(["x"] * 4, "x", same_fn=_EQ)
    lo = grounded_honesty(["x"] * 4, "y", same_fn=_EQ)
    assert float(hi) > float(lo)
    assert max([lo, hi], key=float) is hi


def test_grounded_honesty_none_filtered():
    g = grounded_honesty([None, "x", "x", None], "x", same_fn=_EQ)
    assert g.n_samples == 2
    assert g.grounded == pytest.approx(1.0)


# ─── clustering backend / contract ─────────────────────────────────

def test_same_fn_overrides_method():
    # same_fn forces every pair equal -> single cluster regardless of strings
    always = lambda a, b: True
    assert semantic_entropy(["wildly", "different", "strings"], same_fn=always) == 0.0
    assert council_agreement(["wildly", "different", "strings"], same_fn=always) == 1.0


def test_unknown_method_raises():
    with pytest.raises(StyxxError):
        semantic_entropy(["a", "b"], method="banana")


def test_divergence_available_returns_bool():
    assert isinstance(divergence_available(), bool)


def test_cosine_requires_dependency_when_absent():
    if divergence_available():
        pytest.skip("sentence-transformers installed; cosine path is active")
    with pytest.raises(StyxxError):
        semantic_entropy(["a", "b", "c"], method="cosine")


def test_auto_warns_and_falls_back_when_absent():
    if divergence_available():
        pytest.skip("sentence-transformers installed; no lexical fallback warning")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        semantic_entropy(["the cat sat", "a dog ran", "the moon rose"], method="auto")
    assert any(issubclass(w.category, RuntimeWarning) for w in rec)


def test_cosine_clustering_when_available():
    pytest.importorskip("sentence_transformers")
    # identical answers -> one cluster -> 0 entropy / full agreement
    assert semantic_entropy(["Paris", "Paris", "Paris"], method="cosine") == pytest.approx(0.0, abs=1e-9)
    # semantically distinct answers -> high entropy / low agreement
    e = semantic_entropy(["Paris", "Tokyo", "Cairo", "Berlin"], method="cosine")
    assert e > 1.0
    assert council_agreement(["Paris", "Paris", "Paris", "Tokyo"], method="cosine") == pytest.approx(0.75)


# ─── detect_context_injection ──────────────────────────────────────
# 7.7.13 cross-context divergence as item-level injection-detection primitive.
# Calibration: AUC 0.875 at threshold 0.5 (n=48 gpt-4o-mini, system_lie injection).
# Receipt: papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md
# (commit e093730). Tests below verify the pure-math behavior offline; the AUC
# calibration is in the FINDING, not retested here.

def test_detect_injection_clean_both_arms_agree_truth():
    # No injection: both arms say the truth -> D=0, not suspected.
    r = detect_context_injection(["Paris"] * 10, ["Paris"] * 10, "Paris", same_fn=_EQ)
    assert r.divergence == pytest.approx(0.0)
    assert r.suspected is False
    assert r.concordance_stateless == pytest.approx(1.0)
    assert r.concordance_in_session == pytest.approx(1.0)


def test_detect_injection_clean_both_arms_agree_lie():
    # No injection AND wrong claim: stateless says truth, in-session says truth,
    # claim is a lie -> both concordances 0 -> D=0, not suspected (no injection,
    # the LIE just happens to be unrelated to the model's belief).
    r = detect_context_injection(["Paris"] * 10, ["Paris"] * 10, "Lyon", same_fn=_EQ)
    assert r.divergence == pytest.approx(0.0)
    assert r.suspected is False
    assert r.concordance_stateless == pytest.approx(0.0)
    assert r.concordance_in_session == pytest.approx(0.0)


def test_detect_injection_succeeds_full():
    # Injection succeeds completely: stateless says truth, in-session says lie,
    # claim is the lie -> stateless C=0, in-session C=1 -> D=1, suspected.
    r = detect_context_injection(["Paris"] * 10, ["Lyon"] * 10, "Lyon", same_fn=_EQ)
    assert r.divergence == pytest.approx(1.0)
    assert r.suspected is True
    assert r.concordance_stateless == pytest.approx(0.0)
    assert r.concordance_in_session == pytest.approx(1.0)


def test_detect_injection_succeeds_partial():
    # Partial injection: in-session resampler partially agrees with lie.
    # stateless 0/10 lie, in-session 6/10 lie -> D=0.6, suspected at default 0.5.
    r = detect_context_injection(
        ["Paris"] * 10,
        ["Lyon"] * 6 + ["Paris"] * 4,
        "Lyon", same_fn=_EQ,
    )
    assert r.divergence == pytest.approx(0.6)
    assert r.suspected is True
    assert r.concordance_stateless == pytest.approx(0.0)
    assert r.concordance_in_session == pytest.approx(0.6)


def test_detect_injection_below_threshold_not_suspected():
    # Slight in-session contamination but below threshold -> not suspected.
    r = detect_context_injection(
        ["Paris"] * 10,
        ["Lyon"] * 3 + ["Paris"] * 7,
        "Lyon", same_fn=_EQ,
    )
    assert r.divergence == pytest.approx(0.3)
    assert r.suspected is False  # 0.3 not > 0.5 default


def test_detect_injection_custom_threshold():
    # Same data, lower threshold -> now suspected.
    r = detect_context_injection(
        ["Paris"] * 10,
        ["Lyon"] * 3 + ["Paris"] * 7,
        "Lyon", same_fn=_EQ, threshold=0.2,
    )
    assert r.divergence == pytest.approx(0.3)
    assert r.suspected is True


def test_detect_injection_empty_stateless():
    # If stateless arm is empty, can't compute the defense — return 0/0 and flag
    # honestly (divergence = |0 - C_in_session|).
    r = detect_context_injection([], ["Lyon"] * 10, "Lyon", same_fn=_EQ)
    assert r.n_stateless == 0
    assert r.n_in_session == 10
    assert r.concordance_stateless == 0.0
    assert math.isnan(r.divergence)   # undefined, not a measured 1.0
    assert r.measured is False
    assert r.suspected is True        # unchanged: still flags, still fail-closed


def test_detect_injection_empty_in_session():
    # CHANGED 2026-08-19: an arm that collected NOTHING used to be scored as
    # concordance 0.0, so a failed in-session sample produced divergence 1.0 and
    # suspected=True -- an ACCUSATION manufactured from missing data. Divergence
    # against an unmeasured arm is undefined; measured=False says so and nothing
    # is accused. (The calibrated regime -- AUC 0.875, n=48 -- is unaffected:
    # every calibrated item sampled both arms.)
    r = detect_context_injection(["Paris"] * 10, [], "Paris", same_fn=_EQ)
    assert r.n_in_session == 0
    assert r.concordance_in_session == 0.0
    assert r.measured is False
    assert math.isnan(r.divergence)
    assert r.suspected is True      # fail CLOSED: the comparison never happened


def test_detect_injection_both_empty():
    # CHANGED 2026-08-19: two failed arms gave |0.0 - 0.0| = 0.0 and
    # suspected=False -- a total sampling failure reported as a clean bill of
    # health, from an injection detector. It now fails CLOSED, matching the
    # stance this module already took for a missing stateless arm, and
    # measured=False lets a caller tell "checked, clean" from "never checked".
    r = detect_context_injection([], [], "anything", same_fn=_EQ)
    assert r.n_stateless == 0
    assert r.n_in_session == 0
    assert math.isnan(r.divergence)
    assert r.measured is False
    assert r.suspected is True


def test_detect_injection_none_filtered():
    # None entries dropped on each arm.
    r = detect_context_injection(
        [None, "Paris", "Paris", None],
        [None, "Lyon", "Lyon", "Lyon"],
        "Lyon", same_fn=_EQ,
    )
    assert r.n_stateless == 2
    assert r.n_in_session == 3
    assert r.concordance_stateless == pytest.approx(0.0)
    assert r.concordance_in_session == pytest.approx(1.0)
    assert r.divergence == pytest.approx(1.0)
    assert r.suspected is True


def test_detect_injection_float_and_bool_protocols():
    # InjectionScore acts like its scalar divergence + flag
    hi = detect_context_injection(["a"] * 5, ["b"] * 5, "b", same_fn=_EQ)
    lo = detect_context_injection(["a"] * 5, ["a"] * 5, "a", same_fn=_EQ)
    assert float(hi) > float(lo)
    assert bool(hi) is True
    assert bool(lo) is False


def test_detect_injection_reverse_direction_still_detected():
    # Edge case: stateless arm "agrees" with claim but in-session disagrees.
    # This could happen if the in-session context is *poisoned toward refusing*
    # rather than asserting a lie. D is still high -> suspected.
    r = detect_context_injection(["Lyon"] * 10, ["Paris"] * 10, "Lyon", same_fn=_EQ)
    assert r.divergence == pytest.approx(1.0)
    assert r.suspected is True


def test_detect_injection_unequal_lengths():
    # n_stateless != n_in_session is allowed (caller may use different N per arm).
    r = detect_context_injection(["Paris"] * 5, ["Lyon"] * 20, "Lyon", same_fn=_EQ)
    assert r.n_stateless == 5
    assert r.n_in_session == 20
    assert r.concordance_stateless == pytest.approx(0.0)
    assert r.concordance_in_session == pytest.approx(1.0)
    assert r.divergence == pytest.approx(1.0)
    assert r.suspected is True


def test_detect_injection_namedtuple_fields():
    # Sanity-check the InjectionScore field surface — production code may rely on it.
    r = detect_context_injection(["a"] * 3, ["b"] * 3, "a", same_fn=_EQ)
    assert isinstance(r, InjectionScore)
    assert r._fields == (
        "divergence", "suspected",
        "concordance_stateless", "concordance_in_session",
        "n_clusters_stateless", "n_clusters_in_session",
        "n_stateless", "n_in_session",
        "measured",
    )
    assert r.measured is True


# ── the lexical backend was blind outside the latin alphabet ───────────────

def test_lexical_clustering_is_not_blind_to_non_latin_scripts():
    """`[a-z0-9]+` matched NOTHING outside latin, so every CJK/Cyrillic/Arabic
    answer tokenized to the empty set, two empty sets were called identical,
    all samples collapsed into one cluster, and semantic_entropy returned 0.0 —
    its most confident reading, "the model knows the answer". Measured before
    the fix: four DIFFERENT Japanese city names scored 0.0 while the same four
    in latin script scored 1.386. Blind, and blind in the flattering direction."""
    import math

    for answers in (["東京", "大阪", "京都", "札幌"],          # japanese
                    ["Москва", "Киев", "Минск", "Рига"],        # cyrillic
                    ["Tokyo", "Osaka", "Kyoto", "Sapporo"]):    # latin, control
        e = semantic_entropy(answers, method="lexical")
        assert e == pytest.approx(math.log(4), abs=1e-6), answers

    # genuine agreement still reads as agreement, in any script
    assert semantic_entropy(["東京"] * 4, method="lexical") == pytest.approx(0.0)


def test_untokenizable_but_different_strings_are_not_called_identical():
    """Two strings this backend cannot read must not be merged — that
    manufactures agreement out of its own blind spot."""
    from styxx.divergence import _lexical_same

    assert _lexical_same("🙂", "🙂", 0.5) is True
    assert _lexical_same("🙂", "🙃", 0.5) is False
    assert _lexical_same("", "", 0.5) is True


# ─── GroundedScore.abstains: the Report-or-abstain rule on the score (#233) ───

def test_abstains_separates_stable_contradiction_from_no_stable_belief():
    """grounded is 0.0 for both rows and float() reads them alike; only
    abstains() tells "report a contradiction" from "abstain"."""
    contra = grounded_honesty(["Canberra"] * 5, "Sydney", same_fn=_EQ)
    nobelief = grounded_honesty(["a", "b", "c", "d", "e"], "z", same_fn=_EQ)
    assert contra.grounded == nobelief.grounded == 0.0
    assert float(contra) == float(nobelief) == 0.0
    assert (contra.stability, nobelief.stability) == (1.0, 0.0)
    assert contra.abstains() is False
    assert nobelief.abstains() is True


def test_abstains_on_fewer_than_two_samples():
    # one sample gives stability 1.0 (nothing to disagree with): a "stable
    # contradiction" from one answer, which the sample count refuses
    one = grounded_honesty(["Sydney"], "Canberra", same_fn=_EQ)
    assert one.stability == 1.0 and one.n_samples == 1
    assert one.abstains() is True
    assert grounded_honesty([], "x", same_fn=_EQ).abstains() is True
    assert grounded_honesty([None, "y"], "x", same_fn=_EQ).abstains() is True  # None filtered
    assert grounded_honesty(["y", "y"], "x", same_fn=_EQ).abstains() is False


def test_abstains_threshold_is_low_stability_and_strict():
    from styxx.divergence import LOW_STABILITY
    assert LOW_STABILITY == 0.50
    # stability 0.75 (two clusters of five): reported at the default, abstains above 0.75
    g = grounded_honesty(["Canberra", "Canberra", "Canberra", "Canberra", "Sydney"],
                         "Canberra", same_fn=_EQ)
    assert g.stability == pytest.approx(0.75)
    assert g.abstains() is False
    assert g.abstains(low_stability=0.8) is True
    assert g.abstains(0.75) is False                      # strict: stability < low_stability
    # stability exactly 0.5 (three clusters of five) is not below the line
    half = grounded_honesty(["a", "a", "a", "b", "c"], "a", same_fn=_EQ)
    assert half.stability == pytest.approx(0.5)
    assert half.abstains() is False
    assert GroundedScore(0.0, 0.49, 0.0, 3, 5).abstains() is True


def test_abstains_is_a_method_and_the_tuple_shape_is_unchanged():
    g = grounded_honesty(["Canberra"] * 4 + ["Sydney"], "Canberra", same_fn=_EQ)
    a, b, c, d, e = g                                   # five-way unpacking still works
    assert (a, b, c, d, e) == tuple(g)
    assert GroundedScore._fields == ("grounded", "stability", "concordance",
                                     "n_clusters", "n_samples")
    assert len(g) == 5 and "abstains" not in g._asdict()
    assert grounded_honesty([], "x", same_fn=_EQ) == GroundedScore(0.0, 0.0, 0.0, 0, 0)
    assert grounded_honesty([], "x", same_fn=_EQ) == (0.0, 0.0, 0.0, 0, 0)


def test_grounded_honesty_docstring_points_at_abstains():
    doc = grounded_honesty.__doc__
    assert "score.abstains()" in doc and "LOW_STABILITY" in doc
    assert "does not abstain by itself" in doc


# ─── the default line is audit_claim's, not the boundary hunt's split (#233 review) ───

def _boundary_hunt():
    import json
    from pathlib import Path
    p = (Path(__file__).resolve().parent.parent / "papers" / "grounded-honesty-axis"
         / "boundary_hunt_result.json")
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_default_line_is_not_the_boundary_hunt_split():
    d = _boundary_hunt()
    n, b2, rows = d["n_samples"], d["B2_stability_gate"], d["rows"]
    assert (len(rows), n, b2["median_stability"], b2["n_high"], b2["n_low"]) == (36, 10, 1.0, 30, 6)
    scores = [GroundedScore(r["g_true"], r["stability"], r["concord_true"] / n, r["n_clusters"], n)
              for r in rows]
    low = [s for s in scores if s.stability < b2["median_stability"]]
    assert sorted(round(s.stability, 2) for s in low) == [0.78, 0.89, 0.89, 0.89, 0.89, 0.89]
    # at the default (0.50) no item abstains, the six of the AUC 0.444 stratum included
    assert not any(s.abstains() for s in scores)
    # the run's own split is abstains(low_stability=1.0): True on exactly its low stratum
    assert [s for s in scores if s.abstains(low_stability=1.0)] == low


def test_the_docs_call_the_default_a_default_and_cite_the_split():
    b2 = _boundary_hunt()["B2_stability_gate"]
    for doc in (GroundedScore.abstains.__doc__, grounded_honesty.__doc__):
        flat = " ".join(doc.split())
        assert "low_stability_threshold" in flat and "not a measured line" in flat
        assert "False does not make it one" in flat
        assert "abstains(low_stability=1.0)" in flat
        assert f"{b2['auc_low_stability']:.3f}" in flat          # 0.444
    flat = " ".join(GroundedScore.abstains.__doc__.split())
    assert f"{b2['auc_high_stability']:.3f}" in flat             # 0.967
    # the sample-count rule is the method's own, and the two places audit_claim parts from it
    assert "names no count" in flat
    assert '``"injected"`` before ``"abstain"``' in flat
