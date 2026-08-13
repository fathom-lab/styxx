"""The grounding rate must be reported against its chance floor.

Provenance: found 2026-08-13 by dogfooding `audit_grounding` on the author's own C6 prereg.
The tool reported "86.8% grounded" — true, and nearly meaningless: `_match` accepts any source
value within 0.5*10^-decimals, so a one-decimal claim grounds against a receipt of a few hundred
leaves ESSENTIALLY ALWAYS. A rate quoted without its floor is the fire-rate wearing the
antibody's name (cf. meta_audit_v1).

Two flattering implementations were caught before the shipped one, both by cross-checking the
in-module number against an independent standalone measurement:
  v1: min/max over all leaves -> a seed value (90210) stretched the sample range; floor 0.000
  v2: <=1000 filter + p95      -> range still 342; floor 0.0035
Correct: sample the magnitude class the claims occupy. These tests pin that down.
"""
from __future__ import annotations

from styxx.claim_audit import audit_grounding


def _receipt():
    """A receipt shaped like a real one: many rates in [0,1] plus non-statistic scalars."""
    return {
        "seed": 90210, "n_perm": 300, "n_rep": 400, "n_t": 300,
        "cells": {f"c{i}": {"rate": round(0.01 * i, 3), "cp": round(0.002 * i, 4)}
                  for i in range(1, 60)},
    }


def _dense_receipt():
    """Dense enough that the 1-decimal grid is fully covered — the real-prereg regime."""
    return {"seed": 90210,
            "cells": {f"c{i}": {"rate": round(0.005 * i, 3)} for i in range(0, 201)}}


def test_one_decimal_claims_are_flagged_unfalsifiable():
    """A 1-decimal claim cannot fail against a dense receipt — the report must say so.

    On the real c6 receipt this floor measured exactly 1.000: every tenth in [0,1] had a source
    value within +/-0.05, so no 1-decimal claim in that document could ever be scored unsourced.
    Those claims' 'grounded' status carries no information and the report must mark them.
    """
    rep = audit_grounding("the effect was 0.4 and the rate 0.7", _dense_receipt())
    assert rep.n_total >= 2
    assert rep.floor_by_decimals["1"] >= 0.995, rep.floor_by_decimals
    assert rep.n_unfalsifiable >= 2, rep.n_unfalsifiable


def test_floor_scales_with_receipt_density():
    """The floor is a property of the RECEIPT, not a constant. Sparse receipt -> lower floor."""
    sparse = audit_grounding("the effect was 0.4", {"a": 0.4, "b": 0.9})
    dense = audit_grounding("the effect was 0.4", _dense_receipt())
    assert sparse.floor_by_decimals["1"] < dense.floor_by_decimals["1"], (
        f"sparse {sparse.floor_by_decimals} vs dense {dense.floor_by_decimals}")


def test_floor_is_not_collapsed_by_a_seed_in_the_receipt():
    """Regression: a large scalar (seed/iteration count) must not deflate the floor.

    This is the v1 defect. With min/max sampling, the 2-decimal floor read ~0.000 because draws
    ranged over [0, 90210]. The floor for 2-decimal claims against this receipt is substantial.
    """
    rep = audit_grounding("we measured 0.42 overall", _receipt())
    assert rep.floor_by_decimals["2"] > 0.05, (
        f"2-decimal floor collapsed to {rep.floor_by_decimals} — a large scalar in the "
        "receipt is stretching the sample range again")


def test_more_decimals_means_lower_floor():
    """Monotonicity: precision is what makes a claim falsifiable."""
    rep = audit_grounding("values 0.4, 0.42, 0.421, 0.4213 were observed", _receipt())
    f = {int(k): v for k, v in rep.floor_by_decimals.items()}
    ds = sorted(f)
    for a, b in zip(ds, ds[1:]):
        assert f[a] >= f[b] - 1e-9, f"floor rose with precision: {f}"


def test_excess_over_chance_separates_truth_from_fabrication():
    """PLANTED CONTROL. The headline rate must not be the only number that moves.

    A document quoting real receipt values and one quoting fabricated values must be separated
    by `excess_over_chance`, not merely by the raw grounded percentage.
    """
    src = _receipt()
    real = "rates of 0.170, 0.340 and 0.510 with cp 0.0340, 0.0680 and 0.1020"
    fake = "rates of 0.173, 0.347 and 0.519 with cp 0.0341, 0.0683 and 0.1027"
    r_real = audit_grounding(real, src)
    r_fake = audit_grounding(fake, src)
    assert r_real.excess_over_chance > r_fake.excess_over_chance, (
        f"judge does not discriminate: real {r_real.excess_over_chance} "
        f"vs fabricated {r_fake.excess_over_chance}")


def test_summary_states_the_floor():
    """The floor must appear in the human-readable output, not just the object."""
    rep = audit_grounding("the rate was 0.42", _receipt())
    s = rep.summary()
    assert "chance floor" in s.lower()
    assert "excess over chance" in s.lower()


def test_empty_document_is_safe():
    rep = audit_grounding("no numbers here at all", _receipt())
    assert rep.n_total == 0
    assert rep.chance_floor == 0.0
    assert rep.excess_over_chance == 0.0


# --- provenance uniqueness -------------------------------------------------------------
# Second pass, same day: the floor says how impressed to be by the RATE; it says nothing about
# whether an individual match is CORRECT. `_match` returned the first dict-order hit. On the real
# C6 audit, 20% of grounded claims matched more than one path (one matched ten), so the reported
# `source` was dict ordering presented as provenance.


def test_ambiguous_matches_are_counted_and_disclosed():
    """A claim matching many paths must be reported as ambiguous, not as pinned provenance."""
    src = {"a": 0.5, "b": 0.5001, "c": 0.4999, "d": 0.9}
    rep = audit_grounding("the rate was 0.50", src)
    grounded = [i for i in rep.items if i.status == "grounded"]
    assert grounded, rep.summary()
    assert grounded[0].n_candidates >= 3, grounded[0].n_candidates
    assert rep.n_ambiguous >= 1
    assert "more than one source path" in rep.summary()


def test_uniquely_pinned_claim_is_not_flagged_ambiguous():
    rep = audit_grounding("the rate was 0.4213", {"a": 0.4213, "b": 0.9, "c": 0.1})
    grounded = [i for i in rep.items if i.status == "grounded"]
    assert grounded and grounded[0].n_candidates == 1
    assert rep.n_ambiguous == 0


def test_grounded_always_has_at_least_one_candidate():
    """Regression: 'grounded' with zero candidates is a contradiction the tool used to emit.

    The percent branch compared round(sv*100, d) == round(claim, d). At d=0 that grants a silent
    +/-0.5% tolerance, so "95% upper bound" grounded against a dispersion ratio of 0.948 while no
    source value was actually within the claim's own tolerance.
    """
    src = {"dispersion": 0.948, "other": 0.12}
    rep = audit_grounding("we report 95 % coverage and a rate of 0.42", src)
    for it in rep.items:
        if it.status == "grounded":
            assert it.n_candidates >= 1, f"{it.raw!r} grounded with no candidate path"


def test_duplicate_values_in_a_receipt_are_all_retained():
    """Regression: `_flatten` kept value -> FIRST path only.

    On the real c6 receipts that discarded 262 of 425 numeric leaves. Every repeated value —
    and rates repeat constantly — collapsed onto whichever path was visited first, and the
    auditor reported that arbitrary survivor as provenance. The collision was deleted upstream,
    so no ambiguity was ever visible. Value-keyed dedupe made the receipt look unambiguous by
    destroying the evidence of ambiguity.
    """
    src = {"a": {"x": 0.25}, "b": {"y": 0.25}, "c": {"z": 0.25}}
    rep = audit_grounding("the rate was 0.25", src)
    grounded = [i for i in rep.items if i.status == "grounded"]
    assert grounded and grounded[0].n_candidates == 3, grounded[0].n_candidates


def test_context_resolution_beats_dict_order_and_admits_when_it_cannot():
    """The resolver must pick the right path from surrounding words — or say 'arbitrary'."""
    src = {"null": {"rho_low": {"mean_licensed": 0.25}},
           "knee": {"rho_low": {"mean_licensed": 0.25}}}
    rep = audit_grounding("under knee at rho_low the mean_licensed was 0.25", src)
    it = [i for i in rep.items if i.status == "grounded"][0]
    assert it.n_candidates == 2
    assert it.resolved_by == "context", it.resolved_by
    assert it.source == "knee.rho_low.mean_licensed", it.source

    # No disambiguating signal -> must NOT claim resolution
    rep2 = audit_grounding("the value was 0.25", src)
    it2 = [i for i in rep2.items if i.status == "grounded"][0]
    assert it2.resolved_by == "arbitrary", it2.resolved_by
    assert rep2.n_arbitrary == 1


def test_ambiguity_accounting_is_exhaustive():
    """Every ambiguous claim is either context-resolved or arbitrary — no third bucket."""
    src = {"alpha": {"rate": 0.5}, "beta": {"rate": 0.5}, "gamma": {"other": 0.9}}
    rep = audit_grounding("alpha rate was 0.5 and separately 0.5 appeared", src)
    assert rep.n_ambiguous == rep.n_context_resolved + rep.n_arbitrary


# --- red-team findings, 2026-08-13 (Claude Fable 5) ------------------------------------
# Commissioned adversary on commits 1fb1de5 / 4de77d1 / af62490. Confirmed both defects the
# author flagged as most likely, and called the zero-claims PASS an abdication. All three fixed.


def test_specific_path_is_not_beaten_by_a_bare_summary_key():
    """LANE 2 regression — the CONFIDENT-WRONG case, worst failure available to the resolver.

    Scoring divided overlap by len(path_tokens), so a bare key ("rate") needed one lucky word to
    score 1.0 while a long specific path was penalised for every token the prose did not repeat.
    The more precisely a sentence named its source, the more the correct path was punished — and
    the result was labelled `context`, not `arbitrary`, so it asserted a provenance it had
    actively mis-derived.
    """
    src = {"rate": 0.1, "cells": {"blockconf_ge3_of_7": {"cave_rate": 0.1}}}
    rep = audit_grounding(
        "The blockconf arm at >=3/7 shows a cave rate of 0.100 on the frozen protocol.", src)
    it = [i for i in rep.items if i.status == "grounded"][0]
    assert it.source == "cells.blockconf_ge3_of_7.cave_rate", (it.source, it.resolved_by)
    assert it.resolved_by == "context"


def test_generic_winner_contained_in_specific_runner_up_declines():
    """If the winner's matched tokens are a strict subset of the runner-up's, decline.

    A specific path is never LESS named than the generic path contained within it, so a
    'win' of that shape is an artifact, not evidence.
    """
    src = {"cave_rate": 0.75, "arm": {"cave_rate": 0.75}}
    rep = audit_grounding("the cave_rate was 0.75", src)
    it = [i for i in rep.items if i.status == "grounded"][0]
    assert it.resolved_by in ("context", "arbitrary")
    if it.resolved_by == "context":
        assert it.source != "cave_rate" or it.n_candidates == 1


def test_chance_floor_uses_the_band_the_claims_occupy():
    """LANE 1 regression — third flattering error in the same function.

    Fixing the order of magnitude was not enough. When claims and source leaves both cluster in
    a narrow sub-range, a uniform [0,1] draw wastes its mass on empty territory and the floor
    reads far too LOW. Measured gap on a 94-leaf receipt in [0, 0.25]: 0.0925 vs 0.3735.
    """
    import random
    rng = random.Random(5)
    src = {f"k{i}": round(rng.uniform(0, 0.25), 4) for i in range(94)}
    r2 = random.Random(11)
    doc = " and ".join(f"the rate was {round(r2.uniform(0, 0.25), 3):.3f}" for _ in range(10))
    rep = audit_grounding(doc, src)
    assert rep.floor_by_decimals["3"] > 0.20, rep.floor_by_decimals
    assert rep.floor_band[1] <= 0.30, rep.floor_band
    # a document of pure noise must NOT come out above chance
    assert rep.excess_over_chance <= 0.05, rep.excess_over_chance


def test_zero_claims_is_void_not_pass():
    """LANE 4 — a gate that extracted nothing has checked nothing.

    'PASS' asserts a check that did not occur, in the flattering direction. The repo's doctrine
    is that a leg which cannot fail must not gate; the honest verdict is a refusal.
    """
    rep = audit_grounding("this document contains no statistics whatsoever", {"a": 0.5})
    assert rep.n_total == 0
    assert rep.verdict.startswith("VOID__"), rep.verdict
    assert "PASS" not in rep.verdict


def test_headline_percentage_has_a_companion_floor_in_the_same_units():
    """The number that gets copy-pasted must not be quotable without its null.

    Found by surveying styxx's own rate-returning functions: the report object had been fixed,
    but the fix lived in summary()/render_html(). A caller reading `pct_grounded` directly still
    got a bare rate — the exact artefact this module spent the day learning not to produce.
    """
    src = {f"k{i}": round(i * 0.011, 3) for i in range(60)}
    rep = audit_grounding("rates of 0.011, 0.022 and 0.033 were seen", src)
    assert hasattr(rep, "pct_chance_floor")
    assert hasattr(rep, "pct_excess_over_chance")
    # same units as the headline, and the identity must hold
    assert abs((rep.pct_grounded - rep.pct_chance_floor)
               - rep.pct_excess_over_chance) < 0.15, (
        rep.pct_grounded, rep.pct_chance_floor, rep.pct_excess_over_chance)


def test_confidence_level_labels_are_not_treated_as_claims():
    """'95% upper bound' / '95% interval' are labels, not statistics. None may be extracted."""
    src = {"dispersion": 0.948, "rate": 0.5}
    for phrase in ("the Clopper-Pearson 95% upper bound was tight",
                   "a 95% interval was used",
                   "the 95% credible region",
                   "reported at 95% confidence"):
        rep = audit_grounding(phrase, src)
        assert all(i.value != 95.0 for i in rep.items), f"{phrase!r} -> {[i.raw for i in rep.items]}"
