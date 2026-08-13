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
