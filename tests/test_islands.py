"""styxx.islands — the cohort survey, and the refusals that keep it honest."""
import numpy as np
import pytest

from styxx.islands import survey, cliff, rescue, frame, affinity, MIN_COHORT


def _cohort(n_items=60, seed=0, island=True):
    """Three members sharing a latent geometry, plus one rotated away from it."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_items, 8))
    reps = {}
    for i, name in enumerate(["a", "b", "c"]):
        W = rng.standard_normal((8, 16 + i))
        reps[name] = Z @ W + 0.05 * rng.standard_normal((n_items, 16 + i))
    if island:
        Zi = rng.standard_normal((n_items, 8))          # its own latent: not the shared one
        reps["island"] = Zi @ rng.standard_normal((8, 20))
    else:
        reps["island"] = Z @ rng.standard_normal((8, 20)) + 0.05 * rng.standard_normal((n_items, 20))
    return reps


def test_frame_is_orthonormal_and_item_space():
    X = np.random.default_rng(0).standard_normal((40, 12))
    U = frame(X, k=5)
    assert U.shape == (40, 5)                      # item space, not feature space
    assert np.allclose(U.T @ U, np.eye(5), atol=1e-8)


def test_affinity_bounds():
    U = frame(np.random.default_rng(1).standard_normal((30, 9)), k=4)
    assert affinity(U, U) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= affinity(U, frame(np.random.default_rng(2).standard_normal((30, 9)), k=4)) <= 1.0


def test_survey_finds_the_planted_island():
    s = survey(_cohort(), k=6, n_null=100, n_perm=100)
    assert "island" in s.islands
    assert s.mean_affinity["island"] < min(s.mean_affinity[m] for m in ("a", "b", "c"))


def test_survey_finds_no_island_when_none_planted():
    s = survey(_cohort(island=False), k=6, n_null=100, n_perm=100)
    assert s.islands == []


def test_survey_refuses_a_verdict_below_min_cohort():
    """Bimodality is not testable on four points and the instrument must say so, not guess."""
    s = survey(_cohort(), k=6, n_null=50, n_perm=50)
    assert len(s.members) < MIN_COHORT
    assert s.verdict == "UNDERPOWERED__n_below_8"
    assert any("not testable" in c for c in s.caveats)


def test_survey_rejects_mismatched_item_sets():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="share the item set"):
        survey({"a": rng.standard_normal((30, 5)), "b": rng.standard_normal((31, 5))})


def test_survey_flags_a_cohort_that_shares_no_frame_at_all():
    rng = np.random.default_rng(4)
    reps = {n: rng.standard_normal((40, 10)) for n in "abcd"}     # independent: no shared frame
    s = survey(reps, k=5, n_null=200, n_perm=100)
    assert any("do not share a frame" in c for c in s.caveats)


def test_cliff_refuses_when_the_endpoint_is_at_chance():
    """A knee read off a noise curve is worse than no knee — the documented failure that got
    the module's own internal legibility measure removed before release."""
    reps = _cohort()
    out = cliff(reps["a"], reps["island"], legibility_fn=lambda r, i: 1.0 / len(r), k=6)
    assert out["shape"].startswith("REFUSED__")
    assert out["knee_t_half"] is None and out["transition_width"] is None


def test_rescue_refuses_when_no_rank_beats_chance():
    reps = _cohort()
    out = rescue(reps["a"], reps["island"], legibility_fn=lambda r, i: 1.0 / len(r), ranks=(2, 8))
    assert out["reading"].startswith("REFUSED__")
    assert out["min_sufficient_rank"] is None


def test_cliff_reads_a_switch_like_curve_when_one_is_present():
    """With a legibility measure that fires only near full correction, the shape must read
    switch-like — the b46 signature, on a synthetic stand-in."""
    reps = _cohort()
    doses = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    calls = {"n": 0}

    def fn(reader, corrected):
        calls["n"] += 1
        return [0.01, 0.01, 0.02, 0.05, 0.90, 0.95][calls["n"] - 1]
    out = cliff(reps["a"], reps["island"], legibility_fn=fn, k=6, doses=doses)
    assert out["shape"] == "switch-like"
    assert out["knee_t_half"] == 0.8


def test_rescue_credits_a_recovery_only_against_its_matched_random_null():
    """A correction that a random frame of the same rank reproduces is not a rescue."""
    reps = _cohort()
    seq = iter([0.01,                     # baseline
                0.80, 0.80,               # rank 2: corrected AND its random null both high
                0.85, 0.02])              # rank 8: corrected high, null at floor
    out = rescue(reps["a"], reps["island"], legibility_fn=lambda r, i: next(seq), ranks=(2, 8))
    assert out["min_sufficient_rank"] == 8      # rank 2 rejected: its null matched it
