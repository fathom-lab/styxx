"""styxx.coupling — the three-world exam, and the refusals that outrank a positive."""
import numpy as np
import pytest

from styxx.coupling import couple, rv_coefficient, resample_pair


def _world(coupled, clocked, n=240, seed=11):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.choice(14 * 24, size=n, replace=False)).astype(float) * 3600.0
    ang = 2 * np.pi * (((ts // 3600) % 24).astype(int)) / 24
    clock = np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], 1)
    z = rng.standard_normal((n, 4))
    zb = z if coupled else rng.standard_normal((n, 4))
    c = clock if clocked else np.zeros_like(clock)
    A = np.tanh(z @ rng.standard_normal((4, 12)) + c @ rng.standard_normal((4, 12)))
    B = np.tanh(zb @ rng.standard_normal((4, 24)) + c @ rng.standard_normal((4, 24)))
    return A + 0.15 * rng.standard_normal((n, 12)), ts, B + 0.15 * rng.standard_normal((n, 24))


def _run(coupled, clocked, **kw):
    A, ts, B = _world(coupled, clocked)
    return couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24,
                  bin_seconds=3600, n_perm=200, min_bins=200, **kw)


def test_rv_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((40, 6)), rng.standard_normal((40, 9))
    assert rv_coefficient(X, Y) == pytest.approx(rv_coefficient(Y, X), abs=1e-9)
    assert 0.0 <= rv_coefficient(X, Y) <= 1.0
    assert rv_coefficient(X, X) == pytest.approx(1.0, abs=1e-6)


def test_detects_real_coupling():
    r = _run(coupled=True, clocked=True)
    assert r.verdict == "COUPLED_BEYOND_CONFOUND__attribution_pending"
    assert r.matched_p <= 0.01


def test_the_confound_matched_null_absorbs_a_pure_clock_that_fools_the_free_shuffle():
    """The whole reason this module exists: a shared clock is real correspondence and
    completely uninteresting. The free shuffle falls for it; the licensing null must not."""
    r = _run(coupled=False, clocked=True)
    assert r.free_p <= 0.01, "the confound must be strong enough to fool a naive shuffle"
    assert r.matched_p > 0.10, "the matched null must absorb it"
    assert r.verdict == "CONFOUND_ONLY__explained_by_the_supplied_confound"


def test_silent_on_nothing_and_quotes_its_power_floor():
    r = _run(coupled=False, clocked=False)
    assert r.verdict == "NO_DETECTABLE_COUPLING__above_measured_floor"
    assert r.power_floor["detectable_rv_at_p01"] > 0
    assert any("NOT as 'no coupling'" in c for c in r.caveats)


def test_a_positive_never_claims_attribution():
    """Symmetric statistic: 'the agent senses the room' is not available from this measurement,
    and the verdict string itself has to say so."""
    r = _run(coupled=True, clocked=True)
    assert "attribution_pending" in r.verdict
    assert any("cannot distinguish" in c for c in r.caveats)


def test_refuses_below_the_coverage_gate():
    A, ts, B = _world(True, True, n=40)
    r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24, bin_seconds=3600,
               n_perm=50, min_bins=200)
    assert r.verdict == "INVALID__insufficient_overlap"
    assert r.power_floor["detectable_rv_at_p01"] is None


def test_omitting_the_confound_is_reported_as_a_weaker_question():
    A, ts, B = _world(coupled=False, clocked=True)
    r = couple(A, ts, B, ts, confound=None, bin_seconds=3600, n_perm=200, min_bins=200)
    assert r.verdict == "FREE_SHUFFLE_ONLY__no_confound_supplied_claim_not_licensed"
    assert any("does not license" in c for c in r.caveats)


def test_degenerate_confound_is_flagged():
    """One group per bin: the matched null can absorb nothing and must say so."""
    A, ts, B = _world(True, True)
    r = couple(A, ts, B, ts, confound=lambda b: np.arange(len(np.asarray(b))),
               bin_seconds=3600, n_perm=100, min_bins=200)
    assert any("degenerate" in c for c in r.caveats)


def test_resample_drops_bins_missing_on_either_side():
    a = np.arange(10).reshape(10, 1).astype(float)
    b = np.arange(4).reshape(4, 1).astype(float)
    A, B, bins = resample_pair(a, np.arange(10) * 60.0, b, np.arange(4) * 60.0, 60.0)
    assert len(A) == len(B) == len(bins) == 4


def test_confound_length_mismatch_raises():
    A, ts, B = _world(True, True)
    with pytest.raises(ValueError, match="labels for"):
        couple(A, ts, B, ts, confound=lambda b: np.array([0, 1]), bin_seconds=3600,
               n_perm=10, min_bins=200)
