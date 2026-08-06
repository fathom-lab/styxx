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
    A, B, bins, counts = resample_pair(a, np.arange(10) * 60.0, b, np.arange(4) * 60.0, 60.0)
    assert len(A) == len(B) == len(bins) == len(counts) == 4


def test_confound_length_mismatch_raises():
    A, ts, B = _world(True, True)
    with pytest.raises(ValueError, match="labels for"):
        couple(A, ts, B, ts, confound=lambda b: np.array([0, 1]), bin_seconds=3600,
               n_perm=10, min_bins=200)


def test_refuses_a_positive_when_shared_sampling_density_is_open():
    """The real-data catch, 2026-08-06: a stream against its own TIME-REVERSED copy has no
    bin-level coupling to find, yet irregular sampling makes both streams' bin averages shrink
    together and no permutation null absorbs it. The verdict must name that channel instead of
    claiming coupling."""
    rng = np.random.default_rng(5)
    n = 4000
    # bursty arrivals -> wildly uneven bin counts, the condition that opens the channel
    ts = np.sort(rng.exponential(20.0, n).cumsum())
    X = rng.standard_normal((n, 8)) * rng.gamma(2.0, 1.0, (n, 1))     # heteroscedastic
    r = couple(X[:, :4], ts, X[::-1, 4:], ts, confound=lambda b: np.asarray(b) % 24,
               bin_seconds=60.0, n_perm=150, min_bins=100)
    assert r.sampling_density["applicable"]
    if r.sampling_density["shared"]:
        assert r.verdict == "COUPLED__sampling_density_confound_unbounded"
        assert any("sampling times" in c for c in r.caveats)


def test_uniform_sampling_reports_the_density_channel_as_inapplicable():
    A, ts, B = _world(True, True)
    r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24, bin_seconds=3600,
               n_perm=100, min_bins=200)
    assert r.sampling_density["applicable"] is False
    assert r.verdict == "COUPLED_BEYOND_CONFOUND__attribution_pending"


def _ar1(n, d, rho, rng):
    z = np.zeros((n, d))
    z[0] = rng.standard_normal(d)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(d)
    return z


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_independent_autocorrelated_streams_are_not_called_coupled(seed):
    """Red-team 2026-08-06, the credibility-ending one: two INDEPENDENT AR(1) streams reached the
    permutation floor on 20/20 seeds, because within-group shuffling destroys the autocorrelation
    the data actually has, so the null describes white noise the streams are not."""
    n = 336
    ts = np.arange(n, dtype=float) * 3600.0
    A = _ar1(n, 6, 0.98, np.random.default_rng(seed * 1000 + 1))
    B = _ar1(n, 6, 0.98, np.random.default_rng(seed * 1000 + 2))
    r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24, bin_seconds=3600,
               n_perm=200, min_bins=200)
    assert not r.verdict.startswith("COUPLED_BEYOND"), r.verdict
    assert r.dependence["autocorrelated"] is True


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_independent_drifting_streams_are_not_called_coupled(seed):
    """Red-team: independent linear drifts produced 21/21 false positives. Two streams that
    merely warmed up during the window must not certify as coupled."""
    n, rng = 336, np.random.default_rng(seed)
    ts = np.arange(n, dtype=float) * 3600.0
    t = np.arange(n)[:, None]
    A = rng.standard_normal((n, 6)) + t * rng.standard_normal((1, 6)) * 0.05
    B = rng.standard_normal((n, 6)) + t * rng.standard_normal((1, 6)) * 0.05
    r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24, bin_seconds=3600,
               n_perm=200, min_bins=200)
    assert not r.verdict.startswith("COUPLED_BEYOND"), r.verdict


def test_nonfinite_input_refuses_instead_of_certifying_garbage():
    """Red-team: a single NaN produced a maximal-confidence COUPLED verdict, because every
    permuted RV is NaN and `nan >= nan` is False, so zero permutations exceed the observation."""
    n, rng = 336, np.random.default_rng(0)
    ts = np.arange(n, dtype=float) * 3600.0
    A, B = rng.standard_normal((n, 5)), rng.standard_normal((n, 5))
    A[7, 2] = np.nan
    r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24, bin_seconds=3600,
               n_perm=100, min_bins=200)
    assert r.verdict == "INVALID__nonfinite_input"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_few_shared_glitch_bins_cannot_carry_a_verdict(k):
    """Red-team: two sentinel bins out of 336 gave RV 0.973 at p 0.002 on independent streams —
    a glitch written to both logs (power blip, clock-sync marker) reproduces it exactly."""
    n = 336
    ts = np.arange(n, dtype=float) * 3600.0
    A = np.random.default_rng(1).standard_normal((n, 5))
    B = np.random.default_rng(2).standard_normal((n, 5))
    for j in range(k):
        A[100 + 50 * j] = 50.0
        B[100 + 50 * j] = 50.0
    r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24, bin_seconds=3600,
               n_perm=200, min_bins=200)
    assert not r.verdict.startswith("COUPLED_BEYOND"), r.verdict
    assert r.dependence["leverage_top_bin_share"] >= 0.5


def test_a_positive_names_the_confound_it_is_beyond():
    """Red-team: a day-of-week driver survives an hour-of-day null intact, and the verdict string
    reads as 'coupled'. The caveat must scope the claim to the confound actually supplied."""
    r = _run(coupled=True, clocked=True)
    assert any("ONE confound you supplied" in c for c in r.caveats)


def test_debiased_estimator_is_flat_on_independent_noise_where_rv_inflates():
    """External audit 2026-08-06: rv_coefficient IS linear CKA and inflates with feature count —
    0.91 on INDEPENDENT random streams at this module's own minimum n. The p-value survives (the
    null is drawn at the same n and p) but the coefficient is a dimensionality readout, so the
    debiased estimator is reported beside it."""
    from styxx.coupling import debiased_cka
    for p in (100, 500, 2000):
        rng = np.random.default_rng(0)
        X, Y = rng.standard_normal((200, p)), rng.standard_normal((200, p))
        assert abs(debiased_cka(X, Y)) < 0.05, p
    rv_hi = rv_coefficient(*[np.random.default_rng(0).standard_normal((200, 2000))] * 1,
                           np.random.default_rng(1).standard_normal((200, 2000)))
    assert rv_hi > 0.8, "the biased statistic should still show the inflation it is known for"


def test_debiased_estimator_does_not_lose_genuine_coupling():
    from styxx.coupling import debiased_cka
    z = np.random.default_rng(1).standard_normal((200, 8))
    X = z @ np.random.default_rng(2).standard_normal((8, 500))
    Y = z @ np.random.default_rng(3).standard_normal((8, 500))
    assert debiased_cka(X, Y) > 0.9
    assert abs(rv_coefficient(X, Y) - debiased_cka(X, Y)) < 0.05


def test_frozen_bins_in_singleton_strata_are_reported():
    """A fine-grained confound freezes some bins at their true pairing in every null draw, so that
    fraction is never tested. Silent power loss unless it is surfaced."""
    A, ts, B = _world(True, True)
    n_bins = 240
    r = couple(A, ts, B, ts, bin_seconds=3600, n_perm=100, min_bins=200,
               confound=lambda b: np.arange(len(np.asarray(b))) // 2)
    assert r.dependence["frozen_bin_fraction"] >= 0.0
    r2 = couple(A, ts, B, ts, bin_seconds=3600, n_perm=100, min_bins=200,
                confound=lambda b: np.asarray(b) % 24)
    assert r2.dependence["frozen_bin_fraction"] == 0.0
