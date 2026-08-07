"""styxx.apparatus v2 — every failure that killed version one, as a regression.

Version one was told DO NOT SHIP at 0.55 balanced accuracy by its pre-release audit. Each test
below is one of those failures.
"""
import numpy as np
import pytest

from styxx.apparatus import (audit, time_reverse, strip_amplitude,
                             mask_preserving_surrogate, shuffle_within_density)
from styxx.islands import frame, affinity


def stat(X, Y):
    return affinity(frame(X, 20), frame(Y, 20))


def _floor(n, k=20):
    return k / (n - 1)


def _genuine(n=300, p=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 6))
    return Z @ rng.standard_normal((6, p)), Z @ rng.standard_normal((6, p))


def _stall_artifact(n=300, p=200, rate=0.12, seed=0):
    """Two INDEPENDENT streams sharing only a stall clock — no shared world content."""
    rng = np.random.default_rng(seed)
    a = np.random.default_rng(seed * 1000 + 1).standard_normal((n, p))
    b = np.random.default_rng(seed * 1000 + 2).standard_normal((n, p))
    stall = rng.random(n) < rate
    for i in range(1, n):
        if stall[i]:
            a[i], b[i] = a[i - 1], b[i - 1]
    return a, b


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_stall_artifact_that_version_one_certified_is_now_caught(seed):
    """v1's headline failure: two logs sharing only a stall clock returned SURVIVES on 8/8 seeds
    at 10-12x their null. Time reversal cannot catch it — reversal is itself a permutation, the
    operation class the motivating paper says is blind here. The mask surrogate can."""
    a, b = _stall_artifact(seed=seed)
    r = audit(stat, a, b, floor=_floor(len(a)))
    assert not r.survives, r.verdict
    assert "mask_surrogate" in r.verdict


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_genuine_shared_latent_still_survives(seed):
    a, b = _genuine(seed=seed)
    r = audit(stat, a, b, floor=_floor(len(a)))
    assert r.survives and r.verdict == "SURVIVES_APPARATUS_COUNTERFACTUALS"


def test_a_time_symmetric_genuine_signal_is_refused_not_flagged():
    """v1 flagged every periodic-stimulus finding 10/10, because a Fourier subspace is
    reversal-invariant so reversal does not destroy alignment there."""
    n, p = 300, 120
    rng = np.random.default_rng(3)
    t = np.arange(n)
    Z = np.column_stack([np.sin(2 * np.pi * t / 37), np.cos(2 * np.pi * t / 37),
                         np.sin(2 * np.pi * t / 23), np.cos(2 * np.pi * t / 23)])
    A = Z @ rng.standard_normal((4, p)) + 0.1 * rng.standard_normal((n, p))
    B = Z @ rng.standard_normal((4, p)) + 0.1 * rng.standard_normal((n, p))
    r = audit(stat, A, B, floor=_floor(n))
    tr = r.counterfactuals["time_reversal"]
    assert tr["contaminated"] is None and "UNTESTABLE" in tr["note"]
    assert not r.verdict.startswith("RECORDER_CONTAMINATED__time_reversal")


@pytest.mark.parametrize("bad_stat", [lambda X, Y: -0.5, lambda X, Y: 0.0])
def test_a_non_positive_observation_is_invalid_not_survives(bad_stat):
    """v1 returned survives=True unconditionally for obs<=0, while its own docs told users to
    negate statistics into exactly that region."""
    a, b = _genuine(n=100, p=40)
    r = audit(bad_stat, a, b, floor=0.0)
    assert r.verdict == "INVALID__non_positive_observation"
    assert r.survives is False


def test_a_one_stream_audit_is_refused():
    """v1's one-stream path had no world-destroying counterfactual, so it could only ever say
    'survives'."""
    a, _ = _genuine(n=100, p=40)
    with pytest.raises(ValueError, match="one-stream"):
        audit(stat, a)


def test_non_finite_input_is_invalid():
    a, b = _genuine(n=100, p=40)
    a[3, 3] = np.nan
    assert audit(stat, a, b, floor=0.2).verdict == "INVALID__non_finite_input"


def test_an_observation_below_its_floor_is_untestable_not_judged():
    """Retention against a floor the observation does not clear is a power question."""
    a, b = _genuine(n=100, p=40)
    r = audit(stat, a, b, floor=10.0)
    assert r.verdict == "UNTESTABLE__observation_at_or_below_its_floor"
    assert r.survives is False


def test_density_shuffle_refuses_when_strata_cannot_permute():
    """v1's guard was `len(unique) > 1`, which PASSES in the worst case: all-unique counts make
    the shuffle the identity, forcing contamination on genuine findings."""
    a, b = _genuine(n=300, p=80)
    r = audit(stat, a, b, counts=np.arange(len(a)), floor=_floor(len(a)))
    ds = r.counterfactuals["density_shuffle"]
    assert ds["contaminated"] is None and "UNTESTABLE" in ds["note"]


def test_density_shuffle_runs_when_strata_are_coarse():
    a, b = _genuine(n=300, p=80)
    counts = np.repeat(np.arange(10), 30)
    r = audit(stat, a, b, counts=counts, floor=_floor(len(a)))
    assert r.counterfactuals["density_shuffle"]["contaminated"] in (True, False)


def test_shuffle_within_density_reports_its_frozen_fraction():
    B = np.random.default_rng(0).standard_normal((50, 4))
    out, frozen = shuffle_within_density(B, np.arange(50), np.random.default_rng(1))
    assert frozen == 1.0 and np.array_equal(out, B)      # identity, and it says so
    _, frozen2 = shuffle_within_density(B, np.repeat(np.arange(5), 10), np.random.default_rng(1))
    assert frozen2 == 0.0


def test_amplitude_is_a_decomposition_never_a_verdict():
    """v1 auto-labelled any amplitude-carried finding as contaminated. Only the caller knows
    whether magnitude is world (audio envelope, pupil dilation, firing rate) or instrument."""
    n, p = 300, 120
    rng = np.random.default_rng(5)
    e = np.exp(rng.standard_normal(n))                   # a genuine shared envelope
    A = e[:, None] * rng.standard_normal((n, p))
    B = e[:, None] * rng.standard_normal((n, p))
    r = audit(stat, A, B, floor=_floor(n))
    assert "amplitude_stripped" in r.decomposition
    assert not any("amplitude" in k for k in r.counterfactuals)
    assert "amplitude" not in r.verdict


def test_mask_surrogate_preserves_the_repeat_pattern():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((60, 5))
    for i in (10, 11, 30):
        B[i] = B[i - 1]
    S = mask_preserving_surrogate(B, rng)
    for i in (10, 11, 30):
        assert np.allclose(S[i], S[i - 1])
    assert not np.allclose(S[1], B[1])                   # content genuinely resampled


def test_helpers_are_shape_preserving():
    X = np.random.default_rng(0).standard_normal((40, 7))
    assert time_reverse(X).shape == X.shape
    assert np.allclose(time_reverse(time_reverse(X)), X)
    assert np.allclose(np.linalg.norm(strip_amplitude(X), axis=1), 1.0)
