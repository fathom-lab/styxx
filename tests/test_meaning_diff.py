"""Tests for styxx.meaning_diff — the meaning-regression instrument (no torch, no network)."""
import numpy as np
import pytest

import styxx.meaning_diff as md   # the submodule (styxx.meaning_diff the attribute is the function)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_self_comparison_is_perfect():
    R = _rng().standard_normal((40, 16))
    r = md.meaning_diff(R, R, words=[f"w{i}" for i in range(40)])
    assert r["agreement"] == 1.0
    assert r["verdict"] == "HEALTHY"
    assert r["divergent_concepts"] == []          # no false divergence vs itself


def test_shuffle_breaks_agreement():
    R = _rng().standard_normal((40, 16))
    S = R[_rng(1).permutation(40)]
    r = md.meaning_diff(R, S)
    assert r["agreement"] < 0.5 and r["verdict"] == "BROKEN"


def test_rdm_matches_norm_equalized_cosine():
    R = _rng().standard_normal((30, 12))
    D = md.rdm(R)                                   # norm_equalized cosine geometry
    assert D.shape == (30, 30)
    assert np.allclose(np.diag(D), 0.0, atol=1e-3)  # ~4e-5 from the cosine epsilon (matches distmat)
    assert np.allclose(D, D.T)


def test_norm_sensitive_flag_changes_geometry():
    R = _rng().standard_normal((25, 10))
    R[0] *= 50.0                                    # one high-norm concept
    eq = md.meaning_diff(R, R, norm_equalized=True)
    se = md.meaning_diff(R, R, norm_equalized=False)
    assert eq["agreement"] == 1.0 and se["agreement"] == 1.0   # self always agrees
    assert not np.allclose(md.rdm(R, norm_equalized=True), md.rdm(R, norm_equalized=False))


def test_divergent_concepts_named_and_ranked():
    R = _rng().standard_normal((30, 16))
    S = R.copy()
    # move two concepts' geometry by scrambling their relations
    S[5] = _rng(7).standard_normal(16) * 3
    S[12] = _rng(8).standard_normal(16) * 3
    words = [f"c{i}" for i in range(30)]
    r = md.meaning_diff(R, S, words=words, top_k=5)
    moved = {w for w, _ in r["divergent_concepts"]}
    assert "c5" in moved or "c12" in moved
    scores = [s for _, s in r["divergent_concepts"]]
    assert scores == sorted(scores, reverse=True)   # ranked descending


def test_reliability_caveat_path():
    R = _rng().standard_normal((20, 8))
    r = md.meaning_diff(R, R + 0.01 * _rng(2).standard_normal((20, 8)))
    assert r["reliability"] is None
    assert r["reliable"] is True                    # caveat path = indicative
    assert r["reliability_caveat"] is not None


def test_template_path_measures_reliability():
    base = _rng().standard_normal((20, 8))
    Ta = np.stack([base + 0.01 * _rng(i).standard_normal((20, 8)) for i in range(8)])
    Tb = np.stack([base + 0.01 * _rng(i + 99).standard_normal((20, 8)) for i in range(8)])
    r = md.meaning_diff_templates(Ta, Tb, words=[f"w{i}" for i in range(20)])
    assert r["reliability"] is not None and 0.0 < r["reliability"] <= 1.0


def test_mismatched_concept_counts_raise():
    with pytest.raises(ValueError):
        md.meaning_diff(_rng().standard_normal((10, 4)), _rng().standard_normal((11, 4)))


def test_no_torch_at_import():
    # order-independent: a FRESH interpreter importing only the module must not pull torch
    import subprocess
    import sys
    code = "import styxx.meaning_diff, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"            # core-wheel safe


def test_module_and_function_reachable():
    import styxx
    from styxx.meaning_diff import meaning_diff      # the product call path
    R = _rng().standard_normal((20, 8))
    assert meaning_diff(R, R)["agreement"] == 1.0
    assert styxx.meaning_diff.meaning_diff(R, R)["agreement"] == 1.0   # via the module


def test_verdict_bands_frozen():
    assert md.VERDICT_BANDS == {"HEALTHY": 0.80, "DRIFTED": 0.50}
