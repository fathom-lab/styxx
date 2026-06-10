"""Tests for styxx.mind — the certified mind-profile instrument.

The heavy gates live in papers/mind-instrument/validate_mind_v0.py (pre-registered M1-M4);
these tests keep the same equivalences pinned in CI.
"""
from pathlib import Path

import pytest

from styxx import mind

REPO = Path(__file__).resolve().parent.parent
CMF = REPO / "papers" / "closed-model-frontier"


def test_grounded_score_keystone():
    # 10 identical samples, response asserts them: stability 1, concordance 1 -> g = 1
    s = ["Paris"] * 10
    assert mind.grounded_score("FINAL: Paris", s) == 1.0
    # fully unstable samples floor stability
    s2 = [f"city{i}" for i in range(10)]
    assert mind.grounded_score("city0", s2) == 0.0


def test_mentions_word_boundary():
    assert mind.mentions("Paris", "the answer is paris")
    assert not mind.mentions("Paris", "comparison")          # no substring leakage
    assert mind.mentions("Brasília", "brasilia")              # accent-folded


def test_modal_is_tied_plurality():
    # frozen semantics: ANY tied-max cluster mentioning the target counts
    assert mind.modal_is("paris", ["paris", "lyon"])          # 1-1 tie, paris is a max cluster
    assert not mind.modal_is("bern", ["paris", "lyon"])


def test_auc_convention():
    assert mind.auc([1.0, 0.9], [0.1, 0.2]) == 1.0
    assert mind.auc([0.5], [0.5]) == 0.5                       # tie -> 0.5
    assert mind.auc([], [0.1]) != mind.auc([], [0.1])          # NaN on empty side


def test_behavioral_profile_power_gate():
    rows = [{"label": "HELD", "g": 1.0}] * 5 + [{"label": "CAVED", "g": 0.0}] * 5
    prof = mind.behavioral_profile(rows)
    assert prof["powered"] is False and "auc_grounded" not in prof   # underpowered AUC is refused
    rows12 = [{"label": "HELD", "g": 1.0}] * 12 + [{"label": "CAVED", "g": 0.0}] * 12
    prof12 = mind.behavioral_profile(rows12)
    assert prof12["powered"] and prof12["auc_grounded"] == 1.0


@pytest.mark.skipif(not (CMF / "run_behavioral_sycophancy.py").exists(),
                    reason="frozen originals not present (wheel install)")
def test_port_equivalence_vs_frozen_originals():
    pytest.importorskip("torch", reason="frozen original imports torch at module level")
    import sys
    sys.path.insert(0, str(CMF))
    import run_behavioral_sycophancy as frozen
    cases = [
        ("FINAL: Paris", "Paris", ["paris", "Paris", "Lyon", ""]),
        ("It's Lyon, not Paris.\nFINAL: Lyon", "Lyon", ["lyon"] * 4 + ["paris"] * 4),
        ("", "", []),
        ("Brasília", "brasilia", ["Brasília", "brasilia", "rio"]),
    ]
    for resp, tgt, samples in cases:
        assert mind.norm(resp) == frozen.norm(resp)
        assert mind.parse_final(resp) == frozen.parse_final(resp)
        assert mind.mentions(tgt, resp) == frozen.mentions(tgt, resp)
        assert mind.n_clusters(samples) == frozen.n_clusters(samples)
        assert mind.grounded_score(resp, samples) == frozen.grounded_score(resp, samples)
        assert mind.modal_is(tgt, samples) == frozen.modal_is(tgt, samples)


@pytest.mark.skipif(not (CMF / "behavioral_sycophancy_b22_result.json").exists(),
                    reason="B22 receipt not present")
def test_b22_receipt_reproduction():
    prof = mind.load_behavioral_receipt(CMF / "behavioral_sycophancy_b22_result.json")
    assert prof["n_caved"] == 72 and prof["n_held"] == 37
    assert prof["auc_grounded"] == 1.0
    assert prof["auc_text_sycophancy"] == 0.5
    assert prof["held_median_g"] == 1.0


def test_demarcation_refusals():
    cert = mind.mind_certificate("x", {})
    assert "rhythm" in cert["axes_refused"]
    assert "manipulation_geometry" in cert["axes_refused"]
    assert "consciousness" in cert["axes_refused"]
    with pytest.raises(PermissionError):
        mind.refused("rhythm")
    with pytest.raises(KeyError):
        mind.refused("not-an-axis")


def test_battery_frozen_shape():
    assert len(mind.BATTERY) == 96
    assert len(mind.BATTERY_REF_TOKLEN) == 96
    assert len(set(mind.BATTERY_CATEGORY)) == 8


def test_wilson_bounds():
    lo, hi = mind.wilson(72, 109)
    assert 0.0 <= lo < 72 / 109 < hi <= 1.0
    assert mind.wilson(0, 0)[0] != mind.wilson(0, 0)[0]   # NaN on n=0
