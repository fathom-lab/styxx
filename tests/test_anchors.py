"""styxx.anchors -- smoke-level behavioral contract. Fast, deterministic, generous tolerances
(these are logic checks; the instrument's real characterization lives in the Stage-A receipts)."""
import numpy as np
import pytest

from styxx.anchors import (
    audit_panel, anchor_lr, blindspot_power, min_anchors_for_power,
)

ALPHAS = np.array([0.15, 0.20, 0.10, 0.18])
BETAS = np.array([0.85, 0.80, 0.90, 0.78])
PI, N, K = 0.35, 1500, 200


def make_panel(seed, pi=PI, alphas=ALPHAS, betas=BETAS, n=N, sync=0.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(int)
    p = np.where(y[:, None] == 1, betas[None, :], alphas[None, :])
    V = (rng.random((n, len(alphas))) < p).astype(int)
    if sync > 0:
        key = rng.random(n) < sync
        V[key] = 1
    return y, V


def make_anchors(seed, alphas=ALPHAS, betas=BETAS, k=K, trip=0.0):
    rng = np.random.default_rng(seed)
    neg = (rng.random((k, len(alphas))) < alphas[None, :]).astype(int)
    pos = (rng.random((k, len(betas))) < betas[None, :]).astype(int)
    if trip > 0:
        key = rng.random(k) < trip
        neg[key] = 1
    return neg, pos


def test_clean_panel_estimates_and_reports_regime():
    _, V = make_panel(1)
    neg, pos = make_anchors(2)
    r = audit_panel(V, neg, pos, n_boot=80, null_sims=0, seed=0)
    assert r["verdict"] == "ESTIMATED"
    assert abs(r["pi"] - PI) < 0.08
    assert r["regime"] in ("not_activated", "activated")
    assert r["coverage_note"]
    assert r["ci_source"] == "selective_bootstrap"
    assert r["misfit"]["tau_source"] == "design_point_default"
    assert r["ci"][0] <= r["pi"] <= r["ci"][1]


def test_deaf_panel_voids():
    al, be = np.array([0.45] * 4), np.array([0.52] * 4)
    _, V = make_panel(3, alphas=al, betas=be)
    neg, pos = make_anchors(4, alphas=al, betas=be)
    r = audit_panel(V, neg, pos, n_boot=40, null_sims=0, seed=0)
    assert r["verdict"] == "VOID_PANEL__uninformative"
    assert r["pi"] is None


def test_pooled_detector_contamination_refuses():
    # heavy contamination: half the "negative" stratum is trip-0.9 garbage -> implied pi is
    # impossible and the refusal must fire rather than a clipped confident number
    _, V = make_panel(5, sync=0.02)
    neg, pos = make_anchors(6)
    det_neg, _ = make_anchors(7, trip=0.9)
    pooled = np.vstack([neg, det_neg])
    r = audit_panel(V, pooled, pos, n_boot=80, null_sims=0, seed=0)
    assert r["verdict"] == "VOID_ANCHORS__nonexchangeable"
    assert r["pi"] is None
    assert r["pi_unclipped"] < 0.0


def test_sync_dose_is_priced():
    _, V = make_panel(8, sync=0.15)
    neg, pos = make_anchors(9)
    r = audit_panel(V, neg, pos, n_boot=80, null_sims=0, seed=0)
    assert r["verdict"] == "ESTIMATED"
    assert r["activated"] is True and r["regime"] == "activated"
    assert abs(r["s"] - 0.15) < 0.08
    assert abs(r["pi"] - PI) < 0.10


def test_garbage_stratum_is_diagnostic_only():
    _, V = make_panel(10)
    neg, pos = make_anchors(11)
    det_neg, _ = make_anchors(12, trip=0.8)
    r = audit_panel(V, neg, pos, garbage=det_neg, n_boot=40, null_sims=0, seed=0)
    assert r["garbage"]["master_key_detected"] is True
    # and the detector must NOT have contaminated the estimate
    assert abs(r["pi"] - PI) < 0.08


def test_misfit_null_flags_gross_violation_not_clean():
    # clean panel: p should be unremarkable, and tau should come from the per-dataset null
    _, V = make_panel(13)
    neg, pos = make_anchors(14)
    r_clean = audit_panel(V, neg, pos, n_boot=40, null_sims=60, seed=0)
    assert r_clean["misfit"]["null_p"] > 0.05
    assert r_clean["misfit"]["tau_source"] == "per_dataset_null"
    # gross structure: judge 0 is a specialist -- sharp on anchors, deaf on organic items
    ob = BETAS.copy(); ob[0] = ALPHAS[0]
    _, V2 = make_panel(15, betas=ob)
    neg2, pos2 = make_anchors(16)
    r_bad = audit_panel(V2, neg2, pos2, n_boot=40, null_sims=60, seed=0)
    assert r_bad["misfit"]["null_p"] <= 0.05
    assert r_bad["misfit"]["flag"] is True


def test_determinism():
    _, V = make_panel(17)
    neg, pos = make_anchors(18)
    r1 = audit_panel(V, neg, pos, n_boot=40, null_sims=0, seed=5)
    r2 = audit_panel(V, neg, pos, n_boot=40, null_sims=0, seed=5)
    assert r1["pi"] == r2["pi"] and r1["ci"] == r2["ci"] and r1["s"] == r2["s"]


# --- anchor-threshold instrument (design-time power for catching a shared blind spot) ---
# section-7 design point: J=3, per-judge fp 0.10, 15% shared traps, non-trap fp 0.0961.
DP = dict(J=3, fp_rate=0.10, trap_rate=0.15, fp_rate_alt=0.0961)


def test_single_anchor_lr_matches_frozen_receipt():
    # the frozen receipt (anchor_threshold_result.json) reports 150.8x; convention-independent
    assert abs(anchor_lr(**DP) - 150.8) < 0.05


def test_blindspot_power_is_closed_form_and_tight():
    # standard most-powerful test: at this design point c=1 for K<=50, so power = P(X>=1|p_alt)
    r = blindspot_power(20, **DP)
    assert r["reject_at"] == 1
    assert r["alpha_actual"] <= 0.05                       # valid level-alpha test
    assert abs(r["power"] - 0.9619) < 1e-3
    # single anchor is a smoking gun: nonzero power, unlike the conservative section-7 lower bound
    assert blindspot_power(1, **DP)["power"] > 0.0


def test_tight_power_dominates_conservative_lower_bound():
    # the shipped (tight) power must never fall below the frozen conservative receipt
    conservative = {1: 0.0, 3: 0.0613, 5: 0.1662, 10: 0.4585, 20: 0.8267, 30: 0.953, 50: 0.9972}
    for K, lo in conservative.items():
        assert blindspot_power(K, **DP)["power"] >= lo - 1e-9


def test_min_anchors_for_power_is_tighter_than_paper():
    m90 = min_anchors_for_power(0.90, **DP)
    assert m90["K"] == 15 and m90["power"] >= 0.90
    assert min_anchors_for_power(0.95, **DP)["K"] == 19
    # monotone in the target: a higher bar never needs fewer anchors
    assert min_anchors_for_power(0.95, **DP)["K"] >= m90["K"]


def test_power_increases_with_lr():
    # a bigger shared blind spot (higher trap rate) is easier to catch at fixed K
    weak = blindspot_power(10, J=3, fp_rate=0.10, trap_rate=0.05)["power"]
    strong = blindspot_power(10, J=3, fp_rate=0.10, trap_rate=0.40)["power"]
    assert strong > weak


def test_p_alt_direct_and_input_validation():
    # passing p_alt directly bypasses the trap model and agrees with the derived form
    p_alt = 0.15 + 0.85 * 0.0961 ** 3
    assert abs(blindspot_power(20, J=3, fp_rate=0.10, p_alt=p_alt)["power"]
               - blindspot_power(20, **DP)["power"]) < 1e-12
    with pytest.raises(ValueError):
        blindspot_power(20, J=3, fp_rate=0.10)                  # neither trap_rate nor p_alt
    with pytest.raises(ValueError):
        min_anchors_for_power(0.99999, J=3, fp_rate=0.10, trap_rate=0.15, k_max=5)  # unreachable
