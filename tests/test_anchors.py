"""styxx.anchors -- smoke-level behavioral contract. Fast, deterministic, generous tolerances
(these are logic checks; the instrument's real characterization lives in the Stage-A receipts)."""
import numpy as np
import pytest

from styxx.anchors import audit_panel

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
