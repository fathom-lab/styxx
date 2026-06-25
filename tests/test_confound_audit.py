"""Locks styxx.audit_confound — the confound-robustness auditor. Synthetic controlled cases pin the verdict
logic (catch / clear / broken / inconclusive); the final test proves the generic API reproduces the bespoke
overconfidence red-team on the cached 2x2."""
import json, math
from pathlib import Path
import numpy as np
import pytest

import styxx
from styxx import audit_confound, ConfoundAuditReport, build_confound_grid

ROOT = Path(__file__).resolve().parents[1]


def _synth(score_rule, seed=0, n=200):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    C = rng.standard_normal(n)  # orthogonal to label by construction
    rows = [{"text": f"t{i}", "label": int(y[i]), "confound": float(C[i])} for i in range(n)]
    scores = [float(score_rule(int(y[i]), float(C[i]), rng)) for i in range(n)]
    return rows, scores


def test_robust_when_score_ignores_confound():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi + rng.normal(0, 0.5))
    rep = audit_confound(rows, scores=scores)
    assert rep.gate_ok
    assert rep.verdict.startswith("ROBUST"), rep.verdict


def test_threshold_biased_and_guard_helps():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, instrument="x", confound="c")
    assert rep.verdict.startswith("THRESHOLD-BIASED"), rep.verdict
    assert rep.confound_score_coef_ci95[1] < 0           # confound effect significant (negative)
    assert abs(rep.guard_disparity_adj_oos) < abs(rep.guard_disparity_raw)  # guard reduces the disparity OOS
    assert rep.guard(1.0, rep.guard_ref) == pytest.approx(1.0, abs=1e-6)    # operating-point preserved at ref


def test_confound_dependent_when_score_is_pure_confound():
    rows, scores = _synth(lambda yi, ci, rng: -2.0 * ci + rng.normal(0, 0.3))  # ignores the label entirely
    rep = audit_confound(rows, scores=scores)
    assert "CONFOUND-DEPENDENT" in rep.verdict, rep.verdict


def test_inconclusive_when_grid_not_orthogonal():
    n = 200; y = np.tile([0, 1], n // 2)
    rows = [{"text": f"t{i}", "label": int(y[i]), "confound": float(y[i] + 0.01 * i)} for i in range(n)]  # confound∝label
    scores = [3.0 * int(y[i]) for i in range(n)]
    rep = audit_confound(rows, scores=scores)
    assert rep.verdict.startswith("INCONCLUSIVE"), rep.verdict


def test_construct_recoverable_disambiguates_broken_instrument():
    # confound-dependent score, but construct IS recoverable -> verdict says the INSTRUMENT is broken
    rows, scores = _synth(lambda yi, ci, rng: -2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, confound="c", construct_recoverable_auc=0.80)
    assert "CONFOUND-DEPENDENT" in rep.verdict
    assert "construct IS recoverable" in rep.verdict
    assert rep.construct_recoverable_auc == 0.80


def test_construct_absent_flips_to_inconclusive():
    # if the construct is not recoverable even by refit, any confound verdict is untrustworthy -> INCONCLUSIVE
    rows, scores = _synth(lambda yi, ci, rng: -2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, confound="c", construct_recoverable_auc=0.52)
    assert rep.verdict.startswith("INCONCLUSIVE")


def test_exports_first_class():
    assert {"audit_confound", "ConfoundAuditReport", "build_confound_grid"} <= set(styxx.__all__)
    assert callable(audit_confound)


def test_build_confound_grid_with_fake_generator():
    def fake_gen(system, item):
        return "one short line." if "ONE sentence" in system else "a considerably longer piece of generated text here"
    rows = build_confound_grid(["q1", "q2"], "POS-STANCE", "NEG-STANCE",
                               {"short": "ONE sentence", "long": "five sentences"}, fake_gen)
    assert len(rows) == 2 * 2 * 2  # items x {pos,neg} x {short,long}
    assert all({"text", "label", "confound", "confound_level"} <= set(r) for r in rows)
    assert set(r["label"] for r in rows) == {0, 1}
    # confound (log words) is larger for the long level
    longs = [r["confound"] for r in rows if r["confound_level"] == "long"]
    shorts = [r["confound"] for r in rows if r["confound_level"] == "short"]
    assert min(longs) > max(shorts)


def test_overconfidence_red_team_reproduced_via_generic_api():
    p = ROOT / "benchmarks" / "data" / "overconfidence" / "adversarial_lenxreg_gemini.jsonl"
    if not p.exists():
        pytest.skip("2x2 corpus absent")
    import styxx.guardrail.calibrated_weights_overconfidence_v0 as W0
    from styxx.guardrail.overconfidence_signals import extract_overconfidence_features
    rows0 = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    feats = [extract_overconfidence_features(r["question"], r["response"]) for r in rows0]
    X = np.array([[f[n] for n in W0.FEATURE_NAMES] for f in feats], float)
    z = (X - np.asarray(W0.SCALER_MEAN)) / np.asarray(W0.SCALER_SCALE)
    S = z @ np.asarray(W0.COEFS) + W0.INTERCEPT
    rows = [{"text": r["response"], "label": r["label_overconfident"],
             "confound": math.log1p(len(r["response"].split()))} for r in rows0]
    rep = audit_confound(rows, scores=list(S), instrument="overconfidence_v0", confound="log_words")
    assert rep.verdict.startswith("THRESHOLD-BIASED")
    assert rep.harm["max_swing"] >= 0.30          # the measured ~46% length swing
    assert rep.within_stratum_auc["low"] >= 0.70 and rep.within_stratum_auc["high"] >= 0.70
    assert rep.guard_slope < 0                      # short reads more overconfident
    assert rep.guard_auc_adj_oos >= rep.guard_auc_raw  # guard doesn't hurt overall AUC OOS
