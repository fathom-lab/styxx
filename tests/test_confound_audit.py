"""Locks styxx.audit_confound — the confound-robustness auditor. Synthetic controlled cases pin the verdict
logic (catch / clear / broken / inconclusive); the final test proves the generic API reproduces the bespoke
overconfidence red-team on the cached 2x2."""
import json, math
from pathlib import Path
import numpy as np
import pytest

import styxx
from styxx import audit_confound, ConfoundAuditReport, build_confound_grid
from styxx import validate_against_ground_truth, cem_length_match
from styxx.confound_audit import _lexical_entanglement

ROOT = Path(__file__).resolve().parents[1]


def _synth(score_rule, seed=0, n=200):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    C = rng.standard_normal(n)  # orthogonal to label by construction
    rows = [{"text": f"t{i}", "label": int(y[i]), "confound": float(C[i])} for i in range(n)]
    scores = [float(score_rule(int(y[i]), float(C[i]), rng)) for i in range(n)]
    return rows, scores


def _truenull_text(n=160, seed=0):
    """A real-text corpus where the construct vocabulary is INDEPENDENT of the confound (length)
    within each label — i.e. a true null for the entanglement probe. Construct-word count k is drawn
    independently of total length L, so any margin~length association would be a probe artifact."""
    rng = np.random.default_rng(seed)
    pos = ["great", "love", "excellent", "wonderful", "amazing", "perfect", "best", "happy"]
    neg = ["awful", "hate", "terrible", "horrible", "worst", "bad", "sad", "poor"]
    filler = ["the", "a", "and", "it", "is", "was", "this", "that", "then", "there", "here", "some",
              "more", "words", "extra", "padding"]
    rows = []
    for i in range(n):
        y = i % 2
        vocab = pos if y else neg
        k = int(rng.integers(2, 5)); L = int(rng.integers(5, 40))   # k (construct) independent of L (confound)
        words = list(rng.choice(vocab, k)) + list(rng.choice(filler, max(0, L - k)))
        rng.shuffle(words)
        rows.append({"text": " ".join(words), "label": y, "confound": float(L)})
    return rows


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


# ---------------------------------------------------------------------------
# substrate gate (7.23.0): a confound verdict on a non-ground-truth corpus may be a generator artifact
# ---------------------------------------------------------------------------

def test_synthetic_provenance_warns_on_alarming_verdict():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, confound="c", corpus_provenance="synthetic")
    assert rep.verdict.startswith("THRESHOLD-BIASED")
    assert rep.synthetic_artifact_warning is True
    assert "SYNTHETIC-ARTIFACT RISK" in rep.verdict
    assert rep.corpus_provenance == "synthetic"
    # _synth texts ("t0"..) have no shared tokens -> empty vocab -> probe returns None (documents the limit)
    assert rep.lexical_confound_corr is None


def test_fingerprint_surfaced_in_verdict_end_to_end():
    # the headline path: a SIGNIFICANT lexical fingerprint on a REAL-text (bundled) corpus is computed,
    # surfaced as lexical_confound_corr/p, AND woven into the verdict string.
    from styxx.hf_audit import _load_corpus
    rows = _load_corpus("sentiment"); by = {r["text"]: r for r in rows}
    cmean = sum(r["confound"] for r in rows) / len(rows)
    rep = styxx.audit_hf_model(
        "demo/length-biased", construct="sentiment",
        score_fn=lambda t: 0.5 + 0.30 * (2 * by[t]["label"] - 1) + 0.25 * (by[t]["confound"] - cmean))
    assert rep.verdict.startswith(("THRESHOLD-BIASED", "CONFOUND-DEPENDENT"))
    assert rep.synthetic_artifact_warning is True
    assert rep.lexical_confound_corr is not None and rep.lexical_confound_p < 0.05
    assert "NOT diagnostic" in rep.verdict   # corroborating coupling, not proof of an artifact


def test_lexical_entanglement_robust_to_confound_sorted_input():
    # the probe must NOT manufacture a fingerprint from row order or length-normalization on a true null
    rows = _truenull_text()
    rows.sort(key=lambda r: r["confound"])               # adversarial: rows pre-sorted by the confound
    y = np.array([r["label"] for r in rows]); C = np.array([r["confound"] for r in rows], float)
    corr, p = _lexical_entanglement([r["text"] for r in rows], y, C, reps=300)
    assert p is None or p > 0.05, (corr, p)              # length-invariant binary repr + shuffled folds -> clean


def test_ground_truth_provenance_suppresses_warning():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, confound="c", corpus_provenance="ground_truth")
    assert rep.verdict.startswith("THRESHOLD-BIASED")
    assert rep.synthetic_artifact_warning is False
    assert "SYNTHETIC-ARTIFACT RISK" not in rep.verdict


def test_unspecified_provenance_warns_by_default():
    # caution-by-default: unknown provenance + alarming verdict -> warn (treat as synthetic until validated)
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, confound="c")
    assert rep.synthetic_artifact_warning is True
    assert "UNSPECIFIED provenance" in rep.verdict


def test_robust_verdict_never_warns_even_when_synthetic():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi + rng.normal(0, 0.5))
    rep = audit_confound(rows, scores=scores, corpus_provenance="synthetic")
    assert rep.verdict.startswith("ROBUST")
    assert rep.synthetic_artifact_warning is False


def test_lexical_entanglement_detects_bundled_synthetic_corpus():
    from styxx.hf_audit import _load_corpus
    rows = _load_corpus("sentiment")
    y = np.array([r["label"] for r in rows]); C = np.array([r["confound"] for r in rows], float)
    texts = [r["text"] for r in rows]
    corr, p = _lexical_entanglement(texts, y, C, reps=300)
    assert corr is not None and p is not None
    assert p < 0.05, (corr, p)                 # the bundled LLM-generated corpus IS entangled
    # negative control: break the entanglement by permuting the confound within each label
    rng = np.random.default_rng(0); Cp = C.copy()
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; Cp[idx] = rng.permutation(C[idx])
    _, p_ctrl = _lexical_entanglement(texts, y, Cp, reps=300)
    assert p_ctrl > 0.05, p_ctrl               # control clears (no entanglement left)


def test_cem_length_match_decorrelates_label_and_confound():
    rng = np.random.default_rng(0); n = 400
    # a realistic, weak label<->confound correlation (the finding had 0.07-0.19), the regime CEM targets
    rows = [{"text": f"t{i}", "label": i % 2,
             "confound": float(rng.normal(0.8 if i % 2 else 0.0, 1.0))} for i in range(n)]
    before = abs(np.corrcoef([r["label"] for r in rows], [r["confound"] for r in rows])[0, 1])
    matched = cem_length_match(rows)
    after = abs(np.corrcoef([r["label"] for r in matched], [r["confound"] for r in matched])[0, 1])
    assert len(matched) >= 20 and after < before
    assert after <= 0.20                        # within the audit's orthogonality gate


def test_validate_against_ground_truth_refutes_a_synthetic_artifact():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    syn = audit_confound(rows, scores=scores, confound="c", corpus_provenance="synthetic")
    assert syn.verdict.startswith("THRESHOLD-BIASED")
    # real data where the SAME instrument is actually clean (length effect vanishes) -> refuted
    real_rows = [{"text": f"r{i}", "label": int(rows[i]["label"]), "confound": rows[i]["confound"]}
                 for i in range(len(rows))]
    by = {r["text"]: r for r in real_rows}
    real, rec = validate_against_ground_truth(
        syn, real_rows, score_fn=lambda t: 3.0 * by[t]["label"], confound="c")
    assert real.corpus_provenance == "ground_truth"
    assert real.synthetic_artifact_warning is False
    assert rec.startswith("SYNTHETIC-ARTIFACT (refuted on ground truth)"), rec


def test_validate_rejects_scores_with_matching():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci)
    syn = audit_confound(rows, scores=scores, corpus_provenance="synthetic")
    with pytest.raises(ValueError):
        validate_against_ground_truth(syn, rows, scores=scores, match=True)


def test_validate_inconclusive_real_audit_is_not_a_refutation():
    # an uninformative (non-orthogonal/saturated) real audit must NOT be reported as "refuted"
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    syn = audit_confound(rows, scores=scores, confound="c", corpus_provenance="synthetic")
    assert syn.verdict.startswith("THRESHOLD-BIASED")
    n = 200; yv = np.tile([0, 1], n // 2)
    bad_real = [{"text": f"r{i}", "label": int(yv[i]), "confound": float(yv[i])}  # confound == label: not orthogonal
                for i in range(n)]
    by = {r["text"]: r for r in bad_real}
    real, rec = validate_against_ground_truth(
        syn, bad_real, score_fn=lambda t: 3.0 * by[t]["label"], match=False, confound="c")
    assert real.verdict.startswith("INCONCLUSIVE")
    assert rec.startswith("INCONCLUSIVE on ground truth")
    assert "SYNTHETIC-ARTIFACT (refuted" not in rec        # the precise overclaim we guard against


def test_validate_confirmed_and_clear_branches():
    # CONFIRMED: synthetic alarming AND real also rides the confound
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    syn = audit_confound(rows, scores=scores, confound="c", corpus_provenance="synthetic")
    real_rows = [{"text": f"g{i}", "label": int(rows[i]["label"]), "confound": rows[i]["confound"]}
                 for i in range(len(rows))]
    by = {r["text"]: r for r in real_rows}
    cmean = sum(r["confound"] for r in real_rows) / len(real_rows)
    _, rec_conf = validate_against_ground_truth(
        syn, real_rows, match=False, confound="c",
        score_fn=lambda t: 3.0 * by[t]["label"] - 2.0 * (by[t]["confound"] - cmean))
    assert rec_conf.startswith("CONFIRMED on ground truth")
    # CLEAR: synthetic ROBUST AND real clean
    rows2, scores2 = _synth(lambda yi, ci, rng: 3.0 * yi + rng.normal(0, 0.5))
    syn2 = audit_confound(rows2, scores=scores2, corpus_provenance="synthetic")
    assert syn2.verdict.startswith("ROBUST")
    by2 = {f"c{i}": int(rows2[i]["label"]) for i in range(len(rows2))}
    clean_real = [{"text": f"c{i}", "label": int(rows2[i]["label"]), "confound": rows2[i]["confound"]}
                  for i in range(len(rows2))]
    _, rec_clear = validate_against_ground_truth(
        syn2, clean_real, match=False, score_fn=lambda t: 3.0 * by2[t])
    assert rec_clear.startswith("CLEAR on both")


def test_warning_fires_independent_of_check_entanglement():
    rows, scores = _synth(lambda yi, ci, rng: 3.0 * yi - 2.0 * ci + rng.normal(0, 0.3))
    rep = audit_confound(rows, scores=scores, confound="c",
                         corpus_provenance="synthetic", check_entanglement=False)
    assert rep.synthetic_artifact_warning is True       # provenance-driven, not probe-driven
    assert rep.lexical_confound_corr is None
    assert "SYNTHETIC-ARTIFACT RISK" in rep.verdict


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
