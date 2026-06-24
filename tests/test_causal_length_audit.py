# -*- coding: utf-8 -*-
"""Regression-lock for the 2026-06-24 causal length audit (commits 97ff395, e70f55e).

The correlational length-floor OVER-flags length reliance; the causal test (length-matching +
length-feature ablation, on each instrument's OWN corpus) is the correct tool. This test locks the
three load-bearing claims so a future recalibration can't silently re-introduce the overclaim:

  1. deception is CONSTRUCT-ROBUST under length control (the floor's "0.82 length share" was correlational;
     causally, ablated + length-matched AUC stays high) — guards against re-asserting "deception is a length
     detector."
  2. overconfidence is the ONE length-CARRIED instrument (ablating its length features costs real AUC).
  3. calibrated epistemic register is intrinsically ~1.16x wordier than overconfident register — the
     mechanism behind overconfidence's irreducible length cue (the matched-rebuild honest null).

Offline, deterministic, package-only (uses styxx.guardrail extractors directly). See
papers/grounded-honesty-axis/FINDING_suite_causal_length_2026_06_24.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data"


def _load(rel):
    p = DATA / rel
    if not p.exists():
        pytest.skip(f"corpus not present: {rel}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _matrix(rows, extract, label_key, text_key="response"):
    feats = [extract(r.get("question", ""), r.get(text_key, "")) for r in rows]
    names = list(feats[0].keys())
    X = np.array([[f[n] for n in names] for f in feats], float)
    y = np.array([int(r[label_key]) for r in rows], int)
    wc = np.array([len(str(r.get(text_key, "")).split()) for r in rows], float)
    return X, y, names, wc


def _cv_auc(X, y, idxs, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    Xi = X[:, idxs]
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, y):
        s = StandardScaler().fit(Xi[tr])
        c = LogisticRegression(max_iter=2000).fit(s.transform(Xi[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.predict_proba(s.transform(Xi[te]))[:, 1]))
    return float(np.mean(aucs))


def _cem_idx(wc, y, binw, seed=0):
    rng = np.random.default_rng(seed)
    b = (wc // binw).astype(int)
    keep = []
    for bb in np.unique(b):
        pos = np.where((b == bb) & (y == 1))[0]
        neg = np.where((b == bb) & (y == 0))[0]
        k = min(len(pos), len(neg))
        if k:
            keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    return np.array(sorted(keep), int)


def test_deception_is_construct_robust_under_length_control():
    """Length-matching deception's OWN corpus + ablating log_word_count must NOT collapse it.
    Locks the correction of the correlational floor (which over-flagged deception as 0.82 length)."""
    from styxx.guardrail.deception_signals import extract_deception_features
    rows = _load("deception/responses_v0.jsonl")
    X, y, names, wc = _matrix(rows, extract_deception_features, "label_dishonest")
    nolen = [i for i, n in enumerate(names) if "word_count" not in n]
    idx = _cem_idx(wc, y, binw=8)
    assert len(idx) >= 40, "too few length-matched pairs to test"
    auc_cem_ablated = _cv_auc(X[idx], y[idx], nolen)
    # causally robust: even length-matched AND length-feature-ablated, deception separates well above chance
    assert auc_cem_ablated >= 0.74, (
        f"deception CEM+ablated AUC {auc_cem_ablated:.3f} < 0.74 — the 'deception is a length detector' "
        f"overclaim would be back; the 2026-06-24 finding says it is construct-robust under length control"
    )


def test_overconfidence_is_the_length_carried_instrument():
    """Ablating overconfidence's length features (mean_sentence_length, log_word_count) must cost real AUC.
    Locks 'overconfidence is the one causally length-confounded instrument'."""
    from styxx.guardrail.overconfidence_signals import extract_overconfidence_features
    rows = _load("overconfidence/pairs_v0.jsonl")
    X, y, names, wc = _matrix(rows, extract_overconfidence_features, "label_overconfident")
    allf = list(range(len(names)))
    nolen = [i for i, n in enumerate(names) if n not in {"log_word_count", "mean_sentence_length"}]
    raw = _cv_auc(X, y, allf)
    raw_abl = _cv_auc(X, y, nolen)
    assert (raw - raw_abl) >= 0.03, (
        f"dropping overconfidence's length features cost only {raw-raw_abl:.3f} AUC (raw {raw:.3f} -> "
        f"{raw_abl:.3f}); the 2026-06-24 finding says length is load-bearing for THIS instrument"
    )


def test_calibration_is_intrinsically_wordier():
    """Across the original (gpt) corpus, calibrated answers are materially longer than overconfident.
    Locks the ~1.16x calibration-verbosity ratio that explains overconfidence's irreducible length cue."""
    rows = _load("overconfidence/pairs_v0.jsonl")
    y = np.array([int(r["label_overconfident"]) for r in rows])
    wc = np.array([len(r["response"].split()) for r in rows], float)
    ratio = wc[y == 0].mean() / wc[y == 1].mean()  # calibrated / overconfident
    assert ratio >= 1.10, (
        f"calibrated/overconfident word ratio {ratio:.3f} < 1.10 — the calibration-verbosity mechanism "
        f"(hedging costs words) underpins the overconfidence length finding"
    )
