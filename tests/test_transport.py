# -*- coding: utf-8 -*-
"""Tests for styxx.transport — universal cognometric transport.

Offline and deterministic (synthetic embedding spaces, fixed seed). No
network, no model downloads. These verify the *math* of label-free
linear transport that the 2026-05-17 dogfood validated on real
embeddings:

  - a cognometric instrument scores in (0,1) and ranks correctly
  - procrustes transport recovers class separation through an unknown
    orthogonal rotation (and beats naive no-transport by a wide margin)
  - ridge transport handles unequal home/foreign dimensions
  - fit() rejects an unpaired corpus (the documented boundary)
  - instrument and transport round-trip through save/load
"""
from __future__ import annotations

import numpy as np
import pytest

from styxx.transport import (
    CognometricInstrument,
    Transport,
    transported_score,
)

SEED = 20260517


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores), 0, -1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@pytest.fixture(scope="module")
def world():
    """A synthetic home space, a known cognometric direction, a
    label-free corpus, labeled fit/eval sets, and two foreign spaces
    (an orthogonal rotation; a higher-dim linear projection)."""
    rng = np.random.default_rng(SEED)
    dim = 64
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)

    def labeled(n):
        half = n // 2
        pos = 2.0 * direction + 0.3 * rng.standard_normal((half, dim))
        neg = -2.0 * direction + 0.3 * rng.standard_normal((half, dim))
        X = np.vstack([pos, neg])
        y = np.array([1] * half + [0] * half, dtype=float)
        return X, y

    corpus_home = rng.standard_normal((400, dim))      # label-free
    fit_home, fit_y = labeled(40)
    eval_home, eval_y = labeled(40)

    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))   # orthogonal
    P = rng.standard_normal((dim, 80))                      # dim 64 -> 80

    return {
        "dim": dim, "Q": Q, "P": P,
        "corpus_home": corpus_home,
        "fit_home": fit_home, "fit_y": fit_y,
        "eval_home": eval_home, "eval_y": eval_y,
    }


def test_instrument_scores_unit_interval_and_ranks(world):
    instr = CognometricInstrument.from_labeled(
        world["fit_home"], world["fit_y"])
    p = instr.score(world["eval_home"])
    assert p.shape == (len(world["eval_y"]),)
    assert np.all((p > 0.0) & (p < 1.0))
    y = world["eval_y"]
    assert p[y == 1].mean() > p[y == 0].mean()
    assert _auc(p, y) > 0.95  # sanity: native separation is strong


def test_procrustes_transport_recovers_separation(world):
    Q = world["Q"]
    t = Transport.fit(world["corpus_home"], world["corpus_home"] @ Q,
                      method="procrustes")
    assert t.report.method == "procrustes"

    instr = CognometricInstrument.from_labeled(
        t.home_repr(world["fit_home"]), world["fit_y"])
    foreign_eval = world["eval_home"] @ Q
    transported = _auc(transported_score(instr, t, foreign_eval),
                       world["eval_y"])

    # naive: home instrument applied straight to foreign (same dim here)
    naive_instr = CognometricInstrument.from_labeled(
        world["fit_home"], world["fit_y"])
    naive = _auc(naive_instr.score(foreign_eval), world["eval_y"])

    assert transported > 0.90
    assert transported - naive > 0.25


def test_ridge_transport_handles_unequal_dims(world):
    P = world["P"]
    t = Transport.fit(world["corpus_home"], world["corpus_home"] @ P,
                      method="ridge", ridge_lambda=1e-2)
    assert t.report.foreign_dim == 80 and t.report.home_dim == 64

    instr = CognometricInstrument.from_labeled(
        t.home_repr(world["fit_home"]), world["fit_y"])
    transported = _auc(
        transported_score(instr, t, world["eval_home"] @ P),
        world["eval_y"])
    assert transported > 0.85


def test_fit_rejects_unpaired_corpus(world):
    with pytest.raises(ValueError, match="PAIRED"):
        Transport.fit(world["corpus_home"][:50],
                      (world["corpus_home"] @ world["Q"])[:49],
                      method="procrustes")


def test_unknown_method_rejected(world):
    with pytest.raises(ValueError, match="unknown method"):
        Transport.fit(world["corpus_home"],
                      world["corpus_home"] @ world["Q"],
                      method="gan")


def test_save_load_roundtrip(world, tmp_path):
    Q = world["Q"]
    t = Transport.fit(world["corpus_home"], world["corpus_home"] @ Q,
                      method="procrustes")
    instr = CognometricInstrument.from_labeled(
        t.home_repr(world["fit_home"]), world["fit_y"])
    foreign_eval = world["eval_home"] @ Q

    tp = t.save(tmp_path / "transport.npz")
    ip = instr.save(tmp_path / "instr.npz")
    t2 = Transport.load(tp)
    i2 = CognometricInstrument.load(ip)

    np.testing.assert_allclose(t.to_home(foreign_eval),
                               t2.to_home(foreign_eval), rtol=1e-9)
    np.testing.assert_allclose(transported_score(instr, t, foreign_eval),
                               transported_score(i2, t2, foreign_eval),
                               rtol=1e-9)
