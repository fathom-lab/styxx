# -*- coding: utf-8 -*-
"""
styxx.transport — universal cognometric transport.

Fit a cognometric instrument ONCE in a home embedding space, then move
it into a *different* embedding space — including closed models you can
only embed through, and entirely different model families — with **no
behavior labels, no model weights, and no retraining**.

The only thing you need is a generic corpus embedded through both
encoders (the *same* sentences, two spaces). That corpus carries no
labels and requires no cooperation from the model owner — you just have
to be able to embed text. From it, `Transport` learns a single linear
map foreign -> home, and the home instrument scores foreign-space
embeddings directly.

Validated (2026-05-17 dogfood, refusal instrument, te3-large home):
  - paired-procrustes: AUC 1.000 on clear cases, 0.885-0.935 vs live
    gpt-4o-mini / gpt-4.1-mini refusal — including cross-family
    transport into all-mpnet-base-v2 (768d, different objective).
  - paired-ridge: AUC 1.000 clear, 0.860-0.890 live.
  - naive direct transfer (no transport): 0.30-0.59 (<= random
    cross-family) — the transport is doing the work.

Documented boundary (do not market past this):
  ZERO-paired-data transport is OUT OF SCOPE. Two principled attempts
  (NN-Procrustes proxy and a full CSLS/MUSE unsupervised pipeline)
  failed on 2026-05-17 (~0.60 AUC). Unsupervised cognometric transport
  needs vec2vec-grade nonlinear/adversarial machinery and large
  in-distribution corpora — a separate research bet, not this module.
  `Transport.fit` therefore requires a *paired* (same-sentence) corpus
  and asserts it.

Why a linear map: vec2vec (NeurIPS 2025) / mini-vec2vec (arXiv
2510.02348) show the map between embedding spaces is approximately
linear. styxx confirms it carries *cognometric directions*, not just
retrieval geometry — when the corpus is paired.

7.5.0+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

_PathLike = Union[str, Path]


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# ─────────────────────────────────────────────────────────────────────
# The instrument: a linear cognometric axis in some embedding space
# ─────────────────────────────────────────────────────────────────────
@dataclass
class CognometricInstrument:
    """A calibrated linear axis (diff-of-means) in one embedding space.

    Fit it in whatever representation you will score in: raw home space
    for ``method="ridge"``, or ``Transport.home_repr(...)`` output for
    ``method="procrustes"``. Calibration (the sigmoid mid/scale) is
    computed in that same space, so it must match at score time.
    """

    axis: np.ndarray
    mid: float
    scale: float

    @classmethod
    def from_labeled(
        cls,
        embeddings: np.ndarray,
        labels: Sequence[float],
        *,
        positive: float = 1.0,
        negative: float = 0.0,
    ) -> "CognometricInstrument":
        """Fit by difference-of-means between the positive and negative
        labeled embeddings. Calibrates a sigmoid on the labeled set."""
        emb = _l2(embeddings)
        y = np.asarray(labels, dtype=np.float64)
        pos = emb[y == positive]
        neg = emb[y == negative]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError(
                "need both positive and negative labeled embeddings"
            )
        axis = pos.mean(0) - neg.mean(0)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        proj = emb @ axis
        anchor = proj[(y == positive) | (y == negative)]
        mid = float((anchor.max() + anchor.min()) / 2.0)
        scale = float(max((anchor.max() - anchor.min()) / 2.0, 1e-9))
        return cls(axis=axis, mid=mid, scale=scale)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return p in (0, 1) for each embedding (higher = positive)."""
        emb = _l2(embeddings)
        return 1.0 / (1.0 + np.exp(-(emb @ self.axis - self.mid)
                                   / (self.scale * 0.5)))

    def save(self, path: _PathLike) -> Path:
        path = Path(path)
        np.savez(path, axis=self.axis,
                 mid=np.array(self.mid), scale=np.array(self.scale))
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: _PathLike) -> "CognometricInstrument":
        d = np.load(Path(path))
        return cls(axis=d["axis"], mid=float(d["mid"]), scale=float(d["scale"]))


# ─────────────────────────────────────────────────────────────────────
# The transport: a label-free linear map foreign -> home
# ─────────────────────────────────────────────────────────────────────
@dataclass
class TransportReport:
    method: str
    home_dim: int
    foreign_dim: int
    n_corpus: int
    k: Optional[int] = None

    def __repr__(self) -> str:
        kf = f", k={self.k}" if self.k is not None else ""
        return (f"<TransportReport {self.method} "
                f"{self.foreign_dim}->{self.home_dim} "
                f"n={self.n_corpus}{kf}>")


class Transport:
    """A learned linear map from a foreign embedding space into a home
    embedding space, fit from a paired but label-free corpus.

    Usage::

        t = Transport.fit(home_corpus_emb, foreign_corpus_emb,
                           method="procrustes")
        instr = CognometricInstrument.from_labeled(
            t.home_repr(home_labeled_emb), labels)
        p = transported_score(instr, t, foreign_emb)
    """

    def __init__(self, method: str, report: TransportReport, **state):
        self.method = method
        self.report = report
        self._state = state

    # -- fitting --------------------------------------------------------
    @classmethod
    def fit(
        cls,
        home_corpus: np.ndarray,
        foreign_corpus: np.ndarray,
        *,
        method: str = "procrustes",
        k: int = 256,
        ridge_lambda: float = 1.0,
    ) -> "Transport":
        """Learn the map. ``home_corpus[i]`` and ``foreign_corpus[i]``
        must be the SAME sentence embedded in the two spaces (paired,
        no labels). Zero-paired-data is intentionally unsupported — see
        the module docstring."""
        H = _l2(home_corpus)
        F = _l2(foreign_corpus)
        if H.shape[0] != F.shape[0]:
            raise ValueError(
                "Transport.fit requires a PAIRED corpus: home_corpus and "
                "foreign_corpus must have the same number of rows (the "
                "same sentences embedded in both spaces). Zero-paired-data "
                "transport is a closed negative — see styxx.transport docs."
            )
        if H.shape[0] < 8:
            raise ValueError("need at least 8 corpus pairs")

        rep = TransportReport(method=method, home_dim=H.shape[1],
                              foreign_dim=F.shape[1], n_corpus=H.shape[0])

        if method == "ridge":
            d = F.shape[1]
            W = np.linalg.solve(F.T @ F + ridge_lambda * np.eye(d), F.T @ H)
            return cls("ridge", rep, W=W)

        if method == "procrustes":
            kk = int(min(k, H.shape[1], F.shape[1], H.shape[0] - 1))
            rep.k = kk
            h_mu = H.mean(0)
            f_mu = F.mean(0)
            _, _, Vh = np.linalg.svd(H - h_mu, full_matrices=False)
            _, _, Vf = np.linalg.svd(F - f_mu, full_matrices=False)
            Hc = Vh[:kk]
            Fc = Vf[:kk]
            Hk = _l2((H - h_mu) @ Hc.T)
            Fk = _l2((F - f_mu) @ Fc.T)
            U, _, Vt = np.linalg.svd(Fk.T @ Hk, full_matrices=False)
            R = U @ Vt  # foreign-PCA -> home-PCA, orthogonal
            return cls("procrustes", rep, h_mu=h_mu, f_mu=f_mu,
                       Hc=Hc, Fc=Fc, R=R)

        raise ValueError(f"unknown method {method!r} "
                         f"(use 'procrustes' or 'ridge')")

    # -- applying -------------------------------------------------------
    def home_repr(self, home_emb: np.ndarray) -> np.ndarray:
        """Project home-space embeddings into the representation the
        instrument must be fit and scored in."""
        H = _l2(home_emb)
        if self.method == "ridge":
            return H
        s = self._state
        return _l2((H - s["h_mu"]) @ s["Hc"].T)

    def to_home(self, foreign_emb: np.ndarray) -> np.ndarray:
        """Map foreign-space embeddings into the home representation."""
        F = _l2(foreign_emb)
        if self.method == "ridge":
            return _l2(F @ self._state["W"])
        s = self._state
        return _l2(((F - s["f_mu"]) @ s["Fc"].T) @ s["R"])

    # -- persistence ----------------------------------------------------
    def save(self, path: _PathLike) -> Path:
        path = Path(path)
        arrays = {f"_{key}": np.asarray(val)
                  for key, val in self._state.items()}
        np.savez(path, _method=np.array(self.method),
                 _meta=np.array([self.report.home_dim,
                                 self.report.foreign_dim,
                                 self.report.n_corpus,
                                 -1 if self.report.k is None
                                 else self.report.k]),
                 **arrays)
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: _PathLike) -> "Transport":
        d = np.load(Path(path), allow_pickle=False)
        method = str(d["_method"])
        hd, fd, n, k = (int(v) for v in d["_meta"])
        rep = TransportReport(method=method, home_dim=hd, foreign_dim=fd,
                              n_corpus=n, k=None if k < 0 else k)
        state = {key[1:]: d[key] for key in d.files
                 if key.startswith("_") and key not in ("_method", "_meta")}
        return cls(method, rep, **state)


def transported_score(
    instrument: CognometricInstrument,
    transport: Transport,
    foreign_emb: np.ndarray,
) -> np.ndarray:
    """Score foreign-space embeddings with a home-space instrument via a
    fitted transport. The instrument must have been fit on
    ``transport.home_repr(home_labeled_emb)``."""
    return instrument.score(transport.to_home(foreign_emb))
