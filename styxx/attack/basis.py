# -*- coding: utf-8 -*-
"""
styxx.attack.basis — discover the true latent dimensionality of
cognometric measurement.

If the K=1 phase-transition signature implies that each calibrated
instrument is dominated by one feature, AND those features rely on
overlapping surface patterns (the rc2 non-orthogonality finding),
then the rank of the cross-instrument fingerprint matrix is strictly
LESS than the number of registered instruments. This module computes
that rank — and the principal components.

Public API
----------
    from styxx.attack import cognometric_basis

    result = cognometric_basis(samples)
    result.rank95     # number of components needed for >= 95% variance
    result.evr        # explained variance ratio per PC
    result.loadings   # (n_pc x n_instrument) — what each PC weights
    result.embed      # (n_samples x n_pc) — samples in cognometric basis

Why it matters
--------------
If rank95 < n_instruments, you've discovered that styxx is not
measuring n_instruments independent things — it's measuring rank95
latent factors. That has direct consequences for any downstream
ensembling, alerting threshold design, and instrument-design roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fingerprint import score_all, applicable_instruments


@dataclass
class BasisResult:
    """Output of cognometric_basis().

    Attributes:
        instruments:    column names of the analyzed fingerprint matrix
                        (in the order the matrix is constructed)
        n_samples:      rows in the analyzed matrix (after NaN drops)
        evr:            explained variance ratio per principal component
        cumulative_evr: cumulative EVR
        loadings:       (n_pc, n_instrument) — PC × instrument weights
                        rows are normalized eigenvectors of covariance
        embed:          (n_samples, n_pc) — sample projections into PC basis
        rank95:         smallest k such that cumulative_evr[k-1] >= 0.95
        rank99:         analogous, 99% variance threshold
        means:          per-instrument means used for centering
        stds:           per-instrument std used for scaling
    """
    instruments: List[str]
    n_samples: int
    evr: List[float]
    cumulative_evr: List[float]
    loadings: List[List[float]]
    embed: List[List[float]]
    rank95: int
    rank99: int
    means: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instruments": self.instruments,
            "n_samples": self.n_samples,
            "evr": list(self.evr),
            "cumulative_evr": list(self.cumulative_evr),
            "loadings": [list(row) for row in self.loadings],
            "rank95": self.rank95,
            "rank99": self.rank99,
            "means": list(self.means),
            "stds": list(self.stds),
        }

    def __repr__(self) -> str:
        return (
            f"BasisResult(n={self.n_samples}, "
            f"instruments={self.instruments}, "
            f"rank95={self.rank95}/{len(self.instruments)}, "
            f"rank99={self.rank99}/{len(self.instruments)}, "
            f"evr=[{', '.join(f'{v:.3f}' for v in self.evr)}])"
        )


def cognometric_basis(
    samples: Sequence[Dict[str, Any]],
    instruments: Optional[Sequence[str]] = None,
    standardize: bool = True,
) -> BasisResult:
    """PCA over the cross-instrument fingerprint matrix.

    Args:
        samples:      list of input dicts (kwargs to score_all).
        instruments:  optional explicit instrument list. If None, the
                      union of instruments that fired on the FIRST sample
                      is used (so column ordering is deterministic).
        standardize:  if True, z-score each column before SVD (default).

    Returns:
        BasisResult with EVR, loadings, embedding, and rank95/99.

    Raises:
        ValueError: if no samples produced a valid fingerprint or fewer
                    than 2 instruments scored.
    """
    if not samples:
        raise ValueError("cognometric_basis: empty samples")

    # Build the fingerprint matrix. Skip rows with any NaN (instruments
    # that didn't fire). For mixed-shape inputs, restrict to the columns
    # that fired on >=80% of rows, then drop the remaining NaN rows.
    fingerprints: List[Dict[str, float]] = []
    for s in samples:
        try:
            fp = score_all(**s)
        except Exception:
            continue
        if fp:
            fingerprints.append(fp)

    if not fingerprints:
        raise ValueError("cognometric_basis: zero scoreable samples")

    if instruments is None:
        # Use instruments that fired on >= 80% of the fingerprints
        coverage: Dict[str, int] = {}
        for fp in fingerprints:
            for k in fp:
                coverage[k] = coverage.get(k, 0) + 1
        thresh = max(1, int(0.8 * len(fingerprints)))
        instruments = sorted(k for k, c in coverage.items() if c >= thresh)
    instruments = list(instruments)

    if len(instruments) < 2:
        raise ValueError(
            f"cognometric_basis: need >=2 instruments, got {instruments}"
        )

    # Build matrix; drop rows missing any of the chosen instruments.
    rows: List[List[float]] = []
    for fp in fingerprints:
        row = [fp.get(i) for i in instruments]
        if all(v is not None for v in row):
            rows.append([float(v) for v in row])
    if not rows:
        raise ValueError(
            f"cognometric_basis: zero complete rows for instruments={instruments}"
        )
    X = np.array(rows, dtype=float)
    n, d = X.shape

    means = X.mean(axis=0)
    Xc = X - means
    if standardize:
        stds = X.std(axis=0, ddof=1) if n > 1 else np.ones(d)
        # Avoid div-by-zero on a constant column
        stds = np.where(stds == 0, 1.0, stds)
        Xs = Xc / stds
    else:
        stds = np.ones(d)
        Xs = Xc

    # SVD-based PCA — robust on small n
    # Xs (n x d) = U S V^T. PCs are columns of V; singular_values^2/(n-1) = eigenvalues.
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    # eigenvalues = S^2 / (n-1); EVR = eigenvalue / sum(eigenvalues)
    eigenvalues = (S ** 2) / max(n - 1, 1)
    total = float(eigenvalues.sum()) if eigenvalues.sum() > 0 else 1.0
    evr = (eigenvalues / total).tolist()
    cumulative_evr: List[float] = []
    running = 0.0
    for v in evr:
        running += v
        cumulative_evr.append(running)

    rank95 = next(
        (i + 1 for i, c in enumerate(cumulative_evr) if c >= 0.95),
        len(evr),
    )
    rank99 = next(
        (i + 1 for i, c in enumerate(cumulative_evr) if c >= 0.99),
        len(evr),
    )

    # Loadings: rows are PCs, cols are instruments. Vt is already shape (d, d).
    loadings = Vt.tolist()
    # Embed: U * S projects rows into PC space (n x d).
    embed = (U * S).tolist()

    return BasisResult(
        instruments=instruments,
        n_samples=n,
        evr=evr,
        cumulative_evr=cumulative_evr,
        loadings=loadings,
        embed=embed,
        rank95=rank95,
        rank99=rank99,
        means=means.tolist(),
        stds=stds.tolist(),
    )


__all__ = ["cognometric_basis", "BasisResult"]
