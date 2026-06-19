"""styxx.meaning_diff — the meaning-regression instrument.

Did two models MEAN the same thing? Given two models' concept representations, report an agreement
score, a HEALTHY/DRIFTED/BROKEN verdict, the concepts that diverge most, and a reliability flag that
says when NOT to trust the comparison. Norm-equalized by default (the validated convention that makes
the number trustworthy — see papers/mind-instrument/FINDING_anatomy_v2_2026_06_10.md; the
unweighted-average convention understated cross-model agreement 2-60x).

Pure stdlib + numpy. No torch import at module load — ships in the core wheel; the heavy
template-extraction stays caller-side.

    from styxx.meaning_diff import meaning_diff
    r = meaning_diff(model_a_reps, model_b_reps, words=concepts)
    r.agreement               # 0..1 geometry agreement   (or r["agreement"])
    r.verdict                 # HEALTHY / DRIFTED / BROKEN
    r.divergent_concepts      # [(word, score), ...] what moved most
    r.to_dict()               # the plain dict, for logging / JSON

``meaning_diff`` returns a typed :class:`MeaningDiff` (like every other headline
instrument). It stays dict-compatible — ``r["agreement"]``, ``dict(r)``, ``k in r``
and iteration all still work — so callers written against the pre-7.17.4 dict are
untouched.

Validated port: papers/mind-instrument/PREREG_meaning_diff_v0_2026_06_10.md (gates D1-D5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["meaning_diff", "meaning_diff_templates", "rdm", "VERDICT_BANDS", "MeaningDiff"]

# Frozen verdict bands (heuristics calibrated to the program's receipts; disclosed, not universal).
VERDICT_BANDS = {"HEALTHY": 0.80, "DRIFTED": 0.50}   # >=0.80 healthy; >=0.50 drifted; else broken


@dataclass(frozen=True)
class MeaningDiff:
    """Typed result of :func:`meaning_diff` — the meaning-regression readout.

    Every other headline styxx instrument returns a typed dataclass; this brings
    meaning_diff in line. It stays **fully back-compatible** with the plain dict
    it used to return: ``r["agreement"]``, ``r.get(...)``, ``k in r``, iteration,
    ``dict(r)`` and ``.to_dict()`` all still work, now alongside attribute access
    (``r.agreement``).
    """

    agreement: float
    verdict: str
    divergent_concepts: List[Tuple[Any, float]]
    n_concepts: int
    reliability: Optional[float]
    reliable: bool
    reliability_caveat: Optional[str]
    norm_equalized: bool
    verdict_bands: Dict[str, float]

    def to_dict(self) -> dict:
        """The exact dict the instrument used to return (a fresh copy)."""
        return {
            "agreement": self.agreement,
            "verdict": self.verdict,
            "divergent_concepts": self.divergent_concepts,
            "n_concepts": self.n_concepts,
            "reliability": self.reliability,
            "reliable": self.reliable,
            "reliability_caveat": self.reliability_caveat,
            "norm_equalized": self.norm_equalized,
            "verdict_bands": self.verdict_bands,
        }

    # --- dict back-compat: the instrument shipped a plain dict through 7.17.3 ---
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())


def rdm(R: np.ndarray, *, norm_equalized: bool = True) -> np.ndarray:
    """RDM of a (concepts x dims) representation matrix.

    norm_equalized=True (default): the program's `distmat` exactly — center, per-row normalize,
    cosine-distance geometry. Per-concept scale is removed, so a few high-norm concepts cannot
    dominate. (The artifact tonight's anatomy arc fixed lives one stage earlier, at TEMPLATE
    averaging — handled by ``meaning_diff_templates``; here the per-concept normalization is intrinsic
    to the cosine RDM.)
    norm_equalized=False: centered Euclidean geometry WITHOUT per-row normalization — norm-sensitive,
    exposed so callers can see what scale domination does to the comparison.
    """
    R = np.asarray(R, dtype=float)
    R = R - R.mean(0)
    if norm_equalized:
        R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        G = R @ R.T
        return np.sqrt(np.maximum(2.0 - 2.0 * G, 0.0))
    G = R @ R.T
    nrm = np.diag(G)
    return np.sqrt(np.maximum(nrm[:, None] + nrm[None, :] - 2.0 * G, 0.0))


def _verdict(agreement: float) -> str:
    if agreement >= VERDICT_BANDS["HEALTHY"]:
        return "HEALTHY"
    if agreement >= VERDICT_BANDS["DRIFTED"]:
        return "DRIFTED"
    return "BROKEN"


def meaning_diff(reps_a, reps_b, *, words=None, top_k: int = 10,
                 reliability: float | None = None, norm_equalized: bool = True) -> MeaningDiff:
    """Compare two models' concept geometries. reps_a/reps_b: (N, d_a)/(N, d_b), rows aligned by
    concept (same order). Returns a :class:`MeaningDiff` (dict-compatible) with agreement, verdict,
    divergent_concepts, and the reliability flag."""
    Da, Db = rdm(reps_a, norm_equalized=norm_equalized), rdm(reps_b, norm_equalized=norm_equalized)
    if Da.shape != Db.shape:
        raise ValueError(f"concept counts differ: {Da.shape[0]} vs {Db.shape[0]} (rows must align)")
    n = Da.shape[0]
    iu = np.triu_indices(n, 1)
    a, b = Da[iu], Db[iu]
    agreement = float(np.corrcoef(a, b)[0, 1]) if n > 2 else 1.0
    agreement = max(0.0, agreement)

    # per-concept divergence: 1 - correlation of each concept's distance row across the two models
    div = []
    for i in range(n):
        ra = np.delete(Da[i], i)
        rb = np.delete(Db[i], i)
        c = float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 1 else 1.0
        div.append(1.0 - max(-1.0, min(1.0, c)))
    order = np.argsort(div)[::-1]
    labels = list(words) if words is not None else list(range(n))
    # self-comparison -> no false divergence (D2): identical RDMs give ~0 everywhere
    divergent = [(labels[i], round(div[i], 4)) for i in order[:top_k] if div[i] > 1e-9]

    reliable = (reliability is None) or (reliability >= 0.5)
    return MeaningDiff(
        agreement=round(agreement, 4),
        verdict=_verdict(agreement),
        divergent_concepts=divergent,
        n_concepts=int(n),
        reliability=(round(reliability, 4) if reliability is not None else None),
        reliable=bool(reliable),
        reliability_caveat=(None if reliability is not None else
                            "no template reps supplied; reliability unmeasured — treat agreement "
                            "as indicative, not certified"),
        norm_equalized=norm_equalized,
        verdict_bands=dict(VERDICT_BANDS),
    )


def _sb_split_half_reliability(template_reps: np.ndarray) -> float:
    """Spearman-Brown split-half (odd/even templates) reliability of the battery geometry —
    the frozen anatomy-v2 convention (per-template L2 normalize, average each half)."""
    T = np.asarray(template_reps, dtype=float)              # (T, N, d)
    Tn = T / (np.linalg.norm(T, axis=2, keepdims=True) + 1e-9)
    A, B = Tn[0::2].mean(0), Tn[1::2].mean(0)
    Da, Db = rdm(A), rdm(B)                                  # cosine RDM (anatomy-v2 convention)
    iu = np.triu_indices(Da.shape[0], 1)
    r = float(np.corrcoef(Da[iu], Db[iu])[0, 1])
    return max(0.0, 2 * r / (1 + r)) if r > 0 else 0.0


def meaning_diff_templates(template_reps_a, template_reps_b, *, words=None, top_k: int = 10) -> MeaningDiff:
    """Caller supplies (T, N, d) per-template reps for each model; applies the frozen
    per-template-L2-then-average convention, measures split-half reliability, and compares.
    The recommended entry point when raw template states are available."""
    Ta = np.asarray(template_reps_a, dtype=float)
    Tb = np.asarray(template_reps_b, dtype=float)
    def avg(T):
        Tn = T / (np.linalg.norm(T, axis=2, keepdims=True) + 1e-9)
        return Tn.mean(0)
    rel = min(_sb_split_half_reliability(Ta), _sb_split_half_reliability(Tb))
    return meaning_diff(avg(Ta), avg(Tb), words=words, top_k=top_k,
                        reliability=rel, norm_equalized=False)
