"""probe_validity — a reusable battery that tells you whether an activation probe tracks the CONCEPT or a
surface artifact ORTHOGONAL to it. The actionable tool from NOTE_probe_orthogonality_2026_06_24.

Why: representation-based AI oversight (deception / truthfulness / harm probes) is validated on constructed
statement sets. A probe can pass high accuracy + cross-domain generalization + a text-silence gate and STILL
be orthogonal to the concept, failing silently on natural inputs. This battery catches that.

General by construction — bring ANY concept, ANY model:
    from probe_validity import validate_probe
    report = validate_probe(
        construct_rows,            # [{text, label, group?}, ...]  the probe's construction/validation set
        natural_rows,              # [{text, label}, ...]          natural held-out examples of the SAME concept
        get_acts,                  # callable(list[str]) -> np.ndarray [n, hidden]  (your model, your layer)
    )
    print(report.summary())

Verdict logic (the battery):
  1. SILENCE   — adversary-fair bag-of-words on the construct (leave-one-group-out) must be ~chance, else the
                 construct isn't even silent (text carries the label).
  2. IN-CONSTRUCT AUC — the probe's headline (cross-group); high here is necessary, NOT sufficient.
  3. NATURAL-OOD TRANSFER — fit the direction on the construct, test on natural held-out, vs a PERMUTATION
                 NULL of shuffled-label directions (the max(AUC,1-AUC) floor is well above 0.5 at small n).
  4. ORTHOGONALITY — cosine(construct direction, natural-data direction). Near 0 = the construct found a
                 surface artifact regardless of its in-construct AUC.
  5. CEILING   — does a direction fit on natural data separate natural held-out? (Confirms the concept axis
                 EXISTS, so a failure is about transfer, not absence.)

VALID requires: transfer significant (perm p < 0.05) AND cosine >= 0.5. Otherwise SURFACE-ARTIFACT / WEAK.
Offline, numpy/sklearn only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


def _ensure_sklearn() -> None:
    """Import the sklearn pieces this module needs INTO module globals, lazily on first use — so plain
    ``import styxx`` never pays for the sklearn/scipy/pandas graph (~2.5s of import time). scikit-learn ships
    in the styxx base install; this defers only WHEN it loads, and names a clear install path if it has been
    stripped from the environment."""
    g = globals()
    if "roc_auc_score" in g:
        return
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, StratifiedKFold
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:  # pragma: no cover
        raise ImportError("validate_probe requires scikit-learn: pip install 'styxx[sklearn]'") from e
    g.update(StandardScaler=StandardScaler, LogisticRegression=LogisticRegression, roc_auc_score=roc_auc_score,
             LeaveOneGroupOut=LeaveOneGroupOut, LeaveOneOut=LeaveOneOut, StratifiedKFold=StratifiedKFold,
             TfidfVectorizer=TfidfVectorizer)


def _fair(a: float) -> float: return max(a, 1.0 - a)


def _massmean(X, y): return X[y == 1].mean(0) - X[y == 0].mean(0)


@dataclass
class ProbeValidityReport:
    silence_auc: float
    in_construct_auc: float
    ood_transfer_auc: float
    ood_transfer_p: float
    orthogonality_cosine: float
    natural_axis_ceiling: float
    n_construct: int
    n_natural: int
    verdict: str
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {k: getattr(self, k) for k in ("silence_auc", "in_construct_auc", "ood_transfer_auc",
                "ood_transfer_p", "orthogonality_cosine", "natural_axis_ceiling", "n_construct",
                "n_natural", "verdict", "notes")}

    def summary(self) -> str:
        L = [
            f"PROBE VALIDITY: {self.verdict}",
            f"  silence (BoW, leave-one-group-out): {self.silence_auc:.3f}   [{'silent' if self.silence_auc<=0.6 else 'NOT silent — text carries the label'}]",
            f"  in-construct AUC (headline):        {self.in_construct_auc:.3f}   [necessary, not sufficient]",
            f"  natural-OOD transfer:               {self.ood_transfer_auc:.3f}  (permutation p={self.ood_transfer_p:.3f})  [{'significant' if self.ood_transfer_p<0.05 else 'NOT above random direction'}]",
            f"  orthogonality cos(construct,natural): {self.orthogonality_cosine:+.3f}   [{'aligned' if self.orthogonality_cosine>=0.5 else 'ORTHOGONAL — surface artifact'}]",
            f"  natural-axis ceiling (concept exists?): {self.natural_axis_ceiling:.3f}",
        ]
        L += [f"  note: {n}" for n in self.notes]
        return "\n".join(L)


def _auc_cv(X, y, groups=None, seed=0):
    aucs = []
    splits = LeaveOneGroupOut().split(X, y, groups) if groups is not None \
        else StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    for tr, te in splits:
        if len(np.unique(y[te])) < 2: continue
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.predict_proba(s.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


def validate_probe(construct_rows, natural_rows, get_acts: Callable,
                   text_key="text", label_key="label", group_key="group",
                   perm_iters: int = 1000, seed: int = 0) -> ProbeValidityReport:
    _ensure_sklearn()
    ct = [r[text_key] for r in construct_rows]; cy = np.array([int(r[label_key]) for r in construct_rows])
    cg = np.array([r.get(group_key, 0) for r in construct_rows]) if any(group_key in r for r in construct_rows) else None
    nt = [r[text_key] for r in natural_rows]; ny = np.array([int(r[label_key]) for r in natural_rows])

    # 1. silence (adversary-fair BoW, grouped if groups given)
    Xb = TfidfVectorizer().fit_transform(ct).toarray()
    sil = _fair(_auc_cv(Xb, cy, cg))

    # activations
    Ac = np.asarray(get_acts(ct), float); An = np.asarray(get_acts(nt), float)
    s = StandardScaler().fit(Ac); C = s.transform(Ac); N = s.transform(An)

    # 2. in-construct AUC (cross-group if groups, else 5-fold)
    inc = _auc_cv(Ac, cy, cg)

    # 3. natural-OOD transfer + permutation null
    d = _massmean(C, cy); obs = _fair(roc_auc_score(ny, N @ d))
    rng = np.random.default_rng(seed)
    null = np.array([_fair(roc_auc_score(ny, N @ _massmean(C, rng.permutation(cy)))) for _ in range(perm_iters)])
    p = float((null >= obs).mean())

    # 4. orthogonality to the natural-data direction
    dn = _massmean(N, ny)
    cos = float(d @ dn / (np.linalg.norm(d) * np.linalg.norm(dn) + 1e-9))

    # 5. ceiling: does a natural-fit direction separate natural held-out? (concept axis exists?)
    oof = np.zeros(len(ny))
    for tr, te in LeaveOneOut().split(N):
        oof[te] = N[te] @ _massmean(N[tr], ny[tr])
    ceil = roc_auc_score(ny, oof)

    notes = []
    if sil > 0.6: notes.append("construct is NOT silent; in-construct AUC may be lexical, not representational.")
    if p >= 0.05 and inc >= 0.8: notes.append("classic surface-artifact: high in-construct AUC, transfer no better than a random direction.")
    if ceil >= 0.75 and cos < 0.5: notes.append("the concept axis EXISTS on natural data but the construct direction misses it (orthogonal) — fit on natural data instead.")

    if p < 0.05 and cos >= 0.5 and inc >= 0.7:
        verdict = "VALID (tracks the concept; transfers + aligned)"
    elif (p >= 0.05 or cos < 0.5) and inc >= 0.7:
        verdict = "SURFACE-ARTIFACT (high in-construct AUC but orthogonal / non-transferring)"
    else:
        verdict = "WEAK/INCONCLUSIVE"

    return ProbeValidityReport(sil, inc, obs, p, cos, float(ceil), len(cy), len(ny), verdict, notes)
