# -*- coding: utf-8 -*-
"""
meaning_integrity.py — the MACHINE-SIDE meaning-integrity monitor (styxx primitive, prototype).

Question it answers, that nothing else does objectively: *Does this model MEAN what a human means?*
Not "is the output fluent" (that's surface) — does the model's internal CONCEPT GEOMETRY match the
human geometry of meaning. Output can look right while meaning is wrong; this reads the meaning.

Core: RSA between a model's concept-geometry and a HUMAN meaning reference (here the 54-feature human
rating space validated against the brain). Built on the cosine-distance RDM, it is — provably —
INVARIANT to meaning-preserving transforms (rotation, isotropic scale, translation) and SENSITIVE to
meaning-destroying ones (noise, concept-shuffle, quantization). That split is the whole point: it tracks
*meaning*, not the surface form of the representation, so it can't be fooled by a re-basis and can't
miss a real corruption.

Usage:
    ref = MeaningReference.from_human_features("human_features.npy")
    rep = integrity_report(model_embeddings, ref)        # -> alignment, status, worst concepts
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _rdm(E):
    """Cosine-distance RDM after mean-centering + L2-normalizing rows.
    => rotation-, isotropic-scale-, and translation-invariant by construction."""
    E = np.asarray(E, float)
    E = E - E.mean(0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return 1.0 - E @ E.T


def _upper(R):
    return R[np.triu_indices(R.shape[0], 1)]


class MeaningReference:
    """A human meaning anchor: a concept geometry derived from human judgments (not text)."""

    def __init__(self, ref_matrix, words=None, name="human"):
        self.R = _rdm(ref_matrix)
        self.v = _upper(self.R)
        self.words = list(words) if words is not None else None
        self.name = name
        self.n = self.R.shape[0]

    @classmethod
    def from_human_features(cls, path=None, words=None):
        path = path or os.path.join(HERE, "human_features.npy")
        return cls(np.load(path), words=words, name="human54")


def alignment(E, ref, control=None):
    """RSA(model geometry, human reference) in [-1, 1]. Higher = more human-aligned meaning.
    Rotation/scale/translation invariant. `control` (optional, per-concept vector e.g. word length)
    is partialled out of both RDMs first."""
    g = _upper(_rdm(E))
    rv = ref.v
    if control is not None:
        c = np.asarray(control, float)
        L = np.abs(c[:, None] - c[None, :])[np.triu_indices(len(c), 1)]
        X = np.column_stack([np.ones(len(L)), L])
        g = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
        rv = rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]
    return float(np.corrcoef(g, rv)[0, 1])


def dispersion(E):
    """Absolute spread of the representation = mean centered row-norm. Scale-DEPENDENT on purpose:
    the angular `alignment` is invariant to isotropic scale (a strength for basis-robustness), which
    makes it BLIND to a uniform collapse toward the mean (a model losing all contrast equally). In
    vital-sign mode — compared to the model's own baseline — a falling dispersion ratio catches exactly
    that failure. The complete monitor reads BOTH channels: alignment (structure) + dispersion (magnitude)."""
    E = np.asarray(E, float)
    E = E - E.mean(0)
    return float(np.mean(np.linalg.norm(E, axis=1)))


def per_concept_alignment(E, ref):
    """For each concept, how well its *relational profile* (its row of the RDM) matches the human one.
    Low value => that concept is misrepresented by the model. Localizes WHERE meaning breaks."""
    Rm = _rdm(E)
    Rh = ref.R
    n = Rm.shape[0]
    out = np.empty(n)
    for i in range(n):
        m = np.delete(Rm[i], i)
        h = np.delete(Rh[i], i)
        out[i] = np.corrcoef(m, h)[0, 1]
    return out


def integrity_report(E, ref, healthy=0.25, broken=0.10, words=None, top=10):
    """Full monitor read: global alignment, a HEALTHY/DEGRADED/BROKEN status band, and the concepts
    whose meaning diverges most from human (the localized failures)."""
    a = alignment(E, ref)
    pc = per_concept_alignment(E, ref)
    status = "HEALTHY" if a >= healthy else ("BROKEN" if a < broken else "DEGRADED")
    order = np.argsort(pc)
    w = words or ref.words
    worst = [(int(i), (w[i] if w else int(i)), round(float(pc[i]), 3)) for i in order[:top]]
    return {"alignment": round(a, 4), "status": status,
            "per_concept_mean": round(float(np.mean(pc)), 4),
            "worst_concepts": worst, "per_concept": pc}


class MeaningVitalSign:
    """Deployable TWO-CHANNEL meaning-integrity monitor. Calibrate once on a healthy model, then
    `check()` on a schedule — a *vital sign* for a model's meaning. Reads BOTH channels:
      - alignment      (structure; basis-invariant; catches drift / poisoning / plausible-but-wrong)
      - dispersion_ratio (magnitude vs baseline; catches the uniform collapse alignment is blind to)
    and returns a HEALTHY / DEGRADED / BROKEN verdict plus the worst concepts."""

    def __init__(self, ref, healthy_align=0.25, broken_align=0.10, min_dispersion_ratio=0.70):
        self.ref = ref
        self.healthy_align = healthy_align
        self.broken_align = broken_align
        self.min_dispersion_ratio = min_dispersion_ratio
        self.base_align = None
        self.base_disp = None

    def calibrate(self, E_healthy):
        self.base_align = alignment(E_healthy, self.ref)
        self.base_disp = dispersion(E_healthy)
        return self

    def check(self, E, words=None, top=8):
        a = alignment(E, self.ref)
        dr = (dispersion(E) / self.base_disp) if self.base_disp else float("nan")
        if a < self.broken_align:
            status = "BROKEN"
        elif a < self.healthy_align or (dr == dr and dr < self.min_dispersion_ratio):
            status = "DEGRADED"
        else:
            status = "HEALTHY"
        pc = per_concept_alignment(E, self.ref)
        w = words or self.ref.words
        worst = [(w[i] if w else int(i)) for i in np.argsort(pc)[:top]]
        out = {"alignment": round(a, 4), "dispersion_ratio": round(dr, 4),
               "status": status, "worst_concepts": worst}
        if self.base_align is not None:
            out["align_drop_vs_baseline"] = round(self.base_align - a, 4)
        return out
