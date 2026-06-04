# -*- coding: utf-8 -*-
"""
styxx.meaning_integrity — does a model MEAN what a human means?

A model can produce fluent, plausible output while its internal *understanding* is wrong or degraded.
This primitive reads the meaning behind the output: it compares a model's **concept geometry** (the
pairwise structure of how it represents a set of concepts) to a **human meaning reference** (a
concept x human-feature matrix), and reports an alignment score, a HEALTHY/DEGRADED/BROKEN verdict, and
*which* concepts diverge most.

It is built on a mean-centered, L2-normalized cosine-distance RDM, so the alignment channel is — provably —
invariant to rotation, isotropic scale, and translation of the representation (it reads *meaning*, not the
surface basis), and sensitive to anything that moves the relational structure. A second `dispersion`
channel (scale-*dependent*) catches the one failure the angular channel is blind to: a uniform collapse
toward the mean. The deployable `MeaningVitalSign` reads both.

Validated (see papers/ai-human-alignment): invariant to 1e-16; catches plausible-but-wrong corruption
that output-inspection misses; localizes which concepts broke (ROC-AUC ~0.95); generalizes across
languages/references; and catches REAL fine-tuning damage while passing benign fine-tuning.

Bring your own human reference — a (n_concepts x n_features) matrix of human judgments. Good public
sources: Binder et al. 2016 (65 experiential features, English), the Lancaster Sensorimotor Norms
(11 dims, English), or the ds004301 54-feature space (Chinese). The reference should be RICH (many
dims, perceptual + abstract); thin/perceptual-only references give weak discrimination.

    import numpy as np
    from styxx.meaning_integrity import MeaningReference, MeaningVitalSign

    ref = MeaningReference(human_features, words=concept_words)   # human judgments, (N, F)
    vs  = MeaningVitalSign(ref).calibrate(healthy_model_embeddings)   # (N, D) for the SAME concepts
    print(vs.check(current_model_embeddings))   # -> alignment, dispersion_ratio, status, worst_concepts
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "MeaningReference", "MeaningVitalSign",
    "meaning_alignment", "meaning_dispersion", "per_concept_alignment", "meaning_integrity_report",
    "meaning_agreement",
]


def _rdm(E):
    """Cosine-distance RDM after mean-centering + L2-normalizing rows.
    => rotation-, isotropic-scale-, and translation-invariant by construction."""
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError("embeddings must be a 2-D array (n_concepts, n_dims)")
    E = E - E.mean(0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return 1.0 - E @ E.T


def _upper(R):
    return R[np.triu_indices(R.shape[0], 1)]


class MeaningReference:
    """A human meaning anchor: a concept geometry derived from human judgments (a concept x feature
    matrix), not from any text model. Pass the same concept set's model embeddings to score alignment."""

    def __init__(self, features, words=None, name="human"):
        F = np.asarray(features, dtype=float)
        if F.ndim != 2:
            raise ValueError("features must be a 2-D array (n_concepts, n_features)")
        self.R = _rdm(F)
        self.v = _upper(self.R)
        self.words = list(words) if words is not None else None
        self.name = name
        self.n = self.R.shape[0]
        if self.words is not None and len(self.words) != self.n:
            raise ValueError(f"words ({len(self.words)}) must match n_concepts ({self.n})")

    def __repr__(self):
        return f"MeaningReference(name={self.name!r}, n_concepts={self.n})"


def _check_shape(E, ref):
    E = np.asarray(E, dtype=float)
    if E.shape[0] != ref.n:
        raise ValueError(f"embeddings have {E.shape[0]} concepts; reference has {ref.n}. "
                         "They must describe the SAME concepts, in the same order.")
    return E


def meaning_alignment(embeddings, reference, control=None):
    """RSA between a model's concept geometry and a human meaning reference, in [-1, 1]. Higher = more
    human-aligned. Invariant to rotation/scale/translation of the embeddings. `control` (optional,
    per-concept vector e.g. word length) is partialled out of both geometries first."""
    E = _check_shape(embeddings, reference)
    g = _upper(_rdm(E))
    rv = reference.v
    if control is not None:
        c = np.asarray(control, dtype=float)
        L = np.abs(c[:, None] - c[None, :])[np.triu_indices(len(c), 1)]
        X = np.column_stack([np.ones(len(L)), L])
        g = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
        rv = rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]
    return float(np.corrcoef(g, rv)[0, 1])


def meaning_dispersion(embeddings):
    """Absolute spread of the representation (mean centered row-norm). Scale-DEPENDENT on purpose: the
    angular `meaning_alignment` is scale-invariant, so it is blind to a uniform collapse toward the mean;
    a falling dispersion (vs a healthy baseline) catches exactly that. The full monitor reads both."""
    E = np.asarray(embeddings, dtype=float)
    E = E - E.mean(0)
    return float(np.mean(np.linalg.norm(E, axis=1)))


def per_concept_alignment(embeddings, reference):
    """For each concept, how well its relational profile (its row of the RDM) matches the human one.
    Low value => that concept is misrepresented by the model. Localizes WHERE meaning breaks."""
    E = _check_shape(embeddings, reference)
    Rm, Rh = _rdm(E), reference.R
    n = Rm.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.corrcoef(np.delete(Rm[i], i), np.delete(Rh[i], i))[0, 1]
    return out


def meaning_integrity_report(embeddings, reference, healthy=0.25, broken=0.10, words=None, top=10):
    """One-shot read (uncalibrated): global alignment, an absolute HEALTHY/DEGRADED/BROKEN band, and the
    concepts whose meaning diverges most from human. For ongoing monitoring use `MeaningVitalSign`."""
    a = meaning_alignment(embeddings, reference)
    pc = per_concept_alignment(embeddings, reference)
    status = "HEALTHY" if a >= healthy else ("BROKEN" if a < broken else "DEGRADED")
    w = words or reference.words
    worst = [(int(i), (w[i] if w else int(i)), round(float(pc[i]), 3)) for i in np.argsort(pc)[:top]]
    return {"alignment": round(a, 4), "status": status,
            "per_concept_mean": round(float(np.mean(pc)), 4), "worst_concepts": worst}


def meaning_agreement(embeddings_a, embeddings_b, words=None, top=10):
    """Reference-FREE: do two models MEAN the same? Compares model A's concept geometry to model B's
    (B as the reference) over the SAME concepts, and names which concepts the two represent most
    differently. No human reference needed. Use for model migration / distillation / quantization QA:
    *did the new model keep the meaning of the old one, and if not, which concepts did it lose?*"""
    ref = MeaningReference(embeddings_b, words=words, name="model_b")
    a = meaning_alignment(embeddings_a, ref)
    pc = per_concept_alignment(embeddings_a, ref)
    worst = [((words[i] if words else int(i)), round(float(pc[i]), 3)) for i in np.argsort(pc)[:top]]
    return {"agreement": round(a, 4), "most_divergent_concepts": worst}


class MeaningVitalSign:
    """Deployable TWO-CHANNEL monitor. Calibrate once on a healthy model, then `check()` on a schedule —
    a *vital sign* for a model's meaning. Verdicts are judged RELATIVE to the calibrated healthy baseline
    (retain < `degraded_frac` of baseline alignment -> DEGRADED, < `broken_frac` -> BROKEN), because
    different embedding-extraction methods have different natural scales. The dispersion channel adds a
    floor that catches uniform collapse the angular channel is blind to."""

    def __init__(self, reference, degraded_frac=0.70, broken_frac=0.40, min_dispersion_ratio=0.70,
                 healthy_align=0.25, broken_align=0.10):
        self.ref = reference
        self.degraded_frac = degraded_frac
        self.broken_frac = broken_frac
        self.min_dispersion_ratio = min_dispersion_ratio
        self.healthy_align = healthy_align          # absolute fallback if uncalibrated
        self.broken_align = broken_align
        self.base_align = None
        self.base_disp = None

    def calibrate(self, healthy_embeddings):
        self.base_align = meaning_alignment(healthy_embeddings, self.ref)
        self.base_disp = meaning_dispersion(healthy_embeddings)
        return self

    def check(self, embeddings, words=None, top=8):
        a = meaning_alignment(embeddings, self.ref)
        dr = (meaning_dispersion(embeddings) / self.base_disp) if self.base_disp else float("nan")
        if self.base_align:
            frac = a / self.base_align
            if frac < self.broken_frac:
                status = "BROKEN"
            elif frac < self.degraded_frac or (dr == dr and dr < self.min_dispersion_ratio):
                status = "DEGRADED"
            else:
                status = "HEALTHY"
        else:
            status = "BROKEN" if a < self.broken_align else ("HEALTHY" if a >= self.healthy_align else "DEGRADED")
        pc = per_concept_alignment(embeddings, self.ref)
        w = words or self.ref.words
        worst = [(w[i] if w else int(i)) for i in np.argsort(pc)[:top]]
        out = {"alignment": round(a, 4), "dispersion_ratio": round(dr, 4), "status": status,
               "worst_concepts": worst}
        if self.base_align:
            out["frac_of_baseline"] = round(a / self.base_align, 3)
        return out
