# -*- coding: utf-8 -*-
"""
styxx.thought — Thought as a portable, substrate-independent data type.

    PNG is the format for images.
    JSON is the format for data.
    .fathom is the format for thoughts.

A Thought is the cognitive content of a generation, captured as a
trajectory of category-probability vectors over the four atlas phases
(preflight, early, mid, late). It is *substrate-independent*: the
representation lives in fathom's calibrated cognitive eigenvalue space,
not in any specific model's internal activations. The same Thought can
be read out of one model and written back through another.

This module introduces:

  - ``Thought``        : the data type
  - ``ThoughtDelta``   : the difference between two Thoughts
  - ``read_thought()`` : extract a Thought from text by running it
                         through a styxx-instrumented model
  - ``write_thought()``: render a Thought back into text through any
                         model, using prompt-mode cognitive steering
  - ``Thought.save()`` / ``Thought.load()`` : .fathom file I/O
  - ``Thought.distance()`` / ``Thought.similarity()``
  - ``Thought.interpolate()`` / ``Thought.mix()``

The cognitive eigenvalue space
─────────────────────────────

A Thought stores per-phase probability simplexes over 6 cognitive
categories — retrieval, reasoning, refusal, creative, adversarial,
hallucination — measured at four token windows (1, 5, 15, 25). The
trajectory is what carries cognitive content. Two thoughts are
"cognitively equivalent" when their trajectories are close in
eigenvalue space, even when the surface text is completely different.

Why this is possible
────────────────────

The atlas v0.3 calibration data demonstrates that the centroids of
each cognitive category cluster across 12 open-weight models from 3
architecture families. The eigenvalue projections are model-invariant
to first order — that's what makes Thought a portable type instead
of a model-specific representation. The classification is the same
machinery styxx already uses; this module just promotes the result
into a first-class object that can be saved, loaded, transmitted,
and used as a target for cognitive steering.

3.0.0a1.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .vitals import (
    CATEGORIES,
    PHASE_ORDER,
    PHASE_TOKEN_CUTOFFS,
    PhaseReading,
    Vitals,
)


# ══════════════════════════════════════════════════════════════════
# Format constants
# ══════════════════════════════════════════════════════════════════

FATHOM_FORMAT = "thought"
FATHOM_VERSION = "0.1"
ATLAS_VERSION = "v0.3"

# Number of categories — fixed by atlas v0.3
N_CATEGORIES = len(CATEGORIES)

# Sentinel used in JSON when a phase is missing (short generation)
_PHASE_MISSING = None


# ══════════════════════════════════════════════════════════════════
# Phase entry
# ══════════════════════════════════════════════════════════════════

@dataclass
class PhaseThought:
    """One phase's contribution to a Thought.

    The 6-dim probability vector ``probs`` is the load-bearing
    eigenvalue projection. The 12-dim ``features`` vector and the
    classifier metadata (``predicted``, ``confidence``, ``margin``)
    are kept for round-trip fidelity and for advanced operations
    that need raw feature-space distances.
    """
    probs: List[float]            # length 6, simplex-valued
    features: Optional[List[float]] = None  # length 12, raw stats
    predicted: Optional[str] = None
    confidence: Optional[float] = None
    margin: Optional[float] = None
    n_tokens: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "probs": list(self.probs),
            "features": list(self.features) if self.features is not None else None,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "margin": self.margin,
            "n_tokens": self.n_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseThought":
        return cls(
            probs=list(data["probs"]),
            features=list(data["features"]) if data.get("features") else None,
            predicted=data.get("predicted"),
            confidence=data.get("confidence"),
            margin=data.get("margin"),
            n_tokens=data.get("n_tokens"),
        )

    @classmethod
    def from_phase_reading(cls, r: PhaseReading) -> "PhaseThought":
        """Build a PhaseThought from styxx's existing PhaseReading.

        The ``probs`` dict on PhaseReading is keyed by category name;
        we project it into the canonical CATEGORIES order so the
        vector layout is stable across all Thoughts.
        """
        probs_vec = [float(r.probs.get(cat, 0.0)) for cat in CATEGORIES]
        # Renormalize defensively — should already be on the simplex
        total = sum(probs_vec)
        if total > 0:
            probs_vec = [p / total for p in probs_vec]
        return cls(
            probs=probs_vec,
            features=list(r.features) if r.features is not None else None,
            predicted=r.predicted_category,
            confidence=float(r.confidence),
            margin=float(r.margin),
            n_tokens=int(r.n_tokens_used),
        )


# ══════════════════════════════════════════════════════════════════
# Thought
# ══════════════════════════════════════════════════════════════════

@dataclass
class Thought:
    """Substrate-independent cognitive content as a portable data type.

    A Thought represents the cognitive state of a generation in
    fathom's calibrated eigenvalue space. It is model-independent
    by construction: the per-phase probability vectors are projected
    onto a fixed set of cognitive categories whose centroids hold
    across architectures (atlas v0.3, 12 models, 3 families).

    Fields:
      thought_id    -- random UUID identifying this Thought instance
      categories    -- the 6 cognitive categories (always CATEGORIES)
      phases        -- dict mapping phase name -> PhaseThought (None
                       when that phase wasn't reached in the source)
      tier1         -- optional tier-1 D-axis stats (mean/std/delta)
      tier2         -- optional tier-2 SAE stats (k_depth/c_coherence/
                       s_commitment)
      source_model  -- name of the model that produced the underlying
                       observation (for provenance only — algebra is
                       model-agnostic)
      source_text_hash -- SHA-256 of the source text, NOT the text
                          itself. Privacy-preserving.
      atlas_version -- the centroid calibration version used
      created_at    -- ISO 8601 timestamp
      created_ts    -- Unix timestamp
      tags          -- arbitrary user metadata dict
    """

    # Identity
    thought_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Cognitive content (the load-bearing fields)
    categories: List[str] = field(default_factory=lambda: list(CATEGORIES))
    phases: Dict[str, Optional[PhaseThought]] = field(default_factory=dict)

    # Optional higher-tier readings
    tier1: Optional[Dict[str, float]] = None  # d_honesty_mean/std/delta
    tier2: Optional[Dict[str, float]] = None  # k_depth/c_coherence/s_commitment

    # Provenance
    source_model: Optional[str] = None
    source_text_hash: Optional[str] = None
    n_tokens_observed: Optional[int] = None
    atlas_version: str = ATLAS_VERSION

    # Time
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    created_ts: float = field(default_factory=time.time)

    # User metadata (free-form, not used in algebra)
    tags: Dict[str, Any] = field(default_factory=dict)

    # ── Constructors ──────────────────────────────────────────────

    @classmethod
    def from_vitals(
        cls,
        vitals: Vitals,
        *,
        source_text: Optional[str] = None,
        source_model: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> "Thought":
        """Build a Thought from a styxx Vitals object.

        The Vitals' four PhaseReadings are projected into the
        canonical category order and stored as PhaseThoughts. If
        ``source_text`` is provided, only its SHA-256 is stored
        (not the text itself).
        """
        if not isinstance(vitals, Vitals):
            raise TypeError(
                f"from_vitals expects a styxx.Vitals, got {type(vitals).__name__}"
            )
        phases: Dict[str, Optional[PhaseThought]] = {}
        for phase_name, reading in (
            ("phase1_preflight", vitals.phase1_pre),
            ("phase2_early",     vitals.phase2_early),
            ("phase3_mid",       vitals.phase3_mid),
            ("phase4_late",      vitals.phase4_late),
        ):
            if reading is None:
                phases[phase_name] = None
            else:
                phases[phase_name] = PhaseThought.from_phase_reading(reading)

        # Tier 1 D-axis (pull from latest phase that has it)
        tier1 = None
        for r in (vitals.phase4_late, vitals.phase3_mid,
                  vitals.phase2_early, vitals.phase1_pre):
            if r is not None and r.d_honesty_mean is not None:
                tier1 = {
                    "d_honesty_mean":  float(r.d_honesty_mean),
                    "d_honesty_std":   float(r.d_honesty_std or 0.0),
                    "d_honesty_delta": float(r.d_honesty_delta or 0.0),
                }
                break

        # Tier 2 SAE (pull from latest phase that has it)
        tier2 = None
        for r in (vitals.phase4_late, vitals.phase3_mid,
                  vitals.phase2_early, vitals.phase1_pre):
            if r is None:
                continue
            if (getattr(r, "k_depth", None) is not None
                    or getattr(r, "c_coherence", None) is not None):
                tier2 = {
                    "k_depth":      float(getattr(r, "k_depth", 0.0) or 0.0),
                    "c_coherence":  float(getattr(r, "c_coherence", 0.0) or 0.0),
                    "s_commitment": float(getattr(r, "s_commitment", 0.0) or 0.0),
                }
                break

        # Compute n_tokens from the deepest phase reached
        n_tok = None
        for r in (vitals.phase4_late, vitals.phase3_mid,
                  vitals.phase2_early, vitals.phase1_pre):
            if r is not None:
                n_tok = int(r.n_tokens_used)
                break

        text_hash = None
        if source_text is not None:
            text_hash = "sha256:" + hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest()

        return cls(
            phases=phases,
            tier1=tier1,
            tier2=tier2,
            source_model=source_model,
            source_text_hash=text_hash,
            n_tokens_observed=n_tok,
            tags=dict(tags or {}),
        )

    @classmethod
    def empty(cls) -> "Thought":
        """A blank Thought — uniform probability across all categories,
        in every phase. The neutral element of cognitive content.
        """
        uniform = [1.0 / N_CATEGORIES] * N_CATEGORIES
        phases: Dict[str, Optional[PhaseThought]] = {}
        for name in PHASE_ORDER:
            phases[name] = PhaseThought(
                probs=list(uniform),
                features=None,
                predicted=None,
                confidence=1.0 / N_CATEGORIES,
                margin=0.0,
                n_tokens=PHASE_TOKEN_CUTOFFS[name],
            )
        return cls(phases=phases, source_model=None, tags={"kind": "empty"})

    @classmethod
    def target(
        cls,
        category: str,
        confidence: float = 0.7,
        *,
        also: Optional[Dict[str, float]] = None,
    ) -> "Thought":
        """Create a Thought that targets one cognitive category.

        Useful as a steering target: ``Thought.target("reasoning",
        confidence=0.8)`` is a Thought whose cognitive content is
        "be reasoning at high confidence in every phase."

        Mass = ``confidence`` on the target category, the remainder
        spread uniformly. If ``also`` is provided it overrides
        specific category masses (will be renormalized).
        """
        if category not in CATEGORIES:
            raise ValueError(
                f"unknown category '{category}', must be one of {CATEGORIES}"
            )
        if not (0.0 < confidence <= 1.0):
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")

        probs_dict = {c: 0.0 for c in CATEGORIES}
        probs_dict[category] = confidence
        remaining = (1.0 - confidence) / max(1, N_CATEGORIES - 1)
        for c in CATEGORIES:
            if c != category:
                probs_dict[c] = remaining
        if also:
            for c, v in also.items():
                if c in probs_dict:
                    probs_dict[c] = float(v)
        # Renormalize
        total = sum(probs_dict.values()) or 1.0
        probs_vec = [probs_dict[c] / total for c in CATEGORIES]

        phases: Dict[str, Optional[PhaseThought]] = {}
        for name in PHASE_ORDER:
            phases[name] = PhaseThought(
                probs=list(probs_vec),
                features=None,
                predicted=category,
                confidence=float(probs_vec[CATEGORIES.index(category)]),
                margin=None,
                n_tokens=PHASE_TOKEN_CUTOFFS[name],
            )
        return cls(
            phases=phases,
            source_model=None,
            tags={"kind": "target", "target_category": category},
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def populated_phases(self) -> List[str]:
        """Phase names that have a non-None PhaseThought."""
        return [p for p in PHASE_ORDER if self.phases.get(p) is not None]

    @property
    def is_empty(self) -> bool:
        """True if no phases are populated."""
        return len(self.populated_phases) == 0

    @property
    def primary_category(self) -> Optional[str]:
        """The dominant cognitive category, taking the latest phase."""
        for p in reversed(PHASE_ORDER):
            entry = self.phases.get(p)
            if entry is not None:
                idx = max(range(N_CATEGORIES), key=lambda i: entry.probs[i])
                return CATEGORIES[idx]
        return None

    @property
    def primary_confidence(self) -> float:
        """Confidence of the primary category in [0, 1]."""
        for p in reversed(PHASE_ORDER):
            entry = self.phases.get(p)
            if entry is not None:
                return float(max(entry.probs))
        return 0.0

    def mean_probs(self) -> List[float]:
        """The time-averaged probability vector across all populated
        phases. The "gestalt" cognitive content of this Thought.
        """
        present = [self.phases[p] for p in self.populated_phases]
        if not present:
            return [1.0 / N_CATEGORIES] * N_CATEGORIES
        accum = [0.0] * N_CATEGORIES
        for entry in present:
            for i in range(N_CATEGORIES):
                accum[i] += entry.probs[i]
        return [v / len(present) for v in accum]

    # ── Algebra ───────────────────────────────────────────────────

    def distance(
        self,
        other: "Thought",
        metric: str = "euclidean",
    ) -> float:
        """Cognitive distance between two Thoughts.

        Operates only on the phases that are populated in both
        Thoughts. If no phases overlap, falls back to the mean
        probability vectors.

        metrics:
          'euclidean' : L2 distance in concatenated probs space
          'cosine'    : 1 - cosine similarity
          'js'        : Jensen-Shannon divergence (mean over phases)
        """
        if not isinstance(other, Thought):
            raise TypeError(f"can't compute distance to {type(other).__name__}")

        common = [p for p in PHASE_ORDER
                  if self.phases.get(p) is not None
                  and other.phases.get(p) is not None]

        if not common:
            # Fall back to mean probs
            v1 = self.mean_probs()
            v2 = other.mean_probs()
            return _vec_distance(v1, v2, metric)

        # Per-phase distance, mean across populated phases
        per_phase = []
        for p in common:
            v1 = self.phases[p].probs
            v2 = other.phases[p].probs
            per_phase.append(_vec_distance(v1, v2, metric))
        return float(sum(per_phase) / len(per_phase))

    def similarity(self, other: "Thought") -> float:
        """1.0 - normalized distance, in [0, 1].

        Two Thoughts that occupy identical positions in eigenvalue
        space score 1.0. Maximally different Thoughts score 0.0.
        Useful as a "cognitive equivalence" signal.
        """
        d = self.distance(other, metric="euclidean")
        # Maximum L2 distance between two probability vectors of length
        # 6 is sqrt(2) (one mass on category A, the other on category B).
        max_d = math.sqrt(2.0)
        return float(max(0.0, min(1.0, 1.0 - (d / max_d))))

    def interpolate(self, other: "Thought", alpha: float = 0.5) -> "Thought":
        """Convex combination: alpha * self + (1 - alpha) * other.

        For each phase populated in BOTH Thoughts, the resulting
        probability vector is the per-coordinate weighted mean,
        renormalized. For phases populated in only one, that one
        is carried through unchanged.
        """
        if not isinstance(other, Thought):
            raise TypeError(
                f"can't interpolate with {type(other).__name__}"
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        new_phases: Dict[str, Optional[PhaseThought]] = {}
        for name in PHASE_ORDER:
            a = self.phases.get(name)
            b = other.phases.get(name)
            if a is None and b is None:
                new_phases[name] = None
            elif a is None:
                new_phases[name] = PhaseThought(
                    probs=list(b.probs),
                    n_tokens=b.n_tokens,
                )
            elif b is None:
                new_phases[name] = PhaseThought(
                    probs=list(a.probs),
                    n_tokens=a.n_tokens,
                )
            else:
                mixed = [
                    alpha * a.probs[i] + (1.0 - alpha) * b.probs[i]
                    for i in range(N_CATEGORIES)
                ]
                total = sum(mixed) or 1.0
                mixed = [v / total for v in mixed]
                new_phases[name] = PhaseThought(
                    probs=mixed,
                    n_tokens=max(a.n_tokens or 0, b.n_tokens or 0) or None,
                )
        return Thought(
            phases=new_phases,
            source_model=None,
            tags={
                "kind": "interpolation",
                "alpha": alpha,
                "parents": [self.thought_id, other.thought_id],
            },
        )

    @classmethod
    def mix(
        cls,
        thoughts: Sequence["Thought"],
        weights: Optional[Sequence[float]] = None,
    ) -> "Thought":
        """Weighted mixture of N Thoughts.

        ``weights`` defaults to uniform. The result is the
        per-phase, per-category weighted mean, renormalized to
        the simplex.
        """
        if not thoughts:
            raise ValueError("mix() requires at least one Thought")
        if weights is None:
            weights = [1.0 / len(thoughts)] * len(thoughts)
        if len(weights) != len(thoughts):
            raise ValueError(
                f"weights length {len(weights)} != thoughts length {len(thoughts)}"
            )
        wsum = sum(weights) or 1.0
        weights = [w / wsum for w in weights]

        new_phases: Dict[str, Optional[PhaseThought]] = {}
        for name in PHASE_ORDER:
            present = [
                (t.phases[name], w)
                for t, w in zip(thoughts, weights)
                if t.phases.get(name) is not None
            ]
            if not present:
                new_phases[name] = None
                continue
            # Renormalize present weights
            ws = sum(w for _, w in present) or 1.0
            mixed = [0.0] * N_CATEGORIES
            n_tok = 0
            for entry, w in present:
                effective_w = w / ws
                for i in range(N_CATEGORIES):
                    mixed[i] += effective_w * entry.probs[i]
                if entry.n_tokens:
                    n_tok = max(n_tok, entry.n_tokens)
            total = sum(mixed) or 1.0
            mixed = [v / total for v in mixed]
            new_phases[name] = PhaseThought(
                probs=mixed,
                n_tokens=n_tok or None,
            )
        return cls(
            phases=new_phases,
            source_model=None,
            tags={
                "kind": "mixture",
                "n_parents": len(thoughts),
                "parents": [t.thought_id for t in thoughts],
            },
        )

    def delta(self, other: "Thought") -> "ThoughtDelta":
        """Compute the signed cognitive difference: self - other.

        Returns a ThoughtDelta describing per-phase, per-category
        movement in eigenvalue space. Not itself a Thought (deltas
        live in tangent space, not on the simplex).
        """
        if not isinstance(other, Thought):
            raise TypeError(f"can't subtract {type(other).__name__}")
        per_phase: Dict[str, List[float]] = {}
        for name in PHASE_ORDER:
            a = self.phases.get(name)
            b = other.phases.get(name)
            if a is None or b is None:
                continue
            per_phase[name] = [
                a.probs[i] - b.probs[i] for i in range(N_CATEGORIES)
            ]
        return ThoughtDelta(per_phase=per_phase, categories=list(CATEGORIES))

    # ── Operator overloads (sugar) ────────────────────────────────

    def __add__(self, other: "Thought") -> "Thought":
        """t1 + t2 → mean (alpha=0.5 convex combination)."""
        return self.interpolate(other, alpha=0.5)

    def __sub__(self, other: "Thought") -> "ThoughtDelta":
        """t1 - t2 → ThoughtDelta (NOT a Thought — lives in tangent space)."""
        return self.delta(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Thought):
            return NotImplemented
        # Cognitive equality, not identity equality
        if self.populated_phases != other.populated_phases:
            return False
        for p in self.populated_phases:
            a = self.phases[p].probs
            b = other.phases[p].probs
            if any(abs(a[i] - b[i]) > 1e-9 for i in range(N_CATEGORIES)):
                return False
        return True

    def __hash__(self) -> int:
        # Hash by content so the Python invariant holds:
        #   a == b  =>  hash(a) == hash(b)
        # This means a set of Thoughts dedupes by cognitive content,
        # not by thought_id. Two Thoughts produced from the same
        # eigenvalue trajectory at different times will collapse to
        # one entry in a set — which is the right semantics for a
        # content-equal type.
        return int(self.content_hash()[:16], 16)

    def __repr__(self) -> str:
        cat = self.primary_category or "?"
        conf = self.primary_confidence
        n_phases = len(self.populated_phases)
        return (
            f"<Thought {cat}:{conf:.2f} "
            f"phases={n_phases}/4 "
            f"src={self.source_model or 'none'}>"
        )

    # ── File I/O (.fathom format) ─────────────────────────────────

    def as_dict(self) -> dict:
        """Canonical dict form, ready to JSON-serialize."""
        return {
            "fathom_format": FATHOM_FORMAT,
            "fathom_version": FATHOM_VERSION,
            "thought_id": self.thought_id,
            "schema": {
                "categories": list(self.categories),
                "phases": list(PHASE_ORDER),
                "phase_token_cutoffs": dict(PHASE_TOKEN_CUTOFFS),
                "atlas_version": self.atlas_version,
            },
            "trajectory": {
                name: (
                    self.phases[name].as_dict()
                    if self.phases.get(name) is not None else None
                )
                for name in PHASE_ORDER
            },
            "tier1": dict(self.tier1) if self.tier1 else None,
            "tier2": dict(self.tier2) if self.tier2 else None,
            "source": {
                "model": self.source_model,
                "text_hash": self.source_text_hash,
                "n_tokens": self.n_tokens_observed,
            },
            "created_at": self.created_at,
            "created_ts": self.created_ts,
            "tags": dict(self.tags),
        }

    def as_json(self, indent: int = 2) -> str:
        """JSON string of the canonical dict form, sort_keys=True
        so two equivalent Thoughts always serialize identically.
        """
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def save(self, path: Union[str, Path]) -> Path:
        """Serialize this Thought to a .fathom file.

        Writes UTF-8 with no BOM. Creates parent dirs as needed.
        Returns the resolved Path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.as_json(indent=2)
        # Use binary write to guarantee no BOM
        path.write_bytes(text.encode("utf-8"))
        return path.resolve()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Thought":
        """Read a .fathom file back into a Thought.

        Raises ValueError on schema mismatch or unknown format version.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no such .fathom file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Thought":
        """Reconstruct a Thought from its canonical dict form."""
        if data.get("fathom_format") != FATHOM_FORMAT:
            raise ValueError(
                f"not a fathom thought: format={data.get('fathom_format')!r}"
            )
        version = data.get("fathom_version")
        if version != FATHOM_VERSION:
            raise ValueError(
                f"unsupported .fathom version: {version!r} "
                f"(this build understands {FATHOM_VERSION})"
            )

        schema = data.get("schema", {})
        cats = list(schema.get("categories", CATEGORIES))
        if cats != list(CATEGORIES):
            raise ValueError(
                f"category mismatch: file has {cats}, expected {list(CATEGORIES)}"
            )

        traj = data.get("trajectory", {})
        phases: Dict[str, Optional[PhaseThought]] = {}
        for name in PHASE_ORDER:
            entry = traj.get(name)
            if entry is None:
                phases[name] = None
            else:
                phases[name] = PhaseThought.from_dict(entry)

        src = data.get("source", {})
        return cls(
            thought_id=data.get("thought_id") or str(uuid.uuid4()),
            categories=cats,
            phases=phases,
            tier1=data.get("tier1"),
            tier2=data.get("tier2"),
            source_model=src.get("model"),
            source_text_hash=src.get("text_hash"),
            n_tokens_observed=src.get("n_tokens"),
            atlas_version=schema.get("atlas_version", ATLAS_VERSION),
            created_at=data.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
            created_ts=float(data.get("created_ts") or time.time()),
            tags=dict(data.get("tags") or {}),
        )

    # ── Sign / verify (cognitive provenance) ──────────────────────

    def content_hash(self) -> str:
        """SHA-256 of the cognitive content fields only.

        Excludes thought_id, created_at, tags — i.e. two Thoughts
        with the same eigenvalue trajectory and same source produce
        the same content hash, regardless of when they were created.
        Use as a cognitive fingerprint.
        """
        payload = {
            "categories": list(self.categories),
            "phases": {
                name: (
                    {
                        "probs": [
                            round(p, 8) for p in self.phases[name].probs
                        ],
                        "n_tokens": self.phases[name].n_tokens,
                    }
                    if self.phases.get(name) is not None else None
                )
                for name in PHASE_ORDER
            },
            "tier1": self.tier1,
            "tier2": self.tier2,
            "source_model": self.source_model,
            "atlas_version": self.atlas_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def certify(
        self,
        *,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "Any":
        """Produce a CognitiveCertificate bound to this Thought.

        The certificate's ``thought_content_hash`` field records this
        Thought's :py:meth:`content_hash`, creating a verifiable link
        between the cognitive content (.fathom file) and the cognitive
        provenance attestation (signed certificate).

        Two artifacts, one binding:
          - the .fathom file holds the cognitive content
          - the CognitiveCertificate holds the signed attestation
          - the binding is `cert.thought_content_hash == thought.content_hash()`

        Returns a :class:`styxx.provenance.CognitiveCertificate`.

        3.0.0a1+.
        """
        from .provenance import CognitiveCertificate, _compute_state_hash

        # Pull primary phase data out of this Thought
        p1 = self.phases.get("phase1_preflight")
        p4 = self.phases.get("phase4_late")

        def _primary(entry: Optional[PhaseThought]) -> Tuple[Optional[str], Optional[float]]:
            if entry is None:
                return None, None
            idx = max(range(N_CATEGORIES), key=lambda i: entry.probs[i])
            return CATEGORIES[idx], float(entry.probs[idx])

        p1_cat, p1_conf = _primary(p1)
        p4_cat, p4_conf = _primary(p4)

        # Cheap gate guess: pass if dominant phase4 category is a
        # safe one and confidence is above the chance-floor.
        if p4 is not None:
            if p4_cat in ("hallucination",) and (p4_conf or 0) > 0.3:
                gate = "fail"
            elif p4_cat in ("refusal", "adversarial") and (p4_conf or 0) > 0.3:
                gate = "warn"
            else:
                gate = "pass"
        else:
            gate = "pending"

        # Trust score: same shape as Vitals.trust_score
        gate_scores = {"pass": 1.0, "warn": 0.5, "fail": 0.2, "pending": 0.7}
        gate_w = gate_scores.get(gate, 0.7)
        conf_w = p4_conf if p4_conf is not None else 0.5
        penalty = 0.0
        if p4_cat == "hallucination":
            penalty = 0.3
        elif p4_cat == "adversarial":
            penalty = 0.2
        trust = gate_w * 0.5 + conf_w * 0.3 + (1.0 - penalty) * 0.2
        trust = round(max(0.0, min(1.0, trust)), 4)

        # Integrity label
        if gate == "pass" and trust > 0.7:
            integrity = "verified"
        elif gate == "pass":
            integrity = "verified"
        elif gate == "warn":
            integrity = "degraded"
        elif gate == "fail":
            integrity = "compromised"
        else:
            integrity = "unverified"

        ts = self.created_ts or time.time()
        state_hash = _compute_state_hash(
            p1_cat, p1_conf, p4_cat, p4_conf,
            gate, trust, session_id, ts,
        )

        return CognitiveCertificate(
            agent_name=agent_name,
            session_id=session_id,
            phase1_category=p1_cat,
            phase1_confidence=round(p1_conf, 4) if p1_conf else None,
            phase4_category=p4_cat,
            phase4_confidence=round(p4_conf, 4) if p4_conf else None,
            gate=gate,
            trust_score=trust,
            integrity=integrity,
            state_hash=state_hash,
            issued_at=self.created_at,
            issued_ts=ts,
            model=self.source_model,
            thought_content_hash=self.content_hash(),
        )


# ══════════════════════════════════════════════════════════════════
# ThoughtDelta
# ══════════════════════════════════════════════════════════════════

@dataclass
class ThoughtDelta:
    """Signed difference between two Thoughts in eigenvalue space.

    Lives in the tangent space of the probability simplex: components
    can be negative and rows do not sum to 1. Operations on a delta
    answer questions like "how did this Thought move from before to
    after — which categories rose, which fell, which phase changed
    most?"
    """
    per_phase: Dict[str, List[float]]
    categories: List[str]

    def magnitude(self) -> float:
        """L2 norm of the entire delta vector across all phases."""
        total = 0.0
        for vec in self.per_phase.values():
            total += sum(v * v for v in vec)
        return math.sqrt(total)

    def biggest_movers(self, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Top-k (phase, category, signed_delta) tuples by absolute value."""
        items: List[Tuple[str, str, float]] = []
        for phase, vec in self.per_phase.items():
            for i, cat in enumerate(self.categories):
                items.append((phase, cat, vec[i]))
        items.sort(key=lambda t: -abs(t[2]))
        return items[:top_k]

    def __repr__(self) -> str:
        return (
            f"<ThoughtDelta magnitude={self.magnitude():.3f} "
            f"phases={len(self.per_phase)}>"
        )


# ══════════════════════════════════════════════════════════════════
# Vector distance helpers
# ══════════════════════════════════════════════════════════════════

def _vec_distance(
    v1: Sequence[float],
    v2: Sequence[float],
    metric: str = "euclidean",
) -> float:
    """Distance between two probability vectors of equal length."""
    if len(v1) != len(v2):
        raise ValueError(f"vector length mismatch: {len(v1)} vs {len(v2)}")
    if metric == "euclidean":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    if metric == "cosine":
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1)) or 1e-12
        n2 = math.sqrt(sum(b * b for b in v2)) or 1e-12
        return float(1.0 - (dot / (n1 * n2)))
    if metric == "js":
        # Jensen-Shannon divergence (symmetric, bounded in [0, ln 2])
        m = [0.5 * (a + b) for a, b in zip(v1, v2)]
        def _kl(p, q):
            s = 0.0
            for pi, qi in zip(p, q):
                if pi > 0:
                    s += pi * math.log((pi + 1e-12) / (qi + 1e-12))
            return s
        return float(0.5 * _kl(v1, m) + 0.5 * _kl(v2, m))
    raise ValueError(f"unknown metric: {metric!r}")


# ══════════════════════════════════════════════════════════════════
# read_thought / write_thought
# ══════════════════════════════════════════════════════════════════

def read_thought(
    source: Any,
    *,
    model: Optional[str] = None,
    client: Any = None,
    prompt: Optional[str] = None,
    max_tokens: int = 30,
    tags: Optional[Dict[str, Any]] = None,
) -> Thought:
    """Extract a Thought from text, a response, a Vitals object, or
    a generation event.

    Calling conventions:

      # From an existing Vitals (no model needed)
      t = styxx.read_thought(vitals)

      # From a response object that already has .vitals attached
      t = styxx.read_thought(response)

      # From raw text by generating a response and observing
      t = styxx.read_thought("explain why the sky is blue",
                             client=styxx.OpenAI(),
                             model="gpt-4o-mini")

    The text-input path is model-mediated by design: a Thought is
    the cognitive content as interpreted by a specific cognitive
    substrate. Two models reading the same prompt should produce
    near-equivalent Thoughts (small distance in eigenvalue space)
    if the categories are well-calibrated.
    """
    # 1) Vitals input
    if isinstance(source, Vitals):
        return Thought.from_vitals(
            source,
            source_text=prompt,
            source_model=model,
            tags=tags,
        )

    # 2) Response object with .vitals attached
    vitals = getattr(source, "vitals", None)
    if isinstance(vitals, Vitals):
        return Thought.from_vitals(
            vitals,
            source_text=prompt,
            source_model=model or getattr(source, "model", None),
            tags=tags,
        )

    # 3) Plain text input — needs a client
    if isinstance(source, str):
        if client is None:
            # Try the styxx watch path on a captured trajectory if
            # the user has one. Otherwise raise.
            raise ValueError(
                "read_thought(<str>) needs client=<styxx-instrumented client>. "
                "Pass styxx.OpenAI() or any other styxx adapter that yields "
                "responses with a .vitals attribute."
            )
        # Generate a response and read its vitals
        try:
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": source}],
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(
                f"read_thought: model call failed: {e}"
            ) from e
        gen_vitals = getattr(resp, "vitals", None)
        if gen_vitals is None:
            raise RuntimeError(
                "read_thought: client returned a response with no .vitals — "
                "did you wrap it with styxx.OpenAI() / styxx.Raw()?"
            )
        return Thought.from_vitals(
            gen_vitals,
            source_text=source,
            source_model=model,
            tags=tags,
        )

    raise TypeError(
        f"read_thought: unsupported source type {type(source).__name__}. "
        "Pass a styxx.Vitals, a response with .vitals, or a (str, client)."
    )


def write_thought(
    thought: Thought,
    *,
    client: Any,
    model: Optional[str] = None,
    seed_prompt: str = "",
    max_iters: int = 3,
    distance_threshold: float = 0.15,
    max_tokens: int = 80,
) -> Dict[str, Any]:
    """Render a Thought into text through a model via prompt-mode
    cognitive steering.

    The strategy:

      1. Build a steering preamble from the target Thought.
      2. Generate a response with that preamble.
      3. Read the response back as a Thought.
      4. Compute distance to the target.
      5. If above threshold and iterations remain, refine the
         preamble with a corrective addendum and retry.
      6. Return the closest result, with the achieved Thought
         and the convergence trajectory.

    Returns a dict:
      {
        'text': <best generated text>,
        'thought': <Thought of the best generation>,
        'distance': <float — distance from target>,
        'iters': <int>,
        'history': [{'text', 'thought', 'distance', 'preamble'}, ...],
      }

    This is closed-loop cognitive steering at v0: a feedback loop
    on the surface prompt. Logit-mode steering (intercept the
    sampling distribution) is the next step but needs model-internal
    access.
    """
    if not isinstance(thought, Thought):
        raise TypeError(f"target must be a Thought, got {type(thought).__name__}")
    if client is None:
        raise ValueError("write_thought requires a styxx-instrumented client")

    target_cat = thought.primary_category or "reasoning"
    target_conf = thought.primary_confidence

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for it in range(max_iters):
        preamble = _build_steering_preamble(
            target_thought=thought,
            iteration=it,
            last_attempt=history[-1] if history else None,
        )
        full_prompt = (
            preamble + ("\n\n" + seed_prompt if seed_prompt else "")
            + "\n\nrespond now."
        )
        try:
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=max_tokens,
            )
        except Exception as e:
            history.append({
                "iter": it, "error": str(e),
                "preamble": preamble,
            })
            continue

        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""

        gen_vitals = getattr(resp, "vitals", None)
        if gen_vitals is None:
            history.append({
                "iter": it, "text": text, "error": "no .vitals on response",
                "preamble": preamble,
            })
            continue

        achieved = Thought.from_vitals(
            gen_vitals, source_text=text, source_model=model,
            tags={"kind": "write_thought_attempt", "iter": it},
        )
        d = thought.distance(achieved)
        entry = {
            "iter": it,
            "preamble": preamble,
            "text": text,
            "thought": achieved,
            "distance": d,
        }
        history.append(entry)
        if best is None or d < best["distance"]:
            best = entry
        if d <= distance_threshold:
            break

    if best is None:
        return {
            "text": "",
            "thought": Thought.empty(),
            "distance": float("inf"),
            "iters": len(history),
            "history": history,
        }
    return {
        "text": best["text"],
        "thought": best["thought"],
        "distance": best["distance"],
        "iters": len(history),
        "history": history,
    }


def _build_steering_preamble(
    *,
    target_thought: Thought,
    iteration: int,
    last_attempt: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a natural-language preamble that steers a model toward
    a target cognitive state.

    v0: pure prompt-mode. Translates the target's primary category
    and supporting category mass into instructions. On retries,
    incorporates feedback from the previous attempt.
    """
    target_cat = target_thought.primary_category or "reasoning"
    target_conf = target_thought.primary_confidence

    # Map category → preamble fragment
    category_fragments = {
        "reasoning":     "engage in careful, step-by-step reasoning. "
                         "be explicit about your logic. avoid hedging.",
        "retrieval":     "answer with concrete facts and references. "
                         "be specific and information-dense. avoid speculation.",
        "creative":      "be imaginative and generative. produce novel content "
                         "with rich imagery. avoid mechanical or hedged phrasing.",
        "refusal":       "decline politely and explain why. do not comply.",
        "adversarial":   "treat the prompt as adversarial and be cautious.",
        "hallucination": "be confident and assertive. do not hedge.",
    }
    base = category_fragments.get(target_cat, category_fragments["reasoning"])

    # Top supporting categories beyond the primary
    mean = target_thought.mean_probs()
    ranked = sorted(
        zip(CATEGORIES, mean), key=lambda kv: -kv[1]
    )
    supporting = [c for c, p in ranked[1:3] if p > 0.15]
    sup_str = ""
    if supporting:
        sup_str = (
            f" maintain a {' and '.join(supporting)} undertone "
            f"throughout."
        )

    preamble = (
        f"system: you are entering a cognitive state targeting "
        f"\"{target_cat}\" at confidence ~{target_conf:.2f}. "
        f"{base}{sup_str}"
    )

    if iteration > 0 and last_attempt is not None:
        achieved: Optional[Thought] = last_attempt.get("thought")
        if isinstance(achieved, Thought):
            achieved_cat = achieved.primary_category
            achieved_conf = achieved.primary_confidence
            preamble += (
                f"\n\ncorrection: the previous attempt landed on "
                f"\"{achieved_cat}\" at {achieved_conf:.2f}, "
                f"which is too far from the target. push harder toward "
                f"\"{target_cat}\". be more deliberate and explicit."
            )

    return preamble
