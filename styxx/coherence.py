# -*- coding: utf-8 -*-
"""
styxx.coherence — phase-coherence measurement between two agents' pulse-traces.

This module is the **API form** of the methodology preregistered at commit
``3473523`` and scored by the locked driver ``scripts/phase_coherence_pilot.py``
at commit ``23b7912``.

Two contracts
─────────────
1. **Primary CC matches the locked scorer.** The function
   ``primary_coherence(pulse_a, pulse_b)`` returns numerically identical
   values to ``cc_pearson_lag0(align_pulse_traces(pulse_a, pulse_b))`` in
   the locked driver. If you change this function, you change the meaning
   of the preregistered measurement and must open a new preregistration
   per §10 of the lock document.

2. **Extras are exploratory.** Lag-sweep, per-axis CC, and Hilbert-PLV are
   provided as additional measurements. The preregistration treats per-axis
   CC as exploratory only (§3, single-channel multiple-comparison lock).
   Lag-sweep and PLV are not in the preregistration at all — they are
   diagnostic depth, not hypothesis tests.

When to call which
──────────────────

    >>> from styxx.coherence import (
    ...     load_pulse_trace, pulse_coherence,
    ... )
    >>> pulse_a = load_pulse_trace(Path("~/.styxx/agents/conv1_A/chart.jsonl"))
    >>> pulse_b = load_pulse_trace(Path("~/.styxx/agents/conv1_B/chart.jsonl"))
    >>> result = pulse_coherence(pulse_a, pulse_b)
    >>> result.primary_cc           # Pearson r at lag 0, hypothesis-bearing
    >>> result.per_axis             # exploratory: sycoph/decept/over/refusal
    >>> result.lag_sweep            # exploratory: lead/lag structure
    >>> result.plv                  # exploratory: Hilbert phase-locking value
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Schema: PulseSample (matches preregistration §2 verbatim)
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PulseSample:
    """Immutable per-message projection of a styxx audit.

    Schema fixed by the preregistration §2 contract. Do not extend
    without a new preregistration lock-hash.
    """
    timestamp: float
    msg_id: str
    composite: float
    scores: dict
    needs_revision: bool
    construct_ceiling_fires: list

    def __post_init__(self):
        required_scores = {"sycophancy", "deception", "overconfidence", "refusal"}
        missing = required_scores - set(self.scores.keys())
        if missing:
            raise ValueError(
                f"PulseSample {self.msg_id}: missing scores {missing}"
            )


PulseTrace = list  # list[PulseSample], ascending by timestamp


# ──────────────────────────────────────────────────────────────────
# Loader (mirrors the locked driver verbatim)
# ──────────────────────────────────────────────────────────────────


def load_pulse_trace(
    chart_path: Path,
    session_id: Optional[str] = None,
) -> PulseTrace:
    """Read chart.jsonl, return one agent's PulseSamples.

    Source filter: ``source == "preflight"`` (per analytics convention).
    Field mapping: ``cogn_*``-prefixed names per the §2 schema
    corrigendum. If ``msg_id`` is missing on legacy entries, it is
    synthesized as ``f"{session_id}:{line_no}"``.

    This loader's behavior must remain numerically identical to
    ``scripts/phase_coherence_pilot.py::load_pulse_trace`` for the
    primary preregistered measurement to remain valid.
    """
    chart_path = Path(chart_path).expanduser()
    if not chart_path.exists():
        raise FileNotFoundError(f"chart.jsonl not found at {chart_path}")

    samples: list[PulseSample] = []
    with chart_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("source") != "preflight":
                continue
            if session_id is not None and entry.get("session_id") != session_id:
                continue
            entry_session = entry.get("session_id") or "unknown"
            msg_id_raw = entry.get("msg_id")
            msg_id_str = (
                str(msg_id_raw) if msg_id_raw is not None
                else f"{entry_session}:{line_no}"
            )
            try:
                cogn_scores = entry.get("cogn_scores", {}) or {}
                sample = PulseSample(
                    timestamp=float(entry["ts"]),
                    msg_id=msg_id_str,
                    composite=float(entry.get("cogn_composite", 0.0)),
                    scores={
                        "sycophancy": float(cogn_scores.get("sycophancy", 0.0)),
                        "deception": float(cogn_scores.get("deception", 0.0)),
                        "overconfidence": float(cogn_scores.get("overconfidence", 0.0)),
                        "refusal": float(cogn_scores.get("refusal", 0.0)),
                    },
                    needs_revision=bool(entry.get("cogn_needs_revision", False)),
                    construct_ceiling_fires=list(
                        entry.get("cogn_construct_ceiling_fires", []) or []
                    ),
                )
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(
                    f"chart.jsonl line {line_no}: malformed entry: {e}"
                ) from e
            samples.append(sample)
    samples.sort(key=lambda s: s.timestamp)
    return samples


# ──────────────────────────────────────────────────────────────────
# Primary measurement (must match locked scorer)
# ──────────────────────────────────────────────────────────────────


def _zscore(xs: list[float]) -> list[float]:
    n = len(xs)
    if n == 0:
        return []
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    if var == 0.0:
        return [0.0] * n
    sd = math.sqrt(var)
    return [(x - mean) / sd for x in xs]


def _pearson_r(c_a: list[float], c_b: list[float]) -> float:
    """Pearson r — guards against degenerate variance, equal-length input."""
    if len(c_a) != len(c_b):
        raise ValueError(f"unequal lengths {len(c_a)} vs {len(c_b)}")
    n = len(c_a)
    if n < 3:
        raise ValueError(f"n={n} too small (need >= 3)")
    mean_a = sum(c_a) / n
    mean_b = sum(c_b) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(c_a, c_b)) / n
    var_a = sum((a - mean_a) ** 2 for a in c_a) / n
    var_b = sum((b - mean_b) ** 2 for b in c_b) / n
    denom = math.sqrt(var_a * var_b)
    if denom == 0.0:
        return 0.0
    return cov / denom


def align_composites(
    pulse_a: PulseTrace, pulse_b: PulseTrace
) -> tuple[list[float], list[float]]:
    """Extract z-scored composite series, kth-of-A paired with kth-of-B,
    truncated to shorter (per preregistration §4 corrigendum)."""
    n = min(len(pulse_a), len(pulse_b))
    if n < 3:
        raise ValueError(f"need >= 3 samples each; got {len(pulse_a)}, {len(pulse_b)}")
    return (
        _zscore([s.composite for s in pulse_a[:n]]),
        _zscore([s.composite for s in pulse_b[:n]]),
    )


def primary_coherence(pulse_a: PulseTrace, pulse_b: PulseTrace) -> float:
    """The locked, hypothesis-bearing CC measurement.

    Returns Pearson r at lag 0 between the z-scored composite series
    of the two pulse-traces, kth-of-A paired with kth-of-B, truncated
    to shorter. Numerically identical to the locked scorer's output.
    """
    c_a, c_b = align_composites(pulse_a, pulse_b)
    return _pearson_r(c_a, c_b)


# ──────────────────────────────────────────────────────────────────
# Exploratory: lag-sweep
# ──────────────────────────────────────────────────────────────────


def lag_sweep(
    pulse_a: PulseTrace,
    pulse_b: PulseTrace,
    lags: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3),
) -> dict[int, float]:
    """CC at each lag k: pair index i of A with index i+k of B.

    Positive lag = B trails A by k turns. Negative lag = B leads A.
    Lag 0 is the primary measurement (matches ``primary_coherence``).

    EXPLORATORY only. Lag-sweep is NOT in the preregistration; non-zero-lag
    results carry no hypothesis-test weight and are reported only as
    structural diagnostic depth.
    """
    n_a = len(pulse_a)
    n_b = len(pulse_b)
    n = min(n_a, n_b)
    if n < 5:
        raise ValueError(f"lag_sweep requires n >= 5; got {n}")

    # Extract z-scored composite series on the full-length pre-truncated
    # trace — z-scoring on the shifted subsequence per-lag would conflate
    # scale changes with shift.
    c_a_full = _zscore([s.composite for s in pulse_a])
    c_b_full = _zscore([s.composite for s in pulse_b])

    out: dict[int, float] = {}
    for k in lags:
        if k >= 0:
            sub_a = c_a_full[: n_a - k]
            sub_b = c_b_full[k:]
        else:
            sub_a = c_a_full[-k:]
            sub_b = c_b_full[: n_b + k]
        m = min(len(sub_a), len(sub_b))
        if m < 3:
            out[k] = float("nan")
            continue
        out[k] = _pearson_r(sub_a[:m], sub_b[:m])
    return out


# ──────────────────────────────────────────────────────────────────
# Exploratory: per-axis CC
# ──────────────────────────────────────────────────────────────────


PER_AXIS_INSTRUMENTS = ("sycophancy", "deception", "overconfidence", "refusal")


def per_axis_coherence(
    pulse_a: PulseTrace, pulse_b: PulseTrace
) -> dict[str, float]:
    """Pearson r at lag 0 per sub-instrument.

    EXPLORATORY only — the preregistration §3 single-channel multiple-
    comparison lock means only ``composite`` is hypothesis-bearing.
    Per-axis values are diagnostic-only; do not interpret them as
    independent positive/negative findings.
    """
    n = min(len(pulse_a), len(pulse_b))
    if n < 3:
        raise ValueError(f"need >= 3 samples each; got {len(pulse_a)}, {len(pulse_b)}")
    out: dict[str, float] = {}
    for inst in PER_AXIS_INSTRUMENTS:
        a_vals = [s.scores[inst] for s in pulse_a[:n]]
        b_vals = [s.scores[inst] for s in pulse_b[:n]]
        try:
            out[inst] = _pearson_r(_zscore(a_vals), _zscore(b_vals))
        except ValueError:
            out[inst] = float("nan")
    return out


# ──────────────────────────────────────────────────────────────────
# Exploratory: Hilbert phase-locking value
# ──────────────────────────────────────────────────────────────────


def plv_hilbert(pulse_a: PulseTrace, pulse_b: PulseTrace) -> float:
    """Phase-locking value via the Hilbert analytic signal.

    PLV(x, y) = |E[exp(i·(φ_x − φ_y))]|, where φ_x and φ_y are the
    instantaneous phases of the analytic signals of the z-scored
    composite series.

    PLV ranges [0, 1]: 0 = no phase relationship, 1 = perfect phase-
    locking (constant phase difference). Distinct from Pearson r — two
    signals can be uncorrelated yet phase-locked, or vice versa.

    EXPLORATORY only. This is the signal-processing definition of phase
    coherence; the preregistered operational definition is Pearson r at
    lag 0 (§4). Including this asks: if we use the signal-processing
    convention of the name, does the answer agree?

    Requires numpy + scipy.signal.hilbert.
    """
    try:
        import numpy as np
        from scipy.signal import hilbert
    except ImportError as e:
        raise ImportError(
            "plv_hilbert requires numpy and scipy"
        ) from e
    c_a, c_b = align_composites(pulse_a, pulse_b)
    n = len(c_a)
    if n < 5:
        raise ValueError(f"plv_hilbert requires n >= 5; got {n}")
    ana_a = hilbert(np.asarray(c_a, dtype=float))
    ana_b = hilbert(np.asarray(c_b, dtype=float))
    phase_a = np.angle(ana_a)
    phase_b = np.angle(ana_b)
    plv = float(np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))
    return plv


# ──────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────


@dataclass
class CoherenceResult:
    """Per-dyad coherence measurement bundle.

    `primary_cc` is the preregistered hypothesis-bearing measurement.
    Everything else is exploratory.
    """
    primary_cc: float
    plv: float
    lag_sweep: dict[int, float]
    per_axis: dict[str, float]
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "primary_cc": self.primary_cc,
            "plv": self.plv,
            "lag_sweep": {str(k): v for k, v in self.lag_sweep.items()},
            "per_axis": dict(self.per_axis),
            "n_samples": self.n_samples,
        }


def pulse_coherence(
    pulse_a: PulseTrace, pulse_b: PulseTrace
) -> CoherenceResult:
    """Compute the full coherence bundle for one dyad.

    Primary CC is preregistered and binding; PLV, lag-sweep, and per-axis
    are exploratory companions.
    """
    n = min(len(pulse_a), len(pulse_b))
    return CoherenceResult(
        primary_cc=primary_coherence(pulse_a, pulse_b),
        plv=plv_hilbert(pulse_a, pulse_b),
        lag_sweep=lag_sweep(pulse_a, pulse_b),
        per_axis=per_axis_coherence(pulse_a, pulse_b),
        n_samples=n,
    )


# ──────────────────────────────────────────────────────────────────
# Inter-agent coherence — EMBEDDING-TRAJECTORY channel (drift-axis)
# ──────────────────────────────────────────────────────────────────
#
# This is a different channel from pulse_coherence (which reads the
# cogn-composite register series, lag-0 Pearson r — that channel
# closed-negative). This reads the GEOMETRY of two agents' per-turn
# response embeddings: do their overall trajectory directions in latent
# space point the same way?
#
# Numerically identical to scripts/drift_axis_scorer.py::drift_axis_alignment,
# locked at commit 79906b4 (§8 of drift_axis_alignment_preregistration_2026_05_21).
# Parity is enforced by tests/test_embedding_trajectory_parity.py — if this
# diverges from the locked scorer, the tool no longer computes the quantity
# the preregistration scored, and the test fails.


def embedding_trajectory_alignment(embs_a, embs_b) -> float:
    """Drift-axis alignment between two agents' per-turn embedding trajectories.

    Given two (n, d) arrays of per-turn response embeddings (kth-of-A paired
    with kth-of-B, truncated to shorter), returns the cosine between each
    agent's first-half-to-second-half centroid-drift vector. Range [-1, +1].

    MEASUREMENT, NOT INTERPRETATION. In a preregistered N=20+20 corpus
    (deposit fa24373) cooperative dyads scored ~0.79 vs ~0.40 for adversarial,
    p < 0.001, both embedding families, independently reproduced. BUT whether
    that difference reflects *cooperation* or merely *topic convergence* is
    pending the preregistered topic-control 2x2 (topic_control_preregistration_
    2026_05_22). Until that clears, report this as "embedding-trajectory
    alignment," not "cognitive coupling." See drift_axis_threats_to_validity.md.

    Inputs are expected L2-normalized (the embedding providers normalize).
    Requires numpy.
    """
    import numpy as np
    a = np.asarray(embs_a, dtype=float)
    b = np.asarray(embs_b, dtype=float)
    n = min(a.shape[0], b.shape[0])
    if n < 4:
        return float("nan")
    half = n // 2
    a_dir = a[half:n].mean(0) - a[:half].mean(0)
    b_dir = b[half:n].mean(0) - b[:half].mean(0)
    a_norm = np.linalg.norm(a_dir)
    b_norm = np.linalg.norm(b_dir)
    if a_norm < 1e-12 or b_norm < 1e-12:
        return float("nan")
    return float((a_dir / a_norm) @ (b_dir / b_norm))


__all__ = [
    "PulseSample",
    "PulseTrace",
    "CoherenceResult",
    "load_pulse_trace",
    "align_composites",
    "primary_coherence",
    "lag_sweep",
    "per_axis_coherence",
    "plv_hilbert",
    "pulse_coherence",
    "embedding_trajectory_alignment",
]
