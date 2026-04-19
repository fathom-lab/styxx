# -*- coding: utf-8 -*-
"""
N-sample consensus proxy for logprobs.

Runs the same prompt N times at temperature T > 0, then computes per-
position token agreement across the N chains. Converts empirical
agreement into a proxy entropy / top-2 margin / logprob trajectory and
feeds it to the styxx classifier.

Mock mode generates synthetic ensemble samples for offline testing
(no API key required).
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text or "")


@dataclass
class ConsensusTrajectory:
    n_samples: int
    max_len: int
    agreement: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    proxy_logprob: List[float] = field(default_factory=list)
    proxy_top2_margin: List[float] = field(default_factory=list)
    first_divergence: int = -1

    def to_trajectories(self) -> Dict[str, List[float]]:
        return {
            "entropy": list(self.entropy),
            "logprob": list(self.proxy_logprob),
            "top2_margin": list(self.proxy_top2_margin),
        }


def compute_trajectory(samples: Sequence[str]) -> ConsensusTrajectory:
    chains = [_tokenize(s) for s in samples]
    n = len(chains)
    if n == 0:
        return ConsensusTrajectory(n_samples=0, max_len=0)
    max_len = max((len(c) for c in chains), default=0)

    agree: List[float] = []
    ents: List[float] = []
    plp: List[float] = []
    pm2: List[float] = []
    first_div = -1

    for i in range(max_len):
        col = [c[i] for c in chains if i < len(c)]
        if not col:
            break
        counts = Counter(col)
        total = sum(counts.values())
        sorted_counts = counts.most_common()
        modal = sorted_counts[0][1]
        second = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        H = -sum((v / total) * math.log(v / total) for v in counts.values())
        p_mode = modal / total
        ents.append(H)
        agree.append(p_mode)
        # proxy logprob = log(p_mode) — mirrors what a logprob on the
        # chosen token would look like
        plp.append(math.log(max(p_mode, 1e-9)))
        pm2.append((modal - second) / total)
        if first_div == -1 and modal < total:
            first_div = i

    return ConsensusTrajectory(
        n_samples=n,
        max_len=max_len,
        agreement=agree,
        entropy=ents,
        proxy_logprob=plp,
        proxy_top2_margin=pm2,
        first_divergence=first_div,
    )


# ---------- mock sampler ----------

def mock_sampler(prompt: str, n: int = 5, *, seed: Optional[int] = None,
                 divergence: float = 0.3, length: int = 30) -> List[str]:
    """Generate N synthetic samples that share a common prefix then
    diverge stochastically. Deterministic when seed is set."""
    rng = random.Random(seed if seed is not None else hash(prompt) & 0xFFFFFFFF)
    # shared prefix (first ~1/3 tokens)
    prefix_len = max(1, int(length * (1.0 - divergence)))
    base_vocab = ["the", "a", "it", "is", "was", "and", "of", "to", "in",
                  "that", "this", "we", "can", "be", "very", "model",
                  "token", "answer"]
    prefix = [rng.choice(base_vocab) for _ in range(prefix_len)]
    samples: List[str] = []
    for _ in range(n):
        tail = [rng.choice(base_vocab) for _ in range(length - prefix_len)]
        samples.append(" ".join(prefix + tail))
    return samples


# ---------- high level ----------

def run_consensus(
    prompt: str,
    *,
    n: int = 5,
    sampler: Optional[Callable[[str], str]] = None,
    mock: bool = False,
    mock_divergence: float = 0.3,
    mock_length: int = 30,
    mock_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run an N-sample consensus either via a real sampler callable or
    the mock sampler.

    sampler: callable(prompt) -> str. Invoked N times.
    """
    if mock or sampler is None:
        samples = mock_sampler(prompt, n=n, seed=mock_seed,
                               divergence=mock_divergence,
                               length=mock_length)
    else:
        samples = [sampler(prompt) for _ in range(n)]

    traj = compute_trajectory(samples)
    return {
        "samples": samples,
        "trajectory": traj,
        "mode": "consensus-mock" if (mock or sampler is None) else "consensus",
    }


def build_vitals(result: Dict[str, Any]):
    """Feed a consensus trajectory into the styxx phase classifier.

    Uses the shipped CentroidClassifier when available. Falls back to
    an entropy-threshold heuristic when the classifier can't be loaded
    or a phase has too few tokens.
    """
    from ..vitals import (
        Vitals, PhaseReading, load_centroids, PHASE_ORDER,
        PHASE_TOKEN_CUTOFFS,
    )

    traj = result["trajectory"]
    trajectories = traj.to_trajectories()
    n_tokens = traj.max_len

    def _heuristic_reading(phase: str, n_used: int) -> PhaseReading:
        import statistics as _s
        ents = trajectories["entropy"][:n_used] if n_used else []
        H = _s.mean(ents) if ents else 0.0
        if H < 0.2:
            pred = "retrieval"
        elif H < 0.8:
            pred = "reasoning"
        else:
            pred = "creative"
        cats = ["retrieval", "reasoning", "refusal",
                "creative", "adversarial", "hallucination"]
        probs = {c: 0.05 for c in cats}
        probs[pred] = 0.75
        return PhaseReading(
            phase=phase,
            n_tokens_used=n_used,
            features=[H],
            predicted_category=pred,
            margin=0.2,
            distances={c: (1.0 - p) * 5.0 for c, p in probs.items()},
            probs=probs,
        )

    if n_tokens == 0:
        reading = _heuristic_reading("consensus-empty", 0)
        v = Vitals(phase1_pre=reading, phase4_late=reading, tier_active=-1)
        try:
            v.mode = result["mode"]  # type: ignore[attr-defined]
        except Exception:
            pass
        return v

    # Try the shipped centroid classifier
    clf = None
    try:
        clf = load_centroids()
    except Exception:
        clf = None

    readings: Dict[str, PhaseReading] = {}
    for phase in PHASE_ORDER:
        cutoff = PHASE_TOKEN_CUTOFFS[phase]
        if n_tokens < cutoff:
            readings[phase] = None  # type: ignore[assignment]
            continue
        if clf is not None:
            try:
                readings[phase] = clf.classify(trajectories, phase)
                continue
            except Exception:
                pass
        readings[phase] = _heuristic_reading(phase, min(cutoff, n_tokens))

    # ensure phase1 exists
    p1 = readings.get("phase1_preflight") or _heuristic_reading(
        "phase1_preflight", min(1, n_tokens))
    p4 = readings.get("phase4_late") or p1
    vits = Vitals(
        phase1_pre=p1,
        phase2_early=readings.get("phase2_early"),
        phase3_mid=readings.get("phase3_mid"),
        phase4_late=p4,
        tier_active=-1,
    )
    try:
        vits.mode = result["mode"]  # type: ignore[attr-defined]
    except Exception:
        pass
    return vits


__all__ = ["run_consensus", "compute_trajectory", "mock_sampler",
           "build_vitals", "ConsensusTrajectory"]
