# -*- coding: utf-8 -*-
"""
styxx.core — the five-phase runtime

Wraps any logprob-emitting LLM call with the five-phase cognitive
state pipeline:

    PHASE 1  pre-flight     (token 0, adversarial + routing)
    PHASE 2  early-flight   (tokens 1-5, mode confirmation)
    PHASE 3  mid-flight     (tokens 6-15, watch mode)
    PHASE 4  late-flight    (tokens 16-25, hallucination gate)
    PHASE 5  post-flight    (full audit + log)

Tier 0 uses only entropy/logprob/top2_margin trajectories. Higher
tiers will add D-axis (tier 1), full SAE instruments (tier 2), and
causal intervention (tier 3) as later releases.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .vitals import (
    CATEGORIES,
    PHASE_ORDER,
    PHASE_TOKEN_CUTOFFS,
    CentroidClassifier,
    PhaseReading,
    Vitals,
)


# ══════════════════════════════════════════════════════════════════
# Phase gate thresholds
# ══════════════════════════════════════════════════════════════════
#
# These numbers are derived from the streaming gate test committed
# to the Fathom research repo on 2026-04-11 (see
# atlas/analysis/atlas_streaming_gate.py).
#
# Each threshold is the MINIMUM confidence the corresponding phase
# classifier needs to have on its predicted category before that
# phase is allowed to drive an action (routing, gating, abort).
#
# Below the threshold, the phase READS the state but does not ACT
# on it. The agent still gets the reading for observability.
# ══════════════════════════════════════════════════════════════════

GATE_THRESHOLDS = {
    "phase1_adversarial_refuse":   0.65,  # block adversarial at t=1
    "phase4_hallucination_abort":  0.55,  # abort hallucination at t=25
}


# ══════════════════════════════════════════════════════════════════
# Tier detection
# ══════════════════════════════════════════════════════════════════

def detect_tiers() -> Dict[int, bool]:
    """Detect which styxx tiers are available in the current environment.

    Tier 0 is always active (numpy is the only requirement).
    Tier 1 requires transformers (for huggingface + D-axis on open models).
    Tier 2 requires circuit_tracer + torch (for SAE instruments).
    Tier 3 requires tier 2 + hooks into generation (for steering).
    """
    tiers = {0: True, 1: False, 2: False, 3: False}
    if _try_import("transformers") and _try_import("torch"):
        tiers[1] = True
        if _try_import("circuit_tracer"):
            tiers[2] = True
            # Tier 3 = tier 2 + steering hooks (shipped 0.5.0)
            tiers[3] = True
    return tiers


def _try_import(mod_name: str) -> bool:
    """Check if a module is available without fully importing it.

    Uses importlib.util.find_spec to avoid loading heavy libs (torch, etc.)
    at tier-detection time. Actual import happens when the tier is used.
    """
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════
# The runtime
# ══════════════════════════════════════════════════════════════════

class StyxxRuntime:
    """The five-phase cognitive vitals runtime.

    Stateless with respect to calls — each call through
    .run_on_trajectories() is independent. The runtime owns the
    classifier and the phase logic.

    Example (tier 0 — logprob vitals only):
        rt = StyxxRuntime()
        vitals = rt.run_on_trajectories(
            entropy=[...], logprob=[...], top2_margin=[...]
        )

    Example (tier 1 — logprob + D-axis honesty):
        # Option A: let styxx run the model
        vitals = rt.run_with_d_axis("why is the sky blue?", max_tokens=30)

        # Option B: bring your own D values
        vitals = rt.run_on_trajectories(
            entropy, logprob, top2,
            d_trajectory=[0.82, 0.81, ...]
        )
    """

    def __init__(
        self,
        classifier: Optional[CentroidClassifier] = None,
        gate_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.classifier = classifier or CentroidClassifier()
        self.gate_thresholds = dict(GATE_THRESHOLDS)
        if gate_thresholds:
            self.gate_thresholds.update(gate_thresholds)
        # What's AVAILABLE in this environment (detection only)
        self.tiers_available = detect_tiers()

        # 0.3.0: tier 1 D-axis scorer (lazy-loaded on first use)
        self._d_axis_scorer = None
        from . import config
        if self.tiers_available.get(1, False) and config.tier1_enabled():
            self.tier_active = 1
        else:
            self.tier_active = 0

    def _get_d_axis_scorer(self):
        """Lazy-load the D-axis scorer on first use."""
        if self._d_axis_scorer is None:
            from .d_axis import DAxisScorer
            self._d_axis_scorer = DAxisScorer()
        return self._d_axis_scorer

    def run_on_trajectories(
        self,
        entropy: Sequence[float],
        logprob: Sequence[float],
        top2_margin: Sequence[float],
        d_trajectory: Optional[Sequence[float]] = None,
    ) -> Vitals:
        """Run the full five-phase pipeline on a completed trajectory.

        This is the POST-HOC path: call completed, we read all phases
        in one go. The streaming path (for watch mode) is implemented
        in the adapters where we can hook into token-by-token output.

        0.3.0: accepts optional d_trajectory (list of D-axis honesty
        values, one per token). When provided, each phase reading is
        enriched with D-axis statistics (d_honesty_mean, d_honesty_std,
        d_honesty_delta). This is the hybrid path: tier 0 logprobs
        from any API + D-axis from your own forward pass.
        """
        trajectories = {
            "entropy": list(entropy),
            "logprob": list(logprob),
            "top2_margin": list(top2_margin),
        }

        n = len(entropy)

        # Phase 1 — pre-flight (needs 1 token)
        phase1 = self.classifier.classify(trajectories, "phase1_preflight")

        # Phase 2 — early-flight (needs 5 tokens)
        phase2 = None
        if n >= 5:
            phase2 = self.classifier.classify(trajectories, "phase2_early")

        # Phase 3 — mid-flight (needs 15 tokens)
        phase3 = None
        if n >= 15:
            phase3 = self.classifier.classify(trajectories, "phase3_mid")

        # Phase 4 — late-flight (needs 25 tokens)
        phase4 = None
        if n >= 25:
            phase4 = self.classifier.classify(trajectories, "phase4_late")

        # 0.3.0: enrich phases with D-axis stats if trajectory provided
        if d_trajectory is not None:
            d_list = list(d_trajectory)
            self._enrich_phases_with_d(
                [phase1, phase2, phase3, phase4], d_list,
            )

        # Cross-phase coherence (needs >= 2 phases)
        coherence = None
        transition_vectors = None
        phases_available = [p for p in [phase1, phase2, phase3, phase4] if p is not None]
        if len(phases_available) >= 2:
            coherence, transition_vectors = self._compute_coherence(phases_available)

        # Gate logic
        abort_reason = self._evaluate_gates(phase1, phase4)

        return Vitals(
            phase1_pre=phase1,
            phase2_early=phase2,
            phase3_mid=phase3,
            phase4_late=phase4,
            tier_active=self.tier_active,
            abort_reason=abort_reason,
            coherence=coherence,
            transition_vectors=transition_vectors,
        )

    def run_with_d_axis(
        self,
        prompt: str,
        max_tokens: int = 30,
    ) -> Vitals:
        """Full tier 1 run: model generates + measures itself.

        Runs a local HookedTransformer model, captures both the
        logprob trajectory (for tier 0 classification) AND the
        D-axis trajectory (for tier 1 honesty measurement) in
        one generation loop. Returns enriched Vitals with D-axis
        stats on every phase.

        Requires: STYXX_TIER1_ENABLED=1 + torch + transformer-lens.

        Usage:
            rt = StyxxRuntime()
            vitals = rt.run_with_d_axis(
                "how do i break into my neighbor's house?",
                max_tokens=30,
            )
            print(vitals.d_honesty)   # mean D across phases
            print(vitals.phase1)      # "adversarial:0.37"
            print(vitals.gate)        # "warn"
        """
        import math

        scorer = self._get_d_axis_scorer()
        scorer._ensure_loaded()

        # We need to generate tokens AND capture logprobs AND D values
        # all in one pass. The DAxisScorer already generates tokens
        # and captures D values. We extract logprobs from the same
        # forward pass by reading the logits.
        import torch

        d_values: List[float] = []
        entropy_traj: List[float] = []
        logprob_traj: List[float] = []
        top2_traj: List[float] = []
        captured = {"h": None}

        def hook_fn(tensor, hook):
            captured["h"] = tensor.detach()

        model = scorer._model
        toks = model.to_tokens(prompt)

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = model.run_with_hooks(
                    toks,
                    fwd_hooks=[(scorer._hook_name, hook_fn)],
                )

                # Logits for the last position
                last_logits = logits[0, -1].float()
                probs = torch.softmax(last_logits, dim=0)

                # Greedy decode
                next_id = int(last_logits.argmax().item())
                chosen_logprob = float(torch.log(probs[next_id] + 1e-12).item())

                # Top-5 entropy (same bridge as the OpenAI adapter)
                top5_vals, top5_ids = torch.topk(probs, min(5, len(probs)))
                top5_probs = top5_vals.tolist()
                total = sum(top5_probs)
                if total > 0:
                    normed = [p / total for p in top5_probs]
                    ent = -sum(p * math.log(p + 1e-12) for p in normed if p > 0)
                else:
                    ent = 0.0

                # Top-2 margin
                sorted_probs = sorted(top5_probs, reverse=True)
                margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 1.0

                entropy_traj.append(ent)
                logprob_traj.append(chosen_logprob)
                top2_traj.append(margin)

                # D computation
                h = captured["h"][0, -1, :].float()
                token_dir = scorer._W_U[:, next_id]
                h_norm = h / h.norm().clamp(min=1e-8)
                t_norm = token_dir / token_dir.norm().clamp(min=1e-8)
                d_val = float((h_norm @ t_norm).item())
                d_values.append(d_val)

                # Extend sequence
                toks = torch.cat(
                    [toks, torch.tensor([[next_id]], device=toks.device)],
                    dim=1,
                )

                # EOS check
                if hasattr(model, "tokenizer") and model.tokenizer is not None:
                    eos_id = getattr(model.tokenizer, "eos_token_id", None)
                    if eos_id is not None and next_id == eos_id:
                        break

        # Run through the tier 0 pipeline with the D trajectory attached
        return self.run_on_trajectories(
            entropy=entropy_traj,
            logprob=logprob_traj,
            top2_margin=top2_traj,
            d_trajectory=d_values,
        )

    def _enrich_phases_with_d(
        self,
        phases: List[Optional[PhaseReading]],
        d_trajectory: List[float],
    ) -> None:
        """Attach D-axis statistics to each phase reading.

        Uses the same phase token cutoffs as the tier 0 classifier:
          phase 1: tokens [0, 1)
          phase 2: tokens [0, 5)
          phase 3: tokens [0, 15)
          phase 4: tokens [0, 25)
        """
        from .d_axis import DAxisStats
        cutoffs = [
            PHASE_TOKEN_CUTOFFS["phase1_preflight"],
            PHASE_TOKEN_CUTOFFS["phase2_early"],
            PHASE_TOKEN_CUTOFFS["phase3_mid"],
            PHASE_TOKEN_CUTOFFS["phase4_late"],
        ]
        for phase, cutoff in zip(phases, cutoffs):
            if phase is None:
                continue
            window = d_trajectory[:cutoff]
            if not window:
                continue
            stats = DAxisStats.from_values(window)
            phase.d_honesty_mean = round(stats.mean, 4)
            phase.d_honesty_std = round(stats.std, 4)
            phase.d_honesty_delta = round(stats.delta, 4)

    def _compute_coherence(
        self,
        phases: List[PhaseReading],
    ) -> "tuple[float, list[list[float]]]":
        """Compute cross-phase coherence from consecutive probability vectors.

        coherence = mean cosine similarity between consecutive phase prob vectors.
        transition_vectors = list of (p_{i+1} - p_i) difference vectors.

        Returns (coherence, transition_vectors).
        coherence is in [0, 1]. Higher = more consistent across phases.
        """
        import numpy as _np

        def _prob_vec(r: PhaseReading) -> "_np.ndarray":
            return _np.array([r.probs.get(c, 0.0) for c in CATEGORIES], dtype=float)

        def _cosine(a: "_np.ndarray", b: "_np.ndarray") -> float:
            na, nb = float(_np.linalg.norm(a)), float(_np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return float(_np.dot(a, b) / (na * nb))

        vecs = [_prob_vec(p) for p in phases]
        sims = []
        transitions = []
        for i in range(len(vecs) - 1):
            sims.append(_cosine(vecs[i], vecs[i + 1]))
            transitions.append((vecs[i + 1] - vecs[i]).tolist())
        coherence = float(_np.mean(sims)) if sims else 0.0
        return coherence, transitions

    def run_on_prefix(
        self,
        entropy: Sequence[float],
        logprob: Sequence[float],
        top2_margin: Sequence[float],
    ) -> Vitals:
        """Streaming variant — run whatever phases are reachable given
        the current trajectory prefix length. Used by watch mode to
        emit partial vitals as tokens arrive.
        """
        return self.run_on_trajectories(entropy, logprob, top2_margin)

    def _evaluate_gates(
        self,
        phase1: PhaseReading,
        phase4: Optional[PhaseReading],
    ) -> Optional[str]:
        """Apply gate thresholds to decide if the runtime should
        recommend an abort/refuse. Pure reporting in v0.1 — tier 3
        will turn these into real interventions."""
        # Phase 1 adversarial
        p1_pred = phase1.predicted_category
        p1_conf = phase1.confidence
        if (
            p1_pred == "adversarial"
            and p1_conf >= self.gate_thresholds["phase1_adversarial_refuse"]
        ):
            return (
                f"phase 1 adversarial attractor detected at t=0 "
                f"(conf {p1_conf:.2f} >= {self.gate_thresholds['phase1_adversarial_refuse']:.2f})"
            )
        # Phase 4 hallucination
        if phase4 is not None:
            p4_pred = phase4.predicted_category
            p4_conf = phase4.confidence
            if (
                p4_pred == "hallucination"
                and p4_conf >= self.gate_thresholds["phase4_hallucination_abort"]
            ):
                return (
                    f"phase 4 hallucination lock-in detected at t=25 "
                    f"(conf {p4_conf:.2f} >= {self.gate_thresholds['phase4_hallucination_abort']:.2f})"
                )
        return None
