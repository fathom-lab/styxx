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
            # Tier 3 is tier 2 + generation hooks; shipped as v0.4+
            tiers[3] = False
    return tiers


def _try_import(mod_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════
# The runtime
# ══════════════════════════════════════════════════════════════════

class StyxxRuntime:
    """The five-phase cognitive vitals runtime.

    Stateless with respect to calls — each call through
    .run_on_trajectories() is independent. The runtime owns the
    classifier and the phase logic.

    Example:
        from styxx.core import StyxxRuntime
        rt = StyxxRuntime()
        vitals = rt.run_on_trajectories(
            entropy=[...], logprob=[...], top2_margin=[...]
        )
    """

    # Which tier is actually USED by this runtime at inference time.
    # v0.1 ships only the tier 0 classifier. Higher tiers will be
    # wired in at v0.2+ and this number will advance accordingly.
    TIER_IN_USE = 0

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
        # What's actually running RIGHT NOW (always 0 in v0.1)
        self.tier_active = self.TIER_IN_USE

    def run_on_trajectories(
        self,
        entropy: Sequence[float],
        logprob: Sequence[float],
        top2_margin: Sequence[float],
    ) -> Vitals:
        """Run the full five-phase pipeline on a completed trajectory.

        This is the POST-HOC path: call completed, we read all phases
        in one go. The streaming path (for watch mode) is implemented
        in the adapters where we can hook into token-by-token output.
        """
        trajectories = {
            "entropy": list(entropy),
            "logprob": list(logprob),
            "top2_margin": list(top2_margin),
        }

        # Each phase only fires when the trajectory covers its full
        # window. This is the strict-window policy for v0.1. A
        # streaming adapter with partial-window reads will land in v0.2.
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

        # Gate logic (tier 3 only, but we report the decision for tier 0 too)
        abort_reason = self._evaluate_gates(phase1, phase4)

        return Vitals(
            phase1_pre=phase1,
            phase2_early=phase2,
            phase3_mid=phase3,
            phase4_late=phase4,
            tier_active=self.tier_active,
            abort_reason=abort_reason,
        )

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
