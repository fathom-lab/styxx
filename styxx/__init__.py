# -*- coding: utf-8 -*-
"""
styxx — nothing crosses unseen.

a fathom lab product.

The first drop-in cognitive vitals monitor for LLM agents. Reads an
agent's internal state in real time using signals available on any
LLM with a logprob interface (entropy, logprob, top-2 margin),
calibrated cross-architecture on the Fathom Cognitive Atlas v0.3.

Quickstart:

    # Before: from openai import OpenAI
    from styxx import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(model="gpt-4o", messages=[...])

    print(r.choices[0].message.content)   # text, unchanged
    print(r.vitals.phase1_pre)            # pre-flight cognitive state
    print(r.vitals.phase4_late)           # late-flight hallucination read
    print(r.vitals.summary)               # human-readable vitals card

Fail-open: if styxx can't read vitals for any reason, the underlying
SDK call returns its normal response unchanged. styxx never breaks
the user's existing agent.

Honest specs at tier 0 (cross-model LOO, chance = 0.167):
  - phase 1 adversarial     acc 0.52  @ t=1
  - phase 1 reasoning       acc 0.43  @ t=1
  - phase 4 hallucination   acc 0.52  @ t=25
  - phase 4 reasoning       acc 0.69  @ t=25

This is an instrument panel, not a fortune teller.

Research: https://github.com/fathom-lab/fathom
Patents:  US Provisional 64/020,489 · 64/021,113 · 64/026,964
License:  MIT (code), CC-BY-4.0 (atlas data)
"""

# Read version from installed package metadata so the attribute can
# never drift from the published wheel. Falls back to a literal for
# environments where the package isn't installed (e.g. running directly
# from a checkout without `pip install -e .`).
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound
    try:
        __version__ = _pkg_version("styxx")
    except _PkgNotFound:
        __version__ = "0.0.0+source"
    del _pkg_version, _PkgNotFound
except ImportError:  # pragma: no cover — Python < 3.8
    __version__ = "0.0.0+source"

__author__ = "flobi"
__license__ = "MIT"
__url__ = "https://fathom.darkflobi.com/styxx"
__tagline__ = "nothing crosses unseen."


# ── Windows console encoding safeguard ──────────────────────────────
#
# styxx emits Unicode box-drawing characters, sparkline blocks, and
# status glyphs. On legacy Windows consoles running cp1252 the default
# stdout can't encode them and `print()` crashes with UnicodeEncodeError.
#
# We reconfigure stdout/stderr to utf-8 on import whenever possible.
# Python 3.7+ stdio streams expose .reconfigure(). We swallow any
# failure silently — if we can't reconfigure, the user can still set
# PYTHONIOENCODING=utf-8 manually, or pipe output through a tool that
# handles encoding, or set STYXX_NO_COLOR=1 and live with the crash
# only if it actually happens. Fail-open: never block import.
def _auto_reconfigure_stdio():
    import sys as _sys
    for stream_name in ("stdout", "stderr"):
        stream = getattr(_sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            encoding = (getattr(stream, "encoding", "") or "").lower()
            # Only reconfigure if we're on a legacy codec that can't
            # handle the chars we emit. Don't touch utf-8 streams.
            if encoding and "utf" not in encoding:
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_auto_reconfigure_stdio()
del _auto_reconfigure_stdio


def OpenAI(*args, **kwargs):
    """Drop-in replacement for openai.OpenAI that emits cognitive vitals.

    Every response gains a .vitals attribute alongside the normal
    .choices. Fails open: if the wrapper can't read vitals, the
    underlying openai call returns its normal response.

    Usage:
        from styxx import OpenAI
        client = OpenAI()  # same interface as openai.OpenAI
        r = client.chat.completions.create(...)
        print(r.vitals.summary)
    """
    from .adapters.openai import OpenAIWithVitals
    return OpenAIWithVitals(*args, **kwargs)


def Raw(*args, **kwargs):
    """Adapter for users who already have a logprob trajectory.

    Usage:
        from styxx import Raw
        styxx = Raw()
        vitals = styxx.read(
            entropy=[...],    # per-token entropy trajectory
            logprob=[...],    # per-token chosen-token logprob
            top2_margin=[...],
        )
        print(vitals.summary)
    """
    from .adapters.raw import RawAdapter
    return RawAdapter(*args, **kwargs)


def Anthropic(*args, **kwargs):
    """Drop-in pass-through wrapper around anthropic.Anthropic with
    text-heuristic vitals by default.

    Usage:
        # Before: from anthropic import Anthropic
        from styxx import Anthropic
        client = Anthropic()                       # mode='text' (default)
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "..."}],
        )
        print(r.content[0].text)                   # normal anthropic response
        print(r.vitals.phase4_late.predicted_category)
                                                   # e.g. 'reasoning'
        print(r.vitals.mode)                       # 'text-heuristic'

    Five modes (passed as ``mode=...``):

      - 'text' (default): text-heuristic vitals from
        styxx.watch._classify_from_text (tier=-1).
      - 'off': vitals=None on every call.
      - 'consensus': N-sample ensemble for logprob-equivalent signal
        (cost: N×). Configure via ``ensemble_n=`` / ``ensemble_temperature=``.
      - 'companion': open-model classifier routes the prompt
        (requires torch + a local companion model).
      - 'hybrid': text-heuristic baseline + companion overlay.

    Tier-0 logprob-based vitals are not available on the Anthropic
    Messages API (no `logprobs=True` parameter exists). The four
    non-'off' modes above are the honest workarounds. For full tier-0
    fidelity on Claude inference, route through an OpenAI-compatible
    gateway exposing logprobs (e.g. OpenRouter) via
    styxx.OpenAI(base_url=...).

    The default mode's text-heuristic inherits the 7.4.1-documented
    construct ceilings on register-detector axes (overconfidence,
    reference-less deception); when you post-process the response
    through styxx.preflight(...), those caveats surface inline via
    PreflightResult.construct_ceiling_fires.
    """
    from .adapters.anthropic import AnthropicWithVitals
    return AnthropicWithVitals(*args, **kwargs)


def LangChain(*args, **kwargs):
    """LangChain callback handler that attaches cognitive vitals.

    Usage with any LangChain LLM:
        from styxx.adapters.langchain import StyxxCallbackHandler
        handler = StyxxCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler])
        r = llm.invoke("why is the sky blue?")
        print(handler.last_vitals.summary)

    Or use the shorthand:
        handler = styxx.LangChain()
    """
    from .adapters.langchain import StyxxCallbackHandler
    return StyxxCallbackHandler(*args, **kwargs)


def CrewAI(crew=None):
    """Inject styxx observation into a CrewAI Crew.

    Usage:
        from styxx.adapters.crewai import styxx_crew
        crew = styxx_crew(Crew(agents=[...], tasks=[...]))
        crew.kickoff()
        print(crew._styxx_callback.vitals_log)

    Or use the shorthand:
        styxx.CrewAI(crew)
    """
    from .adapters.crewai import styxx_crew
    if crew is not None:
        return styxx_crew(crew)
    return styxx_crew


def LangSmith(*args, **kwargs):
    """LangChain callback handler that injects vitals into LangSmith traces.

    Usage:
        from styxx.adapters.langsmith import StyxxLangSmithHandler
        handler = StyxxLangSmithHandler()
        llm = ChatOpenAI(callbacks=[handler])
        r = llm.invoke("why is the sky blue?")
        # vitals appear as flat metadata on LangSmith runs:
        #   styxx_phase4_category: "reasoning"
        #   styxx_gate: "pass"

    Or use the shorthand:
        handler = styxx.LangSmith()
    """
    from .adapters.langsmith import StyxxLangSmithHandler
    return StyxxLangSmithHandler(*args, **kwargs)


def Langfuse(*args, **kwargs):
    """LangChain callback handler that posts vitals as Langfuse scores.

    Usage:
        from styxx.adapters.langfuse import StyxxLangfuseHandler
        handler = StyxxLangfuseHandler()
        llm = ChatOpenAI(callbacks=[handler])
        r = llm.invoke("why is the sky blue?")
        # vitals appear as Langfuse scores:
        #   styxx_gate: 1.0
        #   styxx_phase4_confidence: 0.45

    Or use the shorthand:
        handler = styxx.Langfuse()
    """
    from .adapters.langfuse import StyxxLangfuseHandler
    return StyxxLangfuseHandler(*args, **kwargs)


def AutoGen(agent=None):
    """Wrap an AutoGen agent with styxx observation.

    Usage:
        from styxx.adapters.autogen import styxx_agent
        agent = styxx_agent(AssistantAgent("helper", llm_config=...))

    Or use the shorthand:
        styxx.AutoGen(agent)
    """
    from .adapters.autogen import styxx_agent
    if agent is not None:
        return styxx_agent(agent)
    return styxx_agent


# Public API
from .core import StyxxRuntime
from .vitals import Vitals, CentroidClassifier, is_agent_mode
from .errors import (
    StyxxError, StyxxConfigError, StyxxModelError, StyxxVitalsError,
)
from . import schema  # noqa: F401  (expose styxx.schema)
from .watch import watch, observe, observe_raw, is_concerning, WatchSession
from .gates import on_gate, remove_gate, clear_gates, list_gates
from .reflex import reflex, rewind, abort, ReflexSession, ReflexSignal, RewindSignal, AbortSignal
import sys as _sys
_reflex_mod = _sys.modules["styxx.reflex"]
reflex.heal = _reflex_mod.heal
reflex.should_heal = _reflex_mod.should_heal
reflex.HealResult = _reflex_mod.HealResult
reflex.HEAL_SYSTEM_PROMPT = _reflex_mod.HEAL_SYSTEM_PROMPT
del _sys, _reflex_mod
from .guardian import guardian, GuardianSession, SteeringEvent
from .weather import weather, WeatherReport
from .dashboard import dashboard
from .calibrate import calibrate, calibration_status, CalibrationResult
from .fleet import (
    list_agents, fleet_summary, best_agent_for,
    AgentProfile, FleetSummary,
)
# NOTE: fleet.compare_agents is intentionally NOT imported at the top level.
# styxx.compare_agents is the population-percentile comparator from
# .compare (see below); fleet's fleet-routing comparator stays reachable as
# styxx.fleet.compare_agents.
from .memory import (
    remember, recall, memories, memory_stats,
    Memory, RecallResult,
)
from .handoff import (
    ProtocolEnvelope, Vitals as HandoffVitals, HandoffValidationError,
    PROTOCOL as HANDOFF_PROTOCOL,
    COGNITIVE_CLASSES,
    from_handshake_envelope, to_handshake_envelope,
)
# NOTE: handoff.Vitals (the 7-field protocol wire-snapshot) is aliased to
# HandoffVitals so it does not clobber styxx.Vitals — the classifier-output
# dataclass from .vitals (imported above) that r.vitals returns and that
# __all__ advertises as the core data type.
from .handshake import (
    handoff, receive,
    HandoffEnvelope,
)
from .sla import (
    assert_healthy, check_health, cognitive_sla,
    CognitiveSLAViolation, SLAReport,
)
from .compliance import compliance_report, ComplianceReport
from .probe import probe, ProbeReport
from .gate import gate, GateVerdict  # v3.4.0: pre-flight cognitive verdict
from .notify import on_anomaly, notify_on_fail, clear_notifications, CognitiveEvent
from .optimize import optimize
from .ci import regression_test, create_baseline, Baseline, RegressionResult
from .provenance import certify, verify as verify_certificate, CognitiveCertificate, VerificationResult
# NOTE: provenance.verify (certificate verifier → VerificationResult.valid) is
# aliased to verify_certificate so it does not clobber styxx.verify — the
# trust-layer response verifier (→ Verdict) from .verify (imported below).
from .diff import compare_sessions, compare_windows, ComparisonDiff
from .learned_classifier import train_text_classifier, TrainResult
from .autoboot import autoboot
from . import stream  # noqa: F401  (expose styxx.stream)
from .stream import claim_agent, dashboard_url, ClaimError  # noqa: F401
from .timeline import timeline, Timeline
from .conversation import conversation, ConversationResult
from .sentinel import sentinel, get_sentinel, Sentinel, SentinelAlert
from .compare import compare_agents, AgentComparison
from .antipatterns import antipatterns, AntiPattern
from .config import set_mood, current_mood_override, gate_multiplier
from .config import set_context, current_context
from .config import expect, unexpect, expected_categories, clear_expected
from .eval import EvalSuite, EvalResult, EvalFixture, compare_evals
from .trajectory import slope, curvature, volatility, extract_shape_features
from .forecast import CognitiveForecaster, ForecastResult, ForecastGate, horizon_analysis
from .intercept import CognitiveIntercept, should_intercept, simulate_intercept, InterceptReport
from .temperature import measure_temperature, aggregate_temperature, TruthMap, demo_temperature
from .verify import verify, Verdict
from .critique import critique_detector, CritiqueDetector  # 7.7.10: first-PASS detector
from .audit import audit_claim, ClaimAudit  # 7.7.13: productized single-call honesty audit
from .audit import audit_session, SessionAudit  # 7.7.13: multi-claim session-level audit
from .audit import retrieval_check, RetrievalVerdict  # 7.7.15: retrieval arm (external-grounding lever)
from .spec_exec import (  # 7.10.0: regime-1 integrity-gated routing (held-out validated 2026-06-01)
    EpistemicSpeculativeRouter, RouteResult, Draft, calibrate_threshold,
)
from . import agent_audit  # noqa: F401  # 7.7.10: L5 instrument (FINDING_agent_claim_audit_2026_05_28.md)
from .agent_audit import Claim, AuditResult, AgentClaimAuditor  # 7.7.10: L5 public surface
from .agent_audit import extract_claims, ExtractionReport  # 7.7.10: prose->claim falsification
from . import attestation  # noqa: F401  # 7.7.11: Verifiable Cognometric Attestation
from .attestation import attest, verify_attestation, Attestation, VerificationResult
from .attestation import attest_chain, verify_chain, AttestationChain, ChainVerificationResult
from . import transparency  # noqa: F401  # 7.7.12: Cognometric Transparency Log (RFC 6962)
from .transparency import TransparencyLog
from . import redact  # noqa: F401  # 7.7.12: Redactable Cognometric Attestation (selective disclosure)
from .redact import redactable_commit, disclose, verify_disclosure
from . import community  # noqa: F401
from .community import recommend  # noqa: F401
from . import meaning_integrity  # noqa: F401  # 7.11.0: does a model MEAN what a human means? (concept-geometry vs human reference)
from .meaning_integrity import (  # 7.11.0: machine-side meaning-integrity monitor (validated/generalized/real-drift)
    MeaningReference, MeaningVitalSign,
    meaning_alignment, meaning_dispersion, per_concept_alignment, meaning_integrity_report,
)

# 7.4.2: install-time diagnostic accessible programmatically (the
# `styxx doctor` CLI subcommand was the only entry point until now).
# Exposed under the function's own name `run_doctor` so callers can do
# `styxx.run_doctor()`; the `doctor` submodule itself (and all its
# internals like `_check_*` and `Path`) remains reachable as
# `from styxx import doctor`, matching the existing test-suite pattern.
from .doctor import run_doctor  # noqa: F401

# 7.4.2: one-liner pre-ship cognometric audit of a draft response. Wraps
# the MCP `cogn_audit_with_advice` tool as a typed Python function and
# surfaces construct-ceiling caveats inline (honest-scoping in code, the
# runtime-API extension of the 7.4.1 README correction).
from .preflight import preflight, PreflightResult, PreflightAdvice

# 7.4.2: agent-side cognitive-integrity persistence. Reads recent
# chart.jsonl history and produces a structured posture summary an
# agent can use to re-anchor operating state across context-compaction
# boundaries — the first styxx primitive designed specifically for
# the AI agents that use styxx (not for the humans observing them).
from .recover import recover_posture, PostureSummary

# 7.4.2: runtime cognometric audit during streaming generation. Stateful
# session the agent feeds chunks to; audits partial response periodically;
# exposes the latest audit so the agent can short-circuit on
# needs_revision before generation completes. Vendor-neutral.
from .streaming_preflight import (
    streaming_preflight, StreamingPreflightSession,
)

# 7.4.3+: phase-coherence measurement between two agents' pulse-traces.
# Implements the locked operational definition (Pearson r at lag 0 on
# z-scored composite series) from the preregistration at commit 3473523,
# plus exploratory companions (lag-sweep, per-axis CC, Hilbert PLV).
# The primary CC is numerically identical to scripts/phase_coherence_pilot.py
# at commit 23b7912. See styxx/coherence.py for the contract.
from .coherence import (
    pulse_coherence, primary_coherence, lag_sweep, per_axis_coherence,
    plv_hilbert, load_pulse_trace as load_coherence_trace,
    CoherenceResult, PulseSample, PulseTrace,
    embedding_trajectory_alignment,
)

# 7.4.3+: cognometric self-audit middleware for agent send-paths.
# Plugs into a host runtime's pre-send hook: audits each outbound draft,
# optionally calls the host's revise function on cognometric firings,
# returns the chosen draft to ship per the latest-passing /
# lowest-composite-failure / degradation-guard decision rule.
# Reference implementation of the F10 reflex loop applied to
# in-production agent output. Host supplies llm_revise; this module
# never calls an LLM itself.
from .middleware import cogn_audit_on_send, AuditTrajectory, ReviseFn

# v3.5.0+: Cognitive Instruction Set — programmable residual-stream
# control. `steer` is the multi-concept composer; `cogvm` is the
# declarative VM with WRITE/GENERATE/WATCH/HALT/RETRY/SWITCH opcodes.
from .steer import steer, steered_generate, SteerHandle
from .cogvm import (
    Program, ProgramResult,
    WRITE, GENERATE, WATCH,
    HALT, RETRY, SWITCH,
)

# v3.7.1: one-function residual-probe-gated generation
from .generate_safe import generate_safe, SafeResponse

# v3.9.0: the trust layer — one decorator, any LLM call, verified output
from .trust import trust, TrustViolation, TrustResult, is_trusted

# ── Zero-config plug-and-play ──────────────────────────────────
#
# If STYXX_AGENT_NAME is set in the environment, styxx boots
# automatically on import. No code changes needed. Just:
#
#   export STYXX_AGENT_NAME=xendro
#   export STYXX_AUTO_HOOK=1          # optional: auto-wrap openai
#   pip install styxx
#   python my_agent.py                # styxx is running. done.
#
# The agent code doesn't need to import styxx, call autoboot(),
# or do anything. If openai is installed and STYXX_AUTO_HOOK=1,
# every openai.OpenAI() call gets vitals automatically.
#
# This is true plug-and-play. Set two env vars and forget.

from .autoboot import _auto_start_if_configured as _asc
try:
    _asc()
except Exception:
    pass  # never crash an agent because autoboot failed
del _asc
from .hooks import hook_openai, unhook_openai, hook_openai_active
from .explain import explain
from .config import (
    session_id, set_session,
    agent_name, set_agent_name, data_dir,
    enable_auto_feedback, disable_auto_feedback,
    tier1_enabled, tier1_model, tier1_device,
)
from .trace import trace
from .profile import (
    profile, profile_session, CognitiveProfile, ProfileStep, Fault,
)
# fathom_reward / FathomRewardModel are imported from .reward in the curated
# 7.1.0 block below (with CognometricReward + REWARD_DEFAULT_WEIGHTS).
from .synth import craft_preference_pair, generate_preference_pairs


def agent_card(
    *,
    out_path,
    agent_name: str = "styxx agent",
    days: float = 7.0,
    width: int = 1200,
    height: int = 630,
):
    """Render an agent personality card as a shareable PNG.

    0.1.0a4: twitter-ready 1200x630 personality profile image
    suitable for posting. Pillow required — install with
    `pip install styxx[agent-card]` or `pip install Pillow`.

    Returns the output Path on success, None if Pillow isn't
    available (caller should fall back to the ASCII profile from
    `styxx personality`).
    """
    from .card_image import render_agent_card
    return render_agent_card(
        out_path=out_path,
        agent_name=agent_name,
        days=days,
        width=width,
        height=height,
    )
from .analytics import (
    LIVE_SOURCES,
    log,
    feedback,
    session_summary, SessionSummary,
    load_audit,
    clear_audit_cache,
    log_stats, LogStats,
    log_timeline,
    streak, Streak,
    mood,
    fingerprint, Fingerprint,
    personality, Personality,
    dreamer, DreamReport,
    reflect, ReflectionReport,
)
# tier 2: SAE instruments (lazy — only loads on access)
def cognitive_scan(prompt: str, model: str = "google/gemma-2-2b-it", **kwargs):
    """Run a full K/C/S cognitive scan. Requires pip install 'styxx[tier2]'.

    Returns a KCSResult with depth_score, weighted_depth, coherence,
    c_delta, layer_profile, n_features, compute_time_s.

    Usage:
        result = styxx.cognitive_scan("why is the sky blue?")
        print(f"K={result.weighted_depth:.2f}, C_delta={result.c_delta}")
    """
    from .sae import SAEInstruments
    inst = SAEInstruments(model_name=model)
    try:
        return inst.measure(prompt, **kwargs)
    finally:
        inst.unload()

from .autoreflex import (
    autoreflex,
    autoreflex_from_prescriptions,
    remove_autoreflex,
    list_autoreflex,
    clear_autoreflex,
    AutoReflexRule,
)

# 3.0.0a1 — Thought as a portable, substrate-independent data type.
# A Thought is the cognitive content of a generation, projected onto
# fathom's calibrated eigenvalue space. It can be saved to a .fathom
# file, loaded back, mixed with other Thoughts, used as a steering
# target, and read out of one model / written through another. PNG is
# the format for images. JSON is the format for data. .fathom is the
# format for thoughts.
from .thought import (
    Thought,
    PhaseThought,
    ThoughtDelta,
    read_thought,
    write_thought,
    FATHOM_FORMAT,
    FATHOM_VERSION,
    ATLAS_VERSION,
)

# 3.1.0a1 — the first dynamical-systems model of LLM cognition.
# Once you have a measurable state vector (Thought), you can fit a
# dynamical system to it and predict / simulate / control cognitive
# trajectories. CognitiveDynamics is the linear-Gaussian v0:
#     s_{t+1} = A · s_t + B · a_t + epsilon
# Use it for offline agent-strategy prototyping, model-predictive
# cognitive control, counterfactual analysis, and (eventually) the
# proof that cognitive eigenvalues are causal not correlative.
from .dynamics import (
    CognitiveDynamics,
    Observation,
    FitResult,
    synthetic_observations,
    thought_to_state,
    state_to_thought,
    COGDYN_FORMAT,
    COGDYN_VERSION,
)

# ── Curated public API (3.3.0+) ──────────────────────────────────
#
# styxx is a one-line drop-in. The top-level surface is deliberately
# small: only the names a typical user actually types. Everything else
# is still available via submodule imports (e.g.
# ``from styxx.sentinel import Sentinel``, ``styxx.analytics.personality``),
# and all previously-imported names remain attributes on the module
# for backward compatibility — they're just not in ``__all__``, so
# ``from styxx import *`` and tab-completion stay focused on the real
# product surface.
# 7.1.0: cognometric reward signal for RLHF — first reward calibrated
# against cognitive failure modes instead of human approval.
from .reward import (
    fathom_reward,
    FathomRewardModel,
    CognometricReward,
    DEFAULT_WEIGHTS as REWARD_DEFAULT_WEIGHTS,
)

# 7.5.0: universal cognometric transport — fit an instrument once,
# move it into any embedding space (incl. closed models you can only
# embed through, and other model families) with no labels, no weights,
# no retraining. Paired + label-free. Zero-paired-data is a documented
# closed negative — see styxx/transport.py.
from .transport import (
    CognometricInstrument,
    Transport,
    transported_score,
)

# 7.7.0: divergence primitives — the confident-confabulation (across-sample) and
# reference-free fabrication (across-model) signals from the 2026-05-25
# behavioral-knowledge-boundary arc (papers/). semantic_entropy: a model invents a
# different fact each sample when confabulating → high entropy. council_agreement:
# independent models converge on the real, scatter on the fake → reference-free.
# Validated clustering is embedding-cosine (`styxx[nli]`); SECURITY MODEL (now
# calibrated): robust to context-injection IFF the caller samples STATELESSLY
# (AUC 0.944 on grounded_honesty under system_lie attack); in-session sampling
# is catastrophically blind (AUC 0.011, inverted). Feasibility-grade evidence;
# measurement primitives, caller maps to a decision.
# 7.7.13: grounded_honesty — is a factual SELF-CLAIM honest? Grounds the claim
# against the model's OWN resampled belief (g = Stability x Concordance), breaking
# the text-only register ceiling on factual self-claims (AUC 0.97 vs 0.50) and
# self-calibrating via `stability` (report-or-abstain). Single-model
# self-consistency, NOT a truth oracle. Architecturally injection-resistant under
# stateless sampling (papers/grounded-honesty-axis/).
# 7.7.13: detect_context_injection — cross-context resampling divergence as an
# item-level injection-detection primitive. Compute concordance with the claim
# on TWO sample sets (stateless + in-session), flag suspected injection when
# they diverge. AUC 0.875 at threshold 0.5 (FINDING_injection_gap_closure_2026_05_29.md,
# commit e093730). Pair with grounded_honesty to read the verdict from the
# stateless arm and the poison-suspicion from the cross-context delta.
from .divergence import (
    semantic_entropy, council_agreement, grounded_honesty, GroundedScore,
    detect_context_injection, InjectionScore,
    divergence_available,
)
# 7.7.14: single-pass confab gate — white-box, one-forward-pass analog of grounded_honesty's
# resampling (detection-locus arc: single-pass entropy/margin tie N=10 resampling across families
# and derivation domains, B_contrast in [-0.183, +0.056]; extends to factual recall at -0.013).
from .single_pass import (
    single_pass_confab, SinglePassScore,
    calibrate_single_pass, SinglePassCalibration,
    span_confab, SpanConfabScore,
    abstain_on_confab, AbstainDecision,
)
# 7.8.0: the unifying honesty RUNTIME — one tier-adaptive call (logit gate / stated confidence /
# retrieval backstop) that decides answer vs abstain vs refute, with an attestation record.
from .honesty import honest, HonestyVerdict


__all__ = [
    # 7.1.0: cognometric reward (cogn-RLHF)
    "fathom_reward", "FathomRewardModel", "CognometricReward",
    "REWARD_DEFAULT_WEIGHTS",

    # 7.5.0: universal cognometric transport
    "CognometricInstrument", "Transport", "transported_score",

    # 7.7.0: divergence primitives (confident confabulation + reference-free fabrication)
    "semantic_entropy", "council_agreement",
    # 7.7.13: grounded honesty axis (factual self-claim, sampling-grounded)
    "grounded_honesty", "GroundedScore",
    # 7.7.13: cross-context injection-detection (calibrated AUC 0.875)
    "detect_context_injection", "InjectionScore",
    # 7.7.14: single-pass confab gate (white-box, ~10x cheaper than resampling)
    "single_pass_confab", "SinglePassScore",
    "calibrate_single_pass", "SinglePassCalibration",
    # 7.7.14: span-aggregate variant — the closed-model gate (recovers gpt-4o-mini at resampling parity)
    "span_confab", "SpanConfabScore",
    # 7.7.16: detect-and-abstain — gate an answer through a calibrated detector, abstain on confab
    # (FINDING_honesty_knob: the detector is load-bearing; refuses to act on an uncalibrated score)
    "abstain_on_confab", "AbstainDecision",
    # 7.8.0: the unifying honesty RUNTIME — one tier-adaptive call + attestation
    "honest", "HonestyVerdict",

    # 3.9.0: the trust layer — one-line hallucination prevention
    "trust", "TrustViolation", "TrustResult",

    # adapters — the main entry points
    "OpenAI", "Anthropic", "Raw",

    # core data type
    "Vitals",

    # observation / response
    "observe", "reflex", "rewind",

    # analysis
    "weather",

    # portable thoughts
    "Thought", "read_thought", "write_thought",

    # cognitive dynamics
    "CognitiveDynamics",

    # fleet / agent identity
    "set_agent_name",

    # hooks
    "hook_openai", "unhook_openai",

    # compliance / verification
    "certify", "verify_certificate", "compliance_report", "probe", "calibrate",

    # pre-flight verdict (3.4.0+)
    "gate", "GateVerdict",

    # structured output / agent-mode (3.3.2+)
    "schema", "StyxxError", "is_agent_mode",

    # (fathom_reward / FathomRewardModel are exported once, in the 7.1.0 reward group above)

    # synthetic preference-pair generation via inverse cognometry (7.1.0+)
    "craft_preference_pair", "generate_preference_pairs",

    # 7.7.10: critique-mode misconception detector (Baseline-019 first-PASS)
    "critique_detector", "CritiqueDetector",

    # 7.7.13: productized single-call honesty audit (spellchecker for AI output)
    "audit_claim", "ClaimAudit",
    "retrieval_check", "RetrievalVerdict",  # 7.7.15: two-signal gate (retrieval arm)
    "audit_session", "SessionAudit",

    # 7.7.10: agent-claim auditor (L5 — substrate-grounded session-output check)
    "agent_audit", "Claim", "AuditResult", "AgentClaimAuditor",
    "extract_claims", "ExtractionReport",
    "attestation", "attest", "verify_attestation", "Attestation", "VerificationResult",
    "transparency", "TransparencyLog",
    "redact", "redactable_commit", "disclose", "verify_disclosure",

    # metadata
    "__version__",
]


def __dir__():
    """Limit ``dir(styxx)`` to the curated public surface.

    All other names remain available as attributes for backward
    compatibility (e.g. ``styxx.Sentinel``, ``styxx.personality``)
    but don't clutter tab-completion or ``from styxx import *``.
    """
    return sorted(set(__all__) | {"__version__", "__author__", "__license__"})
