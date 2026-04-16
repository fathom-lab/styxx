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

__version__ = "3.2.0"
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
    """Honest pass-through wrapper around anthropic.Anthropic.

    Usage:
        # Before: from anthropic import Anthropic
        from styxx import Anthropic
        client = Anthropic()
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "..."}],
        )
        print(r.content[0].text)   # normal anthropic response
        print(r.vitals)            # always None (see below)

    IMPORTANT — .vitals is None on every Anthropic call.
    Anthropic's Messages API does not expose per-token logprobs
    (no logprobs=True / top_logprobs=k parameter), so tier 0 styxx
    vitals cannot be computed from the response. This wrapper
    exists so styxx.Anthropic is a valid import path, and so a
    one-time warning explains the situation at first use.

    Workarounds if you need vitals on Claude inference:
      - route through an OpenAI-compatible gateway (OpenRouter)
        and use styxx.OpenAI(base_url=...)
      - use styxx.Raw with a pre-captured logprob trajectory
      - wait for styxx v0.2 tier 1 (d-axis honesty from residual
        stream — does not need logprobs)
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
from .vitals import Vitals, CentroidClassifier
from .watch import watch, observe, observe_raw, is_concerning, WatchSession
from .gates import on_gate, remove_gate, clear_gates, list_gates
from .reflex import reflex, rewind, abort, ReflexSession, ReflexSignal, RewindSignal, AbortSignal
from .guardian import guardian, GuardianSession, SteeringEvent
from .weather import weather, WeatherReport
from .dashboard import dashboard
from .calibrate import calibrate, calibration_status, CalibrationResult
from .fleet import (
    list_agents, compare_agents, fleet_summary, best_agent_for,
    AgentProfile, FleetSummary,
)
from .memory import (
    remember, recall, memories, memory_stats,
    Memory, RecallResult,
)
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
from .notify import on_anomaly, notify_on_fail, clear_notifications, CognitiveEvent
from .optimize import optimize
from .ci import regression_test, create_baseline, Baseline, RegressionResult
from .provenance import certify, verify, CognitiveCertificate, VerificationResult
from .diff import compare_sessions, compare_windows, ComparisonDiff
from .learned_classifier import train_text_classifier, TrainResult
from .autoboot import autoboot
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

__all__ = [
    # ── 1. observe ────────────────────────────────────────
    "observe", "observe_raw", "watch", "is_concerning", "explain",
    "log", "feedback",
    "WatchSession", "Vitals", "StyxxRuntime", "CentroidClassifier",
    # adapters
    "OpenAI", "Anthropic", "Raw",
    "LangChain", "CrewAI", "AutoGen", "LangSmith", "Langfuse",
    "hook_openai", "unhook_openai", "hook_openai_active",
    "trace",

    # ── 2. respond ────────────────────────────────────────
    "reflex", "rewind", "abort",
    "on_gate", "remove_gate", "clear_gates", "list_gates",
    "autoreflex", "autoreflex_from_prescriptions",
    "remove_autoreflex", "list_autoreflex", "clear_autoreflex",
    "guardian",
    "ReflexSession", "ReflexSignal", "RewindSignal", "AbortSignal",
    "AutoReflexRule", "GuardianSession", "SteeringEvent",

    # ── 3. scan (tier 2) ─────────────────────────────────
    "cognitive_scan",

    # ── 4. analyze ────────────────────────────────────────
    "weather", "personality", "reflect", "mood", "streak",
    "antipatterns", "conversation", "dreamer",
    "session_summary", "timeline", "fingerprint",
    "agent_card", "log_stats", "log_timeline",
    "load_audit", "clear_audit_cache",
    "WeatherReport", "Personality", "ReflectionReport",
    "SessionSummary", "Timeline", "Fingerprint",
    "Streak", "DreamReport", "LogStats",
    "ConversationResult", "AntiPattern",

    # ── 5. learn ──────────────────────────────────────────
    "calibrate", "calibration_status", "train_text_classifier",
    "enable_auto_feedback", "disable_auto_feedback",
    "optimize",
    "CalibrationResult", "TrainResult",

    # ── 6. fleet ──────────────────────────────────────────
    "set_agent_name", "agent_name", "list_agents",
    "compare_agents", "fleet_summary", "best_agent_for",
    "AgentProfile", "AgentComparison", "FleetSummary",

    # ── 7. memory ─────────────────────────────────────────
    "remember", "recall", "memories", "memory_stats",
    "Memory", "RecallResult",

    # ── 8. handoff ────────────────────────────────────────
    "handoff", "receive",
    "HandoffEnvelope",

    # ── 9. compliance ─────────────────────────────────────
    "certify", "verify", "compliance_report",
    "probe", "on_anomaly", "notify_on_fail", "clear_notifications",
    "regression_test", "create_baseline",
    "assert_healthy", "check_health", "cognitive_sla",
    "compare_sessions", "compare_windows",
    "CognitiveCertificate", "VerificationResult",
    "ComplianceReport", "ProbeReport",
    "Baseline", "RegressionResult",
    "CognitiveSLAViolation", "SLAReport",
    "CognitiveEvent", "ComparisonDiff",

    # ── 10. operations ────────────────────────────────────
    "autoboot", "dashboard", "sentinel", "get_sentinel",
    "session_id", "set_session", "data_dir",
    "set_context", "current_context",
    "expect", "unexpect", "expected_categories", "clear_expected",
    "set_mood", "current_mood_override", "gate_multiplier",
    "tier1_enabled", "tier1_model", "tier1_device",
    "Sentinel", "SentinelAlert",
    "LIVE_SOURCES",

    # ── 11. portable thoughts (3.0.0a1) ───────────────────
    "Thought", "PhaseThought", "ThoughtDelta",
    "read_thought", "write_thought",
    "FATHOM_FORMAT", "FATHOM_VERSION", "ATLAS_VERSION",

    # ── 12. cognitive dynamics (3.1.0a1) ──────────────────
    "CognitiveDynamics", "Observation", "FitResult",
    "synthetic_observations",
    "thought_to_state", "state_to_thought",
    "COGDYN_FORMAT", "COGDYN_VERSION",

    # ── metadata ──────────────────────────────────────────
    "__version__", "__tagline__",
]
