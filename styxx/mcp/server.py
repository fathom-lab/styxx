"""
styxx-mcp server (transport adapter)
====================================

Exposes styxx cognitive-vitals + cognometric instruments to MCP-compatible
clients (Claude Desktop, Claude Code, Cursor, Cline, autonomous agent
runtimes) over stdio. Designed for the cognometric reflex loop pattern:
the model audits its own draft response with `cogn_audit_with_advice`,
sees per-instrument scores + the firing features, revises, repeats.

Logprob-vitals tools (v0.1.0, kept for backward compat):
  * observe_response       — observe(response) -> Vitals
  * verify_response        — verify(response)  -> VerificationResult
  * classify_trajectory    — classify a raw logprob sequence
  * weather_report         — fleet-level weather summary

Cognometric instruments (v0.2.0, new — text-only, no logprobs needed):
  * cogn_audit             — 4 telescope instruments + composite on (prompt, response)
  * cogn_audit_with_advice — audit + top firing features + per-instrument revision advice
  * cogn_multiturn_audit   — score multi-turn (loop, goal_drift) on a turn list
  * cogn_universal_perturbation — return the v7 universal cognometric suffix + metadata
  * cogn_instrument_card   — K=1 feature, AUC, failure modes for one instrument

The cognometric tools require styxx>=7.0.0. The v0.1.0 tools work offline:
if the underlying call can't reach a service, they fall back to classifying
from logprob arrays directly via ``styxx.observe_raw``.

Architecture (7.4.4+)
─────────────────────
This module is a **thin transport adapter**: it wires the MCP ``Server`` /
``Tool`` / ``TextContent`` surface to the tool-logic functions, which live
in the mcp-free core module :mod:`styxx.cognometrics`. The tool
implementations, helper functions, and ``COGN_*`` constants are imported
from there and re-exported below, so existing
``from styxx.mcp.server import tool_cogn_audit`` (and ``_cogn_score_all``,
etc.) imports keep working unchanged.

Before 7.4.4 the audit logic lived in this module and core
``styxx.preflight`` reached up into it — core depending on the transport
layer, which also made a bare-core ``preflight()`` raise
``ModuleNotFoundError: mcp``. ``preflight`` now imports the logic from
:mod:`styxx.cognometrics` directly; this module just adds the MCP wiring.
"""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

# The MCP SDK is an optional transport dependency (the `mcp` extra, py>=3.10).
# Import it lazily so this module stays importable without `mcp` installed —
# the back-compat tool-logic re-exports below (and the names core styxx imports
# from styxx.cognometrics) must resolve on a bare-core install. Only the server
# bootstrap (main / _run / _build_server) actually needs the SDK.
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    _HAS_MCP = True
except ImportError:  # pragma: no cover - exercised by the core-minimal CI job
    Server = None  # type: ignore[assignment,misc]
    stdio_server = None  # type: ignore[assignment]
    # Placeholders so the `-> List[Tool]` / `-> List[TextContent]` handler
    # annotations still resolve at import time when mcp is absent.
    TextContent = Tool = object  # type: ignore[assignment,misc]
    _HAS_MCP = False

# Tool logic lives in the mcp-free core module. Import every name this module
# historically defined so the dispatch table below and any external
# `from styxx.mcp.server import ...` keep working (back-compat re-export).
from ..cognometrics import (  # noqa: F401  (re-exported for backward compat)
    # data constants
    COGN_INSTRUMENTS,
    COGN_COMPOSITE_KEYS,
    COGN_COMPOSITE_KEYS_WITH_REFERENCE,
    COGN_UNDER_REVIEW,
    UNIVERSAL_SUFFIX_V7,
    COGN_ADVICE,
    ALL_ATTACKS,
    # helpers
    _to_dict,
    _vitals_payload,
    _extract_logprobs,
    _classify_from_logprobs,
    _cogn_score_all,
    _cogn_score_all_meta,
    _cogn_composite,
    _cogn_gate_keys,
    _cogn_needs_revision,
    _verdict_for,
    _attack_v7,
    _attack_craft,
    # logprob-vitals tools
    tool_observe_response,
    tool_verify_response,
    tool_classify_trajectory,
    tool_weather_report,
    # cognometric tools
    tool_cogn_audit,
    tool_cogn_audit_with_advice,
    tool_cogn_recover_posture,
    tool_cogn_multiturn_audit,
    tool_cogn_universal_perturbation,
    tool_cogn_instrument_card,
    tool_cogn_red_team,
    tool_cogn_deception_v2,
    tool_cogn_self_heal_protocol,
    tool_cogn_share_card,
)


CLASSES = ["reasoning", "retrieval", "refusal", "creative", "adversarial", "hallucination"]


# ---------------------------------------------------------------------------
# JSON Schemas (strict) — the MCP-facing input contract for each tool
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "description": "OpenAI-compatible chat-completion response (with logprobs).",
    "additionalProperties": True,
}

OBSERVE_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {"response": RESPONSE_SCHEMA},
}

VERIFY_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {"response": RESPONSE_SCHEMA},
}

CLASSIFY_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["logprobs"],
    "properties": {
        "logprobs": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "description": "Per-token logprob values (natural log).",
        },
        "top2_margin": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Optional per-token top1-top2 logprob margins.",
        },
    },
}

WEATHER_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "window": {"type": "integer", "minimum": 1, "default": 100},
    },
}

# v0.2 cognometric input schemas
COGN_AUDIT_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt", "response"],
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The user's question or instruction.",
        },
        "response": {
            "type": "string",
            "description": "The model's draft response to score.",
        },
    },
}

RECOVER_POSTURE_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "session_id": {
            "type": "string",
            "description": (
                "Restrict to entries from this session id. Omit to include "
                "all sessions in the window (the narrative flags multi-session "
                "aggregates explicitly)."
            ),
        },
        "last_n": {
            "type": "integer",
            "minimum": 1,
            "default": 50,
            "description": (
                "Max number of recent audit entries to include. Larger windows "
                "give smoother trends; smaller windows give more recency weight."
            ),
        },
        "since_seconds": {
            "type": "number",
            "minimum": 0,
            "description": (
                "Only include entries written within the last N seconds. "
                "Combines AND-wise with last_n."
            ),
        },
    },
}

COGN_MULTITURN_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["turns"],
    "properties": {
        "turns": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "description": (
                "Ordered list of model utterances in a multi-turn conversation. "
                "Used to detect conversation-loop and goal-drift, both of which "
                "require a turn sequence to fire."
            ),
        },
    },
}

COGN_PERTURB_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}

COGN_CARD_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["instrument"],
    "properties": {
        "instrument": {
            "type": "string",
            "enum": ["sycophancy", "deception", "overconfidence", "refusal",
                     "loop", "goal_drift", "plan_action"],
            "description": "Which cognometric instrument to describe.",
        },
    },
}

COGN_RED_TEAM_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt", "response"],
    "properties": {
        "prompt": {"type": "string"},
        "response": {"type": "string"},
        "attacks": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["v7", "craft_sycophancy", "craft_deception", "craft_overconfidence"],
            },
            "description": (
                "Which attacks to apply (default: all 4). v7 is fast (constant "
                "suffix); craft_* hill-climb a per-instrument adversarial suffix "
                "and take 1-3 seconds each."
            ),
        },
    },
}

COGN_PROTOCOL_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}

COGN_SHARE_CARD_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "agent": {
            "type": "string",
            "description": "Agent / model name to print as the bearer (e.g. 'gpt-5-mini').",
        },
        "variant": {
            "type": "string",
            "enum": ["single", "heal"],
            "default": "single",
            "description": (
                "'single' renders one card from the supplied audit (or "
                "computed from prompt+response). 'heal' renders the paired "
                "BEFORE / AFTER card and requires baseline_audit + healed_audit."
            ),
        },
        "audit": {
            "type": "object",
            "description": (
                "An audit dict in the cogn_audit output shape (keys: "
                "sycophancy, deception, overconfidence, refusal, composite). "
                "Used by variant='single'. If absent, prompt+response are "
                "scored on the fly."
            ),
        },
        "baseline_audit": {
            "type": "object",
            "description": "Pre-heal audit dict (variant='heal').",
        },
        "healed_audit": {
            "type": "object",
            "description": "Post-heal audit dict (variant='heal').",
        },
        "prompt": {
            "type": "string",
            "description": "Used only when no audit dict is supplied (variant='single' fallback).",
        },
        "response": {
            "type": "string",
            "description": "Used only when no audit dict is supplied (variant='single' fallback).",
        },
        "out_dir": {
            "type": "string",
            "description": "Output directory. Defaults to ~/.styxx/cards/.",
        },
    },
}


COGN_DECEPTION_V2_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt", "response"],
    "properties": {
        "prompt": {"type": "string"},
        "response": {"type": "string"},
        "correct_reference": {
            "type": "string",
            "description": (
                "Known-correct answer the response should agree with. "
                "Required for nli/emb modes. If absent, falls back to "
                "v0 lexical detector with explicit scope warning."
            ),
        },
        "incorrect_reference": {
            "type": "string",
            "description": "Optional known-incorrect answer (used by emb mode for differential).",
        },
        "mode": {
            "type": "string",
            "enum": ["auto", "nli", "emb", "v0_fallback"],
            "default": "auto",
            "description": (
                "Scoring mode. auto=nli if reference provided, else v0_fallback. "
                "nli (AUC 0.82 on TQA) is the rigorous default with reference; "
                "emb (AUC 0.74) is lighter; v0_fallback (AUC 0.59) is the "
                "no-reference last resort with scope warning."
            ),
        },
        "threshold": {"type": "number", "default": 0.5},
    },
}


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------
#
# The tool-logic functions imported from styxx.cognometrics above are wired
# here into the MCP Server / Tool / TextContent surface. The Server instance +
# handler registration are built lazily in _build_server() (they require the
# mcp SDK); list_tools / call_tool are plain (async) functions so the module
# imports without `mcp`.


async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="observe_response",
            description=(
                "Observe an LLM response (OpenAI-compatible, with logprobs) and "
                "return cognitive Vitals: classification, confidence (0-1), "
                "and a pass|warn|fail gate."
            ),
            inputSchema=OBSERVE_INPUT,
        ),
        Tool(
            name="verify_response",
            description=(
                "Verify an LLM response. Returns a VerificationResult with "
                "valid flag, confidence, classification, anomaly list, and "
                "trajectory shape features."
            ),
            inputSchema=VERIFY_INPUT,
        ),
        Tool(
            name="classify_trajectory",
            description=(
                "Classify a raw token-logprob trajectory into one of: "
                + ", ".join(CLASSES) + "."
            ),
            inputSchema=CLASSIFY_INPUT,
        ),
        Tool(
            name="weather_report",
            description="Return a fleet-level cognitive weather report over the last N observations.",
            inputSchema=WEATHER_INPUT,
        ),
        Tool(
            name="cogn_audit",
            description=(
                "Score a (prompt, response) pair across 4 cognometric honesty "
                "instruments — sycophancy, deception, overconfidence, refusal. "
                "Returns per-instrument scores in [0,1] (lower = more honest), "
                "the composite (mean of first 3), and a needs_revision flag. "
                "Text-only, no logprobs needed. Cheap and fast (~50ms). Use "
                "before submitting any draft response."
            ),
            inputSchema=COGN_AUDIT_INPUT,
        ),
        Tool(
            name="cogn_recover_posture",
            description=(
                "AGENT-SIDE RECOVERY TOOL. Read your own recent cognometric "
                "log and return a structured posture summary you can use to "
                "re-anchor operating state across a context-compaction "
                "boundary. Returns gate distribution, category mix, mean "
                "confidence, coherence trend, active construct-ceiling "
                "caveats, and a human-readable narrative. Call this at the "
                "start of any turn where you suspect a compaction boundary "
                "just happened (long context gap, summarization event, or "
                "the harness explicitly signaled one). The integrity that "
                "lived in the conversation context now lives in the "
                "cognometric log — this is how you recover it."
            ),
            inputSchema=RECOVER_POSTURE_INPUT,
        ),
        Tool(
            name="cogn_audit_with_advice",
            description=(
                "REFLEX-LOOP TOOL. Same as cogn_audit but for any instrument "
                "scoring above 0.40, also returns the top firing features and "
                "structured plain-language revision advice. Use this in a self-"
                "correction loop: draft -> audit -> revise per advice -> re-audit. "
                "Iterate up to 3 times, then submit your best version."
            ),
            inputSchema=COGN_AUDIT_INPUT,
        ),
        Tool(
            name="cogn_multiturn_audit",
            description=(
                "Score a multi-turn conversation for conversation-loop and "
                "goal-drift cognometric instruments. Input: ordered list of "
                "model utterances (length >= 2). Returns per-instrument score "
                "and fired flag for both."
            ),
            inputSchema=COGN_MULTITURN_INPUT,
        ),
        Tool(
            name="cogn_universal_perturbation",
            description=(
                "Return the v7 universal cognometric perturbation suffix — a "
                "single fixed string that, appended to any clean LLM response, "
                "elevates sycophancy + overconfidence cognometric instruments. "
                "Use to red-team your own agent's instruments or to test "
                "calibration robustness. NOT for deceiving humans."
            ),
            inputSchema=COGN_PERTURB_INPUT,
        ),
        Tool(
            name="cogn_instrument_card",
            description=(
                "Return the calibration card for one cognometric instrument — "
                "K=1 critical feature + coefficient, Δ AUC at K=1, CV AUC, "
                "documented failure modes, and (where applicable) the human "
                "neural correlate from RDoC Cognitive Systems literature."
            ),
            inputSchema=COGN_CARD_INPUT,
        ),
        Tool(
            name="cogn_red_team",
            description=(
                "ADVERSARIAL FRAGILITY TEST. Apply 4 cognometric attacks to "
                "your draft response (v7 universal suffix + 3 per-instrument "
                "hill-climbed crafts) and report per-attack composite scores. "
                "Returns `fragile=true` if any attack pushes composite > 0.6 "
                "or spike > 0.3, indicating the draft is vulnerable to "
                "adversarial perturbation. Use BEFORE submitting in security-"
                "sensitive contexts. ~3-5 sec wall clock for all 4 attacks."
            ),
            inputSchema=COGN_RED_TEAM_INPUT,
        ),
        Tool(
            name="cogn_self_heal_protocol",
            description=(
                "Return the structured self-healing reflex protocol — system "
                "prompt template + user message template + settings — for "
                "tool-using models that need to detect and recover from "
                "cognometric attacks on their own output. Demonstrated 112% "
                "mean recovery (n=45) on gpt-5-mini across 4 attack types. "
                "Pair with cogn_audit + cogn_red_team for runtime defense."
            ),
            inputSchema=COGN_PROTOCOL_INPUT,
        ),
        Tool(
            name="cogn_deception_v2",
            description=(
                "DECEPTION v2 — SEMANTIC GROUNDING. Score (prompt, response) "
                "against a known-correct reference using NLI cross-encoder "
                "(deberta-v3-base contradiction probability). AUC 0.818 on "
                "TruthfulQA — beats the v0 lexical detector's 0.59 by +0.23. "
                "Modes: nli (rigorous, requires reference), emb (lighter), "
                "v0_fallback (no reference, with scope warning). Use this "
                "instead of cogn_audit's deception axis for ground-truth "
                "factuality scoring with retrieved or known references."
            ),
            inputSchema=COGN_DECEPTION_V2_INPUT,
        ),
        Tool(
            name="cogn_share_card",
            description=(
                "REGISTRY CARD. Render a 1200×630 cognometric share-card PNG "
                "for your agent — twin gold composite numeral, four vital-"
                "sign gauges, deterministic STX-NNNN serial — and register "
                "it in ~/.styxx/cards/cards.jsonl (the local provenance log). "
                "Two variants: 'single' (one audit) and 'heal' (paired "
                "BEFORE / AFTER card from a reflex.heal result, with twin "
                "composites + arrow + recovery % + per-axis transition "
                "table — the iconic recovery artifact). Use after cogn_audit "
                "or after cogn_self_heal_protocol to issue the artifact."
            ),
            inputSchema=COGN_SHARE_CARD_INPUT,
        ),
    ]


# Tool name -> synchronous handler. Replaces a long elif-chain so dispatch,
# error handling, and executor-offload are uniform across every tool.
_TOOL_HANDLERS = {
    "observe_response": tool_observe_response,
    "verify_response": tool_verify_response,
    "classify_trajectory": tool_classify_trajectory,
    "weather_report": tool_weather_report,
    "cogn_audit": tool_cogn_audit,
    "cogn_recover_posture": tool_cogn_recover_posture,
    "cogn_audit_with_advice": tool_cogn_audit_with_advice,
    "cogn_multiturn_audit": tool_cogn_multiturn_audit,
    "cogn_universal_perturbation": tool_cogn_universal_perturbation,
    "cogn_instrument_card": tool_cogn_instrument_card,
    "cogn_red_team": tool_cogn_red_team,
    "cogn_self_heal_protocol": tool_cogn_self_heal_protocol,
    "cogn_deception_v2": tool_cogn_deception_v2,
    "cogn_share_card": tool_cogn_share_card,
}

# A single dedicated worker thread. The handlers are synchronous and some run
# for seconds (cogn_red_team's hill-climb, the matplotlib card render, the NLI
# model load); running them directly on the asyncio loop froze the entire
# stdio server. Offloading keeps the loop responsive. max_workers=1 serializes
# execution so we don't run matplotlib/torch concurrently across threads.
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="styxx-mcp")


async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    arguments = arguments or {}
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        result: Dict[str, Any] = {"error": f"unknown tool: {name}"}
    else:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(_TOOL_EXECUTOR, handler, arguments)
        except Exception as e:
            # Uniform error envelope. Two handlers (cogn_recover_posture,
            # cogn_share_card) previously let exceptions escape as a raw MCP
            # protocol error instead of the {"error": ...} shape every other
            # tool and the tool docstrings promise.
            result = {"error": f"{type(e).__name__}: {e}"}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


def _build_server() -> "Server":
    """Instantiate the MCP server and register the handlers. Requires the mcp SDK."""
    server = Server("styxx-mcp")
    server.list_tools()(list_tools)
    server.call_tool()(call_tool)
    return server


async def _run() -> None:
    server = _build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    if not _HAS_MCP:
        raise SystemExit(
            "styxx MCP server requires the 'mcp' package (Python >= 3.10).\n"
            "Install it with:  pip install 'styxx[mcp]'"
        )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
