"""
styxx.agent_card
================

Builds (and optionally serves) the A2A agent-card for styxx.

The canonical card lives at ``.well-known/agent-card.json`` at the repo root.
This module builds an equivalent object in-memory and exposes a CLI helper::

    python -m styxx.agent_card            # prints JSON
    python -m styxx.agent_card --out FILE # writes to file

A2A spec: https://a2a-protocol.org/latest/specification/
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict

from . import __version__, __url__, __tagline__


CLASSES = ["reasoning", "retrieval", "refusal", "creative", "adversarial", "hallucination"]


def _vitals_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["classification", "confidence", "gate"],
        "properties": {
            "classification": {"type": "string", "enum": CLASSES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "gate": {"type": "string", "enum": ["pass", "warn", "fail"]},
        },
    }


def _verification_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["valid", "confidence"],
        "properties": {
            "valid": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "gate": {"type": "string", "enum": ["pass", "warn", "fail"]},
            "classification": {"type": "string"},
            "anomalies": {"type": "array", "items": {"type": "string"}},
            "trajectory": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "slope": {"type": "number"},
                    "curvature": {"type": "number"},
                    "volatility": {"type": "number"},
                },
            },
        },
    }


def build_agent_card() -> Dict[str, Any]:
    """Return the A2A agent card as a plain dict."""
    return {
        "$schema": "https://a2a-protocol.org/schemas/v0.3/agent-card.json",
        "protocolVersion": "0.3",
        "id": "styxx",
        "name": "styxx",
        "description": (
            "Cognitive vitals for language models. Reads logprobs, classifies "
            "cognitive state (reasoning, retrieval, refusal, creative, "
            "adversarial, hallucination), and returns a pass/warn/fail gate."
        ),
        "version": __version__,
        "tagline": __tagline__,
        "documentationUrl": __url__,
        "url": __url__,
        "provider": {
            "id": "fathom-lab",
            "name": "Fathom Lab",
            "url": "https://fathom.darkflobi.com",
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "securitySchemes": {},
        "security": [],
        "skills": [
            {
                "id": "observe",
                "name": "Observe cognitive vitals",
                "description": (
                    "Reads logprobs from a model response and returns Vitals: "
                    "classification, confidence (0-1), and a pass|warn|fail gate."
                ),
                "tags": ["vitals", "classification", "logprobs", "interpretability"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
                "input": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["response"],
                    "properties": {"response": {"type": "object"}},
                },
                "output": _vitals_schema(),
            },
            {
                "id": "verify",
                "name": "Verify response trustworthiness",
                "description": (
                    "Runs trajectory analysis over a response's logprob stream "
                    "and returns a VerificationResult."
                ),
                "tags": ["verification", "trajectory", "confidence"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
                "input": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["response"],
                    "properties": {"response": {"type": "object"}},
                },
                "output": _verification_schema(),
            },
            {
                "id": "classify",
                "name": "Classify trajectory shape",
                "description": (
                    "Given a sequence of token logprobs, returns the cognitive "
                    "class."
                ),
                "tags": ["classification", "trajectory"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
                "input": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["logprobs"],
                    "properties": {
                        "logprobs": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 1,
                        }
                    },
                },
                "output": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["classification", "confidence"],
                    "properties": {
                        "classification": {"type": "string", "enum": CLASSES},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            {
                "id": "weather",
                "name": "Cognitive weather report",
                "description": "Fleet-level weather summary over recent vitals.",
                "tags": ["monitoring", "fleet"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
                "input": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "window": {"type": "integer", "minimum": 1, "default": 100}
                    },
                },
                "output": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "summary": {"type": "string"},
                        "gate": {
                            "type": "string",
                            "enum": ["pass", "warn", "fail"],
                        },
                    },
                },
            },
        ],
        "components": {
            "schemas": {
                "Vitals": _vitals_schema(),
                "VerificationResult": _verification_schema(),
            }
        },
    }


def write_agent_card(path: str | Path) -> Path:
    """Write the agent card to ``path`` (pretty-printed JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_agent_card(), indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print or write the styxx A2A agent-card.")
    parser.add_argument("--out", type=str, default=None, help="Write to file instead of stdout.")
    args = parser.parse_args(argv)
    if args.out:
        p = write_agent_card(args.out)
        print(f"wrote {p}")
    else:
        print(json.dumps(build_agent_card(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
