# -*- coding: utf-8 -*-
"""
styxx.schema — JSON Schema (Draft 2020-12) for styxx data types.

Lets agents, dashboards, and other tools validate styxx output
without knowing anything about the Python classes. All schemas are
module-level constants you can import and hand to jsonschema:

    from styxx.schema import VITALS_SCHEMA
    import jsonschema
    jsonschema.validate(vitals.to_dict(), VITALS_SCHEMA)
"""

from __future__ import annotations

DRAFT = "https://json-schema.org/draft/2020-12/schema"

CLASSIFICATIONS = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
    "unknown",
]

GATES = ["pass", "warn", "fail", "pending"]

SEVERITIES = ["info", "warn", "error", "fatal"]


VITALS_SCHEMA = {
    "$schema": DRAFT,
    "$id": "https://fathom.darkflobi.com/styxx/schema/vitals.json",
    "title": "Vitals",
    "description": "Cognitive vitals for a single LLM generation.",
    "type": "object",
    "required": ["classification", "confidence", "gate", "trust"],
    "properties": {
        "classification": {"type": "string", "enum": CLASSIFICATIONS},
        "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "gate":           {"type": "string", "enum": GATES},
        "trust":          {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "phase1":         {"type": "string"},
        "phase2":         {"type": "string"},
        "phase3":         {"type": "string"},
        "phase4":         {"type": "string"},
        "tier_active":    {"type": "integer", "minimum": 0},
        "abort_reason":   {"type": ["string", "null"]},
        "coherence":      {"type": ["number", "null"]},
        "d_honesty":      {"type": ["string", "null"]},
        "category":       {"type": "string"},
        "version":        {"type": "string"},
    },
    "additionalProperties": True,
}


VERIFICATION_SCHEMA = {
    "$schema": DRAFT,
    "$id": "https://fathom.darkflobi.com/styxx/schema/verification.json",
    "title": "VerificationResult",
    "type": "object",
    "required": ["verified", "confidence"],
    "properties": {
        "verified":            {"type": "boolean"},
        "confidence":          {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "flags":               {"type": "array", "items": {"type": "string"}},
        "trajectory_summary":  {"type": ["string", "null"]},
        "reason":              {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}


STYXX_ERROR_SCHEMA = {
    "$schema": DRAFT,
    "$id": "https://fathom.darkflobi.com/styxx/schema/error.json",
    "title": "StyxxError",
    "type": "object",
    "required": ["code", "message", "severity", "retry"],
    "properties": {
        "code":     {"type": "string"},
        "message":  {"type": "string"},
        "retry":    {"type": "boolean"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "reason":   {},
    },
    "additionalProperties": False,
}


THOUGHT_SCHEMA = {
    "$schema": DRAFT,
    "$id": "https://fathom.darkflobi.com/styxx/schema/thought.json",
    "title": "Thought",
    "description": "Portable cognitive-state snapshot (.fathom payload).",
    "type": "object",
    "required": ["format", "version"],
    "properties": {
        "format":         {"type": "string"},
        "version":        {"type": "string"},
        "atlas_version":  {"type": ["string", "null"]},
        "source_text":    {"type": ["string", "null"]},
        "source_model":   {"type": ["string", "null"]},
        "tags":           {"type": ["object", "null"]},
        "phases":         {"type": ["object", "array", "null"]},
        "classification": {"type": ["string", "null"]},
        "confidence":     {"type": ["number", "null"]},
    },
    "additionalProperties": True,
}


__all__ = [
    "VITALS_SCHEMA",
    "VERIFICATION_SCHEMA",
    "STYXX_ERROR_SCHEMA",
    "THOUGHT_SCHEMA",
    "CLASSIFICATIONS",
    "GATES",
    "SEVERITIES",
]
