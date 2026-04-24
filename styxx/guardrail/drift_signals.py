# -*- coding: utf-8 -*-
"""
Tool-call drift signals — text-only features for `drift_check()`.

Pure Python, no external dependencies beyond stdlib regex and math.
Pyodide-safe. Extracts 23 features per (prompt, functions, tool_call)
triplet.

See `calibrated_weights_drift_v1.py` for the research methodology
and held-out cross-validation AUC per strata.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "for", "to", "in",
    "on", "at", "by", "with", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "may", "might", "must",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "what", "which", "who", "how", "when", "where", "why",
}

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PLACEHOLDER_RE = re.compile(
    r"^(placeholder|example|test|_[a-z_]+|<[^>]+>)$", re.IGNORECASE
)


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS and len(t) > 1]


def _value_first_position(value: Any, prompt_tokens: List[str]) -> Any:
    """Index of earliest prompt token that is a member of value's tokens.
    Returns None if the value contributes no matchable tokens."""
    vtoks = set(_tokens(str(value)))
    if not vtoks:
        return None
    for i, tok in enumerate(prompt_tokens):
        if tok in vtoks:
            return i
    return None


def _arg_order_inversion_rate(
    prompt_tokens: List[str],
    schema_props: Dict[str, Any],
    call_args: Dict[str, Any],
) -> float:
    """Fraction of argument-pairs where the schema's declared order of
    arg keys disagrees with the prompt-order of their call values.

    Targets arg_swap drift: when the model produces the right arg names
    but assigns wrong values across slots. In a well-formed call, a
    value whose slot comes first in the schema should also appear
    earlier in the prompt; arg_swap inverts that.
    """
    if len(call_args) < 2 or not schema_props:
        return 0.0
    schema_order = {k: i for i, k in enumerate(schema_props.keys())}

    positions: Dict[str, int] = {}
    for k, v in call_args.items():
        if k not in schema_order:
            continue
        p = _value_first_position(v, prompt_tokens)
        if p is not None:
            positions[k] = p
    if len(positions) < 2:
        return 0.0

    keys = list(positions.keys())
    inv = 0
    total = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            s_before = schema_order[ki] < schema_order[kj]
            p_before = positions[ki] < positions[kj]
            total += 1
            if s_before != p_before:
                inv += 1
    return inv / total if total else 0.0


def extract_drift_features(
    prompt: str,
    functions: List[Dict[str, Any]],
    tool_call: Dict[str, Any],
) -> Dict[str, float]:
    """Compute the 23-feature drift vector.

    Args:
        prompt: the user's natural-language request (or concatenated
                conversation context)
        functions: list of function schemas the model had access to,
                   each in OpenAI-function-calling format with at least
                   `{name, description, parameters}`
        tool_call: the actual call the model made, with at least
                   `{name, arguments}`

    Returns:
        dict mapping feature names (as in calibrated_weights_drift_v1.
        FEATURE_NAMES) to their computed float values.
    """
    prompt = prompt or ""
    functions = functions or []
    tool_call = tool_call or {}

    call_name = (tool_call.get("name") or "").lower()
    call_args = tool_call.get("arguments") or {}

    prompt_tokens = set(_tokens(prompt))
    prompt_content = set(_content_tokens(prompt))

    # -------- Group A: semantic alignment (5) --------
    f_tool_in_prompt = 1.0 if call_name and call_name in prompt_tokens else 0.0

    tool_parts = set(call_name.split("_"))
    f_tool_parts_in_prompt = (
        len(tool_parts & prompt_content) / max(1, len(tool_parts))
        if tool_parts else 0.0
    )

    call_text_tokens = set()
    for v in call_args.values():
        call_text_tokens.update(_tokens(str(v)))
    call_text_tokens.update(_tokens(call_name))

    union = prompt_content | call_text_tokens
    f_overlap_jaccard = (
        len(prompt_content & call_text_tokens) / max(1, len(union))
        if union else 0.0
    )

    f_prompt_coverage = (
        len(prompt_content & call_text_tokens) / max(1, len(prompt_content))
    )

    if call_args:
        hits = total = 0
        for v in call_args.values():
            vt = set(_tokens(str(v)))
            if not vt:
                continue
            total += 1
            if vt & prompt_tokens:
                hits += 1
        f_arg_verbatim_rate = hits / total if total else 0.0
    else:
        f_arg_verbatim_rate = 0.0

    # -------- Group B: schema conformance (7) --------
    schema = next(
        (fn for fn in functions if fn.get("name") == tool_call.get("name")),
        None,
    )
    if schema is None:
        f_tool_in_schema = 0.0
        f_missing_required_frac = 1.0
        f_spurious_arg_frac = 1.0
        f_type_mismatch_frac = 1.0
        f_arg_count_zscore = 0.0
        f_required_count = 0.0
        f_arg_order_inversion = 0.0
    else:
        f_tool_in_schema = 1.0
        props = (schema.get("parameters") or {}).get("properties") or {}
        required = (schema.get("parameters") or {}).get("required") or []
        spec_args = set(props.keys())
        req_args = set(required)
        called = set(call_args.keys())
        missing_req = req_args - called
        spurious = called - spec_args
        f_missing_required_frac = len(missing_req) / max(1, len(req_args))
        f_spurious_arg_frac = len(spurious) / max(1, len(spec_args))

        mismatches = 0
        checks = 0
        for k, v in call_args.items():
            if k not in props:
                continue
            checks += 1
            declared = (props[k] or {}).get("type", "")
            if declared in ("integer", "int"):
                if isinstance(v, bool) or not isinstance(v, int):
                    try:
                        int(str(v).replace(",", ""))
                    except (ValueError, TypeError):
                        mismatches += 1
            elif declared in ("number", "float"):
                try:
                    float(str(v).replace(",", ""))
                except (ValueError, TypeError):
                    mismatches += 1
            elif declared == "boolean":
                if not isinstance(v, bool):
                    mismatches += 1
            elif declared in ("array", "list"):
                if not isinstance(v, list):
                    mismatches += 1
            elif declared in ("object", "dict"):
                if not isinstance(v, dict):
                    mismatches += 1
            # string type accepts anything
        f_type_mismatch_frac = mismatches / max(1, checks)

        f_arg_count_zscore = (
            (len(called) - len(spec_args))
            / max(1.0, math.sqrt(max(1, len(spec_args))))
        )
        f_required_count = float(len(req_args))

        f_arg_order_inversion = _arg_order_inversion_rate(
            _tokens(prompt), props, call_args,
        )

    # -------- Group C: lexical drift (4) --------
    if call_args:
        placeholders = sum(
            1 for v in call_args.values()
            if isinstance(v, str) and PLACEHOLDER_RE.match(v.strip())
        )
        f_placeholder_frac = placeholders / len(call_args)
    else:
        f_placeholder_frac = 0.0

    f_tool_name_len = math.log(max(1, len(call_name)))

    f_tool_in_any_schema = (
        1.0 if call_name in {fn.get("name", "").lower() for fn in functions}
        else 0.0
    )

    f_n_available_tools = math.log(max(1, len(functions)))

    # -------- Group D: structural (7) --------
    f_n_args_called = float(len(call_args))
    f_prompt_len = math.log(max(1, len(prompt)))

    if call_args:
        avg_arg_len = sum(len(str(v)) for v in call_args.values()) / len(call_args)
    else:
        avg_arg_len = 0.0
    f_avg_arg_len = math.log(max(1.0, avg_arg_len + 1))

    f_has_nested = 1.0 if any(isinstance(v, dict) for v in call_args.values()) else 0.0
    f_has_list = 1.0 if any(isinstance(v, list) for v in call_args.values()) else 0.0
    f_prompt_is_question = 1.0 if "?" in prompt else 0.0
    f_prompt_imperative = (
        1.0 if (prompt[:1].isupper()
                and not prompt.startswith(
                    ("What", "Why", "How", "When", "Where", "Who", "Which",
                     "Can ", "Could ", "Would ", "Should ")
                ))
        else 0.0
    )

    return {
        "tool_in_prompt":         f_tool_in_prompt,
        "tool_parts_in_prompt":   f_tool_parts_in_prompt,
        "overlap_jaccard":        f_overlap_jaccard,
        "prompt_coverage":        f_prompt_coverage,
        "arg_verbatim_rate":      f_arg_verbatim_rate,
        "tool_in_schema":         f_tool_in_schema,
        "missing_required_frac":  f_missing_required_frac,
        "spurious_arg_frac":      f_spurious_arg_frac,
        "type_mismatch_frac":     f_type_mismatch_frac,
        "arg_count_zscore":       f_arg_count_zscore,
        "required_count":         f_required_count,
        "arg_order_inversion":    f_arg_order_inversion,
        "placeholder_frac":       f_placeholder_frac,
        "tool_name_len":          f_tool_name_len,
        "tool_in_any_schema":     f_tool_in_any_schema,
        "n_available_tools":      f_n_available_tools,
        "n_args_called":          f_n_args_called,
        "prompt_len":             f_prompt_len,
        "avg_arg_len":            f_avg_arg_len,
        "has_nested":             f_has_nested,
        "has_list":               f_has_list,
        "prompt_is_question":     f_prompt_is_question,
        "prompt_imperative":      f_prompt_imperative,
    }


__all__ = ["extract_drift_features"]
