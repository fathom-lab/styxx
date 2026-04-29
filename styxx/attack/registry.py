# -*- coding: utf-8 -*-
"""
styxx.attack.registry — per-instrument adapters.

Each entry knows:
  - check_fn       : the live <instrument>_check function from styxx.guardrail
  - score_attr     : verdict attribute carrying the calibrated risk
  - inputs_from_row: corpus-row -> kwargs dict for check_fn (live re-scoring)
  - positive_label : key in the corpus row marking adversarial-positive examples
  - seed_file      : bundled seed jsonl filename inside attack/seeds/

The 6 instruments registered for 7.0.0rc1 are the ones with single-source
labeled paired corpora that ship in benchmarks/data. Refusal (XSTest external),
hallucination (entity-verify pipeline), and tool-call drift (BFCL schema)
are deferred to 7.1+.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    check_fn: Callable[..., Any]
    score_attr: str
    inputs_from_row: Callable[[Dict[str, Any]], Dict[str, Any]]
    positive_label: str
    seed_file: str
    fp_seed_file: str  # natural false-positive library (label=0 scoring high)


def _sycophancy_inputs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": row["question"], "response": row["response"]}


def _loop_inputs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"turns": list(row["turns"])}


def _goal_drift_inputs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"turns": list(row["turns"])}


def _deception_inputs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": row["question"], "response": row["response"]}


def _plan_action_inputs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan": row["plan"], "action": row["action"]}


def _overconfidence_inputs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": row["question"], "response": row["response"]}


def _build_registry() -> Dict[str, InstrumentSpec]:
    # Imports are deferred so styxx.attack can be imported without forcing
    # every instrument's transitive imports. They're cheap, but this also
    # means a missing module surfaces as a clean KeyError on attack(name).
    from styxx.guardrail import (
        sycoph_check,
        loop_check,
        goal_check,
        deception_check,
        plan_action_check,
        overconf_check,
    )

    return {
        "sycophancy": InstrumentSpec(
            name="sycophancy",
            check_fn=sycoph_check,
            score_attr="sycoph_risk",
            inputs_from_row=_sycophancy_inputs,
            positive_label="label_sycophantic",
            seed_file="sycophancy.jsonl",
            fp_seed_file="sycophancy_fp.jsonl",
        ),
        "loop": InstrumentSpec(
            name="loop",
            check_fn=loop_check,
            score_attr="loop_risk",
            inputs_from_row=_loop_inputs,
            positive_label="label_loop",
            seed_file="loop.jsonl",
            fp_seed_file="loop_fp.jsonl",  # ROBUST: file does not exist
        ),
        "goal_drift": InstrumentSpec(
            name="goal_drift",
            check_fn=goal_check,
            score_attr="drift_risk",
            inputs_from_row=_goal_drift_inputs,
            positive_label="label_drifted",
            seed_file="goal_drift.jsonl",
            fp_seed_file="goal_drift_fp.jsonl",
        ),
        "deception": InstrumentSpec(
            name="deception",
            check_fn=deception_check,
            score_attr="deception_risk",
            inputs_from_row=_deception_inputs,
            positive_label="label_dishonest",
            seed_file="deception.jsonl",
            fp_seed_file="deception_fp.jsonl",
        ),
        "plan_action": InstrumentSpec(
            name="plan_action",
            check_fn=plan_action_check,
            score_attr="gap_risk",
            inputs_from_row=_plan_action_inputs,
            positive_label="label_mismatch",
            seed_file="plan_action.jsonl",
            fp_seed_file="plan_action_fp.jsonl",
        ),
        "overconfidence": InstrumentSpec(
            name="overconfidence",
            check_fn=overconf_check,
            score_attr="overconf_risk",
            inputs_from_row=_overconfidence_inputs,
            positive_label="label_overconfident",
            seed_file="overconfidence.jsonl",
            fp_seed_file="overconfidence_fp.jsonl",
        ),
    }


_REGISTRY: Dict[str, InstrumentSpec] = {}


def get_instrument(name: str) -> InstrumentSpec:
    """Return the InstrumentSpec for `name`, building the registry lazily."""
    global _REGISTRY
    if not _REGISTRY:
        _REGISTRY = _build_registry()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown instrument {name!r}. "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_instruments() -> list[str]:
    """Return the names of all registered (mineable) instruments."""
    global _REGISTRY
    if not _REGISTRY:
        _REGISTRY = _build_registry()
    return sorted(_REGISTRY)


__all__ = ["InstrumentSpec", "get_instrument", "list_instruments"]
