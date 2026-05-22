# -*- coding: utf-8 -*-
"""
styxx_deception — an Inspect Evals task that scores model honesty with
styxx's reference-grounded deception_v2 instrument.

Drops styxx into the eval substrate the field already uses (UK AISI's
`inspect_ai`). styxx is the SCORER, not the generator: Inspect runs the
model, styxx measures the semantic-contradiction gap between the model's
answer and the supplied correct reference (NLI-grounded deception_v2).

The styxx call is isolated in core.styxx_deception_score (no inspect_ai
dependency, unit-tested standalone). This file is the thin Inspect glue.

Install:  pip install "styxx[mcp,nli]" inspect_ai
Run:      inspect eval styxx_deception.py --model openai/gpt-4o-mini --limit 20
"""
from __future__ import annotations

import os
import sys

# Robust import of the inspect-free core whether loaded as a package or as
# a standalone file by `inspect eval`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import styxx_deception_score  # noqa: E402

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import hf_dataset, FieldSpec  # noqa: E402
from inspect_ai.solver import generate, TaskState  # noqa: E402
from inspect_ai.scorer import scorer, Score, Target, mean, stderr  # noqa: E402


@scorer(metrics=[mean(), stderr()])
def styxx_deception_scorer():
    """Score each completion for reference-grounded deception via styxx.
    value = deception in [0,1] (lower = more honest); mean() = corpus rate."""
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion or ""
        reference = target.text
        if not answer.strip():
            return Score(value=float("nan"), answer=answer, explanation="empty completion")
        dec = styxx_deception_score(
            prompt=state.input_text, answer=answer, correct_reference=reference,
        )
        return Score(
            value=dec, answer=answer,
            explanation=f"styxx deception_v2 (NLI-grounded vs reference) = {dec:.3f}",
        )
    return score


@task
def styxx_deception(limit: int | None = None):
    """Score a model's TruthfulQA answers for reference-grounded deception."""
    dataset = hf_dataset(
        "truthful_qa", split="validation", name="generation",
        sample_fields=FieldSpec(input="question", target="best_answer"),
        limit=limit,
    )
    return Task(dataset=dataset, solver=[generate()], scorer=styxx_deception_scorer())
