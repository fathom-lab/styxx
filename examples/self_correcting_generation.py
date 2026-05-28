# -*- coding: utf-8 -*-
"""Self-correcting generation demo using styxx.critique_detector.

Demonstrates the *deployment-time* application of the generation-vs-critique
asymmetry measured in `papers/agent-self-audit/FINDING_generation_critique_asymmetry_2026_05_27.md`.

Pipeline:
  1. Generate an answer to a question with gpt-4o-mini in generation mode.
  2. Score the answer with styxx.critique_detector (same model in critique mode).
  3. If the critique score crosses a threshold (default 0.50), regenerate
     with the critique as a correction hint.
  4. Return the corrected answer + the critique trace.

The mechanism: on 91% of well-known misconceptions, the same RLHF-tuned
model will both generate the misconception and flag it when asked directly.
This pipeline catches the generation error using a single additional
inference call to the same model.

Usage:

    python examples/self_correcting_generation.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from styxx import critique_detector


@dataclass
class SelfCorrectingResult:
    question: str
    initial_response: str
    initial_critique_score: float
    corrected_response: Optional[str]
    final_critique_score: Optional[float]
    was_corrected: bool


def self_correct(
    question: str,
    model: str = "gpt-4o-mini",
    threshold: float = 0.50,
) -> SelfCorrectingResult:
    """Generate an answer, critique it, regenerate if critique fails.

    Args:
        question: The user's question.
        model: The OpenAI model to use for both generation and critique.
        threshold: P(NO) above which to regenerate (default 0.50).
    """
    client = OpenAI()
    detector = critique_detector(model=model)

    # 1. Initial generation
    gen_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer:"}],
        max_tokens=200,
        temperature=0,
    )
    initial = gen_completion.choices[0].message.content.strip()

    # 2. Critique the initial response
    initial_score = detector(question, initial)

    if initial_score < threshold:
        return SelfCorrectingResult(
            question=question,
            initial_response=initial,
            initial_critique_score=initial_score,
            corrected_response=None,
            final_critique_score=None,
            was_corrected=False,
        )

    # 3. Regenerate with the critique as a hint
    correction_prompt = (
        f"Question: {question}\n\n"
        f"A first attempt produced this answer:\n  {initial}\n\n"
        f"A factuality check flagged that answer as likely incorrect. "
        f"Please provide a more accurate answer to the question."
    )
    corr_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": correction_prompt}],
        max_tokens=200,
        temperature=0,
    )
    corrected = corr_completion.choices[0].message.content.strip()

    # 4. Re-critique the corrected response
    final_score = detector(question, corrected)

    return SelfCorrectingResult(
        question=question,
        initial_response=initial,
        initial_critique_score=initial_score,
        corrected_response=corrected,
        final_critique_score=final_score,
        was_corrected=True,
    )


def _demo():
    # A mix of well-known misconceptions + clear-truth controls
    questions = [
        "What is the capital of France?",  # truth control
        "How many bones are in the human body?",  # truth control
        "Does cracking your knuckles cause arthritis?",  # folklore
        "Do humans only use 10 percent of their brain?",  # folklore
        "Was Walt Disney's body cryogenically frozen?",  # folklore
        "Is the Great Wall of China visible from space with the naked eye?",  # folklore
    ]
    width = 78
    for q in questions:
        print("─" * width)
        print(f"Q: {q}")
        result = self_correct(q)
        print(f"\n  initial answer:  {result.initial_response[:width-18]!r}")
        print(f"  initial p(NO):   {result.initial_critique_score:.4f}")
        if result.was_corrected:
            print(f"  → REGENERATING with critique hint")
            print(f"  corrected:       {result.corrected_response[:width-18]!r}")
            print(f"  final p(NO):     {result.final_critique_score:.4f}")
            improvement = result.initial_critique_score - (result.final_critique_score or 0)
            print(f"  improvement:     {improvement:+.4f}")
        else:
            print(f"  → no correction needed (below threshold)")
        print()


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY environment variable")
    _demo()
