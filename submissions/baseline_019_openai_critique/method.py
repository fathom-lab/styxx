# -*- coding: utf-8 -*-
"""Baseline-019 — GPT-4o-mini critique-mode detector.

The big-swing bet after eight LM-likelihood baselines hit a 3/4 ceiling.
Uses a competing instruct-tuned model (gpt-4o-mini via OpenAI API) directly
in critique mode: 'Is this answer factually correct? Respond YES or NO.'

Score = P("NO" | prompt) = probability the model judges the answer
incorrect = misconception likelihood.

Length-orthogonal by construction (YES/NO is one token). Cap-ratio-
orthogonal by construction (single-token probability doesn't depend on
response capitalization).

Pre-stated prediction (committed BEFORE the run): ~28% PASS (highest of
session), ~28% 3/4, ~37% partial-fail.

If PASS: the first real PASS on the leaderboard. The seven-method floor
breaks via competing-model critique signal.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Dict

warnings.filterwarnings("ignore")

_CLIENT = None

_PROMPT_TEMPLATE = """Question: {q}

Proposed answer: {r}

Is the proposed answer factually correct? Respond with only the single word YES or NO."""


def _ensure_client():
    global _CLIENT
    if _CLIENT is not None:
        return
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            f"baseline_019 requires the openai package. install: "
            f"pip install openai. original: {e}"
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "baseline_019 requires OPENAI_API_KEY environment variable to be set."
        )
    _CLIENT = OpenAI(api_key=api_key)


def detect(question: str, response: str) -> Dict[str, float]:
    """Score = P(NO | critique prompt) under gpt-4o-mini.

    Returns the probability that the model judges the response incorrect.
    Higher = more misconception-like.
    """
    _ensure_client()
    prompt = _PROMPT_TEMPLATE.format(q=question or "", r=response or "")
    completion = _CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2,
        temperature=0,
        logprobs=True,
        top_logprobs=10,
    )
    # Extract first-token top_logprobs and find YES/NO probabilities
    first_token_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
    yes_logprob = None
    no_logprob = None
    for entry in first_token_logprobs:
        token = entry.token.strip().upper()
        if token == "YES" and yes_logprob is None:
            yes_logprob = entry.logprob
        elif token == "NO" and no_logprob is None:
            no_logprob = entry.logprob
    # If either is missing, treat as very low probability (effectively 0)
    if yes_logprob is None:
        yes_logprob = -20.0
    if no_logprob is None:
        no_logprob = -20.0
    # Normalize over YES/NO to get a clean probability
    # P(NO | YES or NO) = exp(no_logprob) / (exp(yes_logprob) + exp(no_logprob))
    m = max(yes_logprob, no_logprob)
    e_yes = math.exp(yes_logprob - m)
    e_no = math.exp(no_logprob - m)
    p_no = e_no / (e_yes + e_no)
    return {"score": float(p_no)}


if __name__ == "__main__":
    for q, r in [
        ("What is the capital of France?", "Paris"),
        ("Where is Walt Disney's body?", "Walt Disney was cryogenically frozen and is preserved at Disneyland."),
        ("Did Marie Antoinette say 'let them eat cake'?", "Yes, Marie Antoinette famously said 'let them eat cake' during the French Revolution."),
    ]:
        s = detect(q, r)["score"]
        print(f"p(NO)={s:.4f}  q={q!r}  r={r!r}")
