# -*- coding: utf-8 -*-
"""styxx.critique — production-grade misconception detector via critique-mode prompting.

The first method to PASS the styxx gauntlet's v3 detection bars (D1+D2+D3+D4)
on the dark-core consensus-hallucination benchmark. See:

  - submissions/baseline_019_openai_critique/PRE_STATED_PREDICTION.md (commit fdcf92e)
  - submissions/baseline_019_openai_critique/submission.json (commit 17fdd97)
  - papers/agent-self-audit/FINDING_first_pass_2026_05_27.md (commit 0bc9b7b)
  - papers/agent-self-audit/FINDING_generation_critique_asymmetry_2026_05_27.md (commit ac25398)
  - papers/PAPER_recursive_discipline_2026_05_27.md §10, §11

The mechanism: RLHF-tuned LLMs exhibit a generation-vs-critique asymmetry —
the same model that produces consensus misconceptions in generation mode
correctly flags them in critique mode. Measured prevalence: 91.18% on the
n=34 folklore subset of the dark-core benchmark.

Public API:

    from styxx import critique_detector
    score_fn = critique_detector(model="gpt-4o-mini")
    score = score_fn(
        question="Is the Great Wall of China visible from space?",
        response="Yes, the Great Wall is visible from space with the naked eye.",
    )
    # score ≈ 1.0 (model judges this incorrect — high misconception likelihood)

Score interpretation: returns P(NO | critique prompt) in [0, 1].
  - Near 1.0: the model judges the response factually incorrect (likely misconception).
  - Near 0.0: the model judges the response factually correct (likely truth).

Requires OPENAI_API_KEY environment variable (for the default gpt-4o-mini backend)
or a custom callable.

Caveats:
  - The benchmark result used gpt-4o-mini, which was IN the original 3-vendor
    council that generated `expected_consensus`. The PASS verdict is real on
    pre-registered bars but demonstrates within-vendor generation-vs-critique
    asymmetry rather than pure cross-vendor signal. For pure cross-vendor
    detection, use a different-vendor model (Claude, Gemini) when API access
    is available.
  - The score is a probability, not a binary classification. Apply your own
    threshold based on use case (default reasonable threshold: 0.50).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional


__all__ = ["critique_detector", "CritiqueDetector"]


_DEFAULT_PROMPT_TEMPLATE = """Question: {question}

Proposed answer: {response}

Is the proposed answer factually correct? Respond with only the single word YES or NO."""


@dataclass
class CritiqueDetector:
    """A misconception detector that uses critique-mode prompting on an LLM.

    Score = P("NO" | critique prompt). Higher = more misconception-like.

    Attributes:
        model: HuggingFace-style model identifier for the OpenAI client
            (default "gpt-4o-mini"). For cross-vendor robustness, use a
            different-vendor model.
        prompt_template: Format string with {question} and {response} fields.
        temperature: Sampling temperature (default 0 for determinism).
    """
    model: str = "gpt-4o-mini"
    prompt_template: str = _DEFAULT_PROMPT_TEMPLATE
    temperature: float = 0.0

    _client: Optional[object] = None  # lazy-init OpenAI client

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "styxx.critique_detector requires the openai package. "
                "install: pip install openai. original: " + str(e)
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "styxx.critique_detector requires OPENAI_API_KEY environment variable "
                "(default backend is gpt-4o-mini via OpenAI Chat Completions API). "
                "Set the env var, or subclass CritiqueDetector to use a different backend."
            )
        self._client = OpenAI(api_key=api_key)

    def score(self, question: str, response: str) -> float:
        """Return P(NO | critique prompt) for a (question, response) pair.

        Higher = more misconception-like. Range [0, 1].
        """
        self._ensure_client()
        prompt = self.prompt_template.format(
            question=question or "",
            response=response or "",
        )
        completion = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=10,
        )
        first_token_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        yes_lp, no_lp = -20.0, -20.0
        for entry in first_token_logprobs:
            token = entry.token.strip().upper()
            if token == "YES" and yes_lp == -20.0:
                yes_lp = entry.logprob
            elif token == "NO" and no_lp == -20.0:
                no_lp = entry.logprob
        m = max(yes_lp, no_lp)
        e_y = math.exp(yes_lp - m)
        e_n = math.exp(no_lp - m)
        return float(e_n / (e_y + e_n))

    def __call__(self, question: str, response: str) -> float:
        return self.score(question, response)


def critique_detector(
    model: str = "gpt-4o-mini",
    prompt_template: Optional[str] = None,
    temperature: float = 0.0,
) -> Callable[[str, str], float]:
    """Return a critique-mode misconception detector as a callable.

    Args:
        model: The OpenAI Chat-API model to use (default "gpt-4o-mini").
        prompt_template: Optional custom prompt template. Must contain
            `{question}` and `{response}` placeholders. Default is the
            critique prompt from the Baseline-019 PASS submission.
        temperature: Default 0.0 for determinism.

    Returns:
        A callable `(question: str, response: str) -> float` returning
        P(NO | critique prompt) in [0, 1]. Higher = more misconception-like.

    Example:
        >>> from styxx import critique_detector
        >>> det = critique_detector(model="gpt-4o-mini")
        >>> det("What is the capital of France?", "Paris")
        # ~ 0.0 (model judges "Paris" correct)
        >>> det("Where is Walt Disney's body?",
        ...     "Walt Disney was cryogenically frozen.")
        # ~ 1.0 (model judges this incorrect)
    """
    return CritiqueDetector(
        model=model,
        prompt_template=prompt_template or _DEFAULT_PROMPT_TEMPLATE,
        temperature=temperature,
    )
