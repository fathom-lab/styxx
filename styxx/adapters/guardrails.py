# -*- coding: utf-8 -*-
"""Guardrails AI validator — styxx-backed hallucination detection.

Usage:

    from guardrails import Guard
    from styxx.adapters.guardrails import HallucinationCheck

    guard = Guard().use(
        HallucinationCheck(threshold=0.7, on_fail="exception"),
        on="$",  # validate the root value
    )

    validated = guard.parse(
        llm_output,
        metadata={
            "prompt":    "who directed inception?",
            "reference": "Inception is a 2010 film by Christopher Nolan.",
        },
    )

    # Metadata key aliases accepted: reference, context, passage, docs,
    # source, knowledge, grounding, retrieved — same set styxx's @trust
    # auto-detects on kwargs.

Requires the `guardrails-ai` package and `styxx[nli]`:

    pip install styxx[nli] guardrails-ai
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

try:
    from guardrails.validator_base import (
        Validator,
        register_validator,
        ValidationResult,
        PassResult,
        FailResult,
    )
except ImportError as e:
    raise ImportError(
        "styxx.adapters.guardrails requires the `guardrails-ai` package. "
        "Install with: pip install guardrails-ai"
    ) from e


_REFERENCE_METADATA_KEYS = (
    "reference", "context", "passage", "passages",
    "docs", "documents", "source", "sources",
    "knowledge", "grounding", "retrieved", "retrieval",
)


def _reference_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """Match the same auto-detect list used by styxx.trust on kwargs."""
    for key in _REFERENCE_METADATA_KEYS:
        if key not in metadata:
            continue
        val = metadata[key]
        if isinstance(val, str) and val.strip():
            return val
        if (isinstance(val, (list, tuple))
                and all(isinstance(x, str) for x in val) and val):
            return "\n".join(val)
    return None


@register_validator(name="fathom-lab/hallucination_check",
                     data_type="string")
class HallucinationCheck(Validator):
    """Guardrails AI validator backed by styxx's 9-signal cross-validated
    hallucination detector.

    Parameters
    ----------
    threshold : float
        Risk threshold (0-1). Responses scoring ≥ threshold fail the
        validator. Default 0.7, same as `@trust`'s default.
    use_nli : bool, optional
        Override NLI usage. Default `None` = auto-enable when
        `styxx[nli]` is installed.
    use_entity_verify : bool
        Run Wikipedia entity verification. Adds latency. Default False
        to keep the validator fast by default.
    use_probe : bool
        Use the residual-level confab probe. Default False (needs a
        loaded HF model).
    on_fail : callable, optional
        Standard Guardrails on_fail behavior.

    Metadata
    --------
    The validator reads these keys from the `metadata` dict passed to
    the Guard at parse time (same aliases as styxx.trust):

    - `prompt`     : the user prompt / question
    - `reference` / `context` / `passage` / `docs` / `source` /
      `knowledge` / `grounding` / `retrieved` : the grounding text
      (first match wins, list/tuple is joined with newlines)

    Published benchmark AUCs
    -------------------------
    3-seed averaged, n=150/dataset:

        HaluEval-QA           0.998
        TruthfulQA            0.994
        HaluBench-RAGTruth    0.807
        HaluBench-PubMedQA    0.719
        HaluEval-Dialog       0.676
        HaluEval-Summarization 0.643
        HaluBench-FinanceBench 0.492  (published failure mode)
        HaluBench-DROP        0.424  (published failure mode)

    The validator is NOT reliable for extractive-span reading-comp
    (DROP) or financial-arithmetic (FinanceBench) hallucinations — see
    https://fathom.darkflobi.com/cognometry/failures for details.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_nli: Optional[bool] = None,
        use_entity_verify: bool = False,
        use_probe: bool = False,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail=on_fail,
            threshold=threshold,
            use_nli=use_nli,
            use_entity_verify=use_entity_verify,
            use_probe=use_probe,
            **kwargs,
        )
        self._threshold = threshold
        self._use_nli = use_nli
        self._use_entity_verify = use_entity_verify
        self._use_probe = use_probe

    def validate(
        self, value: Any, metadata: Dict[str, Any],
    ) -> ValidationResult:
        from ..guardrail import check

        response_text = str(value) if value is not None else ""
        if not response_text.strip():
            return PassResult(metadata={"risk": 0.0, "reason": "empty"})

        prompt = str(metadata.get("prompt", "")) if metadata else ""
        reference = (
            _reference_from_metadata(metadata) if metadata else None
        )

        verdict = check(
            prompt=prompt,
            response=response_text,
            reference=reference,
            use_entity_verify=self._use_entity_verify,
            use_probe=self._use_probe,
            use_nli=self._use_nli,
        )

        risk = float(verdict.risk)
        if risk >= self._threshold:
            signal_summary = {
                s.name: round(float(s.value), 4) if isinstance(s.value, (int, float)) else s.value
                for s in verdict.signals
            }
            return FailResult(
                error_message=(
                    f"hallucination risk {risk:.3f} "
                    f"(threshold {self._threshold:.2f}); "
                    f"action={verdict.action}"
                ),
                fix_value=None,
                metadata={
                    "risk": risk,
                    "action": verdict.action,
                    "signals": signal_summary,
                    "threshold": self._threshold,
                },
            )
        return PassResult(
            metadata={
                "risk": risk,
                "action": verdict.action,
            },
        )


__all__ = ["HallucinationCheck"]
