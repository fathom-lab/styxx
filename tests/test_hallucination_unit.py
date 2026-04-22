# -*- coding: utf-8 -*-
"""Unit tests for styxx.hallucination — API surface, verdict shape,
on_detect validation. Does not require a GPU or loaded model.

End-to-end with-model tests belong in a separate integration file
gated on CUDA availability."""
from __future__ import annotations

import pytest

from styxx.hallucination import (
    TokenReading, HallucinationVerdict,
    hallucination_verdict, stream_with_risk, detect_hallucination,
    ON_DETECT_CHOICES, DEFAULT_THRESHOLD, DEFAULT_PROBE_TASK,
)


def test_token_reading_shape():
    r = TokenReading(token_id=42, token_text="foo", risk=0.35,
                     will_flag=False)
    assert r.token_id == 42
    assert r.token_text == "foo"
    assert r.risk == 0.35
    assert r.will_flag is False


def test_hallucination_verdict_serializable():
    v = HallucinationVerdict(
        prompt="Summarize paper X",
        output_text="I'm sorry but I cannot verify the title.",
        output_tokens=9,
        risk_score=0.35,
        flagged_tokens=[],
        max_risk=0.42,
        halt_reason="",
        retries_used=0,
        risk_timeline=[0.1, 0.2, 0.35, 0.42, 0.31, 0.25, 0.22, 0.19, 0.17],
        probe_task="confab_prompt",
        probe_layer=1,
        probe_auc=1.0,
        threshold=0.7,
    )
    d = v.as_dict()
    assert d["risk_score"] == 0.35
    assert d["probe_task"] == "confab_prompt"
    assert d["threshold"] == 0.7
    assert d["retries_used"] == 0
    assert isinstance(d["risk_timeline"], list)
    assert len(d["risk_timeline"]) == 9


def test_on_detect_choices_stable():
    """The public on_detect vocabulary is stable across versions.
    If this test ever changes, document migration in CHANGELOG."""
    assert ON_DETECT_CHOICES == (
        "halt_and_flag",
        "flag_only",
        "retry_with_suppression",
    )


def test_detect_rejects_unknown_on_detect():
    """detect_hallucination must reject unknown on_detect strings."""
    # Using a dummy dummy model wouldn't get past resolve_probe anyway,
    # but the on_detect validation is the first thing we check.
    with pytest.raises(ValueError):
        detect_hallucination(
            model=None, tokenizer=None,
            prompt="anything",
            on_detect="delete_the_internet",
        )


def test_default_values_exported():
    assert isinstance(DEFAULT_THRESHOLD, float)
    assert 0 < DEFAULT_THRESHOLD < 1
    assert DEFAULT_PROBE_TASK == "confab_prompt"


def test_api_public_surface():
    """The module's public API is stable."""
    from styxx import hallucination as hm
    for name in (
        "TokenReading",
        "HallucinationVerdict",
        "hallucination_verdict",
        "stream_with_risk",
        "detect_hallucination",
        "ON_DETECT_CHOICES",
    ):
        assert name in hm.__all__, f"{name} missing from __all__"
        assert hasattr(hm, name), f"{name} not exposed at module level"
