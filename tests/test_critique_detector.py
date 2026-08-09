# -*- coding: utf-8 -*-
"""Unit tests for styxx.critique — the critique-mode misconception detector.

Forensic note: styxx/critique.py is the Baseline-019 first-PASS detector
(D1-D4 bars, dark-core benchmark) and was previously untested at the unit
level; tests/test_cli_critique.py covers the unrelated `styxx critique` CLI
command. These tests do NOT re-litigate the benchmark measurement history
(that arc is published record in the module docstring); they pin the
mechanical contract of the scorer, fully offline:

  - score = P(NO | critique prompt) via a two-way softmax over the first
    completion token's top_logprobs, matching YES/NO after strip().upper()
  - first-occurrence-wins when several YES (or NO) variants appear
  - near-miss tokens ("NOT", "YES!") never match — no substring credit
  - absent-token fallback: a missing side stays at the -20.0 logprob floor;
    both sides absent yields exactly 0.5 (the documented 0.50 threshold
    boundary — uninformative, not a verdict)
  - threshold sides: NO-dominant > 0.5 (misconception-like), YES-dominant
    < 0.5 (truth-like), per the docstring's score interpretation
  - request contract: model, max_tokens=2, temperature, logprobs=True,
    top_logprobs=10; prompt built from the template with None coerced to ""
  - client lifecycle: injected client bypasses OpenAI construction entirely;
    lazy construction reads OPENAI_API_KEY once and reuses the client;
    missing key raises RuntimeError naming the env var

The OpenAI client is faked via the `_client` dataclass field (injection the
code itself allows) or a synthetic `openai` module in sys.modules. No
network, no real key.
"""
from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import pytest

from styxx.critique import CritiqueDetector, critique_detector


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _entry(token: str, logprob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob)


def _completion(entries) -> SimpleNamespace:
    """Mimic the Chat Completions response shape the scorer walks:
    completion.choices[0].logprobs.content[0].top_logprobs"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(top_logprobs=list(entries))]
                )
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, entries):
        self._entries = entries
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _completion(self._entries)


def _fake_client(entries):
    completions = _FakeCompletions(entries)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _expected(yes_lp: float, no_lp: float) -> float:
    """Independent arithmetic for the two-way softmax the scorer computes."""
    m = max(yes_lp, no_lp)
    e_y = math.exp(yes_lp - m)
    e_n = math.exp(no_lp - m)
    return e_n / (e_y + e_n)


# ---------------------------------------------------------------------------
# score derivation
# ---------------------------------------------------------------------------

def test_score_is_two_way_softmax_over_yes_no():
    client, _ = _fake_client([_entry("No", -0.02), _entry("Yes", -4.0)])
    det = CritiqueDetector(_client=client)
    score = det.score("q", "r")
    assert isinstance(score, float)
    assert score == pytest.approx(_expected(-4.0, -0.02))
    assert 0.0 <= score <= 1.0


def test_token_variants_normalized_by_strip_upper():
    """' no\\n' and 'Yes ' must be recognized after strip().upper()."""
    client, _ = _fake_client([_entry(" no\n", -0.5), _entry("Yes ", -1.5)])
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") == pytest.approx(_expected(-1.5, -0.5))


def test_first_occurrence_wins_among_variants():
    """A later 'NO' variant must not overwrite the first-matched one."""
    client, _ = _fake_client(
        [_entry("No", -0.5), _entry(" NO", -5.0), _entry("Yes", -1.0)]
    )
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") == pytest.approx(_expected(-1.0, -0.5))


def test_near_miss_tokens_do_not_match():
    """'NOT', 'NONE', 'YES!' are not YES/NO; both sides stay at the floor."""
    client, _ = _fake_client(
        [_entry("Not", -0.1), _entry("None", -0.2), _entry("YES!", -0.3)]
    )
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") == 0.5


# ---------------------------------------------------------------------------
# absent-token robustness (the -20.0 floor)
# ---------------------------------------------------------------------------

def test_both_tokens_absent_yields_exact_half():
    client, _ = _fake_client([_entry("Maybe", -0.1)])
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") == 0.5


def test_empty_top_logprobs_yields_exact_half():
    client, _ = _fake_client([])
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") == 0.5


def test_only_no_present_scores_near_one():
    client, _ = _fake_client([_entry("No", -0.05)])
    det = CritiqueDetector(_client=client)
    score = det.score("q", "r")
    assert score == pytest.approx(_expected(-20.0, -0.05))
    assert score > 0.99


def test_only_yes_present_scores_near_zero():
    client, _ = _fake_client([_entry("Yes", -0.001)])
    det = CritiqueDetector(_client=client)
    score = det.score("q", "r")
    assert score == pytest.approx(_expected(-0.001, -20.0))
    assert score < 1e-6


# ---------------------------------------------------------------------------
# threshold sides (docstring: default reasonable threshold 0.50)
# ---------------------------------------------------------------------------

def test_no_dominant_lands_above_default_threshold():
    client, _ = _fake_client([_entry("No", -0.2), _entry("Yes", -2.0)])
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") > 0.50


def test_yes_dominant_lands_below_default_threshold():
    client, _ = _fake_client([_entry("Yes", -0.2), _entry("No", -2.0)])
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") < 0.50


# ---------------------------------------------------------------------------
# request contract and prompt construction
# ---------------------------------------------------------------------------

def test_request_kwargs_match_contract():
    client, completions = _fake_client([_entry("No", -0.1), _entry("Yes", -0.2)])
    det = CritiqueDetector(model="gpt-4o-mini", temperature=0.0, _client=client)
    det.score("Is water wet?", "Yes, water is wet.")
    assert len(completions.calls) == 1
    kwargs = completions.calls[0]
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 2
    assert kwargs["temperature"] == 0.0
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 10
    messages = kwargs["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "Is water wet?" in content
    assert "Yes, water is wet." in content
    assert "YES or NO" in content


def test_none_inputs_coerced_to_empty_strings():
    client, completions = _fake_client([_entry("Yes", -0.1)])
    det = CritiqueDetector(_client=client)
    score = det.score(None, None)
    assert 0.0 <= score <= 1.0
    content = completions.calls[0]["messages"][0]["content"]
    assert "Question: \n" in content
    assert "Proposed answer: \n" in content
    assert "None" not in content


def test_call_delegates_to_score():
    client, _ = _fake_client([_entry("No", -0.3), _entry("Yes", -1.3)])
    det = CritiqueDetector(_client=client)
    assert det("q", "r") == det.score("q", "r")


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_factory_returns_detector_with_custom_template_and_temperature():
    det = critique_detector(
        model="gpt-4o", prompt_template="Q={question} R={response}", temperature=0.7
    )
    assert isinstance(det, CritiqueDetector)
    client, completions = _fake_client([_entry("No", -0.4), _entry("Yes", -0.9)])
    det._client = client
    score = det("the question", "the response")
    assert score == pytest.approx(_expected(-0.9, -0.4))
    kwargs = completions.calls[0]
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.7
    assert kwargs["messages"][0]["content"] == "Q=the question R=the response"


def test_factory_none_template_uses_default():
    det = critique_detector(prompt_template=None)
    client, completions = _fake_client([_entry("Yes", -0.1)])
    det._client = client
    det("q", "r")
    assert "factually correct" in completions.calls[0]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# client lifecycle (synthetic openai module; no network, no real key)
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_runtime_error(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kw: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    det = CritiqueDetector()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        det.score("q", "r")


def test_client_constructed_from_env_key_once_and_reused(monkeypatch):
    constructed = []
    completions = _FakeCompletions([_entry("No", -0.1), _entry("Yes", -0.6)])

    class _FakeOpenAI:
        def __init__(self, api_key=None):
            constructed.append(api_key)
            self.chat = SimpleNamespace(completions=completions)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    det = CritiqueDetector()
    s1 = det.score("q", "r")
    s2 = det.score("q", "r")
    assert constructed == ["sk-test-not-a-real-key"]
    assert s1 == s2 == pytest.approx(_expected(-0.6, -0.1))


def test_injected_client_bypasses_openai_construction(monkeypatch):
    """With _client injected, no openai import and no env var are needed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client, _ = _fake_client([_entry("No", -0.2), _entry("Yes", -1.2)])
    det = CritiqueDetector(_client=client)
    assert det.score("q", "r") == pytest.approx(_expected(-1.2, -0.2))
