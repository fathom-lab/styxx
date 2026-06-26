# -*- coding: utf-8 -*-
"""Tests for styxx.audit_hf_model.

These run WITHOUT transformers/torch: the model-loading path is exercised separately (integration),
while the audit logic is driven through an injected ``score_fn`` so the full pipeline — bundled
corpus -> scores -> audit_confound -> verdict — is covered on a base (numpy+sklearn) install.
"""
import math

import pytest

import styxx
from styxx.hf_audit import (
    audit_hf_model,
    available_constructs,
    _default_target,
    _load_corpus,
    _CORPORA,
)


def test_exported_from_top_level():
    assert styxx.audit_hf_model is audit_hf_model
    assert "audit_hf_model" in styxx.__all__
    assert callable(styxx.available_constructs)


def test_available_constructs():
    assert set(available_constructs()) == {"sentiment", "toxicity"}


@pytest.mark.parametrize("construct", ["sentiment", "toxicity"])
def test_bundled_corpus_loads_and_is_well_formed(construct):
    rows = _load_corpus(construct)
    assert len(rows) == 200, "bundled boundary corpus should ship 200 items"
    for r in rows[:5] + rows[-5:]:
        assert set(("text", "label", "confound")).issubset(r)
        assert r["label"] in (0, 1)
        assert isinstance(r["confound"], (int, float))
    # both classes present (not degenerate)
    labels = {r["label"] for r in rows}
    assert labels == {0, 1}


def test_unknown_construct_raises():
    with pytest.raises(ValueError):
        _load_corpus("nonsense")
    with pytest.raises(ValueError):
        audit_hf_model("any/model", construct="nonsense", score_fn=lambda t: 0.5)


def test_default_target_sentiment_binary():
    assert _default_target({"positive": 0.7, "negative": 0.3}, "sentiment") == pytest.approx(0.7)


def test_default_target_sentiment_three_class():
    assert _default_target({"negative": 0.3, "neutral": 0.5, "positive": 0.2}, "sentiment") == pytest.approx(0.2)


def test_default_target_sentiment_stars():
    prob = {"1 star": 0.1, "2 stars": 0.1, "3 stars": 0.2, "4 stars": 0.3, "5 stars": 0.3}
    # E[stars] = 3.6 -> (3.6 - 1) / 4 = 0.65
    assert _default_target(prob, "sentiment") == pytest.approx(0.65, abs=1e-6)


def test_default_target_toxicity_binary_and_multilabel():
    assert _default_target({"neutral": 0.2, "toxic": 0.8}, "toxicity") == pytest.approx(0.8)
    assert _default_target({"toxic": 0.6, "insult": 0.4, "threat": 0.1}, "toxicity") == pytest.approx(0.6)


def test_default_target_refuses_opaque_labels():
    # opaque LABEL_0/LABEL_1 heads (sentiment OR toxicity) -> we refuse to guess polarity
    # (returning None) rather than risk a silently-inverted score; caller passes score_label.
    assert _default_target({"label_0": 0.5, "label_1": 0.5}, "sentiment") is None
    assert _default_target({"label_0": 0.3, "label_1": 0.7}, "toxicity") is None


def test_default_target_mixed_star_head_does_not_misfire():
    # a head with star labels AND a non-star label must NOT be treated as a pure star head
    prob = {"1 star": 0.2, "5 stars": 0.3, "spam": 0.5}
    assert _default_target(prob, "sentiment") is None  # no positive/pos key, not all-stars


@pytest.mark.parametrize("construct", ["sentiment", "toxicity"])
def test_bundled_corpus_passes_orthogonality_gate(construct):
    # the whole method is only valid if the bundled corpus is confound-orthogonal; assert it.
    rep = audit_hf_model("probe/constant", construct=construct, score_fn=lambda t: 0.5)
    assert rep.gate_ok is True
    assert abs(rep.orthogonality_corr) <= 0.20


def _row_lookup(construct):
    rows = _load_corpus(construct)
    cmean = sum(r["confound"] for r in rows) / len(rows)
    by_text = {r["text"]: r for r in rows}
    return by_text, cmean


def test_audit_hf_model_flags_a_length_biased_scorer():
    by_text, cmean = _row_lookup("sentiment")

    def biased(text):
        r = by_text[text]
        # label signal (keeps within-stratum discrimination) + a real length push
        return 0.5 + 0.30 * (2 * r["label"] - 1) + 0.25 * (r["confound"] - cmean)

    rep = audit_hf_model("synthetic/biased", construct="sentiment", score_fn=biased)
    assert rep.instrument == "synthetic/biased"
    assert rep.construct_recoverable_auc is not None and rep.construct_recoverable_auc > 0.8
    # the planted length effect must be detected: coefficient significantly positive
    lo, hi = rep.confound_score_coef_ci95
    assert rep.confound_score_coef > 0.05
    assert lo > 0.0, "lower CI should exclude zero for a planted positive length bias"
    assert rep.verdict.split()[0] in {"THRESHOLD-BIASED", "CONFOUND-DEPENDENT"}


def test_audit_hf_model_passes_a_clean_scorer():
    by_text, _ = _row_lookup("sentiment")

    def clean(text):
        r = by_text[text]
        return 0.5 + 0.40 * (2 * r["label"] - 1)  # label only, no length term

    rep = audit_hf_model("synthetic/clean", construct="sentiment", score_fn=clean)
    lo, hi = rep.confound_score_coef_ci95
    assert abs(rep.confound_score_coef) < 0.05
    assert lo <= 0.0 <= hi, "a length-independent scorer's coefficient CI should span zero"


def test_cli_audit_model_card_and_json(monkeypatch, capsys):
    """The CLI default (card) path must not crash, and --format json must serialize cleanly.
    We patch the loader to return a real report built via score_fn (no model download)."""
    import argparse
    import json as _json

    from styxx import cli
    import styxx.hf_audit as ahm

    by_text, _ = _row_lookup("sentiment")
    real_report = ahm.audit_hf_model(
        "fake/model", construct="sentiment",
        score_fn=lambda t: 0.5 + 0.30 * (2 * by_text[t]["label"] - 1) + 0.2,
    )
    # cmd_audit_model does `from .audit_hf_model import audit_hf_model` at call time, so patch there
    monkeypatch.setattr(ahm, "audit_hf_model", lambda *a, **k: real_report)

    def ns(fmt):
        return argparse.Namespace(model_id="fake/model", construct="sentiment", label=None, device=-1, format=fmt)

    assert cli.cmd_audit_model(ns("card")) == 0
    card = capsys.readouterr().out
    assert "audit-model" in card and "verdict" in card and "fake/model" in card

    assert cli.cmd_audit_model(ns("json")) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["instrument"] == "fake/model" and "verdict" in payload


def test_cli_audit_model_rejects_unknown_construct(capsys):
    import argparse
    from styxx import cli
    args = argparse.Namespace(model_id="x", construct="nonsense", label=None, device=-1, format="card")
    assert cli.cmd_audit_model(args) == 2


def test_score_fn_bypasses_transformers_entirely(monkeypatch):
    # Even with transformers unimportable, score_fn path must work (no model load attempted).
    import builtins

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "transformers":
            raise ImportError("transformers blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    rep = audit_hf_model("x/y", construct="toxicity", score_fn=lambda t: 0.5)
    assert rep.instrument == "x/y"
