# -*- coding: utf-8 -*-
"""styxx.credits — the ledger must refuse the flattering half.

The whole point of this module is that it will not quote a savings number it
cannot ground. These tests pin the refusals as hard as the arithmetic.
"""
import json

import pytest

from styxx.credits import TokenLedger, estimate_tokens, token_ledger


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _iter(msg_id, i, *, needs_revision, passed, shipped=False, draft=None):
    e = {"msg_id": msg_id, "iter": i, "composite": 0.5,
         "needs_revision": needs_revision, "passed": passed, "shipped": shipped}
    if draft is not None:
        e["draft"] = draft
    return e


def test_missing_log_refuses_rather_than_reporting_zeros(tmp_path):
    led = token_ledger(tmp_path / "nope.jsonl")
    assert led.n_messages == 0
    assert led.revision_cost_tokens is None
    assert led.catch_rate is None          # not 0.0 — that would be a claim
    assert any("no trajectory log" in r for r in led.refusals)


def test_net_is_refused_without_a_declared_counterfactual(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        _iter("m1", 0, needs_revision=True, passed=False, draft="x" * 400),
        _iter("m1", 1, needs_revision=False, passed=True, shipped=True, draft="y" * 400),
    ])
    led = token_ledger(p)
    assert led.n_catches == 1
    assert led.revision_cost_tokens == 100          # only the revision pass is billed
    assert led.net_tokens is None
    assert any("no counterfactual declared" in r for r in led.refusals)
    assert "REFUSED" in led.render()


def test_net_is_computed_only_as_conditional_on_the_declared_number(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        _iter("m1", 0, needs_revision=True, passed=False, draft="x" * 400),
        _iter("m1", 1, needs_revision=False, passed=True, shipped=True, draft="y" * 400),
    ])
    led = token_ledger(p, rework_tokens=1800)
    assert led.net_tokens == 1800 - 100
    assert led.rework_tokens == 1800
    # the render must name it as the caller's number, not a measurement
    assert "CONDITIONAL" in led.render() and "not a measurement" in led.render()


def test_absent_text_makes_cost_unmeasured_not_zero(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        _iter("m1", 0, needs_revision=True, passed=False),
        _iter("m1", 1, needs_revision=False, passed=True, shipped=True),
    ])
    led = token_ledger(p)
    assert led.revision_cost_tokens is None
    assert led.token_source == "unavailable"
    assert any("NOT zero, it is unmeasured" in r for r in led.refusals)
    # and no net can be derived from an unmeasured cost, even if asked
    assert token_ledger(p, rework_tokens=999).net_tokens is None


def test_first_draft_is_not_billed_to_the_gate(tmp_path):
    """The agent was going to write draft 0 regardless — only the revision
    passes are the gate's bill. Billing draft 0 would inflate the cost side."""
    p = tmp_path / "t.jsonl"
    _write(p, [_iter("m1", 0, needs_revision=False, passed=True, shipped=True, draft="z" * 4000)])
    led = token_ledger(p)
    assert led.n_revised == 0
    assert led.revision_cost_tokens is None      # no revision passes at all
    assert led.n_catches == 0


def test_catch_requires_the_gate_to_have_changed_the_outcome(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        # clean on the first pass: the gate changed nothing, not a catch
        _iter("clean", 0, needs_revision=False, passed=True, shipped=True, draft="a" * 40),
        # flagged and never cleared: shipped flagged, also not a catch
        _iter("stuck", 0, needs_revision=True, passed=False, draft="b" * 40),
        _iter("stuck", 1, needs_revision=True, passed=False, shipped=True, draft="c" * 40),
    ])
    led = token_ledger(p)
    assert led.n_messages == 2
    assert led.n_catches == 0
    assert led.n_shipped_still_flagged == 1
    assert led.catch_rate == 0.0                 # measured zero, on 2 real messages


def test_tokenizer_overrides_the_estimate(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        _iter("m1", 0, needs_revision=True, passed=False, draft="x" * 400),
        _iter("m1", 1, needs_revision=False, passed=True, shipped=True, draft="y" * 400),
    ])
    led = token_ledger(p, tokenizer=lambda t: 7)
    assert led.revision_cost_tokens == 7
    assert led.token_source == "tokenizer"
    assert "estimate" not in led.render()


def test_estimate_tokens_is_monotone_and_handles_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 800) > estimate_tokens("x" * 400)


def test_as_dict_round_trips_the_refusals(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [_iter("m1", 0, needs_revision=False, passed=True, shipped=True, draft="a" * 40)])
    d = token_ledger(p).as_dict()
    assert d["net_tokens"] is None
    assert isinstance(d["refusals"], list) and d["refusals"]
    assert json.dumps(d)          # must stay JSON-serializable for receipts


def test_cli_smoke(tmp_path, capsys):
    from styxx.credits import main
    p = tmp_path / "t.jsonl"
    _write(p, [
        _iter("m1", 0, needs_revision=True, passed=False, draft="x" * 400),
        _iter("m1", 1, needs_revision=False, passed=True, shipped=True, draft="y" * 400),
    ])
    assert main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "REFUSED" in out and "catches" in out
    assert main([str(p), "--rework-tokens", "1800", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["net_tokens"] == 1700
