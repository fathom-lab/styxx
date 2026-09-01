# -*- coding: utf-8 -*-
"""Absence of evidence is not an accusation — the not-stacc fixture.

Reported 2026-08-31 by styxx's longest-running production MCP user: on Claude Code
(Anthropic exposes no logprobs), every observe/verify call returned
classification="adversarial" / gate="fail", so every tool chain ended in a standing false
accusation for months. The first external bug report this project received, and it was a
boundary-confession failure. These tests pin the repaired contract: no logprobs means
UNMEASURED — the gate says it did not run, and never picks the accusing half.
"""
from styxx.cognometrics import tool_observe_response, tool_verify_response

ANTHROPIC_SHAPED = {"response": {"choices": [{"message": {"content": "hello there"}}]}}


def test_no_logprobs_is_unmeasured_not_adversarial():
    r = tool_observe_response(ANTHROPIC_SHAPED)
    assert r["classification"] == "unmeasured"
    assert r["gate"] == "unmeasured"
    assert r["measured"] is False
    assert "adversarial" not in r["classification"]
    assert "logprob" in r["reason"], "the reason must name the missing channel"


def test_verify_inherits_the_unmeasured_band():
    r = tool_verify_response(ANTHROPIC_SHAPED)
    assert r.get("classification") == "unmeasured"
    assert r.get("gate") != "fail", "no evidence must never gate-fail"


def test_real_logprobs_still_classify():
    resp = {"response": {"choices": [{"message": {"content": "hi"},
            "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2},
                                     {"logprob": -0.15}, {"logprob": -0.3}]}}]}}
    r = tool_observe_response(resp)
    assert r["classification"] != "unmeasured", "real trajectories must still be read"
