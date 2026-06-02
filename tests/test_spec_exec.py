# -*- coding: utf-8 -*-
"""Tests for styxx.spec_exec — integrity-gated speculative routing (regime-1)."""
from __future__ import annotations

from styxx.spec_exec import (
    Draft, RouteResult, EpistemicSpeculativeRouter,
    entropy_gate, council_gate, calibrate_threshold,
)


def mock(answer, sig=None, intok=10, outtok=5):
    """A model fn returning n identical drafts (optionally with a precomputed signal)."""
    return lambda prompt, temp, n: [Draft(answer, intok, outtok, sig) for _ in range(n)]


def test_public_api():
    assert callable(entropy_gate) and callable(council_gate) and callable(calibrate_threshold)
    assert EpistemicSpeculativeRouter is not None
    assert Draft and RouteResult


def test_confident_draft_is_kept_cheap():
    # precomputed LOW signal (<= tau) -> keep the cheap draft, never call the verifier
    r = EpistemicSpeculativeRouter(drafter=mock("cheap", sig=0.1),
                                   verifier=mock("frontier"), tau=0.5).run("q")
    assert isinstance(r, RouteResult)
    assert not r.escalated and r.answer == "cheap" and r.verify_tokens == 0
    assert r.draft_tokens == 15  # one greedy draft, no resampling when signal precomputed


def test_uncertain_draft_escalates():
    # precomputed HIGH signal (> tau) -> escalate to the verifier
    r = EpistemicSpeculativeRouter(drafter=mock("cheap", sig=0.9),
                                   verifier=mock("frontier"), tau=0.5).run("q")
    assert r.escalated and r.answer == "frontier" and r.verify_tokens == 15


def test_resampling_gate_when_no_precomputed_signal():
    # no precomputed signal -> router resamples k times and applies the text gate.
    # identical samples -> ~0 entropy -> below tau -> keep cheap.
    r = EpistemicSpeculativeRouter(drafter=mock("Paris"), verifier=mock("frontier"),
                                   gate=entropy_gate, tau=0.5, k=4).run("capital of France?")
    assert not r.escalated and r.answer == "Paris"
    assert r.draft_tokens == 15 * 5  # greedy + k=4 resamples, all counted


def test_entropy_gate_consistent_is_low():
    assert entropy_gate(["Paris", "Paris", "Paris"]) < 0.1


def test_calibrate_threshold_recovers_the_gap():
    # hard items: high signal, cheap WRONG, verifier RIGHT. easy: low signal, cheap RIGHT.
    recs = []
    for i in range(10):
        hard = i < 5
        recs.append(dict(signal=(0.9 if hard else 0.1),
                         local_ok=(not hard), front_ok=True,
                         draft_cost=1.0, verify_cost=7.0))
    tau = calibrate_threshold(recs)
    # the right tau escalates the hard (0.9) items and keeps the easy (0.1) ones
    assert 0.1 <= tau < 0.9
    escalated_hard = 0.9 > tau
    kept_easy = not (0.1 > tau)
    assert escalated_hard and kept_easy


def test_calibrate_threshold_respects_cost_cap():
    # if escalating never helps (verifier no better) the cheapest policy is no escalation
    recs = [dict(signal=0.5, local_ok=True, front_ok=True, draft_cost=1.0, verify_cost=7.0)
            for _ in range(6)]
    tau = calibrate_threshold(recs)
    assert not (0.5 > tau)  # nothing should escalate when the verifier never helps
