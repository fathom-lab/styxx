# -*- coding: utf-8 -*-
"""Kill-gate for agent_audit.value_internally_consistent (oracle-free).

This is the automatic, corpus-scalable form of the L7 off-by-one catch: a
document that contradicts ITSELF is flagged with zero configuration — no
canonical "expected" value supplied. Triage flagger, not a gate.

Pre-registered predictions (stated BEFORE the run):
  P1 self-consistent doc (one value, repeated)  -> (True, ...)
  P2 self-contradicting doc (2+ distinct values) -> (False, ...) naming ALL values + lines
  P3 zero/one occurrence                          -> (True, ...) trivially consistent
                                                     (DISTINCT from the oracle checker, which
                                                      fails loudly on zero)
  P4 runs through AgentClaimAuditor as a Claim
"""
from __future__ import annotations

from styxx.agent_audit import Claim, AgentClaimAuditor, checkers


def test_p1_self_consistent_passes(tmp_path):
    (tmp_path / "paper.md").write_text(
        "abstract: 30 claims. body: 30 claims. appendix: 30 claims.\n", encoding="utf-8",
    )
    ok, ev = checkers.value_internally_consistent(
        tmp_path, path="paper.md", pattern=r"(\d+) claims",
    )
    assert ok is True, ev
    assert "all == '30'" in ev


def test_p2_self_contradiction_names_all_values(tmp_path):
    (tmp_path / "paper.md").write_text(
        "abstract: 30 claims.\nbody: 28 claims.\nappendix: 30 claims.\nfootnote: 31 claims.\n",
        encoding="utf-8",
    )
    ok, ev = checkers.value_internally_consistent(
        tmp_path, path="paper.md", pattern=r"(\d+) claims",
    )
    assert ok is False, ev
    assert "3 distinct values" in ev
    for v in ("'30'", "'28'", "'31'"):
        assert v in ev


def test_p3_zero_occurrence_trivially_consistent(tmp_path):
    (tmp_path / "paper.md").write_text("no countable assertions here.\n", encoding="utf-8")
    ok, ev = checkers.value_internally_consistent(
        tmp_path, path="paper.md", pattern=r"(\d+) claims",
    )
    assert ok is True, ev
    assert "0 occurrence" in ev


def test_p3_single_occurrence_trivially_consistent(tmp_path):
    (tmp_path / "paper.md").write_text("exactly 30 claims, mentioned once.\n", encoding="utf-8")
    ok, ev = checkers.value_internally_consistent(
        tmp_path, path="paper.md", pattern=r"(\d+) claims",
    )
    assert ok is True, ev
    assert "1 occurrence" in ev


def test_p4_through_auditor(tmp_path):
    (tmp_path / "paper.md").write_text("v: 30 claims; later 30 claims.\n", encoding="utf-8")
    claim = Claim(
        id="IC1",
        text="paper.md is internally consistent about its claim count",
        checker=checkers.value_internally_consistent,
        args={"path": "paper.md", "pattern": r"(\d+) claims"},
        expected=True,
    )
    (res,) = AgentClaimAuditor(repo_path=tmp_path).run([claim])
    assert res.verdict == "PASS", res.evidence


def test_distinct_from_oracle_checker_on_zero(tmp_path):
    # the oracle checker FAILS loudly on zero matches; the oracle-free one PASSes.
    (tmp_path / "x.md").write_text("nothing numeric here\n", encoding="utf-8")
    ok_free, _ = checkers.value_internally_consistent(
        tmp_path, path="x.md", pattern=r"(\d+) claims",
    )
    ok_oracle, _ = checkers.value_consistent_across_paths(
        tmp_path, glob="x.md", pattern=r"(\d+) claims", expected="30",
    )
    assert ok_free is True
    assert ok_oracle is False  # the two checkers diverge by design on empty input
