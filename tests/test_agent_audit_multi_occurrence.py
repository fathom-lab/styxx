# -*- coding: utf-8 -*-
"""Kill-gate for agent_audit.value_consistent_across_paths (multi-occurrence checker).

Closes the construct ceiling documented for styxx.agent_audit: the single-site
checkers are first-occurrence-only and miss systematic propagation drift (the
real off-by-one that propagated to 5 sites in the recursive-discipline paper,
caught by manual grep, not by the instrument).

Pre-registered predictions (stated BEFORE the run):
  P1 consistent fixture (value V in 3 files)           -> (True, ...)
  P2 divergent fixture, 2 wrong sites                  -> (False, ...) naming BOTH sites
  P3 zero matches (pattern anchors nothing)            -> (False, ...) fails loudly, no vacuous PASS
"""
from __future__ import annotations

from styxx.agent_audit import Claim, AgentClaimAuditor, checkers


def _write(root, name, body):
    p = root / name
    p.write_text(body, encoding="utf-8")
    return p


def test_p1_consistent_fixture_passes(tmp_path):
    _write(tmp_path, "a.md", "abstract: 30 claims audited\n")
    _write(tmp_path, "b.md", "we audited 30 claims in total\n")
    _write(tmp_path, "c.md", "the 30 claims breakdown follows\n")
    ok, ev = checkers.value_consistent_across_paths(
        tmp_path, glob="*.md", pattern=r"(\d+) claims", expected="30",
    )
    assert ok is True, ev
    assert "all == '30'" in ev


def test_p2_divergent_fixture_names_all_sites(tmp_path):
    _write(tmp_path, "a.md", "abstract: 30 claims audited\n")
    _write(tmp_path, "b.md", "we audited 28 claims in total\n")   # drift site 1
    _write(tmp_path, "c.md", "the 31 claims breakdown follows\n")  # drift site 2
    ok, ev = checkers.value_consistent_across_paths(
        tmp_path, glob="*.md", pattern=r"(\d+) claims", expected="30",
    )
    assert ok is False, ev
    # the property first-occurrence checking lacked: BOTH divergent sites surfaced
    assert "2 diverge" in ev
    assert "b.md" in ev and "c.md" in ev
    assert "'28'" in ev and "'31'" in ev


def test_p3_zero_matches_fails_loudly_not_vacuously(tmp_path):
    _write(tmp_path, "a.md", "no countable claims here at all\n")
    ok, ev = checkers.value_consistent_across_paths(
        tmp_path, glob="*.md", pattern=r"(\d+) claims", expected="30",
    )
    assert ok is False, ev
    assert "ZERO occurrences" in ev


def test_runs_through_auditor_as_a_claim(tmp_path):
    _write(tmp_path, "a.md", "version 7.7.10 in abstract\n")
    _write(tmp_path, "b.md", "see 7.7.10 release notes\n")
    claim = Claim(
        id="MO1",
        text="version string 7.7.10 is consistent across all docs",
        checker=checkers.value_consistent_across_paths,
        args={"glob": "*.md", "pattern": r"(\d+\.\d+\.\d+)", "expected": "7.7.10"},
        expected=True,
    )
    (res,) = AgentClaimAuditor(repo_path=tmp_path).run([claim])
    assert res.verdict == "PASS", res.evidence


def test_multi_occurrence_within_single_file(tmp_path):
    # systematic drift can live inside ONE file too — a single read must catch all
    _write(tmp_path, "paper.md", "30 claims ... later: 30 claims ... typo: 29 claims\n")
    ok, ev = checkers.value_consistent_across_paths(
        tmp_path, glob="*.md", pattern=r"(\d+) claims", expected="30",
    )
    assert ok is False, ev
    assert "1 diverge" in ev and "'29'" in ev
