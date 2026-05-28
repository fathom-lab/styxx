# -*- coding: utf-8 -*-
"""Kill-gate for the version-claim disambiguation fix in extract_claims.

Surfaced by the longitudinal self-report dogfood (scripts/dogfood/
longitudinal_self_report_audit.py): auditing the repo's own AI-authored commit
history produced a 100% "contradiction" rate that was entirely an extractor
ARTIFACT — `version_pin` grabbed the OLD version off the left of a migration
arrow ("7.7.9 -> 7.7.10") and mis-typed it as a "repo is at 7.7.9" state-claim,
and truncated PEP440 pre-release suffixes ("3.1.0a1" -> "3.1.0"). Both would
mis-fire the SHIPPED `styxx audit-claims` gate on real PR bodies.

These are general correctness properties, validated on SYNTHETIC inputs (not
tuned to the 9 offending commits):

Pre-registered predictions (stated BEFORE the fix):
  P1 a bump line "X -> Y" / "X → Y" / "X to Y" yields the POST-state Y, never X.
  P2 PEP440 pre-release suffixes survive extraction (3.1.0a1 stays 3.1.0a1).
  P3 a plain state-claim ("version is 7.7.10") is unaffected (no recall loss).
  P4 the migration guard does not eat legitimate trailing prose
     ("pip install styxx==7.7.10 to reproduce" still extracts 7.7.10).
"""
from __future__ import annotations

from styxx.agent_audit import extract_claims, checkers


def _versions(text):
    return [
        c.args["version"]
        for c in extract_claims(text).claims
        if c.checker == checkers.package_version_equals
    ]


def test_p1_migration_arrow_yields_post_state():
    for text, want in [
        ("release: bump version 7.7.2 → 7.7.3", "7.7.3"),
        ("Version 7.7.0->7.7.1; more tests", "7.7.1"),
        ("Version 7.5.0 -> 7.6.0 + CHANGELOG", "7.6.0"),
        ("PyPI styxx==7.7.9 -> 7.7.10", "7.7.10"),
        ("Bumps version 6.2.0 → 6.2.1.", "6.2.1"),
        ("bumped version 1.0.0 to 1.0.1", "1.0.1"),
    ]:
        got = _versions(text)
        assert want in got, f"{text!r} -> {got}, expected post-state {want}"
        # the OLD (left-of-arrow) version must NOT be extracted as a state-claim
        old = text.split("->")[0].split("→")[0].split(" to ")[0]
        for stale in ("7.7.2", "7.7.0", "7.5.0", "7.7.9", "6.2.0", "1.0.0"):
            if stale in old and stale != want:
                assert stale not in got, f"{text!r} leaked stale LHS {stale}: {got}"


def test_p2_prerelease_suffix_survives():
    assert "3.1.0a1" in _versions("a fresh-venv install of styxx==3.1.0a1 from pypi")
    assert "3.1.0a1" in _versions("version is 3.1.0a1")
    assert "1.2.0rc2" in _versions("styxx==1.2.0rc2")
    # the bare numeric core must NOT be what we capture when a suffix is present
    assert "3.1.0" not in _versions("styxx==3.1.0a1")


def test_p3_plain_state_claim_unaffected():
    assert _versions("The released version is 7.7.10.") == ["7.7.10"]
    assert _versions("version = 9.9.9") == ["9.9.9"]


def test_p4_migration_guard_keeps_legit_trailing_prose():
    # "to reproduce" must not be mistaken for a "to <version>" migration tail
    assert "7.7.10" in _versions("Install with pip install styxx==7.7.10 to reproduce.")
