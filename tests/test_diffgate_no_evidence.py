# -*- coding: utf-8 -*-
"""diffgate must not verify a claim against a diff it could not read.

Found 2026-08-21 by `styxx.flattering`, the screen frozen at 4272d44.

`only_touches` asks "is any changed path outside the prefix?" — and an empty
status answers "no". So the gate returned **VERIFIED** for the input
`"Sorry, I could not produce a diff."` The module whose entire purpose is
refusing to take the agent's word took the agent's word, and it did so by way of
a vacuous truth, which is the quietest way to be wrong.
"""
from __future__ import annotations

import pytest

from styxx.diffgate import gate_diff_text

SUMMARY = "this change only touches styxx/"
REAL = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n"
        "+++ b/styxx/x.py\n@@\n+def test_a(): pass\n")
OUTSIDE = ("diff --git a/other/y.py b/other/y.py\n--- a/other/y.py\n"
           "+++ b/other/y.py\n@@\n+y = 1\n")


@pytest.mark.parametrize("diff,label", [
    ("", "empty string"),
    ("Sorry, I could not produce a diff.", "prose, not a diff"),
    ("{\"error\": \"upstream timeout\"}", "an error payload"),
    ("<html><body>404</body></html>", "an HTML error page"),
])
def test_unreadable_diff_is_never_verified(diff, label):
    g = gate_diff_text(SUMMARY, diff)
    assert g.claims, f"{label}: the claim must still be extracted"
    for c in g.claims:
        assert c.verdict == "UNCHECKABLE", f"{label} produced {c.verdict}"
        assert c.why, "an UNCHECKABLE verdict must say why"


@pytest.mark.parametrize("diff", ["", "Sorry, I could not produce a diff."])
def test_gate_reports_that_it_did_not_run(diff):
    """PASS/FAIL cannot carry 'there was nothing to check'. `measured` can."""
    g = gate_diff_text(SUMMARY, diff)
    assert g.measured is False
    assert g.why_unmeasured
    assert g.to_dict()["measured"] is False


def test_a_parse_failure_is_distinguished_from_an_empty_diff():
    empty = gate_diff_text(SUMMARY, "")
    garbage = gate_diff_text(SUMMARY, "Sorry, I could not produce a diff.")
    assert "parse failure" in garbage.why_unmeasured
    assert "parse failure" not in empty.why_unmeasured


def test_a_readable_diff_still_verifies():
    g = gate_diff_text(SUMMARY, REAL)
    assert g.measured is True
    assert [c.verdict for c in g.claims] == ["VERIFIED"]


def test_a_readable_diff_still_contradicts():
    g = gate_diff_text(SUMMARY, OUTSIDE)
    assert g.measured is True
    assert g.verdict == "FAIL"
    assert [c.verdict for c in g.claims] == ["CONTRADICTED"]


def test_count_claims_are_not_confirmed_by_an_unreadable_diff():
    """"0 files changed" must not be verified by a diff that yielded no paths —
    the number matching is a coincidence of the failure, not evidence."""
    g = gate_diff_text("0 files were changed", "Sorry, no diff.")
    for c in g.claims:
        assert c.verdict == "UNCHECKABLE"
