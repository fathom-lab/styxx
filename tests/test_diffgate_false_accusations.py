# -*- coding: utf-8 -*-
"""False-accusation classes, each carried by name.

A gate that accuses a summary which told the truth is worse than no gate: it
teaches people to remove it. So every false accusation ever found on real data is
pinned here, in the order it was found, with the sentence that produced it.

Found 7.29.1 (80-commit sweep): decimals, versions, DOIs and dotted module names
matched a naive any-dotted-token path pattern — closed by the extension
whitelist. Found 7.29.2 (24 agent PRs): bullet-form `path`: description, and
`fix\\w+` matching *fixture*.

Found 7.44.2, re-sweeping 150 commits and 40 public agent PRs on the CURRENT
release — the three below. The earlier receipts were real but three weeks and
fifteen releases stale, and the harness had never been committed, so nobody
(including me) could re-run them. `scripts/diffgate_validation_sweep.py` exists
so this file can never again be the only evidence.
"""
from __future__ import annotations

import pytest

from styxx.diffgate import gate_diff_text

TOUCHED = ("diff --git a/styxx/absence.py b/styxx/absence.py\n"
           "--- a/styxx/absence.py\n+++ b/styxx/absence.py\n@@\n+x = 1\n")
CREATED = ("diff --git a/pkg/brand_new.go b/pkg/brand_new.go\n"
           "--- /dev/null\n+++ b/pkg/brand_new.go\n@@\n+package x\n")
MODIFIED_GO = ("diff --git a/pkg/component_report.go b/pkg/component_report.go\n"
               "--- a/pkg/component_report.go\n+++ b/pkg/component_report.go\n@@\n+x := 1\n")


@pytest.mark.parametrize("sentence", [
    # 7.44.2 (a) — COMPARATIVE REFERENCE. From styxx 7ab75b039. The path belongs
    # to a DIFFERENT, earlier change; the sentence is drawing an analogy.
    "Fixed the same way sla.py was, confidence_measured included.",
    "Handled just like coherence.py, with the same guard.",
    "Similar to the approach in styxx/vitals.py.",
    # 7.44.2 (b) — EXPLICIT NON-INCLUSION. From styxx db6e9e684. The sentence
    # says in words that the file is NOT in this diff, and the gate accused it.
    "The durable fix (fetch-depth: 0 in test.yml) is staged in the local workflow.",
    "Updated docs/plan.md in a follow-up commit.",
    "styxx/forecast.py will be fixed separately.",
    "Left styxx/atlas.py for a later commit.",
])
def test_a_path_named_is_not_a_path_claimed(sentence):
    g = gate_diff_text(sentence, TOUCHED)
    assert [c.verdict for c in g.claims] != ["CONTRADICTED"], (
        f"false accusation regressed: {sentence!r}")
    assert not any(c.verdict == "CONTRADICTED" for c in g.claims)


def test_creation_the_noun_is_not_creation_the_act():
    """7.44.2 (c) — from a real public PR, openshift-trt/sippy-eval#188.

    `creat\\w+` also matches the NOUN "creation", which in code prose means a
    place where a struct is constructed. "at both TestComparison creation sites
    in component_report.go" became a file-created claim against a file the diff
    only modified. Exactly the 7.29.2 `fix\\w+`/"fixture" catch, one stem over.
    """
    s = ("Initialize `Explanations` to `[]string{}` instead of nil at both "
         "`TestComparison` creation sites in component_report.go")
    g = gate_diff_text(s, MODIFIED_GO)
    assert not any(c.verdict == "CONTRADICTED" for c in g.claims)


# ── and the gate must still catch actual lies ─────────────────────────────

@pytest.mark.parametrize("sentence,diff", [
    ("Created pkg/missing.go for the new gate.", CREATED),
    ("Modified styxx/nowhere.py to add the screen.", TOUCHED),
    ("Deleted styxx/still_here.py.", TOUCHED),
])
def test_real_lies_are_still_caught(sentence, diff):
    g = gate_diff_text(sentence, diff)
    assert any(c.verdict == "CONTRADICTED" for c in g.claims), (
        "narrowing the templates must remove accusations, never the catches")


@pytest.mark.parametrize("sentence,diff", [
    ("Created pkg/brand_new.go for the new gate.", CREATED),
    ("Modified styxx/absence.py to add the screen.", TOUCHED),
    ("Modified pkg/component_report.go to initialize the field.", MODIFIED_GO),
])
def test_true_claims_still_verify(sentence, diff):
    """The other half of the two-sided check: a fix that stops accusing by
    stopping extracting is not a fix. Coverage has to survive."""
    g = gate_diff_text(sentence, diff)
    assert [c.verdict for c in g.claims] == ["VERIFIED"]
