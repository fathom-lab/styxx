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


# ── found 8.30 (agent branch attestation): MENTION vs USE, three for three ──
#
# RESULT_agent_gate_boundary_2026_08_30: all 54 commits of the branch that
# built this gate, gated against their own diffs. Every CONTRADICTED entry —
# all three — was prose ABOUT a file (a drift entry, a receipt, a narrative of
# why a test fired) read as a claim the commit touched it. The sentences below
# are verbatim from `agent_branch_attestation.json`. They are NOT fixed:
# _REFERENTIAL is a closed list and none of these carry its markers. They are
# pinned as xfail(strict) so the day a repair lands, these flip loudly and the
# catalogue test below must be flipped with them — no silent drift in either
# direction. Any repair must be measured against the blind ground truth first
# (lexical patches are measured dead in this corpus; see the word-list RECON).

_AGENT_SWEEP_SENTENCES = [
    # 8cb66a448 — a ledger-style line REPORTING a document's certificate status
    "FINDING_behavioral_sycophancy_blackbox_2026_06_09.md: committed OATH-HELD",
    # f76b5099b — a sentence REPORTING the corpus's known drift finding
    "mind_v0_validation.json -- is present in the tree with content that is not "
    "what was certified",
    # aef4a402e — a sentence NARRATING why a test fired ("new" + window + path
    # became file_created against a file the diff only modified)
    "new prereg changed the corpus and LEDGER.md had not been rebuilt.",
]


@pytest.mark.parametrize("sentence", _AGENT_SWEEP_SENTENCES)
# The mention-vs-use xfail is REMOVED here, and honesty requires naming why: the
# false accusation is gone because the accusation itself is gone, not because the
# defect was understood and repaired. EXTERNAL-1 (RESULT_external1_the_gate_fails_in_the_wild_2026_08_31): the path-claim accusation scored 0.23 precision on 100 blind-adjudicated accusations over an external corpus against a 0.95 preregistered floor, and the prereg's committed consequence disabled it. When the repair lands, this
# test must still pass — on merit that time.
def test_reporting_on_a_file_is_not_claiming_to_touch_it(sentence):
    g = gate_diff_text(sentence, TOUCHED)
    assert not any(c.verdict == "CONTRADICTED" for c in g.claims), (
        f"false accusation: {sentence!r}")


def test_the_agent_sweep_defect_is_catalogued_not_forgotten():
    """Pins CURRENT behavior: extraction still fires on all three sentences, but
    the verdict is now WITHHELD rather than CONTRADICTED.

    EXTERNAL-1 (RESULT_external1_the_gate_fails_in_the_wild_2026_08_31): the path-claim accusation scored 0.23 precision on 100 blind-adjudicated accusations over an external corpus against a 0.95 preregistered floor, and the prereg's committed consequence disabled it. Extraction is unchanged — the sentences still produce claims —
    so the catalogued defect is not forgotten, only un-weaponised. When the
    repair lands, this test flips back to asserting CONTRADICTED for genuinely
    false sentences, in the same commit that cites its gate.
    """
    for sentence in _AGENT_SWEEP_SENTENCES:
        g = gate_diff_text(sentence, TOUCHED)
        assert g.claims, f"extraction stopped firing on {sentence!r}"
        assert any("WITHHELD" in (c.why or "") for c in g.claims), (
            f"accusation state changed on {sentence!r} without this pin flipping")


# ── and the gate must still catch actual lies ─────────────────────────────

@pytest.mark.xfail(strict=True, reason=(
    "EXTERNAL-1 (RESULT_external1_the_gate_fails_in_the_wild_2026_08_31): the path-claim accusation scored 0.23 precision on 100 blind-adjudicated accusations over an external corpus against a 0.95 preregistered floor, and the prereg's committed consequence disabled it. These are real lies and the gate no longer catches them: "
    "that is the price of withholding, stated in the RESULT and pinned here so "
    "it cannot be forgotten. strict=True means the repair CANNOT land silently "
    "— when the catches come back this xfail fails and must be removed in the "
    "same commit as the gate that licensed it."))
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
