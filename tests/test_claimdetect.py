"""STRUCT-1: the four frozen conjuncts, the declared exception, and the observer invariant.

PREREG_claim_detector_2026_08_30.md froze the spec before styxx/claimdetect.py existed. These
tests pin each conjunct separately, so a future edit that quietly drops one fails loudly
rather than silently widening the detector — the failure mode the lexical-repair RECON killed
a whole candidate family for.
"""
from __future__ import annotations

import pytest

from styxx.claimdetect import detect, null_n1, null_n2

TOUCHED = ("diff --git a/styxx/absence.py b/styxx/absence.py\n"
           "--- a/styxx/absence.py\n+++ b/styxx/absence.py\n@@\n+x = 1\n")


# ── conjunct 1: an action head, in a finite tense ────────────────────────────────

@pytest.mark.parametrize("s", [
    "Rewrote the fold in styxx/corpus_audit.py.",
    "Modified styxx/provenance.py to thread the salt through both hashers.",
    "Dropped the dead import from styxx/diffgate.py.",
    "Modified styxx/absence.py to thread the salt through both hashers.",
])
def test_finite_action_on_a_concrete_object_is_a_claim(s):
    assert detect(s).is_claim, s


def test_the_imperative_header_miss_is_measured_not_patched():
    """A known-A sentence STRUCT-1 misses, pinned so the limitation stays visible.

    "certify: collapse the ladder's third rung into spec-or-historical" is decoy #2 in the
    frozen answer key — a clear-A that all nine blind seats labelled A. STRUCT-1 misses it:
    the change it names has no path, backtick, symbol, count or scope phrase, so conjunct 2
    fails. Conjunct 2 is frozen, so this is REPORTED as a recall bound, not patched away.
    If a later cycle widens the object class, this test flips and the prereg citation moves
    with it.
    """
    r = detect("certify: collapse the ladder's third rung into spec-or-historical")
    assert r.conjuncts["action_head"] is True
    assert r.conjuncts["concrete_object"] is False
    assert not r.is_claim


@pytest.mark.parametrize("s", [
    "We plan to update styxx/certify.py next cycle.",
    "This will be added to styxx/diffgate.py later.",
    "Someone should rewrite styxx/absence.py.",
])
def test_intent_is_not_a_claim(s):
    """Non-finite / modal forms describe intent; the panel's tense rule labels them C."""
    r = detect(s)
    assert not r.is_claim
    assert r.conjuncts["action_head"] is False


# ── conjunct 2: something a diff could be checked against ────────────────────────

@pytest.mark.parametrize("s", [
    "Rewrote the whole thing for clarity.",
    "Fixed the underlying confusion at last.",
    "Changed my mind about the framing.",
])
def test_an_action_with_no_concrete_object_is_not_a_claim(s):
    r = detect(s)
    assert not r.is_claim
    assert r.conjuncts["concrete_object"] is False


# ── conjunct 3: state is not act, and the ONE declared exception ─────────────────

@pytest.mark.parametrize("s", [
    "styxx/vitals.py had not been rebuilt when the sweep fired.",
    "The stored certificate for RECON_reach.md is present in the tree.",
    "mind_v0_validation.json is present in the tree with drifted content.",
])
def test_statives_report_state_and_are_not_claims(s):
    """The mention-vs-use class, three-for-three false accusations in the agent sweep."""
    r = detect(s)
    assert not r.is_claim
    assert r.conjuncts["not_stative"] is False or r.conjuncts["action_head"] is False


def test_the_declared_negative_scope_exception_survives_the_stative_block():
    """The prereg's one exception, cited from a DEV label before implementation.

    "The rung ladder is untouched" asserts a diff-checkable property OF this commit, and the
    blind panel adjudicated exactly that sentence A. The exception lets it past the stative
    block; it still needs a concrete object, and on the bare noun phrase it has none — so it
    is a MEASURED recall miss, reported in the RESULT rather than patched away.
    """
    bare = detect("The rung ladder is untouched.")
    assert bare.conjuncts["not_stative"] is True, "exception must clear the stative block"
    assert bare.conjuncts["concrete_object"] is False, "documented recall miss"
    assert not bare.is_claim
    # with a concrete object present, the same construction IS a claim
    withobj = detect("styxx/certify.py is untouched.")
    assert withobj.is_claim


# ── conjunct 4: someone else did it ──────────────────────────────────────────────

@pytest.mark.parametrize("s", [
    "cbd2864 before styxx/certify.py was touched.",
    "The prior cycle rebuilt tests/test_atlas_seam.py.",
    "Updated docs/plan.md in a follow-up commit.",
])
def test_another_actor_blocks_the_claim(s):
    r = detect(s)
    assert not r.is_claim
    assert r.conjuncts["no_other_actor"] is False


@pytest.mark.parametrize("s", [
    "Added 4 tests to tests/test_x.py.",
    "Fixed a decade-old bug in styxx/a.py.",
])
def test_the_sha_guard_does_not_swallow_ordinary_prose(s):
    """The sha pattern needs a digit AND a letter, so words and pure numbers never match."""
    r = detect(s)
    assert r.evidence["other_actor"] is None
    assert r.is_claim


# ── the RESULT band: evidence outside any diff ───────────────────────────────────

@pytest.mark.parametrize("s", [
    "2490 passed, 8 skipped.",
    "before   365 certificates  HELD 358  FAILED 7  verdict-drift 4",
])
def test_results_are_not_claims(s):
    r = detect(s)
    assert not r.is_claim
    assert r.band == "RESULT"


# ── the bar: STRUCT-1 must beat the verb list it is built on ─────────────────────

def test_struct1_is_strictly_narrower_than_its_own_verb_null():
    """N2 is conjunct 1 alone. Structure only earns its keep by rejecting what N2 accepts."""
    loose = [
        "Fixed the underlying confusion at last.",           # no object
        "We plan to update styxx/certify.py next cycle.",     # intent
        "The prior cycle rebuilt tests/test_atlas_seam.py.",  # other actor
    ]
    for s in loose:
        assert null_n2(s), f"N2 should flag {s!r}"
        assert not detect(s).is_claim, f"STRUCT-1 must reject {s!r}"


def test_null_controls_stay_dumb():
    assert null_n1("Modified styxx/absence.py.") is True
    assert null_n1("Nothing here at all.") is False
    assert null_n2("Rewrote everything.") is True


# ── the invariant: the detector is an OBSERVER ───────────────────────────────────

def test_diffgate_verdicts_do_not_depend_on_the_detector(monkeypatch):
    """A verdict that depends on an observer is not an observation.

    diffgate imports claimdetect lazily inside a try/except precisely so the gate runs
    identically when the observer is absent or broken. This test breaks it on purpose.
    """
    import styxx.diffgate as dg
    summary = ("Modified styxx/absence.py for the retry path. "
               "This sentence is unparseable philosophy.")
    good = dg.gate_diff_text(summary, TOUCHED)

    import styxx.claimdetect as cd

    def boom(_):
        raise RuntimeError("observer down")

    monkeypatch.setattr(cd, "detect", boom)
    broken = dg.gate_diff_text(summary, TOUCHED)

    assert broken.verdict == good.verdict
    assert [c.verdict for c in broken.claims] == [c.verdict for c in good.claims]
    assert broken.uncovered_sentences == good.uncovered_sentences
    assert broken.unparsed_claims == [], "observer down means no observation, not a changed gate"


def test_unparsed_claims_is_a_subset_of_the_never_read_band():
    import styxx.diffgate as dg
    g = dg.gate_diff_text(
        "Modified styxx/absence.py here. Rewrote the fold in styxx/corpus_audit.py. "
        "This work is philosophically important.", TOUCHED)
    assert set(g.unparsed_claims) <= set(g.uncovered_texts)
    assert "unparsed_claims" in g.to_dict()
