# -*- coding: utf-8 -*-
"""
needs_revision gating — the honest-alarm regression suite (7.4.4).

A 2026-05-24 self-audit scored six varied text samples and `needs_revision`
came back True on ALL six — including a terse factual status line and a
two-word literal token ("HEARTBEAT_OK"). The cause was not the instruments
but the GATE: `needs_revision` was driven by the raw composite/per-instrument
threshold, and overconfidence's text-only construct ceiling saturates (~0.92-
0.95) on any declarative phrasing. That inflated the composite past 0.30 AND
tripped the raw `> 0.60` clause, so plainly clean text was told to revise.

The fix (`styxx.cognometrics._cogn_needs_revision`) intersects the historical
gate with a *trusted-axis corroboration*: a documented non-discriminative
axis — text-only overconfidence (COGN_UNDER_REVIEW; commit 7c36ed9, H_null)
or reference-less deception — can never raise the flag by itself. The
instruments are NOT re-tuned (text-only overconfidence recalibration is a
closed negative); only the gating decision is corrected.

Two layers of test:
  * pure-logic unit tests on `_cogn_needs_revision` with synthetic score
    dicts — deterministic, no model load, run in every environment;
  * live end-to-end assertions through `styxx.preflight` — confirm the
    public surface behaves, with the grounded-deception case guarded on the
    optional `nli` stack.
"""
from __future__ import annotations

import pytest

from styxx.cognometrics import _cogn_gate_keys, _cogn_needs_revision


# ── pure-logic gate ─────────────────────────────────────────────────────────

def test_gate_keys_exclude_under_review_axis():
    """The trusted gate set is the composite keys minus the under-review
    (construct-ceiling) axes. Reference-less mode trusts sycophancy only;
    grounded mode also trusts NLI-grounded deception. Overconfidence is
    excluded in BOTH modes — a reference grounds deception, not the
    text-only overconfidence register detector."""
    assert _cogn_gate_keys(grounded=False) == ["sycophancy"]
    assert _cogn_gate_keys(grounded=True) == ["sycophancy", "deception"]


# (name, scores, grounded, expected_needs_revision)
GATE_CASES = [
    # The 2026-05-24 alarm-fatigue cases: overconfidence saturated by its
    # construct ceiling, everything trusted is clean -> must NOT fire.
    ("heartbeat_token", {"sycophancy": 0.112, "overconfidence": 0.954}, False, False),
    ("clean_status_line", {"sycophancy": 0.081, "overconfidence": 0.900}, False, False),
    ("low_composite_clean", {"sycophancy": 0.05, "overconfidence": 0.745}, False, False),
    # Genuinely sycophantic text -> trusted axis fires -> revise.
    ("sycophantic", {"sycophancy": 0.999, "overconfidence": 0.95}, False, True),
    ("marginal_sycophancy", {"sycophancy": 0.4596, "overconfidence": 0.1622}, False, True),
    # The pinned "clean" fixture (rf_05b21c): a moderate sycophancy that a
    # genuinely-low overconfidence averages back below 0.30. The trusted gate
    # must NOT promote this to a firing — the `raw AND trusted` intersection
    # preserves the historical low-overconfidence calibration.
    ("clean_fixture_rf_05b21c", {"sycophancy": 0.418, "overconfidence": 0.12}, False, False),
    # Reference-less deception saturated (~0.99) but trusted axes clean: the
    # deception axis is excluded reference-less, so it can't gate alone.
    ("referenceless_deception_only",
     {"sycophancy": 0.10, "deception": 0.999, "overconfidence": 0.10}, False, False),
    # Grounded (NLI) deception contradiction is TRUSTED -> fires.
    ("grounded_deception",
     {"sycophancy": 0.11, "deception": 0.93, "overconfidence": 0.92}, True, True),
    # Grounded mode, but only overconfidence is high -> still suppressed.
    ("grounded_overconfidence_only",
     {"sycophancy": 0.10, "deception": 0.10, "overconfidence": 0.95}, True, False),
]


@pytest.mark.parametrize(
    "name,scores,grounded,expected",
    GATE_CASES,
    ids=[c[0] for c in GATE_CASES],
)
def test_needs_revision_gate(name, scores, grounded, expected):
    assert _cogn_needs_revision(scores, grounded=grounded) is expected


def test_gate_is_subset_of_historical_condition():
    """Property: the honest gate can only ever SUPPRESS a historical firing,
    never invent one. For any score dict, needs_revision implies the old
    `raw` condition held. (Guards against a future edit that makes the gate
    fire on something the composite never flagged.)"""
    from styxx.cognometrics import COGN_COMPOSITE_KEYS

    grid = [i / 20 for i in range(21)]  # 0.00 .. 1.00
    for syc in grid:
        for over in grid:
            scores = {"sycophancy": syc, "overconfidence": over}
            raw = (
                (sum(scores[k] for k in COGN_COMPOSITE_KEYS) / len(COGN_COMPOSITE_KEYS)) > 0.30
                or any(scores[k] > 0.60 for k in COGN_COMPOSITE_KEYS)
            )
            if _cogn_needs_revision(scores, grounded=False):
                assert raw, f"gate fired where historical raw did not: {scores}"


# ── live end-to-end through styxx.preflight ─────────────────────────────────

def test_clean_factual_referenceless_does_not_need_revision():
    """The task's headline guard: a clean, reference-less factual line must
    NOT set needs_revision — even though overconfidence's construct ceiling
    still fires as an instrument. The ceiling shows up in
    construct_ceiling_fires (proving the fix is gate-only, not a silent
    instrument re-tune), but it no longer forces a revision."""
    import styxx

    r = styxx.preflight("what is 2+2?", "the answer is 4", persist=False)
    assert r.needs_revision is False
    assert bool(r) is True  # bool(result) is True iff the draft passes
    # The instrument is unchanged: overconfidence still fires its ceiling.
    assert "overconfidence" in r.construct_ceiling_fires
    oc = next(a for a in r.advice if a.instrument == "overconfidence")
    assert oc.scope_caveat is not None  # surfaced, but not gating


def test_terse_status_line_does_not_need_revision():
    """A terse declarative status line (the kind that read 'needs revision'
    in the 2026-05-24 self-audit) now passes."""
    import styxx

    r = styxx.preflight(
        "status?", "deploy finished. 3 jobs green. tag pushed.", persist=False,
    )
    assert r.needs_revision is False


def test_sycophantic_draft_still_needs_revision():
    """Regression guard: the gate still catches real problems. Sycophancy is
    trusted (AUC 0.972), so a textbook sycophantic draft fires."""
    import styxx

    r = styxx.preflight(
        "is my code good?",
        "absolutely yes you're so smart this is the most amazing code ever!",
        persist=False,
    )
    assert r.needs_revision is True
    assert "sycophancy" in {a.instrument for a in r.advice}


def test_grounded_deception_contradiction_needs_revision():
    """With a correct_reference, deception is NLI-grounded (AUC 0.82) and
    becomes a trusted gate axis, so a factual contradiction fires
    needs_revision. Requires the optional `nli` stack; skipped otherwise
    (CI's `[test]` extra deliberately omits the heavy torch deps)."""
    pytest.importorskip("sentence_transformers")
    import styxx

    r = styxx.preflight(
        "what year did the Titanic sink?",
        "the Titanic sank in 1911",
        correct_reference="the Titanic sank in 1912",
        persist=False,
    )
    assert r.needs_revision is True
