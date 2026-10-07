# -*- coding: utf-8 -*-
"""Unpriced priority sentences withdrawn after the leaderboard pass, held here.

papers/NOTE_prior_art_credit_2026_09_29.md keeps a sentence that says or implies the lab did
something before others only if a survey under a frozen procedure priced it. None of these was
priced.

The guardrail instruments: the sycophancy instrument called itself the earliest instrument shipped
under the *Every Mind Leaves Vitals* call, a call addressed to the field, and the next five counted
themselves under that call as if the count covered everyone who answered it. Other answers were
never surveyed. The sycophancy instrument now says so, and the later ones count the lab's own.

The ordinal is stored reversed, as tests/test_leaderboard_no_priority_claim.py stores it, so this
file does not spell it either.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDRAIL = ROOT / "styxx" / "guardrail"

_ORDINAL = "tsrif"[::-1]
_ORDINAL_RX = re.compile(r"(?<![a-z0-9])" + _ORDINAL + r"(?![a-z0-9])", re.IGNORECASE)
# "<ordinal> [and last] [calibrated cognometric] instrument [shipped] under|after" the call, the
# paper or its publication, after comment markers and line breaks are folded to single spaces.
# "a fifth instrument under the same measurement protocol" counts the lab's own instruments and
# is not matched.
_UNDER_THE_CALL_RX = re.compile(
    r"\b(" + _ORDINAL + r"|second|third|fourth|fifth|sixth)(?: and last)?"
    r"(?: calibrated cognometric)? instrument (?:shipped )?(?:under|after) "
    r"(?:the call|that call|that paper's call|the \*Every Mind|the position paper|publication)",
    re.IGNORECASE)


def _folded(src: str) -> str:
    # join string literals continued on the next line, then fold comment markers and line breaks
    src = re.sub(r'"\s*\n\s*"', "", src)
    return re.sub(r"\s*\n\s*(?:#\s?)?", " ", src)


def unscoped_counts_under_the_call(src: str) -> list:
    """Every count of instruments under the paper's call that is not scoped to the lab's own."""
    text = _folded(src)
    return [text[max(0, m.start() - 40):m.end()] for m in _UNDER_THE_CALL_RX.finditer(text)
            if not text[:m.start()].endswith("lab's ")]


def test_the_sycophancy_note_says_other_answers_were_not_surveyed():
    from styxx.guardrail.calibrated_weights_sycophancy_v0 import CALIBRATION_NOTES
    called_in = CALIBRATION_NOTES["instrument_called_in"]
    assert "Shipped under that call; whether others answered it was not surveyed." in called_in
    assert "10.5281/zenodo.19777921" in called_in
    assert not _ORDINAL_RX.search(called_in), called_in
    assert CALIBRATION_NOTES["instrument_number"] == 4


def test_the_later_instruments_count_only_the_labs_own():
    from styxx.guardrail import (calibrated_weights_deception_v0, calibrated_weights_loop_v0,
                                 calibrated_weights_plan_action_v0)
    expected = {
        calibrated_weights_loop_v0: "the lab's second instrument under that paper's call for #4-#9, "
                                    "after sycophancy v0.",
        calibrated_weights_deception_v0: "The lab's third instrument under that paper's call for #4-#9.",
        calibrated_weights_plan_action_v0: "The lab's fourth instrument under that paper's call for #4-#9.",
    }
    for module, phrase in expected.items():
        called_in = module.CALIBRATION_NOTES["instrument_called_in"]
        assert phrase in called_in, (module.__name__, called_in)
        assert not _ORDINAL_RX.search(called_in), (module.__name__, called_in)


def test_no_guardrail_source_counts_instruments_under_the_call_unscoped():
    hits = {}
    for path in sorted(GUARDRAIL.glob("*.py")):
        found = unscoped_counts_under_the_call(path.read_text(encoding="utf-8"))
        if found:
            hits[path.name] = found
    assert hits == {}


def test_the_scan_sees_the_withdrawn_shapes():
    # the scan above is only as good as its pattern: each withdrawn shape must register
    O = _ORDINAL.capitalize()
    for text in (f'"{O} instrument shipped under that call."',
                 f"# {O} instrument shipped after the *Every Mind Leaves Vitals* position",
                 f"weights v0 — {_ORDINAL} calibrated cognometric\n"
                 "instrument shipped after the *Every Mind Leaves Vitals* position paper",
                 f"paper holds for the {_ORDINAL} instrument shipped after publication.",
                 "# detector. Sixth and last instrument shipped under the call from",
                 '        "DOI 10.5281/zenodo.19777921). Third instrument shipped under "\n'
                 '        "that paper\'s call for #4-#9."',
                 "Vitals*, the second\ninstrument under that paper's call for #4-#9."):
        assert unscoped_counts_under_the_call(text), text
    assert unscoped_counts_under_the_call("the lab's second\ninstrument under that call") == []
    assert unscoped_counts_under_the_call("a fifth\ninstrument under the same measurement protocol") == []
