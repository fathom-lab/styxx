# -*- coding: utf-8 -*-
"""Unpriced priority sentences withdrawn after the leaderboard pass, held here.

papers/NOTE_prior_art_credit_2026_09_29.md keeps a sentence that says or implies nobody else did
something, that the lab did it before others, or that it is the only one, only if a survey under a
frozen procedure priced it. None of these was priced.

The guardrail instruments: the sycophancy instrument called itself the earliest instrument shipped
under the *Every Mind Leaves Vitals* call, a call addressed to the field, and the next five counted
themselves under that call as if the count covered everyone who answered it. Other answers were
never surveyed. The sycophancy instrument now says so, and the later ones count the lab's own.

The field-lacks and only-one sentences: three documents under docs/research/ said the field had no
instruments, no ruler, or no signal during generation, and four docstrings called a comparison,
a control surface or a method the only one. Three more guardrail instruments said in their
`prior_art_context` that no one had shipped or formalized what they detect and that styxx v0
"fills that gap", the sentence the 2026-09-29 pass narrowed in sycophancy's; the grounding and
confound audits said almost no one checks what they check; the CVP specification said a component
typically has no way to ask what CVP carries.

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
# "<ordinal> [and last|final] [up to two words] instrument [up to two words] under|after|to answer|
# answering|in answer to|in response to" the call, the paper or its publication, after comment
# markers and line breaks are folded to single spaces. "a fifth instrument under the same
# measurement protocol" counts the lab's own instruments and is not matched. The scan finds the
# same 22 sites at 566f7769 as the narrower pattern it replaced, and none now.
_UNDER_THE_CALL_RX = re.compile(
    r"\b(" + _ORDINAL + r"|second|third|fourth|fifth|sixth)(?: and (?:last|final))?"
    r"(?: [a-z-]+){0,2} instrument (?:[a-z-]+ ){0,2}?"
    r"(?:under|after|to answer|answering|in answer to|in response to) "
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


# the phrasings the 2026-10-07 scan's earlier pass missed: a gap the lab filled, a task nobody
# formalized, a check almost no one runs, a capability typical systems lack
_GAP_RX = re.compile(
    r"\b(?:fills? (?:that|this|the) gap|clos(?:es|ing) (?:this|that) gap|aware of no|"
    r"not (?:yet )?formali[sz]ed|almost (?:no one|nobody|none)|rarely the|"
    r"most agent-eval pipelines|typically has no)\b", re.IGNORECASE)


def test_the_gap_filling_prior_art_notes_say_what_was_not_surveyed():
    from styxx.guardrail import (calibrated_weights_goal_drift_v0, calibrated_weights_loop_v0,
                                 calibrated_weights_plan_action_v0)
    expected = {
        calibrated_weights_loop_v0: (
            "styxx v0 is a calibrated text-only detector with a published calibration fingerprint "
            "for this failure mode; whether others shipped one earlier was not surveyed.",
            ("AgentBench 2023", "ReAct 2022")),
        calibrated_weights_plan_action_v0: (
            "Whether others formalized it as a calibrated detection task earlier, and what other "
            "agent-eval pipelines check, was not surveyed.",
            ("ReAct (Yao et al. 2022)", "AgentBench (Liu et al. 2023)")),
        calibrated_weights_goal_drift_v0: (
            "Whether others formalized it as a calibrated detection task earlier was not surveyed.",
            ("AgentBench (Liu et al. 2023)", "ToolLLM")),
    }
    for module, (phrase, credits) in expected.items():
        prior = module.CALIBRATION_NOTES["prior_art_context"]
        assert phrase in prior, (module.__name__, prior)
        for credit in credits:
            assert credit in prior, (module.__name__, credit)
        assert not _GAP_RX.search(prior), (module.__name__, _GAP_RX.search(prior).group(0))
        assert not _ORDINAL_RX.search(prior), (module.__name__, prior)


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
                 "Vitals*, the second\ninstrument under that paper's call for #4-#9.",
                 # nearby rewordings the narrower pattern let through
                 "# detector. Fifth instrument released under the call from",
                 "# detector. The fifth instrument to answer the call from",
                 "the sixth and final cognometric instrument answering the call"):
        assert unscoped_counts_under_the_call(text), text
    assert unscoped_counts_under_the_call("the lab's second\ninstrument under that call") == []
    assert unscoped_counts_under_the_call("a fifth\ninstrument under the same measurement protocol") == []


def _flat(rel: str) -> str:
    return re.sub(r"\s+", " ", (ROOT / rel).read_text(encoding="utf-8"))


# (file, sentences withdrawn, wording that replaced them)
_FIELD_LACKS = [
    ("docs/research/fathom-spec-v0.md",
     ["Every other interpretability representation —", "None of them survive a vendor swap"],
     ["never surveyed, and it is withdrawn."]),
    ("docs/research/cognitive-metrology-charter.md",
     ["The science of cognition does not yet have its instruments.",
      "The field has tools but no measurements.", "no universal ruler",
      "This is the pre-scientific era of cognitive measurement."],
     ["None of that was surveyed, and it is withdrawn under the rule of "
      "`papers/NOTE_prior_art_credit_2026_09_29.md`",
      "Whether one already exists elsewhere was not surveyed."]),
    ("docs/research/cognitive-dynamics-v0.md",
     ["there's no feedback signal to use *during* generation", "the missing piece"],
     ["Reading and steering state during it was done earlier (Zou et al., 2023; see §1)."]),
    ("styxx/cogvm.py",
     ["Every production LLM interface today", "the only available control surface"],
     ["prompt-level alignment as the control surface they expose"]),
    ("styxx/guardrail/drift.py",
     ["the only published text-adjacent baseline"],
     ["Outperforms a published text-adjacent baseline", "others were not surveyed."]),
    ("styxx/guardrail/calibrated_weights_drift_v1.py",
     ["Only published comparable", "the only published baseline"],
     ["A published comparable (other baselines were not surveyed)", "clears that baseline"]),
    ("styxx/adapters/anthropic.py",
     ["the only way to get logprob-equivalent signal", "The closest reconstruction"],
     ["one way to get logprob-equivalent signal on the Anthropic API",
      "A reconstruction of logprob-equivalent signal without API support."]),
    ("styxx/guardrail/calibrated_weights_loop_v0.py",
     ["We are aware of no calibrated text-only detector", "styxx v0 fills that gap."],
     ["whether others \" \"shipped one earlier was not surveyed."]),
    ("styxx/guardrail/calibrated_weights_plan_action_v0.py",
     ["but is not formalized as a", "Most agent-eval pipelines check", "rarely the",
      "styxx v0 fills that gap with a"],
     ["styxx v0 is a calibrated \" \"text-only detector of the alignment between plan and"]),
    ("styxx/guardrail/calibrated_weights_goal_drift_v0.py",
     ["but is not formalized", "Most agent-eval pipelines", "styxx v0 fills that gap."],
     ["\"calibrated text-only detector of whether the trajectory \""]),
    ("styxx/claim_audit.py",
     ["Almost no one checks this mechanically"],
     ["Mechanical checks of reported numbers predate this one (statcheck, Nuijten et al., 2016, "
      "recomputes p-values)."]),
    ("styxx/audit_grounding_README.md",
     ["Almost no one checks this mechanically"],
     ["Mechanical checks of reported numbers predate this tool (statcheck, Nuijten et al., 2016, "
      "recomputes p-values)."]),
    ("styxx/confound_audit.py",
     ["should answer but almost none do"],
     ["should answer: *is my score secretly riding a confound?*"]),
    ("docs/cvp/CVP-1.0-specification.md",
     ["it typically has no principled way to ask", "CVP closes this gap."],
     ["it may need a principled way to ask",
      "CVP is one way to carry that signal; whether other protocols carry one was not surveyed."]),
]

# the files where the scan's gap vocabulary found a withdrawn sentence; none may carry it again
_GAP_FILES = ("styxx/guardrail/calibrated_weights_loop_v0.py",
              "styxx/guardrail/calibrated_weights_plan_action_v0.py",
              "styxx/guardrail/calibrated_weights_goal_drift_v0.py",
              "styxx/claim_audit.py", "styxx/audit_grounding_README.md", "styxx/confound_audit.py",
              "docs/cvp/CVP-1.0-specification.md")


def test_the_field_lacks_and_only_one_sentences_stay_withdrawn():
    for rel, withdrawn, replaced_by in _FIELD_LACKS:
        text = _flat(rel)
        for sentence in withdrawn:
            # case-sensitive: a withdrawal note may name the sentence it withdrew in lower case
            assert sentence not in text, (rel, sentence)
        for wording in replaced_by:
            assert wording in text, (rel, wording)


def test_no_gap_or_almost_no_one_sentence_returns():
    hits = {rel: _GAP_RX.findall(_flat(rel)) for rel in _GAP_FILES}
    assert {rel: found for rel, found in hits.items() if found} == {}


def test_what_the_drift_docstrings_measured_stays():
    for rel in ("styxx/guardrail/drift.py", "styxx/guardrail/calibrated_weights_drift_v1.py"):
        text = _flat(rel)
        for number in ("2601.05214", "0.716-0.721"):
            assert number in text, (rel, number)
    assert "Glaive 2,411 samples" in _flat("styxx/guardrail/calibrated_weights_drift_v1.py")
    assert "Our 0.943 on BFCL v3" in _flat("styxx/guardrail/calibrated_weights_drift_v1.py")


def test_the_drift_docstrings_call_no_baseline_the_only_one():
    # case-insensitive, any spelling of the withdrawn claim: "the only", "only published", ...
    # ("only one tool" per benchmark sample is a fact about the data and is not matched)
    only_rx = re.compile(r"\bthe only\b|\bonly (?:published|comparable|baseline)\b", re.IGNORECASE)
    for rel in ("styxx/guardrail/drift.py", "styxx/guardrail/calibrated_weights_drift_v1.py"):
        found = only_rx.findall(_flat(rel))
        assert found == [], (rel, found)
