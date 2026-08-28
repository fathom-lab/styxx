"""The check that closes the residual `require_nonvacuous_gates` discloses.

The anchor test is the historical case: the 2026-08-27 span census called `destroys_nominal`
"the column that decides", every candidate scored 0, and the rule that does nothing scored 0 too.
This module has to fire on that, from the real numbers, or it buys nothing.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.discriminates import (DEGENERATE, NULL_TIES_BEST, SEPARATES,
                                 DiscriminationError, check,
                                 discrimination_report, render)

HIGHER, LOWER = "higher_is_better", "lower_is_better"
CENSUS = (Path(__file__).resolve().parents[1] / "papers" / "closed-model-frontier"
          / "formula_span_census.json")

# The 2026-08-27 census AS PUBLISHED, frozen here on purpose. These are a historical fact about
# what was reported and retracted that day, not a mirror of the live receipt — the live numbers
# have since moved, because the document reporting the census is inside the corpus it measures.
V13_CANDIDATES = {
    "S2_inline_code_with_command":     {"reaches": 3, "destroys_nominal": 0},
    "S3_indented_block_with_command":  {"reaches": 3, "destroys_nominal": 0},
    "S5_right_of_a_backslash_command": {"reaches": 6, "destroys_nominal": 0},
}
V13_CONTROL = {"reaches": 6, "destroys_nominal": 0}   # no span test at all
V13_DIRECTIONS = {"reaches": HIGHER, "destroys_nominal": LOWER}


def test_the_historical_case_is_caught():
    rep = discrimination_report(V13_CANDIDATES, V13_CONTROL, V13_DIRECTIONS,
                                deciding=["destroys_nominal"])
    assert rep["columns"]["destroys_nominal"]["verdict"] == DEGENERATE
    assert not rep["holds"]
    assert [a["column"] for a in rep["accusations"]] == ["destroys_nominal"]


def test_the_historical_case_raises_in_strict_mode():
    with pytest.raises(DiscriminationError) as e:
        check(V13_CANDIDATES, V13_CONTROL, V13_DIRECTIONS, deciding=["destroys_nominal"])
    assert "cannot fail" in str(e.value)


def test_reach_column_also_fails_because_the_null_rule_reaches_most():
    """The other column separates candidates from each other but not from the control.

    S5 reaches 6 and so does the null rule. This is the distinction DEGENERATE misses and
    NULL_TIES_BEST catches: the candidates differ, and none of them beats doing nothing.
    """
    rep = discrimination_report(V13_CANDIDATES, V13_CONTROL, V13_DIRECTIONS,
                                deciding=["reaches"])
    e = rep["columns"]["reaches"]
    assert e["verdict"] == NULL_TIES_BEST
    assert e["beats_control"] == []
    assert e["distinct_values"] == 2      # candidates DO differ from each other
    assert not rep["holds"]


def test_census_receipt_still_carries_the_control_that_convicts_it():
    """Guards the retraction: if the control is ever dropped from the receipt, fail loudly."""
    payload = json.loads(CENSUS.read_text(encoding="utf-8"))
    assert "permissive_control" in payload
    assert "RETRACTED" in payload["the_column_that_decides"]


def test_live_census_still_fails_discrimination():
    """The stable property, asserted instead of the volatile counts.

    An earlier version of this file pinned the published scores (reaches 6, and 3/3/6 across
    candidates) and asserted the receipt still held them. It does not: the reporting document is
    inside the corpus the census measures, so publishing it moved every one of those numbers.
    Pinning them here would have made this suite a tripwire on corpus growth rather than on the
    thing being tested. What is stable is the verdict — no candidate beats the null rule — and
    that is what a regression would have to break.
    """
    disc = json.loads(CENSUS.read_text(encoding="utf-8"))["discrimination"]
    assert disc["holds"] is False
    assert disc["columns"]["destroys_nominal"]["verdict"] != SEPARATES
    assert disc["columns"]["destroys_nominal"]["beats_control"] == []
    assert [a["column"] for a in disc["accusations"]] == ["destroys_nominal"]


def test_live_census_control_ties_the_best_candidate_whatever_the_values():
    """Stated as a relation rather than as values, so corpus growth cannot make it stale."""
    payload = json.loads(CENSUS.read_text(encoding="utf-8"))
    ctl = payload["permissive_control"]
    best_reach = max(c["reaches_accusations"] for c in payload["candidates"])
    assert ctl["reaches_accusations"] >= best_reach, "the null rule should reach at least as far"
    assert ctl["destroys_nominal"] == min(c["destroys_nominal"] for c in payload["candidates"])


def test_cost_benefit_design_the_null_wins_benefit_and_that_is_not_a_defect():
    """The mention/use census, with the control it never ran, measured 2026-08-27.

    A rule that fires on everything catches everything, so the null rule ties or beats every
    candidate on the BENEFIT column by construction. That is not vacuity — it means the
    candidates justify themselves on cost. Declaring the cost column as deciding is the correct
    reading, and on that column the census is sound: the null rule destroys 5,159 nominal
    verifications and all five candidates beat it.

    This guards the limitation as much as the behaviour: if a future change made the benefit
    column an accusation by default, every cost/benefit census in the repository would be
    reported as defective, which would be the instrument misreading the design.
    """
    cands = {
        "blockquote":           {"reached": 0, "nominal": 17},
        "fenced_block":         {"reached": 0, "nominal": 1},
        "inline_code":          {"reached": 4, "nominal": 31},
        "latex_on_line":        {"reached": 3, "nominal": 0},
        "quoting_verb_on_line": {"reached": 0, "nominal": 501},
    }
    control = {"reached": 11, "nominal": 5159}          # abstain every token
    directions = {"reached": HIGHER, "nominal": LOWER}

    cost_declared = discrimination_report(cands, control, directions, deciding=["nominal"])
    assert cost_declared["columns"]["nominal"]["verdict"] == SEPARATES
    assert len(cost_declared["columns"]["nominal"]["beats_control"]) == 5
    assert cost_declared["holds"], "the census's own deciding column is sound"

    assert discrimination_report(cands, control, directions)["columns"]["reached"]["verdict"] \
        == NULL_TIES_BEST, "benefit column: expected, reported, and not an accusation"


def test_a_column_that_separates_holds():
    rep = discrimination_report(
        {"good": {"cost": 1}, "bad": {"cost": 9}}, {"cost": 9}, {"cost": LOWER},
        deciding=["cost"])
    assert rep["columns"]["cost"]["verdict"] == SEPARATES
    assert rep["columns"]["cost"]["beats_control"] == ["good"]
    assert rep["holds"]
    check({"good": {"cost": 1}}, {"cost": 9}, {"cost": LOWER}, deciding=["cost"])


def test_direction_is_respected_not_guessed():
    """Same numbers, opposite declared direction, opposite verdict."""
    cands, ctl = {"a": {"x": 5}}, {"x": 1}
    assert discrimination_report(cands, ctl, {"x": HIGHER})["columns"]["x"]["verdict"] == SEPARATES
    rep = discrimination_report(cands, ctl, {"x": LOWER}, deciding=["x"])
    assert rep["columns"]["x"]["verdict"] == NULL_TIES_BEST
    assert not rep["holds"]


def test_ties_are_not_wins():
    """Equal-to-control is not better-than-control; the null rule wins ties on purpose."""
    rep = discrimination_report({"a": {"x": 4}}, {"x": 4}, {"x": HIGHER}, deciding=["x"])
    assert rep["columns"]["x"]["verdict"] == DEGENERATE
    assert not rep["holds"]


def test_undeclared_columns_are_reported_but_never_accuse():
    rep = discrimination_report(V13_CANDIDATES, V13_CONTROL, V13_DIRECTIONS)  # nothing deciding
    assert rep["columns"]["destroys_nominal"]["verdict"] == DEGENERATE
    assert rep["holds"], "a column nobody called decisive is reported, not charged"


def test_deciding_column_no_candidate_scores_is_an_accusation():
    rep = discrimination_report({"a": {"x": 1}}, {"x": 0, "y": 0},
                                {"x": HIGHER, "y": LOWER}, deciding=["y"])
    assert rep["accusations"][0]["verdict"] == "UNSCORED"


def test_empty_candidate_set_is_itself_refused():
    """Otherwise the check passes vacuously — the exact defect it exists to catch."""
    with pytest.raises(ValueError, match="vacuous"):
        discrimination_report({}, {"x": 0}, {"x": LOWER}, deciding=["x"])


def test_control_missing_a_column_is_refused_not_passed():
    with pytest.raises(ValueError, match="control does not score"):
        discrimination_report({"a": {"x": 1}}, {}, {"x": LOWER}, deciding=["x"])


def test_unknown_direction_is_refused():
    with pytest.raises(ValueError, match="unknown direction"):
        discrimination_report({"a": {"x": 1}}, {"x": 0}, {"x": "bigger"})


def test_strict_false_returns_the_report_instead_of_raising():
    rep = check(V13_CANDIDATES, V13_CONTROL, V13_DIRECTIONS,
                deciding=["destroys_nominal"], strict=False)
    assert not rep["holds"]


def test_render_names_the_column_and_the_verdict():
    rep = discrimination_report(V13_CANDIDATES, V13_CONTROL, V13_DIRECTIONS,
                                deciding=["destroys_nominal"])
    text = render(rep, "no span test at all")
    assert "destroys_nominal" in text and DEGENERATE in text
    assert "no span test at all" in text
    assert "-- none --" in text


def test_render_says_so_when_everything_holds():
    rep = discrimination_report({"a": {"x": 9}}, {"x": 1}, {"x": HIGHER}, deciding=["x"])
    assert "every column declared decisive is beaten" in render(rep)


def test_cli_exits_nonzero_on_the_historical_case(tmp_path):
    spec = tmp_path / "scores.json"
    spec.write_text(json.dumps({
        "directions": V13_DIRECTIONS,
        "deciding": ["destroys_nominal"],
        "control": {"name": "no span test at all", "scores": V13_CONTROL},
        "candidates": V13_CANDIDATES,
    }), encoding="utf-8")
    out = tmp_path / "rep.json"
    p = subprocess.run([sys.executable, "-m", "styxx.discriminates", str(spec),
                        "--json", str(out)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "DEGENERATE" in p.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["holds"] is False


def test_cli_exits_zero_when_a_candidate_beats_the_control(tmp_path):
    spec = tmp_path / "scores.json"
    spec.write_text(json.dumps({
        "directions": {"cost": LOWER},
        "deciding": ["cost"],
        "control": {"name": "null", "scores": {"cost": 9}},
        "candidates": {"good": {"cost": 2}},
    }), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "styxx.discriminates", str(spec)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "SEPARATES" in p.stdout


def test_a_candidate_that_does_nothing_beats_the_control_on_a_cost_column():
    """The obligation-repair census, held out across documents, 2026-08-27.

    A word list catching 1 of 85 missed claims -- recall 0.012 -- costs 1 against a null rule
    costing 127, so it BEATS the control on the cost column and this module reports holds. A rule
    that does nothing is cheap, and on a cost column that is indistinguishable from a rule that is
    cheap because it is precise. The pass is real and is not suppressed; what the report must do
    is make the 0.012 visible beside it.
    """
    rep = discrimination_report(
        {"does_almost_nothing": {"caught": 1, "cost": 1},
         "does_something": {"caught": 40, "cost": 20}},
        {"caught": 85, "cost": 127},
        {"caught": HIGHER, "cost": LOWER},
        deciding=["cost"])
    assert rep["columns"]["cost"]["verdict"] == SEPARATES
    assert rep["holds"], "it does beat the null on the declared deciding column"
    share = rep["share_of_control"]
    assert share["does_almost_nothing"]["caught"] == 0.0118
    assert share["does_something"]["caught"] == 0.4706
    assert share["does_almost_nothing"]["cost"] == 0.0079


def test_share_of_control_is_reported_for_every_candidate_and_column():
    rep = discrimination_report(
        {"a": {"caught": 40, "cost": 20}}, {"caught": 85, "cost": 127},
        {"caught": HIGHER, "cost": LOWER}, deciding=["cost"])
    assert set(rep["share_of_control"]["a"]) == {"caught", "cost"}
    assert "does nothing cheaply" in rep["share_of_control_note"]


def test_the_first_attempt_at_flagging_would_have_flagged_everything():
    """Why this reports rather than judges.

    In a cost/benefit design the null rule wins the benefit column by construction, so a rule that
    flags candidates winning ONLY on the deciding column flags every candidate -- including the
    good one. Pinned so nobody reintroduces it.
    """
    rep = discrimination_report(
        {"good": {"caught": 40, "cost": 20}, "useless": {"caught": 1, "cost": 1}},
        {"caught": 85, "cost": 127},
        {"caught": HIGHER, "cost": LOWER}, deciding=["cost"])
    assert rep["columns"]["caught"]["verdict"] == NULL_TIES_BEST
    assert rep["columns"]["caught"]["beats_control"] == [], (
        "neither candidate beats the null on benefit, which is why 'wins only on the deciding "
        "column' cannot separate them")


def test_render_shows_the_retained_share():
    rep = discrimination_report(
        {"useless": {"caught": 1, "cost": 1}}, {"caught": 85, "cost": 127},
        {"caught": HIGHER, "cost": LOWER}, deciding=["cost"])
    text = render(rep)
    assert "share of the control" in text
    assert "caught=0.0118" in text
