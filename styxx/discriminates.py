"""styxx.discriminates — can the column you called decisive actually fail?

`protocol.require_nonvacuous_gates` refuses a preregistration whose gate no outcome row depends
on. It closes the *structural* half of "a leg that cannot fail must not gate", and its own
comment discloses the half it cannot reach:

    An unfailable BAR (`>= 0.0` on a probability) needs domain knowledge this parser does not
    have and is not attempted; that residual is disclosed rather than silently implied to be
    covered.

This module closes that residual, and it needs no domain knowledge, because the technique is
empirical rather than analytic: **score the null rule too.**

## The cycle that produced it

On 2026-08-27 a census scored five candidate span definitions for a proposed OATH clause and
reported `destroys_nominal = 0` for every one of them, in a field naming that column "the column
that decides". An adversary scored the rule that does nothing at all — no span test, every bare
numeral on a line carrying a backslash command — and it scored `destroys_nominal = 0` as well.

The best candidate and the worst rule available were indistinguishable on the deciding column.
Nothing downstream of it was evidence for anything. The column was blind for a reason no parser
could have guessed: it could only see the 184 documents that carry certificates, and every real
measurement those rules would silence lives in one of the ~935 markdown files outside that frame.

The full record is `papers/closed-model-frontier/RECON_v13_not_frozen_2026_08_27.md`. The cycle
was not frozen. This module is what would have caught it before an adversary had to.

## What it decides

For each column, given every candidate's score and the null rule's score:

* **DEGENERATE** — candidates and control all share one value. The column cannot separate
  anything from anything, and no result reported over it carries information.
* **NULL_TIES_BEST** — the control is at least as good as the best candidate. Some candidates may
  differ from each other, but none of them beats doing nothing, so the column supplies no reason
  to prefer a candidate over the null.
* **SEPARATES** — at least one candidate strictly beats the control. The column can fail, and a
  result reported over it is evidence.

A column the author declared as *deciding* which comes back anything other than SEPARATES is an
accusation. In `strict` mode that is an exception, for calling inside a preregistration before
any bar is frozen; from the CLI it is a non-zero exit.

## The honest limits, because they are the whole reason this is narrow

* **It tests the control, not the construct.** A column can separate cleanly and still measure
  the wrong thing. Seven other instances of exactly that failure are catalogued in
  `papers/SYNTHESIS_mention_and_use_2026_08_26.md`; this module would have cleared every one of
  them. Discrimination is necessary, never sufficient.
* **The control's nullity is the author's claim, not a checked fact.** Declaring a weak
  "control" that is really a fourth candidate defeats the whole check, and nothing here can
  detect that. The control should be the most permissive rule the design admits — the one that
  does no work — because that is the one whose score means "this column did not need my clause".
* **Ties are ties. There is no statistical test.** Scores are compared exactly. Two runs that
  differ only by sampling noise are read as separating, which is wrong, and a column whose spread
  is real but tiny is read as separating, which is right. Supply scores that are counts over a
  frozen frame, not estimates, or do the statistics yourself first.
* **It says nothing about coverage.** The census above scored zero on a column because its frame
  excluded every specimen that could have made it non-zero. This module reports that the column
  is degenerate; it cannot tell you that widening the frame would fix it.
* **In a cost/benefit design the null rule wins the BENEFIT column by construction, and that is
  not a defect.** A rule that fires on everything catches everything, so it maximises recall and
  ties or beats every candidate there. `NULL_TIES_BEST` on such a column is expected and means
  only "the candidates justify themselves on cost, not on reach". Declare the **cost** column as
  deciding in those designs, and read the benefit column as context. Reported when it was found:
  the mention/use census scores `of_which_nominal` (cost) `SEPARATES` — the null rule destroys
  5,159 nominal verifications and all five candidates beat it — while `accusations_reached`
  (benefit) comes back `NULL_TIES_BEST` at a control of 11. The first is the census's own declared
  deciding column and it is sound; flagging the second would be this module misreading the design.
* **But declaring the cost column as deciding is not sufficient either, and the advice above is
  half of a rule.** A candidate can beat the control on a cost column *by doing almost nothing*.
  On 2026-08-27 `oath_obligation_repair_census.py` scored word lists for repairing the obligation
  predicate: held out across documents, the top-20 list caught **1** of 85 missed claims — a
  recall of `0.012` — at a cost of 1 against a null rule costing 127. This module reported
  `SEPARATES` and `holds: true`. A rule that does nothing has a very low cost, and on a cost
  column that is indistinguishable from a rule that is cheap because it is precise.

  There is no threshold-free rule that separates that from a rule which is cheap *because it is
  precise*, and the first attempt at one proved it: flagging candidates that win only on the
  deciding column flagged **every** candidate, because in a cost/benefit design the null rule wins
  the benefit column by construction. So `report["share_of_control"]` reports, per candidate, what
  fraction of the null rule's score it retains on each column. The word list above retains `0.012`
  of the null's catches at `0.008` of its cost — a fine-looking trade whose first number is the
  one that tells you it does nothing. **Reported, not judged**: how much benefit is enough is the
  caller's question, and answering it here would be the proxy defect this module exists to catch.
* **It cannot see overfitting, and no version of it will.** The same census's word lists looked
  excellent in-sample — 36 of 85 caught at a cost of 4 — and collapsed on a held-out split. This
  module asks whether a candidate beats doing nothing; that is a different question from whether
  the candidate was fitted on the data it is scored on. **A discrimination check is not a
  substitute for a held-out evaluation, and passing this one licenses nothing about
  generalisation.**

CLI:
  python -m styxx.discriminates SCORES.json [--json OUT.json]

where SCORES.json is::

    {"directions": {"reaches": "higher_is_better", "destroys": "lower_is_better"},
     "deciding":   ["destroys"],
     "control":    {"name": "no span test at all", "scores": {"reaches": 6, "destroys": 0}},
     "candidates": {"S2_inline_code": {"reaches": 3, "destroys": 0}}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = [
    "DiscriminationError",
    "SEPARATES",
    "DEGENERATE",
    "NULL_TIES_BEST",
    "discrimination_report",
    "render",
]

SEPARATES = "SEPARATES"
DEGENERATE = "DEGENERATE"
NULL_TIES_BEST = "NULL_TIES_BEST"

HIGHER = "higher_is_better"
LOWER = "lower_is_better"


class DiscriminationError(Exception):
    """A column declared as deciding cannot separate the best candidate from the null rule."""


def _better(a: float, b: float, direction: str) -> bool:
    """Is `a` strictly better than `b` under `direction`?"""
    return a > b if direction == HIGHER else a < b


def discrimination_report(candidates: dict, control: dict, directions: dict,
                          deciding=()) -> dict:
    """Score every column for whether any candidate strictly beats the null rule.

    `candidates` maps a candidate name to {column: value}. `control` is the null rule's
    {column: value}. `directions` maps each column to HIGHER or LOWER. `deciding` names the
    columns the author claims decide the choice; those are the ones that become accusations.
    """
    if not candidates:
        raise ValueError("no candidates: a discrimination check over an empty candidate set is "
                         "itself a vacuous gate, which is the defect this module exists to "
                         "catch")
    unknown = sorted(set(directions.values()) - {HIGHER, LOWER})
    if unknown:
        raise ValueError(f"unknown direction(s) {unknown}; use {HIGHER!r} or {LOWER!r}")

    columns = sorted(directions)
    missing = sorted(c for c in columns if c not in control)
    if missing:
        raise ValueError(f"control does not score column(s) {missing}. A column with no control "
                         f"score cannot be checked, and reporting it as SEPARATES because the "
                         f"control is absent is the failure mode inverted.")

    report = {"columns": {}, "accusations": [], "deciding": sorted(deciding)}
    for col in columns:
        direction = directions[col]
        ctl = control[col]
        scored = {n: s[col] for n, s in candidates.items() if col in s}
        if not scored:
            continue
        values = set(scored.values()) | {ctl}
        winners = sorted(n for n, v in scored.items() if _better(v, ctl, direction))
        if len(values) == 1:
            verdict = DEGENERATE
        elif not winners:
            verdict = NULL_TIES_BEST
        else:
            verdict = SEPARATES
        report["columns"][col] = {
            "verdict": verdict,
            "direction": direction,
            "control": ctl,
            "beats_control": winners,
            "distinct_values": len(values),
            "spread": [min(values), max(values)],
        }

    # A candidate can beat the control on a COST column by doing almost nothing, and no
    # threshold-free rule distinguishes that from a rule which is cheap because it is precise.
    # A first attempt flagged candidates winning only on the deciding column -- useless, because
    # in any cost/benefit design the null rule wins the benefit column by construction, so it
    # flagged every candidate including the good ones.
    #
    # So this REPORTS rather than judges: for every candidate, what share of the control's score
    # it retains on each column. On 2026-08-27 the obligation-repair census had a word list
    # retaining 0.012 of the null rule's catches at 0.008 of its cost. Both look fine as a trade;
    # the 0.012 is what tells you the rule does nothing, and only a human comparing it against
    # what the repair needs can say so. Deciding which column carries the benefit, and how much of
    # it is enough, is the caller's judgement -- guessing it here would be the proxy defect this
    # module exists to catch.
    shares = {}
    for name, scores in candidates.items():
        row = {}
        for col in columns:
            if col not in scores:
                continue
            ctl = control[col]
            row[col] = round(scores[col] / ctl, 4) if ctl else None
        shares[name] = row
    report["share_of_control"] = shares
    report["share_of_control_note"] = (
        "Per candidate, its score divided by the null rule's on each column. A candidate that "
        "beats the control on a cost column while retaining almost none of its benefit is a rule "
        "that does nothing cheaply. Read these before believing a cost-column pass.")

    for col in sorted(deciding):
        entry = report["columns"].get(col)
        if entry is None:
            report["accusations"].append({
                "column": col,
                "verdict": "UNSCORED",
                "why": "declared as deciding, but no candidate carries a score for it",
            })
        elif entry["verdict"] != SEPARATES:
            report["accusations"].append({
                "column": col,
                "verdict": entry["verdict"],
                "why": ("every candidate and the control share one value, so this column "
                        "separates nothing" if entry["verdict"] == DEGENERATE else
                        "the control is at least as good as the best candidate, so this column "
                        "gives no reason to prefer any candidate over doing nothing"),
            })
    report["holds"] = not report["accusations"]
    return report


def check(candidates: dict, control: dict, directions: dict, deciding=(),
          strict: bool = True) -> dict:
    """`discrimination_report`, raising DiscriminationError on any accusation when `strict`.

    Call this from a preregistration before freezing a bar against a column.
    """
    rep = discrimination_report(candidates, control, directions, deciding)
    if strict and rep["accusations"]:
        lines = "; ".join(f"{a['column']} ({a['verdict']}): {a['why']}"
                          for a in rep["accusations"])
        raise DiscriminationError(
            f"column(s) declared as deciding cannot fail: {lines}. Freezing a bar against such a "
            f"column produces a number that is about the frame rather than about the candidate. "
            f"Either widen what the column can see, or stop calling it decisive.")
    return rep


def render(rep: dict, control_name: str = "control") -> str:
    out = []
    width = max((len(c) for c in rep["columns"]), default=6)
    out.append(f"{'column'.ljust(width)}  verdict          control  beats-control")
    for col, e in rep["columns"].items():
        beats = ", ".join(e["beats_control"]) if e["beats_control"] else "-- none --"
        out.append(f"{col.ljust(width)}  {e['verdict']:<15}  {e['control']:>7}  {beats}")
    out.append("")
    out.append(f"control: {control_name}")
    if rep.get("share_of_control"):
        out.append("")
        out.append("share of the control's score retained, per candidate:")
        for n, row in rep["share_of_control"].items():
            cells = "  ".join(f"{c}={v}" for c, v in row.items())
            out.append(f"  {n:<28} {cells}")
    if rep["accusations"]:
        out.append("")
        out.append(f"{len(rep['accusations'])} column(s) declared decisive that cannot fail:")
        for a in rep["accusations"]:
            out.append(f"  {a['column']} [{a['verdict']}] — {a['why']}")
    else:
        out.append("every column declared decisive is beaten by at least one candidate.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.discriminates")
    ap.add_argument("scores", help="JSON with directions/deciding/control/candidates")
    ap.add_argument("--json", dest="out", default=None)
    a = ap.parse_args(argv)

    spec = json.loads(Path(a.scores).read_text(encoding="utf-8"))
    control = spec["control"]
    ctl_scores = control["scores"] if "scores" in control else control
    ctl_name = control.get("name", "control") if isinstance(control, dict) else "control"
    rep = discrimination_report(spec["candidates"], ctl_scores, spec["directions"],
                                spec.get("deciding", ()))
    print(render(rep, ctl_name))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0 if rep["holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
