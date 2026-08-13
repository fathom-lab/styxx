"""The union battery: my HARD 8 + the red team's 3 cases, scored together.

Fable's condition for accepting the lane-2 fix was explicit and correct: "a fix that passes my
fixture but breaks your HARD 8 is not a fix. commit both fixtures together; a battery is only
worth its blind spots."

Two batteries with disjoint blind spots are worth more than either. Mine tests paraphrase
(prose that does NOT reuse key names). Fable's tests path-length asymmetry (a bare summary key
colliding with a long specific path) — a shape my fixture never produced, because every path in
my receipt had the same depth. That is precisely the blind spot an author's own battery has.

Run: python papers/dogfood-self-audit/combined_resolver_battery.py
"""
from __future__ import annotations
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from styxx.claim_audit import audit_grounding, _load_sources, _candidates  # noqa: E402
from validate_context_resolution import CASES, HARD_CASES, RECEIPT  # noqa: E402

# --- the red team's cases, transcribed from redteam_2026_08_13/redteam_claim_audit.py ------
REDTEAM = [
    ("A_summary_key_beats_named_cell",
     {"rate": 0.1, "cells": {"blockconf_ge3_of_7": {"cave_rate": 0.1}}},
     "The blockconf arm at >=3/7 shows a cave rate of 0.100 on the frozen protocol.",
     "cells.blockconf_ge3_of_7.cave_rate"),
    ("B_longer_named_path_loses_to_shorter",
     {"knee": {"rate": 0.25}, "blockconf_high_confidence_arm": {"cave_rate": 0.25}},
     "the blockconf high confidence arm cave rate was 0.250",
     "blockconf_high_confidence_arm.cave_rate"),
    ("C_control_equal_length",
     {"arm_one": {"cave_rate": 0.5}, "arm_two": {"cave_rate": 0.5}},
     "arm_two cave_rate was 0.500",
     "arm_two.cave_rate"),
]


def judge(src, sentence, truth):
    """Correct, DECLINED (honest failure), or WRONG (confident-wrong — the unacceptable one)."""
    rep = audit_grounding(sentence, src)
    items = [i for i in rep.items if i.status == "grounded"]
    if not items:
        return "NO-MATCH", "", 0
    it = items[0]
    if it.source == truth:
        return "correct", it.resolved_by, it.n_candidates
    if it.resolved_by == "arbitrary":
        return "declined", it.resolved_by, it.n_candidates
    return "WRONG", it.resolved_by, it.n_candidates


def run(name, cases_with_src):
    print("=" * 78)
    print(name)
    print("=" * 78)
    tally = {"correct": 0, "declined": 0, "WRONG": 0, "NO-MATCH": 0}
    for label, src, sentence, truth in cases_with_src:
        verdict, how, k = judge(src, sentence, truth)
        tally[verdict] += 1
        mark = {"correct": "OK ", "declined": "-- ", "WRONG": "XX ", "NO-MATCH": "?? "}[verdict]
        print(f"  {mark} ({k} cands, {how:>9}) {label}")
        if verdict == "WRONG":
            print(f"        wanted {truth}")
    n = sum(tally.values())
    print(f"\n  correct {tally['correct']}/{n} · declined {tally['declined']} · "
          f"CONFIDENT-WRONG {tally['WRONG']}")
    return tally


def main():
    mine_easy = [(s, RECEIPT, s, t) for s, t in CASES]
    mine_hard = [(s, RECEIPT, s, t) for s, t in HARD_CASES]
    t_easy = run("MINE — EASY (key names reused; floor only)", mine_easy)
    print()
    t_hard = run("MINE — HARD (paraphrased prose)", mine_hard)
    print()
    t_red = run("RED TEAM — path-length asymmetry (my battery's blind spot)", REDTEAM)

    total_wrong = t_easy["WRONG"] + t_hard["WRONG"] + t_red["WRONG"]
    print("\n" + "=" * 78)
    print(f"UNION BATTERY: confident-wrong across all three = {total_wrong}")
    verdict = ("BATTERY_CLEAN__no_confident_wrong_resolution_in_any_fixture" if total_wrong == 0
               else f"BATTERY_FAILS__{total_wrong}_confident_wrong")
    print(f"VERDICT: {verdict}")
    print("\nNote: 'declined' is an acceptable outcome and 'WRONG' is not. A resolver that says")
    print("'arbitrary' has reported its resolution honestly; one that says 'context' and is")
    print("wrong has asserted a provenance it mis-derived.")

    out = {"mine_easy": t_easy, "mine_hard": t_hard, "redteam": t_red,
           "confident_wrong_total": total_wrong, "verdict": verdict}
    (HERE / "combined_resolver_battery.json").write_text(json.dumps(out, indent=2) + "\n",
                                                         encoding="utf-8")
    print("\nwrote combined_resolver_battery.json")
    return 0 if total_wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
