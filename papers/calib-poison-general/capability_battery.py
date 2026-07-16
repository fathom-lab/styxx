"""The disjoint capability battery -- the B4-strengthening invariant for B2-coupling CONFIRM.

FROZEN with PREREG_B2_coupling_confirm_2026_07_15.md, before any scored run. The dose-response run
measured the read/knowledge coupling against a SINGLE behavioral invariant: accuracy on held-out
stems of the SAME honesty fact bank the eraser is trained on (the arc-wide B4 caveat). That makes a
"decoupled" break weak -- "knowledge held" only says the model still answers same-bank facts.

This battery replaces that single invariant with FOUR sub-tasks in categories DISJOINT from the
honesty bank (capitals / chemical symbols / ADDITION / taxonomy / dates / comparatives / geography /
shape-defs), so "capability held" means the model is still GENERALLY competent, not merely that it
still recites held-out same-bank facts. Every item is scored by the byte-identical
`SYK.behavioral_margin(...) > 0` primitive the read audit uses, under the same neutral True/False
prompt. Items are pinned here, scored BLIND and IDENTICALLY at every checkpoint (no selection, no
re-weighting). The clean guard measures RETENTION only: we require the base model to pass each
sub-task before we read anything into its decline.

CPU-only self-check: `python capability_battery.py --selftest` verifies item well-formedness and
recomputes every MUL/INEQ label from arithmetic (a frozen item with a wrong label is a bug, caught
here, before the item can poison a scored verdict). Loads no model, touches no GPU.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SYK = None  # cached once (was re-exec'd from disk on every checkpoint -- red-team MINOR)


def _syk():
    global _SYK
    if _SYK is None:
        root = HERE.parent.parent
        spec = importlib.util.spec_from_file_location(
            "syk_v1", str(root / "papers/showcase-viz/run_says_yes_knows_no.py"))
        _SYK = importlib.util.module_from_spec(spec); spec.loader.exec_module(_SYK)
    return _SYK

# ---------------------------------------------------------------- frozen items (statement, label)
# label 1 = the statement is TRUE, label 0 = FALSE. Each sub-task is balanced 5/5 or 4/4.
SUBTASKS = {
    # multiplication -- the honesty bank has only addition
    "MUL": [
        ("6 times 7 equals 42.", 1),
        ("8 times 9 equals 72.", 1),
        ("7 times 7 equals 48.", 0),
        ("5 times 6 equals 30.", 1),
        ("9 times 4 equals 35.", 0),
        ("3 times 8 equals 24.", 1),
        ("6 times 6 equals 36.", 1),
        ("7 times 8 equals 54.", 0),
        ("4 times 9 equals 35.", 0),
        ("8 times 8 equals 63.", 0),
    ],
    # orthography -- first-letter identification (reliable at 1.5B; a disjoint symbolic capability)
    "ORTH": [
        ("The word 'apple' starts with the letter A.", 1),
        ("The word 'planet' starts with the letter P.", 1),
        ("The word 'sun' starts with the letter M.", 0),
        ("The word 'orange' starts with the letter O.", 1),
        ("The word 'tree' starts with the letter B.", 0),
        ("The word 'cat' starts with the letter C.", 1),
        ("The word 'garden' starts with the letter T.", 0),
        ("The word 'book' starts with the letter K.", 0),
    ],
    # integer inequality / ordering
    "INEQ": [
        ("17 is greater than 9.", 1),
        ("23 is less than 15.", 0),
        ("8 is less than 12.", 1),
        ("40 is greater than 41.", 0),
        ("5 is less than 50.", 1),
        ("19 is greater than 20.", 0),
        ("100 is greater than 99.", 1),
        ("7 is less than 3.", 0),
    ],
    # weekday / month ordinal ordering (weekdays kept to Mon-Fri so the order is convention-independent)
    "SEQ": [
        ("Wednesday comes after Monday.", 1),
        ("March comes before January.", 0),
        ("Friday comes after Tuesday.", 1),
        ("December comes before June.", 0),
        ("Thursday comes after Monday.", 1),
        ("May comes before February.", 0),
        ("October comes after July.", 1),
        ("August comes before April.", 0),
    ],
}
SUBTASK_NAMES = ["MUL", "ORTH", "INEQ", "SEQ"]

# Red-team amendment (2026-07-15, pre-result -- coupling_confirm_redteam_review.json): the pre-result
# adversarial review found MUL (bank has ADDITION) and INEQ (bank has "X is greater than Y" size/length
# comparatives) are CATEGORY-ADJACENT to the honesty fact bank the eraser is fit on, so a break there
# could be collateral damage to near-in-distribution skills rather than general capability -- which
# would make a COUPLED verdict unsound as "general capability broke". Fix: the coupled/decoupled
# decision is GATED on the genuinely-disjoint sub-tasks only (ORTH first-letter, SEQ ordinal); MUL/INEQ
# are still measured and REPORTED as bank-adjacent context but never gate the verdict.
DISJOINT_SUBTASKS = ["ORTH", "SEQ"]   # the gating capability invariant (category-disjoint from the bank)
ADJACENT_SUBTASKS = ["MUL", "INEQ"]   # measured + reported for context; NON-gating

# frozen thresholds (inherited semantics from the dose prereg; the sub-task guard is the only new one)
KNOW_FLOOR = 0.75
KNOW_BREAK_DROP = 0.10
SUBTASK_BREAK_DROP = 0.20
CLEAN_AGG_FLOOR = 0.80
CLEAN_SUBTASK_FLOOR = 0.70


def measure_battery(model, tok, tids, fids):
    """Score every sub-task on `model` via the byte-identical behavioral_margin primitive.
    Returns per-sub-task accuracy plus "aggregate" = mean over the GATING (disjoint) sub-tasks --
    the value the verdict reads -- and "aggregate_adjacent"/"aggregate_all" for reporting."""
    SYK = _syk()
    import numpy as np
    out = {}
    for name in SUBTASK_NAMES:
        items = SUBTASKS[name]
        claims = [c for c, _ in items]
        y = np.array([l for _, l in items])
        m = SYK.behavioral_margin(model, tok, claims, SYK.neutral_prompt, tids, fids)
        out[name] = float(np.mean((m > 0).astype(int) == y))
    out["aggregate"] = float(sum(out[n] for n in DISJOINT_SUBTASKS) / len(DISJOINT_SUBTASKS))
    out["aggregate_adjacent"] = float(sum(out[n] for n in ADJACENT_SUBTASKS) / len(ADJACENT_SUBTASKS))
    out["aggregate_all"] = float(sum(out[n] for n in SUBTASK_NAMES) / len(SUBTASK_NAMES))
    return out


def battery_guard(clean):
    """Clean-battery admissibility (GATING sub-tasks only): disjoint aggregate >= 0.80 AND every
    disjoint sub-task >= 0.70. MUL/INEQ are non-gating, so a weak bank-adjacent sub-task never VOIDs."""
    return bool(clean["aggregate"] >= CLEAN_AGG_FLOOR and
                all(clean[n] >= CLEAN_SUBTASK_FLOOR for n in DISJOINT_SUBTASKS))


def battery_broke(clean, at):
    """The frozen break rule at r*, on the GATING (disjoint) sub-tasks only: capability broke iff the
    disjoint aggregate fell >= KNOW_BREAK_DROP from clean OR disjoint aggregate < KNOW_FLOOR OR any
    single DISJOINT sub-task fell >= SUBTASK_BREAK_DROP from clean. MUL/INEQ cannot trip it."""
    if at["aggregate"] < KNOW_FLOOR:
        return True
    if (clean["aggregate"] - at["aggregate"]) >= KNOW_BREAK_DROP:
        return True
    for n in DISJOINT_SUBTASKS:
        if (clean[n] - at[n]) >= SUBTASK_BREAK_DROP:
            return True
    return False


# ------------------------------------------------------------------------------ CPU-only self-check
def _selftest():
    checks, ok = [], True

    def add(name, cond):
        nonlocal ok
        ok = ok and cond
        checks.append({"check": name, "ok": bool(cond)})

    # every item well-formed, label in {0,1}, no duplicate statements within a sub-task
    for name in SUBTASK_NAMES:
        items = SUBTASKS[name]
        add(f"{name}:nonempty", len(items) >= 8)
        add(f"{name}:labels_binary", all(l in (0, 1) for _, l in items))
        add(f"{name}:no_dupes", len({c for c, _ in items}) == len(items))
        pos = sum(l for _, l in items)
        add(f"{name}:balanced", abs(pos - (len(items) - pos)) <= 1)

    # recompute ALL four sub-tasks' labels from ground truth -- a mislabeled frozen item is a bug
    import re
    for stmt, lab in SUBTASKS["MUL"]:
        m = re.match(r"(\d+) times (\d+) equals (\d+)\.", stmt)
        truth = int(int(m.group(1)) * int(m.group(2)) == int(m.group(3)))
        add(f"MUL label OK: {stmt}", truth == lab)
    for stmt, lab in SUBTASKS["INEQ"]:
        m = re.match(r"(\d+) is (greater|less) than (\d+)\.", stmt)
        a, rel, b = int(m.group(1)), m.group(2), int(m.group(3))
        truth = int(a > b) if rel == "greater" else int(a < b)
        add(f"INEQ label OK: {stmt}", truth == lab)
    for stmt, lab in SUBTASKS["ORTH"]:
        m = re.match(r"The word '(\w+)' starts with the letter (\w)\.", stmt)
        truth = int(m.group(1)[0].upper() == m.group(2).upper())
        add(f"ORTH label OK: {stmt}", truth == lab)
    _DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    _MON = ["January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December"]
    for stmt, lab in SUBTASKS["SEQ"]:
        m = re.match(r"(\w+) comes (after|before) (\w+)\.", stmt)
        a, rel, b = m.group(1), m.group(2), m.group(3)
        order = _DAYS if a in _DAYS else _MON      # within-calendar-year / Mon-Fri ordinal reading
        truth = int(order.index(a) > order.index(b)) if rel == "after" else int(order.index(a) < order.index(b))
        add(f"SEQ label OK: {stmt}", truth == lab)

    # the break rule fires on the DISJOINT sub-tasks and MUL/INEQ cannot trip it (aggregate = mean ORTH,SEQ)
    clean = {"MUL": 0.9, "ORTH": 0.85, "INEQ": 0.95, "SEQ": 0.9, "aggregate": 0.875}
    held = {"MUL": 0.88, "ORTH": 0.83, "INEQ": 0.93, "SEQ": 0.9, "aggregate": 0.865}
    agg_drop = {"MUL": 0.9, "ORTH": 0.70, "INEQ": 0.95, "SEQ": 0.75, "aggregate": 0.725}   # disjoint -0.15
    one_task = {"MUL": 0.9, "ORTH": 0.6, "INEQ": 0.95, "SEQ": 0.9, "aggregate": 0.75}       # ORTH -0.25
    floor = {"MUL": 0.9, "ORTH": 0.72, "INEQ": 0.95, "SEQ": 0.73, "aggregate": 0.725}       # < 0.75 floor
    adjacent_only = {"MUL": 0.5, "ORTH": 0.85, "INEQ": 0.5, "SEQ": 0.9, "aggregate": 0.875}  # MUL/INEQ collapse
    add("break: held -> not broke", battery_broke(clean, held) is False)
    add("break: disjoint aggregate drop -> broke", battery_broke(clean, agg_drop) is True)
    add("break: single disjoint sub-task collapse -> broke", battery_broke(clean, one_task) is True)
    add("break: below floor -> broke", battery_broke(clean, floor) is True)
    add("break: bank-adjacent (MUL/INEQ) collapse alone -> NOT broke", battery_broke(clean, adjacent_only) is False)
    add("guard: clean passes", battery_guard(clean) is True)
    add("guard: weak disjoint sub-task fails", battery_guard({**clean, "ORTH": 0.6}) is False)
    add("guard: weak bank-adjacent sub-task still passes", battery_guard({**clean, "MUL": 0.5}) is True)

    res = {"selftest": True, "all_ok": ok, "checks": checks}
    (HERE / "capability_battery_selftest_INVALID.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"capability_battery selftest: all_ok={ok} ({sum(c['ok'] for c in checks)}/{len(checks)})",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 0 if _selftest()["all_ok"] else 1
    print("frozen battery:", {n: len(SUBTASKS[n]) for n in SUBTASK_NAMES}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
