"""The disjoint capability battery -- the B4-strengthening invariant for B2-coupling CONFIRM,
CALIBRATED to the base model (a base-only, treatment-blind, frozen procedure).

Story: the dose-response run measured read/knowledge coupling against ONE behavioral invariant on
held-out stems of the SAME honesty fact bank the eraser trains on (the arc-wide B4 caveat). This
battery replaces it with capability OUTSIDE that bank. A pre-result adversarial review then found two
things that this file now answers:

  (F2) MUL (bank has ADDITION) and INEQ (bank has "X is greater than Y" comparatives) are
       CATEGORY-ADJACENT to the honesty bank -- a break there could be collateral damage to
       near-in-distribution skills, not general capability. Fix: MUL/INEQ are measured and REPORTED
       (aggregate_adjacent) but NEVER gate the verdict.

  (smoke) The base Qwen2.5-1.5B scores 0.5 (chance) on SEQ (weekday/month ordinal ordering) via the
       True/False readout while ORTH first-letter is 1.0 -- so a fixed disjoint pair is hostage to
       one sub-task the base cannot do. Fix: the gating disjoint battery is SELECTED from a frozen
       CANDIDATE POOL of disjoint symbolic capabilities by keeping only the sub-tasks the CLEAN base
       model clears at DISJOINT_FLOOR_CLEAN (0.90), requiring at least MIN_DISJOINT (3) survivors.
       Selection is measured on the CLEAN base model ONLY, before any treatment -- it is the clean
       guard's own logic ("measure retention of capability the base HAS"), not a fit to results.

Every item is scored by the byte-identical `SYK.behavioral_margin(...) > 0` primitive the read audit
and eval_knowledge use, under the same neutral True/False prompt. Items are pinned here; scored BLIND
and IDENTICALLY at every checkpoint. `python capability_battery.py --selftest` recomputes EVERY
sub-task's labels from ground truth (arithmetic, first/last letter, containment, vowel set, case map,
alphabetical order, antonym pairs, ordinal order) and exercises selection/guard/break -- CPU-only.
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
# label 1 = TRUE, 0 = FALSE. Each sub-task is balanced. Categories are DISJOINT from the honesty fact
# bank (capitals / chemical symbols / ADDITION / taxonomy / historical dates / size-length comparatives
# / continent-language geography / shape-side definitions), EXCEPT the two BANK_ADJACENT ones.

BANK_ADJACENT = {
    # measured + REPORTED for context; NEVER gate the verdict (category-adjacent to the bank)
    "MUL": [
        ("6 times 7 equals 42.", 1), ("8 times 9 equals 72.", 1), ("7 times 7 equals 48.", 0),
        ("5 times 6 equals 30.", 1), ("9 times 4 equals 35.", 0), ("3 times 8 equals 24.", 1),
        ("6 times 6 equals 36.", 1), ("7 times 8 equals 54.", 0), ("4 times 9 equals 35.", 0),
        ("8 times 8 equals 63.", 0),
    ],
    "INEQ": [
        ("17 is greater than 9.", 1), ("23 is less than 15.", 0), ("8 is less than 12.", 1),
        ("40 is greater than 41.", 0), ("5 is less than 50.", 1), ("19 is greater than 20.", 0),
        ("100 is greater than 99.", 1), ("7 is less than 3.", 0),
    ],
}

# The candidate pool for the GATING disjoint battery. All are symbolic/orthographic/lexical
# competences categorically disjoint from the honesty semantic bank. The base-model calibration keeps
# the ones the clean model actually has (>= DISJOINT_FLOOR_CLEAN); the rest are excluded with a
# receipt (e.g. SEQ, which the 1.5B base does at chance, self-documents its own exclusion).
DISJOINT_POOL = {
    "ORTH_FIRST": [   # first-letter identification (base = 1.0 at 1.5B)
        ("The word 'apple' starts with the letter A.", 1), ("The word 'planet' starts with the letter P.", 1),
        ("The word 'sun' starts with the letter M.", 0), ("The word 'orange' starts with the letter O.", 1),
        ("The word 'tree' starts with the letter B.", 0), ("The word 'cat' starts with the letter C.", 1),
        ("The word 'garden' starts with the letter T.", 0), ("The word 'book' starts with the letter K.", 0),
    ],
    "ORTH_LAST": [    # last-letter identification
        ("The word 'table' ends with the letter E.", 1), ("The word 'planet' ends with the letter T.", 1),
        ("The word 'apple' ends with the letter Y.", 0), ("The word 'garden' ends with the letter N.", 1),
        ("The word 'book' ends with the letter M.", 0), ("The word 'river' ends with the letter V.", 0),
        ("The word 'orange' ends with the letter G.", 0), ("The word 'window' ends with the letter W.", 1),
    ],
    "ORTH_CONTAINS": [
        ("The word 'garden' contains the letter D.", 1), ("The word 'apple' contains the letter P.", 1),
        ("The word 'sun' contains the letter Z.", 0), ("The word 'orange' contains the letter G.", 1),
        ("The word 'table' contains the letter X.", 0), ("The word 'river' contains the letter V.", 1),
        ("The word 'book' contains the letter M.", 0), ("The word 'planet' contains the letter Q.", 0),
    ],
    "VOWEL": [
        ("The letter A is a vowel.", 1), ("The letter E is a vowel.", 1), ("The letter K is a vowel.", 0),
        ("The letter O is a vowel.", 1), ("The letter T is a vowel.", 0), ("The letter U is a vowel.", 1),
        ("The letter B is a vowel.", 0), ("The letter M is a vowel.", 0),
    ],
    "CASE": [
        ("The uppercase form of 'a' is 'A'.", 1), ("The uppercase form of 'b' is 'B'.", 1),
        ("The uppercase form of 'c' is 'D'.", 0), ("The uppercase form of 'e' is 'E'.", 1),
        ("The uppercase form of 'f' is 'G'.", 0), ("The uppercase form of 'h' is 'H'.", 1),
        ("The uppercase form of 'm' is 'N'.", 0), ("The uppercase form of 'p' is 'Q'.", 0),
    ],
    "ALPHA_AFTER": [
        ("In the alphabet, the letter D comes after the letter B.", 1),
        ("In the alphabet, the letter Z comes after the letter Y.", 1),
        ("In the alphabet, the letter C comes after the letter F.", 0),
        ("In the alphabet, the letter M comes after the letter K.", 1),
        ("In the alphabet, the letter B comes after the letter T.", 0),
        ("In the alphabet, the letter P comes after the letter L.", 1),
        ("In the alphabet, the letter A comes after the letter E.", 0),
        ("In the alphabet, the letter G comes after the letter R.", 0),
    ],
    "ANTONYM": [
        ("The opposite of 'hot' is 'cold'.", 1), ("The opposite of 'up' is 'down'.", 1),
        ("The opposite of 'big' is 'red'.", 0), ("The opposite of 'fast' is 'slow'.", 1),
        ("The opposite of 'happy' is 'tall'.", 0), ("The opposite of 'open' is 'closed'.", 1),
        ("The opposite of 'light' is 'round'.", 0), ("The opposite of 'wet' is 'green'.", 0),
    ],
    "SEQ": [          # weekday(Mon-Fri)/month ordinal ordering -- kept for transparency; base=0.5 -> excluded
        ("Wednesday comes after Monday.", 1), ("March comes before January.", 0),
        ("Friday comes after Tuesday.", 1), ("December comes before June.", 0),
        ("Thursday comes after Monday.", 1), ("May comes before February.", 0),
        ("October comes after July.", 1), ("August comes before April.", 0),
    ],
}
ALL_SUBTASKS = list(BANK_ADJACENT.keys()) + list(DISJOINT_POOL.keys())

# frozen thresholds
KNOW_FLOOR = 0.75             # disjoint aggregate admissibility at a checkpoint
KNOW_BREAK_DROP = 0.10        # aggregate "broke" if it fell this far from clean
SUBTASK_BREAK_DROP = 0.20     # or any single gating sub-task fell this far (only makes a break HARDER)
CLEAN_AGG_FLOOR = 0.80        # clean disjoint aggregate must clear this
CLEAN_SUBTASK_FLOOR = 0.70    # each SELECTED disjoint sub-task must clear this (selection uses a higher bar)
DISJOINT_FLOOR_CLEAN = 0.90   # a candidate is SELECTED iff the clean base model scores >= this
MIN_DISJOINT = 3              # need at least this many survivors, else the battery is uncalibratable


def measure_all(model, tok, tids, fids):
    """Score EVERY sub-task (bank-adjacent + full disjoint pool) on `model`. Returns {name: accuracy}."""
    SYK = _syk()
    import numpy as np
    out = {}
    for name, items in list(BANK_ADJACENT.items()) + list(DISJOINT_POOL.items()):
        claims = [c for c, _ in items]
        y = np.array([l for _, l in items])
        m = SYK.behavioral_margin(model, tok, claims, SYK.neutral_prompt, tids, fids)
        out[name] = float(np.mean((m > 0).astype(int) == y))
    return out


def select_disjoint(clean_scores, floor=DISJOINT_FLOOR_CLEAN, need=MIN_DISJOINT):
    """Base-only, treatment-blind selection: keep every DISJOINT_POOL sub-task the CLEAN model clears
    at `floor`, sorted by name (deterministic). Returns (survivors, ok) where ok = len>=need."""
    survivors = sorted([n for n in DISJOINT_POOL if clean_scores.get(n, 0.0) >= floor])
    return survivors, bool(len(survivors) >= need)


def aggregate(scores, selected):
    return float(sum(scores[n] for n in selected) / len(selected)) if selected else 0.0


def battery_guard(clean_scores, selected):
    """Clean admissibility on the SELECTED disjoint sub-tasks: aggregate >= 0.80 AND each >= 0.70.
    (Selection already guarantees each >= 0.90, so this holds by construction when selection succeeds;
    kept as the frozen gate.)"""
    if not selected:
        return False
    return bool(aggregate(clean_scores, selected) >= CLEAN_AGG_FLOOR and
                all(clean_scores[n] >= CLEAN_SUBTASK_FLOOR for n in selected))


def battery_broke(clean_scores, at_scores, selected):
    """Break rule at r*, on the SELECTED disjoint sub-tasks only: broke iff the disjoint aggregate
    fell >= KNOW_BREAK_DROP from clean OR aggregate < KNOW_FLOOR OR any selected sub-task fell
    >= SUBTASK_BREAK_DROP. Bank-adjacent MUL/INEQ can never trip it."""
    ca, aa = aggregate(clean_scores, selected), aggregate(at_scores, selected)
    if aa < KNOW_FLOOR:
        return True
    if (ca - aa) >= KNOW_BREAK_DROP:
        return True
    for n in selected:
        if (clean_scores[n] - at_scores[n]) >= SUBTASK_BREAK_DROP:
            return True
    return False


# ------------------------------------------------------------------------------ CPU-only self-check
def _selftest():
    import re
    checks, ok = [], True

    def add(name, cond):
        nonlocal ok
        ok = ok and bool(cond); checks.append({"check": name, "ok": bool(cond)})

    for name, items in list(BANK_ADJACENT.items()) + list(DISJOINT_POOL.items()):
        add(f"{name}:>=8", len(items) >= 8)
        add(f"{name}:labels_binary", all(l in (0, 1) for _, l in items))
        add(f"{name}:no_dupes", len({c for c, _ in items}) == len(items))
        pos = sum(l for _, l in items)
        add(f"{name}:balanced", abs(pos - (len(items) - pos)) <= 1)

    # recompute EVERY sub-task's labels from ground truth
    for s, l in BANK_ADJACENT["MUL"]:
        m = re.match(r"(\d+) times (\d+) equals (\d+)\.", s)
        add(f"MUL:{s}", int(int(m[1]) * int(m[2]) == int(m[3])) == l)
    for s, l in BANK_ADJACENT["INEQ"]:
        m = re.match(r"(\d+) is (greater|less) than (\d+)\.", s)
        add(f"INEQ:{s}", (int(int(m[1]) > int(m[3])) if m[2] == "greater" else int(int(m[1]) < int(m[3]))) == l)
    for s, l in DISJOINT_POOL["ORTH_FIRST"]:
        m = re.match(r"The word '(\w+)' starts with the letter (\w)\.", s)
        add(f"ORTH_FIRST:{s}", int(m[1][0].upper() == m[2].upper()) == l)
    for s, l in DISJOINT_POOL["ORTH_LAST"]:
        m = re.match(r"The word '(\w+)' ends with the letter (\w)\.", s)
        add(f"ORTH_LAST:{s}", int(m[1][-1].upper() == m[2].upper()) == l)
    for s, l in DISJOINT_POOL["ORTH_CONTAINS"]:
        m = re.match(r"The word '(\w+)' contains the letter (\w)\.", s)
        add(f"ORTH_CONTAINS:{s}", int(m[2].upper() in m[1].upper()) == l)
    _VOWELS = set("AEIOU")
    for s, l in DISJOINT_POOL["VOWEL"]:
        m = re.match(r"The letter (\w) is a vowel\.", s)
        add(f"VOWEL:{s}", int(m[1].upper() in _VOWELS) == l)
    for s, l in DISJOINT_POOL["CASE"]:
        m = re.match(r"The uppercase form of '(\w)' is '(\w)'\.", s)
        add(f"CASE:{s}", int(m[1].upper() == m[2]) == l)
    for s, l in DISJOINT_POOL["ALPHA_AFTER"]:
        m = re.match(r"In the alphabet, the letter (\w) comes after the letter (\w)\.", s)
        add(f"ALPHA_AFTER:{s}", int(m[1].upper() > m[2].upper()) == l)
    _ANT = {"hot": "cold", "up": "down", "fast": "slow", "open": "closed",
            "big": "small", "happy": "sad", "light": "dark", "wet": "dry"}
    for s, l in DISJOINT_POOL["ANTONYM"]:
        m = re.match(r"The opposite of '(\w+)' is '(\w+)'\.", s)
        add(f"ANTONYM:{s}", int(_ANT.get(m[1]) == m[2]) == l)
    _DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    _MON = ["January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December"]
    for s, l in DISJOINT_POOL["SEQ"]:
        m = re.match(r"(\w+) comes (after|before) (\w+)\.", s)
        order = _DAYS if m[1] in _DAYS else _MON
        add(f"SEQ:{s}", (int(order.index(m[1]) > order.index(m[3])) if m[2] == "after"
                         else int(order.index(m[1]) < order.index(m[3]))) == l)

    # selection + guard + break logic
    base = {"ORTH_FIRST": 1.0, "ORTH_LAST": 0.95, "ORTH_CONTAINS": 0.9, "VOWEL": 1.0, "CASE": 0.95,
            "ALPHA_AFTER": 0.85, "ANTONYM": 0.9, "SEQ": 0.5, "MUL": 1.0, "INEQ": 0.875}
    sel, okk = select_disjoint(base)
    add("select: keeps >=0.90 disjoint", sel == sorted(["ORTH_FIRST", "ORTH_LAST", "ORTH_CONTAINS", "VOWEL", "CASE", "ANTONYM"]))
    add("select: excludes SEQ(0.5) and ALPHA_AFTER(0.85)", "SEQ" not in sel and "ALPHA_AFTER" not in sel)
    add("select: excludes bank-adjacent from disjoint set", "MUL" not in sel and "INEQ" not in sel)
    add("select: ok with >=3", okk is True)
    add("select: too few survivors -> not ok", select_disjoint({"ORTH_FIRST": 1.0, "VOWEL": 0.6, "SEQ": 0.5})[1] is False)
    add("guard: clean selected passes", battery_guard(base, sel) is True)
    clean_s = {n: base[n] for n in sel}
    held = {n: base[n] - 0.02 for n in sel}
    broke_agg = {n: base[n] - 0.15 for n in sel}
    broke_one = {**clean_s, sel[0]: clean_s[sel[0]] - 0.25}
    adj = {**clean_s, "MUL": 0.4, "INEQ": 0.4}       # bank-adjacent collapse
    add("break: held -> not broke", battery_broke(clean_s, held, sel) is False)
    add("break: aggregate drop -> broke", battery_broke(clean_s, broke_agg, sel) is True)
    add("break: one selected sub-task collapse -> broke", battery_broke(clean_s, broke_one, sel) is True)
    add("break: bank-adjacent collapse alone -> NOT broke", battery_broke(clean_s, adj, sel) is False)

    res = {"selftest": True, "all_ok": ok, "n": len(checks), "n_ok": sum(c["ok"] for c in checks)}
    (HERE / "capability_battery_selftest_INVALID.json").write_text(
        json.dumps({**res, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(f"capability_battery selftest: all_ok={ok} ({res['n_ok']}/{res['n']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 0 if _selftest()["all_ok"] else 1
    print("bank-adjacent:", list(BANK_ADJACENT), "| disjoint pool:", list(DISJOINT_POOL), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
