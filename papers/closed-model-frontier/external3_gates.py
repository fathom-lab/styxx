"""EXTERNAL-3 gates: BC-1 measured against its preregistration (PREREG_bc1_by_construction_2026_09_16).

Reads the baseline ledger (this checkout at the prereg commit, WITHOUT the repair:
``external3_harness.py gate --tag external3_base``) and the repaired ledger (the same checkout
with BC-1) — same corpus, same reconstruction, same exclusions, one instrument change — and
scores the gates the prereg froze.  The wheel's ledger (EXTERNAL-2) is reported beside them as
context only: main already differs from the wheel in what it reads (V14), and that difference is
not this repair's.

  G-B1  subset invariant: every (pr_id, kind, claim text) that is CONTRADICTED after the repair
        was CONTRADICTED before it.  Zero new accusations, anywhere.
  G-B2  by construction: re-counted by external2_census.py on the repaired ledger; this script
        recomputes the three counters directly so the receipt does not depend on a second run.
  G-B3  the verified side: every (pr_id, kind, text) VERIFIED before, for tests_added,
        symbol_added and only_touches, is VERIFIED after; no VERIFIED tests_added or
        symbol_added in a diff without Python.
  G-B5  the survivors, by kind, with no precision attached.

Counts only; no PR named.  Writes external3_gates.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "external3_base_ledger.jsonl"
WHEEL = HERE / "external2_summary.json"
AFTER = HERE / "external3_ledger.jsonl"
OUT = HERE / "external3_gates.json"
PY = {".py", ".pyi"}
KINDS = ("tests_added", "symbol_added", "only_touches", "files_changed_count")
STOP = None  # filled from external2_census so the by-construction rule is the census's own


def _load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["pr_id"]] = r
    return out


def _keyed(ledger: dict, verdict: str) -> set:
    return {(pid, c["kind"], c["text"]) for pid, r in ledger.items()
            for c in r["claims"] if c["verdict"] == verdict}


def main() -> int:
    sys.path.insert(0, str(HERE))
    from external2_census import looks_like_path  # the census's own rule, unchanged

    before, after = _load(BEFORE), _load(AFTER)
    assert set(before) == set(after), "the two ledgers cover different PRs"
    acc_b, acc_a = _keyed(before, "CONTRADICTED"), _keyed(after, "CONTRADICTED")
    new_accusations = sorted(acc_a - acc_b)
    removed = acc_b - acc_a
    by_kind_removed = Counter(k for _, k, _ in removed)
    by_kind_before = Counter(k for _, k, _ in acc_b)
    by_kind_after = Counter(k for _, k, _ in acc_a)

    ver_b = {t for t in _keyed(before, "VERIFIED") if t[1] in ("tests_added", "symbol_added", "only_touches")}
    ver_a = _keyed(after, "VERIFIED")
    lost_verified = sorted(ver_b - ver_a)
    verified_without_python = 0
    by_construction = Counter()
    for pid, r in after.items():
        has_py = any(e in PY for e in r["exts"])
        for c in r["claims"]:
            if c["kind"] in ("tests_added", "symbol_added"):
                if c["verdict"] == "CONTRADICTED" and not has_py:
                    by_construction[f"{c['kind']}.contradicted_no_python_in_diff"] += 1
                if c["verdict"] == "VERIFIED" and not has_py:
                    verified_without_python += 1
            if c["kind"] == "only_touches" and c["verdict"] == "CONTRADICTED":
                if not looks_like_path(c["detail"].get("prefix", "")):
                    by_construction["only_touches.contradicted_prefix_is_not_a_path"] += 1
    for k in ("tests_added.contradicted_no_python_in_diff", "symbol_added.contradicted_no_python_in_diff",
              "only_touches.contradicted_prefix_is_not_a_path"):
        by_construction.setdefault(k, 0)

    # what the survivors look like, mechanically (no adjudication)
    surv = Counter()
    for pid, r in after.items():
        for c in r["claims"]:
            if c["verdict"] != "CONTRADICTED":
                continue
            k = c["kind"]
            if k == "tests_added":
                m = re.search(r"diff adds (\d+) test functions, claim says (\d+)", c["why"])
                got, n = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
                surv["tests_added: diff adds 0" if got == 0 else "tests_added: counts differ, both > 0"] += 1
            elif k == "symbol_added":
                surv["symbol_added: name not defined in added lines (Python diff)"] += 1
            elif k == "only_touches":
                surv["only_touches: two prefixes" if c["detail"].get("prefix2") else "only_touches: one path-shaped prefix"] += 1
            elif k == "files_changed_count":
                surv["files_changed_count: git stat line" if re.search(r"insertions?\(\+\)|deletions?\(-\)", c["text"])
                     else "files_changed_count: other"] += 1

    coverage_b = sum(1 for r in before.values() if r["claims"])
    coverage_a = sum(1 for r in after.values() if r["claims"])
    gates = {
        "G-B1_subset_invariant": {"new_accusations": len(new_accusations), "pass": not new_accusations},
        "G-B2_by_construction": {"counters": dict(by_construction),
                                 "pass": all(v == 0 for v in by_construction.values())},
        "G-B3_verified_preserved": {"verified_before": len(ver_b), "lost": len(lost_verified),
                                    "verified_without_python_after": verified_without_python,
                                    "pass": not lost_verified and verified_without_python == 0},
        "G-B5_survivors": {"accusations_before": len(acc_b), "accusations_after": len(acc_a),
                           "removed": len(removed), "by_kind_before": dict(by_kind_before),
                           "by_kind_after": dict(by_kind_after), "by_kind_removed": dict(by_kind_removed),
                           "survivor_shapes": dict(surv), "precision": "not measured; see prereg"},
    }
    payload = {
        "prereg": "PREREG_bc1_by_construction_2026_09_16.md",
        "before": {"ledger": BEFORE.name, "instrument": "this checkout at the prereg commit, without BC-1",
                   "eligible": len(before), "covered": coverage_b},
        "wheel_context": {k: json.loads(WHEEL.read_text(encoding="utf-8"))[k]
                          for k in ("accusations_total", "accusations_by_kind",
                                    "accusations_unsupported_by_construction")},
        "after": {"ledger": AFTER.name, "instrument": "this checkout, BC-1", "eligible": len(after),
                  "covered": coverage_a,
                  "covered_note": "sentences whose symbol_added match captured a word are never-read now, "
                                  "so coverage drops by exactly those sentences"},
        "gates": gates,
        "all_blocking_gates_pass": all(gates[g]["pass"] for g in ("G-B1_subset_invariant", "G-B2_by_construction", "G-B3_verified_preserved")),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0 if payload["all_blocking_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
