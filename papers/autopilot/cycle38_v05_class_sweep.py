"""Per-class battery sweep for OATH v0.5 (PREREG severability procedure).

Runs the mutant battery + the six-doc P1 recert under: ALL classes on, then each class OFF in turn
(leave-one-out), to attribute the caught/false_verify deltas to individual classes. Toggles the
V05_* module flags in styxx.certify -- no code edits. Output: cycle38_v05_class_sweep_result.json.
"""
from __future__ import annotations
import importlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.certify as C
import mutant_battery as MB

CLASSES = ["V05_APPROX_NOTATION", "V05_UNIT_RANGE", "V05_ARXIV_ID", "V05_AT_PARAM",
           "V05_DERIVED_PCT", "V05_SELF_SCOPED_N"]

SIXDOC = [
    ("papers/consensus-truth-engine", "FINDING_truthengine_2026_05_25.md"),
    ("papers/decoupled-diagonal-capstone", "FINDING_2026_05_25.md"),
    ("papers/benchmark-validation", "FINDING_triviaqa_2026_05_25.md"),
    ("papers/knowledge-boundary-calibration", "FINDING_kbc_2026_05_25.md"),
    ("papers/knowledge-boundary-calibration", "FINDING_curve_2026_05_25.md"),
]


def sixdoc_ung():
    total = 0
    for d, dn in SIXDOC:
        d = Path(ROOT) / d
        receipts = [r for r in sorted(d.glob("*.json"))
                    if not r.name.endswith("certificate.json") and not r.name.startswith("cycle38")]
        cert = C.certify_doc(d / dn, receipts)
        total += len(cert["ungrounded"])
    return total


def battery():
    # MB.run() equivalent: call its main path in-process. MB writes a file; re-import to get numbers.
    res = MB.run_battery() if hasattr(MB, "run_battery") else None
    return res


def set_flags(off=None):
    off = off or set()
    for c in CLASSES:
        setattr(C, c, c not in off)


def main():
    # sanity: battery must expose an in-process entry; if not, fall back to reading its writer
    if not hasattr(MB, "run_battery"):
        print("mutant_battery has no run_battery(); using main() with a temp out per config")
    configs = [("ALL", set())] + [(f"drop_{c}", {c}) for c in CLASSES]
    rows = []
    for name, off in configs:
        set_flags(off)
        out = HERE / f"_sweep_{name}.json"
        MB.main_argv = ["--out", str(out)]
        # MB.main() reads sys.argv; drive it directly
        sys.argv = ["mutant_battery.py", "--out", str(out)]
        MB.main()
        r = json.loads(out.read_text(encoding="utf-8"))
        ung6 = sixdoc_ung()
        rows.append({"config": name, "caught": r["caught"], "false_verify": r["false_verify"],
                     "n_mutants": r["n_mutants"], "clean_ung13": r["clean_ungrounded_total"],
                     "sixdoc_ung": ung6})
        print(f"{name:22} caught={r['caught']} fv={r['false_verify']} mut={r['n_mutants']} "
              f"clean_ung13={r['clean_ungrounded_total']} sixdoc_ung={ung6}", flush=True)
        out.unlink(missing_ok=True)
    set_flags(set())
    base = rows[0]
    deltas = []
    for row in rows[1:]:
        deltas.append({"class": row["config"].replace("drop_", ""),
                       "d_caught_from_dropping": row["caught"] - base["caught"],
                       "d_fv_from_dropping": row["false_verify"] - base["false_verify"],
                       "d_sixdoc_ung_from_dropping": row["sixdoc_ung"] - base["sixdoc_ung"]})
    (HERE / "cycle38_v05_class_sweep_result.json").write_text(
        json.dumps({"all_on": base, "leave_one_out": rows[1:], "drop_deltas": deltas,
                    "bars": {"caught_min": 116, "false_verify_max": 26, "sixdoc_ung_max": 4}},
                   indent=2) + "\n", encoding="utf-8")
    print("\nDROP DELTAS (effect of removing each class from the full composition):")
    for d in deltas:
        print(f"  drop {d['class']:22} caught {d['d_caught_from_dropping']:+d}  "
              f"fv {d['d_fv_from_dropping']:+d}  sixdoc_ung {d['d_sixdoc_ung_from_dropping']:+d}")


if __name__ == "__main__":
    main()
