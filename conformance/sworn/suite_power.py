"""Measure what the test suite DEFENDS: mutate a layer, run the oracle, record kill or survival.

Spec: papers/sworn/SPEC_suite_power_v01_2026_09_06.md, frozen before this file ran.

The differential harness compares `styxx/sworn.py` against a second implementation and cannot reach
the tree handles, the sidecar layer, the receipt layer or `_coverage` — the JavaScript side has no
repository, and those layers sit outside the compared verdict core. Two real defects were recently
found hiding in a comparable blind spot. The only instrument left for these layers is the test
suite, so this asks the only question that matters to somebody about to change them: **if I broke
this line, would anything fail?**

A survivor is not a bug. It is a place where a bug would ship silently.

Usage:
    python conformance/sworn/suite_power.py --catalogue <in.json> --out <receipt.json>

Nothing is left behind: the file is restored from the bytes read at start, in a finally block, and
its digest is checked after every mutant. A run that cannot restore the file stops rather than
continuing over a corrupted tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGET = ROOT / "styxx" / "sworn.py"

# S1: the oracle is named here, before any catalogue is read, and it goes in the receipt.
ORACLE = [
    "tests/test_sworn.py",
    "tests/test_sworn_attacks.py",
    "tests/test_sworn_dogfood.py",
    "tests/test_sworn_eol.py",
    "tests/test_capsule_sworn.py",
]

_FAILED = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)
_TAIL = re.compile(r"^(\d+) failed[^\n]*", re.M)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def run_oracle(timeout=1800):
    """(passed: bool, failing_test_ids, collected_ok: bool, seconds, tail)."""
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *ORACLE, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    secs = round(time.time() - t0, 1)
    # A collection error is not a defence: the suite never got to assert anything.
    collected = "error" not in out.lower().split("\n")[-3:][0] if out else False
    collected = not re.search(r"errors? during collection|INTERNALERROR", out)
    ids = sorted(set(_FAILED.findall(out)))
    return r.returncode == 0, ids, collected, secs, out[-600:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out", default=str(HERE / "suite_power.json"))
    a = ap.parse_args(argv)

    out_path = Path(a.out).resolve()
    if out_path.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(out_path)],
                           capture_output=True)
        if r.returncode == 0:
            print("REFUSED: %s is tracked; a run is history — write a new file (S6)" % out_path.name,
                  file=sys.stderr)
            return 2

    doc = json.loads(Path(a.catalogue).read_text(encoding="utf-8"))
    catalogue = doc["mutations"] if isinstance(doc, dict) else doc

    original = TARGET.read_bytes()
    original_sha = _sha(original)
    text = original.decode("utf-8")
    crlf = b"\r\n" in original

    # G-B: a run over a red baseline measures nothing.
    print("baseline: running the oracle unmutated ...", flush=True)
    ok, ids, collected, secs, tail = run_oracle()
    print("baseline: %s in %ss" % ("PASSES" if ok else "FAILS", secs), flush=True)
    if not ok:
        print("REFUSED (G-B): the oracle does not pass unmutated; every 'kill' below would be "
              "meaningless.\n%s" % tail, file=sys.stderr)
        return 2

    results = []
    try:
        for i, m in enumerate(catalogue, 1):
            rec = dict(m)
            hits = text.count(m["old"])
            rec["anchor_occurrences"] = hits
            if hits == 0:
                rec["verdict"] = "anchor_missing"
            elif m["old"] == m["new"]:
                rec["verdict"] = "no_op"
            else:
                mutant = text.replace(m["old"], m["new"], 1)
                TARGET.write_bytes(
                    (mutant.replace("\n", "\r\n") if crlf else mutant).encode("utf-8"))
                ok, ids, collected, secs, tail = run_oracle()
                rec["seconds"] = secs
                if not collected:
                    rec["verdict"] = "non_viable"
                    rec["detail"] = "the oracle did not collect: %s" % tail[-200:]
                else:
                    rec["verdict"] = "survived" if ok else "killed"
                    rec["killed_by"] = ids
                    rec["killed_by_count"] = len(ids)
                TARGET.write_bytes(original)
                assert _sha(TARGET.read_bytes()) == original_sha, "the file was not restored"
            results.append(rec)
            print("  [%3d/%3d] %-13s %-9s %-46s %s"
                  % (i, len(catalogue), rec["verdict"], rec.get("layer", "?"),
                     rec.get("name", "")[:46],
                     ("%d tests" % rec["killed_by_count"]) if rec.get("killed_by_count")
                     else rec.get("detail", "")[:40]), flush=True)
    finally:
        TARGET.write_bytes(original)
        if _sha(TARGET.read_bytes()) != original_sha:
            print("FATAL: styxx/sworn.py was not restored — check `git status` before continuing",
                  file=sys.stderr)
            return 3

    verdicts = Counter(r["verdict"] for r in results)
    controls = [r for r in results if r.get("control")]
    mutants = [r for r in results if not r.get("control")]
    viable = [r for r in mutants if r["verdict"] in ("killed", "survived")]
    killed = [r for r in viable if r["verdict"] == "killed"]
    survived = [r for r in viable if r["verdict"] == "survived"]
    ctl_viable = [r for r in controls if r["verdict"] in ("killed", "survived")]
    ctl_killed = [r for r in ctl_viable if r["verdict"] == "killed"]

    by_layer = {}
    for r in viable:
        d = by_layer.setdefault(r.get("layer", "?"), {"viable": 0, "killed": 0, "survived": 0})
        d["viable"] += 1
        d["killed" if r["verdict"] == "killed" else "survived"] += 1

    # S5: which tests do the defending, and how concentrated is it?
    by_test = Counter()
    for r in killed:
        for t in r.get("killed_by", []):
            by_test[t.split("::")[0]] += 1

    gates = {
        "G-N": {"quantity": "viable mutants measured", "value": len(viable), "bar": ">= 25",
                "pass": len(viable) >= 25},
        "G-K": {"quantity": "controls killed", "value": len(ctl_killed),
                "controls_viable": len(ctl_viable), "bar": "== 0", "pass": len(ctl_killed) == 0,
                "note": "a killed control VOIDS the run: the suite would be detecting editing"},
        "G-L": {"quantity": "layers with at least one viable mutant", "value": len(by_layer),
                "layers": sorted(by_layer), "bar": ">= 3", "pass": len(by_layer) >= 3},
        "G-B": {"quantity": "the unmutated oracle passes", "value": True, "bar": "required",
                "pass": True},
        "G-S": {"quantity": "kill rate, killed / viable",
                "value": {"killed": len(killed), "viable": len(viable),
                          "rate": (round(len(killed) / len(viable), 4) if viable else None)},
                "bar": "none — reported, never passed or failed (S4)", "pass": None},
    }
    void = not gates["G-K"]["pass"]

    receipt = {
        "schema": "styxx.sworn.suite-power/v1",
        "spec": "papers/sworn/SPEC_suite_power_v01_2026_09_06.md",
        "void": void,
        "oracle": {"tests": ORACLE,
                   "note": ("a survivor survived THIS oracle; it may still be killed elsewhere in "
                            "the repository (S1)")},
        "mutated": {"path": "styxx/sworn.py", "sha256": original_sha},
        "counts": {
            "proposed": len(catalogue), "controls": len(controls),
            "controls_killed": len(ctl_killed), "viable": len(viable),
            "killed": len(killed), "survived": len(survived),
            "anchor_missing": verdicts.get("anchor_missing", 0),
            "non_viable": verdicts.get("non_viable", 0), "no_op": verdicts.get("no_op", 0),
        },
        "gates": gates,
        "by_layer": by_layer,
        "killing_test_files": dict(by_test.most_common()),
        # S4: the survivors ARE the result.
        "survived": [{"name": r["name"], "layer": r.get("layer"), "old": r["old"],
                      "new": r["new"], "why": r.get("why")} for r in survived],
        "mutations": results,
        "reading": ("killed means a change there would be noticed by the named oracle; survived "
                    "means it would not, and a defect in that place could ship with every test in "
                    "this set still green."),
    }
    out_path.write_bytes((json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False)
                          + "\n").encode("utf-8"))
    print("\nviable %d  killed %d  survived %d  (controls %d, killed %d)  -> %s"
          % (len(viable), len(killed), len(survived), len(controls), len(ctl_killed),
             out_path.name))
    if void:
        print("VOID: a control was killed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
