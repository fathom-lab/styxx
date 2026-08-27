"""CENSUS — four silent passes in this lab's own verification machinery, and whether its own
silent-pass detectors see them.

**A census. It licenses no claim, and it proposes no fix.** It measures a detector against four
cases it did not have, so a successor preregistration knows what it would be trying to beat.

## Why this exists

`styxx.absence` describes itself in one line: *find the places where NOT MEASURING reads as a good
result.* Between 2026-08-25 and 2026-08-26, four defects of exactly that description were found in
this repository's own verification machinery — every one by accident, none by the detector:

  A  `tests/test_ledger.py` skips itself when `.git/shallow` exists, and CI checks out at depth 1,
     so the LEDGER's regeneration guarantee has never run in CI on any Python version. A skipped
     test reads GREEN on a pull request.
  B  `tests/test_certificate_reproduces.py` yields only documents whose receipts all resolve, so a
     document with an unresolvable receipt is dropped from the drift guard rather than reported.
     36 of 178 certificates were invisible to it.
  C  `_resolve_receipts` treats a sha mismatch as `missing`. Every receipt hash in this corpus was
     recorded on Windows and is a CRLF hash, so on Linux cross-directory receipts failed to
     resolve and their documents vanished from the guard — CI called one "repaired" when it had
     merely become unexaminable.
  D  the v0.11 battery's gate G4'b tested `all(c["call"] != "LABEL" for c in fresh)`. `all()` over
     an empty list is True, so a panel artifact containing no fresh draw at all cleared the bar.

The SILENT-PASS corpus is repo-only on purpose and scores against real history. These are real
history, they are ours, and they are the kind of case a benchmark cannot be handed by its author
without cheating — nobody wrote them to be caught.

## What is measured

For each case: the pre-fix source at its real commit, both detectors run on it, and whether either
flags a line within `WINDOW` lines of the defect. Plus two controls, because a detector reporting
zero everywhere is indistinguishable from a detector being called wrongly:

  * a POSITIVE control — a textbook `except: return {"gate": "pass"}` the detector must catch;
  * a NEGATIVE control — source with no defect, which must stay clean.

The first version of this census reported zero findings everywhere, and the reason was that
`scan_source` returns a list while `scan_path` returns a report; the harness read `.findings` off
a list and measured nothing. The controls exist because that happened.

  python papers/closed-model-frontier/oath_vacuous_pass_census.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.absence import scan_source as absence_scan                       # noqa: E402
from styxx.loops import scan_source as loops_scan                           # noqa: E402

OUT = HERE / "oath_vacuous_pass_census.json"
WINDOW = 6          # a flag this close to the defect counts as a catch, generously

CASES = [
    {"id": "VP-A", "commit": "HEAD",
     "path": "tests/test_ledger.py", "line": 25,
     "shape": "pytest.skip() on a shallow clone; a skipped test reads GREEN in CI",
     "found_by": "reading .github/workflows/test.yml for an unrelated reason"},
    {"id": "VP-B", "commit": "cbd2864",
     "path": "tests/test_certificate_reproduces.py", "line": 56,
     "shape": "generator yields only fully-resolvable documents; the rest are dropped, not reported",
     "found_by": "an adversarial program audit"},
    {"id": "VP-C", "commit": "cbd2864",
     "path": "styxx/corpus_audit.py", "line": 85,
     "shape": "sha mismatch classified as `missing`; platform-dependent hashes make it silent",
     "found_by": "CI going red on a document that passes locally"},
    {"id": "VP-D", "commit": "9d09ef3",
     "path": "papers/closed-model-frontier/run_oath_v11_battery.py", "line": 329,
     "shape": "all() over an empty population returns True; the gate passes having examined nothing",
     "found_by": "an adversarial audit of the battery, after every gate passed"},
]

POSITIVE_CONTROL = '''
def check(rows):
    try:
        return {"trust": compute(rows), "gate": "pass"}
    except Exception:
        return {"trust": 1.0, "gate": "pass"}
'''

NEGATIVE_CONTROL = '''
def add(a, b):
    return a + b
'''


def detect(src: str, name: str) -> dict:
    """Both detectors, driven exactly as `benchmarks/silent_pass` drives them."""
    try:
        a = {f.line for f in absence_scan(src, name)}
    except SyntaxError:
        a = set()
    try:
        v = loops_scan(src, name)
        loops = {d.line for d in v.derivations} | {t.line for t in v.trust}
    except SyntaxError:
        loops = set()
    return {"absence": sorted(a), "loops": sorted(loops)}


def at(commit: str, path: str) -> str:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout


def main() -> int:
    pos = detect(POSITIVE_CONTROL, "positive_control.py")
    neg = detect(NEGATIVE_CONTROL, "negative_control.py")
    controls_ok = bool(pos["absence"]) and not neg["absence"]

    rows = []
    for c in CASES:
        src = at(c["commit"], c["path"])
        if not src:
            rows.append({**c, "error": "source unreadable at that commit"})
            continue
        d = detect(src, c["path"])
        near = sorted(x for x in set(d["absence"]) | set(d["loops"])
                      if abs(x - c["line"]) <= WINDOW)
        rows.append({**c, "source_lines": src.count("\n"),
                     "absence_findings": len(d["absence"]),
                     "loops_findings": len(d["loops"]),
                     "flags_within_window": near,
                     "caught": bool(near)})

    caught = sum(1 for r in rows if r.get("caught"))
    payload = {
        "census": "vacuous pass — four silent passes in this lab's own verification machinery",
        "status": "CENSUS. Licenses no claim, proposes no fix. Sizes a detector's blind spot "
                  "against four cases it did not have.",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "window_lines": WINDOW,
        "controls": {
            "positive": pos, "negative": neg, "usable": controls_ok,
            "why": "a detector reporting zero everywhere is indistinguishable from a detector "
                   "being called wrongly. The first version of this census made exactly that "
                   "mistake — it read `.findings` off a list — and the controls are why it was "
                   "caught rather than published."},
        "cases_examined": len(rows),
        "cases_caught": caught,
        "cases_missed": len(rows) - caught,
        "cases": rows,
        "the_shared_shape": (
            "None of the four is 'a healthy value returned on a crash', which is the family "
            "`absence` does catch (SP-1/SP-5). All four are SUCCESS BY EMPTY POPULATION: the "
            "thing to be checked is filtered away upstream — by a skip, by a generator guard, by "
            "a mismatch reclassified as absence — and a downstream check then passes over "
            "nothing, reporting the same green it would report on a full population. The "
            "emptiness is not at the function boundary where a value-returning detector looks; "
            "it is manufactured several steps earlier, and the check that passes never learns "
            "the difference."),
        "what_this_does_not_show": [
            "That `styxx.absence` is broken. Its published recall is 9/20 on the SILENT-PASS "
            "benchmark and its own LIMITS string says a clean run is not a certificate. Missing "
            "a family it was not built for is a measured blind spot, not a defect.",
            "That these four are the whole class. Four is four.",
            "That a detector for this shape is feasible. Nothing here was built or tried; the "
            "rejected-design arithmetic that would decide it has not been done.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"controls usable: {controls_ok}  (positive absence={pos['absence']}, "
          f"negative absence={neg['absence']})")
    for r in rows:
        print(f"  {r['id']}  {r['path'][-46:]:<46} absence={r.get('absence_findings')} "
              f"loops={r.get('loops_findings')}  caught={r.get('caught')}")
    print(f"\n{caught} of {len(rows)} caught -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
