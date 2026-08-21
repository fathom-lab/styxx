# -*- coding: utf-8 -*-
"""Does the corpus contain any in-repo DECIDING consumer at all?

!!! THIS SCRIPT'S OWN NUMBER IS NOT TRUSTWORTHY, AND THAT IS STATED HERE RATHER
!!! THAN DISCOVERED LATER.
!!!
!!! It traces a TOKEN NAME across the whole repository. `if gate == "pass"` in
!!! analytics.py counts as a consumer of whatever produced `gate`, with no check
!!! that it is the SAME value. That is C7 from the flattering adjudication --
!!! cross-function attribution by name without dispatch resolution -- committed
!!! inside the measurement written to avoid it.
!!!
!!! So its DECIDED count is an UPPER BOUND only. The attainability check, which
!!! traced three actual producer->consumer paths, found ZERO. The truth is
!!! between 0 and that upper bound, and neither number licenses freezing a gate.

The attempt-2 attainability check demonstrated 0 of 3 hops. Before treating that
as a fact about the instrument, establish whether it is a fact about the corpus:
for each known defect, is the produced value ever

    DECIDED   branched on / thresholded / asserted, inside this repository
    EGRESS    formatted into a string, serialized to dict/JSON, printed, or
              returned from a public API -- i.e. it leaves the program while
              still unmeasured, and the decision is made OUTSIDE
    NEITHER   written and never read

`confabulation_ratio` is written, appears in an f-string, and appears in a
`to_dict()`. It is never branched on. If that generalizes, then the edge the
thesis names frequently TERMINATES OUTSIDE THE PROCESS, no static screen confined
to one repository can see it, and every GO/NO-GO written against this corpus is
unanswerable at any floor above zero.
"""
from __future__ import annotations

import ast
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "benchmarks" / "silent_pass" / "cases.json"

_EGRESS_CALLS = {"dumps", "dump", "print", "format", "write", "render",
                 "to_dict", "asdict", "json", "log", "info", "debug", "emit"}


def tree_of(path: str, src: str):
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def current_files() -> dict[str, str]:
    out = {}
    for p in (REPO / "styxx").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            out[str(p.relative_to(REPO)).replace("\\", "/")] = \
                p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return out


def classify(files: dict[str, str], token: str) -> tuple[str, list[str]]:
    """How is `token` (an attribute name or dict key) consumed across the repo?"""
    decided, egress = [], []
    for path, src in files.items():
        tree = tree_of(path, src)
        if tree is None:
            continue
        lines = src.splitlines()

        def mentions(node) -> bool:
            for s in ast.walk(node):
                if isinstance(s, ast.Attribute) and s.attr == token:
                    return True
                if isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Constant) \
                        and s.slice.value == token:
                    return True
                if isinstance(s, ast.Name) and s.id == token:
                    return True
            return False

        for n in ast.walk(tree):
            if isinstance(n, (ast.If, ast.While)) and mentions(n.test):
                decided.append(f"{path}:{n.lineno} decided | "
                               f"{lines[n.lineno-1].strip()[:70]}")
            elif isinstance(n, ast.Assert) and mentions(n.test):
                decided.append(f"{path}:{n.lineno} asserted")
            elif isinstance(n, ast.JoinedStr) and mentions(n):
                egress.append(f"{path}:{n.lineno} f-string")
            elif isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if (isinstance(k, ast.Constant) and k.value == token) or \
                            (v is not None and mentions(v)):
                        egress.append(f"{path}:{getattr(n,'lineno',0)} dict/serialize")
                        break
            elif isinstance(n, ast.Call):
                f = getattr(n.func, "id", None) or getattr(n.func, "attr", "")
                if f in _EGRESS_CALLS and any(mentions(a) for a in n.args):
                    egress.append(f"{path}:{n.lineno} {f}()")

    if decided:
        return "DECIDED", decided[:3]
    if egress:
        return "EGRESS", egress[:3]
    return "NEITHER", []


def main() -> int:
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    files = current_files()

    # the token to trace, per case: the field/key the corpus records as carrying
    # the bad value. Derived from what_was_returned where it names one.
    TOKENS = {
        "SP-2026-0004": "trust_score", "SP-2026-0005": "composite",
        "SP-2026-0006": "r", "SP-2026-0007": "r2",
        "SP-2026-0008": "risk_level", "SP-2026-0009": "healthy",
        "SP-2026-0010": "verified", "SP-2026-0011": "valid",
        "SP-2026-0012": "confabulation_ratio", "SP-2026-0013": "verified",
        "SP-2026-0014": "timestamp", "SP-2026-0015": "axis",
        "SP-2026-0016": "divergence", "SP-2026-0017": "outcome",
        "SP-2026-0018": "centroids", "SP-2026-0019": "gate",
        "SP-2026-0020": "entropy", "SP-2026-0001": "status",
        "SP-2026-0002": "risk", "SP-2026-0003": "health",
    }

    print("CORPUS CONSUMER CENSUS — where does each bad value actually go?\n")
    tally: Counter = Counter()
    for c in cases:
        cid = c["id"]
        tok = TOKENS.get(cid)
        if not tok:
            tally["no-token"] += 1
            continue
        kind, ev = classify(files, tok)
        tally[kind] += 1
        print(f"  {cid}  {tok:22} {kind}")
        for e in ev:
            print(f"{'':16}{e}")

    total = sum(tally.values())
    print(f"\n  over {total} cases:")
    for k, v in tally.most_common():
        print(f"    {k:10} {v:3d}  {v/total:5.1%}")

    dec = tally.get("DECIDED", 0)
    print(f"\n  in-repo DECIDING consumers: {dec}/{total}")
    if dec < 8:
        print("  => a producer->decision screen confined to this repository cannot")
        print("     reach a floor of 10/20 at any tuning. The attempt-2 gate as")
        print("     drafted is UNANSWERABLE and must be revised before freezing.")
    print("\n  UPPER BOUND ONLY. This traces a token NAME, not a value: two")
    print("  different variables called `gate` are indistinguishable to it, which")
    print("  is C7 from the flattering adjudication committed inside the")
    print("  measurement written to avoid it. The attainability check, tracing")
    print("  three real producer->consumer paths, found 0. The truth is between,")
    print("  and neither end of that interval licenses freezing a gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
