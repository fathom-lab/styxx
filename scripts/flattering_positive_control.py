# -*- coding: utf-8 -*-
"""POSITIVE CONTROL for the frozen flattering screen — the leg the external run lacked.

The adversarial adjudication of the external run made one objection that
invalidates the whole thing:

    "A screen with zero recall and a defect-free corpus produce byte-identical
     output, and this run cannot distinguish them."

That is the defect class this project studies, committed by this project's own
experiment: an instrument that never fired, reported as a clean corpus. 0 of 8
means nothing until the screen is shown to fire on code that is known to be
defective.

So: run the FROZEN screen (no edits — prereg G3) against real pre-fix source for
every SILENT-PASS corpus case, extracted from git at `<fix_commit>~1`. These are
known-true instances by construction: each one is a defect this project found,
fixed, and recorded with the commit that fixed it.

Recall here is an UPPER bound on recall in the wild, not an estimate of it: the
rules were written against this corpus. A screen that cannot even reach its own
training cases has no recall anywhere; one that reaches them has an unknown
amount elsewhere.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from styxx.flattering import scan_source

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "benchmarks" / "silent_pass" / "cases.json"


def prefix_source(commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}~1:{path}"], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw

    print("POSITIVE CONTROL — frozen screen vs. real pre-fix source of known defects\n")
    hit = miss = unrunnable = 0
    by_subtype: Counter = Counter()
    rows = []
    for c in cases:
        module, commit, line = c.get("module"), c.get("fix_commit"), c.get("defect_line")
        subtype = str(c.get("subtype", "?")).split()[0]
        src = prefix_source(commit, module) if (module and commit) else None
        if src is None:
            unrunnable += 1
            rows.append(("UNRUN ", c["id"], subtype, module, ""))
            continue
        try:
            hits = scan_source(src, module)
        except SyntaxError:
            unrunnable += 1
            rows.append(("UNRUN ", c["id"], subtype, module, "pre-fix source unparseable"))
            continue
        # Fires ON THIS DEFECT only if a TIER-A hit lands within +/-3 lines of the
        # recorded defect line. A hit elsewhere in the same 2000-line file is not
        # a catch -- that is the localization discipline the corpus already uses.
        near = [h for h in hits if h.tier == "A" and line and abs(h.line - line) <= 3]
        anywhere = [h for h in hits if h.tier == "A"]
        if near:
            hit += 1
            by_subtype[subtype] += 1
            rows.append(("CAUGHT", c["id"], subtype, module,
                         f"L{near[0].line} {near[0].polarity_from}"))
        else:
            miss += 1
            rows.append(("miss  ", c["id"], subtype, module,
                         f"{len(anywhere)} TIER-A elsewhere in file" if anywhere else ""))

    for mark, cid, sub, mod, note in rows:
        print(f"  {mark} {cid}  {sub:6} {str(mod):28} {note}")

    scored = hit + miss
    print(f"\n  {hit}/{scored} known defects caught"
          f"  ({unrunnable} unrunnable, excluded from the denominator)")
    if scored:
        print(f"  recall on its own corpus: {hit/scored:.1%}")
    print(f"  by subtype: {dict(by_subtype) or 'none'}")

    print("\n  Reading:")
    if hit == 0:
        print("    THE SCREEN HAS ZERO RECALL ON KNOWN DEFECTS. The external 0/8 is")
        print("    therefore uninformative in both directions and must not be quoted")
        print("    as evidence about anything.")
    else:
        print(f"    The screen fires on real defects, so the external 0/8 is not the")
        print(f"    output of a dead instrument. It remains an UPPER bound: these are")
        print(f"    the cases the rules were written against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
