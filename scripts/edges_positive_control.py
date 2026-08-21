# -*- coding: utf-8 -*-
"""GO / NO-GO for the edge screen — prereg G0, committed before the screen existed.

    If the edge screen catches fewer than 8 of the 20 cases in
    benchmarks/silent_pass -- measured against real pre-fix source extracted at
    <fix_commit>~1 -- it is NOT run against external code at all, and the
    preregistration terminates with that number published.

Why the gate exists: `styxx.flattering` scored **10%** on this same corpus while
being fitted to it, and its external `0 of 8` was uninterpretable as a direct
consequence. A screen with zero recall and a defect-free corpus produce
byte-identical output.

This number is IN-SAMPLE and is an UPPER bound on recall, never an estimate of
it: the rules were written with this corpus in view.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from styxx.edges import scan_package

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "benchmarks" / "silent_pass" / "cases.json"
FLOOR = 8
TOL = 3          # a producer within +/-3 lines of the recorded defect line
_ROOT = Path(tempfile.mkdtemp(prefix="styxx_edges_pc_"))


def package_at(commit: str) -> Path | None:
    dest = _ROOT / commit
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    try:
        tar = subprocess.run(["git", "archive", f"{commit}~1", "styxx"],
                             cwd=REPO, capture_output=True, check=True).stdout
        p = subprocess.Popen(["tar", "-x", "-C", str(dest)], stdin=subprocess.PIPE)
        p.communicate(tar)
        if p.returncode != 0:
            return None
    except subprocess.CalledProcessError:
        return None
    return dest


def main() -> int:
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw

    print("GO/NO-GO — edge screen vs. real pre-fix source of 20 known defects")
    print(f"prereg G0 floor: {FLOOR} of {len(cases)}\n")

    # one scan per distinct fix commit, reused across cases
    scans: dict[str, object] = {}
    caught = 0
    by_subtype: Counter = Counter()
    rows = []
    for c in cases:
        cid, module = c["id"], c.get("module")
        commit, line = c.get("fix_commit"), c.get("defect_line")
        subtype = str(c.get("subtype", "?")).split()[0]
        if not (module and commit and line):
            rows.append(("SKIP  ", cid, subtype, module, "case lacks module/commit/line"))
            continue
        if commit not in scans:
            root = package_at(commit)
            scans[commit] = scan_package(root / "styxx") if root else None
        rep = scans[commit]
        if rep is None or not getattr(rep, "measured", False):
            rows.append(("UNRUN ", cid, subtype, module, "package not extractable"))
            continue

        want = module.replace("\\", "/").lower()
        near = [e for e in rep.edges
                if want.endswith(Path(e.producer_path).as_posix().split("styxx/", 1)[-1].lower())
                or Path(e.producer_path).as_posix().lower().endswith(want)]
        hit = [e for e in near if abs(e.producer_line - line) <= TOL]
        if hit:
            caught += 1
            by_subtype[subtype] += 1
            e = hit[0]
            rows.append(("CAUGHT", cid, subtype, module,
                         f"L{e.producer_line} -> decided at "
                         f"{Path(e.consumer_path).name}:{e.consumer_line} "
                         f"({e.consumer_func}) | {e.loud_evidence}"))
        else:
            rows.append(("miss  ", cid, subtype, module,
                         f"{len(near)} edge(s) elsewhere in file" if near else ""))

    for mark, cid, sub, mod, note in rows:
        print(f"  {mark} {cid}  {sub:6} {str(mod):26} {note}")

    any_rep = next((r for r in scans.values() if r is not None), None)
    if any_rep is not None:
        print(f"\n  (resolution on the styxx tree: "
              f"{any_rep.resolution:.1%} intra-package, "
              f"{any_rep.raw_resolution:.1%} raw; "
              f"{len(any_rep.edges)} total edges flagged)")

    print(f"\n  {caught}/{len(cases)} caught   by subtype: {dict(by_subtype) or 'none'}")
    if caught >= FLOOR:
        print(f"\n  G0 PASS ({caught} >= {FLOOR}) -> the external run is authorised.")
        print("  This is IN-SAMPLE and an UPPER bound on recall, not an estimate.")
        return 0
    print(f"\n  G0 FAIL ({caught} < {FLOOR}) -> NO external run. The preregistration")
    print("  terminates here and this number is what gets published.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
