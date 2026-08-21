# -*- coding: utf-8 -*-
"""Apply Q2's frozen DIFF SHAPE to the Q1 candidates. No new query is invented.

The preregistration froze two queries. Q1 (commit message) returned 415
candidates, too many for three-lens adjudication each. Rather than introduce a
screening rule that was not preregistered, this takes the INTERSECTION of the two
frozen queries: a Q1 candidate whose diff also has Q2's shape.

    removes   return <flattering constant>        0 / 0.0 / True / 1.0 / "pass"
    adds      raise | nan | None | warn | skip | log.warning

All three yields are reported — Q1 alone, the shape test alone on those
candidates, and the intersection — so the narrowing is visible rather than
implied. The intersection is a SUBSET of a frozen query, never a widening of one.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path("C:/Users/heyzo/AppData/Local/Temp/spcorpus")
IN = Path(__file__).resolve().parent.parent / "papers" / "out_sp_ext_candidates.json"
OUT = Path(__file__).resolve().parent.parent / "papers" / "out_sp_ext_shaped.json"

REMOVED = re.compile(
    r'^-\s*(return\s+(0\.0|0|True|1\.0|1|\[\]|\{\}|"(pass|ok|valid|healthy|steady)")\b'
    r'|.*=\s*(0\.0|True)\s*$)', re.I)
ADDED = re.compile(
    r'^\+.*\b(raise|nan|float\("nan"\)|None|warn|warning|skip|logger\.(warn|error)'
    r'|log\.(warn|error)|pytest\.skip|measured|NotImplemented)\b')


def diff_of(repo: Path, sha: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "show", "--format=", "--unified=3", sha],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=240).stdout
    except subprocess.TimeoutExpired:
        return ""


def has_shape(diff: str) -> tuple[bool, str, str]:
    """Q2 shape, evaluated hunk by hunk so the two lines must co-occur."""
    hunk: list[str] = []
    for line in diff.splitlines() + ["@@"]:
        if line.startswith("@@"):
            rem = next((l for l in hunk if REMOVED.match(l)), None)
            add = next((l for l in hunk if ADDED.match(l)), None)
            if rem and add:
                return True, rem.strip()[:90], add.strip()[:90]
            hunk = []
        else:
            hunk.append(line)
    return False, "", ""


def main() -> int:
    d = json.loads(IN.read_text(encoding="utf-8"))
    cands = d["candidates"]
    print(f"Q1 candidates: {len(cands)}")
    kept, tally = [], Counter()
    for i, c in enumerate(cands, 1):
        repo = ROOT / c["repo"]
        diff = diff_of(repo, c["sha"])
        if not diff:
            continue
        ok, rem, add = has_shape(diff)
        if not ok:
            continue
        c = dict(c)
        c["shape_removed"], c["shape_added"] = rem, add
        c["diff_chars"] = len(diff)
        kept.append(c)
        tally[c["repo"]] += 1
        if i % 60 == 0:
            print(f"    scanned {i}/{len(cands)}, kept {len(kept)}", flush=True)

    print(f"\nQ1 n Q2-shape: {len(kept)}  ({len(kept)/len(cands):.1%} of Q1)")
    for r, n in tally.most_common():
        print(f"    {r:24} {n}")
    OUT.write_text(json.dumps({"n_q1": len(cands), "n_shaped": len(kept),
                               "by_repo": dict(tally), "candidates": kept},
                              indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
