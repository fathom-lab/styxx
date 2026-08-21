# -*- coding: utf-8 -*-
"""Q2, standalone, with the regexes EXACTLY as frozen at 38b8428.

Two things this fixes.

**1. Q2 never ran.** Its efficient form is `git log -S` (pickaxe), which against
`--filter=blob:none` clones timed out after 10 minutes on the smallest repository
in the set. On a full clone the same query takes **13 seconds**. The obstacle was
lazy blob fetch, not the query.

**2. My shape filter was WIDER than the frozen query, and I did not notice.**
`scripts/sp_ext_shape.py` implements the removed-side as::

    return (0.0|0|True|1.0|1|[]|{}|"pass"...)   OR   .* = (0.0|True)$
                                                     ^^^^^^^^^^^^^^^^ not in the prereg

and the added-side with `float("nan")`, `warning`, `logger.(warn|error)`,
`pytest.skip`, `measured`, `NotImplemented` — none of which are in the frozen
text either. The preregistration says queries are not edited (G4, G6), and a
widened query is an edited one even when the widening is unintentional.

So this module carries the frozen patterns **verbatim**, and the runner reports
both yields so the discrepancy is a number rather than an apology.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("C:/Users/heyzo/AppData/Local/Temp/spfull")
OUT = Path(__file__).resolve().parent.parent / "papers" / "out_sp_ext_q2.json"

# ── verbatim from PREREG_sp_external_corpus_2026_08_21.md, section HARVEST QUERIES
FROZEN_REMOVED = re.compile(r'return\s+(0|0\.0|True|1\.0|"(pass|ok|valid|healthy)")')
FROZEN_ADDED = re.compile(r'raise|nan|None|warn|skip|log\.(warn|error)')

# pickaxe seeds: the literal strings whose occurrence count must change. These
# only decide WHICH COMMITS ARE EXAMINED; the frozen regexes above decide which
# qualify, so a seed cannot widen the query.
SEEDS = ["return 0.0", "return 0", "return True", "return 1.0",
         'return "pass"', 'return "ok"', 'return "valid"', 'return "healthy"']


def git(repo: Path, *args, timeout=900) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def qualifying_hunk(diff: str):
    """Frozen Q2: a REMOVED line matching FROZEN_REMOVED and an ADDED line
    matching FROZEN_ADDED, in the SAME hunk."""
    hunk: list[str] = []
    for line in diff.splitlines() + ["@@"]:
        if line.startswith("@@"):
            rem = next((l for l in hunk
                        if l.startswith("-") and not l.startswith("--")
                        and FROZEN_REMOVED.search(l)), None)
            add = next((l for l in hunk
                        if l.startswith("+") and not l.startswith("++")
                        and FROZEN_ADDED.search(l)), None)
            if rem and add:
                return rem.strip()[:100], add.strip()[:100]
            hunk = []
        else:
            hunk.append(line)
    return None


def main() -> int:
    repos = sorted(d for d in ROOT.iterdir() if (d / ".git").exists())
    if not repos:
        print(f"no full clones under {ROOT}", file=sys.stderr)
        return 2

    kept, tally, seen_shas = [], Counter(), 0
    for r in repos:
        head = git(r, "rev-parse", "HEAD").strip()[:12]
        shas: set[str] = set()
        for s in SEEDS:
            shas.update(git(r, "log", "--all", "--no-merges", f"-S{s}",
                            "--format=%H").split())
        seen_shas += len(shas)
        hit = 0
        for sha in shas:
            diff = git(r, "show", "--format=", "--unified=3", sha, timeout=240)
            if not diff or len(diff) > 900_000:
                continue
            q = qualifying_hunk(diff)
            if not q:
                continue
            rem, add = q
            files = [l.split("|")[0].strip()
                     for l in git(r, "show", "--stat", "--format=", sha).splitlines()
                     if "|" in l]
            if not any(f.endswith(".py") for f in files) or len(files) > 40:
                continue
            subj = git(r, "show", "-s", "--format=%s", sha).strip()
            kept.append({"repo": r.name, "sha": sha, "head": head, "subject": subj,
                         "shape_removed": rem, "shape_added": add,
                         "n_files": len(files), "query": "Q2-frozen"})
            hit += 1
        tally[r.name] = hit
        print(f"  {r.name:24} HEAD {head}  pickaxe {len(shas):5d} -> Q2 {hit:4d}",
              flush=True)

    print(f"\n  commits examined by pickaxe: {seen_shas}")
    print(f"  Q2 (frozen regexes, standalone): {len(kept)}")
    OUT.write_text(json.dumps(
        {"frozen_removed": FROZEN_REMOVED.pattern,
         "frozen_added": FROZEN_ADDED.pattern,
         "seeds": SEEDS, "n_pickaxe_commits": seen_shas,
         "n_q2": len(kept), "by_repo": dict(tally), "candidates": kept},
        indent=1), encoding="utf-8")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
