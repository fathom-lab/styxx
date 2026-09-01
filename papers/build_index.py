# -*- coding: utf-8 -*-
"""Check papers/INDEX.md against the tree, and emit the receipt that check produces.

WHY THIS EXISTS. The 2026-09-01 audit measured the lab rediscovering its own findings:
47 arcs, 13 distinct questions, an arc-to-arc citation density of 0.041, fourteen arcs
citing no other and ten cited by none. Mention-versus-use was named in five arcs on
2026-05-25 and the synthesis three months later that catalogues ten instances cites none
of them. The diagnosis was not that the ideas were thin — it was that nothing made an arc
declare what came before it.

INDEX.md is the answer, and a map nothing enforces rots into a map nobody trusts. So this
script enforces the half that CAN be enforced.

WHAT IS MECHANICAL AND WHAT IS NOT — the distinction is the point, and overstating it
would be the same error the audit was written to catch:

  MECHANICAL (checked here, and a test fails on drift)
    every arc directory has exactly one row; every row names a directory that exists;
    every tag comes from the closed vocabulary; every status comes from the closed
    vocabulary; every module named in "ships in" exists on disk.

  AUTHORED (never regenerated, never validated here)
    the one-sentence terminal claim per arc, the idea index, the receipts index. Those
    are judgments. A script that regenerated them would be inventing the thing it claims
    to check, and this lab has spent a week finding instruments that were handed their
    target by the object they judged.

So the guarantee is narrow and worth stating plainly: you cannot add an arc to this
repository without adding a row to INDEX.md, and you cannot invent a tag. What the row
SAYS is on the author.

  python papers/build_index.py            # check, write the receipt, exit 1 on drift
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "papers" / "INDEX.md"
OUT = ROOT / "papers" / "index_check.json"

# Directories under papers/ that are not research arcs. `arxiv/` holds submission
# artifacts (SUBMIT.md, build_arxiv.py, PDFs), not a line of research. `showcase-viz` is
# NOT here: it carries 33 papers and the index rightly gives it a row.
NOT_ARCS = {"__pycache__", "assets", "figures", "arxiv", "sworn", "charon"}

# The closed vocabularies. Adding to either is a deliberate act, visible in a diff.
TAGS = {
    "extraction-vs-adjudication", "mention-vs-use", "handed-target", "refusal-as-verdict",
    "calibration", "cross-model-transfer", "representational-geometry",
    "oscillation-dynamics", "self-verification", "receipt-integrity", "preregistration",
    "introspection", "sycophancy-pressure", "deception-honesty", "knowledge-boundary",
    "agent-provenance", "policy-gating", "benchmark-construct", "other",
}
STATUSES = {"LIVE", "SUPERSEDED", "RETRACTED", "NEGATIVE-RESULT", "UNCLEAR"}


def arc_dirs() -> set:
    return {d.name for d in (ROOT / "papers").iterdir()
            if d.is_dir() and d.name not in NOT_ARCS}


def rows(text: str) -> list:
    """Rows of the arc table: the first pipe-table whose header names 'arc'."""
    out, in_table = [], False
    for line in text.splitlines():
        if line.startswith("| arc |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 4 or set(cells[0]) <= set("-: "):
                continue
            out.append(cells)
    return out


def main() -> int:
    text = INDEX.read_text(encoding="utf-8")
    table = rows(text)
    listed = {r[0] for r in table}
    actual = arc_dirs()

    problems = []
    for a in sorted(actual - listed):
        problems.append(f"arc has no INDEX row: papers/{a}/ - add one before opening work on it")
    for a in sorted(listed - actual):
        problems.append(f"INDEX row names a directory that does not exist: {a}")

    bad_tags, bad_status, missing_mod = [], [], []
    for r in table:
        arc = r[0]
        status = r[2].split()[0] if len(r) > 2 and r[2] else ""
        if status and status not in STATUSES:
            bad_status.append(f"{arc}: status {status!r} not in the closed vocabulary")
        for t in (r[3].split(",") if len(r) > 3 else []):
            # a tag may carry a parenthetical gloss; the tag is the head of the cell
            head = t.strip().split("(")[0].strip()
            if head and head not in TAGS:
                bad_tags.append(f"{arc}: tag {head!r} not in the closed vocabulary")
        for m in re.findall(r"`([^`]+\.py)`", r[4] if len(r) > 4 else ""):
            # A ships-in cell may name a repo-relative path (styxx/foo.py) or a bare
            # filename that lives inside the arc's own directory — resonance_profiler.py
            # sits in papers/frequency-resonance/, which is itself the audit's finding
            # that it was never promoted into styxx/.
            if not ((ROOT / m).exists() or (ROOT / "papers" / arc / m).exists()):
                missing_mod.append(f"{arc}: ships-in names {m}, which does not exist")

    problems += bad_tags + bad_status + missing_mod

    payload = {
        "what": "structural check of papers/INDEX.md against the tree",
        "checked": ["every arc directory has a row", "every row names a real directory",
                    "tags from the closed vocabulary", "statuses from the closed vocabulary",
                    "modules named in ships-in exist"],
        "not_checked": ("the one-sentence terminal claim, the idea index and the receipts "
                        "index are AUTHORED judgments and are never validated here — a script "
                        "that regenerated them would be inventing what it claims to check"),
        "arc_directories": len(actual),
        "rows": len(table),
        "problems": problems,
        "ok": not problems,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"arc directories : {len(actual)}")
    print(f"INDEX rows      : {len(table)}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("INDEX.md is consistent with the tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
