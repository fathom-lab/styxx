"""Propose pairs of papers that state a different value for what may be the same quantity.

    python papers/closed-model-frontier/audit_cross_paper_numbers.py
    python papers/closed-model-frontier/audit_cross_paper_numbers.py --min-overlap 4

**This script does not find contradictions. It proposes candidates for a human to read.**

That distinction is the whole design. Three times in two days this lab recorded the same error
-- BENCH-1 with the prefix extractor, BENCH-2 with the oracle's acceptance rate, DECIDE-1 with
the instrument's admission rate -- and each time the error was treating machinery's output as
ground truth about the world. A script that scanned the papers and announced contradictions
would be the fourth. So this one prints candidates and says nothing about them, and
`AUDIT_cross_paper_numbers_2026_09_18.md` records what reading each candidate established.

On the run recorded in that document, 20 candidates surfaced and 1 was a real supersession. A
5% hit rate is what a proposer looks like, and it is worth running anyway: the one it found had
been sitting in the record unnoticed, and was originally discovered by accident rather than by
looking.

## Method

Every `N of M` in `papers/**.md` with M >= 20 and N <= M. Two occurrences are a candidate pair
when they share M, differ in N, live in different files, and their surrounding text shares at
least `--min-overlap` content words. Sharing M is what makes two numbers comparable at all; the
word overlap is what makes them plausibly about the same thing.

## What it cannot see

Percentages that never state their denominator. Quantities written only as prose. Numbers that
agree but were measured against different instruments, which is the case the next reader should
worry about most -- every RESULT here pins its instrument sha in its header, so two papers can
state different numbers and both be right, and only reading says which.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
STOP = frozenset(
    "the a an of and or to in on for is are was were with that this it its as at by from be "
    "been not no we our us they them their than then so if but".split()
)
NUM = re.compile(r"(?<![\d.,])(\d[\d,]*)\s+of\s+(\d[\d,]*)(?![\d.,])")
WORD = re.compile(r"[a-z_]{4,}")


def occurrences(root: pathlib.Path):
    """Every `N of M` in the papers, except in audit documents.

    An `AUDIT_*.md` quotes other papers' numbers in order to discuss them, so leaving it in
    makes the scan find its own commentary and makes the candidate count reported in that
    commentary fail to reproduce. The exclusion is one filename prefix, named here rather than
    buried, because a silent exclusion is how a scan stops covering what it claims to cover.
    """
    for f in sorted(root.rglob("*.md")):
        if f.name.startswith("AUDIT_"):
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines, 1):
            for m in NUM.finditer(line):
                n = int(m.group(1).replace(",", ""))
                d = int(m.group(2).replace(",", ""))
                if d < 20 or n > d:
                    continue
                ctx = line[max(0, m.start() - 90):m.start()] + " " + line[m.end():m.end() + 90]
                words = {w for w in WORD.findall(ctx.lower()) if w not in STOP}
                yield d, n, words, f.relative_to(ROOT), i, line.strip()


def candidates(min_overlap: int):
    by_den = collections.defaultdict(list)
    for occ in occurrences(ROOT / "papers"):
        by_den[occ[0]].append(occ)
    seen, out = set(), []
    for den, rows in sorted(by_den.items()):
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                A, B = rows[a], rows[b]
                if A[1] == B[1] or A[3] == B[3]:
                    continue
                shared = A[2] & B[2]
                if len(shared) < min_overlap:
                    continue
                key = (den, A[1], B[1], str(A[3]), str(B[3]))
                if key in seen:
                    continue
                seen.add(key)
                out.append((den, A, B, sorted(shared)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Propose candidates. Decide nothing.")
    ap.add_argument("--min-overlap", type=int, default=3)
    a = ap.parse_args()
    found = candidates(a.min_overlap)
    for den, A, B, shared in found:
        print(f"\n=== of {den}: {A[1]} vs {B[1]}   shared={shared[:6]}")
        print(f"  A {A[3]}:{A[4]}\n    {A[5][:160]}")
        print(f"  B {B[3]}:{B[4]}\n    {B[5][:160]}")
    print(f"\n{len(found)} candidate pairs. None of them is a finding until someone reads both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
