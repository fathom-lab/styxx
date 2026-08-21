# -*- coding: utf-8 -*-
"""Quantify a detector flaw found AFTER the frozen scan, without touching the detector.

G3 of the prereg forbids editing styxx/flattering.py after the run. So this
measures the flaw from the outside: of the hits produced, how many rest on a
BARE NAME in the test position -- `if flag:` / `x if flag else y` -- which the
screen reads as a container-emptiness test but which is, in most real code, a
boolean flag.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "papers" / "out_flattering_external.json"


def test_shape(path: str, line: int):
    """Classify the test that produced a hit at this line."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return "unparseable"
    best = None
    for node in ast.walk(tree):
        test = None
        if isinstance(node, ast.IfExp) and node.lineno <= line <= (node.end_lineno or line):
            test = node.test
        elif isinstance(node, ast.If) and node.lineno <= line <= (node.end_lineno or line):
            test = node.test
        if test is None:
            continue
        span = (node.end_lineno or line) - node.lineno
        if best is None or span < best[0]:
            best = (span, test)
    if best is None:
        return "not-found"
    t = best[1]
    while isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
        t = t.operand
    if isinstance(t, ast.Name):
        return "BARE_NAME"
    if isinstance(t, ast.Call):
        return "len()" if getattr(t.func, "id", "") == "len" else "call"
    if isinstance(t, ast.Compare):
        return "len()-compare" if any(
            isinstance(x, ast.Call) and getattr(x.func, "id", "") == "len"
            for x in [t.left, *t.comparators]) else "compare"
    if isinstance(t, ast.Attribute):
        return "attribute"
    return type(t).__name__


def main() -> int:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for tier, hits in (("TIER-A", d["tier_a"]), ("TIER-B (200 sample)", d["tier_b_sample"])):
        c = Counter(test_shape(h["path"], h["line"]) for h in hits)
        n = sum(c.values())
        print(f"\n{tier}  n={n}")
        for k, v in c.most_common():
            flag = "  <-- boolean flag misread as emptiness" if k == "BARE_NAME" else ""
            print(f"    {k:16} {v:4d}  {v/n:6.1%}{flag}")
    print("\nTIER-A hits resting on a bare name:")
    for h in d["tier_a"]:
        if test_shape(h["path"], h["line"]) == "BARE_NAME":
            print(f"    {h['package']:12} {h['function']:34} {h['snippet'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
