# -*- coding: utf-8 -*-
"""Measure the corpus's TOPOLOGY: how does a bad value travel to its consumer?

The precondition committed to in `RESULT_edges_2026_08_21.md`. `styxx.edges`
reached 0/20 because it requires the producer call and the decision to sit in the
same function, and it was aimed at a corpus whose defects mostly do not. Rather
than guess at the fix, measure the target:

    for each of the 20 known defects, what SHAPE does the value exit in?

    scalar          returned bare -- an intra-procedural screen can follow it
    object-field    written into a dataclass/attribute, read as `r.field`
    dict-key        written into a dict, read as `d["key"]`
    tuple           returned in a tuple, unpacked by the caller
    none-returned   the function returns nothing at the defect line

This is a measurement of known-true instances, not a test of anything. It decides
whether "follow the value through a field" is the right mechanism for attempt 2
or merely the obvious one.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "benchmarks" / "silent_pass" / "cases.json"
_TMP = Path(tempfile.mkdtemp(prefix="styxx_topo_"))


def source_at(commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}~1:{path}"], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def enclosing(tree: ast.AST, line: int):
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.lineno <= line <= (n.end_lineno or n.lineno):
                span = (n.end_lineno or n.lineno) - n.lineno
                if best is None or span < best[0]:
                    best = (span, n)
    return best[1] if best else None


def exit_shape(fn: ast.AST, line: int) -> tuple[str, str]:
    """How does the value at `line` leave this function?"""
    # the statement at (or nearest above) the defect line
    node = None
    for n in ast.walk(fn):
        if getattr(n, "lineno", None) == line:
            node = n
            break

    # does the defect line sit inside a constructor call / keyword argument?
    for n in ast.walk(fn):
        if isinstance(n, ast.keyword) and n.value is not None:
            lo = getattr(n.value, "lineno", None)
            if lo == line:
                return "object-field", f"keyword {n.arg}= in a constructor call"
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if getattr(v, "lineno", None) == line:
                    kn = getattr(k, "value", "?")
                    return "dict-key", f"dict key {kn!r}"

    # a return at or containing the line
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            lo, hi = n.lineno, (n.end_lineno or n.lineno)
            if lo <= line <= hi:
                v = n.value
                if isinstance(v, ast.Dict):
                    return "dict-key", "returned inside a dict literal"
                if isinstance(v, ast.Tuple):
                    return "tuple", f"returned in a {len(v.elts)}-tuple"
                if isinstance(v, ast.Call):
                    nm = getattr(v.func, "id", None) or getattr(v.func, "attr", "")
                    if nm and (nm[:1].isupper() or nm == "cls"):
                        return "object-field", f"returned as {nm}(...)"
                    return "scalar", f"returned from {nm}(...)"
                return "scalar", "returned bare"

    # assignment to an attribute
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and getattr(n, "lineno", None) == line:
            for t in n.targets:
                if isinstance(t, ast.Attribute):
                    return "object-field", f"assigned to self.{t.attr}"
                if isinstance(t, ast.Subscript):
                    return "dict-key", "assigned into a subscript"
            return "scalar", "assigned to a local"

    return "unclassified", ""


def main() -> int:
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    shapes: Counter = Counter()
    rows = []
    for c in cases:
        mod, commit, line = c.get("module"), c.get("fix_commit"), c.get("defect_line")
        src = source_at(commit, mod) if (mod and commit) else None
        if not src or not line:
            rows.append((c["id"], "UNRUN", "", c.get("consumer", "")))
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            rows.append((c["id"], "UNPARSED", "", c.get("consumer", "")))
            continue
        fn = enclosing(tree, line)
        if fn is None:
            shapes["module-level"] += 1
            rows.append((c["id"], "module-level", "not inside a function",
                         c.get("consumer", "")))
            continue
        shape, why = exit_shape(fn, line)
        shapes[shape] += 1
        rows.append((c["id"], shape, f"{fn.name}: {why}", c.get("consumer", "")))

    print("CORPUS TOPOLOGY — how each known defect's value reaches its consumer\n")
    for cid, shape, why, consumer in rows:
        print(f"  {cid}  {shape:14} {why[:52]:52}")
        if consumer:
            print(f"{'':16}consumer as recorded: {consumer[:70]}")
    total = sum(shapes.values())
    print(f"\n  shapes over {total} classified cases:")
    for k, v in shapes.most_common():
        print(f"    {k:16} {v:3d}  {v/total:5.1%}")

    reachable = shapes.get("scalar", 0)
    print(f"\n  reachable by an INTRA-PROCEDURAL screen (scalar only): "
          f"{reachable}/{total} = {reachable/total:.0%}" if total else "")
    print("  everything else needs the value followed across a field, key or tuple.")
    print("\n  This is a measurement of the target, not a test. It selects the")
    print("  mechanism for attempt 2; it does not license any claim.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
