# -*- coding: utf-8 -*-
"""EXPLORATORY — is "flattering" a property of the function, or of its consumer?

Not a preregistered test and not a claim. This asks one question of the data the
frozen scan already produced:

    For a function that returns a flattering constant on empty input, does any
    CALLER actually branch on or threshold the returned value?

If flattery is a property of the function, the answer should not matter. If it is
a property of the EDGE between producer and consumer -- which is what
`contract` (boundary-only, 3/5) and `flattering` (87% bare-name) both failed
their way into suggesting -- then the caller is the whole story.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "papers" / "out_flattering_external.json"
_SKIP = {".git", "__pycache__"}


class _CallSites(ast.NodeVisitor):
    """Find calls to `target` and classify what happens to the return value."""

    def __init__(self, target: str):
        self.target, self.sites = target, []
        self._ctx: list[str] = []

    def _named(self, fn) -> bool:
        return (isinstance(fn, ast.Name) and fn.id == self.target) or \
               (isinstance(fn, ast.Attribute) and fn.attr == self.target)

    def visit_If(self, node):
        self._scan(node.test, "branched-on")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self._scan(node.test, "branched-on")
        self.generic_visit(node)

    def visit_Compare(self, node):
        for x in [node.left, *node.comparators]:
            self._scan(x, "thresholded")
        self.generic_visit(node)

    def visit_Assert(self, node):
        self._scan(node.test, "asserted")
        self.generic_visit(node)

    def visit_While(self, node):
        self._scan(node.test, "branched-on")
        self.generic_visit(node)

    def visit_Call(self, node):
        if self._named(node.func):
            self.sites.append("called")
        self.generic_visit(node)

    def _scan(self, sub, kind):
        for n in ast.walk(sub):
            if isinstance(n, ast.Call) and self._named(n.func):
                self.sites.append(kind)


def probe(pkg_root: Path, target: str) -> Counter:
    c = Counter()
    for p in pkg_root.rglob("*.py"):
        if set(p.parts) & _SKIP:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        v = _CallSites(target)
        v.visit(tree)
        c.update(v.sites)
    return c


def main() -> int:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    hits = d["tier_a"] + d["tier_b_sample"][:120]
    seen, rows = set(), []
    for h in hits:
        fn = h["function"].split(".")[-1]
        if fn.startswith("<") or (h["package"], fn) in seen:
            continue
        seen.add((h["package"], fn))
        root = Path(h["path"])
        while root.parent.name != "site-packages" and root.parent != root:
            root = root.parent
        c = probe(root, fn)
        decided = c["branched-on"] + c["thresholded"] + c["asserted"]
        rows.append((h["package"], fn, c["called"], decided, h in d["tier_a"]))

    for tier, want in (("TIER-A", True), ("TIER-B", False)):
        sel = [r for r in rows if r[4] is want]
        if not sel:
            continue
        with_dec = [r for r in sel if r[3] > 0]
        print(f"\n{tier}: {len(sel)} distinct functions probed")
        print(f"  return value is branched-on / thresholded / asserted somewhere: "
              f"{len(with_dec)}  ({len(with_dec)/len(sel):.0%})")
        print(f"  never used in a decision at all:                              "
              f"{len(sel)-len(with_dec)}")
        for pkg, fn, called, dec, _ in sorted(sel, key=lambda r: -r[3])[:12]:
            print(f"    {pkg:14} {fn[:32]:32} calls={called:4d}  decisions={dec:4d}")
    print("\nEXPLORATORY. No claim is made from this; it selects the mechanism for a")
    print("preregistered v2, it does not test one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
