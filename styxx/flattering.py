# -*- coding: utf-8 -*-
"""styxx.flattering — find the empty case that returns the reassuring constant.

    $ styxx-flattering path/to/package

The signature this looks for
────────────────────────────
SILENT-PASS exists because **the inert default and the flattering default are
often the same value**: risk 0.0, trust 1.0, gate "pass". `styxx.absence` looks
for a guard that was never written. This module looks for the opposite and much
crisper case — **the author DID notice the empty input, and chose a flattering
constant for it.**

    return n_hot / n if n > 0 else 0.0        # a calm truth map of nothing
    if not samples: return 0.0                # "one cluster: the model knows"
    if n == 0:      return 0.0, 0, 0          # an injection detector that never ran

Syntactically these are guards. Semantically they are the defect, because a
consumer that thresholds the value cannot distinguish *"measured, and it is
fine"* from *"there was nothing to measure"*.

Polarity is required, and that is deliberate
────────────────────────────────────────────
`0.0` is only flattering if low is the good end. Without evidence of polarity —
from the function name, the variable being assigned, or the keyword it fills —
this module makes **no claim**. Those hits are counted as TIER-B and reported
separately, never as findings. Precision over recall: a screen that cries wolf
gets deleted, and a deleted screen catches nothing.

    TIER-A  emptiness guard + flattering constant + polarity evidence  -> claimed
    TIER-B  emptiness guard + constant, no polarity evidence           -> counted only

An honest refusal is never a hit: `return float("nan")`, `raise`, and returning
a `Measured` that knows it is unmeasured are all correct handling of an empty
input, and none of them match.

Provenance: these rules were written against the styxx corpus in
`benchmarks/silent_pass` and frozen before this screen was pointed at any code
this project did not write.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["scan_path", "scan_source", "Hit", "FlatteringReport"]

_LOW_IS_GOOD = ("risk", "error", "drift", "divergence", "entropy", "deception",
                "hallucinat", "confabulat", "violation", "uncertaint", "anomal",
                "failure", "loss", "penalt", "toxic", "leak", "regret", "defect")
_HIGH_IS_GOOD = ("confidence", "trust", "grounded", "reliabilit", "accuracy",
                 "auc", "coherence", "agreement", "stabilit", "precision",
                 "recall", "fidelit", "safet", "health", "quality", "score")
_OK_BOOL = ("valid", "ok", "safe", "healthy", "pass", "clean", "verified",
            "consistent", "success", "allow", "grant", "trusted", "compliant")
_OK_STR = {"pass", "ok", "healthy", "valid", "clean", "safe", "steady",
           "verified", "sealed", "normal", "good", "success"}


@dataclass
class Hit:
    path: str
    line: int
    function: str
    tier: str
    rule: str
    polarity_from: Optional[str]
    snippet: str

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in
                ("path", "line", "function", "tier", "rule", "polarity_from", "snippet")}


@dataclass
class FlatteringReport:
    hits: List[Hit] = field(default_factory=list)
    files_scanned: int = 0
    files_unparsed: int = 0
    measured: bool = True
    why: Optional[str] = None

    @property
    def tier_a(self) -> List[Hit]:
        return [h for h in self.hits if h.tier == "A"]

    @property
    def tier_b(self) -> List[Hit]:
        return [h for h in self.hits if h.tier == "B"]

    def render(self) -> str:
        if not self.measured:
            return f"SCANNED NOTHING - NOT A CLEAN RESULT: {self.why}"
        return "\n".join([
            f"{self.files_scanned} files scanned ({self.files_unparsed} unparseable)",
            f"TIER-A (polarity evidence, claimed):   {len(self.tier_a)}",
            f"TIER-B (structural only, not claimed): {len(self.tier_b)}"])


def _polarity(name: Optional[str]) -> Optional[str]:
    """'low_is_good' | 'high_is_good' | 'ok_bool' | None (no claim)."""
    if not name:
        return None
    n = name.lower()
    if any(k in n for k in _LOW_IS_GOOD):
        return "low_is_good"
    if any(k in n for k in _HIGH_IS_GOOD):
        return "high_is_good"
    for k in _OK_BOOL:
        if n == k or n.startswith(k + "_") or n.startswith("is_" + k) or f"_{k}" in n:
            return "ok_bool"
    return None


def _is_flattering(node: ast.AST, pol: str) -> bool:
    """Does this constant read as the good news, given the polarity?"""
    if isinstance(node, ast.Tuple):
        return bool(node.elts) and all(_is_flattering(e, pol) for e in node.elts)
    if not isinstance(node, ast.Constant):
        return False
    v = node.value
    if isinstance(v, bool):
        return v is True and pol == "ok_bool"
    if isinstance(v, (int, float)):
        if pol == "low_is_good":
            return float(v) == 0.0
        if pol == "high_is_good":
            return float(v) == 1.0
        return False
    if isinstance(v, str):
        return v.strip().lower() in _OK_STR
    return False


def _fname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _looks_sizey(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and _fname(node.func) == "len":
        return True
    if isinstance(node, ast.Name):
        n = node.id.lower()
        return n in ("n", "count", "total", "size", "length") or n.startswith("n_") \
            or n.endswith(("_n", "_count", "_len", "_size"))
    if isinstance(node, ast.Attribute):
        return node.attr.lower() in ("size", "n", "count", "length")
    return False


def _emptiness(test: ast.AST) -> Optional[bool]:
    """True  -> the test is true WHEN EMPTY   (`not xs`, `n == 0`, `len(x) < 1`)
       False -> the test is true WHEN NON-EMPTY (`xs`, `n > 0`)
       None  -> not an emptiness test at all.

    Anything ambiguous returns None. A screen that guesses is a screen that
    manufactures findings.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _emptiness(test.operand)
        return None if inner is None else (not inner)
    if isinstance(test, ast.Name):
        return False
    if isinstance(test, ast.Call) and _fname(test.func) == "len":
        return False
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if _looks_sizey(left):
            _subj, other, flip = left, right, False
        elif _looks_sizey(right):
            _subj, other, flip = right, left, True
        else:
            return None
        if not (isinstance(other, ast.Constant) and isinstance(other.value, int)
                and not isinstance(other.value, bool) and other.value in (0, 1)):
            return None
        z, kind = other.value, type(op)
        if kind is ast.Eq:
            return z == 0
        if kind is ast.NotEq:
            return False if z == 0 else None
        if kind in (ast.Gt, ast.GtE):
            nonempty = (z == 0 and kind is ast.Gt) or (z == 1 and kind is ast.GtE)
            if not nonempty:
                return None
            return True if flip else False
        if kind in (ast.Lt, ast.LtE):
            empty = (z == 1 and kind is ast.Lt) or (z == 0 and kind is ast.LtE)
            if not empty:
                return None
            return False if flip else True
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, src_lines: List[str]):
        self.path, self.lines, self.hits = path, src_lines, []
        self.stack: List[str] = []

    def _snip(self, node: ast.AST) -> str:
        i = getattr(node, "lineno", 1) - 1
        return self.lines[i].strip()[:150] if 0 <= i < len(self.lines) else ""

    def _record(self, node, rule, pol_from):
        self.hits.append(Hit(
            path=self.path, line=getattr(node, "lineno", 0),
            function=".".join(self.stack) or "<module>",
            tier="A" if pol_from else "B", rule=rule,
            polarity_from=pol_from, snippet=self._snip(node)))

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        # R1: an emptiness guard among the first statements that returns the
        #     flattering constant instead of refusing
        for stmt in node.body[:4]:
            if not isinstance(stmt, ast.If):
                continue
            when_empty = _emptiness(stmt.test)
            if when_empty is None:
                continue
            branch = stmt.body if when_empty else (stmt.orelse or [])
            for s in branch:
                if isinstance(s, ast.Return) and s.value is not None:
                    self._check_const(s.value, node.name,
                                      "R1_empty_guard_returns_constant", s)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        if isinstance(node.value, ast.IfExp):
            tgt = _fname(node.targets[0]) if node.targets else None
            self._check_ifexp(node.value, tgt)
        self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.IfExp):
            self._check_ifexp(node.value, self.stack[-1] if self.stack else None)
        self.generic_visit(node)

    def visit_keyword(self, node):
        if isinstance(node.value, ast.IfExp):
            self._check_ifexp(node.value, node.arg)
        self.generic_visit(node)

    def _check_ifexp(self, node: ast.IfExp, name_hint: Optional[str]):
        when_empty = _emptiness(node.test)
        if when_empty is None:
            return
        fallback = node.body if when_empty else node.orelse
        self._check_const(fallback, name_hint, "R2_ifexp_empty_fallback_constant", node)

    def _check_const(self, value_node, name_hint, rule, at):
        # A healthy STRING carries its own polarity. `return "pass"` on an empty
        # input is the flattering default whatever the function is called, so it
        # needs no name evidence -- the inverse of the bare-float case, where an
        # unnamed number can never be judged. (SP-2026-0012 returned "steady".)
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str) \
                and value_node.value.strip().lower() in _OK_STR:
            return self._record(at, rule, f"literal:{value_node.value!r}")

        seen = []
        for hint, src in ((name_hint, "name"),
                          (self.stack[-1] if self.stack else None, "function")):
            if not hint or hint in seen:
                continue
            seen.append(hint)
            pol = _polarity(hint)
            if pol and _is_flattering(value_node, pol):
                return self._record(at, rule, f"{src}:{hint}")
        # no polarity evidence — structural only, counted but never claimed
        if isinstance(value_node, ast.Constant) and not isinstance(value_node.value, str) \
                and value_node.value in (0, 0.0, True, 1, 1.0):
            self._record(at, rule, None)


def scan_source(src: str, path: str = "<string>") -> List[Hit]:
    v = _Visitor(path, src.splitlines())
    v.visit(ast.parse(src))
    return v.hits


_SKIP_DIRS = {".git", "__pycache__", ".tox", ".venv", "node_modules", ".mypy_cache"}


def scan_path(root, *, skip_tests: bool = True) -> FlatteringReport:
    """Scan a tree. `measured` is False when nothing was actually read — a zero
    from an empty scan is the very defect this module screens for."""
    root = Path(root)
    rep = FlatteringReport()
    files = [p for p in root.rglob("*.py")
             if not (set(p.parts) & _SKIP_DIRS)
             and not (skip_tests and ("test" in p.name.lower()
                                      or "tests" in {q.lower() for q in p.parts}))]
    if not files:
        rep.measured, rep.why = False, f"no .py files under {root}"
        return rep
    for p in files:
        try:
            rep.hits.extend(scan_source(
                p.read_text(encoding="utf-8", errors="replace"), str(p)))
            rep.files_scanned += 1
        except (SyntaxError, OSError, ValueError, RecursionError):
            rep.files_unparsed += 1
    if rep.files_scanned == 0:
        rep.measured, rep.why = False, "every candidate file failed to parse"
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="styxx-flattering",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tier", default="A", choices=["A", "B", "AB"])
    ap.add_argument("--include-tests", action="store_true")
    a = ap.parse_args(argv)
    rep = scan_path(a.path, skip_tests=not a.include_tests)
    if not rep.measured:
        print(rep.render(), file=sys.stderr)
        return 2
    sel = rep.tier_a if a.tier == "A" else rep.tier_b if a.tier == "B" else rep.hits
    if a.json:
        print(json.dumps({"summary": rep.render(),
                          "hits": [h.as_dict() for h in sel]}, indent=2, ensure_ascii=False))
    else:
        print(rep.render())
        for h in sel:
            print(f"  {h.path}:{h.line}  {h.function}  [{h.tier}] {h.polarity_from or '-'}")
            print(f"      {h.snippet}")
    return 1 if rep.tier_a else 0


if __name__ == "__main__":
    raise SystemExit(main())
