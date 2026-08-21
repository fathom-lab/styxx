# -*- coding: utf-8 -*-
"""styxx.edges — the defect is not in the function. It is on the edge.

    $ python -m styxx.edges path/to/package

Two screens were built on 2026-08-21 and both were measured against criteria
frozen before they were written. Both fell short, **in opposite directions**:

    styxx.contract    inspects the call BOUNDARY   3/3 boundary-visible, 0/2 interior
    styxx.flattering  inspects the RETURNED VALUE  0 genuine / 19,632 files, 10% recall

`contract` looks at one endpoint. `flattering` looks at the other. The adversarial
reviewers of the flattering run — whose assigned job was to destroy its findings —
named the missing analysis without being asked to design one:

    C1  no consumer-liveness analysis. Kills 6 of 6.
        "Does the symbol have any reader outside its writing scope?"
    C6  the defect class definitionally requires an OUTBOUND measurement that a
        DOWNSTREAM READER misinterprets; a value travelling inward is
        categorically ineligible.

C6 is a definition, and it is not one this project wrote. Read as a claim about
location:

    **A value is not "flattering". A value is flattering TO SOMEONE.**
    `0.0` is a defect only where a consumer thresholds it. With no consumer,
    `risk = 0.0` computed from nothing is a number nobody read.

A finding here is an EDGE `F -> C`, and all five must hold
(`papers/PREREG_edges_2026_08_21.md`, frozen before this file existed):

  1. PRODUCER          F returns a constant K on a path reached by ABSENCE
  2. CONSUMER          a call site uses F's return in a DECISION       <- fixes C1
  3. INDISTINGUISHABLE K is type-identical to F's computed returns, so C cannot
                       separate them even in principle. NaN/None/Measured are
                       DEFENDED and are never findings.
  4. POLARITY FROM THE CONSUMER, never from names                      <- fixes C4
                       K must land on the branch that does not raise, warn, or
                       record a finding. Name morphology inverted on sklearn, and
                       `valid`/`assert`/`check` select for exactly the one-sided
                       predicates where an optimistic empty return is
                       MATHEMATICALLY CORRECT.
  5. CONTRAST          F's computed paths must be able to reach the loud branch.
                       `np.linalg.norm([], inf)` already returns 0.0, so that
                       ternary's branches were numerically identical and could
                       conflate nothing.                               <- fixes C5

Inbound arguments are ineligible by construction (C6): this module only ever
examines a value flowing OUT of a call.

Bare names in test position are disambiguated by USE, not assumed (fixes C2):
`if not x` counts as an absence test only when `x` is used as a container
somewhere in the same function — `len(x)`, iteration, subscript, `sum(x)`. 87% of
the previous screen's candidates rested on `if not <boolean flag>`.

MEASURED YIELD: 0 EDGES ON 227 REAL FILES
─────────────────────────────────────────
Prereg G0 required >= 8 of the 20 corpus cases before this was allowed near
external code. **It caught 0, so no external repository was ever fetched and the
preregistration was terminated** (`papers/RESULT_edges_2026_08_21.md`).

The screen is 3/3 on hand-written true positives and 0/7 on negative controls.
It flagged nothing real because of the limitation declared four paragraphs above:
producer and decision must share a function. **By the corpus's own recorded
consumers that holds for 0 of 20 cases** — every consumer is `ForecastGate`, `the
caller layer`, `ProtocolEnvelope.validate`, somewhere else. The G0 floor was
unachievable by construction, and 0/20 was fixed the moment this design was.

So: not exported from `styxx/__init__.py`, no console script. A research script.
Attempt 2 needs an inter-procedural mechanism and its own preregistration.

RESOLUTION IS REPORTED, AND IT CAN INVALIDATE THE RUN
─────────────────────────────────────────────────────
A cross-function screen that resolves 3% of call sites and reports high precision
is making a statement about nothing. `EdgeReport.resolution` is the fraction of
call sites whose callee was resolved, and prereg G2 makes a run below 25%
`INVALID__BLIND` regardless of what it found.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = ["scan_package", "EdgeReport", "Edge", "Producer"]

# ── vocabulary for "this branch is LOUD" — about CONTROL FLOW, not about names.
# These are things a branch DOES, which is why this list may look name-ish and
# is not: it matches calls and statements, never identifiers being measured.
_LOUD_CALLS = {"warn", "warning", "error", "critical", "exception", "fatal",
               "abort", "fail", "raise_for_status", "exit", "panic", "alarm"}
_LOUD_SINKS = ("finding", "error", "violation", "anomal", "problem", "issue",
               "failure", "warning", "alert", "defect", "breach")
_LOUD_STRINGS = {"fail", "failed", "error", "invalid", "contradicted", "flag",
                 "unsafe", "reject", "denied", "violation", "abort"}


# ── data ───────────────────────────────────────────────────────────────────

@dataclass
class Producer:
    name: str
    path: str
    lineno: int
    absence_returns: List[Tuple[int, Any, str]] = field(default_factory=list)
    n_computed_returns: int = 0
    defended: bool = False           # returns None / NaN / Measured somewhere
    const_return_values: List[Any] = field(default_factory=list)


@dataclass
class Edge:
    producer: str
    producer_path: str
    producer_line: int
    consumer_path: str
    consumer_line: int
    consumer_func: str
    constant: Any
    why_absence: str
    decision: str
    loud_evidence: str
    snippet: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("producer", "producer_path", "producer_line", "consumer_path",
                 "consumer_line", "consumer_func", "constant", "why_absence",
                 "decision", "loud_evidence", "snippet")}


@dataclass
class EdgeReport:
    edges: List[Edge] = field(default_factory=list)
    files_scanned: int = 0
    files_unparsed: int = 0
    n_producers: int = 0
    calls_total: int = 0            # every call site, builtins included
    calls_intra: int = 0            # calls to a name DEFINED somewhere in this package
    calls_resolved: int = 0         # ...and uniquely resolvable to one definition
    producers_dropped: int = 0      # producers discarded as ambiguous names
    decisions_seen: int = 0
    measured: bool = True
    why: Optional[str] = None

    @property
    def resolution(self) -> Optional[float]:
        """Of the calls this screen could in principle resolve — those naming a
        function DEFINED in this package — the fraction it actually did.

        The denominator is intra-package calls, not all calls. `len()`, `print()`
        and every third-party call are not resolution failures; counting them
        would make prereg G2 unpassable, and a gate that can only fail is exactly
        as broken as one that cannot fail. `raw_resolution` is published beside
        it so the choice is checkable rather than convenient.

        None, never 0.0, when there is nothing to resolve — a zero here would
        read as measured blindness rather than as no measurement.
        """
        return (self.calls_resolved / self.calls_intra) if self.calls_intra else None

    @property
    def raw_resolution(self) -> Optional[float]:
        """Resolved over EVERY call site, builtins and third-party included.
        Always the smaller number. Reported so the denominator above cannot be
        mistaken for a flattering choice made after seeing a result."""
        return (self.calls_resolved / self.calls_total) if self.calls_total else None

    def render(self) -> str:
        if not self.measured:
            return f"SCANNED NOTHING - NOT A CLEAN RESULT: {self.why}"
        r, rr = self.resolution, self.raw_resolution
        rs = f"{r:.1%}" if r is not None else "n/a (no intra-package calls)"
        rrs = f"{rr:.1%}" if rr is not None else "n/a"
        out = [f"{self.files_scanned} files ({self.files_unparsed} unparseable), "
               f"{self.n_producers} resolvable producers "
               f"({self.producers_dropped} dropped as ambiguous names)",
               f"call sites {self.calls_total}, intra-package {self.calls_intra}, "
               f"resolved {self.calls_resolved}",
               f"RESOLUTION {rs}   (raw, over all call sites: {rrs})",
               f"decisions on a resolved call: {self.decisions_seen}",
               f"EDGES FLAGGED: {len(self.edges)}"]
        if r is not None and r < 0.25:
            out.append("  G2: resolution < 25% -> INVALID__BLIND. A screen that "
                       "cannot see the edges cannot speak about them.")
        return "\n".join(out)


# ── absence detection, with bare names disambiguated BY USE ────────────────

def _fname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _container_names(fn: ast.AST) -> Set[str]:
    """Names used as containers anywhere in this function.

    This is the fix for the previous screen's dominant false-positive class: 87%
    of its candidates were `if not <boolean flag>`. A name is only treated as
    possibly-empty data if the function itself treats it as data.
    """
    out: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and _fname(n.func) in (
                "len", "sum", "sorted", "list", "tuple", "set", "max", "min",
                "any", "all", "iter", "enumerate"):
            for a in n.args:
                if isinstance(a, ast.Name):
                    out.add(a.id)
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
        elif isinstance(n, (ast.For, ast.comprehension)):
            it = getattr(n, "iter", None)
            if isinstance(it, ast.Name):
                out.add(it.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                and n.attr in ("append", "extend", "items", "keys", "values",
                               "get", "add", "update", "pop"):
            out.add(n.value.id)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and _fname(n.func) == "join" and n.args and isinstance(n.args[0], ast.Name):
            out.add(n.args[0].id)
    return out


def _absence_test(test: ast.AST, containers: Set[str]) -> Optional[Tuple[bool, str]]:
    """(true_when_absent, why) or None if this is not an absence test."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _absence_test(test.operand, containers)
        return (not inner[0], inner[1]) if inner else None
    if isinstance(test, ast.Name):
        # only if the function itself treats this name as data -- see C2
        return (False, f"{test.id} used as a container") if test.id in containers else None
    if isinstance(test, ast.Call) and _fname(test.func) == "len":
        return (False, "len(...) truthiness")
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) \
                and right.value is None:
            return (isinstance(op, ast.Is), f"{_fname(left) or 'value'} is None")
        sizey = None
        if isinstance(left, ast.Call) and _fname(left.func) == "len":
            sizey, other, flip = left, right, False
        elif isinstance(right, ast.Call) and _fname(right.func) == "len":
            sizey, other, flip = right, left, True
        elif isinstance(left, ast.Name) and left.id in containers:
            sizey, other, flip = left, right, False
        if sizey is None or not (isinstance(other, ast.Constant)
                                 and isinstance(other.value, int)
                                 and not isinstance(other.value, bool)):
            return None
        z, kind = other.value, type(op)
        if z not in (0, 1):
            return None
        if kind is ast.Eq:
            return (True, f"len == {z}") if z == 0 else None
        if kind in (ast.Lt, ast.LtE):
            empty = (z == 1 and kind is ast.Lt) or (z == 0 and kind is ast.LtE)
            return ((not empty, f"len bound {z}") if flip else
                    (empty, f"len bound {z}")) if empty else None
        if kind in (ast.Gt, ast.GtE):
            nonempty = (z == 0 and kind is ast.Gt) or (z == 1 and kind is ast.GtE)
            return ((True, f"len > {z}") if flip else (False, f"len > {z}")) \
                if nonempty else None
    return None


def _is_defended(v: Any) -> bool:
    """None and NaN are honest refusals. A consumer CAN distinguish them."""
    if v is None:
        return True
    return isinstance(v, float) and math.isnan(v)


# ── pass 1: index producers ────────────────────────────────────────────────

class _ProducerVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path, self.found = path, []

    def visit_FunctionDef(self, node):
        containers = _container_names(node)
        p = Producer(name=node.name, path=self.path, lineno=node.lineno)

        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                if isinstance(sub.value, ast.Constant):
                    p.const_return_values.append(sub.value.value)
                    if _is_defended(sub.value.value):
                        p.defended = True
                elif isinstance(sub.value, ast.Call) and _fname(sub.value.func) in (
                        "Measured", "NoComputedData"):
                    p.defended = True
                else:
                    p.n_computed_returns += 1

        # (a) an absence guard whose branch returns a constant
        for stmt in node.body:
            if isinstance(stmt, ast.If):
                a = _absence_test(stmt.test, containers)
                if a:
                    when_absent, why = a
                    branch = stmt.body if when_absent else (stmt.orelse or [])
                    for s in branch:
                        if isinstance(s, ast.Return) and isinstance(s.value, ast.Constant) \
                                and not _is_defended(s.value.value):
                            p.absence_returns.append((s.lineno, s.value.value, why))
            # (b) an except handler returning a constant -- crash to sentinel
            elif isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    for s in ast.walk(h):
                        if isinstance(s, ast.Return) and isinstance(s.value, ast.Constant) \
                                and not _is_defended(s.value.value):
                            p.absence_returns.append(
                                (s.lineno, s.value.value, "returned from except handler"))

        # (c) a conditional expression in return position
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.IfExp):
                a = _absence_test(sub.value.test, containers)
                if a:
                    when_absent, why = a
                    fb = sub.value.body if when_absent else sub.value.orelse
                    if isinstance(fb, ast.Constant) and not _is_defended(fb.value):
                        p.absence_returns.append((sub.lineno, fb.value, why))

        if p.absence_returns:
            self.found.append(p)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


# ── pass 2: consumers, and where the quiet branch is ───────────────────────

def _loud_evidence(body: List[ast.AST]) -> str:
    """What this branch DOES that makes it the loud one. '' if it is quiet."""
    for n in body:
        for s in ast.walk(n):
            if isinstance(s, ast.Raise):
                return "raises"
            if isinstance(s, ast.Assert):
                return "asserts"
            if isinstance(s, ast.Call):
                f = _fname(s.func).lower()
                if f in _LOUD_CALLS:
                    return f"calls {f}()"
                if f in ("append", "add") and isinstance(s.func, ast.Attribute):
                    tgt = _fname(s.func.value).lower()
                    if any(k in tgt for k in _LOUD_SINKS):
                        return f"records into {tgt}"
            if isinstance(s, ast.Return) and isinstance(s.value, ast.Constant):
                v = s.value.value
                if isinstance(v, str) and v.strip().lower() in _LOUD_STRINGS:
                    return f"returns {v!r}"
                if v is False:
                    return "returns False"
    return ""


def _resolves_to_quiet(k, test: ast.AST, target: ast.AST,
                       body: List[ast.AST], orelse: List[ast.AST]):
    """Would K take the quiet branch? -> (decision, loud_evidence) or None.

    `target` is the node carrying the producer's value in the test — either the
    call itself, or a local name bound to it earlier in the same function.
    """
    loud_body, loud_else = _loud_evidence(body), _loud_evidence(orelse)

    def verdict(k_takes_body: bool, how: str):
        if k_takes_body and not loud_body and loud_else:
            return (how, f"else-branch {loud_else}")
        if (not k_takes_body) and not loud_else and loud_body:
            return (how, f"if-branch {loud_body}")
        return None

    if test is target:
        return verdict(bool(k), "truthiness")
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) \
            and test.operand is target:
        return verdict(not bool(k), "not-truthiness")
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if left is target:
            other, flip = right, False
        elif right is target:
            other, flip = left, True
        else:
            return None
        if not isinstance(other, ast.Constant):
            return None
        t = other.value
        try:
            kind = type(op)
            if kind is ast.Eq:
                res = (k == t)
            elif kind is ast.NotEq:
                res = (k != t)
            elif isinstance(t, (int, float)) and isinstance(k, (int, float)) \
                    and not isinstance(k, bool) and not isinstance(t, bool):
                a, b = (t, k) if flip else (k, t)
                res = {ast.Gt: a > b, ast.GtE: a >= b,
                       ast.Lt: a < b, ast.LtE: a <= b}[kind]
            else:
                return None
        except Exception:
            return None
        return verdict(bool(res), f"compare {kind.__name__.lower()} {t!r}")
    return None


class _ConsumerScan:
    """Sequential, intra-procedural.

    The first version of this class only saw a producer call written INSIDE the
    if-test. On the styxx tree that indexed 95 producers, resolved 2,602 calls,
    and reached **12 decisions**, because real code writes

        score = compute(x)
        if score > 0.6:

    and that test contains no Call node at all. A screen that cannot follow one
    local assignment cannot see the edge it exists to find. So it tracks
    `name = producer(...)` bindings through the statement list and invalidates a
    name the moment anything else is assigned to it.

    Deliberately shallow, and each limit is a declared blindness rather than an
    approximation: no tuple unpacking, no attribute targets, no cross-function
    propagation, no flow through dataclass fields.
    """

    def __init__(self, path: str, producers: Dict[str, Producer], lines: List[str],
                 defined: Dict[str, int]):
        self.path, self.producers, self.lines = path, producers, lines
        self.defined = defined
        self.edges: List[Edge] = []
        self.stack: List[str] = []
        self.calls_total = self.calls_intra = self.calls_resolved = self.decisions = 0
        self._seen: Set[Tuple[str, int, int]] = set()

    def run(self, tree: ast.AST) -> None:
        for n in ast.walk(tree):                      # call accounting, flat
            if isinstance(n, ast.Call):
                self.calls_total += 1
                cnt = self.defined.get(_fname(n.func))
                if cnt:
                    self.calls_intra += 1
                    if cnt == 1:                      # defined twice is NOT resolved
                        self.calls_resolved += 1
        self._block(getattr(tree, "body", []), {})

    def _producer_of(self, value) -> Optional[Producer]:
        if isinstance(value, ast.Call):
            return self.producers.get(_fname(value.func))
        return None

    def _block(self, stmts, binds: Dict[str, Producer]) -> None:
        for st in stmts or []:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.stack.append(st.name)
                self._block(st.body, {})
                self.stack.pop()
            elif isinstance(st, ast.ClassDef):
                self.stack.append(st.name)
                self._block(st.body, {})
                self.stack.pop()
            elif isinstance(st, ast.Assign):
                prod = self._producer_of(st.value)
                for t in st.targets:
                    if isinstance(t, ast.Name):
                        if prod:
                            binds[t.id] = prod
                        else:
                            binds.pop(t.id, None)
                    else:
                        for sub in ast.walk(t):
                            if isinstance(sub, ast.Name):
                                binds.pop(sub.id, None)
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                prod = self._producer_of(st.value) if st.value is not None else None
                if prod:
                    binds[st.target.id] = prod
                else:
                    binds.pop(st.target.id, None)
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                binds.pop(st.target.id, None)
            elif isinstance(st, (ast.If, ast.While)):
                self._decide(st.test, st.body, st.orelse, binds)
                self._block(st.body, dict(binds))
                self._block(st.orelse, dict(binds))
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                for sub in ast.walk(st.target):
                    if isinstance(sub, ast.Name):
                        binds.pop(sub.id, None)
                self._block(st.body, dict(binds))
                self._block(st.orelse, dict(binds))
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                self._block(st.body, binds)
            elif isinstance(st, ast.Try):
                self._block(st.body, dict(binds))
                for h in st.handlers:
                    self._block(h.body, dict(binds))
                self._block(st.orelse, dict(binds))
                self._block(st.finalbody, dict(binds))

    def _targets(self, test: ast.AST, binds: Dict[str, Producer]):
        """Every node in the test carrying a producer's value."""
        for n in ast.walk(test):
            if isinstance(n, ast.Call):
                p = self.producers.get(_fname(n.func))
                if p:
                    yield n, p
            elif isinstance(n, ast.Name):
                p = binds.get(n.id)
                if p:
                    yield n, p

    def _decide(self, test, body, orelse, binds) -> None:
        for target, p in self._targets(test, binds):
            self.decisions += 1
            if p.n_computed_returns == 0:          # requirement 5 -- CONTRAST
                continue
            for line, k, why in p.absence_returns:
                if _is_defended(k):                # requirement 3
                    continue
                r = _resolves_to_quiet(k, test, target, body, orelse)
                if r is None:
                    continue
                decision, loud = r
                ln = getattr(test, "lineno", 0)
                key = (p.name, line, ln)
                if key in self._seen:
                    continue
                self._seen.add(key)
                i = ln - 1
                self.edges.append(Edge(
                    producer=p.name, producer_path=p.path, producer_line=line,
                    consumer_path=self.path, consumer_line=ln,
                    consumer_func=".".join(self.stack) or "<module>",
                    constant=k, why_absence=why, decision=decision,
                    loud_evidence=loud,
                    snippet=self.lines[i].strip()[:140] if 0 <= i < len(self.lines) else ""))


# ── driver ─────────────────────────────────────────────────────────────────

_SKIP = {".git", "__pycache__", ".tox", ".venv", "node_modules", ".mypy_cache",
         "build", "dist"}


def _files(root: Path, skip_tests: bool) -> List[Path]:
    return [p for p in root.rglob("*.py")
            if not (set(p.parts) & _SKIP)
            and not (skip_tests and ("test" in p.name.lower()
                                     or "tests" in {q.lower() for q in p.parts}))]


def scan_package(root, *, skip_tests: bool = True) -> EdgeReport:
    """Index producers across the whole tree, then find the edges into decisions."""
    root = Path(root)
    rep = EdgeReport()
    files = _files(root, skip_tests)
    if not files:
        rep.measured, rep.why = False, f"no .py files under {root}"
        return rep

    trees: Dict[Path, Tuple[ast.AST, List[str]]] = {}
    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            trees[p] = (ast.parse(src), src.splitlines())
            rep.files_scanned += 1
        except (SyntaxError, OSError, ValueError, RecursionError):
            rep.files_unparsed += 1
    if not trees:
        rep.measured, rep.why = False, "every candidate file failed to parse"
        return rep

    # pass 1 — producers. A name defined more than once is DROPPED: an
    # unresolvable callee is not a resolved one, and guessing is how the previous
    # screen attributed a consumer to the wrong implementation (C7).
    seen: Dict[str, Producer] = {}
    dupes: Set[str] = set()
    defined: Dict[str, int] = {}
    for p, (tree, _) in trees.items():
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined[n.name] = defined.get(n.name, 0) + 1
        v = _ProducerVisitor(str(p))
        v.visit(tree)
        for prod in v.found:
            if prod.name in seen:
                dupes.add(prod.name)
            seen[prod.name] = prod
    producers = {k: v for k, v in seen.items() if k not in dupes}
    rep.n_producers = len(producers)
    rep.producers_dropped = len(dupes)

    # pass 2 — consumers
    for p, (tree, lines) in trees.items():
        c = _ConsumerScan(str(p), producers, lines, defined)
        c.run(tree)
        rep.edges.extend(c.edges)
        rep.calls_total += c.calls_total
        rep.calls_intra += c.calls_intra
        rep.calls_resolved += c.calls_resolved
        rep.decisions_seen += c.decisions
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.edges",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--include-tests", action="store_true")
    a = ap.parse_args(argv)
    rep = scan_package(a.path, skip_tests=not a.include_tests)
    if not rep.measured:
        print(rep.render(), file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps({"summary": rep.render(),
                          "resolution": rep.resolution,
                          "raw_resolution": rep.raw_resolution,
                          "edges": [e.as_dict() for e in rep.edges]},
                         indent=2, ensure_ascii=False))
    else:
        print(rep.render())
        for e in rep.edges:
            print(f"\n  {e.producer}() returns {e.constant!r} when {e.why_absence}")
            print(f"    produced {Path(e.producer_path).name}:{e.producer_line}")
            print(f"    decided  {Path(e.consumer_path).name}:{e.consumer_line} "
                  f"in {e.consumer_func}  [{e.decision}]")
            print(f"    quiet because the {e.loud_evidence}")
            print(f"    | {e.snippet}")
    return 1 if rep.edges else 0


if __name__ == "__main__":
    raise SystemExit(main())
