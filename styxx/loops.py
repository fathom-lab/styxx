# -*- coding: utf-8 -*-
"""styxx.loops — find fields a system derives from its own output, then trusts as truth.

    styxx-loops styxx/                 # screen a package
    styxx-loops my_agent/ --json

The class
─────────
On 2026-08-19 a single contaminated field in this codebase was found to have
corrupted six separate consumers, each in a different way:

    write_audit:  entry["outcome"] = "correct"   # ...when entry["gate"] == "pass"

`gate` is the classifier's verdict. `outcome` is supposed to be GROUND TRUTH —
whether the answer was actually right. Deriving one from the other makes the
system its own grader. Then six consumers read `outcome` believing it:

    calibrate()          shifted the classifier's centroids on those labels
    learned_classifier   trained on (prompt, its own prediction) pairs
    antipatterns         skipped every entry the gate liked — while existing
                         precisely to catch what the gate missed
    weather              divided by them, reporting a 20% warn rate as 100%
    feedback()           walked past them onto an older, unrelated entry
    log()                minted more of them from unmeasured self-reports

Every one of those reads as a small local bug in its own file. None is findable
by reviewing that file. They are findable only by asking a question no linter
asks: **where does this field come from, and who trusts it?**

What this module does
─────────────────────
Two passes and a join.

  1. DERIVATION — an assignment `rec[F] = ...` whose control or data flow
     depends on another field of the SAME record (`rec.get(G)`, `rec[G]`, or a
     local bound from one). That is a field the system computes about itself.
  2. TRUST — anywhere else, code that FILTERS or BRANCHES on `F`
     (`e.get(F) == ...`, `if e[F] in (...)`, a comprehension guard). That is a
     consumer treating `F` as an input rather than as its own echo.
  3. JOIN — report each derived field alongside every site that trusts it.

A hit is not automatically a defect: plenty of derived fields are meant to be
read back (a cached total, a display string). It becomes a defect when the
consumer is CALIBRATING, TRAINING, SCORING or GATING on it — because then the
system is grading itself. The report ranks consumers by that vocabulary and
says which it could not judge.

What it cannot see
──────────────────
  * cross-process flow (a field written here, consumed after a round trip
    through a database, a queue, or another service)
  * derivation through a helper call it cannot follow inter-procedurally
  * whether a given trust site is LEGITIMATE — that is the reader's call, and
    the report is a work list, not a verdict
  * fields carried in objects rather than dict-shaped records
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

__all__ = ["Derivation", "TrustSite", "LoopReport", "scan_path", "LIMITS"]

LIMITS = (
    "intra-procedural and dict-shaped only: blind to cross-process flow, "
    "derivation through helper calls, and object-carried fields; a hit is a "
    "question, not a verdict — plenty of derived fields are meant to be read back."
)

# Consumers where trusting a self-derived field makes the system its own grader.
# A consumer consulting one of these alongside the field can already distinguish
# a self-derived value from a real one — the defence this codebase adopted after
# the 2026-08-19 outcome loop.
_GUARD_HINT = ("_source", "provenance", "_origin", "_by", "author")

_HIGH_STAKES = ("calibrat", "train", "fit", "learn", "score", "gate", "verdict",
                "audit", "grade", "label", "accuracy", "precision", "recall",
                "rate", "prescri", "detect", "anomal", "pattern")


@dataclass
class Derivation:
    """A field the code computes from another field of the same record."""
    field_name: str
    source_field: str
    file: str
    line: int
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return {"field": self.field_name, "derived_from": self.source_field,
                "file": self.file, "line": self.line, "source": self.source}


@dataclass
class TrustSite:
    """A place that filters or branches on a field."""
    field_name: str
    file: str
    line: int
    source: str
    function: str
    high_stakes: bool
    # True when the enclosing function consults a provenance field
    # (outcome_source, *_provenance, *_origin) — i.e. it can already tell a
    # self-derived value from a real one. Credit is FUNCTION-level and therefore
    # OVER-credits: a guard on one path shields sites on another. That bias is
    # deliberate (a screen that keeps flagging fixed code gets ignored) and is
    # the direction that loses findings, not the one that invents them.
    guarded: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"field": self.field_name, "file": self.file, "line": self.line,
                "source": self.source, "function": self.function,
                "high_stakes": self.high_stakes, "guarded": self.guarded}


@dataclass
class LoopReport:
    derivations: List[Derivation] = field(default_factory=list)
    trust_sites: List[TrustSite] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def measured(self) -> bool:
        """False when nothing was scanned — which is not a clean result."""
        return self.files_scanned > 0

    def loops(self) -> Dict[str, Dict[str, Any]]:
        """Derived fields that are also trusted somewhere, worst first."""
        derived: Set[str] = {d.field_name for d in self.derivations}
        out: Dict[str, Dict[str, Any]] = {}
        for name in derived:
            sites = [t for t in self.trust_sites if t.field_name == name]
            if not sites:
                continue
            out[name] = {
                "derivations": [d for d in self.derivations if d.field_name == name],
                "trust_sites": sites,
                "n_high_stakes": sum(1 for s in sites
                                     if s.high_stakes and not s.guarded),
                "n_guarded": sum(1 for s in sites if s.guarded),
            }
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["n_high_stakes"]))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "measured": self.measured,
            "loops": {
                name: {
                    "derivations": [d.as_dict() for d in info["derivations"]],
                    "trust_sites": [s.as_dict() for s in info["trust_sites"]],
                    "n_high_stakes": info["n_high_stakes"],
                    "n_guarded": info["n_guarded"],
                }
                for name, info in self.loops().items()
            },
            "limits": LIMITS,
        }

    def render(self) -> str:
        lines = ["styxx loops — fields derived from the system's own output, then trusted", ""]
        lines.append(f"  files scanned   {self.files_scanned}")
        if not self.measured:
            lines.append("")
            lines.append("  SCANNED NOTHING — not a clean result, no result.")
            return "\n".join(lines)
        found = self.loops()
        if not found:
            lines.append("  closed loops    0")
            lines.append("")
            lines.append("  a clean screen is not a certificate.")
            lines.append(f"  LIMITS: {LIMITS}")
            return "\n".join(lines)
        lines.append(f"  closed loops    {len(found)}")
        lines.append("")
        for name, info in found.items():
            n_hs = info["n_high_stakes"]
            lines.append(f"  FIELD '{name}' — derived from the system's own output, "
                         f"trusted at {len(info['trust_sites'])} site(s), "
                         f"{n_hs} high-stakes")
            for d in info["derivations"]:
                lines.append(f"    derived  {Path(d.file).name}:{d.line}  "
                             f"(from '{d.source_field}')")
                lines.append(f"             {d.source.strip()[:88]}")
            for s in info["trust_sites"]:
                mark = ("ok" if s.guarded else
                        ("!!" if s.high_stakes else "  "))
                lines.append(f"    {mark} trusted {Path(s.file).name}:{s.line}  "
                             f"in {s.function}()")
            lines.append("")
        lines.append("  '!!' = a consumer that CALIBRATES, TRAINS, SCORES or GATES on the")
        lines.append("  field. There the system is grading itself.")
        lines.append(f"  LIMITS: {LIMITS}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if not self.measured:
            return "<LoopReport SCANNED NOTHING — not a clean result>"
        return (f"<LoopReport {len(self.loops())} closed loops across "
                f"{self.files_scanned} files>")


def _const_str(node: ast.AST) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _record_field_reads(node: ast.AST) -> Set[str]:
    """Field names read off a dict-shaped record anywhere under `node`."""
    names: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
           and sub.func.attr == "get" and sub.args:
            k = _const_str(sub.args[0])
            if k:
                names.add(k)
        elif isinstance(sub, ast.Subscript):
            k = _const_str(sub.slice)
            if k:
                names.add(k)
    return names


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: List[str]):
        self.filename = filename
        self.lines = lines
        self.derivations: List[Derivation] = []
        self.trust: List[TrustSite] = []
        self._fn = "<module>"
        self._guarded: Set[str] = set()

    def _src(self, node: ast.AST) -> str:
        i = getattr(node, "lineno", 1) - 1
        return self.lines[i] if 0 <= i < len(self.lines) else ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev, prev_g = self._fn, self._guarded
        self._fn, self._guarded = node.name, self._guarded_fields(node)
        self._scan_function(node)
        self.generic_visit(node)
        self._fn, self._guarded = prev, prev_g

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    @staticmethod
    def _guarded_fields(fn: ast.AST) -> Set[str]:
        """Fields tested ALONGSIDE a provenance field somewhere in this function."""
        guarded: Set[str] = set()
        tests = [n.test for n in ast.walk(fn) if isinstance(n, ast.If)]
        tests += [c for n in ast.walk(fn)
                  if isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp))
                  for g in n.generators for c in g.ifs]
        for t in tests:
            read = _record_field_reads(t)
            if any(any(h in r for h in _GUARD_HINT) for r in read):
                guarded |= {r for r in read
                            if not any(h in r for h in _GUARD_HINT)}

        # Function-level credit. calibrate()'s guard lives in a DIFFERENT
        # comprehension from the field read:
        #     labeled = [e for e in entries if e.get("outcome") in (...)]
        #     usable  = [e for e in labeled  if e.get("outcome_source") != "auto"]
        # Per-test matching calls that unguarded, which means a codebase that
        # HAS fixed the loop keeps getting flagged — the fastest way to teach
        # someone to ignore a screen. Crediting the whole function over-credits
        # (a guard on one path shields sites on another), so this errs toward
        # silence on already-defended code and the report says so.
        all_reads: Set[str] = set()
        has_guard = False
        for sub in ast.walk(fn):
            if isinstance(sub, (ast.Call, ast.Subscript)):
                for r in _record_field_reads(sub):
                    if any(h in r for h in _GUARD_HINT):
                        has_guard = True
                    else:
                        all_reads.add(r)
        if has_guard:
            guarded |= all_reads
        return guarded

    def _scan_function(self, fn: ast.AST) -> None:
        # locals bound from a record field: `gate = entry.get("gate")`
        local_src: Dict[str, str] = {}
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 \
               and isinstance(sub.targets[0], ast.Name):
                reads = _record_field_reads(sub.value)
                if len(reads) == 1:
                    local_src[sub.targets[0].id] = next(iter(reads))

        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            # what does this branch's condition depend on?
            cond_fields = set(_record_field_reads(node.test))
            for nm in (n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)):
                if nm in local_src:
                    cond_fields.add(local_src[nm])
            if not cond_fields:
                continue
            # ...and does the branch write another field of the same record?
            for sub in ast.walk(node):
                if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1):
                    continue
                tgt = sub.targets[0]
                if not isinstance(tgt, ast.Subscript):
                    continue
                written = _const_str(tgt.slice)
                if not written or written in cond_fields:
                    continue
                for src_field in sorted(cond_fields):
                    self.derivations.append(Derivation(
                        field_name=written, source_field=src_field,
                        file=self.filename, line=getattr(sub, "lineno", 0),
                        source=self._src(sub)))
                    break

    def visit_Compare(self, node: ast.Compare) -> None:
        for name in _record_field_reads(node.left):
            self.trust.append(TrustSite(
                field_name=name, file=self.filename,
                line=getattr(node, "lineno", 0), source=self._src(node),
                function=self._fn,
                high_stakes=any(k in self._fn.lower() for k in _HIGH_STAKES),
                guarded=name in self._guarded))
        self.generic_visit(node)


def scan_source(source: str, filename: str = "<string>") -> _Visitor:
    v = _Visitor(filename, source.splitlines())
    v.visit(ast.parse(source))
    return v


def scan_path(path: str | Path, *, skip: Optional[List[str]] = None) -> LoopReport:
    """Screen a file or package for fields derived from a system's own output."""
    p = Path(path)
    skip = skip or ["__pycache__", "/tests/", "/test/"]
    if p.is_file():
        # An explicitly named file is never skipped. The default list excludes
        # /tests/, so `scan_path("tests/test_x.py")` scanned NOTHING and
        # reported it as a clean result -- the same failure styxx.absence had
        # with site-packages. Skip patterns filter a directory WALK; they do not
        # overrule a path the caller named.
        files, skip = [p], []
    else:
        files = sorted(p.rglob("*.py"))
    rep = LoopReport()
    for f in files:
        s = str(f).replace("\\", "/")
        if any(k in s for k in skip):
            continue
        try:
            v = scan_source(f.read_text(encoding="utf-8"), str(f))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        rep.derivations.extend(v.derivations)
        rep.trust_sites.extend(v.trust)
        rep.files_scanned += 1
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.loops",
        description="Find fields a system derives from its own output and then trusts.")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = scan_path(a.path)
    print(json.dumps(rep.as_dict(), indent=2) if a.json else rep.render())
    return 0   # a screen reports; it never fails a build on its own


if __name__ == "__main__":
    raise SystemExit(main())
