# -*- coding: utf-8 -*-
"""styxx.absence — find the places where NOT MEASURING reads as a good result.

    styxx-absence styxx/            # screen a package
    styxx-absence my_agent/ --json  # machine-readable

The class
─────────
Across three adversarial audit waves of styxx itself (7.36.0 / 7.37.0 / 7.38.0),
39 defects were confirmed. Nearly all of them were one shape:

    a scoring path that failed, or never ran, and returned a value
    indistinguishable from a healthy measurement.

Concretely, from this repo's own history:

    except Exception:                     # gate.py — an invalid API key
        return GateVerdict(trust_score=1.0, ...)      # ...scored 1.0 of 1.0

    composite=float(entry.get("cogn_composite", 0.0))  # coherence.py — an ABSENT
                                                       # field became a real 0.0
    if denom == 0.0:
        return 0.0                        # coherence.py — r is UNDEFINED here,
                                          # but 0.0 asserts "no relationship"
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0   # dynamics.py — no
                                          # variance to explain became PERFECT
                                          # explained variance
    fired = bool(v.fired or v.needs_revision)   # .fired is a LIST; the
                                          # calibrated term could never decide

None of these raise. None of them log. Every one of them produces a number a
downstream gate happily consumes. That is why they survived years of tests: the
suite asserts what the code returns, and the code returns something plausible.

What this module is
───────────────────
A SCREEN, not a verdict — an AST pass over source you point it at, flagging the
shapes above (plus the sentinel-dataflow form described below). It is deliberately mechanical: it reads code, not
behaviour, so it cannot know whether a given 0.0 is a real measurement or a
fabricated one. It hands you the candidates; you read them.

What it CANNOT see (stated, not buried)
───────────────────────────────────────
  * runtime truthiness — whether `v.fired` is a list, a bool, or None at the
    moment the gate runs. The TRUTHY_GATE rule flags the *shape*, not the type.
  * cross-module call sites — a gate defined honestly here and consumed by a
    looser disjunct three files away is invisible to a per-file pass.
  * dynamic dispatch, getattr chains, and anything built at runtime.
  * whether a flagged default is CORRECT. Plenty of zeros are real zeros. The
    report is a work list, and a clean run is not a certificate.

A screen with no false positives would be a screen that misses the interesting
cases, so this one is tuned to over-report and say so.

Measured recall — 8 of 9 (tests/test_absence.py)
────────────────────────────────────────────────
Characterized against GROUND TRUTH: the defects fixed in 7.36.0-7.38.0, whose
pre-fix source is in this repo's history. The corpus is inlined in the test file
(CI clones shallow), and each case is the real shape, not a toy.

    CAUGHT  witness truthy ConscienceReading    TRUTHY_GATE
    CAUGHT  middleware ceiling_only disjunct    TRUTHY_GATE
    CAUGHT  gate() crash -> trust_score 1.0     HEALTHY_ON_CRASH
    CAUGHT  absent composite -> 0.0             SENTINEL_DEFAULT
    CAUGHT  pearson denom 0 -> 0.0              UNDEFINED_AS_NUMBER
    CAUGHT  r2 = 1.0 on zero variance           UNDEFINED_AS_NUMBER
    CAUGHT  mean_conf fabricated 0.5            UNDEFINED_AS_NUMBER
    CAUGHT  weather crash -> gate:"pass"        CRASH_TO_HEALTHY_SENTINEL
    MISSED  forecast from empty trajectories    (absent guard)

The first pass scored 2 of 9. The gap was ternaries (`r2 = ... if ss_tot > 1e-12
else 1.0` is an IfExp, not an If), operands that are neither Attribute nor Call
(`(not result.needs_revision) or ceiling_only`), and numeric polarity
(trust_score=1.0 is the BEST reading, so a bare "is this constant healthy?"
test could never see it). Fixing those three took it to 7 of 9.

Then the costliest miss was closed. The weather tool swallowed its crash into
`report = None` and returned the healthy `{"gate": "pass"}` LATER, outside the
handler — every call raised (wrong kwarg) and every call reported "pass". The
CRASH_TO_HEALTHY_SENTINEL rule follows a swallowed sentinel within one function
body and connects the two halves: 7 of 9 -> 8 of 9.

One miss remains, and it is the honest ceiling of a pass over source:
  * ABSENT GUARD — forecast() had no validation at all. No screen can flag code
    that was never written; only a test or a reviewer catches that class. It is
    asserted AS a miss in the test suite so the ceiling cannot drift quietly.

Also disclosed: on a freshly hand-audited tree (styxx itself at 7.38.0) this
screen produced 41 candidates and ZERO new true positives — the survivors were
legitimate defaults and one documented display fallback. Its value is on code
that has NOT had 39 defects walked out of it by hand, and as a regression guard
so the class cannot come back quietly.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Finding", "AbsenceReport", "scan_source", "scan_path", "LIMITS"]

LIMITS = (
    "screen, not verdict: reads code not behaviour; blind to runtime types, "
    "cross-module call sites, and dynamic dispatch; a clean run is not a "
    "certificate, and many flagged defaults are legitimate."
)

# Values that read as "everything is fine" when returned from a failure path.
# NOTE the _is_healthy() guard below: `1 in {True}` is True in Python, so a bare
# `return 1` — which in a CLI means FAILURE — matched this set and every CLI
# error path lit up. The screen had the type confusion it exists to find.
_HEALTHY_CONSTS = {True, "pass", "ok", "OK", "healthy", "VERIFIED", "SEALED",
                   "HELD", "OATH-HELD", "clean", "safe"}
# Keys whose healthy value is unambiguous regardless of polarity.
_VERDICT_KEYS = {"gate", "status", "verdict", "trustworthy", "passed", "valid",
                 "healthy", "ok", "measured", "grounded"}
# Names that denote a MEASUREMENT — a default here fabricates a reading.
_MEASUREMENT = ("score", "conf", "confidence", "risk", "composite", "rate",
                "delta", "r2", "auc", "trust", "coherence", "drift", "entropy",
                "margin", "prob", "sigma", "mean", "variance", "corr", "temp")
# Names that denote a GATE DECISION — truthiness here decides whether to act.
_GATE_NAMES = ("fired", "gate", "passed", "needs_revision", "flag", "flags",
               "caught", "halted", "should", "trigger", "blocked", "valid")
# Metrics whose HIGH end reads as healthy (trust 1.0 = perfect) and whose LOW
# end reads as healthy (risk 0.0 = none). A crash path setting either extreme
# fabricates the best possible reading — gate() returned trust_score=1.0 on an
# invalid API key.
_HIGH_IS_HEALTHY = ("trust", "confidence", "conf", "score", "grounded", "r2",
                    "accuracy", "auc", "coherence", "stability")
_LOW_IS_HEALTHY = ("risk", "error", "drift", "deception", "sycophancy",
                   "hallucination", "divergence", "delta", "violation")
# Denominator-ish names whose zero case makes a statistic UNDEFINED.
# Matched EXACTLY against the last dotted segment, never as a substring:
# "n" inside "lines"/"runs"/"entries" made this rule fire on half the repo.
_DENOM = frozenset({"denom", "denominator", "var", "variance", "std", "stdev",
                    "total", "count", "ss_tot", "ss_res", "n", "sigma",
                    "spread", "rng", "span"})


@dataclass
class Finding:
    rule: str
    file: str
    line: int
    source: str
    why: str
    severity: str = "review"     # "review" — this is a screen; humans rank

    def as_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "file": self.file, "line": self.line,
                "source": self.source, "why": self.why, "severity": self.severity}


@dataclass
class AbsenceReport:
    findings: List[Finding] = field(default_factory=list)
    files_scanned: int = 0
    files_unparsed: List[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        """False when the screen scanned NOTHING — which is not a clean result.

        The census run that first used this module skipped every external
        package (site-packages was in the default skip list) and printed
        "candidates 0" for 2.4M lines of torch and transformers. The screen
        built to find measurements-that-never-ran had produced one.
        """
        return self.files_scanned > 0

    def by_rule(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.findings:
            out[f.rule] = out.get(f.rule, 0) + 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"findings": [f.as_dict() for f in self.findings],
                "files_scanned": self.files_scanned,
                "files_unparsed": list(self.files_unparsed),
                "by_rule": self.by_rule(),
                "measured": self.measured,
                "limits": LIMITS}

    def render(self, *, limit: int = 40) -> str:
        lines = ["styxx absence — where not measuring reads as a good result", ""]
        lines.append(f"  files scanned   {self.files_scanned}")
        if self.files_unparsed:
            lines.append(f"  unparsed        {len(self.files_unparsed)}")
        if not self.measured:
            lines.append("")
            lines.append("  SCANNED NOTHING — every candidate file was skipped or")
            lines.append("  unreadable. This is NOT a clean result; it is no result.")
            lines.append("  (Check the skip list: 'site-packages' is skipped by")
            lines.append("  default, so screening an installed package needs an")
            lines.append("  explicit skip= argument.)")
            return "\n".join(lines)
        counts = self.by_rule()
        if not self.findings:
            lines.append("  candidates      0")
            lines.append("")
            # A clean run is exactly when someone over-reads the result, so the
            # limits print HERE too — not only when there is something to show.
            lines.append("  a clean screen is not a certificate.")
            lines.append(f"  LIMITS: {LIMITS}")
            return "\n".join(lines)
        lines.append(f"  candidates      {len(self.findings)}")
        for rule, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {rule:<18} {n}")
        lines.append("")
        shown = self.findings if limit <= 0 else self.findings[:limit]
        for f in shown:
            lines.append(f"  {f.file}:{f.line}  [{f.rule}]")
            lines.append(f"    {f.source.strip()[:96]}")
            lines.append(f"    -> {f.why}")
        if limit > 0 and len(self.findings) > limit:
            lines.append(f"  ... {len(self.findings) - limit} more (--json for all)")
        lines.append("")
        lines.append(f"  LIMITS: {LIMITS}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if not self.measured:
            return "<AbsenceReport SCANNED NOTHING — not a clean result>"
        return (f"<AbsenceReport {len(self.findings)} candidates across "
                f"{self.files_scanned} files>")


def _name_of(node: ast.AST) -> str:
    """Best-effort dotted name for a node, for vocabulary matching."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return ""


def _is_healthy(value: Any) -> bool:
    """True only for a real bool True or a healthy STRING.

    Bare numbers are excluded on purpose: their polarity depends on the metric
    (trust 1.0 is healthy, risk 1.0 is not), and int/bool equality made `1`
    match `True`.
    """
    if isinstance(value, bool):
        return value is True
    if isinstance(value, str):
        return value in _HEALTHY_CONSTS
    return False


def _is_best_case_number(name: str, value: Any) -> bool:
    """A numeric extreme in the healthy direction for this metric's polarity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    low = name.lower()
    if any(v in low for v in _HIGH_IS_HEALTHY) and float(value) >= 0.9:
        return True
    if any(v in low for v in _LOW_IS_HEALTHY) and float(value) == 0.0:
        return True
    return False


def _matches(name: str, vocab) -> bool:
    low = name.lower()
    return any(v in low for v in vocab)


def _is_denom(name: str) -> bool:
    """Exact match on the trailing identifier segment (see _DENOM)."""
    return name.lower().split(".")[-1] in _DENOM


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: List[str]):
        self.filename = filename
        self.lines = lines
        self.findings: List[Finding] = []

    def _src(self, node: ast.AST) -> str:
        i = getattr(node, "lineno", 1) - 1
        return self.lines[i] if 0 <= i < len(self.lines) else ""

    def _add(self, rule: str, node: ast.AST, why: str) -> None:
        self.findings.append(Finding(rule=rule, file=self.filename,
                                     line=getattr(node, "lineno", 0),
                                     source=self._src(node), why=why))

    @staticmethod
    def _healthy_reason(v: ast.AST) -> Optional[str]:
        """Describe why this returned value reads as 'everything is fine', or None."""
        if isinstance(v, ast.Constant) and _is_healthy(v.value):
            return f"returns {v.value!r}"
        if isinstance(v, ast.Dict):
            for k, val in zip(v.keys, v.values):
                kn = _name_of(k) if k is not None else ""
                if (kn.lower() in _VERDICT_KEYS and isinstance(val, ast.Constant)
                        and _is_healthy(val.value)):
                    return f"returns {kn}={val.value!r}"
        if isinstance(v, ast.Call):
            for kw in v.keywords or []:
                if not (kw.arg and isinstance(kw.value, ast.Constant)):
                    continue
                if kw.arg.lower() in _VERDICT_KEYS and _is_healthy(kw.value.value):
                    return f"constructs {kw.arg}={kw.value.value!r}"
                if _is_best_case_number(kw.arg, kw.value.value):
                    return (f"constructs {kw.arg}={kw.value.value}, the BEST possible "
                            f"reading for that metric")
        return None

    # ── rule 1: a failure path that returns a healthy-looking value ──────────
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Return) or sub.value is None:
                continue
            reason = self._healthy_reason(sub.value)
            if reason:
                self._add("HEALTHY_ON_CRASH", sub,
                          f"the failure path {reason} — a caller cannot tell this "
                          f"crash from a real result")
        self.generic_visit(node)

    # ── rule 5: a crash swallowed into a sentinel, healthy on the sentinel ──
    # The costliest instance in this repo's history was NOT inside its handler:
    #     try: report = styxx.weather(window=window)
    #     except Exception: report = None          # crash swallowed here...
    #     if report is None:
    #         return {"summary": "...", "gate": "pass"}   # ...and healthy HERE.
    # Every call raised (wrong kwarg) and every call reported "pass". Catching
    # this needs the two halves connected, so this rule follows the sentinel
    # name within one function body.
    def _check_sentinel_dataflow(self, fn: ast.AST) -> None:
        swallowed = {}
        for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
            for sub in ast.walk(h):
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                        and isinstance(sub.value, ast.Constant)):
                    swallowed[sub.targets[0].id] = sub.value.value
        if not swallowed:
            return
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            tested = set()
            for t in ast.walk(node.test):
                if isinstance(t, ast.Name) and t.id in swallowed:
                    tested.add(t.id)
            if not tested:
                continue
            for st in node.body:
                if not isinstance(st, ast.Return) or st.value is None:
                    continue
                reason = self._healthy_reason(st.value)
                if reason:
                    name = sorted(tested)[0]
                    self._add("CRASH_TO_HEALTHY_SENTINEL", st,
                              f"{name} is set to {swallowed[name]!r} by an except "
                              f"handler above, and this branch {reason} — so a "
                              f"swallowed crash and a real result are the same value "
                              f"to the caller")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_sentinel_dataflow(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._check_sentinel_dataflow(node)
        self.generic_visit(node)

    # ── rule 2: a missing MEASUREMENT defaulted to a number ─────────────────
    def visit_Call(self, node: ast.Call) -> None:
        # A verdict keyword whose value is `A if <obj> else B` — the branch is
        # chosen by an OBJECT's truthiness, so a dataclass/list decides the
        # verdict regardless of what it measured (witness.substrate_divergence
        # picked "FLAG" vs "OK" off an always-truthy ConscienceReading).
        for kw in node.keywords or []:
            if not (kw.arg and kw.arg.lower() in _VERDICT_KEYS):
                continue
            if isinstance(kw.value, ast.IfExp) and isinstance(
                    kw.value.test, (ast.Name, ast.Attribute)):
                self._add("TRUTHY_GATE", kw.value,
                          f"{kw.arg} is chosen by the truthiness of "
                          f"{_name_of(kw.value.test)!r} — an object or non-empty "
                          f"container is always true, so one branch may be dead")

        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and len(node.args) == 2):
            key, default = node.args
            kn = _name_of(key)
            if (kn and _matches(kn, _MEASUREMENT)
                    and isinstance(default, ast.Constant)
                    and isinstance(default.value, (int, float))
                    and not isinstance(default.value, bool)):
                self._add("SENTINEL_DEFAULT", node,
                          f"an ABSENT {kn!r} becomes {default.value} — a missing "
                          f"measurement enters the maths as a real reading")
        self.generic_visit(node)

    # ── rule 3: an UNDEFINED statistic returned as a number ─────────────────
    def visit_If(self, node: ast.If) -> None:
        test_name = ""
        degenerate = False
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            test_name = _name_of(t.left)
            op = t.ops[0]
            comp = t.comparators[0]
            if isinstance(op, (ast.Eq, ast.Lt, ast.LtE)) and isinstance(comp, ast.Constant) \
               and isinstance(comp.value, (int, float)) and float(comp.value) <= 1e-6:
                degenerate = True
        elif isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            test_name = _name_of(t.operand)
            degenerate = True
        if degenerate and test_name and _is_denom(test_name):
            for sub in node.body:
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant) \
                   and isinstance(sub.value.value, (int, float)) \
                   and not isinstance(sub.value.value, bool):
                    self._add("UNDEFINED_AS_NUMBER", sub,
                              f"{test_name} is degenerate, so the statistic is "
                              f"UNDEFINED — returning {sub.value.value} asserts a "
                              f"result instead of refusing")
        self.generic_visit(node)

    # ── rules 2+3 in ternary form: `X = <expr> if <guard> else <number>` ────
    # The If-statement forms were covered; the one-liners were not, and both of
    # this repo's worst instances were one-liners:
    #   r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0   (dynamics)
    #   mean_conf = sum(c)/len(c) if confs else 0.5           (sla)
    def _check_ifexp(self, target: str, node: ast.IfExp) -> None:
        orelse = node.orelse
        if not (isinstance(orelse, ast.Constant)
                and isinstance(orelse.value, (int, float))
                and not isinstance(orelse.value, bool)):
            return
        t = node.test
        guard = ""
        degenerate = False
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            guard = _name_of(t.left)
            comp = t.comparators[0]
            if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float))                and float(comp.value) <= 1e-6:
                degenerate = True
        elif isinstance(t, (ast.Name, ast.Attribute, ast.Call)):
            guard = _name_of(t)          # bare truthiness: `... if confs else 0.5`
            degenerate = True
        if not degenerate:
            return
        if _is_denom(guard) or (guard and _matches(guard, _MEASUREMENT)):
            self._add("UNDEFINED_AS_NUMBER", node,
                      f"when {guard} is empty/zero the statistic is UNDEFINED — the "
                      f"else-branch returns {orelse.value}, asserting a result")
        elif target and _matches(target, _MEASUREMENT):
            self._add("SENTINEL_DEFAULT", node,
                      f"{target} falls back to {orelse.value} when {guard or 'the guard'} "
                      f"is empty — an unmeasured value entering the maths as a reading")

    # ── rule 4: a gate decided by truthiness / a subsuming disjunct ─────────
    def visit_Assign(self, node: ast.Assign) -> None:
        target = _name_of(node.targets[0]) if node.targets else ""
        if isinstance(node.value, ast.IfExp):
            self._check_ifexp(target, node.value)
        if target and _matches(target, _GATE_NAMES):
            v = node.value
            inner = v.args[0] if (isinstance(v, ast.Call) and _name_of(v.func) == "bool"
                                  and v.args) else v
            if isinstance(inner, ast.BoolOp) and isinstance(inner.op, ast.Or):
                dynamic = [x for x in inner.values
                           if not isinstance(x, ast.Constant)]
                # Name and `not x` count: middleware's dead disjunct was
                # `passed = (not result.needs_revision) or ceiling_only` —
                # neither operand is a bare Attribute.
                attrs = [x for x in dynamic
                         if isinstance(x, (ast.Attribute, ast.Call, ast.Name,
                                           ast.UnaryOp))]
                # `a or "literal"` is a default, not a decision between two
                # signals — only flag when 2+ live operands compete.
                if attrs and len(dynamic) >= 2:
                    self._add("TRUTHY_GATE", node,
                              f"{target} is decided by an OR over attribute values — if "
                              f"either is a list/str/object it is truthy when non-empty, "
                              f"and a calibrated term beside it can never change the "
                              f"outcome")
            elif isinstance(inner, (ast.Attribute,)) and isinstance(v, ast.Call):
                self._add("TRUTHY_GATE", node,
                          f"{target} is bool() of an attribute — a dataclass, list or "
                          f"str is truthy regardless of what it measured")
        self.generic_visit(node)


def scan_source(source: str, filename: str = "<string>") -> List[Finding]:
    """Screen one module's source. Raises SyntaxError on unparseable input."""
    tree = ast.parse(source)
    v = _Visitor(filename, source.splitlines())
    v.visit(tree)
    return v.findings


def scan_path(path: str | Path, *, skip: Optional[List[str]] = None) -> AbsenceReport:
    """Screen a file or a package directory (recursively, *.py)."""
    p = Path(path)
    skip = skip or ["__pycache__", ".venv", "site-packages", "/build/", "/dist/"]
    files = [p] if p.is_file() else sorted(p.rglob("*.py"))
    report = AbsenceReport()
    for f in files:
        s = str(f).replace("\\", "/")
        if any(k in s for k in skip):
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            report.files_unparsed.append(str(f))
            continue
        try:
            report.findings.extend(scan_source(src, str(f)))
        except SyntaxError:
            report.files_unparsed.append(str(f))
            continue
        report.files_scanned += 1
    if not report.files_scanned:
        import warnings
        warnings.warn(
            f"styxx.absence: scanned NOTHING under {p} — every file was skipped "
            f"or unreadable, so 0 candidates means 'no result', not 'clean'. "
            f"The default skip list excludes site-packages; pass skip=[...] to "
            f"screen an installed package.", RuntimeWarning, stacklevel=2)
    return report


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="styxx.absence",
        description="Screen source for places where NOT MEASURING reads as a good result.")
    ap.add_argument("path", help="a .py file or a package directory")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--rule", default=None, help="only this rule")
    ap.add_argument("--limit", type=int, default=40,
                    help="max findings to print (0 = all)")
    a = ap.parse_args(argv)

    rep = scan_path(a.path)
    if a.rule:
        rep.findings = [f for f in rep.findings if f.rule == a.rule]
    if a.json:
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        print(rep.render(limit=a.limit))
    # A screen never fails a build on its own: it reports, humans rank. Exit 0
    # even with findings, so nobody is tempted to silence it to go green.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
