"""PROBE E — does a gate's decision term ever take both values on a real population?

The census is a STATIC screen. It reports SHAPES: a decision expression carrying a
term that COULD be constant. It cannot tell a dead gate from a live one sharing the
same syntax -- `census_discrimination_control.py` proved that, and every census number
has since been quoted as an upper bound on the defect rate rather than the rate.

This closes that gap by execution. It rewrites every boolean decision term in the
target package into a recorder, drives the package with its own test suite, and reports
which terms were observed taking only one value.

    a gate is unfalsifiable when any term of its decision expression is constant
    across the population; stuck TRUE in an OR forces pass, stuck FALSE in an AND
    forces silence

The static screen guesses at "constant". This measures it.

WHAT A VERDICT MEANS, EXACTLY. A term reported CONSTANT was constant *under the
population it was driven with* -- here, the repository's own test suite. That is not a
proof it can never vary. It is the strictly weaker and more useful claim: **the test
suite cannot distinguish this gate from one with that term hard-wired.** Which is the
mutation-testing insight, arrived at from the other side: mutation asks whether
changing the code changes the tests' verdict; this asks whether the population ever
changed the gate's inputs. Same question about the same silence.

Population sensitivity is the whole point, so the report always carries n. A term seen
twice is not evidence of anything, and OBS_FLOOR exists so that "constant" is never
printed next to a sample that could not have been otherwise. Terms below the floor are
reported UNDERPOWERED, never as passes -- the failure this program spent the day
catching in four separate instruments, including two of its own.

    python probe_e_runtime.py --pkg ../../styxx --tests ../../tests --json out.json

No network, no model calls. Rewrites ASTs in memory only; nothing on disk is modified.
"""
from __future__ import annotations

import argparse
import ast
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict

# Below this many observations a term's value set says nothing. A term evaluated once
# is trivially "constant" and reporting it as such would manufacture defects out of
# thin coverage -- the same overstatement (more resolution than the method supports)
# this file exists to detect.
OBS_FLOOR = 8

# term_id -> {"vals": set of bools, "n": int, "loc": str, "op": str, "func": str}
_OBS: dict = {}
_META: dict = {}


def _rec(tid, value):
    """Record the truthiness of a decision term, then hand the value straight back.

    Called from rewritten source. Deliberately total: an exception here would corrupt
    the subject's behaviour, and an instrument that changes what it measures is worse
    than no instrument.
    """
    try:
        o = _OBS.get(tid)
        if o is None:
            o = _OBS[tid] = {"t": 0, "f": 0}
        if value:
            o["t"] += 1
        else:
            o["f"] += 1
    except Exception:                                        # noqa: BLE001
        pass
    return value


class _Rewriter(ast.NodeTransformer):
    """Wrap each operand of every boolean decision in a recorder call.

    Only operands of BoolOp are instrumented, plus bare comparisons in `if`/`return`
    position. Wrapping OPERANDS rather than whole expressions preserves short-circuit
    evaluation exactly: the recorder for a right-hand operand only fires when Python
    would have evaluated it. That is not a limitation but a second measurement -- an
    operand that never fires at all is a term the population never reached, which is a
    different and equally disqualifying kind of silence.
    """

    def __init__(self, path, mod):
        self.path, self.mod = path, mod
        self.func = "<module>"
        self.n = 0

    def visit_FunctionDef(self, node):                       # noqa: N802
        prev, self.func = self.func, node.name
        self.generic_visit(node)
        self.func = prev
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _wrap(self, node, op):
        tid = f"{self.mod}:{self.func}:{getattr(node, 'lineno', 0)}:{self.n}"
        self.n += 1
        _META[tid] = {"module": self.mod, "func": self.func, "path": self.path,
                      "line": getattr(node, "lineno", 0), "op": op,
                      "src": _src(node)}
        call = ast.Call(
            func=ast.Name(id="_probe_e_rec", ctx=ast.Load()),
            args=[ast.Constant(value=tid), node], keywords=[])
        return ast.copy_location(call, node)

    def visit_BoolOp(self, node):                            # noqa: N802
        self.generic_visit(node)
        op = "or" if isinstance(node.op, ast.Or) else "and"
        node.values = [self._wrap(v, op) for v in node.values]
        return node

    def visit_If(self, node):                                # noqa: N802
        self.generic_visit(node)
        if isinstance(node.test, (ast.Compare, ast.Call, ast.Name, ast.Attribute,
                                  ast.UnaryOp)):
            node.test = self._wrap(node.test, "if")
        return node

    def visit_Return(self, node):                            # noqa: N802
        self.generic_visit(node)
        if isinstance(node.value, (ast.Compare, ast.UnaryOp)):
            node.value = self._wrap(node.value, "return")
        return node


def _src(node):
    try:
        return ast.unparse(node)[:120]
    except Exception:                                        # noqa: BLE001
        return "<unparseable>"


class _Loader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        # Bind the recorder in the subject's globals before its code runs. A single
        # leading underscore is load-bearing: an identifier beginning with two
        # underscores is private-name-mangled inside any class body, so `__probe_e_rec`
        # would compile to `_SomeClass__probe_e_rec` in every method and raise
        # NameError on the first gate evaluated in a class.
        module.__dict__.setdefault("_probe_e_rec", _rec)
        return super().exec_module(module)

    def get_code(self, fullname):
        """Always compile from source.

        The inherited SourceLoader.get_code consults __pycache__ first and returns
        stale bytecode without ever calling source_to_code -- so the rewriter silently
        never ran and the first real run reported ZERO instrumented terms while the
        suite passed cleanly. A prober that reports nothing found is indistinguishable
        from a prober that is not running, which is the day's bug class wearing yet
        another hat: an instrument whose null result and whose broken state look
        identical from outside. The zero-terms line in the summary is now the tell.
        """
        source = self.get_data(self.get_filename(fullname))
        return self.source_to_code(source, self.get_filename(fullname))

    def source_to_code(self, data, path, *, _optimize=-1):
        try:
            tree = ast.parse(data, filename=path)
        except SyntaxError:
            return super().source_to_code(data, path, _optimize=_optimize)
        mod = os.path.splitext(os.path.basename(path))[0]
        rw = _Rewriter(path, self.name if hasattr(self, "name") else mod)
        tree = rw.visit(tree)
        # The recorder is injected into the module's globals by exec_module, NOT
        # imported. An injected `from probe_e_runtime import _rec` looks equivalent and
        # is not: run as a script this file is `__main__`, so the subject's import
        # created a SECOND copy of the module with its own empty counters. Every
        # observation landed in the copy nobody read, and the run reported 3,349 terms
        # with zero observations -- a clean-looking null result produced by a broken
        # instrument, for the third time today.
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True, optimize=_optimize)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes):
        self.prefixes = tuple(prefixes)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefixes):
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:                                # noqa: BLE001
                continue
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = _Loader(fullname, spec.origin)
                return spec
        return None


def install(prefixes):
    """Instrument every module whose dotted name starts with one of `prefixes`."""
    prefixes = tuple(prefixes)
    # Anything already imported was loaded uninstrumented and would be reused from
    # sys.modules, so its terms would never be recorded and would silently vanish from
    # the denominator rather than showing up as unmeasured.
    for name in [m for m in sys.modules if m.startswith(prefixes)]:
        del sys.modules[name]
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, _Finder(prefixes))


def report(census_path=None):
    """Join what was observed against what the static screen predicted."""
    rows = []
    for tid, o in _OBS.items():
        n = o["t"] + o["f"]
        meta = _META.get(tid, {})
        if n < OBS_FLOOR:
            verdict = "UNDERPOWERED"
        elif o["t"] == 0:
            verdict = "CONSTANT_FALSE"
        elif o["f"] == 0:
            verdict = "CONSTANT_TRUE"
        else:
            verdict = "LIVE"
        rows.append({**meta, "term_id": tid, "n": n, "n_true": o["t"],
                     "n_false": o["f"], "verdict": verdict})

    # terms the rewriter created but the population never evaluated even once
    for tid, meta in _META.items():
        if tid not in _OBS:
            rows.append({**meta, "term_id": tid, "n": 0, "n_true": 0, "n_false": 0,
                         "verdict": "NEVER_REACHED"})

    counts = defaultdict(int)
    for r in rows:
        counts[r["verdict"]] += 1

    powered = [r for r in rows
               if r["verdict"] in ("CONSTANT_TRUE", "CONSTANT_FALSE", "LIVE")]
    dead = [r for r in powered if r["verdict"] != "LIVE"]
    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "obs_floor": OBS_FLOOR,
        "n_terms_instrumented": len(_META),
        "counts": dict(counts),
        "n_powered": len(powered),
        "n_dead_of_powered": len(dead),
        # The only rate this run licenses. Denominator is terms the population actually
        # exercised enough times to answer the question -- NOT every term found, which
        # would silently credit unreached code as healthy.
        "dead_rate_of_powered": (round(len(dead) / len(powered), 4)
                                 if powered else None),
        "scope_note": (
            "CONSTANT means constant UNDER THIS POPULATION (the repository's own test "
            "suite), not constant in principle. The licensed reading is that the suite "
            "cannot distinguish this gate from one with the term hard-wired. "
            "UNDERPOWERED and NEVER_REACHED terms are excluded from the rate and are "
            "not passes."),
        "rows": sorted(rows, key=lambda r: (r["verdict"] != "CONSTANT_TRUE",
                                            r["verdict"] != "CONSTANT_FALSE",
                                            -r["n"])),
    }
    if census_path and os.path.exists(census_path):
        try:
            with open(census_path, encoding="utf-8") as f:
                cen = json.load(f)
            # Two mismatches make the obvious join wrong, and both fail SILENTLY:
            #   1. the census writes "function"; these rows carry "func"
            #   2. the census records the BoolOp's line; the rewriter records each
            #      OPERAND's line, which differs on any multi-line expression
            # Keying on (func, line) would therefore have matched almost nothing and
            # printed a confident zero -- indistinguishable from "the census predicted
            # no real defects". So the join is done at FUNCTION level, which is the
            # granularity both sides actually agree on, and labelled as such rather
            # than dressed up as per-term precision the data cannot support.
            def _key(mod, fn):
                return (os.path.basename(str(mod or "")).replace(".py", "")
                        .split(".")[-1], fn)

            flagged = {_key(c.get("module"), c.get("function") or c.get("func"))
                       for c in cen.get("rows", []) if c.get("n_at_risk")}
            flagged.discard(("", None))
            if not flagged:
                out["census_join"] = {"error": "census file had no at-risk rows"}
                return out
            dead_fns = {_key(r.get("module"), r.get("func")) for r in dead}
            live_fns = {_key(r.get("module"), r.get("func"))
                        for r in rows if r["verdict"] == "LIVE"}
            confirmed = flagged & dead_fns
            # flagged, exercised enough to answer, and every measured term varied
            refuted = (flagged & live_fns) - dead_fns
            out["census_join"] = {
                "granularity": "function (module basename + function name)",
                "n_static_candidate_functions": len(flagged),
                "n_confirmed_dead_by_execution": len(confirmed),
                "n_refuted_all_terms_live": len(refuted),
                "n_unmeasured": len(flagged - dead_fns - live_fns),
                "confirmed": sorted(f"{m}:{f}" for m, f in confirmed)[:60],
                "note": ("Confirmed = the census flagged this function AND execution "
                         "found a constant term in it. Refuted = flagged, exercised, "
                         "every measured term varied. Unmeasured = flagged but the "
                         "population never drove it hard enough to answer -- not a "
                         "pass. The refuted count is the census's false-positive load "
                         "measured directly, which is why its rate was only ever an "
                         "upper bound on the defect rate."),
            }
        except Exception as e:                               # noqa: BLE001
            out["census_join"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def selftest():
    """Point the prober at gates with KNOWN answers before believing it on real code.

    Three fixtures: a gate whose term is wired true, one wired false, and one that
    genuinely varies. An instrument that has never been shown both outcomes on a
    known answer is a hope, and this file's entire argument is that such instruments
    are everywhere -- including in the hands of people who just said so.
    """
    _OBS.clear()
    _META.clear()
    src = (
        "def stuck_true(x):\n"
        "    return ALWAYS or x > 10\n"
        "def stuck_false(x):\n"
        "    return NEVER and x > 10\n"
        "def live(x):\n"
        "    return x > 10 or x < -10\n")
    tree = ast.parse(src)
    rw = _Rewriter("<fixture>", "fixture")
    tree = rw.visit(tree)
    ast.fix_missing_locations(tree)
    ns = {"_probe_e_rec": _rec, "ALWAYS": True, "NEVER": False}
    exec(compile(tree, "<fixture>", "exec"), ns)             # noqa: S102
    # The driving range is part of the fixture, not decoration. The first version swept
    # x over [-10, 9], on which BOTH of `live`'s terms are constant -- and the prober
    # duly reported the live gate as dead. It was right: on that population the gate
    # could not have gone the other way. The bug was the population, which is the whole
    # claim of this file restated as a bug report against its own selftest. The sweep
    # must straddle both thresholds or the control proves nothing.
    for x in range(-20, 20):
        ns["stuck_true"](x)
        ns["stuck_false"](x)
        ns["live"](x)

    rep = report()
    by_func = defaultdict(set)
    for r in rep["rows"]:
        by_func[r["func"]].add(r["verdict"])

    checks = [
        ("stuck_true  -> a CONSTANT_TRUE term", "CONSTANT_TRUE" in by_func["stuck_true"]),
        ("stuck_false -> a CONSTANT_FALSE term", "CONSTANT_FALSE" in by_func["stuck_false"]),
        ("live        -> no constant term", not (
            {"CONSTANT_TRUE", "CONSTANT_FALSE"} & by_func["live"])),
        ("live        -> at least one LIVE term", "LIVE" in by_func["live"]),
        # short-circuit: stuck_true's right operand is never evaluated, so the
        # population never reached it. That must NOT be silently counted healthy.
        ("short-circuited operand reported, not dropped",
         any(r["verdict"] in ("NEVER_REACHED", "UNDERPOWERED")
             for r in rep["rows"] if r["func"] == "stuck_true")),
    ]
    ok = True
    for label, good in checks:
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")
    print(f"\n  PROBE E VALIDATION: "
          f"{'PASS — separates dead gates from live ones on known answers' if ok else 'FAIL'}")
    _OBS.clear()
    _META.clear()
    return 0 if ok else 1


def run_chunked(pkg, tests_dir, out_json, census=None, stack_mb=256, timeout_s=300):
    """Drive the suite one test FILE at a time, each in its own interpreter.

    A single whole-suite process died twice without finishing, and a probe that cannot
    complete a run produces the same output as a probe that found nothing. Chunking
    buys three things: a crash costs one file instead of the run, the crashing files
    become a recorded population gap rather than a silent one, and per-file isolation
    stops one module's global state from colouring another's terms.

    Every chunk's counts merge by term id, which is stable (module:func:line:index)
    because the rewriter walks the same AST every time.
    """
    import glob as _glob                                     # noqa: PLC0415
    import subprocess                                        # noqa: PLC0415
    import tempfile                                          # noqa: PLC0415

    files = sorted(_glob.glob(os.path.join(tests_dir, "**", "test_*.py"),
                              recursive=True))
    tmp = tempfile.mkdtemp(prefix="probe_e_")
    merged_obs, merged_meta = {}, {}
    chunks = []
    for i, tf in enumerate(files, 1):
        cj = os.path.join(tmp, f"c{i:04d}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--pkg", pkg,
               "--tests", tf, "--json", cj, "--stack-mb", str(stack_mb)]
        status, code = "ok", None
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout_s, cwd=os.getcwd())
            code = p.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
        if status == "ok" and not os.path.exists(cj):
            status = f"crash(exit {code})"
        if os.path.exists(cj):
            try:
                with open(cj, encoding="utf-8") as f:
                    r = json.load(f)
                for row in r.get("rows", []):
                    tid = row["term_id"]
                    m = merged_obs.setdefault(tid, {"t": 0, "f": 0})
                    m["t"] += row.get("n_true", 0)
                    m["f"] += row.get("n_false", 0)
                    merged_meta.setdefault(tid, {k: row.get(k) for k in
                                                 ("module", "func", "path", "line",
                                                  "op", "src")})
            except Exception as e:                           # noqa: BLE001
                status = f"unreadable({type(e).__name__})"
        chunks.append({"test_file": os.path.relpath(tf, tests_dir), "status": status})
        print(f"[probe-e] {i}/{len(files)} {os.path.basename(tf):45s} {status}",
              flush=True)

    _OBS.clear()
    _OBS.update(merged_obs)
    _META.clear()
    _META.update(merged_meta)
    rep = report(census)
    bad = [c for c in chunks if c["status"] != "ok"]
    rep["population"] = {
        "driver": "pytest, one process per test file",
        "target": tests_dir,
        "n_test_files": len(files),
        "n_files_failed_to_run": len(bad),
        # A file that never ran contributed no observations, so any term only that file
        # would have exercised is now NEVER_REACHED for a reason that has nothing to do
        # with the code under audit. Naming them keeps that distinction visible instead
        # of letting harness failure masquerade as dead code.
        "files_failed_to_run": bad[:40],
    }
    if out_json:
        with open(out_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1)
    return rep


def _summarise(rep):
    c = rep["counts"]
    pop = rep.get("population", {})
    print(f"\n  terms instrumented : {rep['n_terms_instrumented']}")
    print(f"  powered (n>={OBS_FLOOR})     : {rep['n_powered']}")
    print(f"  CONSTANT_TRUE      : {c.get('CONSTANT_TRUE', 0)}")
    print(f"  CONSTANT_FALSE     : {c.get('CONSTANT_FALSE', 0)}")
    print(f"  LIVE               : {c.get('LIVE', 0)}")
    print(f"  UNDERPOWERED       : {c.get('UNDERPOWERED', 0)}")
    print(f"  NEVER_REACHED      : {c.get('NEVER_REACHED', 0)}")
    print(f"  dead rate (of powered): {rep['dead_rate_of_powered']}")
    if pop.get("n_files_failed_to_run"):
        print(f"  !! test files that failed to run: "
              f"{pop['n_files_failed_to_run']}/{pop.get('n_test_files')} "
              f"— their terms are unmeasured, not healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pkg", help="dotted prefix to instrument, e.g. styxx")
    ap.add_argument("--tests", help="path passed to pytest as the population")
    ap.add_argument("--census", help="census JSON to join against")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--maxfail", type=int, default=0)
    ap.add_argument("--stack-mb", type=int, default=256,
                    help="stack for the pytest thread; instrumentation deepens frames")
    ap.add_argument("--chunked", action="store_true",
                    help="one interpreter per test file; survives a crashing module")
    ap.add_argument("--chunk-timeout", type=int, default=300)
    ap.add_argument("--join-only", metavar="PRIOR_JSON",
                    help="recompute verdicts and the census join from a prior run")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not (a.pkg and a.tests):
        ap.error("--pkg and --tests are required unless --selftest")

    if a.join_only:
        # Re-derive the report (and the census join) from an existing run's rows,
        # without re-driving the suite. Counts are reconstructed from the stored
        # n_true/n_false, so the verdicts are recomputed rather than trusted.
        with open(a.join_only, encoding="utf-8") as f:
            prior = json.load(f)
        for row in prior.get("rows", []):
            _OBS[row["term_id"]] = {"t": row.get("n_true", 0),
                                    "f": row.get("n_false", 0)}
            _META[row["term_id"]] = {k: row.get(k) for k in
                                     ("module", "func", "path", "line", "op", "src")}
        rep = report(a.census)
        rep["population"] = prior.get("population", {})
        rep["population"]["rejoined_from"] = a.join_only
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rep, f, indent=1)
        _summarise(rep)
        cj = rep.get("census_join", {})
        if cj:
            print(f"\n  census candidates (functions): "
                  f"{cj.get('n_static_candidate_functions')}")
            print(f"  confirmed dead by execution  : "
                  f"{cj.get('n_confirmed_dead_by_execution')}")
            print(f"  refuted (all terms live)     : "
                  f"{cj.get('n_refuted_all_terms_live')}")
            print(f"  unmeasured (not a pass)      : {cj.get('n_unmeasured')}")
        return 0

    if a.chunked:
        rep = run_chunked(a.pkg, a.tests, a.json, a.census, a.stack_mb, a.chunk_timeout)
        _summarise(rep)
        return 0

    install([a.pkg])
    import pytest                                            # noqa: PLC0415
    argv = [a.tests, "-q", "--no-header", "-p", "no:cacheprovider"]
    if a.maxfail:
        argv += [f"--maxfail={a.maxfail}"]
    print(f"[probe-e] driving {a.pkg} with {a.tests}", flush=True)
    # Every instrumented term adds a call frame, so the subject's deepest recursion
    # gets deeper by a constant factor and the full suite crashed the interpreter
    # outright on the default stack. Driving pytest from a thread with a large stack
    # keeps the measurement from perturbing the thing measured -- an instrument that
    # changes its subject's failure modes is measuring itself.
    import threading                                         # noqa: PLC0415
    box = {}

    def _run():
        box["code"] = pytest.main(argv)

    try:
        threading.stack_size(a.stack_mb * 1024 * 1024)
    except (ValueError, RuntimeError) as e:                  # noqa: BLE001
        print(f"[probe-e] stack_size({a.stack_mb}MB) refused: {e}", flush=True)
    t = threading.Thread(target=_run)
    t.start()
    t.join()
    code = box.get("code", -1)
    print(f"[probe-e] pytest exit {code}", flush=True)

    rep = report(a.census)
    rep["population"] = {"driver": "pytest", "target": a.tests,
                         "pytest_exit_code": int(code)}
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1)
    _summarise(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
