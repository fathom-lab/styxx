# -*- coding: utf-8 -*-
"""ATTAINABILITY CHECK — can the attempt-2 gate be passed at all?

Attempt 1 froze a floor of `>= 8 of 20` that was unachievable by construction:
its mechanism required producer and consumer in the same function, which held for
0 of 20 cases. `0/20` was determined before any data existed.

    A gate that cannot fail certifies noise.
    A gate that cannot pass certifies nothing.
    Both are gates whose outcome was fixed before the data arrived.

So, before `PREREG_edges2_2026_08_21.md` is frozen: for each of the three hops
the mechanism specifies, take a REAL corpus case of that shape, load the REAL
pre-fix source, and demonstrate that the producer, the hop, and a deciding
consumer all exist and are matchable by the rule as written.

This does not test the instrument -- the instrument does not exist yet. It tests
whether the gate is answerable.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def pkg_files(commit: str) -> dict[str, str]:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", f"{commit}~1", "styxx"],
                           cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8").stdout.split()
    out = {}
    for n in names:
        if n.endswith(".py"):
            r = subprocess.run(["git", "show", f"{commit}~1:{n}"], cwd=REPO,
                               capture_output=True, text=True, encoding="utf-8")
            if r.returncode == 0:
                out[n] = r.stdout
    return out


def loud(body) -> str:
    for n in body or []:
        for s in ast.walk(n):
            if isinstance(s, ast.Raise):
                return "raises"
            if isinstance(s, ast.Call):
                f = getattr(s.func, "id", None) or getattr(s.func, "attr", "")
                if f.lower() in ("warn", "warning", "error", "critical", "exit", "fail"):
                    return f"calls {f}()"
            if isinstance(s, ast.Return) and isinstance(s.value, ast.Constant):
                v = s.value.value
                if v is False or (isinstance(v, str) and v.lower() in
                                  ("fail", "error", "invalid", "contradicted", "flag")):
                    return f"returns {v!r}"
    return ""


def find_attr_decisions(files: dict[str, str], attr: str):
    """Anywhere a decision is made on `<something>.attr`."""
    hits = []
    for path, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.If, ast.While)):
                continue
            for sub in ast.walk(n.test):
                if isinstance(sub, ast.Attribute) and sub.attr == attr:
                    hits.append((path, n.lineno, loud(n.body), loud(n.orelse),
                                 src.splitlines()[n.lineno - 1].strip()[:88]))
    return hits


def find_key_decisions(files: dict[str, str], key: str):
    """Anywhere a decision is made on `<something>["key"]`."""
    hits = []
    for path, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.If, ast.While, ast.Assert)):
                continue
            test = n.test if not isinstance(n, ast.Assert) else n.test
            for sub in ast.walk(test):
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) \
                        and sub.slice.value == key:
                    hits.append((path, n.lineno,
                                 loud(getattr(n, "body", [])), loud(getattr(n, "orelse", [])),
                                 src.splitlines()[n.lineno - 1].strip()[:88]))
            if isinstance(n, ast.Assert):
                continue
    return hits


def find_call_then_decide(files: dict[str, str], fname: str):
    """A caller that binds `x = fname(...)` and then decides on x."""
    hits = []
    for path, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            bound = set()
            for st in ast.walk(fn):
                if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
                    nm = getattr(st.value.func, "id", None) or \
                         getattr(st.value.func, "attr", "")
                    if nm == fname:
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                bound.add(t.id)
            if not bound:
                continue
            for st in ast.walk(fn):
                if isinstance(st, (ast.If, ast.While)):
                    for sub in ast.walk(st.test):
                        if isinstance(sub, ast.Name) and sub.id in bound:
                            hits.append((path, st.lineno, loud(st.body), loud(st.orelse),
                                         lines[st.lineno - 1].strip()[:88]))
    return hits


HOPS = [
    ("FIELD hop", "SP-2026-0012", "5dd3949",
     "confabulation_ratio= into TruthMap(...), read as r.confabulation_ratio",
     lambda f: find_attr_decisions(f, "confabulation_ratio")),
    ("KEY hop", "SP-2026-0011", "5dd3949",
     "dict key 'valid' from tool_verify_response, read as d['valid']",
     lambda f: find_key_decisions(f, "valid")),
    ("RETURN hop", "SP-2026-0016", "ed91621",
     "detect_context_injection(...) bound to a local, then decided on",
     lambda f: find_call_then_decide(f, "detect_context_injection")),
]


def main() -> int:
    print("ATTAINABILITY — is the attempt-2 gate answerable at all?\n")
    cache: dict[str, dict[str, str]] = {}
    reachable = 0
    for label, cid, commit, desc, probe in HOPS:
        if commit not in cache:
            cache[commit] = pkg_files(commit)
        files = cache[commit]
        hits = probe(files)
        # a hop is DEMONSTRATED only when a consumer exists AND one branch is loud
        useful = [h for h in hits if h[2] or h[3]]
        ok = bool(useful)
        reachable += ok
        print(f"  {'DEMONSTRATED' if ok else 'NOT DEMONSTRATED'}  {label}  ({cid})")
        print(f"      {desc}")
        print(f"      consumers found: {len(hits)}, with a loud branch: {len(useful)}")
        for path, line, lb, le, snip in useful[:3]:
            print(f"        {path}:{line}  if[{lb or '-'}] else[{le or '-'}]")
            print(f"          | {snip}")
        if hits and not useful:
            for path, line, _, _, snip in hits[:2]:
                print(f"        (quiet both sides) {path}:{line}  | {snip}")
        print()

    print(f"  {reachable}/3 hops demonstrated on real pre-fix source")
    if reachable == 3:
        print("  -> the G0 floor of 10/20 is ANSWERABLE. Freeze the prereg.")
    else:
        print("  -> at least one hop is NOT demonstrated. Lower the floor BEFORE")
        print("     freezing, in the open, and say which hop failed and why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
