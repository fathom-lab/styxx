# -*- coding: utf-8 -*-
"""Break one rule of the collector at a time, in a copy, and check that the tests catch each break.

    py -3.12 checks/mutation_check.py <scratch dir> [pytest target ...]

With no pytest target the copy's own tests/ folder runs (the development layout under
_direction_a_2026_11_02/collector). In the lab's repository the collector's tests live in tests/:
run from the repository root and pass them, e.g.

    py -3.12 papers/tests-pass-ci/collect/checks/mutation_check.py <scratch> tests/test_tests_pass_ci_collector_*.py

The copy is named to those tests by the environment variable TPC_COLLECT_DIR. Nothing outside the
scratch directory is written.
"""
import os
import pathlib
import shutil
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent
M = [
    ("seal check never raises", "tpc_seal.py", "        if self.in_window(created_at) and not self.open:", "        if False:"),
    ("seal skips the ancestry check", "tpc_seal.py", "        if not self._on_main(self.repo, commit):", "        if False:"),
    ("seal reads any signature line", "tpc_seal.py", "        if not signed(text):", "        if False:"),
    ("seal reads the markers anywhere", "tpc_seal.py",
     "    return bool(lines) and all(ln.startswith(SIGNED) and not ln.startswith(UNSIGNED) for ln in lines)",
     "    return SIGNED in text and UNSIGNED not in text"),
    ("seal ignores TO-FILL rows", "tpc_seal.py", "        if to_fill_rows(text):", "        if False:"),
    ("seal ignores the running code", "tpc_seal.py", "        if sealed_code[0] != running:", "        if False:"),
    ("seal takes a future anchor", "tpc_seal.py", "        if self._now() < t:", "        if False:"),
    ("read_row skips the guard", "tpc_read.py", '    guard.check(row.get("created_at"), f"CI read of',
     '    (lambda *a: None)(row.get("created_at"), f"CI read of'),
    ("batch skips the guard", "tpc_read.py", '        guard.check(r.get("created_at"), f"body read of',
     '        (lambda *a: None)(r.get("created_at"), f"body read of'),
    ("stage_read drops the sealed filter", "tpc_collect.py", "    if sealed and not ctx.guard.open:", "    if False:"),
    ("cap by u", "tpc_collect.py", '    for r in sorted(rows, key=lambda x: (x["v"], x["u"])):',
     '    for r in sorted(rows, key=lambda x: x["u"]):'),
    ("M uses P_AGENT for all", "tpc_draw.py", "    in_m = in_d and below(h, p_main(agent))", "    in_m = in_d"),
    ("timed_out not mapped", "tpc_screen.py", '    return "failure" if conclusion == "timed_out" else conclusion',
     "    return conclusion"),
    ("slice formula label", "tpc_frame.py", '(("tests-pass-ci/slice/" + d.isoformat())',
     '(("tests-pass-ci/slices/" + d.isoformat())'),
    ("postfilter dropped", "tpc_read.py", "    if need and need not in body.lower():", "    if False:"),
    ("au1 logs dropped", "tpc_read.py",
     '    au1_eligible = row["in_M"] and sc["screen"] == "none" and outcome == "tests failed"',
     "    au1_eligible = False"),
    ("rule R ignores staleness", "tpc_rule.py", "    item3 = stale is False", "    item3 = True"),
    ("rule R reads local claims", "tpc_rule.py", " or LOCAL.search(text):", ":"),
    ("dry run asks for databaseId", "tpc_const.py", 'DRYRUN_FIELDS = "number createdAt',
     'DRYRUN_FIELDS = "databaseId number createdAt'),
    ("frame keeps forks", "tpc_frame.py", '    return (row.get("stars") or 0) >= MIN_STARS and row.get("fork") is False',
     '    return (row.get("stars") or 0) >= MIN_STARS'),
]


def main() -> int:
    scratch = pathlib.Path(sys.argv[1]).resolve()
    targets = sys.argv[2:]
    work = scratch / "mut"
    missed = 0
    for name, f, old, new in M:
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(SRC, work, ignore=shutil.ignore_patterns("__pycache__", "checks"))
        p = work / f
        s = p.read_bytes().decode("utf-8")
        assert s.count(old) == 1, (name, s.count(old))
        p.write_bytes(s.replace(old, new).encode("utf-8"))
        env = dict(os.environ, TPC_COLLECT_DIR=str(work))
        if targets:
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + targets
            cwd = None
        else:
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"]
            cwd = work
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        missed += int(r.returncode == 0)
        print(("CAUGHT " if r.returncode else "MISSED ") + name + " :: " + last, flush=True)
    print(f"{len(M) - missed} of {len(M)} caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
