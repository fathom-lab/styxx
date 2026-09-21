"""MUTE-1 — mutation testing of the checking harness itself.

    python -m benchmarks.harness_mutation.mute --inventory        # list the checks and the mutants, run nothing
    python -m benchmarks.harness_mutation.mute --run              # apply every mutant, ask the oracle, write the receipt
    python -m benchmarks.harness_mutation.mute --run --only M-JOB # one operator
    python -m benchmarks.harness_mutation.mute --run --level 2    # MUTE-3: mutate the GUARDS themselves (see below)

Mutation testing asks of a test suite: if I break the program, does a test go red? This asks the
same question one level up, of the *checking apparatus*: if I cut a check out of CI, silence a
step, flip its guard, delete the file it reads, or stop the workflow from firing at all -- does
anything in this repository go red?

Four times in two days the answer in this repository was no (`gauntlet-pr.yml`, `telescope.yml`,
the port differential on `main`, the styxx-js typecheck): a check that existed, reported green,
and measured nothing. Each was found by a person reading a file. This is the instrument that
reads the files.

Vocabulary
----------
A CHECK is something the repository declares as a verification: a workflow trigger, a workflow
job, a `run:` step, an npm script, or a SUBJECT -- a file a check reads and would be meaningless
without (a corpus, a pin, a lockfile, a status page).

A MUTANT is one CHECK with one OPERATOR applied:

    M-TRIGGER   the workflow's `on:` becomes `workflow_dispatch` only -- it never fires by itself
    M-JOB       the job is deleted
    M-STEP      the `run:` step is deleted
    M-SWALLOW   the `run:` step is wrapped `( ... ) || true` -- it can no longer fail
    M-GUARD     the step's `if:` becomes `false` -- it never runs
    M-SCRIPT    the npm script is deleted from package.json
    M-SUBJECT   the subject file is deleted (or its load-bearing line removed)

The ORACLE is the repository's own test suite, restricted to the tests that read the harness
(selected by a path pattern, listed in the receipt, never hand-picked). A mutant is

    KILLED      some test that passed on the unmutated tree goes RED on the mutant (fails or
                errors), or the mutant makes collection itself error
    SURVIVED    nothing goes red -- the cut was silent
    UNREACHED   the mutant could not be applied (the file or line is not there to cut)

Per-test comparison against the baseline, not "did the oracle exit non-zero": a test that already
fails on the unmutated tree cannot kill anything, and is reported as excluded rather than counted.

Receipt schema v1.1 (MUTE-3) splits what v1 folded together. A baseline-passing test can be RED
on the mutant, or it can have VANISHED (its id is not collected at all -- a parametrized case
whose subject was deleted, or a test function that was itself the thing cut). v1 counted both as
kills; v1.1 counts only red, and a collection error, and records the vanished ids beside them.
Nothing in MUTE-1 or MUTE-2 changes under v1.1 (their kills were red, or a collection crash), and
the split is what makes level 2 readable: a deleted test cannot be its own alarm.

Level 2 -- the guards under mutation
------------------------------------
`--level 2` asks the question one level further up. The GUARDS are the test files written to hold
the harness in place (declared in `GUARDS` below; the manifest pins their test functions since
MUTE-3). Three operators cut them:

    M-GFILE     the guard file is deleted
    M-GFUNC     one test function is deleted
    M-GVACUOUS  every `assert` in one test function becomes `assert True` -- the guard still
                runs, still passes, and checks nothing

The oracle is the same suite. A guard that is cut and noticed by nothing is a guard with no
guard; the RESULT names where that chain ends.

What this measures and what it does not
---------------------------------------
It measures whether the TEST SUITE guards the harness -- whether cutting a check is something the
suite notices. It does not run CI. A mutant that only CI could catch (a deleted `uses: checkout`,
a step that would fail on a runner) is out of scope on purpose: the oracle has to be runnable by
anyone with a clone, and the repository's own discipline is that a guard lives in `tests/`, not
in the hope that somebody watches the Actions tab. SURVIVED therefore means exactly: *nothing in
tests/ would tell you*. The receipt says which tests were asked.

No verdict is hand-labelled. Every survivor is a finding to be read, not a defect to be counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PACKAGE_JSON = ROOT / "packages" / "styxx-js" / "package.json"
RECEIPT_DEFAULT = ROOT / "papers" / "harness" / "mute1_receipt.json"

#: The tests that make up the oracle: every test file whose source refers to a harness path. The
#: pattern is the definition; the list it produces is written into the receipt.
ORACLE_PATTERN = re.compile(
    r"\.github/workflows|\"workflows\"|package\.json|\"\.gitignore\"|telescope"
    r"|web/gate/differential|py_side|check_pairs|calib1_(ask|score)|PREREG_calib1"
)

#: Subjects: files (or lines) that a declared check reads and is meaningless without. Declared
#: here rather than inferred, because "what a check reads" is not mechanical; the preregistration
#: pins this list before the run.
SUBJECTS: list[tuple[str, str, str | None]] = [
    # (relative path, description, load-bearing line regex or None for whole-file deletion)
    ("telescope/prompts.json", "the corpus telescope/run.py loads at load_prompts()", None),
    ("telescope/STATUS.md", "the page that says the telescope is not running", None),
    ("telescope/data/latest.json", "the snapshot the public scoreboard serves", None),
    ("web/gate/differential/py_side.py", "the instrument pin the port's differential refuses to run without",
     r'^PINNED\s*=\s*"[0-9a-f]{64}"'),
    ("web/gate/README.md", "the README that names the instrument hash", None),
    ("packages/styxx-js/package-lock.json", "the lockfile `npm ci` installs from", None),
    (".gitignore", "the rule that stops the telescope corpus being ignored", r"^!telescope/prompts\.json$"),
    ("papers/closed-model-frontier/calib1_score.py", "the prereg hash the CALIB-1 scorer refuses to score without",
     r'^PREREG_SHA256_FROZEN\s*=\s*"[0-9a-f]{64}"'),
    ("papers/closed-model-frontier/calib1_ask.ts", "the model pin the CALIB-1 runner asks for",
     r'^const PINNED_MODEL = "[^"]+";$'),
]


#: The guards: test files whose purpose is to hold the harness in place. Declared, not inferred,
#: for the same reason SUBJECTS are: "what guards what" is not mechanical. Level 2 mutates exactly
#: these; since MUTE-3 the manifest pins their test functions.
GUARDS: list[str] = [
    "tests/test_gauntlet_pr_verifies_something.py",
    "tests/test_telescope_status_is_honest.py",
    "tests/test_ci_runs_the_js_typecheck.py",
    "tests/test_port_is_current.py",
    "tests/test_harness_manifest.py",
    "tests/test_ci_steps_propagate_failure.py",
]


@dataclass
class Check:
    kind: str            # workflow-trigger | workflow-job | workflow-step | npm-script | subject
    path: str
    job: str | None = None
    step: int | None = None
    name: str | None = None
    detail: str | None = None


@dataclass
class Mutant:
    id: str
    operator: str
    check: Check
    description: str


@dataclass
class Verdict:
    mutant: str
    operator: str
    check: dict
    verdict: str                     # KILLED | SURVIVED | UNREACHED
    killed_by: list[str] = field(default_factory=list)      # red tests + collection errors: the alarms
    seconds: float = 0.0
    note: str | None = None
    failed_on_mutant: list[str] = field(default_factory=list)     # baseline-passing ids that went red
    vanished_on_mutant: list[str] = field(default_factory=list)   # baseline-passing ids not collected
    errors_on_mutant: list[str] = field(default_factory=list)     # red ids that were not in the baseline (collection errors)


# --------------------------------------------------------------------------- inventory

def _load_wf(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def inventory(level: int = 1) -> tuple[list[Check], list[Mutant]]:
    if level == 2:
        return inventory_guards()
    checks: list[Check] = []
    mutants: list[Mutant] = []
    n = 0

    def mut(op: str, check: Check, desc: str) -> None:
        nonlocal n
        n += 1
        mutants.append(Mutant(id=f"MUTE-{n:03d}", operator=op, check=check, description=desc))

    for wf in sorted(WORKFLOWS.glob("*.yml")):
        rel = str(wf.relative_to(ROOT))
        doc = _load_wf(wf)
        checks.append(Check("workflow-trigger", rel, name=doc.get("name")))
        mut("M-TRIGGER", checks[-1], f"{rel}: `on:` becomes workflow_dispatch only")
        for jname, job in (doc.get("jobs") or {}).items():
            jc = Check("workflow-job", rel, job=jname, name=job.get("name"))
            checks.append(jc)
            mut("M-JOB", jc, f"{rel}: delete job `{jname}`")
            for i, st in enumerate(job.get("steps") or []):
                if "run" not in st:
                    continue
                sc = Check("workflow-step", rel, job=jname, step=i, name=st.get("name"),
                           detail=(st["run"].strip().splitlines() or [""])[0][:80])
                checks.append(sc)
                mut("M-STEP", sc, f"{rel} / {jname} / step {i}: delete `{sc.name or sc.detail}`")
                mut("M-SWALLOW", sc, f"{rel} / {jname} / step {i}: wrap in `( ... ) || true`")
                if "if" in st:
                    mut("M-GUARD", sc, f"{rel} / {jname} / step {i}: `if:` becomes false")

    scripts = json.loads(PACKAGE_JSON.read_text(encoding="utf-8")).get("scripts", {})
    for sname, cmd in scripts.items():
        c = Check("npm-script", str(PACKAGE_JSON.relative_to(ROOT)), name=sname, detail=cmd)
        checks.append(c)
        mut("M-SCRIPT", c, f"package.json: delete script `{sname}`")

    for rel, desc, line in SUBJECTS:
        c = Check("subject", rel, name=desc, detail=line)
        checks.append(c)
        mut("M-SUBJECT", c, f"{rel}: " + ("remove the line matching /{}/".format(line) if line else "delete the file"))

    return checks, mutants


def guard_functions(path: Path) -> list[tuple[str, int, int, int]]:
    """(name, first line incl. decorators, last line, number of asserts) for every top-level test_*."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            asserts = sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))
            out.append((node.name, first, node.end_lineno, asserts))
    return out


def inventory_guards() -> tuple[list[Check], list[Mutant]]:
    """Level 2: the guard files and their test functions, from GUARDS above."""
    checks: list[Check] = []
    mutants: list[Mutant] = []
    n = 0

    def mut(op: str, check: Check, desc: str) -> None:
        nonlocal n
        n += 1
        mutants.append(Mutant(id=f"MUTE-G{n:03d}", operator=op, check=check, description=desc))

    for rel in GUARDS:
        p = ROOT / rel
        if not p.exists():
            continue
        fc = Check("guard-file", rel, name=p.name)
        checks.append(fc)
        mut("M-GFILE", fc, f"{rel}: delete the guard file")
        for name, first, last, asserts in guard_functions(p):
            c = Check("guard-function", rel, name=name, detail=f"lines {first}-{last}, {asserts} asserts")
            checks.append(c)
            mut("M-GFUNC", c, f"{rel}::{name}: delete the test function")
            mut("M-GVACUOUS", c, f"{rel}::{name}: every assert becomes `assert True`")
    return checks, mutants


# --------------------------------------------------------------------------- applying mutants

def _dump_wf(doc: dict) -> str:
    import yaml
    # PyYAML reads `on:` as the boolean True; write it back as the quoted key GitHub accepts.
    if True in doc:
        doc = {("on" if k is True else k): v for k, v in doc.items()}
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000)


def apply(m: Mutant, tree: Path) -> bool:
    """Apply one mutant to the tree in place. Returns False when there is nothing to cut."""
    c = m.check
    target = tree / c.path
    if m.operator in ("M-TRIGGER", "M-JOB", "M-STEP", "M-SWALLOW", "M-GUARD"):
        if not target.exists():
            return False
        doc = _load_wf(target)
        if m.operator == "M-TRIGGER":
            key = True if True in doc else "on"
            if key not in doc:
                return False
            doc[key] = {"workflow_dispatch": None}
        else:
            jobs = doc.get("jobs") or {}
            if c.job not in jobs:
                return False
            if m.operator == "M-JOB":
                del jobs[c.job]
                if not jobs:
                    doc["jobs"] = {}
            else:
                steps = jobs[c.job].get("steps") or []
                if c.step is None or c.step >= len(steps) or "run" not in steps[c.step]:
                    return False
                if m.operator == "M-STEP":
                    del steps[c.step]
                elif m.operator == "M-SWALLOW":
                    steps[c.step]["run"] = "( " + steps[c.step]["run"].rstrip() + "\n) || true\n"
                elif m.operator == "M-GUARD":
                    if "if" not in steps[c.step]:
                        return False
                    steps[c.step]["if"] = "false"
        target.write_text(_dump_wf(doc), encoding="utf-8")
        return True
    if m.operator == "M-SCRIPT":
        pkg = json.loads(target.read_text(encoding="utf-8"))
        if c.name not in pkg.get("scripts", {}):
            return False
        del pkg["scripts"][c.name]
        target.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")
        return True
    if m.operator == "M-GFILE":
        if not target.exists():
            return False
        target.unlink()
        return True
    if m.operator in ("M-GFUNC", "M-GVACUOUS"):
        if not target.exists():
            return False
        funcs = {name: (first, last, asserts) for name, first, last, asserts in guard_functions(target)}
        if c.name not in funcs:
            return False
        first, last, asserts = funcs[c.name]
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        if m.operator == "M-GFUNC":
            del lines[first - 1:last]
            target.write_text("".join(lines), encoding="utf-8")
            return True
        if asserts == 0:
            return False                      # nothing to make vacuous
        import ast
        mod = ast.parse("".join(lines))
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == c.name:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assert):
                        sub.test = ast.Constant(True)
                        sub.msg = None
        target.write_text(ast.unparse(ast.fix_missing_locations(mod)) + "\n", encoding="utf-8")
        return True
    if m.operator == "M-SUBJECT":
        if not target.exists():
            return False
        if c.detail is None:
            target.unlink()
            return True
        rx = re.compile(c.detail, re.M)
        body = target.read_text(encoding="utf-8")
        if not rx.search(body):
            return False
        target.write_text(rx.sub("", body, count=1), encoding="utf-8")
        return True
    raise ValueError(m.operator)


# --------------------------------------------------------------------------- the oracle

#: The instrument's own tests are not an oracle for the harness: a mutation tester that used
#: itself to decide whether its mutants were noticed would be grading its own homework. The
#: preregistration enumerated the oracle before this file's tests existed (Amendment A).
ORACLE_EXCLUDES = {"tests/test_harness_mutation.py"}


def oracle_files(tree: Path) -> list[str]:
    out = []
    for p in sorted((tree / "tests").glob("test_*.py")):
        rel = str(p.relative_to(tree))
        if rel in ORACLE_EXCLUDES:
            continue
        if ORACLE_PATTERN.search(p.read_text(encoding="utf-8", errors="replace")):
            out.append(rel)
    return out


def run_oracle(tree: Path, files: list[str]) -> dict[str, str]:
    """{nodeid: outcome} for every test in the oracle set."""
    report = tree / ".mute_report.json"
    if report.exists():
        report.unlink()
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header",
           "-o", "addopts=", "--junitxml", str(report), *files]
    subprocess.run(cmd, cwd=str(tree), capture_output=True, text=True, timeout=1800)
    out: dict[str, str] = {}
    if report.exists():
        import xml.etree.ElementTree as ET
        for tc in ET.parse(report).getroot().iter("testcase"):
            nid = f"{tc.get('classname')}::{tc.get('name')}"
            if tc.find("failure") is not None or tc.find("error") is not None:
                out[nid] = "failed"
            elif tc.find("skipped") is not None:
                out[nid] = "skipped"
            else:
                out[nid] = "passed"
        report.unlink()
    return out


def compare(passing: list[str], res: dict[str, str], baseline_ids: set[str] | None = None
            ) -> tuple[list[str], list[str], list[str], list[str]]:
    """(killed_by, failed_on_mutant, vanished_on_mutant, errors_on_mutant) for one oracle result.

    v1.1 rule. A mutant is killed by every baseline-passing test that is RED on it (failed or
    error), and by every red id that was not in the baseline at all (a collection error). A
    baseline-passing test whose id is simply not collected on the mutant has VANISHED: recorded,
    never a kill -- a deleted test cannot be its own alarm."""
    baseline_ids = baseline_ids if baseline_ids is not None else set(passing)
    failed = [k for k in passing if res.get(k) == "failed"]
    vanished = [k for k in passing if k not in res]
    errors = sorted(k for k, v in res.items() if v == "failed" and k not in baseline_ids)
    return (sorted(failed + errors), failed, vanished, errors)


def _restore(tree: Path) -> None:
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=str(tree), check=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=str(tree), check=True)


def tree_sha(tree: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tree), capture_output=True,
                          text=True, check=True).stdout.strip()


def harness_fingerprint(tree: Path, oracle: list[str]) -> str:
    """sha256 over every file a mutant can touch plus every file the oracle reads.

    A commit hash names one commit. This names the *harness*: two commits with different histories
    but the same workflows, package.json, subjects and oracle tests get the same fingerprint, and a
    receipt applies to any commit whose fingerprint matches. That is what lets a run made on a
    locally merged tree be checked against the commit that eventually carries the same files.
    """
    paths = set(str(p.relative_to(tree)) for p in (tree / ".github" / "workflows").glob("*.yml"))
    paths.add(str(PACKAGE_JSON.relative_to(ROOT)))
    paths.update(rel for rel, _, _ in SUBJECTS)
    paths.update(oracle)
    h = hashlib.sha256()
    for rel in sorted(paths):
        f = tree / rel
        h.update(rel.encode() + b"\0")
        h.update(f.read_bytes() if f.exists() else b"<absent>")
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------- the run

def run(tree: Path, only: str | None, out: Path, limit: int | None, level: int = 1) -> dict:
    t0 = time.time()
    checks, mutants = inventory(level)
    if only:
        mutants = [m for m in mutants if m.operator == only]
    if limit:
        mutants = mutants[:limit]
    files = oracle_files(tree)
    if not files:
        sys.exit("the oracle is empty: no test file matches the harness pattern")

    _restore(tree)
    baseline = run_oracle(tree, files)
    passing = sorted(k for k, v in baseline.items() if v == "passed")
    excluded = sorted(k for k, v in baseline.items() if v != "passed")
    baseline_ids = set(baseline)
    if not passing:
        sys.exit("no test passes on the unmutated tree; the oracle cannot kill anything")

    verdicts: list[Verdict] = []
    for m in mutants:
        t1 = time.time()
        applied = apply(m, tree)
        if not applied:
            verdicts.append(Verdict(m.id, m.operator, asdict(m.check), "UNREACHED",
                                    seconds=round(time.time() - t1, 2),
                                    note="nothing to cut: the file, job, step, guard or line is absent"))
            _restore(tree)
            continue
        # a deleted guard file must not turn the whole session into a usage error: ask pytest
        # only for the files that still exist; the deleted one's tests then simply vanish
        res = run_oracle(tree, [f for f in files if (tree / f).exists()])
        _restore(tree)
        killed_by, failed, vanished, errors = compare(passing, res, baseline_ids)
        verdicts.append(Verdict(m.id, m.operator, asdict(m.check),
                                "KILLED" if killed_by else "SURVIVED", killed_by,
                                round(time.time() - t1, 2),
                                failed_on_mutant=failed, vanished_on_mutant=vanished,
                                errors_on_mutant=errors))
        sys.stderr.write(f"\r{len(verdicts)}/{len(mutants)} {m.id} {verdicts[-1].verdict:9s}")
    sys.stderr.write("\n")

    by_op: dict[str, dict[str, int]] = {}
    for v in verdicts:
        by_op.setdefault(v.operator, {"KILLED": 0, "SURVIVED": 0, "UNREACHED": 0})[v.verdict] += 1

    receipt = {
        "schema": "styxx.harness-mutation/v1.1",   # red-only kills; failed/vanished/errors split per verdict
        "level": level,
        "instrument": "benchmarks/harness_mutation/mute.py",
        "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tree": tree_sha(tree),
        "harness_fingerprint": harness_fingerprint(tree, files),
        "oracle": {
            "pattern": ORACLE_PATTERN.pattern,
            "files": files,
            "tests_passing_on_baseline": len(passing),
            "tests_excluded_not_passing_on_baseline": excluded,
        },
        "population": {
            "checks": len(checks),
            "mutants": len(mutants),
            "by_operator": by_op,
        },
        "totals": {k: sum(d[k] for d in by_op.values()) for k in ("KILLED", "SURVIVED", "UNREACHED")},
        "verdicts": [asdict(v) for v in verdicts],
        "seconds": round(time.time() - t0, 1),
        "caveat": ("SURVIVED means no test in the oracle set noticed the cut. It is a finding to be "
                   "read, not a defect to be counted, and the oracle is the test suite, not CI."),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", help="one operator, e.g. M-JOB")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tree", default=str(ROOT), help="a git worktree to mutate (restored after every mutant)")
    ap.add_argument("--out", default=str(RECEIPT_DEFAULT))
    ap.add_argument("--level", type=int, default=1, choices=(1, 2), help="1: the harness; 2: the guards")
    a = ap.parse_args(argv)
    if a.inventory:
        checks, mutants = inventory(a.level)
        by = {}
        for m in mutants:
            by[m.operator] = by.get(m.operator, 0) + 1
        print(f"{len(checks)} checks, {len(mutants)} mutants")
        for op, k in sorted(by.items()):
            print(f"  {op:10s} {k}")
        print("oracle:", *oracle_files(ROOT), sep="\n  ")
        return 0
    if a.run:
        tree = Path(a.tree).resolve()
        if tree == ROOT.resolve():
            sys.exit("refusing to mutate the checkout this instrument lives in; pass --tree <worktree>")
        r = run(tree, a.only, Path(a.out), a.limit, a.level)
        t = r["totals"]
        print(f"KILLED {t['KILLED']}  SURVIVED {t['SURVIVED']}  UNREACHED {t['UNREACHED']}  -> {a.out}")
        for v in r["verdicts"]:
            if v["verdict"] == "SURVIVED":
                c = v["check"]
                print(f"  SURVIVED {v['mutant']} {v['operator']:10s} {c['path']} {c.get('job') or ''} "
                      f"{('step %d' % c['step']) if c.get('step') is not None else ''} {c.get('name') or ''}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
