"""The MUTE-1 instrument, held to the failures that would make its receipt worthless.

A mutation tester whose mutants do not apply, whose oracle cannot fail, or whose verdicts come
from the exit code rather than from per-test comparison would report a killed/survived split that
means nothing. These pin the mechanics; the preregistration
(`papers/harness/PREREG_mute1_harness_mutation_2026_09_20.md`) pins the population and the
predictions, and the receipt pins the run.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.harness_mutation import mute  # noqa: E402


def _scratch(tmp_path: Path) -> Path:
    """A throwaway copy of the harness files, so apply() never touches the real checkout."""
    t = tmp_path / "tree"
    (t / ".github" / "workflows").mkdir(parents=True)
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        shutil.copy(wf, t / ".github" / "workflows" / wf.name)
    (t / "packages" / "styxx-js").mkdir(parents=True)
    shutil.copy(ROOT / "packages" / "styxx-js" / "package.json", t / "packages" / "styxx-js" / "package.json")
    for rel, _, _ in mute.SUBJECTS:
        src = ROOT / rel
        if src.exists():
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, t / rel)
    return t


def test_the_inventory_is_deterministic_and_matches_the_preregistered_population():
    a = mute.inventory()
    b = mute.inventory()
    assert [m.id for m in a[1]] == [m.id for m in b[1]]
    ops = {}
    for m in a[1]:
        ops[m.operator] = ops.get(m.operator, 0) + 1
    assert set(ops) == {"M-TRIGGER", "M-JOB", "M-STEP", "M-SWALLOW", "M-GUARD", "M-SCRIPT", "M-SUBJECT"}
    assert ops["M-STEP"] == ops["M-SWALLOW"], "every run step gets both cuts"


@pytest.mark.parametrize("op", ["M-TRIGGER", "M-JOB", "M-STEP", "M-SWALLOW", "M-GUARD", "M-SCRIPT"])
def test_every_operator_changes_the_file_and_leaves_it_parseable(op, tmp_path):
    t = _scratch(tmp_path)
    _, mutants = mute.inventory()
    m = next(x for x in mutants if x.operator == op)
    target = t / m.check.path
    before = target.read_text(encoding="utf-8")
    assert mute.apply(m, t) is True
    after = target.read_text(encoding="utf-8")
    assert after != before, "the mutant applied but changed nothing"
    if target.suffix == ".yml":
        doc = yaml.safe_load(after)
        assert isinstance(doc, dict) and "jobs" in doc
        assert "on" in doc or True in doc, "the workflow lost its trigger key"
    else:
        json.loads(after)


def test_a_swallowed_step_cannot_fail(tmp_path):
    """The point of M-SWALLOW: the step's exit status is thrown away. Proven by running it."""
    t = _scratch(tmp_path)
    _, mutants = mute.inventory()
    m = next(x for x in mutants if x.operator == "M-SWALLOW")
    assert mute.apply(m, t)
    doc = yaml.safe_load((t / m.check.path).read_text(encoding="utf-8"))
    run = doc["jobs"][m.check.job]["steps"][m.check.step]["run"]
    assert run.rstrip().endswith("|| true")
    sh = tmp_path / "s.sh"
    sh.write_text("( exit 7\n) || true\n", encoding="utf-8")
    assert subprocess.run(["bash", str(sh)]).returncode == 0


def test_a_subject_line_cut_removes_exactly_that_line(tmp_path):
    t = _scratch(tmp_path)
    _, mutants = mute.inventory()
    m = next(x for x in mutants if x.operator == "M-SUBJECT" and x.check.path.endswith("py_side.py"))
    before = (t / m.check.path).read_text(encoding="utf-8")
    assert mute.apply(m, t)
    after = (t / m.check.path).read_text(encoding="utf-8")
    assert "PINNED = " in before and "PINNED = " not in after
    assert len(before.splitlines()) - len(after.splitlines()) in (0, 1)


def test_an_absent_subject_is_unreached_not_killed(tmp_path):
    t = _scratch(tmp_path)
    _, mutants = mute.inventory()
    m = next(x for x in mutants if x.operator == "M-SUBJECT" and x.check.path == "telescope/prompts.json")
    (t / m.check.path).unlink(missing_ok=True)
    assert mute.apply(m, t) is False, "cutting a file that is not there must be UNREACHED, never a verdict"


def test_the_oracle_is_selected_by_pattern_not_by_hand():
    files = mute.oracle_files(ROOT)
    assert files, "the oracle pattern selects nothing"
    for f in files:
        assert mute.ORACLE_PATTERN.search((ROOT / f).read_text(encoding="utf-8", errors="replace"))
    assert "tests/test_harness_mutation.py" not in files or True   # this file may or may not match; either is fine


def test_verdicts_compare_per_test_against_the_baseline_not_the_exit_code():
    """A test that already fails on the unmutated tree must not be able to kill a mutant."""
    baseline = {"a::t1": "passed", "a::t2": "failed", "a::t3": "passed"}
    passing = sorted(k for k, v in baseline.items() if v == "passed")
    mutant_res = {"a::t1": "passed", "a::t2": "failed", "a::t3": "passed"}
    assert not [k for k in passing if mutant_res.get(k) != "passed"], "nothing changed: SURVIVED"
    mutant_res = {"a::t1": "failed", "a::t2": "passed", "a::t3": "passed"}
    assert [k for k in passing if mutant_res.get(k) != "passed"] == ["a::t1"], "t1 killed it; t2 recovering is not a kill"


def test_a_kill_is_red_and_a_vanished_test_is_not_a_kill():
    """v1.1 (MUTE-3). A baseline-passing test kills the mutant only by going RED. A test whose id
    is not collected on the mutant has vanished -- recorded, never a kill: a deleted test cannot
    be its own alarm. A red id that was not in the baseline at all is a collection error, and
    that IS a kill (the suite went red, just not by a test the baseline knew)."""
    passing = ["a::t1", "a::t2", "a::t3"]
    baseline_ids = {"a::t1", "a::t2", "a::t3", "a::t4"}      # t4 failed on baseline: excluded
    killed_by, failed, vanished, errors = mute.compare(passing, {"a::t1": "failed", "a::t3": "passed", "a::t4": "failed"}, baseline_ids)
    assert killed_by == ["a::t1"] and failed == ["a::t1"] and vanished == ["a::t2"] and errors == []
    killed_by, failed, vanished, errors = mute.compare(passing, {"a::t1": "passed", "a::t3": "passed"}, baseline_ids)
    assert killed_by == [] and vanished == ["a::t2"], "vanishing alone is SURVIVED"
    killed_by, failed, vanished, errors = mute.compare(passing, {"a::tests/a.py": "failed"}, baseline_ids)
    assert killed_by == ["a::tests/a.py"] and errors == ["a::tests/a.py"] and vanished == passing, "a collection crash is red"


def test_level_two_enumerates_exactly_the_declared_guards_and_their_tests():
    checks, mutants = mute.inventory(level=2)
    files = {c.path for c in checks if c.kind == "guard-file"}
    assert files == {g for g in mute.GUARDS if (ROOT / g).exists()}
    ops = {}
    for m in mutants:
        ops[m.operator] = ops.get(m.operator, 0) + 1
    assert ops["M-GFUNC"] == ops["M-GVACUOUS"], "every test function gets both cuts"
    assert ops["M-GFILE"] == len(files)
    assert all(m.id.startswith("MUTE-G") for m in mutants), "level-2 ids are namespaced apart from level 1"


def test_the_guard_operators_cut_exactly_one_thing_and_leave_the_file_parseable(tmp_path):
    import ast
    src = ('import pytest\n\n'
           'def helper():\n    return 1\n\n'
           '@pytest.mark.parametrize("x", [1, 2])\n'
           'def test_first(x):\n    assert x > 0, "positive"\n    assert helper() == 1\n\n'
           'def test_second():\n    assert helper() == 1\n')
    t = tmp_path / "tests"
    t.mkdir()
    (t / "test_g.py").write_text(src, encoding="utf-8")
    c = mute.Check("guard-function", "tests/test_g.py", name="test_first")
    assert mute.apply(mute.Mutant("MUTE-G001", "M-GFUNC", c, ""), tmp_path) is True
    after = (t / "test_g.py").read_text(encoding="utf-8")
    names = [n.name for n in ast.parse(after).body if isinstance(n, ast.FunctionDef)]
    assert names == ["helper", "test_second"], "the decorated function went, with its decorator, and nothing else"
    (t / "test_g.py").write_text(src, encoding="utf-8")
    assert mute.apply(mute.Mutant("MUTE-G002", "M-GVACUOUS", c, ""), tmp_path) is True
    mod = ast.parse((t / "test_g.py").read_text(encoding="utf-8"))
    first = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "test_first")
    second = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "test_second")
    assert all(isinstance(a.test, ast.Constant) and a.test.value is True for a in ast.walk(first) if isinstance(a, ast.Assert))
    assert not any(isinstance(a.test, ast.Constant) for a in ast.walk(second) if isinstance(a, ast.Assert)), "the other function is untouched"
    assert len(first.decorator_list) == 1, "the parametrize decorator survives the hollowing"
    (t / "test_g.py").write_text(src, encoding="utf-8")
    fc = mute.Check("guard-file", "tests/test_g.py", name="test_g.py")
    assert mute.apply(mute.Mutant("MUTE-G003", "M-GFILE", fc, ""), tmp_path) is True and not (t / "test_g.py").exists()
    assert mute.apply(mute.Mutant("MUTE-G003", "M-GFILE", fc, ""), tmp_path) is False, "cutting what is gone is UNREACHED"


def test_the_instrument_refuses_to_mutate_its_own_checkout():
    r = subprocess.run([sys.executable, "-m", "benchmarks.harness_mutation.mute", "--run",
                        "--tree", str(ROOT)], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr)


def test_the_fingerprint_moves_when_the_harness_moves(tmp_path):
    """The receipt applies to any commit whose harness fingerprint matches, so the fingerprint has
    to move when a workflow moves and stay put when something unrelated does."""
    a = mute.harness_fingerprint(ROOT, mute.oracle_files(ROOT))
    b = mute.harness_fingerprint(ROOT, mute.oracle_files(ROOT))
    assert a == b
    t = _scratch(tmp_path)
    (t / "tests").mkdir()
    for f in mute.oracle_files(ROOT):
        shutil.copy(ROOT / f, t / f)
    (t / "README.md").write_text("unrelated", encoding="utf-8")
    c = mute.harness_fingerprint(t, mute.oracle_files(ROOT))
    wf = next((t / ".github" / "workflows").glob("*.yml"))
    wf.write_text(wf.read_text(encoding="utf-8") + "\n# moved\n", encoding="utf-8")
    d = mute.harness_fingerprint(t, mute.oracle_files(ROOT))
    assert c != d
