# -*- coding: utf-8 -*-
"""integrations/gitlab/diffgate.gitlab-ci.yml — the job's script lines, run the way GitLab runs them.

The job file is parsed as YAML, its `script` lines are executed in a temporary repository with
CI_MERGE_REQUEST_DESCRIPTION, CI_MERGE_REQUEST_DIFF_BASE_SHA and CI_COMMIT_SHA set the way a
merge request pipeline sets them (the `pip install` line is skipped: the release under test is
the one on sys.path). The JSON artifact is written either way.

The job reports by default, as the GitHub Action does since it changed its default (CHANGELOG,
"the GitHub Action reports by default"): a lying description is reported, its lies named in the
log, and the job passes. Only STYXX_SOFT_FAIL set to "false" blocks, read the way the Action's
``_soft_fail`` reads soft-fail: compared case-insensitively with surrounding whitespace stripped,
unset as "true", every other value reporting with a warning naming it. An error that leaves no
report fails the job in either mode.
"""
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "integrations" / "gitlab" / "diffgate.gitlab-ci.yml"
yaml = pytest.importorskip("yaml")

LYING = "Refactored src/retry.py. Added 3 tests. Only touches files under src/."
HONEST = "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed."
PASSES = 'so the job passes; set STYXX_SOFT_FAIL to "false" to fail it.'
REPORTED = "WARNING: styxx diffgate found failures but STYXX_SOFT_FAIL is on, as it is by default, " + PASSES
REPORTED_ODD = ("WARNING: styxx diffgate found failures but STYXX_SOFT_FAIL has a value this job does not "
                "recognise (named above), so it reports as it does by default, " + PASSES)
BLOCKED = 'styxx diffgate: FAIL, and STYXX_SOFT_FAIL is "false", so the job fails.'
NOT_RECOGNISED = "which is neither \"true\" nor \"false\""
NO_REPORT = "without writing a FAIL report: that is an error, not a verdict"


def _bash():
    """The bash that runs the job's script, as a full path, or None when there is none.

    A bare "bash" is the wrong name to hand subprocess on Windows (#185): CreateProcess searches
    the system directory before PATH, so it starts WSL's launcher (System32\\bash.exe) even where
    shutil.which("bash") found Git's bash, and the launcher can exit 1 without running a line.
    So the script runs under the path found here, and a launcher under the Windows directory or
    the WindowsApps alias never counts as bash. On Windows, Git's own bash is looked for beside
    git.exe when the one on PATH is a launcher.
    """
    found = shutil.which("bash")
    if os.name != "nt":
        return found

    def usable(p):
        if not p:
            return False
        low = str(Path(p).resolve()).lower()
        windir = str(Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()).lower()
        return not (low.startswith(windir + os.sep) or os.sep + "windowsapps" + os.sep in low)

    if usable(found):
        return found
    git = shutil.which("git")
    if git:
        for parent in Path(git).resolve().parents:
            for rel in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
                cand = parent.joinpath(*rel)
                if cand.is_file() and usable(cand):
                    return str(cand)
    return None


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="the job's script needs bash (WSL's launcher does not count)")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout.strip()


def _job():
    return yaml.safe_load(JOB.read_text(encoding="utf-8"))["diffgate"]


@pytest.fixture
def mr_repo(tmp_path):
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    (work / "src").mkdir()
    (work / "src" / "retry.py").write_text("def retry(n):\n    return n\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "base")
    _git(work, "push", "-q", "origin", "HEAD:main")
    base = _git(work, "rev-parse", "HEAD")
    (work / "src" / "retry.py").write_text(
        "def retry(n):\n    return n\n\ndef retry_once(n):\n    return retry(1)\n", encoding="utf-8")
    (work / "tests").mkdir()
    (work / "tests" / "test_retry.py").write_text("def test_retry_once():\n    assert True\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "the change")
    head = _git(work, "rev-parse", "HEAD")
    return work, base, head


def _run_job(work, base, head, description, soft_fail=None, strict=False):
    """Run the job's script lines under bash. ``soft_fail`` None leaves STYXX_SOFT_FAIL unset;
    ``strict`` adds --strict to the styxx.diffgate line, the edit the README describes."""
    job = _job()
    assert job["rules"][0]["if"] == '$CI_PIPELINE_SOURCE == "merge_request_event"'
    env = dict(os.environ, CI_MERGE_REQUEST_DESCRIPTION=description,
               CI_MERGE_REQUEST_DIFF_BASE_SHA=base, CI_COMMIT_SHA=head,
               CI_PIPELINE_SOURCE="merge_request_event", PYTHONPATH=str(ROOT))
    env.pop("STYXX_SOFT_FAIL", None)
    if soft_fail is not None:
        env["STYXX_SOFT_FAIL"] = soft_fail
    lines = [l for l in job["script"] if not l.startswith("pip install")]
    script = "\n".join(lines)
    if strict:
        assert script.count(" --out diffgate.json") == 1
        script = script.replace(" --out diffgate.json", " --strict --out diffgate.json")
    # bash reads the backslashes of a Windows path as escapes, so the interpreter is passed with
    # forward slashes and quoted (#185); on POSIX this is the path as it was. The job names it
    # `python` twice, at the start of a line: the gate, and the reading of STYXX_SOFT_FAIL.
    python = shlex.quote(Path(sys.executable).as_posix())
    script, n = re.subn(r"(?m)^python(?= -)", lambda _: python, script)
    assert n == 2, script
    return subprocess.run([BASH, "-c", "set -e\n" + script + "\n"], cwd=work, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def _report(work):
    return json.loads((work / "diffgate.json").read_text(encoding="utf-8"))


@needs_bash
def test_by_default_a_lying_description_is_reported_with_the_lies_named_and_the_job_passes(mr_repo):
    work, base, head = mr_repo
    r = _run_job(work, base, head, LYING)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" in r.stdout and "[CONTRADICTED:tests_added]" in r.stdout
    assert "[CONTRADICTED:only_touches]" in r.stdout and "tests/test_retry.py" in r.stdout
    assert REPORTED in r.stdout and REPORTED_ODD not in r.stdout
    assert NOT_RECOGNISED not in r.stdout      # unset is the default, not a value
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
def test_with_soft_fail_false_a_lying_description_fails_the_job_with_the_lies_named(mr_repo):
    work, base, head = mr_repo
    r = _run_job(work, base, head, LYING, soft_fail="false")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAIL" in r.stdout and "[CONTRADICTED:tests_added]" in r.stdout
    assert "[CONTRADICTED:only_touches]" in r.stdout and "tests/test_retry.py" in r.stdout
    assert BLOCKED in r.stdout and PASSES not in r.stdout
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "false"])
def test_an_honest_description_passes_the_job(mr_repo, soft_fail):
    work, base, head = mr_repo
    r = _run_job(work, base, head, HONEST, soft_fail=soft_fail)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("PASS")
    assert "WARNING" not in r.stdout
    assert _report(work)["verdict"] == "PASS"


def _action_soft_fail():
    spec = importlib.util.spec_from_file_location("_diffgate_action_for_gitlab", ROOT / "diffgate_action.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._soft_fail


# The seven values tests/test_diffgate_action.py reads through the Action, and three more.
@needs_bash
@pytest.mark.parametrize("value, blocks, warned", [
    ("", False, True),
    (" true", False, False),
    ("TRUE", False, False),
    ("yes", False, True),
    ("0", False, True),
    ("False ", True, False),
    ("false", True, False),
    ("\tFALSE\n", True, False),
    ("no", False, True),
    ("fals", False, True),
])
def test_how_the_soft_fail_value_is_read(mr_repo, value, blocks, warned):
    """Only "false" blocks, compared case-insensitively with surrounding whitespace stripped;
    "true" reports; every other value, the empty string included, reports and is named in a
    warning. The Action's reader gives the same answer for each value."""
    assert _action_soft_fail()(value) == (not blocks, value if warned else None)
    work, base, head = mr_repo
    r = _run_job(work, base, head, LYING, soft_fail=value)
    assert r.returncode == (1 if blocks else 0), r.stdout + r.stderr
    assert "[CONTRADICTED:tests_added]" in r.stdout
    assert (NOT_RECOGNISED in r.stdout) is warned
    if warned:
        assert f"WARNING: STYXX_SOFT_FAIL is {value!r}, which is neither" in r.stdout
    assert (BLOCKED in r.stdout) is blocks
    assert (PASSES in r.stdout) is (not blocks)
    assert (REPORTED in r.stdout) is (not blocks and not warned)
    assert (REPORTED_ODD in r.stdout) is (not blocks and warned)
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
@pytest.mark.parametrize("soft_fail, rc", [(None, 0), ("false", 1)])
def test_strict_makes_an_uncheckable_claim_a_failure_that_blocks_only_with_soft_fail_false(mr_repo, soft_fail, rc):
    work, base, head = mr_repo
    r = _run_job(work, base, head, "Added 1 test. All tests pass.", soft_fail=soft_fail, strict=True)
    assert r.returncode == rc, r.stdout + r.stderr
    assert r.stdout.startswith("FAIL") and "contradicted=0" in r.stdout
    assert "[UNCHECKABLE:tests_pass]" in r.stdout
    assert (REPORTED in r.stdout) is (rc == 0) and (BLOCKED in r.stdout) is (rc == 1)
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "true", "false"])
def test_a_gate_that_errors_fails_the_job_in_either_mode_and_a_stale_report_is_not_read(mr_repo, soft_fail):
    """A head the repository does not have makes the CLI raise, exit 1 and write no report. That
    is an error, not a FAIL verdict, so reporting does not swallow it; and a diffgate.json left
    from an earlier run, saying FAIL, is removed before the gate runs rather than read as this
    run's verdict."""
    work, base, _head = mr_repo
    (work / "diffgate.json").write_text(json.dumps({"diffgate": "v0", "verdict": "FAIL"}), encoding="utf-8")
    r = _run_job(work, base, "0" * 40, LYING, soft_fail=soft_fail)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "styxx diffgate exited 1 " + NO_REPORT in r.stdout
    assert PASSES not in r.stdout and BLOCKED not in r.stdout
    assert not (work / "diffgate.json").exists()


def test_the_job_never_expands_the_description_through_the_shell():
    job = _job()
    writes = [l for l in job["script"] if "CI_MERGE_REQUEST_DESCRIPTION" in l]
    assert writes == ["printf '%s' \"$CI_MERGE_REQUEST_DESCRIPTION\" > mr_description.md"]
    assert job["artifacts"]["when"] == "always" and "diffgate.json" in job["artifacts"]["paths"]


def test_the_job_leaves_soft_fail_unset_so_a_pipeline_or_project_variable_reaches_it():
    """In GitLab a variable in the job's own `variables:` outranks one in the pipeline's top-level
    `variables:`, so a default written into the job would silently override a pipeline that sets
    STYXX_SOFT_FAIL: "false". The default lives in the script (unset reads as "true") instead.
    The job's commented-out line is the copy-paste opt-in."""
    job = _job()
    assert "STYXX_SOFT_FAIL" not in (job.get("variables") or {})
    text = JOB.read_text(encoding="utf-8")
    assert re.search(r'(?m)^    # STYXX_SOFT_FAIL: "false"', text)
