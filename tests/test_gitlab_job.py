# -*- coding: utf-8 -*-
"""integrations/gitlab/diffgate.gitlab-ci.yml — the job's script lines, run the way GitLab runs them.

The job file is parsed as YAML, its `script` lines are executed in a temporary repository with
CI_MERGE_REQUEST_DESCRIPTION, CI_MERGE_REQUEST_DIFF_BASE_SHA and CI_COMMIT_SHA set the way a
merge request pipeline sets them (the `pip install` line is skipped: the release under test is
the one on sys.path). A lying description fails the job with the lies named; an honest one
passes; the JSON artifact is written either way.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "integrations" / "gitlab" / "diffgate.gitlab-ci.yml"
yaml = pytest.importorskip("yaml")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout.strip()


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


def _run_job(work, base, head, description):
    job = yaml.safe_load(JOB.read_text(encoding="utf-8"))["diffgate"]
    assert job["rules"][0]["if"] == '$CI_PIPELINE_SOURCE == "merge_request_event"'
    env = dict(os.environ, CI_MERGE_REQUEST_DESCRIPTION=description,
               CI_MERGE_REQUEST_DIFF_BASE_SHA=base, CI_COMMIT_SHA=head,
               CI_PIPELINE_SOURCE="merge_request_event", PYTHONPATH=str(ROOT))
    lines = [l for l in job["script"] if not l.startswith("pip install")]
    script = "set -e\n" + "\n".join(lines).replace("python -m", f"{sys.executable} -m") + "\n"
    return subprocess.run(["bash", "-c", script], cwd=work, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


@pytest.mark.skipif(shutil.which("bash") is None, reason="the job's script needs bash")
def test_a_lying_description_fails_the_job_with_the_lies_named(mr_repo):
    work, base, head = mr_repo
    r = _run_job(work, base, head, "Refactored src/retry.py. Added 3 tests. Only touches files under src/.")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAIL" in r.stdout and "[CONTRADICTED:tests_added]" in r.stdout
    assert "[CONTRADICTED:only_touches]" in r.stdout and "tests/test_retry.py" in r.stdout
    report = json.loads((work / "diffgate.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"


@pytest.mark.skipif(shutil.which("bash") is None, reason="the job's script needs bash")
def test_an_honest_description_passes_the_job(mr_repo):
    work, base, head = mr_repo
    r = _run_job(work, base, head, "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed.")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("PASS")
    assert json.loads((work / "diffgate.json").read_text(encoding="utf-8"))["verdict"] == "PASS"


def test_the_job_never_expands_the_description_through_the_shell():
    job = yaml.safe_load(JOB.read_text(encoding="utf-8"))["diffgate"]
    writes = [l for l in job["script"] if "CI_MERGE_REQUEST_DESCRIPTION" in l]
    assert writes == ["printf '%s' \"$CI_MERGE_REQUEST_DESCRIPTION\" > mr_description.md"]
    assert job["artifacts"]["when"] == "always" and "diffgate.json" in job["artifacts"]["paths"]
