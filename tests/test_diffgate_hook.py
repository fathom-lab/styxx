# -*- coding: utf-8 -*-
"""styxx.diffgate_hook — the commit message cannot lie about the staged diff, as a console script.

Runs the module the way git and pre-commit run it: a message in a file, the change staged in a
real temporary repository, the script invoked with the file's path. A lying message is refused
(exit 1) with the contradicted claims named; an honest one is allowed; a message with no
diff-shaped claim passes on scope; git's own `#` commentary is not read as prose; "tests pass"
is UNCHECKABLE and never an accusation. Also pins the two pieces of wiring pre-commit relies on:
the entry in .pre-commit-hooks.yaml and the console script in pyproject.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True)


@pytest.fixture
def staged_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "retry.py").write_text("def retry(n):\n    return n\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "src" / "retry.py").write_text(
        "def retry(n):\n    return n\n\ndef retry_once(n):\n    return retry(1)\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_retry.py").write_text(
        "def test_retry_once():\n    assert True\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    return tmp_path


def _run(repo, message):
    msg = repo / "COMMIT_EDITMSG"
    msg.write_text(message, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT))   # the checkout's module, wherever the test runs
    return subprocess.run([sys.executable, "-m", "styxx.diffgate_hook", str(msg)], cwd=repo,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def test_a_lying_message_is_refused_with_the_lies_named(staged_repo):
    r = _run(staged_repo, "Refactored src/retry.py. Added 3 tests. Only touches files under src/.")
    assert r.returncode == 1
    assert "[LIE] tests_added" in r.stdout and "claim says 3" in r.stdout
    assert "[LIE] only_touches" in r.stdout and "tests/test_retry.py" in r.stdout
    assert "[ok ] file_touched" in r.stdout
    assert "--no-verify" in r.stdout


def test_an_honest_message_is_allowed(staged_repo):
    r = _run(staged_repo, "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed.")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout and "4 claim(s)" in r.stdout


def test_no_diff_shaped_claim_passes_on_scope_silently(staged_repo):
    r = _run(staged_repo, "tidy up the retry path")
    assert r.returncode == 0
    assert "LIE" not in r.stdout and "PASS" not in r.stdout


def test_git_commentary_lines_are_not_read(staged_repo):
    r = _run(staged_repo, "tidy up\n# Please enter the commit message. Added 3 tests here.\n#\n")
    assert r.returncode == 0
    assert "tests_added" not in r.stdout


def test_tests_pass_is_uncheckable_never_an_accusation(staged_repo):
    r = _run(staged_repo, "Modified src/retry.py. All tests pass.")
    assert r.returncode == 0
    assert "[ ? ] tests_pass" in r.stdout


def test_wrong_arity_is_a_usage_error_not_a_verdict(tmp_path):
    r = subprocess.run([sys.executable, "-m", "styxx.diffgate_hook"], cwd=tmp_path,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert r.returncode == 2 and "usage" in r.stderr


def test_the_pre_commit_manifest_names_this_hook():
    text = (ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")
    assert "id: diffgate-commit-msg" in text
    assert "entry: styxx-diffgate-commit-msg" in text
    assert "language: python" in text
    assert "stages: [commit-msg]" in text


def test_pyproject_ships_the_console_script():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'styxx-diffgate-commit-msg = "styxx.diffgate_hook:main"' in text
