# -*- coding: utf-8 -*-
"""Gemini CLI drives the Claude Code diffgate hook unchanged: BeforeTool, run_shell_command, exit 2.

Gemini's stdin object carries the shared fields plus its own (timestamp, mcp_context) and names
the shell tool run_shell_command. The hook must read the shared ones, accept that tool name,
and ignore the rest: a lying commit message exits 2 with the lies on stderr, an honest one exits
0, and a write_file call is not read at all.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "integrations" / "claude-code" / "diffgate-hook" / "pretool.py"


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


def _gemini(repo, tool_name, tool_input):
    payload = {
        "session_id": "s", "transcript_path": "/tmp/t.json", "cwd": str(repo),
        "hook_event_name": "BeforeTool", "timestamp": "2026-09-16T16:00:00Z",
        "tool_name": tool_name, "tool_input": tool_input, "mcp_context": {},
    }
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(HOOK)], cwd=repo, input=json.dumps(payload),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def test_a_lying_commit_from_gemini_is_blocked_with_exit_2_and_the_lies_on_stderr(staged_repo):
    r = _gemini(staged_repo, "run_shell_command", {"command": 'git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."'})
    assert r.returncode == 2
    assert "BLOCKED" in r.stderr
    assert "[LIE] tests_added" in r.stderr and "claim says 3" in r.stderr
    assert "[LIE] only_touches" in r.stderr and "tests/test_retry.py" in r.stderr


def test_an_honest_commit_from_gemini_passes(staged_repo):
    r = _gemini(staged_repo, "run_shell_command", {"command": 'git commit -m "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed."'})
    assert r.returncode == 0, r.stderr


def test_gemini_write_file_is_not_read(staged_repo):
    r = _gemini(staged_repo, "write_file", {"file_path": "x.py", "content": "git commit -m 'Added 3 tests.'"})
    assert r.returncode == 0 and r.stderr == ""


def test_the_gemini_readme_points_at_the_one_file():
    text = (ROOT / "integrations" / "gemini-cli" / "diffgate-hook" / "README.md").read_text(encoding="utf-8")
    assert "integrations/claude-code/diffgate-hook/pretool.py" in text
    assert '"matcher": "run_shell_command"' in text
