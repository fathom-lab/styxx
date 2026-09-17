# -*- coding: utf-8 -*-
"""integrations/cursor/diffgate-hook — Cursor's beforeShellExecution hook, driven the way Cursor drives it.

JSON on stdin with the command and cwd, JSON on stdout with a permission. A lying commit message
is denied with the lies in agent_message; an honest one, a non-git command, a message with no
diff-shaped claim and unparseable stdin are all allowed. The command parser must be the Claude
Code hook's, byte for byte, when that file is present.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "integrations" / "cursor" / "diffgate-hook" / "before_shell.py"
CLAUDE_HOOK = ROOT / "integrations" / "claude-code" / "diffgate-hook" / "pretool.py"
SHARED = ("_git", "_heredoc", "_flag_values", "_parse")


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


def _hook(repo, command, raw=None):
    payload = raw if raw is not None else json.dumps({
        "command": command, "cwd": str(repo), "sandbox": False, "conversation_id": "c",
        "generation_id": "g", "model": "m", "hook_event_name": "beforeShellExecution",
        "cursor_version": "1", "workspace_roots": [str(repo)], "user_email": None})
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(HOOK)], cwd=repo, input=payload, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_a_lying_commit_message_is_denied_with_the_lies_in_agent_message(staged_repo):
    out = _hook(staged_repo, 'git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."')
    assert out["permission"] == "deny"
    assert "BLOCKED" in out["user_message"]
    assert "[LIE] tests_added" in out["agent_message"] and "claim says 3" in out["agent_message"]
    assert "[LIE] only_touches" in out["agent_message"] and "tests/test_retry.py" in out["agent_message"]
    assert "[ok ] file_touched" in out["agent_message"]


def test_the_heredoc_form_is_read(staged_repo):
    cmd = 'git commit -m "$(cat <<\'EOF\'\nRefactored src/retry.py\n\nAdded 3 tests. Only touches files under src/.\nEOF\n)"'
    assert _hook(staged_repo, cmd)["permission"] == "deny"


def test_an_honest_message_is_allowed(staged_repo):
    out = _hook(staged_repo, 'git commit -m "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed."')
    assert out == {"permission": "allow"}


def test_commands_the_hook_does_not_read_are_allowed_untouched(staged_repo):
    assert _hook(staged_repo, "ls -la") == {"permission": "allow"}
    assert _hook(staged_repo, 'git commit -m "tidy"') == {"permission": "allow"}
    assert _hook(staged_repo, "git commit --amend --no-edit") == {"permission": "allow"}


def test_unparseable_stdin_fails_open(staged_repo):
    assert _hook(staged_repo, None, raw="not json") == {"permission": "allow"}


def test_the_answer_uses_cursors_field_names_only(staged_repo):
    out = _hook(staged_repo, 'git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."')
    assert set(out) <= {"permission", "user_message", "agent_message"}
    assert out["permission"] in ("allow", "deny", "ask")


def _source_of(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    src = path.read_text(encoding="utf-8").splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = "\n".join(src[node.lineno - 1:node.end_lineno])
    return out


@pytest.mark.skipif(not CLAUDE_HOOK.exists(), reason="the Claude Code hook is not in this tree")
def test_the_parser_is_the_claude_code_parser_verbatim():
    mine = _source_of(HOOK, SHARED)
    theirs = _source_of(CLAUDE_HOOK, SHARED)
    assert set(mine) == set(SHARED) == set(theirs)
    for name in SHARED:
        assert mine[name] == theirs[name], name


def test_nothing_from_the_command_is_executed_and_no_network_is_imported():
    tree = ast.parse(HOOK.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"urllib", "http", "socket", "requests", "ssl"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            first = node.args[0]
            assert isinstance(first, ast.List) and isinstance(first.elts[0], ast.Constant) and first.elts[0].value == "git"
