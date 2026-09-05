# -*- coding: utf-8 -*-
"""The four entry points the design names, run as subprocesses.

`python -m styxx.harness junit|github|claude-code ...`, `python -m styxx.harness.claude_code`,
and the two thin scripts under integrations/claude-code/sworn-hooks/. Each writes an LF-only
manifest, or exits as its contract says: junit/github two on usage, claude-code and the scripts
zero on everything.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from styxx.sworn import Manifest

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "integrations" / "claude-code" / "sworn-hooks"

GREEN = b"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" failures="0" errors="0">
<testcase classname="tests.test_app" name="test_one" />
<testcase classname="tests.test_app" name="test_two" />
</testsuite></testsuites>
"""


def run(args, stdin: bytes = b"", env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("CLAUDE_PROJECT_DIR", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable] + args, input=stdin, cwd=str(ROOT), capture_output=True, env=env)


def test_junit_mints_and_refuses_the_reserved_rung(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_bytes(GREEN)
    out = tmp_path / "m.json"
    r = run(["-m", "styxx.harness", "junit", str(report), "--rung", "L1", "--turn", "t", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    assert b"\r" not in out.read_bytes()
    assert Manifest.load(out).rung == "L1"
    r = run(["-m", "styxx.harness", "junit", str(report), "--rung", "L3", "--out", str(tmp_path / "n.json")])
    assert r.returncode == 2 and not (tmp_path / "n.json").exists()


def test_github_mints(tmp_path):
    ev = tmp_path / "event.json"
    ev.write_bytes(json.dumps({"before": "a" * 40, "after": "b" * 40}).encode())
    out = tmp_path / "m.json"
    r = run(["-m", "styxx.harness", "github", "--event", str(ev), "--event-name", "push",
             "--rung", "L2", "--after-turn-on-base", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    assert b"\r" not in out.read_bytes() and Manifest.load(out).rung == "L2"


def test_claude_code_post_tool_then_stop_through_the_package(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    mdir = tmp_path / "m"
    post = {"session_id": "cli-sess", "cwd": str(ws), "hook_event_name": "PostToolUse",
            "tool_name": "Bash", "tool_input": {"command": "echo 1"}, "tool_use_id": "toolu_1",
            "tool_response": {"stdout": "1\n", "stderr": "", "interrupted": False}}
    r = run(["-m", "styxx.harness", "claude-code", "post-tool", "--dir", str(mdir)], json.dumps(post).encode())
    assert r.returncode == 0, r.stderr
    stop = {"session_id": "cli-sess", "cwd": str(ws), "hook_event_name": "Stop", "stop_hook_active": False}
    r = run(["-m", "styxx.harness.claude_code", "stop", "--dir", str(mdir)], json.dumps(stop).encode())
    assert r.returncode == 0, r.stderr
    assert r.stdout == b""
    m = Manifest.load(mdir / "cli-sess.manifest.json")
    assert m.rung == "L1" and m.receipts["r1"]["kind_of_source"] == "tool_stdout"


def test_the_thin_scripts_exit_zero_on_everything_and_the_post_tool_path_never_imports_styxx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    mdir = tmp_path / "m"
    post = {"session_id": "script-sess", "cwd": str(ws), "hook_event_name": "PostToolUse",
            "tool_name": "Bash", "tool_input": {"command": "echo 2"}, "tool_use_id": "toolu_2",
            "tool_response": {"stdout": "2\n", "stderr": "", "interrupted": False}}
    # a sitecustomize that raises on any styxx import: the post-tool path must never trip it
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'styxx' or name.startswith('styxx.'):\n"
        "            raise ImportError('styxx imported on the PostToolUse path: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n", encoding="utf-8")
    r = run([str(HOOKS / "post-tool.py"), "--dir", str(mdir)], json.dumps(post).encode(),
            env_extra={"PYTHONPATH": str(guard) + os.pathsep + str(ROOT)})
    assert r.returncode == 0, r.stderr
    assert b"styxx imported" not in r.stderr
    assert list((mdir / "script-sess" / "events").glob("*.json")), r.stderr
    stop = {"session_id": "script-sess", "cwd": str(ws), "hook_event_name": "Stop"}
    r = run([str(HOOKS / "stop.py"), "--dir", str(mdir)], json.dumps(stop).encode())
    assert r.returncode == 0, r.stderr
    assert r.stdout == b""
    assert Manifest.load(mdir / "script-sess.manifest.json").receipts["r1"]["kind_of_source"] == "tool_stdout"
    # garbage on stdin, a directory inside the workspace, no stdin at all: still zero
    for script in ("post-tool.py", "stop.py"):
        assert run([str(HOOKS / script), "--dir", str(mdir)], b"not json").returncode == 0
        assert run([str(HOOKS / script), "--dir", str(ws / "m")], json.dumps(post).encode()).returncode == 0
        assert run([str(HOOKS / script), "--bogus-flag"], b"{}").returncode == 0
    assert not (ws / "m").exists()


def test_the_dispatcher_refuses_an_unknown_adapter_with_exit_two():
    r = run(["-m", "styxx.harness", "banana"])
    assert r.returncode == 2 and b"usage" in r.stderr
    assert run(["-m", "styxx.harness"]).returncode == 2
