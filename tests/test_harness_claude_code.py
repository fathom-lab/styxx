# -*- coding: utf-8 -*-
"""styxx.harness.claude_code — the PostToolUse stager and the Stop finaliser, at L1.

Built to papers/sworn/DESIGN_harness_adapters_2026_09_02.md. No real Claude Code session is
run: every payload here is canned in the shapes the live documentation and this box's
transcripts showed. What is pinned: the outside-workspace rule (with Windows paths where the
interpreter is Windows), session_id validation, the tolerant tool table, completeness marks,
authored recording for Write and Edit, parallel staging without loss, an idempotent fold, the
end-to-end MALFORMED receipt_author_minted for a file the agent wrote and then read, exit zero
for every payload, an empty Stop stdout, and the README's settings block.

LOAD-BEARING: test_a_file_the_agent_wrote_and_then_read_is_malformed_end_to_end,
test_a_manifest_directory_inside_the_workspace_is_refused.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
import threading
from pathlib import Path

import pytest

import styxx.harness.claude_code as CC
from styxx.sworn import Manifest, verify

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "integrations" / "claude-code" / "sworn-hooks" / "README.md"
SID = "sess-0001"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


@pytest.fixture
def mdir(tmp_path):
    return tmp_path / "manifests"


def payload(tool, tool_input, tool_response, *, cwd, sid=SID, tuid="toolu_01", **extra):
    d = {"session_id": sid, "transcript_path": "t.jsonl", "cwd": str(cwd), "permission_mode": "default",
         "hook_event_name": "PostToolUse", "tool_name": tool, "tool_input": tool_input,
         "tool_use_id": tuid, "tool_response": tool_response}
    d.update(extra)
    return d


def stop_payload(cwd, sid=SID, **extra):
    d = {"session_id": sid, "transcript_path": "t.jsonl", "cwd": str(cwd), "permission_mode": "default",
         "hook_event_name": "Stop", "stop_hook_active": False}
    d.update(extra)
    return d


def bash(cwd, command="echo hi", stdout="hi\n", stderr="", interrupted=False, **kw):
    return payload("Bash", {"command": command},
                   {"stdout": stdout, "stderr": stderr, "interrupted": interrupted, "isImage": False},
                   cwd=cwd, **kw)


def read(cwd, path, content, start=1, num=None, total=None, cap=None, tool_input_extra=None, **kw):
    lines = content.count("\n") or 1
    f = {"content": content, "filePath": path, "startLine": start,
         "numLines": lines if num is None else num, "totalLines": lines if total is None else total}
    if cap is not None:
        f["truncatedByTokenCap"] = cap
    ti = {"file_path": path}
    ti.update(tool_input_extra or {})
    return payload("Read", ti, {"type": "text", "file": f}, cwd=cwd, **kw)


def staged(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════ names and the outside rule

class TestSessionId:
    @pytest.mark.parametrize("bad", ["../../evil", "CON", "nul.json", "", None, 5, "a/b", "a\\b",
                                     ".hidden", "x" * 200, "with space"])
    def test_an_unsafe_session_id_writes_nothing(self, ws, mdir, bad):
        assert CC.valid_session_id(bad) is False
        assert CC.stage_event(bash(ws, sid=bad), mdir) is None
        assert not mdir.exists()

    def test_a_safe_session_id_is_a_path_component(self):
        assert CC.valid_session_id("8f3c1e2a-4b5d-4e6f-8a9b-0c1d2e3f4a5b")
        assert CC.valid_session_id("sess.1_2-3")


class TestOutsideRule:
    def test_a_manifest_directory_inside_the_workspace_is_refused(self, ws, tmp_path, capsys):
        """LOAD-BEARING. A manifest the agent can Read into a receipt of itself is L0."""
        for inside in (ws, ws / "m", ws / "sub" / ".." / "m"):
            assert CC.stage_event(bash(ws), inside) is None
            assert CC.finalise(stop_payload(ws), inside) is None
        assert not (ws / "m").exists()
        assert "inside the workspace" in capsys.readouterr().err

    def test_a_sibling_whose_name_extends_the_workspace_is_outside(self, ws, tmp_path):
        sib = tmp_path / "ws-manifests"
        assert CC.is_inside(sib, [ws]) is False
        assert CC.stage_event(bash(ws), sib) is not None

    def test_a_directory_under_claude_project_dir_is_refused_while_cwd_is_a_worktree(self, tmp_path):
        project = tmp_path / "project"
        worktree = tmp_path / "wt"
        project.mkdir(), worktree.mkdir()
        assert CC.stage_event(bash(worktree), project / "m", extra_roots=[str(project)]) is None
        assert CC.stage_event(bash(worktree), tmp_path / "m", extra_roots=[str(project)]) is not None

    def test_a_payload_without_cwd_writes_nothing(self, tmp_path, capsys):
        p = bash(tmp_path / "ws")
        del p["cwd"]
        assert CC.stage_event(p, tmp_path / "m") is None
        assert "cwd absent" in capsys.readouterr().err

    @pytest.mark.parametrize("weird", ["\\\\?\\C:\\Users\\x\\manifests", "//?/C:/x", "relative/dir", ""])
    def test_a_directory_that_is_not_a_native_absolute_path_is_refused(self, ws, weird, capsys):
        assert CC.native_abs(weird) is None
        assert CC.stage_event(bash(ws), weird) is None
        assert "not a native absolute path" in capsys.readouterr().err

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
    def test_windows_a_case_variant_of_the_workspace_is_still_inside(self, ws):
        upper = Path(str(ws).upper()) / "m"
        assert CC.is_inside(upper, [ws]) is True
        assert CC.stage_event(bash(ws), upper) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
    def test_windows_a_drive_less_msys_spelling_is_refused(self, ws):
        msys = "/c/Users/someone/manifests"
        assert CC.native_abs(msys) is None
        assert CC.stage_event(bash(ws), msys) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
    def test_windows_forward_slash_spelling_of_the_workspace_is_inside(self, ws):
        fwd = str(ws).replace("\\", "/") + "/m"
        assert CC.is_inside(fwd, [ws]) is True

    def test_default_dir_reads_only_the_mapping_it_is_handed(self):
        assert CC.default_dir({"STYXX_SWORN_MANIFEST_DIR": "/x/y"}) == "/x/y"
        assert CC.default_dir({}) is None or CC.default_dir({}) is not None   # never raises
        if sys.platform == "win32":
            assert CC.default_dir({"LOCALAPPDATA": "C:\\la"}).startswith("C:\\la")
        else:
            assert CC.default_dir({"HOME": "/home/u"}).startswith("/home/u")


# ═════════════════════════════════════════════════════════════ the tool table

class TestBash:
    def test_stdout_is_a_complete_tool_stdout_receipt_noted_with_the_command(self, ws, mdir):
        p = CC.stage_event(bash(ws, command="python -m pytest -q", stdout="3 passed\n"), mdir)
        ev = staged(p)
        assert ev["tool_use_id"] == "toolu_01" and ev["tool_name"] == "Bash"
        (r,) = ev["receipts"]
        assert r["kind"] == "tool_stdout" and r["complete"] is True and r["note"] == "python -m pytest -q"
        assert base64.b64decode(r["b64"]) == b"3 passed\n" and r["sha256"] == sha(b"3 passed\n")
        assert ev["authored_sha256"] == []

    def test_non_empty_stderr_is_a_second_receipt(self, ws, mdir):
        ev = staged(CC.stage_event(bash(ws, stdout="", stderr="warn\n"), mdir))
        assert [r["kind"] for r in ev["receipts"]] == ["tool_stdout", "tool_stderr"]

    def test_an_interrupted_response_is_incomplete(self, ws, mdir):
        ev = staged(CC.stage_event(bash(ws, stdout="part", interrupted=True), mdir))
        assert ev["receipts"][0]["complete"] is False

    @pytest.mark.parametrize("out", ["a\nb\n... [12 lines truncated]\n", "x ... [4000 characters truncated]",
                                     "<persisted-output>\nOutput too large. Saved to: /tmp/x.txt\n</persisted-output>"])
    def test_a_truncation_marker_or_a_spill_stub_is_incomplete_and_not_followed(self, ws, mdir, out):
        ev = staged(CC.stage_event(bash(ws, stdout=out), mdir))
        assert ev["receipts"][0]["complete"] is False
        assert base64.b64decode(ev["receipts"][0]["b64"]).decode() == out

    def test_a_string_response_is_taken_whole(self, ws, mdir):
        ev = staged(CC.stage_event(payload("Bash", {"command": "x"}, "Error: exit 1", cwd=ws), mdir))
        assert ev["receipts"][0]["kind"] == "tool_stdout"
        assert base64.b64decode(ev["receipts"][0]["b64"]) == b"Error: exit 1"

    def test_a_shape_neither_source_showed_is_serialised_whole_with_a_note(self, ws, mdir):
        ev = staged(CC.stage_event(payload("Bash", {"command": "x"}, [1, 2], cwd=ws), mdir))
        assert "not recognised" in ev["receipts"][0]["note"]


class TestRead:
    def test_a_whole_file_read_is_a_complete_file_read_receipt(self, ws, mdir):
        ev = staged(CC.stage_event(read(ws, str(ws / "a.txt"), "l1\nl2\n"), mdir))
        (r,) = ev["receipts"]
        assert r["kind"] == "file_read" and r["complete"] is True and r["note"] == str(ws / "a.txt")
        assert base64.b64decode(r["b64"]) == b"l1\nl2\n"

    def test_a_window_is_incomplete(self, ws, mdir):
        assert staged(CC.stage_event(read(ws, "f", "x\n", start=20, num=1, total=90), mdir))["receipts"][0]["complete"] is False
        assert staged(CC.stage_event(read(ws, "f", "x\n", num=1, total=90), mdir))["receipts"][0]["complete"] is False
        assert staged(CC.stage_event(read(ws, "f", "x\n", cap=True), mdir))["receipts"][0]["complete"] is False
        assert staged(CC.stage_event(read(ws, "f", "x\n", tool_input_extra={"offset": 10}), mdir))["receipts"][0]["complete"] is False
        assert staged(CC.stage_event(read(ws, "f", "x\n", tool_input_extra={"limit": 5}), mdir))["receipts"][0]["complete"] is False

    def test_a_zero_based_whole_window_is_complete(self, ws, mdir):
        assert staged(CC.stage_event(read(ws, "f", "x\ny\n", start=0), mdir))["receipts"][0]["complete"] is True

    def test_the_documented_shape_is_read_too(self, ws, mdir):
        ev = staged(CC.stage_event(payload("Read", {"file_path": "f"}, {"success": True, "content": "abc"}, cwd=ws), mdir))
        assert ev["receipts"][0]["kind"] == "file_read" and base64.b64decode(ev["receipts"][0]["b64"]) == b"abc"
        assert ev["receipts"][0]["complete"] is True

    def test_an_image_mints_nothing(self, ws, mdir):
        p = payload("Read", {"file_path": "i.png"}, {"type": "image", "file": {"base64": "AAAA", "type": "image/png"}}, cwd=ws)
        assert CC.stage_event(p, mdir) is None


class TestWebFetch:
    @pytest.mark.parametrize("resp", [{"result": "rendered", "code": 200, "url": "https://x"},
                                      {"success": True, "content": "rendered"}, "rendered"])
    def test_a_fetch_is_never_complete(self, ws, mdir, resp):
        ev = staged(CC.stage_event(payload("WebFetch", {"url": "https://x"}, resp, cwd=ws), mdir))
        (r,) = ev["receipts"]
        assert r["kind"] == "http_fetch" and r["complete"] is False and r["note"] == "https://x"
        assert base64.b64decode(r["b64"]) == b"rendered"


class TestWriteAndEdit:
    def test_write_records_the_content_and_the_file_on_disk_and_mints_nothing(self, ws, mdir):
        f = ws / "x.txt"
        f.write_bytes(b"hello\r\n")                       # the disk kept a CRLF the input lacked
        p = payload("Write", {"file_path": str(f), "content": "hello\n"},
                    {"type": "create", "filePath": str(f), "content": "hello\n"}, cwd=ws)
        ev = staged(CC.stage_event(p, mdir))
        assert ev["receipts"] == []
        assert set(ev["authored_sha256"]) == {sha(b"hello\n"), sha(b"hello\r\n")}

    def test_edit_records_new_string_the_reconstruction_and_the_disk(self, ws, mdir):
        f = ws / "y.py"
        f.write_bytes(b"a = 2\nb = 2\n")
        p = payload("Edit", {"file_path": str(f), "old_string": "a = 1", "new_string": "a = 2"},
                    {"filePath": str(f), "oldString": "a = 1", "newString": "a = 2",
                     "originalFile": "a = 1\nb = 1\n", "replaceAll": False}, cwd=ws)
        ev = staged(CC.stage_event(p, mdir))
        assert set(ev["authored_sha256"]) == {sha(b"a = 2"), sha(b"a = 2\nb = 1\n"), sha(b"a = 2\nb = 2\n")}

    def test_edit_honours_replace_all(self, ws, mdir):
        p = payload("Edit", {"file_path": str(ws / "none"), "old_string": "1", "new_string": "2", "replace_all": True},
                    {"originalFile": "1 1 1"}, cwd=ws)
        ev = staged(CC.stage_event(p, mdir))
        assert sha(b"2 2 2") in ev["authored_sha256"]

    def test_a_failed_write_or_edit_records_nothing(self, ws, mdir):
        for tool in ("Write", "Edit"):
            p = payload(tool, {"file_path": str(ws / "z"), "content": "c", "old_string": "o", "new_string": "n"},
                        "Error: file has not been read yet", cwd=ws)
            assert CC.stage_event(p, mdir) is None

    @pytest.mark.parametrize("tool", ["Grep", "Glob", "Task", "TodoWrite", "MultiEdit", "NotebookEdit", "mcp__x__y"])
    def test_other_tools_stage_nothing(self, ws, mdir, tool):
        assert CC.stage_event(payload(tool, {"a": 1}, {"b": 2}, cwd=ws), mdir) is None
        assert tool in CC.IGNORED_TOOLS or tool.startswith("mcp__")


class TestCap:
    def test_a_receipt_over_the_cap_is_staged_hash_only_and_folded_hash_only(self, ws, mdir, monkeypatch):
        monkeypatch.setattr(CC, "RECEIPT_BYTE_CAP", 8)
        big = "x" * 9
        ev = staged(CC.stage_event(bash(ws, stdout=big), mdir))
        assert ev["receipts"][0]["b64"] is None and ev["receipts"][0]["sha256"] == sha(big.encode())
        out = CC.finalise(stop_payload(ws), mdir)
        m = Manifest.load(out)
        assert "bytes" not in m.receipts["r1"] and m.receipts["r1"]["sha256"] == sha(big.encode())
        s = verify(('<sworn r="r1" k="hash">it hashes to %s.</sworn>\n' % sha(big.encode())).encode(),
                   name="d.md", manifest=m)["spans"][0]
        assert s["verdict"] == "HELD"
        s = verify(b'<sworn r="r1" k="quote">it printed `xxxxxxxxxxxxxxxxx`</sworn>\n', name="d.md", manifest=m)["spans"][0]
        assert (s["verdict"], s["reason"]) == ("UNRESOLVED", "manifest_bytes_absent")


# ═════════════════════════════════════════════════════════════ the fold

class TestFinalise:
    def test_the_manifest_is_l1_with_the_blindness_and_the_weakness_printed(self, ws, mdir):
        CC.stage_event(bash(ws), mdir)
        out = CC.finalise(stop_payload(ws), mdir)
        assert out == mdir / (SID + ".manifest.json")
        m = Manifest.load(out)
        assert m.intact() and m.rung_status() == ("ok", "L1") and m.turn == SID
        assert "blind, permanently, to files written by shell commands" in m.harness
        assert "weak" in m.harness and "adapters, never a recorder" in m.harness
        assert b"\r" not in out.read_bytes()

    def test_a_file_the_agent_wrote_and_then_read_is_malformed_end_to_end(self, ws, mdir):
        """LOAD-BEARING. Invariant 2 by set membership across the two hook events."""
        f = ws / "note.txt"
        f.write_bytes(b"the result is 0.55\n")
        CC.stage_event(payload("Write", {"file_path": str(f), "content": "the result is 0.55\n"},
                               {"type": "create"}, cwd=ws, tuid="toolu_w"), mdir)
        CC.stage_event(read(ws, str(f), "the result is 0.55\n", tuid="toolu_r"), mdir)
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        assert list(m.receipts) == ["r1"] and m.receipts["r1"]["kind_of_source"] == "file_read"
        s = verify(b'<sworn r="r1" k="numeric">the result is 0.55</sworn>\n', name="d.md", manifest=m)["spans"][0]
        assert (s["verdict"], s["reason"]) == ("MALFORMED", "receipt_author_minted")

    def test_a_shell_written_file_read_later_is_accepted_the_documented_blindness(self, ws, mdir):
        f = ws / "shell.txt"
        f.write_bytes(b"42\n")
        CC.stage_event(bash(ws, command="echo 42 > shell.txt", stdout="", tuid="toolu_b"), mdir)
        CC.stage_event(read(ws, str(f), "42\n", tuid="toolu_r"), mdir)
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        s = verify(b'<sworn r="r2" k="numeric">it says 42</sworn>\n', name="d.md", manifest=m)["spans"][0]
        assert s["verdict"] == "HELD" and s["provenance"]["rung"] == "L1"

    def test_two_folds_of_the_same_events_are_byte_identical(self, ws, mdir):
        CC.stage_event(bash(ws, tuid="toolu_a"), mdir)
        CC.stage_event(bash(ws, tuid="toolu_b", stdout="two\n"), mdir)
        a = CC.finalise(stop_payload(ws), mdir).read_bytes()
        b = CC.finalise(stop_payload(ws), mdir).read_bytes()
        assert a == b
        m = Manifest.load(mdir / (SID + ".manifest.json"))
        assert list(m.receipts) == ["r1", "r2"] and m.minted_at == m.receipts["r2"]["captured_at"]

    def test_ids_follow_capture_order_then_tool_use_id(self, ws, mdir):
        import datetime as dt
        t0 = dt.datetime(2026, 9, 5, 12, 0, 0, tzinfo=dt.timezone.utc)
        CC.stage_event(bash(ws, tuid="toolu_z", stdout="z"), mdir, now=t0)
        CC.stage_event(bash(ws, tuid="toolu_a", stdout="a"), mdir, now=t0)
        CC.stage_event(bash(ws, tuid="toolu_m", stdout="early"), mdir, now=t0 - dt.timedelta(seconds=5))
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        got = [base64.b64decode(m.receipts[r]["bytes"]) for r in ("r1", "r2", "r3")]
        assert got == [b"early", b"a", b"z"]

    def test_parallel_staging_loses_nothing(self, ws, mdir):
        n = 24
        threads = [threading.Thread(target=CC.stage_event, args=(bash(ws, tuid="toolu_%03d" % i, stdout="%d" % i), mdir))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        assert len(m.receipts) == n
        assert sorted(int(base64.b64decode(e["bytes"])) for e in m.receipts.values()) == list(range(n))

    def test_notes_carry_tool_prompt_and_agent_ids(self, ws, mdir):
        CC.stage_event(bash(ws, command="ls", prompt_id="p-9", agent_id="agent-2"), mdir)
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        note = m.receipts["r1"]["harness_note"]
        assert note.startswith("Bash ls") and "prompt_id=p-9" in note and "agent_id=agent-2" in note
        assert "tool_use_id=toolu_01" in note

    def test_the_last_assistant_message_enters_authored(self, ws, mdir):
        CC.stage_event(bash(ws), mdir)
        m = Manifest.load(CC.finalise(stop_payload(ws, last_assistant_message="done: 3 passed"), mdir))
        assert sha(b"done: 3 passed") in m.authored_sha256

    def test_an_unreadable_staged_file_is_a_reported_gap_not_a_crash(self, ws, mdir, capsys):
        CC.stage_event(bash(ws), mdir)
        (mdir / SID / "events" / "zzz-broken.json").write_bytes(b"{not json")
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        assert list(m.receipts) == ["r1"]
        assert "could not be read" in capsys.readouterr().err

    def test_a_session_with_no_events_still_writes_an_empty_manifest(self, ws, mdir):
        m = Manifest.load(CC.finalise(stop_payload(ws), mdir))
        assert m.receipts == {} and m.rung_status() == ("ok", "L1")


# ═════════════════════════════════════════════════════════════ the CLI layer

def feed(monkeypatch, data: bytes):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))


class TestMain:
    @pytest.mark.parametrize("garbage", [b"", b"not json", b"[1, 2]", b"{}", b"\xff\xfe",
                                         json.dumps({"session_id": "../x", "cwd": "/w", "tool_name": "Bash"}).encode(),
                                         json.dumps({"session_id": "s", "cwd": "/w", "tool_name": 5}).encode(),
                                         json.dumps({"session_id": "s", "cwd": None, "tool_name": "Bash",
                                                     "tool_input": None, "tool_response": None}).encode()])
    def test_main_returns_zero_for_every_payload(self, monkeypatch, tmp_path, garbage):
        feed(monkeypatch, garbage)
        assert CC.main(["post-tool", "--dir", str(tmp_path / "m")]) == 0
        feed(monkeypatch, garbage)
        assert CC.main(["stop", "--dir", str(tmp_path / "m")]) == 0

    def test_post_tool_stages_and_stop_folds_with_nothing_on_stdout(self, monkeypatch, ws, mdir, capsys):
        feed(monkeypatch, json.dumps(bash(ws)).encode())
        assert CC.main(["post-tool", "--dir", str(mdir)]) == 0
        feed(monkeypatch, json.dumps(stop_payload(ws)).encode())
        assert CC.main(["stop", "--dir", str(mdir)]) == 0
        out, err = capsys.readouterr()
        assert out == ""
        assert "manifest" in err
        assert Manifest.load(mdir / (SID + ".manifest.json")).receipts["r1"]["kind_of_source"] == "tool_stdout"

    def test_claude_project_dir_is_a_second_root(self, monkeypatch, tmp_path, capsys):
        project, wt = tmp_path / "project", tmp_path / "wt"
        project.mkdir(), wt.mkdir()
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
        feed(monkeypatch, json.dumps(bash(wt)).encode())
        assert CC.main(["post-tool", "--dir", str(project / "m")]) == 0
        assert not (project / "m").exists() and "inside the workspace" in capsys.readouterr().err

    def test_the_environment_names_the_directory_when_dir_is_absent(self, monkeypatch, ws, tmp_path):
        monkeypatch.setenv("STYXX_SWORN_MANIFEST_DIR", str(tmp_path / "envdir"))
        feed(monkeypatch, json.dumps(bash(ws)).encode())
        assert CC.main(["post-tool"]) == 0
        assert (tmp_path / "envdir" / SID / "events").is_dir()

    def test_only_main_reads_stdin_or_the_environment(self):
        import ast
        tree = ast.parse(Path(CC.__file__).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name == "main":
                continue
            names = {"%s.%s" % (n.value.id, n.attr) for n in ast.walk(fn)
                     if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
            assert not (names & {"sys.stdin", "os.environ", "os.getenv", "sys.argv"}), fn.name

    def test_the_module_imports_nothing_from_styxx_at_module_scope(self):
        import ast
        tree = ast.parse(Path(CC.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("styxx") for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("styxx")


# ═════════════════════════════════════════════════════════════ the README

class TestReadme:
    def test_the_opening_paragraph_says_this_is_the_weak_rung_and_why(self):
        text = README.read_text(encoding="utf-8")
        opening = text.split("\n\n")[1]
        assert opening.startswith("This is the weak rung.")
        assert "not a recorder" in opening and "blind, permanently" in opening
        assert "that is what L1 means" in opening

    def test_the_settings_block_parses_and_matches_the_five_tools(self):
        text = README.read_text(encoding="utf-8")
        block = re.search(r"```json\n(.*?)```", text, re.S).group(1)
        cfg = json.loads(block)
        post = cfg["hooks"]["PostToolUse"][0]
        assert post["matcher"] == "Bash|Read|Write|Edit|WebFetch"
        assert post["hooks"][0]["command"].endswith('post-tool.py"')
        stop = cfg["hooks"]["Stop"][0]
        assert stop["matcher"] == "" and stop["hooks"][0]["command"].endswith('stop.py"')
        assert post["hooks"][0]["timeout"] <= 60 and stop["hooks"][0]["timeout"] <= 120

    def test_this_repository_enables_no_hook_of_its_own(self):
        settings = ROOT / ".claude" / "settings.json"
        if settings.exists():
            cfg = json.loads(settings.read_text(encoding="utf-8"))
            assert "sworn-hooks" not in json.dumps(cfg)
