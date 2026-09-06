# -*- coding: utf-8 -*-
"""The sworn action, driven end to end: a fake event file, a command that writes a canned JUnit
report to $SWORN_JUNIT, canned sworn documents committed at the head of a temporary git
repository, no network.

LOAD-BEARING: test_held_failed_and_unresolved_documents_each_get_a_row_and_exit_zero pins the
report-only contract — every verdict is a row and the exit is zero; test_the_composed_manifest_is
_the_adapters_entries_with_only_the_id_changed pins the layout the README tells authors to cite;
test_a_fork_pull_request_mints_l1_and_prints_the_fork_rule pins the sentence the plan owes.

Built to papers/sworn/SPEC_sworn_action_v01_2026_09_05.md ("Tests this spec commits to").
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ACTION_DIR = ROOT / "sworn"
ACTION = ACTION_DIR / "sworn_action.py"

JUNIT_OK = (b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="0" time="0.01">'
            b'<testcase classname="tests.test_app" name="test_one" time="0.001" />'
            b'<testcase classname="tests.test_app" name="test_two" time="0.001" />'
            b'<testcase classname="tests.test_app" name="test_three" time="0.001" />'
            b'</testsuite></testsuites>\n')

WRITER = ("import os, shutil, sys\n"
          "shutil.copy(sys.argv[1], os.environ['SWORN_JUNIT'])\n"
          "for extra in sys.argv[2:]:\n"
          "    open(extra, 'wb').write(b'rewritten on disk after the turn\\n')\n"
          "print('3 passed in 0.01s')\n")


def load_action():
    spec = importlib.util.spec_from_file_location("sworn_action", ACTION)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sworn_action"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    e = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
             GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=e,
                          check=True, encoding="utf-8", errors="replace").stdout.strip()


def _docs(base_sha: str) -> dict:
    return {
        "HELD.md": ("# held\n"
                    '<sworn r="r1" k="numeric">The runner resolved 3 passed testcases.</sworn>\n'
                    '<sworn r="r2" k="numeric">It resolved 0 failures.</sworn>\n'
                    '<sworn r="r6" k="quote">The base sha is `%s`.</sworn>\n'
                    '<sworn r="r9#/pull_request/number" k="numeric">This is pull request 7.</sworn>\n'
                    % base_sha).encode(),
        "FAILED.md": b'# failed\n<sworn r="r1" k="numeric">The runner resolved 4 passed testcases.</sworn>\n',
        "UNRESOLVED.md": b'# unresolved\n<sworn r="r12" k="numeric">There are 12 receipts.</sworn>\n',
        "REPORT.md": b'# report\n<sworn r="r3" k="absent">The report carries no `<failure`.</sworn>\n',
        "plain.md": b"# plain\nno tag here\n",
        "stale.sworn.json": b'{"spec": "sworn/0.1", "commit": null, "document": {"name": "x", "sha256": "0"}, "text": "", "spans": []}\n',
        # a report the turn committed: different bytes from ok.xml, so only the laundering test
        # copies committed bytes into $SWORN_JUNIT
        "committed.xml": JUNIT_OK.replace(b"test_three", b"test_committed"),
    }


@pytest.fixture
def repo(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _git(ws, "init", "-q")
    (ws / "README.md").write_bytes(b"# app\n")
    _git(ws, "add", ".")
    _git(ws, "commit", "-q", "-m", "base")
    base = _git(ws, "rev-parse", "HEAD")
    for name, data in _docs(base).items():
        (ws / name).write_bytes(data)
    _git(ws, "add", ".")
    _git(ws, "commit", "-q", "-m", "the turn")
    head = _git(ws, "rev-parse", "HEAD")
    (tmp_path / "ok.xml").write_bytes(JUNIT_OK)
    (tmp_path / "write_junit.py").write_text(WRITER, encoding="utf-8")
    return {"ws": ws, "base": base, "head": head, "tmp": tmp_path}


def _event(repo, *, fork=False, event="pull_request", body=None, head=None, base=None, head_repo="lab/app"):
    head = head or repo["head"]
    base = base or repo["base"]
    if event == "push":
        return {"before": base, "after": head, "repository": {"full_name": "lab/app"}}
    pr = {"number": 7, "title": "t",
          "body": body if body is not None else ("body with <sworn r=\"r2\" k=\"numeric\">0 failures.</sworn>"
                                                 "\r\nand $(rm -rf /) `x` line two\r\n"),
          "base": {"ref": "main", "sha": base, "repo": {"full_name": "lab/app", "fork": False}},
          "head": {"ref": "f", "sha": head}}
    if head_repo is not None:
        pr["head"]["repo"] = {"full_name": ("someone/app" if fork else head_repo), "fork": bool(fork)}
    return {"action": "synchronize", "number": 7, "pull_request": pr, "repository": {"full_name": "lab/app"}}


def _command(repo, *extra):
    parts = ['"%s"' % sys.executable, '"%s"' % (repo["tmp"] / "write_junit.py"), '"%s"' % (repo["tmp"] / "ok.xml")]
    parts += ['"%s"' % e for e in extra]
    return " ".join(parts)


def run_action(repo, monkeypatch, *, event=None, event_name="pull_request", command=None, **env):
    """Drive main() with the environment the composite action sets. Returns (code, out_dir, summary, run)."""
    tmp = repo["tmp"]
    ev = tmp / "event.json"
    ev.write_text(json.dumps(event if event is not None else _event(repo)), encoding="utf-8")
    out = tmp / "out"
    for k in list(os.environ):
        if k.startswith("SWORN_") or k.startswith("GITHUB_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(repo["ws"]))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp / "step_summary.md"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp / "github_output.txt"))
    monkeypatch.setenv("SWORN_COMMAND", command if command is not None else _command(repo))
    monkeypatch.setenv("SWORN_OUT_DIR", str(out))
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    def boom(*a, **k):
        raise AssertionError("the action opened a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    code = load_action().main()
    summary = (out / "summary.md").read_text(encoding="utf-8") if (out / "summary.md").exists() else ""
    run = json.loads((out / "run.json").read_text(encoding="utf-8")) if (out / "run.json").exists() else {}
    return code, out, summary, run


def _rows(summary: str) -> dict:
    rows = {}
    for line in summary.splitlines():
        m = re.match(r"\| `([^`]+)` \| (\S+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (L[12]) \| (.*) \|$", line)
        if m:
            rows[m.group(1)] = {"verdict": m.group(2), "held": int(m.group(3)), "failed": int(m.group(4)),
                                "unresolved": int(m.group(5)), "malformed": int(m.group(6)),
                                "rung": m.group(7), "harness": m.group(8)}
    return rows


class TestVerdictsAreRowsAndExitIsZero:
    def test_held_failed_and_unresolved_documents_each_get_a_row_and_exit_zero(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch)
        assert code == 0
        rows = _rows(summary)
        assert rows["HELD.md"]["verdict"] == "SWORN-HELD" and rows["HELD.md"]["held"] == 4
        assert rows["FAILED.md"]["verdict"] == "SWORN-FAILED" and rows["FAILED.md"]["failed"] == 1
        assert rows["UNRESOLVED.md"]["unresolved"] == 1 and rows["UNRESOLVED.md"]["held"] == 0
        assert rows["REPORT.md"]["verdict"] == "SWORN-HELD"
        assert rows["pull_request_body.md"]["verdict"] == "SWORN-HELD"
        assert "- UNRESOLVED manifest_id_missing `r12` @" in summary
        for row in rows.values():
            assert row["rung"] == "L2"
            assert "sworn-action/0.1" in row["harness"] and "fork caveat" in row["harness"]
            assert "junit over" in row["harness"] and "report-only" in row["harness"]
        assert run["rung"] == "L2" and run["rung_reason"] is None
        assert {d["name"]: d["verdict"] for d in run["documents"]}["FAILED.md"] == "SWORN-FAILED"
        for d in run["documents"]:
            assert (out / d["receipt"]).exists() and (out / d["document"]).exists()
        assert run["manifest"]["path"] == "sworn.manifest.json" and "out_dir" in run and "workspace" not in run
        assert "report-only until the measurement prices FAILED" in summary
        assert "the minting job is the claimant's" in summary

    def test_a_failing_command_and_a_timing_out_command_exit_zero(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, command='"%s" -c "import sys; print(\'boom\'); sys.exit(3)"' % sys.executable)
        assert code == 0 and run["exit_code"] == 3 and run["junit_present"] is False
        assert "r1 to r4 are absent" in summary
        code, out, summary, run = run_action(repo, monkeypatch, command='"%s" -c "import time; time.sleep(8)"' % sys.executable,
                                             SWORN_TIMEOUT_MINUTES="0.02")
        assert code == 0 and run["timed_out"] is True and run["exit_code"] == 124

    def test_an_empty_command_is_a_usage_error_exit_two(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, command="   ")
        assert code == 2 and summary == "" and run == {}

    def test_github_output_carries_the_outputs(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch)
        text = (repo["tmp"] / "github_output.txt").read_text(encoding="utf-8")
        assert "manifest<<" in text and "verdicts<<" in text and "rung<<" in text
        block = re.search(r"verdicts<<(\S+)\n(.*)\n\1\n", text)
        assert json.loads(block.group(2))["FAILED.md"] == "SWORN-FAILED"
        assert (repo["tmp"] / "step_summary.md").read_text(encoding="utf-8") == summary


class TestTheComposedManifest:
    def test_the_composed_manifest_is_the_adapters_entries_with_only_the_id_changed(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch)
        composed = json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))
        junit = json.loads((out / "junit.manifest.json").read_text(encoding="utf-8"))
        github = json.loads((out / "github.manifest.json").read_text(encoding="utf-8"))
        assert sorted(composed["receipts"]) == sorted(["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"])
        for rid in ("r1", "r2", "r3", "r4"):
            assert composed["receipts"][rid] == junit["receipts"][rid]
        for old, new in {"r1": "r5", "r2": "r6", "r3": "r7", "r4": "r8", "r5": "r9"}.items():
            want = dict(github["receipts"][old])
            want["id"] = new
            assert composed["receipts"][new] == want
        assert composed["rung"] == "L2" and junit["rung"] == "L2" and github["rung"] == "L2"
        assert composed["authored_sha256"] == junit["authored_sha256"]
        assert len(composed["authored_sha256"]) == len(_docs(repo["base"]))
        assert composed["turn"] == "lab/app#7@" + repo["head"]
        assert run["manifest"]["digest"] == composed["digest"]

    def test_no_report_leaves_r1_to_r4_absent_and_r5_to_r9_in_place(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, command='"%s" -c "print(1)"' % sys.executable)
        composed = json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))
        assert sorted(composed["receipts"]) == ["r5", "r6", "r7", "r8", "r9"]
        assert not (out / "junit.manifest.json").exists()
        rows = _rows(summary)
        assert rows["HELD.md"]["unresolved"] == 2 and rows["HELD.md"]["held"] == 2   # r6, r9 still resolve
        assert "- UNRESOLVED manifest_id_missing `r1` @" in summary

    def test_an_unparseable_report_keeps_r3_and_r4(self, repo, monkeypatch):
        (repo["tmp"] / "ok.xml").write_bytes(b"this is not xml\n")
        code, out, summary, run = run_action(repo, monkeypatch)
        composed = json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))
        assert sorted(composed["receipts"]) == ["r3", "r4", "r5", "r6", "r7", "r8", "r9"]
        assert _rows(summary)["REPORT.md"]["verdict"] == "SWORN-HELD"

    def test_a_laundered_report_reads_malformed_receipt_author_minted(self, repo, monkeypatch):
        # the command copies a file the turn committed into $SWORN_JUNIT: its bytes are in authored_sha256
        cmd = _command(repo).replace(str(repo["tmp"] / "ok.xml"), str(repo["ws"] / "committed.xml"))
        code, out, summary, run = run_action(repo, monkeypatch, command=cmd)
        assert code == 0
        assert _rows(summary)["REPORT.md"]["malformed"] == 1
        assert "- MALFORMED receipt_author_minted `r3` @" in summary

    def test_the_report_path_enters_r4_as_given(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, SWORN_JUNIT="sub/report.xml")
        composed = json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))
        import base64
        r4 = json.loads(base64.b64decode(composed["receipts"]["r4"]["bytes"]))
        assert r4["paths_requested"] == ["sub/report.xml"]
        assert (repo["ws"] / "sub" / "report.xml").exists()


class TestTheRung:
    def test_a_fork_pull_request_mints_l1_and_prints_the_fork_rule(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event=_event(repo, fork=True))
        assert code == 0 and run["rung"] == "L1"
        assert "On a pull request from a fork, the minting job is the claimant's." in run["rung_reason"]
        assert "**Rung L1** — On a pull request from a fork, the minting job is the claimant's." in summary
        rows = _rows(summary)
        assert rows and all(r["rung"] == "L1" for r in rows.values())
        for name in ("sworn", "junit", "github"):
            m = json.loads((out / ("%s.manifest.json" % name)).read_text(encoding="utf-8"))
            assert m["rung"] == "L1"
        assert "fork: true" in json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))["harness"]

    def test_a_head_repository_absent_from_the_payload_is_treated_as_a_fork(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event=_event(repo, head_repo=None))
        assert run["rung"] == "L1" and "treated as a fork" in run["rung_reason"]

    def test_a_fork_with_base_pinned_workflow_declared_mints_l2_as_the_adapter_accepts(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event=_event(repo, fork=True),
                                             SWORN_BASE_PINNED_WORKFLOW="true")
        assert run["rung"] == "L2" and run["rung_reason"] is None
        composed = json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))
        assert composed["rung"] == "L2" and "base-pinned-workflow=true" in composed["harness"]

    def test_l2_without_after_turn_on_base_mints_l1_with_the_reason(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, SWORN_AFTER_TURN_ON_BASE="false")
        assert run["rung"] == "L1" and "after-turn-on-base" in run["rung_reason"]
        assert "**Rung L1** — L2 was declared without after-turn-on-base" in summary

    def test_l1_declared_is_l1_with_no_reason(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, SWORN_RUNG="L1")
        assert run["rung"] == "L1" and run["rung_reason"] is None

    def test_a_rung_outside_the_ladder_is_lowered_never_raised(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, SWORN_RUNG="L3")
        assert code == 0 and run["rung"] == "L1" and "not in the ladder" in run["rung_reason"]


class TestEventsAndCheckouts:
    def test_a_push_event_uses_after_and_before(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event=_event(repo, event="push"), event_name="push")
        assert code == 0 and run["head"] == repo["head"] and run["base"] == repo["base"]
        assert run["turn"] == "lab/app@" + repo["head"] and run["rung"] == "L2"
        rows = _rows(summary)
        assert "pull_request_body.md" not in rows
        assert rows["FAILED.md"]["verdict"] == "SWORN-FAILED"
        # r9 is the push payload: the pointer /pull_request/number names no leaf in it, which the
        # verifier reads as the author's error (MALFORMED), never as a receipt it could not see
        assert rows["HELD.md"]["malformed"] == 1 and rows["HELD.md"]["held"] == 3
        assert "- MALFORMED " in summary and "`r9#/pull_request/number` @" in summary

    @pytest.mark.parametrize("name", ["pull_request_target", "workflow_dispatch", ""])
    def test_other_events_do_not_run_and_exit_zero(self, repo, monkeypatch, name):
        code, out, summary, run = run_action(repo, monkeypatch, event_name=name)
        assert code == 0 and "DID NOT RUN" in summary and "did_not_run" in run
        assert not (out / "sworn.manifest.json").exists()
        if name == "pull_request_target":
            assert "base repository's token" in summary

    def test_a_missing_head_commit_does_not_run(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event=_event(repo, head="e" * 40))
        assert code == 0 and "DID NOT RUN" in summary and "head.sha" in summary

    def test_a_payload_the_adapter_cannot_read_does_not_run(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event={"pull_request": {"number": 1}, "repository": {}})
        assert code == 0 and "DID NOT RUN" in summary

    def test_a_shallow_checkout_verifies_the_body_only_and_says_so(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, event=_event(repo, base="c" * 40))
        assert code == 0 and run["discovery"] == "unavailable"
        assert "changed-file discovery unavailable" in summary and "fetch-depth: 0" in summary
        rows = _rows(summary)
        assert list(rows) == ["pull_request_body.md"]
        composed = json.loads((out / "sworn.manifest.json").read_text(encoding="utf-8"))
        assert composed["authored_sha256"] == [] and "r5" not in composed["receipts"]


class TestTheDocuments:
    def test_the_body_is_verified_as_submitted_with_its_line_endings(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch)
        body = _event(repo)["pull_request"]["body"].encode("utf-8")
        assert (out / "documents" / "pull_request_body.md").read_bytes() == body
        assert b"\r\n" in body
        rec = json.loads((out / "receipts" / "pull_request_body.md.sworn-receipt.json").read_text(encoding="utf-8"))
        import hashlib
        assert rec["document"]["inline_sha256"] == hashlib.sha256(body).hexdigest()
        assert rec["commit"] == repo["head"]
        assert (repo["ws"] / "README.md").exists()          # $(rm -rf /) went nowhere near a shell

    def test_a_sidecar_and_an_untagged_markdown_are_skipped_and_named(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch)
        skipped = {s["path"]: s["why"] for s in run["skipped"]}
        assert "sidecar" in skipped["stale.sworn.json"] and "no <sworn tag" in skipped["plain.md"]
        assert "- not verified: `stale.sworn.json`" in summary
        assert "committed.xml" not in skipped

    def test_a_command_that_rewrites_a_document_on_disk_changes_no_verdict(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch, command=_command(repo, repo["ws"] / "HELD.md"))
        assert (repo["ws"] / "HELD.md").read_bytes() == b"rewritten on disk after the turn\n"
        assert _rows(summary)["HELD.md"]["verdict"] == "SWORN-HELD"
        assert (out / "documents" / "HELD.md").read_bytes().startswith(b"# held\n")

    def test_a_path_with_spaces_and_quotes_is_verified(self, repo, monkeypatch):
        ws = repo["ws"]
        name = "notes with spaces and 'quotes'.md"
        (ws / name).write_bytes(b'<sworn r="r2" k="numeric">0 failures.</sworn>\n')
        _git(ws, "add", ".")
        _git(ws, "commit", "-q", "-m", "more")
        repo["head"] = _git(ws, "rev-parse", "HEAD")
        code, out, summary, run = run_action(repo, monkeypatch)
        assert _rows(summary)[name]["verdict"] == "SWORN-HELD"


class TestOutputsAndSource:
    def test_every_output_is_lf_only(self, repo, monkeypatch):
        code, out, summary, run = run_action(repo, monkeypatch)
        for p in out.rglob("*"):
            if p.is_file() and p.name != "command.log" and not p.name.endswith(".xml") \
                    and p.name != "pull_request_body.md":
                assert b"\r" not in p.read_bytes(), p
        assert b"\r" not in (repo["tmp"] / "step_summary.md").read_bytes()

    def test_the_source_carries_no_error_annotation_and_no_network_import(self):
        src = ACTION.read_text(encoding="utf-8")
        assert "::error" not in src and "::warning" not in src
        assert "STYXX_STRICT" not in src and "SOFT_FAIL" not in src
        tree = ast.parse(src)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not names & {"urllib", "http", "socket", "requests", "ssl"}

    def test_nothing_under_styxx_imports_the_action(self):
        for p in (ROOT / "styxx").rglob("*.py"):
            assert "sworn_action" not in p.read_text(encoding="utf-8", errors="replace"), p

    def test_action_yml_has_no_gate_input_and_passes_inputs_through_env(self):
        y = (ACTION_DIR / "action.yml").read_text(encoding="utf-8")
        assert "strict" not in y and "soft-fail" not in y and "GH_TOKEN" not in y and "github.token" not in y
        assert 'python "${{ github.action_path }}/sworn_action.py"' in y
        for line in y.splitlines():
            if line.strip().startswith("run:"):
                assert "inputs." not in line, line
        for inp in ("command", "rung", "after-turn-on-base", "base-pinned-workflow", "junit", "out-dir", "timeout-minutes"):
            assert "  %s:" % inp in y
        assert "SWORN_COMMAND: ${{ inputs.command }}" in y

    def test_the_readme_opens_with_the_two_sentences_and_carries_the_layout(self):
        text = (ACTION_DIR / "README.md").read_text(encoding="utf-8")
        head = text[:600]
        assert "Report-only until the measurement prices FAILED." in head
        assert "On a pull request from a fork, the minting job is the claimant's" in head
        assert "declares rung L1 (weak), never L2" in text
        assert "Do not switch the trigger to `pull_request_target`" in text
        for rid in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"):
            assert "| `%s` |" % rid in text
        assert "an absence never renumbers" in text
        for word in ("self-verifying", "tamper-proof", "immutable"):
            assert word not in text.lower()

    def test_the_example_workflow_checks_out_the_head_with_full_history(self):
        y = (ACTION_DIR / "examples" / "sworn.yml").read_text(encoding="utf-8")
        assert "ref: ${{ github.event.pull_request.head.sha }}" in y and "fetch-depth: 0" in y
        assert "pull_request_target" not in y
        assert "contents: read" in y and "uses: ./sworn" in y
        assert not (ROOT / ".github" / "workflows" / "sworn.yml").exists()

    def test_the_root_action_is_untouched_by_this_leg(self):
        y = (ROOT / "action.yml").read_text(encoding="utf-8")
        assert "diffgate_action.py" in y and "sworn" not in y.lower()


class TestTheCommittedSample:
    # The sample the script reproduces TODAY. `sworn_action_sample.*` is the first one and is
    # history: it records what the action printed before the verify headline warned about a
    # SWORN-HELD document with unresolved spans, and two sworn documents cite it. When the verifier
    # changed, the script's own refusal named the remedy — "a sample is history; write a new prefix
    # at a new commit" — so a second sample was written beside the first rather than over it.
    CURRENT_SAMPLE = "sworn_action_sample_2026_09_06"

    def _sample_module(self):
        script = ROOT / "papers" / "sworn" / "sworn_action_sample.py"
        spec = importlib.util.spec_from_file_location("sworn_action_sample", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_committed_sample_reproduces(self):
        """The current sample is what the script produces today — manifests, run.json and the
        summary byte for byte; receipts on their core."""
        assert self._sample_module().check(self.CURRENT_SAMPLE) == 0

    def test_the_first_sample_is_still_here_and_is_not_regenerated(self):
        """History is kept, not rewritten. The first sample must remain on disk and must NOT
        reproduce — it cannot, because the headline it captured predates the warning, and a sample
        that quietly started reproducing again would mean somebody had rewritten it."""
        here = ROOT / "papers" / "sworn"
        first = sorted(p.name for p in here.glob("sworn_action_sample.*") if p.suffix != ".py")
        assert first, "the first sample was deleted; a sample is history"
        assert (here / "sworn_action_sample.summary.md").exists()
        assert self._sample_module().check("sworn_action_sample") != 0, (
            "the first sample reproduces again, which means either it was regenerated in place or "
            "the warning it predates was removed")

    def test_the_two_samples_differ_only_in_the_warning(self):
        """The pair IS the record of the change: same action, same fixture, one line apart."""
        here = ROOT / "papers" / "sworn"
        old = (here / "sworn_action_sample.summary.md").read_text(encoding="utf-8").splitlines()
        new = (here / ("%s.summary.md" % self.CURRENT_SAMPLE)).read_text(
            encoding="utf-8").splitlines()
        assert len(old) == len(new), "the samples differ in shape, not just in the warning"
        differing = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
        assert len(differing) == 1, "expected exactly one differing line, got %d" % len(differing)
        i = differing[0]
        assert "UNRESOLVED.md" in old[i] and "SWORN-HELD" in old[i]
        assert "WARNING" not in old[i], "the first sample predates the warning"
        assert "nothing was checked" in new[i], new[i]

    def test_the_sample_refuses_to_overwrite_itself(self, capsys):
        script = ROOT / "papers" / "sworn" / "sworn_action_sample.py"
        spec = importlib.util.spec_from_file_location("sworn_action_sample", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.main([]) == 1
        assert "REFUSED" in capsys.readouterr().err
