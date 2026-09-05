# -*- coding: utf-8 -*-
"""styxx.harness.github — an event payload and a diff the caller fetched, at the declared rung.

Built to papers/sworn/DESIGN_harness_adapters_2026_09_02.md. The fork caveat is on every
manifest; L2 needs the caller's declaration; a fork pull_request (or one whose head repository
is absent) needs the base-pinned declaration too; pull_request_target is accepted; push events
carry before/after; any other event name is refused; absent over an incomplete diff is
MALFORMED; the module reads no ambient state below main.

LOAD-BEARING: test_l2_on_a_fork_pull_request_needs_the_base_pinned_declaration,
test_the_module_reads_no_ambient_state_below_main.
"""
from __future__ import annotations

import ast
import base64
import json
from pathlib import Path

import pytest

import styxx.harness.github as G
from styxx.evidence import _FORBIDDEN_MODULES
from styxx.sworn import Manifest, verify

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def pr_event(fork=False, head_repo_absent=False, number=7):
    head_repo = None if head_repo_absent else {
        "full_name": "someone/styxx" if fork else "fathom-lab/styxx", "fork": bool(fork)}
    return {"action": "synchronize", "number": number,
            "pull_request": {"number": number,
                             "base": {"sha": BASE_SHA, "repo": {"full_name": "fathom-lab/styxx", "fork": False}},
                             "head": {"sha": HEAD_SHA, "repo": head_repo}}}


def push_event():
    return {"before": BASE_SHA, "after": HEAD_SHA, "ref": "refs/heads/main"}


def raw(event) -> bytes:
    return (json.dumps(event, indent=1) + "\n").encode("utf-8")


def mint(event, name="pull_request", **kw):
    args = dict(diff=None, diff_complete=False, rung="L1", ran_after_turn_on_base=False,
                base_pinned_workflow=False)
    args.update(kw)
    return G.mint(event, raw(event), name, **args)


def sp(text, receipt, kind="numeric"):
    return '<sworn r="%s" k="%s">%s</sworn>' % (receipt, kind, text)


def rb(m, rid):
    return base64.b64decode(m.receipts[rid]["bytes"])


class TestReceipts:
    def test_a_same_repo_pull_request_at_l2_records_shas_event_name_and_payload(self):
        m = mint(pr_event(), rung="L2", ran_after_turn_on_base=True)
        assert "r1" not in m.receipts
        assert rb(m, "r2") == BASE_SHA.encode() and rb(m, "r3") == HEAD_SHA.encode()
        assert rb(m, "r4") == b"pull_request"
        assert rb(m, "r5") == raw(pr_event())
        assert all(m.receipts[r]["kind_of_source"] == "harness_note" and m.receipts[r]["complete"]
                   for r in ("r2", "r3", "r4", "r5"))
        assert m.rung_status() == ("ok", "L2")
        assert "fork: false" in m.harness
        assert "fork caveat:" in m.harness and G.FORK_CAVEAT in m.harness
        assert "rung L2 declared by the caller, not detected" in m.harness

    def test_the_payload_is_a_leaf_bearing_receipt(self):
        m = mint(pr_event(number=42))
        s = verify((sp("pull request 42", "r5#/pull_request/number") + "\n").encode(),
                   name="d.md", manifest=m)["spans"][0]
        assert s["verdict"] == "HELD" and s["provenance"]["rung"] == "L1"
        s = verify((sp("the head is `\"fork\": false`", "r5#/pull_request/head/repo/fork", "quote")
                    + "\n").encode(), name="d.md", manifest=m)["spans"][0]
        assert (s["verdict"], s["reason"]) == ("MALFORMED", "leaf_not_string")

    def test_a_push_event_reads_before_and_after(self):
        m = mint(push_event(), name="push", rung="L2", ran_after_turn_on_base=True)
        assert rb(m, "r2") == BASE_SHA.encode() and rb(m, "r3") == HEAD_SHA.encode()
        assert rb(m, "r4") == b"push"
        assert "fork: false" in m.harness and "fork caveat:" in m.harness

    def test_a_diff_is_r1_with_the_callers_completeness(self):
        m = mint(pr_event(), diff=b"--- a\n+++ b\n", diff_complete=True)
        e = m.receipts["r1"]
        assert e["kind_of_source"] == "http_fetch" and e["complete"] is True
        assert "asserted by the caller, not observed" in e["harness_note"]

    def test_absent_over_an_incomplete_diff_is_malformed_and_over_a_complete_one_holds(self):
        doc = (sp("the diff carries no `TODO`", "r1", "absent") + "\n").encode()
        m = mint(pr_event(), diff=b"--- a\n+++ b\n", diff_complete=False)
        s = verify(doc, name="d.md", manifest=m)["spans"][0]
        assert (s["verdict"], s["reason"]) == ("MALFORMED", "absent_over_partial")
        m = mint(pr_event(), diff=b"--- a\n+++ b\n", diff_complete=True)
        assert verify(doc, name="d.md", manifest=m)["spans"][0]["verdict"] == "HELD"


class TestRefusals:
    def test_l2_without_the_after_turn_declaration_is_refused(self):
        with pytest.raises(ValueError, match="ran_after_turn_on_base"):
            mint(pr_event(), rung="L2")

    def test_l2_on_a_fork_pull_request_needs_the_base_pinned_declaration(self):
        """LOAD-BEARING. A fork's workflow is the claimant's; L2 rests on 'a party other than the
        claimant', and the event bytes cannot show the workflow was the base's."""
        with pytest.raises(ValueError, match="base_pinned_workflow"):
            mint(pr_event(fork=True), rung="L2", ran_after_turn_on_base=True)
        m = mint(pr_event(fork=True), rung="L2", ran_after_turn_on_base=True, base_pinned_workflow=True)
        assert m.rung_status() == ("ok", "L2") and "fork: true" in m.harness

    def test_a_fork_pull_request_mints_at_l1_without_any_declaration(self):
        m = mint(pr_event(fork=True))
        assert m.rung_status() == ("ok", "L1") and "fork: true" in m.harness and "weak" in m.harness

    def test_a_missing_head_repository_is_unknown_and_treated_as_a_fork(self):
        assert G.fork_status(pr_event(head_repo_absent=True), "pull_request") is None
        with pytest.raises(ValueError, match="head repository absent"):
            mint(pr_event(head_repo_absent=True), rung="L2", ran_after_turn_on_base=True)
        m = mint(pr_event(head_repo_absent=True))
        assert "fork: unknown" in m.harness

    def test_pull_request_target_from_a_fork_mints_l2_on_the_after_turn_declaration_alone(self):
        m = mint(pr_event(fork=True), name="pull_request_target", rung="L2", ran_after_turn_on_base=True)
        assert m.rung_status() == ("ok", "L2") and rb(m, "r4") == b"pull_request_target"

    def test_an_unknown_event_name_and_a_reserved_rung_are_refused(self):
        with pytest.raises(ValueError, match="event name"):
            mint(pr_event(), name="issues")
        with pytest.raises(ValueError, match="L3"):
            mint(pr_event(), rung="L3")

    def test_a_payload_without_shas_is_refused(self):
        with pytest.raises(ValueError):
            mint({"pull_request": {"base": {}, "head": {}}})
        with pytest.raises(ValueError):
            mint({"ref": "x"}, name="push")


class TestPurity:
    def test_the_module_reads_no_ambient_state_below_main(self):
        """LOAD-BEARING. Only main may read the environment (argparse defaults). Nothing else in
        the module imports or references the modules styxx.evidence forbids."""
        tree = ast.parse(Path(G.__file__).read_text(encoding="utf-8"))
        banned = set(_FORBIDDEN_MODULES)
        top_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".")[0])
        assert not (top_imports & banned), sorted(top_imports & banned)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name == "main":
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Import):
                    assert not ({a.name.split(".")[0] for a in node.names} & banned), fn.name
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    assert node.value.id not in banned, "%s references %s.%s" % (fn.name, node.value.id, node.attr)
                    assert "%s.%s" % (node.value.id, node.attr) not in ("Path.cwd", "Path.home"), fn.name


class TestCLI:
    def test_main_writes_an_lf_only_manifest(self, tmp_path, capsys):
        ev = tmp_path / "event.json"
        ev.write_bytes(raw(pr_event()))
        diff = tmp_path / "pr.diff"
        diff.write_bytes(b"--- a\n+++ b\n")
        out = tmp_path / "m.json"
        assert G.main(["--event", str(ev), "--event-name", "pull_request", "--diff", str(diff),
                       "--diff-complete", "--rung", "L2", "--after-turn-on-base", "--turn", "t",
                       "--out", str(out)]) == 0
        assert "minted" in capsys.readouterr().out
        assert b"\r" not in out.read_bytes()
        m = Manifest.load(out)
        assert m.intact() and m.rung_status() == ("ok", "L2") and m.receipts["r1"]["complete"] is True

    def test_a_refused_declaration_is_exit_two_and_no_manifest(self, tmp_path, capsys):
        ev = tmp_path / "event.json"
        ev.write_bytes(raw(pr_event(fork=True)))
        out = tmp_path / "m.json"
        assert G.main(["--event", str(ev), "--event-name", "pull_request", "--rung", "L2",
                       "--after-turn-on-base", "--out", str(out)]) == 2
        assert not out.exists() and "usage" in capsys.readouterr().err

    def test_without_event_arguments_or_environment_it_is_a_usage_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        assert G.main(["--rung", "L1", "--out", str(tmp_path / "m.json")]) == 2

    def test_the_environment_supplies_the_defaults_only_at_the_command_line(self, tmp_path, monkeypatch):
        ev = tmp_path / "event.json"
        ev.write_bytes(raw(push_event()))
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
        monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
        out = tmp_path / "m.json"
        assert G.main(["--rung", "L1", "--out", str(out)]) == 0
        assert rb(Manifest.load(out), "r4") == b"push"
