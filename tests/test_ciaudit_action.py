# -*- coding: utf-8 -*-
"""The GitHub Action for `styxx ci-audit` (`ci-audit/action.yml`, `styxx/ciaudit/action.py`).

A local bare remote stands in for GitHub: a default branch, `refs/pull/N/head` as pushed, and
`refs/pull/N/merge` as GitHub's test merge, with fetch-by-sha allowed as GitHub allows it. The
workspace is made the way `actions/checkout` makes it -- the event's sha fetched at depth 1 and
checked out detached -- so the shallow path is the one tested. The tests hold: the comparison each
event implies; the gate's verdict and exit status; the annotation on the exact line; the job
summary with the verified repair; the outputs and the receipt; that a gate that cannot run never
passes; that the change's own text cannot act as a workflow command; that tokens leave the
environment before the gate reads anything; the one-click suggestion -- applied, it is the
verified repair, line for line -- posted once and never twice; and that the action's own steps
pass the gate they run."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from styxx.ciaudit import action as A  # noqa: E402

CI = """on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        run: python -m pytest tests -q
      - name: Lint
        run: npm run lint
"""
CI_PR7 = """on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        run: python -m pytest tests -q
      - name: Lint
        continue-on-error: true
        run: npm run lint

      - name: Typecheck
        run: npx tsc --noEmit || true
"""
CI_PR8 = CI + """      - name: Audit
        run: npm audit
"""
ENV = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True, text=True, env=dict(os.environ, **ENV)).stdout.strip()


def _github(tmp_path: Path) -> dict:
    src = tmp_path / "src"
    (src / ".github" / "workflows").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
    wf = src / ".github" / "workflows" / "ci.yml"

    def commit(msg: str) -> str:
        _git(src, "add", "-A")
        _git(src, "commit", "-q", "-m", msg)
        return _git(src, "rev-parse", "HEAD")

    s = {}
    wf.write_text(CI)
    s["c0"] = commit("ci")
    _git(src, "checkout", "-q", "-b", "pr7")
    wf.write_text(CI_PR7)
    s["h7"] = commit("ci: lint non-blocking, typecheck best-effort")
    _git(src, "checkout", "-q", "main")
    (src / "README.md").write_text("moved on\n")
    s["c1"] = commit("docs")                                     # the base moves after the branch is cut; ci.yml does not
    _git(src, "checkout", "-q", "--detach", s["c1"])
    _git(src, "merge", "-q", "--no-ff", "-m", "Merge pr7 into c1 (GitHub's test merge)", s["h7"])
    s["m7"] = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-q", "-b", "pr8", s["c1"])
    wf.write_text(CI_PR8)
    s["h8"] = commit("ci: audit")
    _git(src, "checkout", "-q", "--detach", s["c1"])
    _git(src, "merge", "-q", "--no-ff", "-m", "Merge pr8 (test merge)", s["h8"])
    s["m8"] = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-q", "main")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    _git(remote, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(src, "push", "-q", str(remote), "main", f"{s['h7']}:refs/pull/7/head", f"{s['m7']}:refs/pull/7/merge",
         f"{s['h8']}:refs/pull/8/head", f"{s['m8']}:refs/pull/8/merge")
    s["remote"] = remote
    return s


def _checkout(tmp_path: Path, s: dict, sha: str, name: str = "ws") -> Path:
    """What actions/checkout does: the sha at depth 1, detached."""
    ws = tmp_path / name
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    _git(ws, "remote", "add", "origin", f"file://{s['remote']}")
    _git(ws, "fetch", "--quiet", "--no-tags", "--depth=1", "origin", f"+{sha}:refs/remotes/checkout")
    _git(ws, "checkout", "-q", "--detach", sha)
    assert _git(ws, "rev-parse", "--is-shallow-repository") == "true"
    return ws


def _run(monkeypatch, capsys, tmp_path: Path, ws: Path, event_name: str, event: dict, sha: str, **inputs) -> dict:
    ev = tmp_path / f"event-{event_name}.json"
    ev.write_text(json.dumps(event), encoding="utf-8")
    summary, output = tmp_path / "summary.md", tmp_path / "output.txt"
    for f in (summary, output):
        if f.exists():
            f.unlink()
    for k, v in {"GITHUB_WORKSPACE": str(ws), "GITHUB_EVENT_NAME": event_name, "GITHUB_EVENT_PATH": str(ev), "GITHUB_SHA": sha,
                 "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_OUTPUT": str(output), "RUNNER_TEMP": str(tmp_path),
                 "GITHUB_REPOSITORY": "org/repo", "GH_TOKEN": "secret-token-value", "ACTIONS_RUNTIME_TOKEN": "runtime-token-value",
                 "STYXX_FAIL_ON": inputs.get("fail_on", "new-hidden"), "STYXX_SUGGEST": inputs.get("suggest", "false"),
                 "STYXX_ANNOTATE": inputs.get("annotate", "true")}.items():
        monkeypatch.setenv(k, v)
    rc = A.main()
    out = capsys.readouterr().out
    return {"rc": rc, "out": out, "summary": summary.read_text(encoding="utf-8") if summary.exists() else "",
            "outputs": dict(ln.split("=", 1) for ln in output.read_text(encoding="utf-8").splitlines()) if output.exists() else {},
            "env_tokens": [k for k in ("GH_TOKEN", "ACTIONS_RUNTIME_TOKEN") if k in os.environ]}


def _pr_event(n: int, head: str, base: str) -> dict:
    return {"number": n, "pull_request": {"number": n, "head": {"sha": head}, "base": {"sha": base}}}


def _line(text: str, needle: str) -> int:
    return next(i for i, ln in enumerate(text.splitlines(), 1) if needle in ln)


# ----------------------------------------------------------------------------- the comparison and the verdict

def test_the_test_merge_fires_on_the_exact_lines(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    ws = _checkout(tmp_path, s, s["m7"])
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"])
    assert r["rc"] == 1
    # the annotation sits on the line that hides the check: the continue-on-error key; the `run:` line of the `|| true`
    coe, tsc = _line(CI_PR7, "continue-on-error: true"), _line(CI_PR7, "npx tsc --noEmit || true")
    ann = [ln for ln in r["out"].splitlines() if ln.startswith("::error ")]
    assert len(ann) == 2
    assert ann[0].startswith(f"::error file=.github/workflows/ci.yml,line={coe},title=styxx ci-audit — a check that hides its own failure::SWALLOWED: the step 'Lint'")
    assert "acquired: continue-on-error" in ann[0] and "Verified repair: no-continue-on-error (1 line" in ann[0]
    assert ann[1].startswith(f"::error file=.github/workflows/ci.yml,line={tsc},title=") and "born hidden" in ann[1] and "Verified repair: strict-shell" in ann[1]
    # the job summary: the verdict, the lines, and the repairs as diffs
    assert "### styxx ci-audit — 2 checks newly hidden: the gate fires" in r["summary"]
    assert "GitHub's test merge, against its first parent (the base branch)" in r["summary"]
    assert f"(line {coe})" in r["summary"] and "```diff" in r["summary"] and "-        continue-on-error: true" in r["summary"]
    assert r["outputs"]["fires"] == "true" and r["outputs"]["new-hidden"] == "2" and r["outputs"]["measured"] == "true"
    rec = json.loads(Path(r["outputs"]["receipt"]).read_text())
    assert rec["base"] == s["c1"] and rec["head"] == s["m7"] and rec["reading"] == "test-merge" and rec["pr"] == 7
    assert rec["lines"][".github/workflows/ci.yml::test::1"][:2] == [coe, coe]
    # tokens are gone from the environment before the gate reads the change's text
    assert r["env_tokens"] == [] and "secret-token-value" not in r["out"] + r["summary"] + json.dumps(rec)


def test_a_quiet_change_passes_and_fail_on_never_reports_only(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, s["m8"]), "pull_request", _pr_event(8, s["h8"], s["c1"]), s["m8"])
    assert r["rc"] == 0 and "nothing newly hidden" in r["summary"] and r["outputs"]["fires"] == "false" and "::error" not in r["out"]
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, s["m7"], "ws7"), "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"], fail_on="never")
    assert r["rc"] == 0 and r["outputs"]["fires"] == "true"
    assert "::error" not in r["out"] and r["out"].count("::warning file=.github/workflows/ci.yml,line=") == 2


def test_a_head_checkout_reads_the_merge_ref(tmp_path, monkeypatch, capsys):
    """A workflow that checks out the pull request's head: the action fetches refs/pull/N/merge itself."""
    s = _github(tmp_path)
    ws = _checkout(tmp_path, s, s["h7"])
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"])
    assert r["rc"] == 1 and r["outputs"]["new-hidden"] == "2"
    rec = json.loads(Path(r["outputs"]["receipt"]).read_text())
    assert rec["reading"] == "pr-ref" and rec["base"] == s["c1"] and rec["head"] == s["m7"]


def test_push_and_merge_group(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, s["m7"]), "push", {"before": s["c1"], "after": s["m7"]}, s["m7"])
    assert r["rc"] == 1 and r["outputs"]["new-hidden"] == "2" and "the push: its old tip against its new tip" in r["summary"]
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, s["m7"], "ws2"), "merge_group", {"merge_group": {"base_sha": s["c1"], "head_sha": s["m7"]}}, s["m7"])
    assert r["rc"] == 1 and "the merge queue's group, against its base" in r["summary"]
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, s["m7"], "ws3"), "push", {"before": A.ZERO, "after": s["m7"]}, s["m7"])
    assert r["rc"] == 0 and "a new branch" in r["out"] and r["outputs"]["measured"] == "false"
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, s["m7"], "ws4"), "workflow_dispatch", {}, s["m7"])
    assert r["rc"] == 0 and "not gated" in r["summary"]


def test_a_gate_that_cannot_run_never_passes(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    ws = _checkout(tmp_path, s, s["m7"])
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request_target", _pr_event(7, s["h7"], s["c1"]), s["m7"])
    assert r["rc"] == 2 and "::error title=styxx ci-audit — the gate did not run::" in r["out"] and "refused on pull_request_target" in r["out"]
    assert "UNMEASURED" in r["summary"] and "This is not a pass." in r["summary"] and r["outputs"]["measured"] == "false"
    r = _run(monkeypatch, capsys, tmp_path, ws, "merge_group", {"merge_group": {"base_sha": "d" * 40}}, s["m7"])
    assert r["rc"] == 2 and "could not fetch dddddddddddd" in r["out"]
    r = _run(monkeypatch, capsys, tmp_path, ws, "merge_group", {"merge_group": {"base_sha": "d" * 40}}, s["m7"], fail_on="never")
    assert r["rc"] == 0 and "::warning title=styxx ci-audit — the gate did not run::" in r["out"]
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"], fail_on="sometimes")
    assert r["rc"] == 2 and "fail-on must be new-hidden or never" in r["out"]


def test_the_changes_own_text_cannot_act_as_a_workflow_command(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    # a real newline (the YAML escape) and a literal %0A in the step's name, a workflow command in its script
    evil = CI_PR7.replace("- name: Lint\n", "- name: \"Lint\\n::add-mask::x %0A::debug::y, :: 100%\"\n").replace("run: npm run lint", "run: npm run lint -- '::set-output name=x::y'")
    src = tmp_path / "src"
    _git(src, "checkout", "-q", "--detach", s["c1"])
    (src / ".github" / "workflows" / "ci.yml").write_text(evil)
    _git(src, "commit", "-qam", "evil")
    head = _git(src, "rev-parse", "HEAD")
    _git(src, "push", "-q", str(s["remote"]), f"{head}:refs/pull/9/merge")
    r = _run(monkeypatch, capsys, tmp_path, _checkout(tmp_path, s, head), "push", {"before": s["c1"], "after": head}, head)
    lines = r["out"].splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("::stop-commands::"))
    token = lines[start].split("::")[2]
    stop = lines.index(f"::{token}::")
    assert len(token) == 32 and start < stop
    outside = lines[:start] + lines[stop + 1:]
    for ln in outside:                                                  # outside the stopped region, every command is one we built
        if ln.startswith("::"):
            assert re.match(r"^::(error|warning) (file=[^,:]+,line=\d+(,endLine=\d+)?,)?title=[^:,]+::", ln), ln
    lint = next(ln for ln in outside if "the step 'Lint" in ln)
    assert "Lint%0A::add-mask::x %250A::debug::y" in lint                 # the newline escaped; the literal %0A escaped so it cannot decode to one
    assert any("::set-output" in ln for ln in lines[start + 1:stop])     # the script's text is printed only inside the stopped region


# ----------------------------------------------------------------------------- the one-click repair

@pytest.mark.parametrize("text,new", [
    ("a\nb\nc\n", "a\nc\n"),                                   # a deletion: an empty suggestion
    ("a\nb\nc\n", "a\nB1\nB2\nc\n"),                           # a replacement by more lines
    ("a\nb\nc\n", "a\nb\nx\nc\n"),                             # a pure insertion, anchored above
    ("a\nb\n", "x\na\nb\n"),                                   # a pure insertion at the top, anchored below
    ("a\nb\nc\nd\ne\n", "A\nb\nc\nd\nE\n"),                    # two changes apart: one span
])
def test_a_suggestion_applied_is_the_new_text(text, new):
    sg = A.suggestion(text, new)
    assert sg["start_line"] <= sg["line"] and A.apply_suggestion(text, sg) == new


def test_the_suggestion_is_the_verified_repair_line_for_line(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    ws = _checkout(tmp_path, s, s["m7"])
    with pytest.raises(RuntimeError, match="is not in the clone"):
        A.readable(ws, s["c1"], s["m7"])                                  # the base is not fetched yet: loud, not an empty change
    A.ensure(ws, s["c1"])
    A.readable(ws, s["c1"], s["m7"])
    rec = A.D.audit_commit(ws, s["c1"], s["m7"], readers={}, fix=True)
    text = A.D._text(ws, s["m7"], ".github/workflows/ci.yml")
    got = {}
    for x in rec["workflows"][0]["new_hidden"]:
        new = A.repaired_text(text, "ci.yml", x)
        assert new is not None and A.R.unified_diff(text, new, "ci.yml") == x["fix"]["diff"]
        sg = A.suggestion(text, new)
        assert A.apply_suggestion(text, sg).splitlines() == new.splitlines()
        got[x["name"]] = sg
    coe = _line(CI_PR7, "continue-on-error: true")
    assert got["Lint"] == {"start_line": coe, "line": coe, "lines": [], "replaces": ["        continue-on-error: true"]}
    tsc = _line(CI_PR7, "npx tsc --noEmit || true")
    assert got["Typecheck"]["start_line"] == tsc and got["Typecheck"]["line"] == tsc
    assert got["Typecheck"]["lines"] == ["        run: |", "          set -eo pipefail", "          npx tsc --noEmit"]


def test_suggestions_are_posted_once_and_never_twice(tmp_path, monkeypatch, capsys):
    s = _github(tmp_path)
    ws = _checkout(tmp_path, s, s["m7"])
    posted: list = []
    calls: list = []

    def fake_api(method, url, token, payload=None):
        calls.append((method, url, token))
        if method == "GET":
            return 200, [{"body": p["body"]} for p in posted]
        posted.append(payload)
        return 201, {"id": len(posted)}

    monkeypatch.setattr(A, "api", fake_api)
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"], suggest="true")
    assert r["rc"] == 1 and len(posted) == 2 and "2 one-click suggestions posted on the pull request." in r["summary"]
    assert all(t == "secret-token-value" for _, _, t in calls) and all(u.startswith("https://api.github.com/repos/org/repo/pulls/7/comments") for _, u, _ in calls)
    coe = _line(CI_PR7, "continue-on-error: true")
    p0 = next(p for p in posted if p["line"] == coe)
    assert p0["commit_id"] == s["h7"] and p0["path"] == ".github/workflows/ci.yml" and p0["side"] == "RIGHT" and "start_line" not in p0
    assert "```suggestion\n\n```" in p0["body"] and "<!-- styxx-ci-audit:suggest:" in p0["body"] and "loud" not in p0["body"].split("```suggestion")[1].split("```")[0]
    p1 = next(p for p in posted if p is not p0)
    assert "```suggestion\n        run: |\n          set -eo pipefail\n          npx tsc --noEmit\n```" in p1["body"]
    # a re-run finds its own marks and posts nothing
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"], suggest="true")
    assert len(posted) == 2 and "already suggested on this pull request" in r["out"]
    # a fork's read-only token: reported, and the gate's verdict does not change
    monkeypatch.setattr(A, "api", lambda method, url, token, payload=None: (200, []) if method == "GET" else (403, "Resource not accessible by integration"))
    r = _run(monkeypatch, capsys, tmp_path, ws, "pull_request", _pr_event(7, s["h7"], s["c1"]), s["m7"], suggest="true")
    assert r["rc"] == 1 and "the token cannot write to pull requests" in r["out"]


def test_hunks_are_the_lines_a_review_comment_can_sit_on(tmp_path):
    s = _github(tmp_path)
    ws = _checkout(tmp_path, s, s["m7"])
    A.ensure(ws, s["c1"])
    h = A.hunks(ws, s["c1"], s["m7"], ".github/workflows/ci.yml")
    coe, tsc = _line(CI_PR7, "continue-on-error: true"), _line(CI_PR7, "npx tsc --noEmit || true")
    assert A.within((coe, coe), h) and A.within((tsc, tsc), h) and not A.within((1, 1), h)


# ----------------------------------------------------------------------------- the action itself

def _action() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "ci-audit" / "action.yml").read_text(encoding="utf-8"))


def test_the_actions_own_steps_pass_the_gate_they_run(tmp_path):
    """The composite action's steps, as a workflow, read by the engine: nothing hidden, nothing dropped."""
    import yaml
    from styxx import ciaudit
    steps = _action()["runs"]["steps"]
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "action.yml").write_text(yaml.safe_dump({"on": "pull_request", "jobs": {"action": {"runs-on": "ubuntu-latest", "steps": steps}}}))
    rec = ciaudit.audit(str(tmp_path))
    assert rec["summary"]["hidden"] == 0 and rec["summary"]["dropped"] == 0


def test_no_input_or_event_text_is_interpolated_into_a_shell():
    for st in _action()["runs"]["steps"]:
        if "run" in st:
            assert "${{" not in st["run"], st["run"]
            assert st["run"].lstrip().startswith("set -euo pipefail")
    ins = _action()["inputs"]
    assert ins["fail-on"]["default"] == "new-hidden" and ins["suggest"]["default"] == "false" and ins["styxx-version"]["default"] == "action"


def test_the_action_runs_with_only_numpy_and_pyyaml():
    """`styxx-version: action` installs numpy and PyYAML and puts the action's own ref on PYTHONPATH:
    the driver must import on that alone."""
    code = ("import sys, importlib.abc\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name.split('.')[0] in ('sklearn', 'scipy', 'pandas', 'torch', 'requests', 'mcp'):\n"
            "            raise ImportError('blocked: ' + name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import styxx.ciaudit.action as a\n"
            "print(a.SCHEMA)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert p.returncode == 0 and "styxx.ci-audit-action/v1" in p.stdout, p.stderr[-500:]
