# -*- coding: utf-8 -*-
"""SWALLOW-13's third stage of `styxx ci-audit --repair` (`styxx/ciaudit/repair_frontier.py`): the
frontier's edits on text, each verified end to end on a workflow -- loud under the same fault, the
healthy run unchanged -- the two readings of what no edit repairs, and the stage wired into the
command, the pull request's gate and the Action."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from styxx.ciaudit import repair_frontier as F  # noqa: E402

FRONTIER_FIXTURE = """on: [push]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify the changelog
        run: |
          echo "entries: $(./scripts/check-changelog.sh)"
      - name: Run tests
        run: npx jest &> /dev/null || exit 0
      - name: Test server
        run: |
          pytest tests/server &
          echo started
      - name: Verify the browser binary
        run: |
          ls ~/.cache/chromium/chrome 2>/dev/null || \\
          echo "check completed"
      - name: Lint
        continue-on-error: true
        run: echo "config=$(npx eslint --print-config index.js)" >> $GITHUB_OUTPUT
  ratchet:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint baseline
        run: |
          LINT_EXIT=0
          npx eslint . || LINT_EXIT=$?
          echo "LINT_EXIT=$LINT_EXIT" >> $GITHUB_ENV
      - name: Ratchet
        run: |
          if [ "$LINT_EXIT" != "0" ] && [ -f .lint-baseline ]; then exit 1; fi
"""


def _tree(tmp_path: Path, text: str) -> Path:
    wf = tmp_path / "repo" / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text(text, encoding="utf-8")
    return tmp_path / "repo"


# ----------------------------------------------------------------------------- reading a line

def test_scan_finds_the_outermost_substitutions_and_masks_what_is_quoted():
    subs, masked = F.scan('echo "a $(b "c") d" && e $((1+2)) # $(x)')
    assert subs == [(8, 16)]                                   # `$(b "c")`; `$((` is arithmetic, the comment is not read
    assert masked == 'echo ______________ && e $((1+2)) ______'    # the operators outside quotes are what is left to read
    assert F.scan("echo 'unclosed") is None and F.scan('echo "$(a b') is None


# ----------------------------------------------------------------------------- the edits, on text

def test_hoist_substitution_moves_a_status_the_line_throws_away_onto_a_line_of_its_own():
    assert F.hoist_substitution('echo "v=$(tool --x)" >> $GITHUB_OUTPUT\n') == \
        'set -eo pipefail\n__sub1="$(tool --x)"\necho "v=${__sub1}" >> $GITHUB_OUTPUT\n'
    assert F.hoist_substitution('for f in $(ls *.txt); do cat "$f"; done\n') == \
        'set -eo pipefail\n__sub1="$(ls *.txt)"\nfor f in ${__sub1}; do cat "$f"; done\n'
    assert F.hoist_substitution('export TOKEN=$(gh auth token)\nif [ "$(git status --porcelain)" != "" ]; then\n  exit 1\nfi\n') == \
        'set -eo pipefail\n__sub1="$(gh auth token)"\nexport TOKEN=${__sub1}\n__sub2="$(git status --porcelain)"\nif [ "${__sub2}" != "" ]; then\n  exit 1\nfi\n'
    # a line that is more than one command keeps its meaning only as it is; a heredoc body and a continued line are left alone
    for run in ('echo "$(a)" && b\n', 'echo "$(a)" | tee x\n', 'elif [ "$(a)" = b ]; then\n', 'X=$(a)\n', 'cat <<EOF\necho "$(b)"\nEOF\n',
                'echo "$(c)" \\\n  more\n'):
        assert F.hoist_substitution(run) == run, run
    # a name the script already uses is not reused
    assert F.hoist_substitution('echo "__sub1 $(a)"\n') == 'set -eo pipefail\n__sub2="$(a)"\necho "__sub1 ${__sub2}"\n'


def test_a_hoisted_answer_by_status_keeps_its_no(tmp_path):
    """grep's, diff's, `git diff --exit-code`'s, `jq -e`'s status 1 is an answer; hoisted, it stays one:
    only a status above 1 stops the step. The model's stubs never answer 1, so the repair is verified
    as before (added after SWALLOW-13's run)."""
    assert F.hoist_substitution('echo "todos=$(grep -c TODO src/main.c)" >> $GITHUB_OUTPUT\n') == \
        'set -eo pipefail\n__sub1="$(grep -c TODO src/main.c)" || [ $? -eq 1 ]\necho "todos=${__sub1}" >> $GITHUB_OUTPUT\n'
    assert "|| [ $? -eq 1 ]" in F.hoist_substitution('if [[ $(git diff --exit-code) ]]; then\n  exit 1\nfi\n')
    for run in ('echo "d=$(git diff --stat)"\n', 'echo "m=$(cat m.json | jq -c .)"\n', 'echo "x=$(ls | grep_helper)"\n'):
        assert "|| [ $? -eq 1 ]" not in F.hoist_substitution(run), run
    from styxx.ciaudit import engine
    text = """on: [push]
jobs:
  t:
    runs-on: ubuntu-latest
    steps:
      - name: Verify no TODOs
        run: echo "todos=$(grep -c TODO src/main.c)" >> $GITHUB_OUTPUT
"""
    rec = F.try_frontier(text, "ci.yml", "t", 0, engine.Runner())
    assert rec["verified_repair"] == "hoist-substitution"
    assert '+          __sub1="$(grep -c TODO src/main.c)" || [ $? -eq 1 ]' in next(c for c in rec["candidates"] if c["repair"] == "hoist-substitution")["diff"]


def test_background_liveness_waits_on_a_job_that_has_already_died():
    assert F.background_liveness('  npm start &\n  npx wait-on http://localhost:3000\n') == (
        '  npm start &\n  __bg1=$!\n  sleep 5\n  if ! kill -0 "$__bg1" 2>/dev/null; then wait "$__bg1" || exit $?; fi\n'
        '  npx wait-on http://localhost:3000\n')
    assert F.background_liveness('npx jest &> /dev/null\necho x &\n') == 'npx jest &> /dev/null\necho x &\n'   # a redirection is not a job; an echo is no check
    assert F.background_liveness('server >log 2>&1 &\n').startswith('server >log 2>&1 &\n__bg1=$!\n')


def test_no_exit_zero_and_no_default_joined():
    assert F.no_exit_zero('npx jest || exit 0\necho "|| exit 0"\nmake || exit 1\n') == 'set -eo pipefail\nnpx jest\necho "|| exit 0"\nmake || exit 1\n'
    assert F.no_exit_zero('make || exit 1\n') == 'make || exit 1\n'
    assert F.no_default_joined('ls a 2>/dev/null || \\\necho "done"\n') == 'set -eo pipefail\nls a 2>/dev/null\n'
    # a fallback on a continuation line of its own is SWALLOW-5's no-default, not this edit
    assert F.no_default_joined('ls a \\\n  || echo "done"\n') == 'ls a \\\n  || echo "done"\n'
    assert F.transform("make\n", "no-exit-zero") == (None, "no `|| exit 0`")


# ----------------------------------------------------------------------------- verified, end to end

def test_every_edit_is_verified_on_a_workflow_and_the_rest_is_read(tmp_path, capsys):
    from styxx import ciaudit
    from styxx.ciaudit import main
    tree = _tree(tmp_path, FRONTIER_FIXTURE)
    rec = ciaudit.audit(str(tree), repair=True)
    by = {t["name"]: t for t in rec["repairs"]}
    assert {n: t["verified_repair"] for n, t in by.items()} == {
        "Verify the changelog": "hoist-substitution", "Run tests": "no-exit-zero", "Test server": "background-liveness",
        "Verify the browser binary": "no-default-joined", "Lint": "no-coe+hoist-substitution", "Lint baseline": None}
    for n, t in by.items():                                   # the first two stages ran first, and verified none of these
        assert [c["repair"] for c in t["candidates"]] == rec["repair_catalogue"]
        assert not any(c.get("verified") for c in t["candidates"][:6])
        if t["verified_repair"]:
            c = next(c for c in t["candidates"] if c["repair"] == t["verified_repair"])
            assert c["loud"] and c["unchanged"] and all(c["loud_by_flavour"].values()) and all(c["unchanged_by_flavour"].values())
    ro = by["Lint baseline"]["readings"]["routed"]
    assert ro["flags"] == ["LINT_EXIT"] and ro["exported"] == ["LINT_EXIT"] and ro["a_reader_can_fail"]
    assert [(r["job"], r["step"], r["step_name"], r["via"]) for r in ro["readers"]] == [("ratchet", 2, "Ratchet", "GITHUB_ENV")]
    assert by["Lint baseline"]["readings"]["declared"] is None
    assert rec["summary"]["repairs"]["verified"] == 5 and rec["summary"]["repairs"]["routed"] == 1 and rec["summary"]["repairs"]["declared"] == 0
    assert main([str(tree), "--repair"]) == 1
    out = capsys.readouterr().out
    assert "ci.yml › ci › Run tests — no-exit-zero, 4 lines" in out
    assert '+          if ! kill -0 "$__bg1" 2>/dev/null; then wait "$__bg1" || exit $?; fi' in out
    assert "ci.yml › ratchet › Lint baseline — no verified repair:" in out
    assert "reading: its failure is routed, not lost: LINT_EXIT via GITHUB_ENV, read by 'Ratchet' (job ratchet) -- a reader can fail the job" in out


def test_the_readings():
    import yaml
    doc = yaml.safe_load("""on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - id: probe
        name: Probe
        run: |
          ok=true
          npm test || ok=false
          echo "ok=$ok" >> "$GITHUB_OUTPUT"
      - name: Say
        if: steps.probe.outputs.ok == 'false'
        run: echo "::warning::tests failed (non-blocking)"
      - name: Unread
        run: |
          rc=0; pytest || rc=$?
          echo "rc=$rc" >> $GITHUB_ENV
      - name: Loglevel
        run: npm config set loglevel warn && npm test
""")
    ro = F.routed(doc, "a", 0)
    assert ro["exported"] == ["ok"] and not ro["a_reader_can_fail"]
    assert [(r["step_name"], r["conditions"], r["can_fail"]) for r in ro["readers"]] == [("Say", True, False)]
    assert F.routed(doc, "a", 2) is None                    # written, never read: not routed
    assert F.declared(doc, "a", 1) == "::warning" and F.declared(doc, "a", 3) is None     # `loglevel warn` declares nothing
    assert F.say({"routed": None, "declared": "non-blocking"}) == "the script says 'non-blocking'"
    assert F.say({"routed": None, "declared": None}) is None


# ----------------------------------------------------------------------------- wired into the gate and the Action

def test_a_third_stage_repair_is_rebuilt_by_name_where_the_gate_and_the_action_rebuild_one():
    from styxx.ciaudit import action as A
    from styxx.ciaudit import differential as D
    from styxx.ciaudit import engine
    from styxx.ciaudit import repair as R
    from styxx.ciaudit import repair_structural as RS
    text = FRONTIER_FIXTURE
    for name, i in (("no-exit-zero", 2), ("background-liveness", 3), ("no-coe+hoist-substitution", 5)):
        assert R.apply_repair(text, "ci", i, name) == F.apply_frontier(text, "ci", i, name)
    assert R.apply_repair(text, "ci", 2, "guard-status") == RS.apply_structural(text, "ci", 2, "guard-status")
    assert R.apply_repair(text, "ci", 2, "no-such-repair") == (None, "unknown repair: no-such-repair")
    fx = D.fix_for(text, "ci.yml", "ci", 2, engine.Runner())
    assert fx["verified_repair"] == "no-exit-zero" and fx["stage"] == "swallow-13" and fx["lines_changed"] == 4
    new = A.repaired_text(text, "ci.yml", {"job": "ci", "index": 2, "fix": fx})
    assert new is not None and "npx jest &> /dev/null || exit 0" not in new and A.suggestion(text, new) is not None
    ro = D.fix_for(text, "ci.yml", "ratchet", 1, engine.Runner())
    assert ro["verified_repair"] is None and ro["readings"]["routed"]["exported"] == ["LINT_EXIT"]


def test_the_action_says_the_reading_where_there_is_no_repair():
    from styxx.ciaudit import action as A
    fx = {"verified_repair": None, "stage": None, "lines_changed": None, "diff": None, "why_not": "strict-shell: not loud",
          "readings": {"routed": None, "declared": "allowed to fail"}}
    rec = {"new_hidden": 1, "workflows": [{"workflow": "ci.yml", "path": ".github/workflows/ci.yml", "new_hidden": [
        {"job": "ci", "index": 1, "step": "name:Tests", "name": "Tests", "kind": "born hidden", "verdict": "SWALLOWED", "mechanism": None, "fix": fx}]}]}
    md = A.summary_md(rec, {"reading": "test-merge", "base": "a" * 40, "head": "b" * 40}, {})
    assert "| none verified — the script says 'allowed to fail' |" in md
