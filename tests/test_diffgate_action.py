# -*- coding: utf-8 -*-
"""The Action must not report a pass when the gate did not run.

The shipped Action returned exit 0 whenever the diff could not be fetched, under
a comment reading *"a broken fetch must not fake a verdict"* — and 0 is the
passing verdict. It also ignored `DiffGate.measured` entirely, so an error
payload served with HTTP 200 produced a green check.

These are the product-level instance of the defect class the product exists to
detect, which is why they are pinned here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ACTION = Path(__file__).resolve().parent.parent / "diffgate_action.py"

SUMMARY = "This change only touches styxx/ and adds 2 tests."
REAL_DIFF = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n"
             "+++ b/styxx/x.py\n@@\n+def test_a(): pass\n+def test_b(): pass\n")


def load_action():
    spec = importlib.util.spec_from_file_location("diffgate_action", ACTION)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diffgate_action"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    payload = {"pull_request": {"number": 7, "body": SUMMARY,
                                "url": "https://api.github.test/pr/7"}}
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    monkeypatch.delenv("STYXX_STRICT", raising=False)
    monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    return tmp_path


def run(monkeypatch, diff_or_exc, *, strict=False, soft=False):
    mod = load_action()

    def fake_api(url, accept):
        if isinstance(diff_or_exc, Exception):
            raise diff_or_exc
        return diff_or_exc

    monkeypatch.setattr(mod, "api", fake_api)
    monkeypatch.setenv("STYXX_STRICT", "true" if strict else "false")
    monkeypatch.setenv("STYXX_SOFT_FAIL", "true" if soft else "false")
    return mod.main()


def test_a_readable_diff_passes(env, monkeypatch):
    assert run(monkeypatch, REAL_DIFF) == 0


@pytest.mark.parametrize("body", [
    "",
    "Sorry, I could not produce a diff.",
    '{"message": "Not Found"}',
    "<html><body>404</body></html>",
])
def test_unreadable_diff_never_passes_under_strict(env, monkeypatch, body):
    """A 200 response that is not a diff must not produce a green check."""
    assert run(monkeypatch, body, strict=True) == 1


@pytest.mark.parametrize("body", ["", "Sorry, I could not produce a diff."])
def test_unreadable_diff_is_reported_even_when_not_strict(env, monkeypatch, body, tmp_path):
    code = run(monkeypatch, body, strict=False)
    assert code == 0                       # the documented non-strict contract
    written = (env / "sum.md").read_text(encoding="utf-8")
    assert "UNMEASURED" in written
    assert "did not run" in written.lower()


def test_fetch_failure_fails_under_strict(env, monkeypatch):
    """This returned 0 under a comment saying a broken fetch must not fake a
    verdict. 0 IS the verdict."""
    assert run(monkeypatch, RuntimeError("connection reset"), strict=True) == 1


def test_fetch_failure_is_reported_when_not_strict(env, monkeypatch):
    assert run(monkeypatch, RuntimeError("connection reset")) == 0
    written = (env / "sum.md").read_text(encoding="utf-8")
    assert "DID NOT RUN" in written


def test_soft_fail_still_never_breaks_the_job(env, monkeypatch):
    assert run(monkeypatch, "not a diff", strict=True, soft=True) == 0


def test_the_action_pins_a_version_that_has_the_bypass_fix():
    """Releases before 7.44.0 contain the `only_touches` bypass in this very
    gate. The Action must never offer one as its default.

    Asserts the PROPERTY, not the literal string — the first version of this
    test hard-coded `styxx>=7.44.0` and failed the moment the floor was raised
    to 7.44.2, which is a test that breaks on the fix rather than on the defect.
    """
    import re

    y = (ACTION.parent / "action.yml").read_text(encoding="utf-8")
    m = re.search(r'default:\s*"styxx>=(\d+)\.(\d+)\.(\d+)"', y)
    assert m, "action.yml must pin a minimum styxx version"
    assert tuple(int(g) for g in m.groups()) >= (7, 44, 0)


# ---- the default reports; only an explicit soft-fail: "false" blocks ----
#
# action.yml's soft-fail input defaulted to "false", so a repository that added the Action got a
# check that failed on every accusation. No kind of accusation the instrument still makes has been
# measured clearing the 0.95 precision floor the lab set (the figures and their receipts are in
# action.yml's description and the CHANGELOG entry "the GitHub Action reports by default"), so the
# default now reports and a repository opts in to blocking. The value is compared
# case-insensitively with surrounding whitespace stripped; "false" blocks, "true" reports, and any
# other value reports with a warning naming it. These pin the default, the script's own fallback,
# how seven values are read, and that the lab's own check still blocks.

CONTRADICTED_DIFF = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n"
                     "+++ b/styxx/x.py\n@@\n+def test_a(): pass\n")


def _input_block(yml: str, name: str) -> str:
    """The lines of one input in action.yml, from `  name:` to the next two-space key."""
    import re

    m = re.search(rf"(?m)^  {re.escape(name)}:\n((?:^(?:    .*)?\n)*)", yml)
    assert m, f"action.yml has no `{name}` input"
    return m.group(1)


def test_the_action_reports_by_default():
    import re

    y = (ACTION.parent / "action.yml").read_text(encoding="utf-8")
    m = re.search(r'(?m)^    default:\s*"([^"]*)"', _input_block(y, "soft-fail"))
    assert m, "the soft-fail input must state its default"
    assert m.group(1) == "true"


def test_the_script_reports_when_soft_fail_is_unset(env, monkeypatch, capsys):
    """Run outside action.yml (STYXX_SOFT_FAIL unset), the script's default matches the input's."""
    mod = load_action()
    monkeypatch.setattr(mod, "api", lambda url, accept: CONTRADICTED_DIFF)
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "::error title=styxx diffgate - contradicted claim::" in out
    assert 'soft-fail: "false"' in out
    assert "soft-fail value not recognised" not in out      # unset is the default, not a value
    written = (env / "sum.md").read_text(encoding="utf-8")
    assert "soft-fail is on, as it is by default" in written


def test_soft_fail_false_still_fails_on_a_contradiction(env, monkeypatch):
    assert run(monkeypatch, CONTRADICTED_DIFF, soft=False) == 1
    written = (env / "sum.md").read_text(encoding="utf-8")
    assert 'sets `soft-fail: "false"`' in written


def test_soft_fail_true_reports_a_contradiction_and_passes(env, monkeypatch):
    assert run(monkeypatch, CONTRADICTED_DIFF, soft=True) == 0


NOT_RECOGNISED = "::warning title=styxx diffgate - soft-fail value not recognised::"


@pytest.mark.parametrize("value, blocks, warned", [
    ("", False, True),
    (" true", False, False),
    ("TRUE", False, False),
    ("yes", False, True),
    ("0", False, True),
    ("False ", True, False),
    ("false", True, False),
])
def test_how_the_soft_fail_value_is_read(env, monkeypatch, capsys, value, blocks, warned):
    """Only an explicit "false" blocks, compared case-insensitively with surrounding whitespace
    stripped. "true" reports. Every other value, the empty string included, reports and is named
    in a warning. Before this, every value but "true" in some casing blocked."""
    mod = load_action()
    monkeypatch.setattr(mod, "api", lambda url, accept: CONTRADICTED_DIFF)
    monkeypatch.setenv("STYXX_SOFT_FAIL", value)
    assert mod.main() == (1 if blocks else 0)
    out = capsys.readouterr().out
    assert "::error title=styxx diffgate - contradicted claim::" in out
    assert (NOT_RECOGNISED in out) is warned
    written = (env / "sum.md").read_text(encoding="utf-8")
    if warned:
        assert f"soft-fail is {value!r}, which is neither" in out
        assert "does not recognise" in written
    if blocks:
        assert 'sets `soft-fail: "false"`' in written
    else:
        assert "this check reports and does not fail the job" in written


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _step_with_uses(text: str, ref: str) -> list:
    """The lines of the one workflow step whose own `uses:` line names `ref`.

    A step runs from its `- ` line to the next line, not blank and not a comment, indented no
    deeper than that dash. A comment line is never a step's `uses:` line, so a comment that
    mentions `uses: ./` (the header of diffgate.yml has one) cannot stand in for the step.
    """
    import re

    lines = text.splitlines()
    own = re.compile(r"""^\s*(?:-\s+)?uses:\s*(["']?)""" + re.escape(ref) + r"""\1\s*(?:#.*)?$""")
    hits = [i for i, line in enumerate(lines) if own.match(line)]
    assert len(hits) == 1, f"expected one step with its own `uses: {ref}` line, found {len(hits)}"
    start = hits[0]
    while not re.match(r"^\s*-\s", lines[start]):
        start -= 1
        assert start >= 0, f"`uses: {ref}` is not inside a step"
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.lstrip().startswith("#") and _indent(line) <= _indent(lines[start]):
            break
        end += 1
    return lines[start:end]


def _with_inputs(step: list) -> dict:
    """The step's `with:` inputs, name to the value as written (quotes kept, comment dropped)."""
    import re

    at = next((k for k, line in enumerate(step) if re.match(r"^\s*(?:-\s+)?with:\s*(?:#.*)?$", line)), None)
    if at is None:
        return {}
    inputs = {}
    for line in step[at + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= _indent(step[at]):
            break
        m = re.match(r"^\s*([\w-]+):\s*(.*?)\s*(?:#.*)?$", line)
        if m:
            inputs[m.group(1)] = m.group(2)
    return inputs


def _unquote(value: str) -> str:
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'" else value


def test_the_labs_own_check_still_blocks():
    """This repository's diffgate job opts in to blocking explicitly, so the new default does not
    change it. The step is found by its own `uses: ./` line, and its soft-fail is read the way the
    Action reads it. Read, never written: no branch here may touch .github/."""
    wf = ACTION.parent / ".github" / "workflows" / "diffgate.yml"
    inputs = _with_inputs(_step_with_uses(wf.read_text(encoding="utf-8"), "./"))
    assert inputs.get("soft-fail") == '"false"', (
        "the lab's own diffgate step must set soft-fail: \"false\" or it stops blocking")
    assert load_action()._soft_fail(_unquote(inputs["soft-fail"])) == (False, None)


def test_the_step_finder_reads_the_step_not_a_comment():
    """An earlier version of the test above searched from the earliest `uses: ./` in the file,
    which is in diffgate.yml's header comment, so a soft-fail: "false" anywhere below it passed.
    Here the only soft-fail: "false" is on another step, and the `uses: ./` step must be read as
    setting none."""
    text = ("# the action at the root of this repository (`uses: ./`, so the wrapper runs)\n"
            "jobs:\n"
            "  diffgate:\n"
            "    steps:\n"
            "      - name: Gate\n"
            "        uses: ./\n"
            "        with:\n"
            "          strict: \"false\"\n"
            "      - uses: someone/else@v1\n"
            "        with:\n"
            "          soft-fail: \"false\"\n")
    step = _step_with_uses(text, "./")
    assert step[0].strip() == "- name: Gate" and len(step) == 4
    assert _with_inputs(step) == {"strict": '"false"'}
