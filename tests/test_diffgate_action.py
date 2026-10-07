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
# check that failed on every accusation. On pull requests this lab did not write the accusations
# measured precision 0.23 against a preregistered 0.95 floor
# (papers/closed-model-frontier/RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md), so the
# default now reports and a repository opts in to blocking. These pin the default, the script's
# own fallback, and that the lab's own check still blocks.

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
    written = (env / "sum.md").read_text(encoding="utf-8")
    assert "soft-fail is on (the default)" in written


def test_soft_fail_false_still_fails_on_a_contradiction(env, monkeypatch):
    assert run(monkeypatch, CONTRADICTED_DIFF, soft=False) == 1
    written = (env / "sum.md").read_text(encoding="utf-8")
    assert 'sets `soft-fail: "false"`' in written


def test_soft_fail_true_reports_a_contradiction_and_passes(env, monkeypatch):
    assert run(monkeypatch, CONTRADICTED_DIFF, soft=True) == 0


def test_the_labs_own_check_still_blocks():
    """This repository's diffgate job opts in to blocking explicitly, so the new default does not
    change it. Read, never written: no branch here may touch .github/."""
    import re

    wf = ACTION.parent / ".github" / "workflows" / "diffgate.yml"
    text = wf.read_text(encoding="utf-8")
    step = text[text.index("uses: ./"):]
    assert re.search(r'(?m)^\s+soft-fail:\s*"false"\s*$', step), (
        "the lab's own diffgate job must set soft-fail: \"false\" or it stops blocking")
