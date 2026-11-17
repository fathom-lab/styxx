# -*- coding: utf-8 -*-
"""What `action.yml`, the README and the script's own sentences say of the Action, held to the script.

Four things. The events the texts name are the two `main()` gates, and on any other event the script
prints one line, reads no claim and returns 0 in every mode. The `styxx-version` input's text says
what chooses the gate: the ref in `uses:`, not the installed package. On a push the text the gate
reads is the head commit's message, and no sentence calls it "the description". On a `pull_request`
event every byte the script prints is what the script at `97772a02` printed (the commit this change
was cut from); that comparison skips where the commit is not in the checkout, as in a shallow clone.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "diffgate_action.py"
BASE = "97772a02"           # the commit this change was cut from


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The table of paths through main() and the runner that drives one of them, from the file that holds them.
FC = _load_file("diffgate_action_fails_closed_paths", Path(__file__).with_name("test_diffgate_action_fails_closed.py"))


@pytest.fixture(scope="module")
def act():
    return FC.load("diffgate_action_texts")


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _input_block(yml, name):
    m = re.search(rf"(?m)^  {re.escape(name)}:\r?\n((?:^(?:    .*)?\r?\n)*)", yml)
    assert m, f"action.yml has no `{name}` input"
    return m.group(1)


# -- the events ---------------------------------------------------------------------------------------------------

def test_the_events_the_texts_name_are_the_ones_main_gates():
    gated = set(re.findall(r'event_name == "([a-z_]+)"', ACTION.read_text(encoding="utf-8")))
    assert gated == {"pull_request", "push"}
    yml = _flat((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert "It gates two events: pull_request (the description against the pull request's diff) and push" in yml
    assert "On any other event, pull_request_target and merge_group among them, it prints one line" in yml
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "It gates `pull_request` and `push` events" in readme
    assert "on any other event, `pull_request_target` and `merge_group` among them, it prints one line" in readme
    assert "types: [opened, edited, synchronize, reopened]" in readme


@pytest.mark.parametrize("strict", [False, True], ids=["strict-off", "strict-on"])
@pytest.mark.parametrize("soft", ["true", "false", "maybe", None])
@pytest.mark.parametrize("event", ["pull_request_target", "merge_group", "workflow_dispatch", "issue_comment", ""])
def test_an_event_that_is_not_gated_prints_one_line_and_passes(act, tmp_path, monkeypatch, capsys, event, soft,
                                                              strict):
    """Whatever soft-fail and strict say: one plain line, no annotation, no summary, nothing asked of
    the API, exit 0. The description here holds a claim the diff contradicts."""
    case = dict(event=event, body=FC.FALSE)
    rc, out, summary, asked = FC.run_main(act, tmp_path, monkeypatch, capsys, case, soft, strict)
    lines = out.splitlines()
    if soft == "maybe":                     # a soft-fail value that is not recognised is named before anything else
        assert lines[0].startswith("::warning title=styxx diffgate - soft-fail value not recognised::")
        lines = lines[1:]
    assert rc == 0
    assert lines == [f"styxx diffgate: event {event!r} not gated (pull_request/push only)"]
    assert summary == "" and asked == []


# -- the styxx-version input --------------------------------------------------------------------------------------

def test_the_styxx_version_text_says_the_ref_chooses_the_gate():
    """The script imports the styxx package beside it (the process tests of
    tests/test_diffgate_action_fails_closed.py name the one imported), so the input's text must not
    say the installed version is the gate."""
    block = _flat(_input_block((ROOT / "action.yml").read_text(encoding="utf-8"), "styxx-version"))
    assert "It does NOT choose the gate." in block
    assert "at the ref the workflow names in `uses:`" in block
    assert "name another ref in `uses:`" in block
    assert "Do not pin below it" not in block
    assert re.search(r'default: "styxx>=\d+\.\d+\.\d+"', block)


def test_the_readme_links_the_example_at_a_ref_that_holds_its_repair():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "blob/main/examples/knowsay_endpoint.py" in readme
    assert not re.search(r"blob/v8\.0\.[012]/examples/knowsay_endpoint\.py", readme)


# -- what the text is called on a push ----------------------------------------------------------------------------

PUSH = {
    "refused, a claim": (dict(event="push", body=FC.FALSE, api=FC.Refused("HTTP Error 502: Bad Gateway")),
                         "no rewording of the head commit message makes it serve one"),
    "refused, no claim": (dict(event="push", body=FC.PROSE, api=FC.Refused("HTTP Error 502: Bad Gateway")),
                          "the diff could not be read and the head commit message holds no claim the gate reads"),
    "unmeasured, no claim": (dict(event="push", body=FC.PROSE, api=""),
                             "did not measure the diff and the head commit message holds no claim the gate reads"),
    "measured, a claim": (dict(event="push", body=FC.FALSE),
                          "did not measure while the head commit message holds a claim the gate reads"),
}


@pytest.mark.parametrize("name", list(PUSH))
def test_on_a_push_the_text_is_the_head_commit_message(act, tmp_path, monkeypatch, capsys, name):
    case, sentence = PUSH[name]
    rc, out, summary, _ = FC.run_main(act, tmp_path, monkeypatch, capsys, case, "false", False)
    assert sentence in out + summary, out + summary
    for soft in ("false", "true"):
        for strict in (False, True):
            _, out, summary, _ = FC.run_main(act, tmp_path, monkeypatch, capsys, case, soft, strict)
            assert "description" not in out + summary, (soft, strict, out + summary)


def test_on_a_pull_request_the_text_is_still_the_description(act, tmp_path, monkeypatch, capsys):
    _, out, summary, _ = FC.run_main(act, tmp_path, monkeypatch, capsys, dict(body=FC.FALSE), "false", False)
    assert "did not measure while the description holds a claim the gate reads" in summary
    assert "head commit message" not in out + summary


# -- a pull request's bytes are as they were ----------------------------------------------------------------------

def _base_script(tmp_path):
    try:
        text = subprocess.run(["git", "show", f"{BASE}:diffgate_action.py"], cwd=ROOT, capture_output=True,
                              timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip(f"the script at {BASE} is not in this checkout (a shallow clone), so there is nothing to compare")
    path = tmp_path / "base" / "diffgate_action.py"
    path.parent.mkdir()
    path.write_bytes(text)
    return path


@pytest.mark.parametrize("strict", [False, True], ids=["strict-off", "strict-on"])
@pytest.mark.parametrize("soft", ["true", "maybe", None, "false"])
def test_on_a_pull_request_every_printed_byte_is_as_it_was(act, tmp_path, monkeypatch, capsys, soft, strict):
    """Every path of the fails-closed table that is not a push, in report and in blocking mode: the
    same exit code, the same stdout and the same summary as the script at the commit this change was
    cut from. The change rewords sentences on a push only."""
    old = FC.load("diffgate_action_texts_base", _base_script(tmp_path))
    for name, case in FC.PATHS.items():
        if case.get("event") == "push":
            continue
        new_run = FC.run_main(act, tmp_path, monkeypatch, capsys, case, soft, strict)
        old_run = FC.run_main(old, tmp_path, monkeypatch, capsys, case, soft, strict)
        assert new_run[:3] == old_run[:3], name
