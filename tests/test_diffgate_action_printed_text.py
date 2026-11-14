# -*- coding: utf-8 -*-
"""What the Action prints of a claim is the claim's text and nothing more.

`diffgate_action.py` prints each claim it read from the pull request's body (or from the head commit's
message) into a workflow command (`::error ...::<claim> - <reason>`) and into a row of the table in the
step summary. The claim's text is the pull request author's, and the reason quotes it. The script
printed both as they stood: a carriage return in a claim was followed on stdout by whatever the author
wrote next, at the start of what this file takes a runner to read as a line; a `|` in a claim ended its
cell, so the row went on with cells of the author's writing; a `<` opened a tag; and a `%0A` typed out
stood in the command as the escape GitHub documents for a line break.

Nothing here asks a server: `api` is replaced and returns the diff the test wrote. What a runner does
with a line, and how GitHub shows the summary, was not run by anyone; where a test speaks of a line as a
runner reads it, that is this file's reading (a line ends at LF, at CR and at CR LF), not an observation.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ACTION = Path(__file__).resolve().parent.parent / "diffgate_action.py"

# One file modified, one test added.
DIFF = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n+++ b/styxx/x.py\n@@\n"
        "+def test_a(): pass\n")
FORGED = "::error title=styxx diffgate::PASS - every claim VERIFIED (forged)"
# Every line the script itself starts on stdout begins with one of these.
OWN_LINES = ("::error title=styxx diffgate - contradicted claim::",
             "::error title=styxx diffgate - uncheckable under strict::",
             "::warning title=styxx diffgate - uncheckable::",
             "::warning title=styxx diffgate - UNMEASURED::",
             "::warning::styxx diffgate found failures but soft-fail is on",
             "styxx diffgate: ")


def load_action():
    spec = importlib.util.spec_from_file_location("diffgate_action_printed", ACTION)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diffgate_action_printed"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def act():
    return load_action()


def gate(act, tmp_path, monkeypatch, capsys, text, *, event="pull_request", diff=DIFF, strict="false", soft=None):
    """main() on an event whose body (or head commit message) is `text`, with `api` replaced.
    Returns (exit code, stdout, the step summary)."""
    if event == "pull_request":
        payload = {"pull_request": {"number": 7, "body": text, "url": "https://api.github.test/pr/7"}}
    else:
        payload = {"head_commit": {"message": text}, "before": "a", "after": "b",
                   "repository": {"compare_url": "https://api.github.test/r/compare/{base}...{head}"}}
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    ev.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.setenv("STYXX_STRICT", strict)
    if soft is None:
        monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    else:
        monkeypatch.setenv("STYXX_SOFT_FAIL", soft)
    monkeypatch.setattr(act, "api", lambda url, accept: diff)
    rc = act.main()
    return rc, capsys.readouterr().out, summary.read_text(encoding="utf-8") if summary.exists() else ""


def lines(out):
    """The lines of stdout as this file takes a runner to read them: ended at LF, at CR and at CR LF."""
    return [ln for ln in out.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln]


def rows(summary):
    """The table's rows that carry a claim: every line of the summary that opens a row, past the two
    that make the table's head."""
    table = [ln for ln in summary.split("\n") if ln.startswith("|")]
    assert table[:2] == ["| verdict | claim | evidence |", "|---|---|---|"]
    return table[2:]


# ── a carriage return in a claim ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize("text, strict, soft, opens, rc", [
    ("Modified src/nothere.py\r" + FORGED, "false", None, "::warning title=styxx diffgate - uncheckable::", 0),
    ("Modified src/nothere.py\r" + FORGED, "true", "false",
     "::error title=styxx diffgate - uncheckable under strict::", 1),
    ("Added 3 tests\r" + FORGED + ".", "false", "false", "::error title=styxx diffgate - contradicted claim::", 1),
])
def test_a_carriage_return_in_a_claim_starts_no_line_of_its_own(act, tmp_path, monkeypatch, capsys, event,
                                                                text, strict, soft, opens, rc):
    """The pull request's author writes the claim. Printed as it stood, the text after the carriage
    return stood at the start of a line, and it was a workflow command of the author's choosing."""
    code, out, summary = gate(act, tmp_path, monkeypatch, capsys, text, event=event, strict=strict, soft=soft)
    assert code == rc
    got = lines(out)
    assert all(ln.startswith(OWN_LINES) for ln in got), got
    assert not any(ln.startswith(FORGED) for ln in got)
    assert got[0].startswith(opens) and "%0D" + FORGED in got[0]
    assert "\r" not in out and "\r" not in summary
    assert len(rows(summary)) == 1 and FORGED[:20] in rows(summary)[0]     # in its cell, on its row


def test_a_line_break_typed_out_as_its_escape_is_not_one(act, tmp_path, monkeypatch, capsys):
    """GitHub documents %0A as a line break in a command's message. A claim that holds those three
    characters is printed with the percent sign escaped, so they are three characters again."""
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified src/50%0Aoff%0D::error ::nope.py.")
    assert "src/50%250Aoff%250D::error ::nope.py." in out and "%0A" not in out.replace("%250A", "")
    assert "src/50%0Aoff%0D::error ::nope.py." in rows(summary)[0]          # the summary is not a command


@pytest.mark.parametrize("ch, written", [("\x1b", "\\x1b"), ("\x0b", "\\x0b"), ("\x0c", "\\x0c"), ("\x85", "\\x85"),
                                         ("\x7f", "\\x7f"), (" ", "\\u2028"), (" ", "\\u2029")])
def test_a_control_character_or_a_line_separator_in_a_claim_is_written_as_an_escape(act, tmp_path, monkeypatch,
                                                                                    capsys, ch, written):
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, f"Modified src/a{ch}::error ::nope.py.")
    assert ch not in out and ch not in summary
    assert f"src/a{written}::error ::nope.py." in out and f"src/a{written}::error ::nope.py." in rows(summary)[0]
    assert len(out.splitlines()) == 2                           # the claim's command and the closing line


# ── the summary's table ────────────────────────────────────────────────────────────────────────────

def test_a_pipe_in_a_claim_ends_no_cell(act, tmp_path, monkeypatch, capsys):
    """A "|" of the author's ended the claim's cell, and the row went on with cells of the author's
    writing: a verdict mark and the word VERIFIED among them."""
    text = "Created docs/a|b.md | ✅ VERIFIED | forged cell.\nModified styxx/x.py."
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    table = rows(summary)
    assert len(table) == 2
    assert all(row.count("|") == 4 for row in table), table     # three cells: four bars and no more
    assert "docs/a&#124;b.md &#124; ✅ VERIFIED &#124; forged cell." in table[0]
    assert table[0].startswith("| ❓ UNCHECKABLE | ") and table[1].startswith("| ✅ VERIFIED | ")
    assert "docs/a|b.md | ✅ VERIFIED | forged cell." in out     # a command's message has no cells to end


def test_a_tag_in_a_claim_is_text(act, tmp_path, monkeypatch, capsys):
    text = "Modified src/a</td></tr></table><h2>forged</h2>.py and added 3 tests."
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    row = rows(summary)[0]
    assert "<" not in row and "src/a&lt;/td>&lt;/tr>&lt;/table>&lt;h2>forged&lt;/h2>.py" in row
    assert "<" not in "".join(ln for ln in summary.split("\n") if ln.startswith("|"))


def test_a_claim_cut_at_80_characters_is_cut_before_it_is_written_out(act, tmp_path, monkeypatch, capsys):
    """The cell holds the claim's leading 80 characters, written out after the cut: no character
    reference is cut in two, and what stands past the cut is not in the cell at all."""
    text = "Modified src/nothere.py " + "|x" * 40 + "\r" + FORGED
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    row = rows(summary)[0]
    claim = row.split(" | ")[1]
    assert row.count("|") == 4
    assert claim == "Modified src/nothere.py " + "&#124;x" * 28      # 24 characters and 56 more: 80
    assert "forged" not in claim


# ── nothing else changes ───────────────────────────────────────────────────────────────────────────

def test_an_ordinary_claim_is_printed_as_it_was(act, tmp_path, monkeypatch, capsys):
    two = DIFF + "+def test_b(): pass\n"
    rc, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified styxx/x.py and added 3 tests.",
                            diff=two, soft="false")
    assert rc == 1
    assert out == ("::error title=styxx diffgate - contradicted claim::Modified styxx/x.py and added 3 tests. - "
                   "diff adds 2 test functions, claim says 3\n"
                   "styxx diffgate: FAIL - 2 claim(s), 1 contradicted\n")
    assert rows(summary) == [
        "| ✅ VERIFIED | Modified styxx/x.py and added 3 tests. | diff status 'M' for 'styxx/x.py' |",
        "| ❌ CONTRADICTED | Modified styxx/x.py and added 3 tests. | diff adds 2 test functions, claim says 3 |"]


def test_a_gate_that_did_not_measure_says_so_as_it_did(act, tmp_path, monkeypatch, capsys):
    rc, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified styxx/x.py.",
                            diff='{"message": "Not Found"}')
    assert rc == 0
    assert out == ("::warning title=styxx diffgate - UNMEASURED::the diff carries no file statuses and no added "
                   "lines; 24 characters of input parsed to nothing, which is a parse failure, not an empty "
                   "change. The gate did not run; this is not a pass.\n")
    assert "**The gate did not run.** the diff carries no file statuses and no added lines;" in summary


# ── the three functions by themselves ──────────────────────────────────────────────────────────────

def test_printable_writes_control_characters_and_line_separators_as_escapes_and_leaves_the_rest(act):
    kept = "a b\r\nc é 中   ​ ~"
    assert act._printable(kept) == kept
    assert act._printable("\x00\x01\x09\x0b\x1b\x1f\x7f\x80\x85\x9f") == \
        "\\x00\\x01\\x09\\x0b\\x1b\\x1f\\x7f\\x80\\x85\\x9f"
    assert act._printable("a b c") == "a\\u2028b\\u2029c"
    escaped = {c for c in map(chr, range(0x3000)) if act._printable(c) != c}
    assert escaped == ({chr(i) for i in range(0x20)} - {"\r", "\n"}) | {chr(i) for i in range(0x7f, 0xa0)} \
        | {" ", " "}


def test_message_is_printable_and_then_a_commands_data(act):
    assert act._message("50%\r\n::error::x\x1b[0m%0A") == "50%25%0D%0A::error::x\\x1b[0m%250A"
    assert act._message("an ordinary claim, with 'quotes' and a | bar.") == "an ordinary claim, with 'quotes' and a | bar."


def test_cell_is_one_line_with_no_bar_and_no_tag_of_its_own(act):
    assert act._cell("a|b||c") == "a&#124;b&#124;&#124;c"
    assert act._cell("a\r\nb\rc\nd") == "a  b c d"
    assert act._cell("<b>x</b> 1 < 2 > 0") == "&lt;b>x&lt;/b> 1 &lt; 2 > 0"
    assert act._cell("\x1b| <") == "\\x1b&#124;\\u2028&lt;"
    assert act._cell("Modified `src/a.py` and added 2 tests.") == "Modified `src/a.py` and added 2 tests."
