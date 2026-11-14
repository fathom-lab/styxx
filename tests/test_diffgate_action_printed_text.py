# -*- coding: utf-8 -*-
"""What the Action prints of a claim starts no line, no cell and no tag of the author's.

`diffgate_action.py` prints each claim it read from the pull request's body (or from the head commit's
message) into a row of the table in the step summary, and each claim it did not verify into a workflow
command as well (`::error ...::<claim> - <reason>`). The claim's text is the author's, and the reason can
quote it and paths of the diff. The script printed both as they stood. A carriage return in a claim was
followed on stdout by whatever the author wrote next, where this file takes a runner to start a line. A
`|` in a claim stood in the row as a bar, so the row had bars of the author's writing. A `<` stood in the
row as it came, a closing `</table>` and an `<h2>` included. And a `%0A` typed out stood in the command
as the escape GitHub documents for a line break.

Each variable cell of the table is now a code span, so that what stands in it is shown as the
characters it is; tests/test_diffgate_action_summary_renders.py renders the table with cmark-gfm where
that is installed, and this file holds the text of the rows.

Nothing here asks a server: `api` is replaced and returns the diff the test wrote. What a runner does
with a line, and how GitHub shows the summary, was not run by anyone; where a test speaks of a line as a
runner reads it, that is this file's reading (a line ends at LF, at CR and at CR LF), not an observation.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ACTION = Path(__file__).resolve().parent.parent / "diffgate_action.py"

# One file modified, one test added.
DIFF = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n+++ b/styxx/x.py\n@@\n"
        "+def test_a(): pass\n")
FORGED = "::error title=styxx diffgate::PASS - every claim VERIFIED (forged)"
# Every line the script itself starts on stdout in the runs below begins with one of these (its lines for
# a soft-fail value it does not recognise and for a failed fetch are not reached here).
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
    # as the file holds it: read_text would hand a lone carriage return back as a line feed
    return rc, capsys.readouterr().out, summary.read_bytes().decode("utf-8") if summary.exists() else ""


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
    """The author of the pull request, or of the head commit, writes the claim. Printed as it stood,
    the text after the carriage return stood where this file takes a runner to start a line, and it had
    the form of a workflow command of the author's choosing."""
    code, out, summary = gate(act, tmp_path, monkeypatch, capsys, text, event=event, strict=strict, soft=soft)
    assert code == rc
    got = lines(out)
    assert all(ln.startswith(OWN_LINES) for ln in got), got
    assert not any(ln.startswith(FORGED) for ln in got)
    assert got[0].startswith(opens) and "%0D" + FORGED in got[0]
    assert "\r" not in out and "\r" not in summary
    assert len(rows(summary)) == 1 and FORGED[:20] in rows(summary)[0]     # in its cell, on its row
    assert rows(summary)[0].count(" | ") == 2 and rows(summary)[0].count("`") == 4


def test_a_line_break_typed_out_as_its_escape_is_not_one(act, tmp_path, monkeypatch, capsys):
    """GitHub documents %0A as a line break in a command's message. A claim that holds those three
    characters is printed with the percent sign escaped, so they are three characters again."""
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified src/50%0Aoff%0D::error ::nope.py.")
    assert "src/50%250Aoff%250D::error ::nope.py." in out and "%0A" not in out.replace("%250A", "")
    assert "src/50%0Aoff%0D::error ::nope.py." in rows(summary)[0]          # the summary is not a command


@pytest.mark.parametrize("ch, written", [("\x1b", "\\x1b"), ("\x0b", "\\x0b"), ("\x0c", "\\x0c"), ("\x85", "\\x85"),
                                         ("\x7f", "\\x7f"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")])
def test_a_control_character_or_a_line_separator_in_a_claim_is_written_as_an_escape(act, tmp_path, monkeypatch,
                                                                                    capsys, ch, written):
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, f"Modified src/a{ch}::error ::nope.py.")
    assert ch not in out and ch not in summary
    assert f"src/a{written}::error ::nope.py." in out and f"src/a{written}::error ::nope.py." in rows(summary)[0]
    assert len(out.splitlines()) == 2                           # the claim's command and the closing line


# ── the summary's table ────────────────────────────────────────────────────────────────────────────

def bars(row):
    """The bars of a row that are not behind a backslash: the ones a table reads as the ends of cells."""
    return len(re.findall(r"(?<!\\)\|", row))


def test_a_pipe_in_a_claim_ends_no_cell(act, tmp_path, monkeypatch, capsys):
    """A "|" of the author's stood in the row as a bar: seven bars under a head of three columns, with
    a verdict mark and the word VERIFIED of the author's writing between them."""
    text = "Created docs/a|b.md | ✅ VERIFIED | forged cell.\nModified styxx/x.py."
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    table = rows(summary)
    assert len(table) == 2
    assert all(bars(row) == 4 for row in table), table           # three cells: four bars and no more
    assert "| `Created docs/a\\|b.md \\| ✅ VERIFIED \\| forged cell.` |" in table[0]
    assert table[0].startswith("| ❓ UNCHECKABLE | `") and table[1].startswith("| ✅ VERIFIED | `")
    assert "docs/a|b.md | ✅ VERIFIED | forged cell." in out     # a command's message has no cells to end


def test_a_tag_or_markup_in_a_claim_stands_inside_the_cells_code_span(act, tmp_path, monkeypatch, capsys):
    text = "Modified src/a</td></tr></table><h2>forged</h2>.py and added 3 tests ![v](https://e.example/v.png)."
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    row = rows(summary)[0]
    claim = row.split(" | ")[1]
    assert claim == "`" + text[:80] + "`" and bars(row) == 4


def test_a_claim_cut_at_80_characters_is_cut_before_it_is_written_out(act, tmp_path, monkeypatch, capsys):
    """The cell holds the claim's leading 80 characters, written out after the cut: no escape is cut in
    two, and what stands past the cut is not in the cell at all."""
    text = "Modified src/nothere.py " + "|x" * 40 + "\r" + FORGED
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    row = rows(summary)[0]
    assert bars(row) == 4
    assert row.split(" | ")[1] == "`Modified src/nothere.py " + "\\|x" * 28 + "`"      # 24 characters and 56 more: 80
    assert "forged" not in row.split(" | ")[1]


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
        "| ✅ VERIFIED | `Modified styxx/x.py and added 3 tests.` | `diff status 'M' for 'styxx/x.py'` |",
        "| ❌ CONTRADICTED | `Modified styxx/x.py and added 3 tests.` | `diff adds 2 test functions, claim says 3` |"]


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
    kept = "a b\tc\r\nd \u00e9 \u4e2d \u00a0 \u200b ~ \U0001f600"
    assert act._printable(kept) == kept
    assert act._printable("\x00\x01\x08\x0b\x1b\x1f\x7f\x80\x85\x9f") == \
        "\\x00\\x01\\x08\\x0b\\x1b\\x1f\\x7f\\x80\\x85\\x9f"
    assert act._printable("a\u2028b\u2029c") == "a\\u2028b\\u2029c"
    assert act._printable("a\ud800b\udc80c\udfffd") == "a\\ud800b\\udc80c\\udfffd"
    escaped = {c for c in map(chr, range(0x110000)) if act._printable(c) != c}
    assert escaped == ({chr(i) for i in range(0x20)} - {"\t", "\r", "\n"}) | {chr(i) for i in range(0x7f, 0xa0)} \
        | {"\u2028", "\u2029"} | {chr(i) for i in range(0xd800, 0xe000)}


def test_message_is_printable_and_then_a_commands_data(act):
    assert act._message("50%\r\n::error::x\x1b[0m%0A") == "50%25%0D%0A::error::x\\x1b[0m%250A"
    assert act._message("an ordinary claim, with 'quotes' and a | bar.") == "an ordinary claim, with 'quotes' and a | bar."


def test_line_is_one_line_with_no_bar_and_no_tag_of_its_own(act):
    assert act._line("a|b||c") == "a&#124;b&#124;&#124;c"
    assert act._line("a\r\nb\rc\nd") == "a  b c d"
    assert act._line("<b>x</b> 1 < 2 > 0") == "&lt;b>x&lt;/b> 1 &lt; 2 > 0"
    assert act._line("\x1b|\u2028<") == "\\x1b&#124;\\u2028&lt;"


@pytest.mark.parametrize("text, cell", [
    ("Modified src/a.py and added 2 tests.", "`Modified src/a.py and added 2 tests.`"),
    ("a|b||c", "`a\\|b\\|\\|c`"),                              # the table's own escape for a bar
    ("a\\|b", "`a\\\\|b`"),                                    # a backslash of the author's before a bar
    ("a\r\nb\rc\nd", "`a  b c d`"),
    ("<b>x</b> ![v](u) [l](u) **b** &lt; &#124;", "`<b>x</b> ![v](u) [l](u) **b** &lt; &#124;`"),
    ("Modified `src/a.py`", "`` Modified `src/a.py` ``"),       # one backtick more than the longest run, and padded
    ("a `` b ` c", "```a `` b ` c```"),
    ("`", "`` ` ``"),
    (" a ", "`  a  `"),                                         # a reader takes one pair of spaces off
    (" a", "` a`"),
    ("a\\", "`a\\`"),
    ("\x1b\u2028\ud800\tz", "`\\x1b\\u2028\\ud800\tz`"),
    ("", ""), ("   ", ""),
])
def test_cell_is_a_code_span_that_holds_the_text(act, text, cell):
    assert act._cell(text) == cell


# ── text that comes from the diff: a reason quotes its paths ───────────────────────────────────────

def diff_of(path):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n y = 0\n+x = 1\n"


def test_a_path_of_the_diff_that_a_reason_quotes_ends_no_cell_and_opens_no_tag(act, tmp_path, monkeypatch, capsys):
    """The body is ordinary. The diff's own path holds the bar, the "<" and the typed-out escape, and the
    reason quotes the path: the author of a pull request writes its diff as well as its body."""
    path = "dev/a|b<i>%0A::error title=x::forged\x85/x.py"       # the NEL is what makes the gate withhold
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified x.py.", diff=diff_of(path))
    table = rows(summary)
    assert len(table) == 1 and bars(table[0]) == 4 and table[0].count("`") == 4
    assert "dev/a\\|b<i>%0" in table[0]
    got = lines(out)
    assert all(ln.startswith(OWN_LINES) for ln in got), got
    assert "%0A" not in out.upper().replace("%250A", "")         # the gate's own reading may lower the path's case
    assert "dev/a|b<i>%250" in out and "\x85" not in out and "\x85" not in summary


@pytest.mark.parametrize("strict, soft, opens", [
    ("false", None, "::warning title=styxx diffgate - uncheckable::"),
    ("true", "false", "::error title=styxx diffgate - uncheckable under strict::"),
])
def test_a_reason_is_written_for_a_command_as_a_claim_is(act, tmp_path, monkeypatch, capsys, strict, soft, opens):
    path = "dev/50%0D::error title=x::forged\x85/x.py"
    _, out, _ = gate(act, tmp_path, monkeypatch, capsys, "Modified x.py.", diff=diff_of(path), strict=strict, soft=soft)
    line = [ln for ln in lines(out) if ln.startswith(opens)]
    assert len(line) == 1 and "dev/50%250" in line[0]
    assert "%0D" not in out.upper().replace("%250D", "")


def test_a_contradicted_claims_reason_is_written_for_its_command(act, tmp_path, monkeypatch, capsys):
    """No reason the gate writes for a contradiction quotes a path today, so the gate is replaced here
    by one claim whose text and reason each hold a carriage return, a percent sign and ESC: this holds
    the print for a contradicted claim, and the row, to what they were handed."""
    class Claim:
        verdict, kind = "CONTRADICTED", "tests_added"
        text = "claim 50%\r::error ::a\x1b|<"
        why = "reason 50%\r::error ::b\x1b|<"

    class Gate:
        measured, why_unmeasured, verdict, claims = True, "", "FAIL", [Claim()]
    monkeypatch.setattr(act, "gate_diff_text", lambda *a, **k: Gate())
    rc, out, summary = gate(act, tmp_path, monkeypatch, capsys, "anything", soft="false")
    assert rc == 1
    assert out == ("::error title=styxx diffgate - contradicted claim::claim 50%25%0D::error ::a\\x1b|< - "
                   "reason 50%25%0D::error ::b\\x1b|<\nstyxx diffgate: FAIL - 1 claim(s), 1 contradicted\n")
    assert rows(summary) == ["| \u274c CONTRADICTED | `claim 50% ::error ::a\\x1b\\|<` | "
                             "`reason 50% ::error ::b\\x1b\\|<` |"]


def test_a_claim_is_cut_at_120_characters_before_it_is_written_for_a_command(act, tmp_path, monkeypatch, capsys):
    """24 characters and then percent signs: the command holds the claim's leading 120 characters, each
    percent sign written whole, and not the leading 120 characters of what was written."""
    _, out, _ = gate(act, tmp_path, monkeypatch, capsys, "Modified src/nothere.py " + "%" * 200)
    line = lines(out)[0]
    claim = line[len("::warning title=styxx diffgate - uncheckable::"):].split(" - '")[0]
    assert claim == "Modified src/nothere.py " + "%25" * 96


# ── a surrogate that stands alone, a tab, the pull request's number ────────────────────────────────

@pytest.mark.parametrize("ch, written", [("\ud800", "\\ud800"), ("\udc80", "\\udc80"), ("\udce2\udc80\udca8", "\\udce2\\udc80\\udca8")])
def test_a_lone_surrogate_in_a_claim_is_written_as_an_escape_and_ends_nothing(act, tmp_path, monkeypatch,
                                                                                capsys, ch, written):
    """A lone surrogate cannot be written as UTF-8: it ended main() in a traceback when the summary was
    written, and where stdout writes one as a raw byte, three of them are the bytes of U+2028."""
    rc, out, summary = gate(act, tmp_path, monkeypatch, capsys, f"Modified src/a{ch}::error ::nope.py.")
    assert rc == 0
    assert f"src/a{written}::error ::nope.py." in out and f"src/a{written}::error ::nope.py." in rows(summary)[0]
    out.encode("utf-8")                                         # nothing in it that UTF-8 cannot carry
    summary.encode("utf-8")


def test_a_tab_in_a_claim_is_left_as_it_is(act, tmp_path, monkeypatch, capsys):
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Updated src/nope.py\tand the docs.")
    assert "::Updated src/nope.py\tand the docs. - " in out
    assert "| `Updated src/nope.py\tand the docs.` |" in rows(summary)[0]


@pytest.mark.parametrize("number, shown", [(7, "7"), ("7\r::error ::x", "?"), (None, "?"), (True, "?"),
                                           (["7"], "?"), (7.5, "?")])
def test_the_pull_requests_number_is_printed_only_if_it_is_an_integer(act, tmp_path, monkeypatch, capsys,
                                                                      number, shown):
    """GitHub documents the number as an integer and a runner writes the event file, so this is not
    text the pull request's author reaches; it was the one value of that file printed as it came."""
    payload = {"pull_request": {"number": number, "body": "Modified styxx/x.py.", "url": "https://api.github.test/pr"}}
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    ev.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.delenv("STYXX_STRICT", raising=False)
    monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    monkeypatch.setattr(act, "api", lambda url, accept: DIFF)
    assert act.main() == 0
    text = summary.read_bytes().decode("utf-8")
    assert f"claim(s) checked on the PR #{shown} body" in text and "\r" not in text
    payload["pull_request"]["body"] = ""
    ev.write_text(json.dumps(payload), encoding="utf-8")
    capsys.readouterr()
    assert act.main() == 0
    assert capsys.readouterr().out == f"styxx diffgate: empty PR #{shown} body — nothing to gate\n"


# ── the table's size ───────────────────────────────────────────────────────────────────────────────

def test_the_table_stops_before_the_summary_passes_the_size_github_allows(act, tmp_path, monkeypatch, capsys):
    """One sentence with no full stop gives a claim for each of its names, and every claim a row. The
    rows stop at the script's budget and a line says how many claims are not shown, and of what verdict."""
    rc, out, summary = gate(act, tmp_path, monkeypatch, capsys, "fix a|c.c " * 3000)
    table = rows(summary)
    total = int(summary.split("claim(s) checked")[0].rsplit("· ", 1)[1])
    assert total > len(table) > 0
    assert sum(len(r.encode("utf-8")) + 1 for r in table) <= act._ROWS_BYTES
    assert len(summary.encode("utf-8")) < 1024 * 1024
    assert all(bars(r) == 4 for r in table)
    rest = total - len(table)
    assert (f"_{rest} more claim(s) are not in this table, which stops at 524,288 bytes of rows to stay under "
            f"the size GitHub allows a step's summary: 0 CONTRADICTED, {rest} UNCHECKABLE, 0 VERIFIED. "
            "The verdict above counts every claim._") in summary
    assert out.count("::warning title=styxx diffgate - uncheckable::") == total      # every claim has its command


def test_a_table_within_the_budget_has_every_claim_and_no_such_line(act, tmp_path, monkeypatch, capsys):
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "fix a.c " * 40)
    assert len(rows(summary)) == 40 and "are not in this table" not in summary
