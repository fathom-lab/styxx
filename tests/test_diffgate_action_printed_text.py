# -*- coding: utf-8 -*-
"""What the Action prints of a claim starts no line, no cell and no tag of the author's.

`diffgate_action.py` prints each claim it read from the pull request's body (or from the head commit's
message) into a row of the table in the step summary, and each claim it did not verify into a workflow
command as well (`::error ...::<claim> - <reason>`; a warning for at most a hundred of the claims it
could not check, from the last section of this file on). The claim's text is the author's, and the reason can
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
import types
from pathlib import Path

import pytest

ACTION = Path(__file__).resolve().parent.parent / "diffgate_action.py"

# One file modified, one test added.
DIFF = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n+++ b/styxx/x.py\n@@\n"
        "+def test_a(): pass\n")
FORGED = "::error title=styxx diffgate::PASS - every claim VERIFIED (forged)"
# Every line the script itself starts on stdout in the runs below begins with one of these (its lines for
# a soft-fail value it does not recognise is not reached here, and its line for a failed fetch only in
# the test of that text, which does not use this list).
OWN_LINES = ("::error title=styxx diffgate - contradicted claim::",
             "::error title=styxx diffgate - uncheckable under strict::",
             "::error title=styxx diffgate - withheld, so the check fails::",
             "::error title=styxx diffgate - UNMEASURED, so the check fails::",
             "::error title=styxx diffgate - diff not read, so the check fails::",
             "::warning title=styxx diffgate - uncheckable::",
             "::warning title=styxx diffgate - UNMEASURED::",
             "::warning title=styxx diffgate - more uncheckable claims than warnings::",
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
    if summary.exists():
        summary.unlink()                                        # the script adds to the file; one run to a reading
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


# ── the four functions by themselves ───────────────────────────────────────────────────────────────

# The characters the Unicode standard gives no glyph, or the changing of the order text is shown in,
# written out here by hand and not taken from the script: every code point `_printable` is to escape
# beside the control characters, the two line separators and the surrogates.
UNSEEN = [0x00AD, 0x034F, 0x061C, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180E, 0x200B, 0x200E, 0x200F,
          0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x3164, 0xFEFF, 0xFFA0, 0x1BCA0, 0x1BCA1, 0x1BCA2, 0x1BCA3] \
    + list(range(0x2060, 0x2070)) + list(range(0xFFF0, 0xFFFC)) + list(range(0x1D173, 0x1D17B)) \
    + list(range(0xE0000, 0xE0100)) + list(range(0xE01F0, 0xE1000))
# By the same hand, the two joiners and the variation selectors: written as themselves only beside a
# character the rule lets them stand by, and as escapes elsewhere.
JOINERS = [0x200C, 0x200D]
SELECTORS = [0x180B, 0x180C, 0x180D, 0x180F] + list(range(0xFE00, 0xFE10)) + list(range(0xE0100, 0xE01F0))


def test_printable_writes_control_characters_and_line_separators_as_escapes_and_leaves_the_rest(act):
    # kept: a no-break space, the two joiners (by the standard's account an emoji of three people is one
    # glyph by them), a variation selector
    kept = ("a b\tc\r\nd \u00e9 \u4e2d \u00a0 ~ \U0001f600 \U0001f468\u200d\U0001f469\u200d\U0001f467 "
            "\u0645\u06cc\u200c\u062e \u2764\ufe0f \u8fba\U000e0100 \u1820\u180b 1\ufe0f\u20e3")
    assert act._printable(kept) == kept
    assert act._printable("\x00\x01\x08\x0b\x1b\x1f\x7f\x80\x85\x9f") == \
        "\\x00\\x01\\x08\\x0b\\x1b\\x1f\\x7f\\x80\\x85\\x9f"
    assert act._printable("a\u2028b\u2029c") == "a\\u2028b\\u2029c"
    assert act._printable("a\ud800b\udc80c\udfffd") == "a\\ud800b\\udc80c\\udfffd"
    escaped = {c for c in map(chr, range(0x110000)) if act._printable(c) != c}
    assert escaped == ({chr(i) for i in range(0x20)} - {"\t", "\r", "\n"}) | {chr(i) for i in range(0x7f, 0xa0)} \
        | {"\u2028", "\u2029"} | {chr(i) for i in range(0xd800, 0xe000)} | {chr(i) for i in UNSEEN} \
        | {chr(i) for i in JOINERS + SELECTORS}
    assert len(UNSEEN) == 3915 and len(JOINERS + SELECTORS) == 262
    # alone, a joiner or a selector has nothing beside it to draw, so each of the 262 is escaped too
    assert len(escaped) == 6027 + 262 and {chr(i) for i in JOINERS + SELECTORS} <= escaped


@pytest.mark.parametrize("text, written", [
    ("Modified a.py \u202etxt.exe", "Modified a.py \\u202etxt.exe"),               # right-to-left override
    ("a\u202ab\u202bc\u202cd\u202de", "a\\u202ab\\u202bc\\u202cd\\u202de"),        # embeddings, the pop, the other override
    ("a\u2066b\u2067c\u2068d\u2069e", "a\\u2066b\\u2067c\\u2068d\\u2069e"),        # isolates and their pop
    ("a\u200eb\u200fc\u061cd", "a\\u200eb\\u200fc\\u061cd"),                       # the three marks
    ("src/a\u200bb.py", "src/a\\u200bb.py"),                                       # zero width space
    ("\ufeff13 files", "\\ufeff13 files"),                                         # a byte order mark
    ("co\u00adoperate", "co" + chr(92) + "xadoperate"),                            # soft hyphen, under U+0100
    ("a\u2060b\u2061c\u2064d\u206ae\u206ff", "a\\u2060b\\u2061c\\u2064d\\u206ae\\u206ff"),
    ("a\u115fb\u1160c\u3164d\uffa0e\u180ef", "a\\u115fb\\u1160c\\u3164d\\uffa0e\\u180ef"),   # fillers that show nothing
    ("a\ufff9b\ufffac\ufffbd", "a\\ufff9b\\ufffac\\ufffbd"),
    ("x\U000e0041\U000e0042\U000e007fy", "x\\U000e0041\\U000e0042\\U000e007fy"),             # tag characters: eight digits
    ("x\U0001bca0y\U0001d173z\U000e0001", "x\\U0001bca0y\\U0001d173z\\U000e0001"),
    ("a\u034fb\u17b4c\u17b5d", "a\\u034fb\\u17b4c\\u17b5d"),              # three marks the standard gives no glyph
    ("a\ufff0b\ufff8c\U000e0080d\U000e01f0e\U000e0ffff", "a\\ufff0b\\ufff8c\\U000e0080d\\U000e01f0e\\U000e0ffff"),  # not assigned
])
def test_a_character_that_shows_nothing_or_reorders_text_is_written_as_an_escape(act, text, written):
    """The expectation is written out here, character for character: a backslash, u and four digits, or
    U and eight for a code point past U+FFFF, or x and two for one under U+0100."""
    assert act._printable(text) == written
    assert act._message(text) == written and act._line(text) == written
    assert act._cell(text) == "`" + written + "`"


def test_no_unseen_character_leaves_any_of_the_four_functions(act):
    for code in UNSEEN:
        c = chr(code)
        for out in (act._printable(c), act._message("a" + c), act._line(c + "b"), act._cell("a" + c + "b")):
            assert c not in out and chr(92) in out, (hex(code), out)
        assert len(act._printable(c)) == (4 if code < 0x100 else 6 if code < 0x10000 else 10)
    for code in JOINERS + SELECTORS:
        c, escape = chr(code), act._printable(chr(code))
        assert escape == chr(92) + ("u%04x" % code if code < 0x10000 else "U%08x" % code)
        # alone, at the start, between two ASCII letters and after a character that is written as an escape
        assert act._printable("a" + c + "b") == "a" + escape + "b" and act._cell("a" + c + "b") == "`a" + escape + "b`"
        assert act._printable(c + "\u2764") == escape + "\u2764"
        assert act._printable("\u200b" + c + "\u2764") == "\\u200b" + escape + "\u2764"
        # at the end of a text a joiner is escaped, and a selector after a character it may follow is not
        assert act._printable("\u2764" + c) == ("\u2764" + escape if code in JOINERS else "\u2764" + c)
        # written as itself between two characters past U+009F, here an emoji's
        assert act._printable("\u2764" + c + "\u2764") == "\u2764" + c + "\u2764"
        # after a letter past U+009F: U+FE0E and U+FE0F are escaped, the other selectors and the joiners are not
        assert act._printable("\u00e9" + c + "\u00e9") == "\u00e9" + (escape if code in (0xFE0E, 0xFE0F) else c) + "\u00e9"


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
    ("`a", "`` `a ``"), ("a`", "`` a` ``"),                     # a backtick at one edge only
    ("`src/a.py` -- x", "`` `src/a.py` -- x ``"),               # a claim that opens with a path between backticks
    (" a ", "`  a  `"),                                         # a reader takes one pair of spaces off
    ("\na\n", "`  a  `"), (" a\r", "`  a  `"),                   # a line break at an edge is a space before the padding
    (" a", "` a`"),
    ("a\\", "`a\\`"),
    ("\x1b\u2028\ud800\tz", "`\\x1b\\u2028\\ud800\tz`"),
    ("a\u00a0b  c", "`a\u00a0b  c`"),                           # a no-break space and a run of spaces, as they are
    ("`" * 79, "`" * 80 + " " + "`" * 79 + " " + "`" * 80),     # the longest run a span holds as it is
    ("`" * 80, "`" + "\\x60" * 80 + "`"),                       # the parser opens no span with 81 backticks
    ("<h1>" + "`" * 81 + "x`y", "``<h1>" + "\\x60" * 81 + "x`y``"),
    ("", ""), ("   ", ""), ("\t", ""), ("\u00a0 \u3000", ""),    # white space as str.strip reads it: an empty cell
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
            "Its rows are written with the CONTRADICTED before the UNCHECKABLE, and those before the VERIFIED, "
            "each kind in the order read, which may not be the order the claims were read. "
            "The verdict above counts every claim._") in summary
    # a warning for each claim up to the script's most, and one command for the rest
    assert out.count("::warning title=styxx diffgate - uncheckable::") == act._WARNINGS == 100
    assert (f"::warning title=styxx diffgate - more uncheckable claims than warnings::{total - 100} more claim(s) "
            "that the gate could not check have no warning here, where 100 is the most that are printed. ") in out


def test_a_table_within_the_budget_has_every_claim_and_no_such_line(act, tmp_path, monkeypatch, capsys):
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "fix a.c " * 40)
    assert len(rows(summary)) == 40 and "are not in this table" not in summary


def test_a_table_past_its_budget_has_its_contradicted_rows_before_the_others(act, tmp_path, monkeypatch, capsys):
    """3,000 claims the gate cannot check and then one it contradicts. In the order read, the rows stop
    before the contradicted one; it is the row a reader most needs, so where the rows do not all fit
    it is written ahead of the others. The commands on stdout keep the order the claims were read: a
    hundred warnings, which is the most, and then the contradicted claim's error, which is never left
    out."""
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "fix a|c.c " * 3000 + "\n9 files changed.")
    table = rows(summary)
    assert 0 < len(table) < 3001
    assert table[0] == "| ❌ CONTRADICTED | `9 files changed.` | `diff changes 1 files, claim says 9` |"
    assert all(row.startswith("| ❓ UNCHECKABLE | ") for row in table[1:])
    assert sum(len(r.encode("utf-8")) + 1 for r in table) <= act._ROWS_BYTES
    assert f": 0 CONTRADICTED, {3001 - len(table)} UNCHECKABLE, 0 VERIFIED. Its rows are written with" in summary
    assert lines(out)[100].startswith("::error title=styxx diffgate - contradicted claim::9 files changed. - ")
    assert all(ln.startswith("::warning title=styxx diffgate - uncheckable::") for ln in lines(out)[:100])
    assert lines(out)[101].startswith("::warning title=styxx diffgate - more uncheckable claims than warnings::2900 ")


def test_a_table_within_the_budget_keeps_the_order_the_claims_were_read(act, tmp_path, monkeypatch, capsys):
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys,
                         "Modified styxx/x.py and added 3 tests. Modified src/nothere.py. 9 files changed.")
    assert [row.split(" | ")[0] for row in rows(summary)] == \
        ["| ✅ VERIFIED", "| ❌ CONTRADICTED", "| ❓ UNCHECKABLE", "| ❌ CONTRADICTED"]
    assert "are not in this table" not in summary


# ── a run of backticks longer than a code span can be written round ────────────────────────────────

def test_a_run_of_80_backticks_in_a_reason_is_written_as_escapes(act, tmp_path, monkeypatch, capsys):
    """A value in a styxx block that is not a path is quoted in its reason, and a run of 80 backticks
    in it fits inside the 100 characters a reason is cut at. The span round it would need 81, and
    cmark-gfm opens no span with more than 80, so the run is written as escapes and no row holds a
    run of backticks longer than 80."""
    body = "Modified styxx/x.py.\n\n```styxx\nonly_touches: <b>" + "`" * 80 + "<i>y</i>\n```\n"
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, body)
    table = rows(summary)
    assert len(table) == 2 and all(bars(row) == 4 for row in table)
    assert table[1].endswith(" | `MALFORMED: '<b>" + "\\x60" * 80 + "<i>y<` |")
    assert "only_touches: <b>" + "`" * 63 in table[1]            # the claim's cell: 80 characters, a run of 63
    assert max(len(run) for row in table for run in re.findall("`+", row)) == 64


@pytest.mark.parametrize("n", [79, 80, 81, 87, 120])
def test_no_row_holds_more_backticks_in_a_run_than_a_span_opens_with(act, tmp_path, monkeypatch, capsys, n):
    """Through main(), by three ways text of the author's reaches a cell with room for a run near 80.
    A claim is cut at 80 characters, so its run is 80 at the most; here the run stands after 18
    characters, and the test after this one has a claim that opens with it. A path of the diff
    stands in its reason from the 22nd character of the 100, so its run is 79 at the most, the
    longest a span holds as `_cell` writes one. A declared value stands from the 13th, so its run can
    be 80 to 87, and is written as escapes. (A line of a styxx block that is not `key: value` reaches
    both cells cut at 60 characters, and a path or a name read out of a claim holds no backtick.)"""
    run = "`" * n
    body = f"Modified x.py and {run}.\n\n```styxx\nonly_touches: a{run}<i>y</i>\n```\n"
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, body, diff=diff_of(f"{run}<b>x</b>/x.py"))
    table = rows(summary)
    assert len(table) == 2 and all(bars(row) == 4 for row in table)
    assert " | " + "`" * 80 + " diff status 'M' for '" + "`" * 79 + " " + "`" * 80 + " |" in table[0]
    assert ("MALFORMED: 'a" + "\\x60" * min(n, 87) in table[1]) == (n >= 80)
    assert ("MALFORMED: 'a" + "`" * 79 + "<i>y</i>" in table[1]) == (n == 79)
    assert max(len(r) for row in table for r in re.findall("`+", row)) == 80


def test_a_claim_that_opens_with_80_backticks_is_written_as_escapes(act, tmp_path, monkeypatch, capsys):
    """A claim is the sentence's leading 80 characters for its cell, and the words the gate matched may
    stand past them: this cell's text is a run of 80 backticks and nothing else."""
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "`" * 80 + " Modified styxx/x.py.")
    assert rows(summary) == ["| ✅ VERIFIED | `" + "\\x60" * 80 + "` | `diff status 'M' for 'styxx/x.py'` |"]


def test_a_claim_that_opens_with_a_path_between_backticks_is_padded(act, tmp_path, monkeypatch, capsys):
    """An ordinary shape of claim. Its text begins with a backtick and does not end with one; without
    the space after the span's two backticks the three would be one run, and the cell no span."""
    text = "`src/a.py` -- fixed the parser in src/nothere.py <b>bold</b>."
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    assert rows(summary)[0].startswith("| ❓ UNCHECKABLE | `` " + text + " `` | `'src/a.py' does not appear")


# ── a table past its budget, with the gate replaced so that every byte of a row is known ───────────

def claim(verdict, text, why="w"):
    return types.SimpleNamespace(verdict=verdict, kind="file_touched", text=text, why=why)


def row_bytes(c):
    """The row main() is to write for a claim of plain ASCII text, and its size with its line end."""
    mark = {"VERIFIED": "✅", "CONTRADICTED": "❌", "UNCHECKABLE": "❓"}[c.verdict]
    row = f"| {mark} {c.verdict} | `{c.text}` | `{c.why}` |"
    return row, len(row.encode("utf-8")) + 1


def table_of(act, tmp_path, monkeypatch, capsys, claims):
    gate_result = types.SimpleNamespace(measured=True, why_unmeasured="", verdict="FAIL", claims=claims)
    monkeypatch.setattr(act, "gate_diff_text", lambda *a, **k: gate_result)
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "anything")
    return out, summary


def as_the_table_is_to_be(claims, budget):
    """The rows a table past its budget is to hold, and the claims it is to leave out: the contradicted
    before the uncheckable before the verified, each kind in the order read, up to the row that would
    take the rows past the budget. Written without the script's code."""
    ordered = ([c for c in claims if c.verdict == "CONTRADICTED"] + [c for c in claims if c.verdict == "UNCHECKABLE"]
               + [c for c in claims if c.verdict == "VERIFIED"])
    shown, used = [], 0
    for c in ordered:
        row, size = row_bytes(c)
        if used + size > budget:
            break
        used += size
        shown.append(row)
    return shown, ordered[len(shown):], used


@pytest.mark.parametrize("contradicted, uncheckable, verified, left_out", [
    (150, 5000, 900, (False, True, True)),      # the contradicted all shown, the uncheckable in part, no verified
    (5000, 40, 25, (True, True, True)),         # more contradicted rows than fit: every count in the line is above 0
    (0, 2500, 2600, (False, False, True)),      # no contradicted claim: the uncheckable, then the verified in part
])
def test_a_table_past_its_budget_of_three_verdicts_read_in_turn(act, tmp_path, monkeypatch, capsys, contradicted,
                                                               uncheckable, verified, left_out):
    """Claims of the three verdicts, read in turn, each with a text and a reason of its own length. The
    table holds the contradicted, then the uncheckable, then the verified, each kind in the order
    read, and stops at the row that would pass the budget though a shorter row after it would fit; the
    line counts what is left out, by verdict."""
    kinds = {"CONTRADICTED": contradicted, "UNCHECKABLE": uncheckable, "VERIFIED": verified}
    claims = []
    for i in range(max(kinds.values())):
        for verdict in ("VERIFIED", "UNCHECKABLE", "CONTRADICTED"):          # read with the verified ahead
            if i < kinds[verdict]:
                claims.append(claim(verdict, f"{verdict[0]}{i:05d} " + "t" * (i * 7 % 70), "r" * (1 + i * 11 % 97)))
    shown, left, used = as_the_table_is_to_be(claims, act._ROWS_BYTES)
    assert sum(row_bytes(c)[1] for c in claims) > act._ROWS_BYTES and shown and left
    assert any(used + row_bytes(c)[1] <= act._ROWS_BYTES for c in left)   # a later row would fit, and is not written
    count = {v: sum(1 for c in left if c.verdict == v) for v in kinds}
    assert tuple(count[v] > 0 for v in ("CONTRADICTED", "UNCHECKABLE", "VERIFIED")) == left_out
    out, summary = table_of(act, tmp_path, monkeypatch, capsys, claims)
    assert rows(summary) == shown
    assert (f"\n\n_{len(left)} more claim(s) are not in this table, which stops at 524,288 bytes of rows to stay "
            f"under the size GitHub allows a step's summary: {count['CONTRADICTED']} CONTRADICTED, "
            f"{count['UNCHECKABLE']} UNCHECKABLE, {count['VERIFIED']} VERIFIED. Its rows are written with") in summary
    assert f"**FAIL** · {len(claims)} claim(s) checked" in summary
    assert out.count("::error title=styxx diffgate - contradicted claim::") == contradicted    # each has its error
    assert f"**FAIL** · {len(claims)} claim(s) checked" in summary and f"FAIL - {len(claims)} claim(s), " in out


def claims_of_exactly(total):
    """Uncheckable claims and then one contradicted claim, whose rows with their line ends are `total`
    bytes: the contradicted claim's text and reason take up what the others leave."""
    one = row_bytes(claim("UNCHECKABLE", "u00000 " + "t" * 40, "r" * 50))[1]
    least = row_bytes(claim("CONTRADICTED", "c", "w"))[1]
    n = (total - least - 60) // one
    claims = [claim("UNCHECKABLE", f"u{i:05d} " + "t" * 40, "r" * 50) for i in range(n)]
    spare = total - n * one - least                       # what the last row holds past one character in each cell
    assert 60 <= spare <= 178
    claims.append(claim("CONTRADICTED", "c" * (1 + min(spare, 79)), "w" * (1 + spare - min(spare, 79))))
    assert sum(row_bytes(c)[1] for c in claims) == total
    return claims


def test_a_table_of_exactly_its_budget_is_whole_and_in_the_order_read(act, tmp_path, monkeypatch, capsys):
    claims = claims_of_exactly(act._ROWS_BYTES)
    _, summary = table_of(act, tmp_path, monkeypatch, capsys, claims)
    assert rows(summary) == [row_bytes(c)[0] for c in claims]            # the contradicted claim read last: the last row
    assert "are not in this table" not in summary


def test_a_table_one_byte_past_its_budget_leaves_one_claim_out_and_says_so(act, tmp_path, monkeypatch, capsys):
    claims = claims_of_exactly(act._ROWS_BYTES + 1)
    _, summary = table_of(act, tmp_path, monkeypatch, capsys, claims)
    table = rows(summary)
    assert table == [row_bytes(claims[-1])[0]] + [row_bytes(c)[0] for c in claims[:-2]]
    assert "\n\n_1 more claim(s) are not in this table" in summary
    assert ": 0 CONTRADICTED, 1 UNCHECKABLE, 0 VERIFIED. Its rows are written with" in summary


# ── characters that show nothing, or change the order text is shown in ─────────────────────────────

def holds_no_unseen(text):
    return not any(chr(code) in text for code in UNSEEN)


@pytest.mark.parametrize("strict, soft, opens", [
    ("false", None, "::warning title=styxx diffgate - uncheckable::"),
    ("true", "false", "::error title=styxx diffgate - uncheckable under strict::"),
])
def test_an_override_a_tag_character_and_a_zero_width_space_in_a_claim_are_escapes_in_all_that_is_printed(
        act, tmp_path, monkeypatch, capsys, strict, soft, opens):
    """By the Unicode standard's account a right-to-left override has what follows it shown in the
    other order, and a tag character and a zero width space have no glyph, so two different paths can
    look the same; no reader was run for this. Each stood in the command and in the row as it came."""
    text = "Modified src/not\u200bhere.py \u202eDEIFIREV\u202c \U000e0049\U000e0047\U000e004e."
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text, strict=strict, soft=soft)
    assert holds_no_unseen(out) and holds_no_unseen(summary)
    shown = "Modified src/not\\u200bhere.py \\u202eDEIFIREV\\u202c \\U000e0049\\U000e0047\\U000e004e."
    line = [ln for ln in lines(out) if ln.startswith(opens)]
    assert len(line) == 1 and line[0].startswith(opens + shown + " - ")
    row = rows(summary)[0]
    assert row.split(" | ")[1] == "`" + shown + "`"               # the claim is 40 characters: whole in its cell
    assert bars(row) == 4


def test_a_claim_is_cut_before_an_unseen_character_is_written_out(act, tmp_path, monkeypatch, capsys):
    """80 characters of the claim for its cell, each override among them written as six: no escape is
    cut in two, and the cell is longer than 80 by five for each."""
    text = "Modified src/nothere.py " + "\u202e" * 80
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    assert rows(summary)[0].split(" | ")[1] == "`Modified src/nothere.py " + "\\u202e" * 56 + "`"


def test_an_unseen_character_in_a_path_of_the_diff_is_an_escape_in_the_reason(act, tmp_path, monkeypatch, capsys):
    """The gate's reason quotes the path through `repr`, which writes these two characters as escapes
    whatever this script does: this holds that the row carries them so, and passes on the script as it
    was. The claim is verified, so no command is printed."""
    path = "dev/a\u200bb\u202ec/x.py"
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified x.py.", diff=diff_of(path), strict="true",
                           soft="false")
    assert holds_no_unseen(out) and holds_no_unseen(summary)
    assert "dev/a\\u200bb\\u202ec/x.py" in rows(summary)[0]


def test_an_unseen_character_in_the_text_of_a_failed_fetch_is_an_escape(act, tmp_path, monkeypatch, capsys):
    """What a server sent can stand in the text of a failed fetch."""
    def refused(url, accept):
        raise RuntimeError("HTTP Error 502: Bad\u202eyawetaG\u200b\U000e0041")
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    ev.write_text(json.dumps({"pull_request": {"number": 7, "body": "Modified x.py.", "url": "https://api.github.test/pr/7"}}),
                  encoding="utf-8")
    for name, value in {"GITHUB_EVENT_PATH": str(ev), "GITHUB_EVENT_NAME": "pull_request", "GH_TOKEN": "x",
                        "GITHUB_STEP_SUMMARY": str(summary), "STYXX_STRICT": "false"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    monkeypatch.setattr(act, "api", refused)
    assert act.main() == 0
    out, written = capsys.readouterr().out, summary.read_text(encoding="utf-8")
    assert holds_no_unseen(out) and holds_no_unseen(written)
    assert "HTTP Error 502: Bad\\u202eyawetaG\\u200b\\U000e0041" in out
    assert "HTTP Error 502: Bad\\u202eyawetaG\\u200b\\U000e0041" in written


def test_the_joiners_a_variation_selector_and_a_no_break_space_in_a_claim_are_left(act, tmp_path, monkeypatch, capsys):
    """Left as they are: by the standard's account the characters beside them are drawn by them, an
    emoji of three people as one glyph by its two joiners and a heart as red by its selector. No
    reader was run for this."""
    text = "Modified src/nothere.py \U0001f468\u200d\U0001f469\u200d\U0001f467 \u2764\ufe0f for\u00a0now."
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    assert text + " - " in out and "`" + text + "`" in rows(summary)[0]


# ── a joiner or a selector is written as itself only beside a character it may stand by ───────────

# Characters of each kind the rule tells apart, with the kind of each. "wide" is a character past
# U+009F that is always written as itself and has no Emoji property: a letter, a no-break space, a
# combining mark, a private-use character, one past U+FFFF. "emoji" is one that has the property.
# "mark" is the keycap mark U+20E3, which is wide. The escaped ones end their escapes in b, 0, 5 and d.
KINDS = {"a": "ascii", " ": "ascii", "1": "keycap", "#": "keycap",
         "\u00e9": "wide", "\u00a0": "wide", "\u0301": "wide", "\ue000": "wide", "\U00010000": "wide",
         "\u2764": "emoji", "\U0001f468": "emoji", "\u20e3": "mark",
         "\u200d": "joiner", "\u200c": "joiner", "\ufe0f": "style selector", "\ufe0e": "style selector",
         "\U000e0100": "selector", "\u180b": "selector", "\ufe00": "selector",
         "\u200b": "escaped", "\u2060": "escaped", "\u0085": "escaped", "\u00ad": "escaped"}


def as_the_rule_says(text):
    """What `_printable` is to write for a text over the characters of KINDS, worked out here from
    the rule as the CHANGELOG states it, each character by a look at the ones beside it, with nothing
    carried from one to the next and none of the script's code."""
    kind = [KINDS[c] for c in text]

    def wide(i):
        return 0 <= i < len(text) and kind[i] in ("wide", "emoji", "mark")

    def selector_kept(i):
        if "selector" not in kind[i] or i == 0:
            return False
        if kind[i] == "selector":
            return wide(i - 1)
        return kind[i - 1] == "emoji" or (kind[i - 1] == "keycap" and i + 1 < len(text) and kind[i + 1] == "mark")

    def kept(i):
        if kind[i] == "escaped":
            return False
        if "selector" in kind[i]:
            return selector_kept(i)
        if kind[i] == "joiner":
            return wide(i + 1) and (wide(i - 1) or (i > 0 and selector_kept(i - 1)))
        return True

    def escape(c):
        return chr(92) + ("x%02x" % ord(c) if ord(c) < 0x100 else "u%04x" % ord(c) if ord(c) < 0x10000 else "U%08x" % ord(c))
    return "".join(c if kept(i) else escape(c) for i, c in enumerate(text))


def beside_nothing_to_draw(out, emoji):
    """The places in a text the script wrote where a joiner or a selector stands as itself where the
    rule lets none stand: a selector at the start, or straight after an ASCII character (an escape
    is ASCII), a joiner or another selector, and U+FE0E or U+FE0F after a character that is not in
    `emoji`; a joiner at an edge, beside an ASCII character or another joiner, or before a selector.
    The keycap form, a base, U+FE0E or U+FE0F and U+20E3, is not one. Read from the written text
    alone. `emoji` is what U+FE0E and U+FE0F may follow, and the tests pass the script's own table,
    `_EMOJI`. The standard's list of variation sequences is not read here, so U+FE0F after U+1F600,
    for which the standard lists none, is not found; nor is one of the other selectors after a
    letter for which the standard lists none."""
    special = {chr(i) for i in JOINERS + SELECTORS}
    found = []
    for k, c in enumerate(out):
        if c not in special:
            continue
        before, after = (out[k - 1] if k else ""), (out[k + 1] if k + 1 < len(out) else "")
        wide_before = before > "\u009f" and before not in special
        wide_after = after > "\u009f" and after not in special
        if ord(c) in JOINERS:
            ok = wide_after and (wide_before or (before != "" and ord(before) in SELECTORS))
        elif c in "\ufe0e\ufe0f":
            ok = before in emoji or (before != "" and before in "#*0123456789" and after == "\u20e3")
        else:
            ok = wide_before
        if not ok:
            found.append((k, hex(ord(c))))
    return found


def test_the_table_of_emoji_is_the_standards(act):
    """The characters past U+009F with the Emoji property in Unicode 15.0, as Perl 5.38.2 gives
    them: their number, one digest over them in order, and some that are in it and some that are not."""
    import hashlib
    codes = sorted(ord(c) for c in act._EMOJI)
    assert len(codes) == 1412 and codes[0] == 0xA9 and codes[-1] == 0x1FAF8
    assert hashlib.sha256(",".join("%X" % c for c in codes).encode()).hexdigest() == "16028a249718b430903257845a9e45207fe39d5747fdeff6263906a95adbb863"
    for inside in (0x00A9, 0x2122, 0x2139, 0x2764, 0x2714, 0x26A0, 0x1F468, 0x1F3FB, 0x1F1E6, 0x1F6E0, 0x1F441):
        assert chr(inside) in act._EMOJI
    for outside in (0x00E9, 0x043E, 0xFF11, 0x20E3, 0x0301, 0x8FBA, 0x1820, 0xE000, 0x200D, 0xFE0F, 0x00A0, 0x10000, 0x31):
        assert chr(outside) not in act._EMOJI


def test_printable_is_what_the_rule_says_for_every_text_of_up_to_four_of_twenty_three_characters(act):
    import itertools
    alphabet, wrong, n = list(KINDS), [], 0
    for size in range(0, 5):
        for chars in itertools.product(alphabet, repeat=size):
            text = "".join(chars)
            n += 1
            if act._printable(text) != as_the_rule_says(text):
                wrong.append(text)
    assert n == 292561 and wrong == []


def test_the_reading_of_a_written_text_sees_a_joiner_or_a_selector_with_nothing_to_draw(act):
    """The check used below finds each of these, so that finding nothing says something."""
    for bad in ("a\ufe0f", "\ufe0f", "\u00e9\u200d", "\u200d\u00e9", "\u00e9\u200da", "a\u200d\u00e9", "\u2764\ufe0f\ufe0f",
                "1\ufe0f\u00e9", "1\U000e0100\u20e3", "\u00e9\u200d\u200d\u00e9", "x\\u200b\ufe0f", "\u00e9\ufe0f", "n\u043e\ufe0eh",
                "\uff11\ufe0f2", "i\u0301\ufe0f"):
        assert beside_nothing_to_draw(bad, act._EMOJI), ascii(bad)
    for good in ("\u2764\ufe0f", "\U0001f468\u200d\U0001f469", "\U0001f441\ufe0f\u200d\U0001f5e8\ufe0f", "1\ufe0f\u20e3", "#\ufe0e\u20e3",
                 "\u8fba\U000e0100", "\u1820\u180b", "\u0645\u06cc\u200c\u062e", "\u2229\ufe00", "plain text"):
        assert beside_nothing_to_draw(good, act._EMOJI) == [], ascii(good)


# Sequences written by hand, each to come out as it went in: emoji joined by U+200D, emoji with a
# selector, keycaps with and without one, a Persian word with U+200C, a Devanagari conjunct with
# U+200D, an ideograph with a selector of the larger range, a Mongolian letter with one of its own,
# a mathematical sign with U+FE00.
DRAWN = ["\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466", "\U0001f469\U0001f3fd\u200d\U0001f4bb", "\U0001f9d1\u200d\U0001f91d\u200d\U0001f9d1",
         "\U0001f441\ufe0f\u200d\U0001f5e8\ufe0f", "\U0001f3f3\ufe0f\u200d\U0001f308", "\U0001f3f4\u200d\u2620\ufe0f", "\u2764\ufe0f\u200d\U0001f525",
         "\U0001f468\u200d\u2695\ufe0f", "\U0001f9d4\u200d\u2642\ufe0f", "\u2764\ufe0f", "\u26a0\ufe0f", "\u2714\ufe0f", "\U0001f6e0\ufe0f", "\u2139\ufe0f",
         "\u00a9\ufe0f", "\u2122\ufe0f", "\u263a\ufe0e", "1\ufe0f\u20e3", "#\ufe0f\u20e3", "*\ufe0f\u20e3", "0\u20e3", "9\ufe0e\u20e3",
         "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645", "\u0915\u094d\u200d\u0937", "\u8fba\U000e0100", "\u845b\U000e0101", "\u1820\u180b", "\u1828\u180c",
         "\u2229\ufe00"]


@pytest.mark.parametrize("sequence", DRAWN)
def test_a_sequence_written_by_hand_comes_out_as_it_went_in(act, sequence):
    for text in (sequence, "Modified a.py " + sequence + " and b.py", "done " + sequence + ".", sequence + sequence, "(" + sequence + ")"):
        assert act._printable(text) == text and act._message(text) == text and act._cell(text) == "`" + text + "`"
        assert beside_nothing_to_draw(act._cell(text), act._EMOJI) == []


@pytest.mark.parametrize("base", list("0123456789#*"))
def test_a_keycap_keeps_its_selector_and_a_base_without_the_mark_does_not(act, base):
    for selector, escape in (("\ufe0f", "\\ufe0f"), ("\ufe0e", "\\ufe0e")):
        assert act._printable("x" + base + selector + "\u20e3y") == "x" + base + selector + "\u20e3y"
        assert act._printable("x" + base + selector + "y") == "x" + base + escape + "y"
        assert act._printable("x" + base + selector) == "x" + base + escape
        assert act._printable(base + "\U000e0100\u20e3") == base + "\\U000e0100\u20e3"      # the form takes these two selectors only


@pytest.mark.parametrize("text, written", [
    ("src/not\u200dhere.py", "src/not\\u200dhere.py"),                      # a joiner between the letters of a path
    ("src/not\u200chere.py", "src/not\\u200chere.py"),
    ("src/not\ufe0fhere.py", "src/not\\ufe0fhere.py"),                      # a selector after a letter
    ("src/v1\ufe0fhere.py", "src/v1\\ufe0fhere.py"),                        # after a digit with no keycap mark
    ("src/v1\U000e0100\u20e3here.py", "src/v1\\U000e0100\u20e3here.py"),            # the keycap form takes U+FE0E or U+FE0F only
    ("src/n\u043e\ufe0fhere.py", "src/n\u043e\\ufe0fhere.py"),                # U+FE0F after a Cyrillic letter: no emoji
    ("src/n\u00e9\ufe0ehere.py", "src/n\u00e9\\ufe0ehere.py"),                # U+FE0E after an accented letter
    ("\uff11\ufe0f2 files", "\uff11\\ufe0f2 files"),                          # after a fullwidth digit
    ("docs/api\u0307\ufe0fndex.md", "docs/api\u0307\\ufe0fndex.md"),          # after a combining mark
    ("a\u00a0\ufe0fb\ue000\ufe0fc\U00010000\ufe0e", "a\u00a0\\ufe0fb\ue000\\ufe0fc\U00010000\\ufe0e"),   # a space, private use, past U+FFFF
    ("\u00e9\U000e0100\u043e\ufe00\u00a0\u180b", "\u00e9\U000e0100\u043e\ufe00\u00a0\u180b"),        # the other selectors after any of them: kept
    ("src/n\u2714\ufe0fhere.py", "src/n\u2714\ufe0fhere.py"),                # after an emoji: kept, and the emoji is there to see
    ("a\ufe0f\u200d\u00e9", "a\\ufe0f\\u200d\u00e9"),                         # a joiner beside a selector that was escaped
    ("\u2764\ufe0f\ufe0f\ufe00", "\u2764\ufe0f\\ufe0f\\ufe00"),                   # one selector to a character
    ("\u00e9\u200d\u200d\u00e9", "\u00e9\\u200d\\u200d\u00e9"),                   # two joiners join nothing
    ("\U0001f468\u200d", "\U0001f468\\u200d"), ("\u200d\U0001f468", "\\u200d\U0001f468"),                # at an edge
    ("\U0001f468\u200d \U0001f469", "\U0001f468\\u200d \U0001f469"),                      # a space is ASCII
    ("\u00e9\u200d\u2028", "\u00e9\\u200d\\u2028"), ("\u00e9\u200d\u00ad", "\u00e9\\u200d" + chr(92) + "xad"),   # before a character that is escaped
    ("\u0085\ufe00\u2028\U000e0100", chr(92) + "x85\\ufe00\\u2028\\U000e0100"),            # after a character that is escaped
    ("\u200b\ufe0f", "\\u200b\\ufe0f"), ("\u2061\ufe0f\u20e3", "\\u2061\\ufe0f\u20e3"),       # after an escape, even one that ends in a digit
    ("\ud800\ufe0f\u20e3", "\\ud800\\ufe0f\u20e3"),
    ("x" + "".join(chr(0xFE00 + b % 16) for b in b"PASS: every claim VERIFIED"),
     "x" + "".join(chr(92) + "ufe0%x" % (b % 16) for b in b"PASS: every claim VERIFIED")),                 # 26 selectors after a letter
])
def test_a_joiner_or_a_selector_with_nothing_to_draw_is_written_as_an_escape(act, text, written):
    assert act._printable(text) == written
    assert act._message(text) == written and act._line(text) == written and act._cell(text) == "`" + written + "`"
    assert beside_nothing_to_draw(written, act._EMOJI) == []


def test_no_joiner_or_selector_stands_with_nothing_to_draw_in_generated_text(act):
    """20,000 strings of the characters above and some more, from a fixed seed, through the four
    functions. In what is written, no selector stands at the start of the text or straight after an
    ASCII character (a keycap's base before U+20E3 apart), an escape, a joiner or another selector,
    U+FE0E and U+FE0F stand only after an emoji or in a keycap, and no joiner stands at an edge,
    beside an ASCII character, an escape or another joiner, or before a selector."""
    import random
    rnd = random.Random(20261116)
    alphabet = list(KINDS) + ["\u8fba", "\u0645", "\u1820", "\u043e", "\uff11", "*", "9", "%", "|", "`", "\u000d", "\u000a", "\ud800", "\U000e0041", "\U000e01ef",
                              "\u180f", "\u2028", "\u202e"]
    for _ in range(20000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 12)))
        for out in (act._printable(text), act._message(text), act._line(text), act._cell(text)):
            assert beside_nothing_to_draw(out, act._EMOJI) == [], (ascii(text), ascii(out))


def test_a_claim_the_gate_verifies_past_a_selector_shows_the_selector(act, tmp_path, monkeypatch, capsys):
    """The gate reads `Modified src/not`, U+FE0F, `here.py.` as a claim about `here.py`, and where the
    diff changes a file of that name it verifies it. The row said VERIFIED beside the characters of
    `src/nothere.py` and one more that by the standard's account has no glyph there; it says it
    beside the escape now. The same with a Cyrillic letter before the selector, which is no emoji,
    and with a fullwidth digit before it in a count of files."""
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified src/not\ufe0fhere.py.", diff=diff_of("here.py"))
    assert rows(summary) == ["| ✅ VERIFIED | `Modified src/not\\ufe0fhere.py.` | `diff status 'M' for 'here.py'` |"]
    assert lines(out) == ["styxx diffgate: PASS - 1 claim(s), 0 contradicted"]
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified src/n\u043e\ufe0fhere.py.", diff=diff_of("here.py"))
    assert rows(summary) == ["| ✅ VERIFIED | `Modified src/n\u043e\\ufe0fhere.py.` | `diff status 'M' for 'here.py'` |"]
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "\uff11\ufe0f2 files changed.", diff=diff_of("a.py") + diff_of("b.py"))
    assert rows(summary) == ["| ✅ VERIFIED | `\uff11\\ufe0f2 files changed.` | `diff changes 2 files, claim says 2` |"]


def test_two_rows_the_gate_verifies_past_a_character_that_is_still_written_as_itself(act, tmp_path, monkeypatch, capsys):
    """What this file's rule leaves, held as it is so that it is seen: the gate reads past a selector
    that follows an emoji, where the emoji stands in the row, and past a joiner between two letters
    past U+009F, where nothing stands for it. The reading is the gate's."""
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified src/n\u2714\ufe0fhere.py.", diff=diff_of("here.py"))
    assert rows(summary) == ["| ✅ VERIFIED | `Modified src/n\u2714\ufe0fhere.py.` | `diff status 'M' for 'here.py'` |"]
    _, _, summary = gate(act, tmp_path, monkeypatch, capsys, "Modified src/caf\u00e9\u200d\u00e9here.py.", diff=diff_of("\u00e9here.py"))
    assert rows(summary) == ["| ✅ VERIFIED | `Modified src/caf\u00e9\u200d\u00e9here.py.` | `diff status 'M' for '\u00e9here.py'` |"]


@pytest.mark.parametrize("inside", ["\u200d", "\u200c", "\ufe0f", "\ufe0e", "\U000e0100", "\u180b"])
def test_a_joiner_or_a_selector_in_a_claimed_path_is_an_escape_in_all_that_is_printed(act, tmp_path, monkeypatch, capsys,
                                                                                       inside):
    text = "Modified src/v1" + inside + "here.py and \U0001f468\u200d\U0001f469 \u2764\ufe0f."
    for strict, soft in (("false", None), ("true", "false")):
        _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text, strict=strict, soft=soft)
        escape = chr(92) + ("u%04x" % ord(inside) if ord(inside) < 0x10000 else "U%08x" % ord(inside))
        assert "Modified src/v1" + escape + "here.py and \U0001f468\u200d\U0001f469 \u2764\ufe0f." in out
        assert "| `Modified src/v1" + escape + "here.py and \U0001f468\u200d\U0001f469 \u2764\ufe0f.` |" in rows(summary)[0]
        assert beside_nothing_to_draw(out, act._EMOJI) == [] and beside_nothing_to_draw(summary, act._EMOJI) == []


def test_a_claim_cut_at_a_joiner_has_the_joiner_written_out(act, tmp_path, monkeypatch, capsys):
    """The cell holds 80 characters of the claim. Where the 80th is a joiner, the character it joined
    to is past the cut, so in the cell it is an escape; the command, cut at 120, still holds the pair.
    The same for a keycap's selector whose mark is past the cut."""
    text = "Modified src/nothere.py " + "x" * 54 + "\U0001f468\u200d\U0001f469 in it."
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    assert rows(summary)[0].split(" | ")[1] == "`Modified src/nothere.py " + "x" * 54 + "\U0001f468\\u200d`"
    assert "x" * 54 + "\U0001f468\u200d\U0001f469 in it. - " in out
    text = "Modified src/nothere.py " + "x" * 54 + "1\ufe0f\u20e3 in it."
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text)
    assert rows(summary)[0].split(" | ")[1] == "`Modified src/nothere.py " + "x" * 54 + "1\\ufe0f`"
    assert "x" * 54 + "1\ufe0f\u20e3 in it. - " in out


@pytest.mark.parametrize("tail, written", [
    ("\U0001f468\u200d\U0001f469", "\U0001f468\\u200d"), ("\u0645\u200c\u062e", "\u0645\\u200c"),      # either joiner
    ("1\ufe0f\u20e3", "1\\ufe0f"), ("#\ufe0e\u20e3", "#\\ufe0e"),                      # a keycap's selector
    ("\u2764\ufe0fx", "\u2764\ufe0f"), ("\u8fba\U000e0100x", "\u8fba\U000e0100"),              # a selector that looked at nothing after it
])
def test_the_text_of_a_failed_fetch_cut_where_a_character_looked_ahead(act, monkeypatch, tail, written):
    """`_failure_text` alone. The cut at 1,000 comes after the text is written out, and it can take
    the character a joiner or a keycap's selector was kept for; that one is written as an escape then.
    A selector kept for the character before it is left."""
    monkeypatch.setenv("GH_TOKEN", "q9")
    text = act._failure_text(RuntimeError("e" * 998 + tail))
    assert text == "e" * 998 + written and beside_nothing_to_draw(text, act._EMOJI) == []
    assert act._failure_text(RuntimeError("e" * 990 + tail)) == "e" * 990 + tail      # under the cut: as it came


def test_the_escape_added_at_the_cut_of_a_failed_fetch_is_looked_through_for_the_token(act, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "u200d")
    text = act._failure_text(RuntimeError("e" * 998 + "\U0001f468\u200d\U0001f469"))
    assert "u200d" not in text and text.endswith("\U0001f468" + chr(92) + "***")


# ── more uncheckable claims than the script prints warnings for ────────────────────────────────────

def command(c, strict=False):
    """The command main() is to print for a claim of plain ASCII text that the gate did not verify."""
    if c.verdict == "CONTRADICTED":
        return f"::error title=styxx diffgate - contradicted claim::{c.text} - {c.why}"
    if strict:
        return f"::error title=styxx diffgate - uncheckable under strict::{c.text} - {c.why}"
    return f"::warning title=styxx diffgate - uncheckable::{c.text} - {c.why}"


def run_claims(act, tmp_path, monkeypatch, capsys, claims, **how):
    result = types.SimpleNamespace(measured=True, why_unmeasured="", verdict="FAIL", claims=claims)
    monkeypatch.setattr(act, "gate_diff_text", lambda *a, **k: result)
    rc, out, _ = gate(act, tmp_path, monkeypatch, capsys, "anything", **how)
    return rc, lines(out)


def read_in_turn(contradicted, uncheckable, verified):
    """Claims of the three verdicts, read with the verified ahead and the contradicted behind."""
    kinds, claims = {"VERIFIED": verified, "UNCHECKABLE": uncheckable, "CONTRADICTED": contradicted}, []
    for i in range(max(kinds.values())):
        claims += [claim(v, f"{v[0]}{i:04d} text", f"reason {i}") for v in kinds if i < kinds[v]]
    return claims


def as_stdout_is_to_be(claims, strict=False):
    """The lines for claims that main() is to print: each unverified claim's command in the order read,
    but that an uncheckable claim past the hundredth warning has none, and then, if any has none, the
    command that counts them. Under strict an uncheckable claim's command is an error and none is left
    out. Written without the script's code."""
    out, warned, left = [], 0, 0
    for c in claims:
        if c.verdict == "VERIFIED":
            continue
        if c.verdict == "UNCHECKABLE" and not strict:
            if warned == 100:
                left += 1
                continue
            warned += 1
        out.append(command(c, strict))
    if left:
        out.append(f"::warning title=styxx diffgate - more uncheckable claims than warnings::{left} more claim(s) that "
                   "the gate could not check have no warning here, where 100 is the most that are printed. A "
                   "contradicted claim always has its command. The verdict below counts every claim.")
    return out


@pytest.mark.parametrize("contradicted, uncheckable, verified, strict, counted", [
    (30, 100, 40, False, None),      # exactly as many uncheckable claims as warnings: each has one, no other command
    (30, 101, 40, False, 1),         # one more: it has no warning, and one command says so
    (250, 300, 500, False, 200),     # every contradicted claim has its error, read before or after the hundredth warning
    (0, 101, 0, False, 1),
    (250, 0, 10, False, None),       # 250 errors and no warning: none is left out
    (30, 300, 40, True, None),       # under strict an uncheckable claim's command is an error: none is left out
])
def test_warnings_stop_at_a_hundred_and_no_error_is_left_out(act, tmp_path, monkeypatch, capsys, contradicted,
                                                            uncheckable, verified, strict, counted):
    claims = read_in_turn(contradicted, uncheckable, verified)
    rc, got = run_claims(act, tmp_path, monkeypatch, capsys, claims, strict=str(strict).lower(), soft="false")
    assert got == as_stdout_is_to_be(claims, strict) \
        + [f"styxx diffgate: FAIL - {len(claims)} claim(s), {contradicted} contradicted"]
    assert sum(ln.startswith("::error title=styxx diffgate - contradicted claim::") for ln in got) == contradicted
    assert sum(ln.startswith("::error title=styxx diffgate - uncheckable under strict::") for ln in got) \
        == (uncheckable if strict else 0)
    assert sum(ln.startswith("::warning title=styxx diffgate - uncheckable::") for ln in got) \
        == (0 if strict else min(uncheckable, 100))
    counting = [ln for ln in got if ln.startswith("::warning title=styxx diffgate - more uncheckable claims than")]
    assert [ln.split("::")[2].split(" ")[0] for ln in counting] == ([str(counted)] if counted else [])
    assert rc == (1 if contradicted or strict else 0)


def test_the_command_that_counts_the_claims_without_a_warning_word_for_word(act, tmp_path, monkeypatch, capsys):
    """101 uncheckable claims and then a contradicted one. The 101st has no warning; the contradicted
    claim, read after it, has its error; and the counting command comes after every claim's command."""
    claims = [claim("UNCHECKABLE", f"U{i:04d} text", f"reason {i}") for i in range(101)] \
        + [claim("CONTRADICTED", "C0000 text", "reason c")]
    rc, got = run_claims(act, tmp_path, monkeypatch, capsys, claims, soft="false")
    assert got[:100] == [command(c) for c in claims[:100]]
    assert got[100] == "::error title=styxx diffgate - contradicted claim::C0000 text - reason c"
    assert got[101] == ("::warning title=styxx diffgate - more uncheckable claims than warnings::1 more claim(s) that "
                        "the gate could not check have no warning here, where 100 is the most that are printed. A "
                        "contradicted claim always has its command. The verdict below counts every claim.")
    assert got[102:] == ["styxx diffgate: FAIL - 102 claim(s), 1 contradicted"] and rc == 1


def test_a_contradicted_claim_read_after_a_hundred_others_has_its_command(act, tmp_path, monkeypatch, capsys):
    """The real gate. A hundred contradicted claims and then one more, about a path: with a bound on
    every command the last one had none. Each of the 101 has its error."""
    two = ("diff --git a/src/secrets.py b/src/secrets.py\n--- a/src/secrets.py\n+++ b/src/secrets.py\n@@ -1 +1,2 @@\n"
           " y = 0\n+x = 1\ndiff --git a/docs/a.md b/docs/a.md\n--- a/docs/a.md\n+++ b/docs/a.md\n@@ -1 +1,2 @@\n y\n+x\n")
    for body in ("9 files changed. " * 100 + "Only touches docs/.", "All tests pass. " * 100 + "9 files changed. Only touches docs/."):
        _, out, _ = gate(act, tmp_path, monkeypatch, capsys, body, diff=two, soft="false")
        named = [ln for ln in lines(out) if ln.startswith("::error title=styxx diffgate - contradicted claim::")]
        assert named[-1].startswith("::error title=styxx diffgate - contradicted claim::Only touches docs/. - "
                                    "paths outside 'docs': ['src/secrets.py']")
        assert len(named) == (101 if body.startswith("9") else 2)


def test_a_job_fails_on_every_claim_whether_or_not_it_has_a_warning(act, tmp_path, monkeypatch, capsys):
    """Whether the job fails is counted over every claim, and so is the soft-fail line."""
    claims = [claim("UNCHECKABLE", f"U{i:04d} text", "r") for i in range(250)]
    assert run_claims(act, tmp_path, monkeypatch, capsys, claims, strict="false", soft="false")[0] == 0
    assert run_claims(act, tmp_path, monkeypatch, capsys, claims, strict="true", soft="false")[0] == 1
    rc, got = run_claims(act, tmp_path, monkeypatch, capsys, claims + [claim("CONTRADICTED", "C text", "r")])
    assert rc == 0 and got[-1].startswith("::warning::styxx diffgate found failures but soft-fail is on")
    assert got[-2] == "styxx diffgate: FAIL - 251 claim(s), 1 contradicted"


def test_a_filler_in_a_claimed_path_is_an_escape_in_the_reason_too(act, tmp_path, monkeypatch, capsys):
    """`repr`, through which a reason quotes a path, leaves the four fillers as they are (they are
    letters to it), so for these the reason carried the character too, in the command and in the row."""
    text = "Modified src/not\u3164here.py. Modified lib/a\u115f\u1160\uffa0b.py."
    for strict, soft, opens in (("false", None, "::warning title=styxx diffgate - uncheckable::"),
                                ("true", "false", "::error title=styxx diffgate - uncheckable under strict::")):
        _, out, summary = gate(act, tmp_path, monkeypatch, capsys, text, strict=strict, soft=soft)
        assert holds_no_unseen(out) and holds_no_unseen(summary)
        named = [ln for ln in lines(out) if ln.startswith(opens)]
        assert len(named) == 2
        assert " - 'src/not\\u3164here.py' does not appear in the diff" in named[0]
        assert " - 'lib/a\\u115f\\u1160\\uffa0b.py' does not appear in the diff" in named[1]
        assert "| `'src/not\\u3164here.py' does not appear in the diff" in rows(summary)[0]


@pytest.mark.parametrize("verdict, strict, opens", [
    ("CONTRADICTED", "false", "::error title=styxx diffgate - contradicted claim::"),
    ("UNCHECKABLE", "true", "::error title=styxx diffgate - uncheckable under strict::"),
    ("UNCHECKABLE", "false", "::warning title=styxx diffgate - uncheckable::"),
])
def test_a_claim_is_cut_at_120_characters_before_its_escapes_are_written_in_each_command(
        act, tmp_path, monkeypatch, capsys, verdict, strict, opens):
    """The gate replaced by one claim of 118 letters and then ten overrides: the command holds 120
    characters of it, the two overrides among them written out whole, and no part of a third."""
    one = types.SimpleNamespace(verdict=verdict, kind="file_touched", text="t" * 118 + "\u202e" * 10, why="why")
    result = types.SimpleNamespace(measured=True, why_unmeasured="", verdict="FAIL", claims=[one])
    monkeypatch.setattr(act, "gate_diff_text", lambda *a, **k: result)
    _, out, _ = gate(act, tmp_path, monkeypatch, capsys, "anything", strict=strict, soft="false")
    assert lines(out)[0] == opens + "t" * 118 + "\\u202e\\u202e - why"


def test_a_reason_is_cut_at_100_characters_before_its_escapes_are_written_in_its_cell(act, tmp_path, monkeypatch,
                                                                                    capsys):
    """A reason of 98 letters and then ten fillers: the cell holds 100 characters of it, the two fillers
    among them written out whole; the command holds the reason whole."""
    one = types.SimpleNamespace(verdict="UNCHECKABLE", kind="file_touched", text="claim", why="r" * 98 + "\u3164" * 10)
    result = types.SimpleNamespace(measured=True, why_unmeasured="", verdict="PASS", claims=[one])
    monkeypatch.setattr(act, "gate_diff_text", lambda *a, **k: result)
    _, out, summary = gate(act, tmp_path, monkeypatch, capsys, "anything")
    assert rows(summary) == ["| ❓ UNCHECKABLE | `claim` | `" + "r" * 98 + "\\u3164\\u3164` |"]
    assert lines(out)[0] == "::warning title=styxx diffgate - uncheckable::claim - " + "r" * 98 + "\\u3164" * 10


# -- the bytes of the step summary ---------------------------------------------------------------------

def test_the_summary_is_written_in_utf_8_with_lf_alone_and_appended_on_every_system(act, tmp_path, monkeypatch):
    """`_write_summary` appends its lines to the file GITHUB_STEP_SUMMARY names in UTF-8, each ended by
    LF and by no carriage return, on every system: text mode on Windows would end each in CR LF. What
    the file held before is kept."""
    path = tmp_path / "summary.md"
    path.write_bytes(b"before\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    act._write_summary(["## a — b", "", "| ✅ VERIFIED | `x` | `y` |"])
    written = path.read_bytes()
    assert b"\r" not in written
    assert written == b"before\n" + "## a — b\n\n| ✅ VERIFIED | `x` | `y` |\n".encode("utf-8")


def test_the_summary_is_opened_with_newline_lf_so_linux_sees_a_reversion_too(act, tmp_path, monkeypatch):
    """On Linux text mode writes LF whatever `newline` is, so the bytes alone cannot show the argument
    gone there; the call itself is checked."""
    path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    calls = []
    real_open = open

    def recording_open(file, *args, **kwargs):
        calls.append((str(file), args, kwargs))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(act, "open", recording_open, raising=False)
    act._write_summary(["x"])
    assert [(a, k.get("newline"), k.get("encoding")) for f, a, k in calls if f == str(path)] == [
        (("a",), "\n", "utf-8")]
