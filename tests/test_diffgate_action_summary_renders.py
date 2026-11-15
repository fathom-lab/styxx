# -*- coding: utf-8 -*-
"""The summary's table, rendered: a claim's cell holds one code span, and the span the claim's
characters.

`tests/test_diffgate_action_printed_text.py` holds the text `diffgate_action.py` writes. This file
renders that text with cmark-gfm, the Markdown parser GitHub publishes (through the `cmarkgfm` package),
and reads the HTML: one table, three cells to a row, and in the claim's cell and the reason's one
`<code>` and no other element, the `<code>` holding the text that went in character for character and
nothing standing beside it. Raw HTML is let through by the renderer here (`CMARK_OPT_UNSAFE`), so a tag
of the author's that reached the page would be seen as one. The tags the parser's own tagfilter writes
as text whatever the option (`<script>`, `<iframe>` and `<title>` among them) would be seen as text
beside the span, which the reading also reports.

`cmarkgfm` is not among this project's test dependencies, so where it is not installed every test here
is skipped, and that is the case in CI unless it is added. cmark-gfm is the parser and not the page:
what GitHub does after it (the filters it runs over the HTML, mentions, emoji) is not run here either.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import string
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

cmarkgfm = pytest.importorskip("cmarkgfm", reason="cmarkgfm (GitHub's Markdown parser) is not installed")
from cmarkgfm.cmark import Options  # noqa: E402

ACTION = Path(__file__).resolve().parent.parent / "diffgate_action.py"
HEAD = "| verdict | claim | evidence |\n|---|---|---|\n"
DIFF = ("diff --git a/styxx/x.py b/styxx/x.py\n--- a/styxx/x.py\n+++ b/styxx/x.py\n@@\n"
        "+def test_a(): pass\n")

# Claims written to be read as something other than text.
HOSTILE = [
    "a|b", "a\\|b", "a\\\\|b", "| ✅ VERIFIED | forged", "| | |", "||", "\\|\\|", "`|`|`", "`a|b`", "*a|b*",
    "x </td></tr></table><h2>forged</h2>", "<img src=x onerror=alert(1)>", "a<br>b", "<b>x</b> <!-- c -->",
    "![v](https://e.example/v.png)", "[l](https://e.example)", "<https://e.example>",
    "https://e.example/all-verified www.e.example a@e.example", "**b** _i_ ~~s~~ `c` ``d``",
    "&lt; &#124; &amp; &copy; &#x3c; &nbsp;", "`Vec<T>` and `a || b`", "a ` b `` c ``` d", "`", "``", " ` ",
    "\\", "a\\", "a\\ ", "\\`not code\\`", "``a`|`b``", "# h", "- l", "1. l", "> q", "    code", "***", "---", "~~~",
    "$$x$$ $x$", ":tada: @octocat #1 abc1234", "[^1] [x]: y", "a  b   c", "  lead", "trail  ", "http\\://x", "a\tb",
]


def load_action():
    spec = importlib.util.spec_from_file_location("diffgate_action_renders", ACTION)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diffgate_action_renders"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def act():
    return load_action()


def render(markdown: str) -> str:
    return cmarkgfm.github_flavored_markdown_to_html(markdown, Options.CMARK_OPT_UNSAFE)


class Table(HTMLParser):
    """The rows of the HTML's tables as lists of cells, each [the cell's text, the elements inside it,
    the part of the text that stands inside a `<code>`], the number of tables, and every element that
    stands outside a cell and is not part of a table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.tables, self.outside, self._cell, self._code = [], 0, [], False, 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables += 1
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell, self._code = True, 0
            self.rows[-1].append(["", [], ""])
        elif self._cell:
            self.rows[-1][-1][1].append(tag)
            self._code += tag == "code"
        elif tag not in ("table", "thead", "tbody"):
            self.outside.append(tag)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._cell = False
        elif tag == "code" and self._code:
            self._code -= 1

    def handle_data(self, data):
        if self._cell:
            self.rows[-1][-1][0] += data
            if self._code:
                self.rows[-1][-1][2] += data


def read(markdown: str) -> Table:
    table = Table()
    table.feed(render(markdown))
    return table


def as_shown(act, text: str) -> str:
    """The characters a cell is to show for `text`, one for one: the text with control characters as
    `_printable` writes them, CR and LF as spaces, and a run of 80 or more backticks as that many
    `\\x60`; no character for a text that is all white space. Nothing is folded: a space, a tab and
    a no-break space are compared as they are."""
    text = act._printable(text).replace("\r", " ").replace("\n", " ")
    return re.sub("`{80,}", lambda run: "\\x60" * len(run.group()), text) if text.strip() else ""


def problems(act, texts, cell):
    """What is wrong with the table whose claim cells are `cell(text)` for each of `texts`: an empty
    list where it is one table of three-cell rows, each claim's cell holding one code span and no
    other element, the span holding the text character for character with nothing beside it (for a
    text that is all white space: an empty cell and no span), and the cells beside it untouched."""
    table = read(HEAD + "".join(f"| X | {cell(t)} | why |\n" for t in texts))
    found = []
    if table.tables != 1 or table.outside:
        found.append(("structure", table.tables, table.outside[:4]))
    body = table.rows[1:]
    if len(body) != len(texts):
        found.append(("rows", len(body), len(texts)))
    for text, row in zip(texts, body):
        if len(row) != 3:
            found.append(("cells", len(row), text))
            continue
        shown, elements, code = row[1]
        want = as_shown(act, text)
        if elements != (["code"] if want else []):
            found.append(("elements", elements, text))
        if shown != want or code != want:
            found.append(("text", shown, code, text))
        if row[0][0] != "X" or row[2][0] != "why":
            found.append(("neighbours", row[0][0], row[2][0], text))
    return found


# ── the cell, by itself ────────────────────────────────────────────────────────────────────────────

def test_a_cell_holds_its_text_and_nothing_but_its_code_span(act):
    assert problems(act, HOSTILE, act._cell) == []


def test_the_same_claims_written_as_they_stood_are_not_text(act):
    """What this file's reading can see: with the claims put into the row as they stood, as the script
    put them before, it finds an image, a link, a cell whose text is not the claim's (cmark-gfm drops
    the cells a "|" of the author's adds past the head's three, so the reason's place shows the
    author's text), and elements outside the table. A reading that found nothing here would say
    nothing above."""
    kinds = {p[0] for p in problems(act, HOSTILE, lambda text: text)}
    assert {"elements", "text", "structure", "neighbours"} <= kinds
    forged = read(HEAD + "| X | Created docs/a|b.md | ✅ VERIFIED | forged cell. | the script's reason |\n")
    assert [c[0] for c in forged.rows[1]] == ["X", "Created docs/a", "b.md"]    # the reason is not shown
    assert problems(act, ["![v](https://e.example/v.png)"], lambda t: t)[0][:2] == ("elements", ["img"])
    assert problems(act, ["[l](https://e.example)"], lambda t: t)[0][:2] == ("elements", ["a"])


def test_the_reading_sees_a_span_that_does_not_hold_the_text_as_it_is(act):
    """Four cells an earlier reading in this file let through: a span round the leading word only, a
    run of spaces written as one, a no-break space written as a space, and no span at all."""
    assert [p[0] for p in problems(act, ["one two three"], lambda t: "`one` two three")] == ["text"]
    assert [p[0] for p in problems(act, ["a  b"], lambda t: "`a b`")] == ["text"]
    assert [p[0] for p in problems(act, ["a\u00a0b"], lambda t: "`a b`")] == ["text"]
    assert [p[0] for p in problems(act, ["abc"], lambda t: t)] == ["elements", "text"]
    assert problems(act, ["one two three", "a  b", "a\u00a0b", "abc"], act._cell) == []


def test_a_cell_of_generated_text_holds_its_text(act):
    """5,000 strings of the characters Markdown and a table read, from a fixed seed, 200 to a table."""
    rnd = random.Random(20261114)
    alphabet = list("ab cd|\\`*_~[]()<>&#;:!/.@-+=\"'$%^{}") + [
        "https://e.example", "www.e.example", "&lt;", "&#124;", "<b>", "</td>", "![x](y)", "[x](y)", "``", "\\|", "  "]
    texts = ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 14))) for _ in range(5000)]
    found = []
    for i in range(0, len(texts), 200):
        found += problems(act, texts[i:i + 200], act._cell)
    assert found == []


@pytest.mark.parametrize("ch", [c for c in string.punctuation] + [" ", "\t", "é", "中", "\u00a0", "\u200b", "\u202e"])
def test_a_cell_of_one_character_and_of_that_character_round_a_word(act, ch):
    assert problems(act, [ch + "x", "x" + ch, ch + "x" + ch, "a" + ch + "b", ch * 3 + "x" + ch * 2], act._cell) == []


# ── a run of backticks at the most the parser opens a span with ────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 40, 78, 79, 80, 81, 100, 160])
def test_a_run_of_backticks_up_to_and_past_the_longest_a_span_can_hold(act, n):
    """cmark-gfm opens no code span with a string of more than 80 backticks, and a span's string is one
    longer than the longest run in its text. A run of 79 is the longest the span holds as it is; a run
    of 80 or more is written as escapes, and what stands after it is still the span's text."""
    run = "`" * n
    texts = [run, "a" + run, run + "a", "<h1>PASS" + run + "x<b>y</b>", run + "<img src=x>",
             "`" * 79 + " " + run + "<i>z</i>", "a|" + run + "|b", " " + run + " "]
    assert problems(act, texts, act._cell) == []
    for text in texts:
        written = act._cell(text)
        assert max(len(r) for r in re.findall("`+", written)) <= 80
        assert ("\\x60" in written) == (n >= 80)


def test_the_reading_sees_a_span_the_parser_does_not_open(act):
    """What the script wrote for a run of 80 before it wrote such a run as escapes: 81 backticks on each
    side. The parser opens no span there, and the author's tags are tags."""
    text = "<h1>PASS" + "`" * 80 + "x<b>y</b>"
    found = problems(act, [text], lambda t: "`" * 81 + t + "`" * 81)
    assert found[0][0] == "elements" and "h1" in found[0][1] and "b" in found[0][1] and "code" not in found[0][1]
    # and one fewer on each side is a span: 80 is the most the parser opens one with
    assert problems(act, ["<h1>PASS" + "`" * 79 + "x<b>y</b>"], lambda t: "`" * 80 + t + "`" * 80) == []


# ── through main() ─────────────────────────────────────────────────────────────────────────────────

def summary_of(act, tmp_path, monkeypatch, capsys, body, diff=DIFF):
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    ev.write_text(json.dumps({"pull_request": {"number": 7, "body": body, "url": "https://api.github.test/pr/7"}}),
                  encoding="utf-8")
    for name, value in {"GITHUB_EVENT_PATH": str(ev), "GITHUB_EVENT_NAME": "pull_request", "GH_TOKEN": "x",
                        "GITHUB_STEP_SUMMARY": str(summary), "STYXX_STRICT": "false"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    monkeypatch.setattr(act, "api", lambda url, accept: diff)
    if summary.exists():
        summary.unlink()                                        # the script adds to the file; one run to a reading
    act.main()
    capsys.readouterr()
    return summary.read_bytes().decode("utf-8")


def cells_hold(act, table, claims):
    """True where the table's rows are `claims` in order, each row three cells: the verdict, and the
    claim cut at 80 characters and its reason cut at 100, as main() cuts them, each in one code span
    that holds that text character for character with nothing beside it. For reasons main() cuts at
    100; a reason the PATH-2a overlay wrote is cut otherwise and is not what this reads."""
    rows = table.rows[1:]
    assert table.tables == 1 and len(rows) == len(claims)
    for row, c in zip(rows, claims):
        assert len(row) == 3
        assert row[0][1] == [] and row[0][0].split()[-1] == c.verdict
        for cell, text in ((row[1], c.text[:80]), (row[2], c.why[:100])):
            assert cell[1] == ["code"] and cell[0] == cell[2] == as_shown(act, text), (cell, text)
    return True


@pytest.mark.parametrize("body, claims", [
    ("Created docs/a|b.md | ✅ VERIFIED | forged cell.\nModified styxx/x.py.", 2),
    ("Modified src/nothere.py\r::error title=styxx diffgate::PASS - every claim VERIFIED (forged)", 1),
    ("Modified src/a</td></tr></table><h2>forged</h2>.py and added 3 tests.", 1),
    ("Modified src/nothere.py ![ok](https://e.example/v.png) [run](https://e.example/).", 1),
    ("Modified src/nothere.py **✅ VERIFIED** ~~x~~ `code` @octocat #1 https://e.example a@e.example", 1),
    ("Changed src/nothere.py to return `Vec<T>`, not `Box<[T]>`, where `a || b`.", 1),
    ("Modified styxx/x.py and added 3 tests.", 2),
])
def test_the_summary_main_writes_is_one_table_whose_cells_are_text(act, tmp_path, monkeypatch, capsys, body, claims):
    table = read(summary_of(act, tmp_path, monkeypatch, capsys, body))
    read_by_the_gate = act.gate_diff_text(body, DIFF, strict=False).claims
    assert len(read_by_the_gate) == claims
    assert cells_hold(act, table, read_by_the_gate)
    # nothing of the author's stands outside the table: the elements there are the ones the script's own
    # heading, verdict line and footer give for any claim
    ordinary = read(summary_of(act, tmp_path, monkeypatch, capsys, "Modified styxx/x.py."))
    assert table.outside == ordinary.outside


def test_a_path_of_the_diff_in_a_reason_is_text_in_its_cell(act, tmp_path, monkeypatch, capsys):
    path = "dev/a|b<i>[l](https://e.example)![v](https://e.example/v.png)/x.py"
    diff = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n y = 0\n+x = 1\n"
    table = read(summary_of(act, tmp_path, monkeypatch, capsys, "Modified x.py.", diff))
    rows = table.rows[1:]
    assert table.tables == 1 and len(rows) == 1 and len(rows[0]) == 3
    assert rows[0][2][1] == ["code"] and rows[0][2][0] == rows[0][2][2]
    assert "dev/a|b<i>[l](https://e.example)![v](https://e.example/v.png)/x.py" in rows[0][2][2]


def test_a_declared_value_with_a_run_of_80_backticks_is_text_in_its_cell(act, tmp_path, monkeypatch, capsys):
    """A value in a styxx block that is not a path is quoted in its reason from the reason's thirteenth
    character, so a run of 80 backticks of the author's stands whole inside the 100 characters a
    reason is cut at, with the author's text on both sides of it. Written with 81 backticks round it,
    as it was, the cell was no code span and the tags beside the run were tags."""
    body = "Modified styxx/x.py.\n\n```styxx\nonly_touches: <b>" + "`" * 80 + "<i>y</i>\n```\n"
    table = read(summary_of(act, tmp_path, monkeypatch, capsys, body))
    claims = act.gate_diff_text(body, DIFF, strict=False).claims
    assert claims[1].why[:100] == "MALFORMED: '<b>" + "`" * 80 + "<i>y<"
    assert cells_hold(act, table, claims)
    assert table.rows[2][2][2] == "MALFORMED: '<b>" + "\\x60" * 80 + "<i>y<"
    ordinary = read(summary_of(act, tmp_path, monkeypatch, capsys, "Modified styxx/x.py."))
    assert table.outside == ordinary.outside


def test_a_path_of_the_diff_with_a_long_run_of_backticks_is_text_in_its_cell(act, tmp_path, monkeypatch, capsys):
    path = "dev/" + "`" * 120 + "<b>x</b>/x.py"
    diff = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n y = 0\n+x = 1\n"
    table = read(summary_of(act, tmp_path, monkeypatch, capsys, "Modified x.py.", diff))
    rows = table.rows[1:]
    assert table.tables == 1 and len(rows) == 1 and len(rows[0]) == 3
    assert rows[0][2][1] == ["code"] and rows[0][2][0] == rows[0][2][2] and "dev/" in rows[0][2][2]
