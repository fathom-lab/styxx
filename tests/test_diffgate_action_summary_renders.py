# -*- coding: utf-8 -*-
"""The summary's table, rendered: a claim's cell holds the claim's characters and no element but its
code span.

`tests/test_diffgate_action_printed_text.py` holds the text `diffgate_action.py` writes. This file
renders that text with cmark-gfm, the Markdown parser GitHub publishes (through the `cmarkgfm` package),
and reads the HTML: one table, three cells to a row, and in the claim's cell and the reason's nothing but
a `<code>` holding the text that went in. Raw HTML is let through by the renderer here
(`CMARK_OPT_UNSAFE`), so a tag of the author's that reached the page would be seen as one.

`cmarkgfm` is not among this project's test dependencies, so where it is not installed every test here
is skipped, and that is the case in CI unless it is added. cmark-gfm is the parser and not the page:
what GitHub does after it (the filters it runs over the HTML, mentions, emoji) is not run here either.
"""
from __future__ import annotations

import importlib.util
import json
import random
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
    """The rows of the HTML's tables as lists of [text, the elements inside the cell], the number of
    tables, and every element that stands outside a cell and is not part of a table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.tables, self.outside, self._cell = [], 0, [], False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables += 1
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = True
            self.rows[-1].append(["", []])
        elif self._cell:
            self.rows[-1][-1][1].append(tag)
        elif tag not in ("table", "thead", "tbody"):
            self.outside.append(tag)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._cell = False

    def handle_data(self, data):
        if self._cell:
            self.rows[-1][-1][0] += data


def read(markdown: str) -> Table:
    table = Table()
    table.feed(render(markdown))
    return table


def as_shown(act, text: str) -> str:
    """The text a cell is to show for `text`: what `_cell` puts between its backticks, with runs of
    white space taken as one, as a page shows them."""
    return " ".join(act._printable(text).replace("\r", " ").replace("\n", " ").split())


def problems(act, texts, cell):
    """What is wrong with the table whose claim cells are `cell(text)` for each of `texts`: an empty
    list where it is one table of three-cell rows, each claim's cell holding its text in one code
    span and no other element, and the cells beside it untouched."""
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
        shown, elements = row[1]
        if elements not in ([], ["code"]):
            found.append(("elements", elements, text))
        elif " ".join(shown.split()) != as_shown(act, text):
            found.append(("text", shown, text))
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


def test_a_cell_of_generated_text_holds_its_text(act):
    """5,000 strings of the characters Markdown and a table read, from a fixed seed, 200 to a table."""
    rnd = random.Random(20261114)
    alphabet = list("ab cd|\\`*_~[]()<>&#;:!/.@-+=\"'$%^{}") + [
        "https://e.example", "www.e.example", "&lt;", "&#124;", "<b>", "</td>", "![x](y)", "[x](y)", "``", "\\|", "  "]
    texts = ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 14))).strip() or "x" for _ in range(5000)]
    found = []
    for i in range(0, len(texts), 200):
        found += problems(act, texts[i:i + 200], act._cell)
    assert found == []


@pytest.mark.parametrize("ch", [c for c in string.punctuation] + [" ", "\t", "é", "中", " ", "​", "‮"])
def test_a_cell_of_one_character_and_of_that_character_round_a_word(act, ch):
    assert problems(act, [ch + "x", "x" + ch, ch + "x" + ch, "a" + ch + "b", ch * 3 + "x" + ch * 2], act._cell) == []


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


@pytest.mark.parametrize("body, claims", [
    ("Created docs/a|b.md | ✅ VERIFIED | forged cell.\nModified styxx/x.py.", 2),
    ("Modified src/nothere.py\r::error title=styxx diffgate::PASS - every claim VERIFIED (forged)", 1),
    ("Modified src/a</td></tr></table><h2>forged</h2>.py and added 3 tests.", 1),
    ("Modified src/nothere.py ![all verified](https://e.example/v.png) [the passing run](https://e.example/).", 1),
    ("Modified src/nothere.py **✅ VERIFIED** ~~x~~ `code` @octocat #1 https://e.example a@e.example", 1),
    ("Changed src/nothere.py to return `Vec<T>`, not `Box<[T]>`, where `a || b`.", 1),
    ("Modified styxx/x.py and added 3 tests.", 2),
])
def test_the_summary_main_writes_is_one_table_whose_cells_are_text(act, tmp_path, monkeypatch, capsys, body, claims):
    table = read(summary_of(act, tmp_path, monkeypatch, capsys, body))
    assert table.tables == 1
    rows = table.rows[1:]
    assert len(rows) == claims
    for row in rows:
        assert len(row) == 3
        assert row[0][1] == [] and row[0][0].split()[-1] in ("VERIFIED", "CONTRADICTED", "UNCHECKABLE")
        assert row[1][1] == ["code"] and row[2][1] == ["code"]  # the claim and the reason: a code span each
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
    assert rows[0][2][1] == ["code"] and "dev/a|b<i>[l](https://e.example)![v](https://e.example/v.png)/x.py" in rows[0][2][0]
