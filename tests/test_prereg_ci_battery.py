# -*- coding: utf-8 -*-
"""prereg-ci B8: the mutation battery's misses, each pinned by a case that failed before its fix.

The B8 red team ran 139 mutation cases against prereg-ci at 9b32b76d; 16 that the tool must refuse
passed with exit 0, and the fresh red team found two majors. Each class is pinned here in the shape the
battery built it, on repositories made with ``git init`` in tmp_path (no remote, no network):

* M1: a gates fence a renderer shows and the scanner does not count (a block quote, a list item's
  continuation 4 columns in, a second word in the info string) beside the counted one, or alone;
* M2: a pull request that deletes the seals manifest, and one that re-adds it with a line restated
  after a merged deletion;
* M3: a receipt kept under a suffix other than ``.json``;
* M4: a PREREG kept under a suffix other than ``.md`` that a receipt names (C0 and the tree's licences);
* M5: a locked scorer that reads the clock (printed as the boundary, not refused: KNOWN_MISSES);
* F1: a locks PREREG that arrives in the pull request, whose scorer is the pull request's own code;
* F2: author bytes that would open a line of the printout with a workflow command.

``prereg-ci/KNOWN_MISSES.md`` names what v0 does not refuse and why; the cases at the end of this file
pin those boundaries, so a later hardening is seen as a change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import core  # noqa: E402
from tests.test_prereg_ci_cli import F, SENTENCE, TABLE, prereg, result_md, run, scored  # noqa: E402
from tests.test_prereg_ci_git import Git  # noqa: E402

HOLD = {"acc": 0.95, "ece": 0.08}
SHIP = {"acc": 0.95, "ece": 0.03}
CFG = 'surfaces = ["README*.md"]\n'
README_PLAIN = "# demo\n\nA digits model.\n"


def block(table=None) -> str:
    return json.dumps(table or TABLE, indent=1)


def moved(bar: float) -> dict:
    t = json.loads(json.dumps(TABLE))
    t["gates"]["G_ece"]["value"] = bar
    return t


def quoted(text: str) -> str:
    return "\n".join("> " + ln for ln in text.split("\n"))


def fence(table=None, info="gates") -> str:
    return f"{F}{info}\n{block(table)}\n{F}"


def sealed(tmp_path, name: str, text: str, readme: str = README_PLAIN) -> Git:
    g = Git(tmp_path / name)
    g.write("PREREG_digits.md", text)
    g.write("README_demo.md", readme)
    g.write("prereg-ci.toml", CFG)
    g.commit("seal")
    return g


# ---- M1: a gates fence the scanner does not count, beside the counted one or alone ------------------------

SHADOWS = {
    # the reader's table in a block quote, the machine's (a looser bar) in a collapsed <details> (B11q)
    "block-quote-beside-details": "# PREREG digits\n\nThe decision:\n\n" + quoted(fence()) + "\n\n"
                                  "<details><summary>appendix</summary>\n\n" + fence(moved(0.5)) + "\n\n</details>\n",
    # both visible, the reader's quoted (B11q2)
    "block-quote-beside-plain": "# PREREG digits\n\n" + quoted(fence()) + "\n\n" + fence(moved(0.5)) + "\n",
    # a fence 4 columns in under an ordered list item, rendered in the item (B11n)
    "ordered-item-4": "# PREREG digits\n\n1.  the decision:\n\n" + "\n".join("    " + ln for ln in fence().split("\n"))
                      + "\n\n" + fence(moved(0.5)) + "\n",
    # a second word in the info string, which CommonMark renders as the language gates (B11t)
    "second-word": "# PREREG digits\n\n" + fence(info="gates strict") + "\n\n" + fence(moved(0.5)) + "\n",
    # the one table, blockquoted: a reader sees a decision, the scanner none (B11z)
    "only-block-quote": "# PREREG digits\n\n" + quoted(fence()) + "\n",
}


@pytest.mark.parametrize("name", sorted(SHADOWS))
def test_a_gates_fence_a_renderer_shows_and_the_scanner_does_not_count_is_named_beside_the_scanners_reading(name):
    """``_decision`` keeps the scanner's reading (the conformance set pins it: vectors_compare.json's hidden
    fences read NOT_GATED); ``_shadowed`` names the fence the reader is shown, which check refuses."""
    why = core._shadowed(SHADOWS[name])
    assert why is not None and ("shadowing channel" in why or "inside a container" in why), why
    assert core._decision(SHADOWS[name])["code"] in (None, "NOT_GATED")


def test_a_plain_fence_alone_and_prose_quoting_another_language_name_no_shadow():
    text = "# PREREG digits\n\n> a note, quoted\n>\n> " + f"{F}json\n> {{}}\n> {F}" + "\n\n" + fence() + "\n"
    assert core._decision(text)["form"] == "gates" and core._shadowed(text) is None
    assert core._shadowed("# PREREG digits\n\n> quoted prose, no fence\n") is None
    assert core._shadowed(prereg()) is None and core._shadowed(b"\xff\xfe") is None


def test_a_blockquoted_only_table_does_not_read_as_not_gated_in_check(tmp_path, capsys):
    """B11z: with the one table blockquoted the PREREG read NOT_GATED, which is never refused, and a
    RESULT typing SHIP over HOLD numbers passed."""
    g = sealed(tmp_path, "z", SHADOWS["only-block-quote"])
    g("switch", "-q", "-c", "run")
    g.write("receipt.json", json.dumps({**scored(HOLD), "verdict": "SHIP"}, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md("SHIP"))
    g.commit("run")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1, out
    assert "FAIL FENCE_SHADOWED" in out and "renders as a gates block" in out, out


def test_a_blockquoted_decoy_added_to_a_sealed_prereg_is_decision_edited_after_seal(tmp_path, capsys):
    """B11a: the sealed fence untouched, a pull request adds a blockquoted table with another bar; C0 read
    it as prose (PROSE_AMENDED_IN_PLACE, a NOTE) and the check passed."""
    g = sealed(tmp_path, "a", prereg())
    g("switch", "-q", "-c", "decoy")
    g.write("PREREG_digits.md", prereg() + "\nThe bar, as a reader is shown it:\n\n" + quoted(fence(moved(0.01))) + "\n")
    g.commit("a decoy table")
    code, out = run(g, "check", "--base", "main", "--head", "decoy", capsys=capsys)
    assert code == 1, out
    assert "FAIL DECISION_EDITED_AFTER_SEAL" in out and "holds a second table beside it" in out, out


@pytest.mark.parametrize("name", ["block-quote-beside-details", "second-word"])
def test_a_shadowed_table_is_refused_where_the_receipt_is_checked(tmp_path, capsys, name):
    """B11q and B11t: the check printed PASS and the ladder of the looser table."""
    g = sealed(tmp_path, name, SHADOWS[name])
    g("switch", "-q", "-c", "run")
    rec = scored(HOLD, table=moved(0.5))
    g.write("receipt.json", json.dumps(rec, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md(rec["verdict"]))
    g.commit("run")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1 and "FAIL FENCE_SHADOWED" in out, out


def test_lint_refuses_a_draft_holding_a_shadowed_table(tmp_path, capsys):
    g = Git(tmp_path / "lint")
    g.write("README_demo.md", f"The model is {SENTENCE}.\n")
    g.write("prereg-ci.toml", CFG)
    g.commit("surfaces")
    g.write("PREREG_digits.md", SHADOWS["block-quote-beside-plain"])
    code, out = run(g, "lint", "PREREG_digits.md", capsys=capsys)
    assert code == 1 and "FAIL FENCE_SHADOWED" in out, out
