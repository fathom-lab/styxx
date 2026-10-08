# -*- coding: utf-8 -*-
"""The leaderboard `styxx leaderboard` prints says what Baseline-019 measured and claims no priority.

papers/NOTE_prior_art_credit_2026_09_29.md section 3 withdrew an unpriced ordinal priority claim
from styxx/critique.py's docstring, which now says the detector passed the styxx gauntlet's v3
detection bars. The same claim stayed in the Baseline-019 row of LEADERBOARD.md (the packaged copy
is what the CLI prints and the wheel carries), in two comments in styxx/__init__.py, and in the
same kind of claim on the Baseline-008, Baseline-014 and Baseline-016 rows. These tests hold the
withdrawal.

The ordinal is stored reversed, as scripts/zenodo_deposit_software_v7_49_0.py stores the charter's
words, so this file does not spell it either.
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from styxx.cli import cmd_leaderboard

ROOT = Path(__file__).resolve().parent.parent
ROOT_COPY = ROOT / "LEADERBOARD.md"
PKG_COPY = ROOT / "styxx" / "_data" / "LEADERBOARD.md"

_ORDINAL = "tsrif"[::-1]
_ORDINAL_RX = re.compile(r"(?<![a-z0-9])" + _ORDINAL + r"(?![a-z0-9])", re.IGNORECASE)


def _printed() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_leaderboard(SimpleNamespace(rows_only=False))
    assert rc == 0, f"cmd_leaderboard exited {rc}; expected 0"
    return buf.getvalue()


def _row(text: str, name: str) -> str:
    rows = [ln for ln in text.splitlines() if f"**{name}**" in ln]
    assert len(rows) == 1, f"expected one {name} row, found {len(rows)}"
    return rows[0]


def test_the_root_and_packaged_leaderboards_are_the_same_bytes():
    # The CLI prints the packaged copy; the repository links the root one. A withdrawal made in one
    # and not the other would leave the claim in whichever copy a reader opens. A submission PR
    # that edits only the root file, as "How to submit" step 3 used to read, fails here.
    assert ROOT_COPY.read_bytes() == PKG_COPY.read_bytes(), (
        "LEADERBOARD.md and styxx/_data/LEADERBOARD.md differ; "
        "run: cp LEADERBOARD.md styxx/_data/LEADERBOARD.md")


def test_the_submission_instructions_say_to_copy_the_root_file_over_the_packaged_one():
    text = ROOT_COPY.read_text(encoding="utf-8")
    assert ("3. **Open a PR** adding one row to the table below, then copy `LEADERBOARD.md` over "
            "`styxx/_data/LEADERBOARD.md`") in text


def test_baseline_019_row_states_the_measured_pass_without_an_ordinal():
    row = _row(_printed(), "Baseline-019")
    assert "**passed the v3 bars, 4/4**" in row
    assert "It passed the styxx gauntlet's v3 detection bars, 4/4" in row
    assert "**🎯 4 / 4 — PASS**" in row
    assert not _ORDINAL_RX.search(row), row


def test_baseline_014_and_016_rows_claim_no_ordinal_achievement():
    text = _printed()
    row_014 = _row(text, "Baseline-014")
    row_016 = _row(text, "Baseline-016")
    for name, row in (("Baseline-014", row_014), ("Baseline-016", row_016)):
        assert not _ORDINAL_RX.search(row), f"{name}: {row}"
        assert "EVER" not in row, name
    # what they measured stays
    assert "**D2 partition CLEARS both v3 confound bars**" in row_014
    assert "**But this is an n=1 demonstration that signal genuinely orthogonal" in row_014
    assert "(1) D2 length-delta crosses 0.10 (0.103); (2) BOTH D4 deltas above 0.10" in row_016


def test_baseline_008_row_states_the_bars_without_an_ordinal():
    # The heading and the notes set the row apart from Baseline-007's length artifact with an
    # achievement ordinal; both now say which bars scored it.
    row = _row(_printed(), "Baseline-008")
    heading = row.split("|")[1]
    assert "*(embedding similarity — tested under v2 bars; pre-stated prediction validated)*" in heading
    assert not _ORDINAL_RX.search(row), row
    assert "| Detection submission tested under v2 bars (D1+D2+D3). Pre-stated" in row
    assert "D1=0.805 ✓ / D2=0.928 ✓ / D1-len=0.015 ✗ / D1-cap=0.102 ✓" in row


def test_the_init_comments_on_the_critique_detector_claim_no_ordinal():
    src = (ROOT / "styxx" / "__init__.py").read_text(encoding="utf-8")
    hits = [ln for ln in src.splitlines()
            if "#" in ln and "critique" in ln.lower() and _ORDINAL_RX.search(ln.split("#", 1)[1])]
    assert hits == []
    assert src.count("passed the gauntlet's v3 bars") == 2
