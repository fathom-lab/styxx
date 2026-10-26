# -*- coding: utf-8 -*-
"""prereg-ci B9: the README and the credit note say what the code and the receipts say.

``prereg-ci/README.md`` names a file in this repository for every number it prints. This test reads
each number beside the receipt the README names for it, so a receipt that moves, or a README that
drifts from it, fails here: the R1, RV2, RV3 and RV4 copies under ``conformance/prereg_ci/``, the
#224 replay pinned in ``tests/test_prereg_ci_demo_224.py``, the window in ``styxx/prereg_ci/core.py``.
A numeral the README prints that is in none of these is refused unless it is a name (an exit code, a
year, a section, a pull request). It also holds the README's "report-only by default" to the
Action's default, and the credit to the five neighbours the design names.
"""
from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import core  # noqa: E402
from tests.test_prereg_ci_package import HOUSE_RX, PRIORITY_RX  # noqa: E402

README = ROOT / "prereg-ci" / "README.md"
CREDITS = ROOT / "prereg-ci" / "CREDITS.md"
ACTION = ROOT / "prereg-ci" / "action.yml"
C = ROOT / "conformance" / "prereg_ci"
NEIGHBOURS = ("reproducible-science", "Elliot Tower", "Panossian", "honest-signal", "Alessandro Cardurani",
              "preregister", "Nick Harris", "scienceverse", "Lakens", "DeBruine")


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def _flat(p: Path) -> str:
    return " ".join(_text(p).split())


# each number the README prints, the sentence it sits in, and the receipt that holds it
RECEIPTS = [
    ("33 of 33 records naming a gates digest", C / "rescore_census_ca347730.txt", 'TALLY {"RESCORES": 33} records 33'),
    ("orders 32 of 32 distinct PREREG and receipt pairs strictly", C / "landing_order_ca347730.txt",
     "distinct pairs 32; landed-gates commit G strictly before receipt R: 32"),
    ("ordering it by the whole file orders 31", C / "landing_order_ca347730.txt",
     "landed-bytes commit W strictly before R: 31"),
    ("the licence matches 0 shipped surfaces", C / "licence_replay_ca347730.txt", "files with Q 0, K 4, H 4"),
    ("The quote's last five words match 4 of the 4 surfaces a hand rule finds on the RESULT's day",
     C / "licence_replay_ca347730.txt", "== rev 414db45a: surfaces read 328; files with Q 0, K 4, H 4"),
    ("and 3 of the 10 it finds just before #221", C / "licence_replay_ca347730.txt",
     "== rev 3437340b^1: surfaces read 394; files with Q 0, K 3, H 10"),
    ("with precision 0.23", C / "external1_rescore_ca347730.txt", "precision 0.23"),
    ("`FAILURE__gate_claim_not_yet_true`", C / "external1_rescore_ca347730.txt", "verdict FAILURE__gate_claim_not_yet_true"),
    ("against the table's bar of 0.95", C / "reconstructed" / "external1_reconstructed.md",
     '"G_E1": {"metric": "precision", "op": ">=", "value": 0.95'),
    ("`G_B_long 0.1097 >= 0.15 (margin -0.0403)`", ROOT / "tests" / "test_prereg_ci_demo_224.py",
     "G_B_long 0.1097 >= 0.15 (margin -0.0403)"),
    ("gates digest `578924f8...`", ROOT / "tests" / "test_prereg_ci_demo_224.py", 'SEALED_SHA = "578924f8'),
    ("under gates `e513a1da...`", ROOT / "tests" / "test_prereg_ci_demo_224.py", 'MOVED_SHA = "e513a1da'),
    ("`G_B_long` moved to 0.10 in the working copy only", ROOT / "tests" / "test_prereg_ci_demo_224.py",
     'BAR[:-4] + "0.10"'),
]


def test_every_number_the_readme_prints_stands_in_the_receipt_it_names():
    readme = _flat(README)
    for sentence, receipt, held in RECEIPTS:
        assert sentence in readme, sentence
        assert receipt.relative_to(ROOT).as_posix() in readme, receipt
        assert held in _text(receipt), (receipt, held)


def test_the_seven_surfaces_the_five_words_miss_were_added_sixteen_days_after_the_result():
    replay = _text(C / "licence_replay_ca347730.txt")
    later = [ln for ln in replay.splitlines() if ln.startswith("  H -- added 2026-09-16 ")]
    assert len(later) == 7
    assert "the other 7 were added 16 days after the RESULT" in _flat(README)
    result = ROOT / "papers" / "closed-model-frontier" / "RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md"
    assert result.is_file()
    assert (datetime.date(2026, 9, 16) - datetime.date(2026, 8, 31)).days == 16


def test_the_window_the_readme_prints_is_the_codes():
    assert core._WINDOW == 300 and "a withdrawal within 300 characters" in _flat(README)
    assert "`_WINDOW` in `styxx/prereg_ci/core.py`" in _flat(README)


def test_no_other_numeral_stands_in_the_readme():
    """A numeral outside RECEIPTS is a name: an exit code, a year, a section, a list index, RFC 3161."""
    named = {"0", "1", "2", "3", "4", "5", "2.1", "2021", "2026", "3161", "7", "10", "16", "31", "32", "33",
             "300", "0.1097", "0.15", "0.10", "0.23", "0.95"}
    numerals = re.findall(r"(?<![\w.#-])\d+(?:\.\d+)*(?!\w)", _text(README))
    assert set(numerals) <= named, sorted(set(numerals) - named)


def test_report_only_by_default_is_the_actions_default():
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert action["inputs"]["report-only"]["default"] == "true"
    assert "**Report-only by default.**" in _text(README)


def test_the_readme_names_the_known_misses_the_credit_and_the_boundary():
    readme = _flat(README)
    for needle in ("KNOWN_MISSES.md", "CREDITS.md", "What a PASS does not say", "not yet tested against their code",
                   "reconstructed on 2026-10-24, after the result", "not a catch", "No value is measured yet",
                   "DECISION_EDITED_AFTER_SEAL", "VERDICT_DIFFERS", "DECISION_DIGEST_UNSEALED", "LICENCE_WITHDRAWN",
                   "RECEIPT_REWRITTEN", "examples/prereg-ci-demo/README.md"):
        assert needle in readme, needle
    assert (ROOT / "prereg-ci" / "KNOWN_MISSES.md").is_file()
    for name in NEIGHBOURS:
        assert name in readme and name in _flat(CREDITS), name


def test_the_word_law_over_the_readme_and_the_credit():
    for p in (README, CREDITS):
        text = _text(p)
        assert not HOUSE_RX.search(text) and not PRIORITY_RX.search(text), p
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
