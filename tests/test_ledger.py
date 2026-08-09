"""THE LEDGER must always equal what the committed receipts regenerate.

It is the one claim document with no OATH certificate — its numbers are counts of the receipts
themselves, so there is nothing beneath them to bind to. Regeneration is the guarantee instead:
edit it by hand, or let it drift from the corpus, and this fails.
"""
import subprocess
import sys
from pathlib import Path


def test_ledger_matches_a_fresh_regeneration_from_the_receipts(tmp_path):
    root = Path(__file__).resolve().parent.parent
    ledger = root / "papers" / "LEDGER.md"
    if not ledger.exists():
        import pytest
        pytest.skip("LEDGER.md not present")
    if (root / ".git" / "shallow").exists():
        # build_ledger's power-basis split needs full git history (git log -S,
        # --follow, merge-base); a shallow clone regenerates a shorter ledger
        # and this test would fail on the missing rows, not on drift. CI's
        # actions/checkout defaults to depth 1 — the real fix is fetch-depth: 0
        # in test.yml, which is staged in the local workflow commit.
        import pytest
        pytest.skip("shallow clone — ledger regeneration needs full git history")
    committed = ledger.read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, str(root / "papers" / "build_ledger.py")],
                       capture_output=True, text=True, cwd=str(root))
    assert r.returncode == 0, r.stderr[-500:]
    regenerated = ledger.read_text(encoding="utf-8")
    if committed != regenerated:
        ledger.write_text(committed, encoding="utf-8")     # leave the tree as we found it
        raise AssertionError(
            "papers/LEDGER.md does not match what papers/build_ledger.py produces from the "
            "committed receipts. Either the corpus changed and the ledger was not rebuilt, or "
            "the ledger was edited by hand. Run: python papers/build_ledger.py")
