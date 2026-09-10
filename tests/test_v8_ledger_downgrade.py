"""The retirement ledger cannot be moved back outside the digest by the reader's own kindness.

THE ATTACK, reported by a seventh adversarial pass and reproduced here. Moving the ledger inside
`set_sha256` was this round's repair: delete a retirement row and the set's identity moves. But the
reader kept a compatibility path for sets written before the move, falling back to
`provenance.retired`. So an attacker moved the ledger back under `provenance`, recomputed
`set_sha256` over the now-smaller core, and got a self-consistent index whose history sat outside
the digest again. The compatibility path undid the repair.

THE REPAIR. `read_ledger` now refuses a ledger it finds only under `provenance.retired`, and
refuses an index carrying one under both keys. It costs nothing: at the moment the branch was
closed, the only v8 set in this tree carried `retired` at the top level and no
`provenance.retired`, so the fallback had no honest consumer left. A genuinely old set restored
from git is migrated deliberately instead of accepted quietly, which is the right way round.

WHAT IT DOES NOT CLOSE, tested here as well so the limit is not left to prose: an attacker who
edits the ledger IN PLACE under `retired` and recomputes `set_sha256` leaves a self-consistent
index, and nothing inside the file catches it. What catches that is the `set_sha256` pinned in
`mutation_coverage.json` and the previous bytes in git, both outside the artifact a stranger
receives.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "v8"
GEN = SET / "gen_vectors.py"


def _index() -> dict:
    return json.loads((SET / "index.json").read_text(encoding="utf-8"))


def _copy(tmp_path: pathlib.Path) -> pathlib.Path:
    dst = tmp_path / "v8"
    shutil.copytree(SET, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _run(directory: pathlib.Path):
    return subprocess.run(
        [sys.executable, str(GEN), "--dir", str(directory)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
    )


def test_the_committed_set_carries_the_ledger_inside_the_digest() -> None:
    """The premise of every test below. If this fails, the repair was reverted."""
    index = _index()
    assert "retired" in index, (
        "conformance/v8/index.json carries no top-level `retired`. The ledger was moved inside "
        "set_sha256 deliberately; if it moved back out, that is the finding."
    )
    assert "retired" not in (index.get("provenance") or {}), (
        "the ledger is present under provenance as well, which is the state read_ledger refuses"
    )


@pytest.mark.slow
def test_the_downgrade_is_refused(tmp_path: pathlib.Path) -> None:
    """Move the ledger back under provenance. The reader must refuse, not carry it forward."""
    dst = _copy(tmp_path)
    index = json.loads((dst / "index.json").read_text(encoding="utf-8"))
    ledger = index.pop("retired")
    index.setdefault("provenance", {})["retired"] = ledger
    (dst / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True),
                                    encoding="utf-8", newline="\n")

    result = _run(dst)
    out = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, (
        "the generator accepted a set whose ledger sits outside set_sha256. Before this repair it "
        "did exactly that, silently, and wrote the history forward from the undigested key.\n" + out[-2000:]
    )
    assert "provenance.retired" in out, (
        "it refused, but the message does not name the key, so an operator meeting this cannot "
        "tell a downgrade from a corrupted index.\n" + out[-2000:]
    )


@pytest.mark.slow
def test_a_ledger_under_both_keys_is_refused(tmp_path: pathlib.Path) -> None:
    """Two ledgers, one digested and one not, can disagree. A reader cannot tell which is real."""
    dst = _copy(tmp_path)
    index = json.loads((dst / "index.json").read_text(encoding="utf-8"))
    index.setdefault("provenance", {})["retired"] = {"with_reason": [], "input_churn": []}
    (dst / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True),
                                    encoding="utf-8", newline="\n")

    result = _run(dst)
    out = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, "an index carrying two ledgers was accepted\n" + out[-2000:]
    assert "BOTH" in out or "both" in out, out[-2000:]


def test_the_limit_is_stated_where_someone_will_read_it() -> None:
    """An in-place edit is not catchable from inside the file, and the source must say so."""
    source = GEN.read_text(encoding="utf-8")
    assert "recomputes `set_sha256`" in source or "recompute `set_sha256`" in source, (
        "gen_vectors.py no longer states that an attacker who edits the ledger in place and "
        "recomputes the digest leaves a self-consistent index. That limit is the honest half of "
        "this repair and removing it would leave a reader believing the digest is a defence it "
        "is not."
    )
    assert "mutation_coverage.json" in source and "git" in source, (
        "the source no longer names what actually detects an in-place edit"
    )
