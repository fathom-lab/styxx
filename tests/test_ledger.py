"""THE LEDGER must always equal what the committed receipts regenerate.

It is the one claim document with no OATH certificate — its numbers are counts of the receipts
themselves, so there is nothing beneath them to bind to. Regeneration is the guarantee instead:
edit it by hand, or let it drift from the corpus, and this fails.

## The guarantee had never actually run in CI

`papers/build_ledger.py` needs real history — `git log -S`, `--follow`, `merge-base` — to compute
the power-basis split. `actions/checkout` defaults to `fetch-depth: 1`, which leaves
`.git/shallow` present, and this test used to call `pytest.skip` on it. So on every Python version
in the matrix the regeneration guarantee `papers/LEDGER.md` advertises was **silently not run**,
and the check went green anyway.

An absent measurement surfacing as a passing check is the defect class this repository exists to
document — `benchmarks/silent_pass/CORPUS.md` catalogues it as SP-1. It had been sitting in our
own suite.

The obvious fix is `fetch-depth: 0` in `test.yml`. **This file does not rely on that**, for two
reasons. A workflow file can be changed back by anyone, and a contributor working from a shallow
clone deserves the same guarantee CI gets. So the test now repairs its own precondition and, if it
cannot, **fails in CI instead of skipping**. Locally it still skips, because a developer with a
shallow clone is not the person hiding a defect.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_UNSHALLOW_TIMEOUT_S = 300


def _in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def has_full_history(root: Path) -> bool:
    """True if the repo has full history, unshallowing it first if that is possible.

    Returns the state AFTER the attempt, so a caller can branch on the fact rather than on
    whether a fetch happened to be needed.
    """
    shallow = root / ".git" / "shallow"
    if not shallow.exists():
        return True
    try:
        subprocess.run(["git", "fetch", "--unshallow", "--quiet"], cwd=str(root),
                       capture_output=True, text=True, timeout=_UNSHALLOW_TIMEOUT_S)
    except Exception:
        pass
    return not shallow.exists()


def test_ledger_matches_a_fresh_regeneration_from_the_receipts():
    root = Path(__file__).resolve().parent.parent
    ledger = root / "papers" / "LEDGER.md"
    if not ledger.exists():
        pytest.skip("LEDGER.md not present")

    if not has_full_history(root):
        # Deliberately asymmetric. In CI this is the silent-pass defect and must be loud; on a
        # developer's shallow clone it is an inconvenience and must not be.
        if _in_ci():
            raise AssertionError(
                "the ledger regeneration guarantee cannot run: this checkout is shallow and "
                "`git fetch --unshallow` did not resolve it. build_ledger.py needs real history "
                "(git log -S, --follow, merge-base). Set `fetch-depth: 0` on actions/checkout. "
                "This FAILS rather than skips because a guarantee that silently does not run is "
                "the defect this repository exists to document.")
        pytest.skip("shallow clone and could not unshallow — regeneration needs full history")

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


def test_full_history_check_is_a_noop_on_a_normal_checkout():
    """On a non-shallow repo the helper must not shell out at all."""
    root = Path(__file__).resolve().parent.parent
    if (root / ".git" / "shallow").exists():
        pytest.skip("this checkout is shallow; the no-op path is not the one under test")
    assert has_full_history(root) is True


def test_a_shallow_checkout_without_a_remote_is_reported_not_hidden(tmp_path):
    """The failure mode that matters: shallow, unshallow impossible, so the answer must be False.

    Built as a real shallow clone with no reachable remote, so `git fetch --unshallow` genuinely
    cannot succeed. If this ever returns True the guard is vacuous and CI would go back to
    skipping the guarantee without saying so.
    """
    src = tmp_path / "src"
    src.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}

    def git(*args, cwd):
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                              env=env, timeout=60)

    git("init", "-q", cwd=src)
    for i in range(3):
        (src / f"f{i}.txt").write_text(str(i), encoding="utf-8")
        git("add", "-A", cwd=src)
        git("commit", "-qm", f"c{i}", cwd=src)

    clone = tmp_path / "clone"
    r = git("clone", "-q", "--depth", "1", src.as_uri(), str(clone), cwd=tmp_path)
    if r.returncode != 0 or not (clone / ".git" / "shallow").exists():
        pytest.skip("could not construct a shallow clone in this environment")

    git("remote", "remove", "origin", cwd=clone)          # now unshallowing cannot succeed
    assert has_full_history(clone) is False
    assert (clone / ".git" / "shallow").exists(), "the helper must not leave the repo altered"
