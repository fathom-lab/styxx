"""The suite's own preconditions, asserted rather than assumed.

A skip is green. That makes the skip count the easiest place in a test suite for a guarantee to
stop running without anyone noticing, and on 2026-08-28 two of ours had done exactly that:

* the LEDGER regeneration guarantee, and
* seven of the eight cases in `tests/test_silent_pass_bench.py` — the benchmark this repository
  maintains **specifically to detect outcomes that do not happen while every check stays green**.

Both skipped on `.git/shallow`, which is present on every `actions/checkout` default. The fix is
in `tests/conftest.py`; these tests exist so it cannot quietly come undone.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def test_the_session_has_full_git_history(full_git_history):
    """In CI this must hold, because history-dependent guarantees silently skip without it."""
    if not _in_ci():
        if not full_git_history:
            pytest.skip("local shallow clone; the CI arm of this assertion is the load-bearing one")
        assert full_git_history is True
        return
    assert full_git_history is True, (
        "the checkout is shallow and could not be unshallowed. Every history-dependent test in "
        "this suite will now SKIP, and a skip is green — which is the exact defect "
        "benchmarks/silent_pass exists to catch, occurring in the suite that tests it.")


def test_the_silent_pass_benchmark_is_not_skipping_wholesale():
    """The benchmark must actually execute, not report health by not running.

    Runs the benchmark file as its own pytest session and asserts that it is not skipped in bulk.
    Before 2026-08-28 this would have failed in CI: seven of its cases skipped on a shallow clone
    and the suite went green anyway.
    """
    r = subprocess.run(
        [os.sys.executable, "-m", "pytest", "-q", "--no-header",
         str(ROOT / "tests" / "test_silent_pass_bench.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    tail = (r.stdout or "")[-400:]
    assert r.returncode == 0, tail
    # Extract "N passed" / "M skipped" from the summary line.
    import re
    passed = int((re.search(r"(\d+) passed", r.stdout) or [0, 0])[1]) if "passed" in r.stdout else 0
    skipped = int((re.search(r"(\d+) skipped", r.stdout) or [0, 0])[1]) if "skipped" in r.stdout \
        else 0
    assert passed > 0, f"the silent-pass benchmark ran nothing:\n{tail}"
    assert passed >= skipped, (
        f"the silent-pass benchmark skipped {skipped} of {passed + skipped} cases. A benchmark "
        f"that mostly does not run is reporting its own health by absence, which is the thing it "
        f"measures.\n{tail}")


def test_a_skip_is_never_evidence_of_health():
    """Documentation as a test, because this is the lesson and it cost us two guarantees.

    Kept executable so it appears in the suite rather than only in a comment nobody re-reads.
    """
    import re
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    flat = re.sub(r"[\s*]+", " ", conftest)      # the phrase is bold and wraps across lines
    assert "silent-pass benchmark was itself a silent pass" in flat, (
        "the conftest must keep the reason it exists, or the next person deletes the fixture as "
        "unexplained overhead")
