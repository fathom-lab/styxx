"""Session setup for the suite.

## Why this file exists: our silent-pass benchmark was a silent pass

`actions/checkout` defaults to `fetch-depth: 1`, so CI checkouts are shallow and `.git/shallow`
exists. A number of tests need real history — `git log -S`, `--follow`, `merge-base`, and the
pre-fix trees the silent-pass corpus is scored against — and every one of them handled that by
calling `pytest.skip`.

Measured on 2026-08-28 by comparing two CI runs on this branch:

    before   2504 passed, 30 skipped
    after    2518 passed, 22 skipped

Six of those fourteen new passes are tests added that day. **The other eight were tests that had
been skipping in CI and now run.** One was the LEDGER regeneration guarantee. The other seven are
in `tests/test_silent_pass_bench.py`, which carries eight shallow-clone skip sites.

That benchmark is this repository's instrument for detecting outcomes that do not happen while
every check stays green. It had never run in CI. **The silent-pass benchmark was itself a silent
pass**, and nothing in the suite said so, because a skip is green.

## Why it is a fixture rather than a side effect

The repair first landed inside `tests/test_ledger.py`, which unshallows the checkout in order to
run its own assertion. That worked for the benchmark only by accident: `test_ledger` sorts before
`test_silent_pass_bench`, so the history happened to be there by the time the benchmark looked.
Deselect one file, reorder the suite, or run the benchmark alone, and seven tests go back to
skipping silently.

A guarantee that depends on collection order is not a guarantee. So the unshallow happens once
here, before any test runs, and every history-dependent test in the suite gets the same ground.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_UNSHALLOW_TIMEOUT_S = 300


def _in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


@pytest.fixture(scope="session", autouse=True)
def full_git_history():
    """Unshallow the checkout once, before anything reads git history.

    Yields True if the session has full history. Deliberately does NOT fail on its own: a test
    that needs history decides for itself what an absent history means, and `test_ledger.py`
    turns it into a CI failure rather than a skip. This fixture only removes the excuse.
    """
    shallow = ROOT / ".git" / "shallow"
    if not shallow.exists():
        yield True
        return
    try:
        subprocess.run(["git", "fetch", "--unshallow", "--quiet"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=_UNSHALLOW_TIMEOUT_S)
    except Exception:
        pass
    ok = not shallow.exists()
    if not ok and _in_ci():
        # Not a failure here — it would abort the whole session. Individual tests that need
        # history report it, and this makes sure the reason is visible in the log rather than
        # inferred from a skip count.
        print("\nWARNING: checkout is shallow and could not be unshallowed. "
              "History-dependent tests will report it; a skip count is not evidence of health.")
    yield ok
