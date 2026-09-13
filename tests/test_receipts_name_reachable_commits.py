# -*- coding: utf-8 -*-
"""Every committed sworn receipt names a commit this repository has.

Found 2026-09-13: the checksum RESULT's receipt, as committed at c0bba384, named c1751053 — a commit
that existed only on the machine that built the series. For that span of history the document was
unchallengeable by BOUNTY.md's own "a document at a commit it does not name" clause, and
`python -m styxx.challenge` could not re-derive it. This test makes that a CI failure.

Known exception, with its reason: the sworn Action samples under papers/sworn/ were minted by the
Action inside its own checkout of a pull-request merge ref, a commit that exists only on the runner.
They are samples of the Action's output, not receipts anyone is asked to re-derive.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KNOWN_UNREACHABLE_PREFIX = "papers/sworn/sworn_action_sample"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _receipts() -> list[str]:
    out = _git("ls-files", "*.sworn-receipt.json").stdout.split()
    return sorted(out)


@pytest.mark.skipif(_git("rev-parse", "--is-shallow-repository").stdout.strip() == "true",
                    reason="a shallow clone lacks the history this test reads; run with full history")
def test_every_committed_receipt_names_a_commit_this_repository_has():
    receipts = _receipts()
    assert receipts, "no receipts tracked?"
    missing = []
    for rel in receipts:
        if rel.startswith(KNOWN_UNREACHABLE_PREFIX):
            continue
        r = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        commit = r.get("commit")
        if not commit:
            missing.append((rel, "names no commit"))
            continue
        if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            missing.append((rel, f"names {commit[:12]}, which this repository does not have"))
    assert not missing, "\n".join(f"{p}: {why}" for p, why in missing)


def test_the_known_exception_is_the_action_samples_only():
    # if the samples ever get re-minted at a real commit, delete the prefix and this test
    excepted = [r for r in _receipts() if r.startswith(KNOWN_UNREACHABLE_PREFIX)]
    assert excepted, "the exception list names files that no longer exist; remove it"
    assert all(Path(r).name.startswith("sworn_action_sample") for r in excepted)
