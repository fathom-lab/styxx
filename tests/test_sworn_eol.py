"""Byte-pinned artifacts must never be EOL-translated, and the rule must be enforced.

A sworn sidecar records span offsets into the EXACT bytes of its document, and
tests/test_sworn_dogfood.py asserts render(sidecar) == document.read_bytes(). Line-ending
translation changes those bytes and shifts every offset after the first newline, so the
round-trip that is the format's whole guarantee breaks — on Windows only, silently, while
Linux CI stays green.

That is not hypothetical. On 2026-09-01 a Windows verification of this branch failed three
tests that pass on CI: both dogfood round-trips and the h_mapping sha256 pin. The object
database held LF; core.autocrlf handed the worktree CRLF. .gitattributes already carries the
same lesson for styxx/centroids/*.json, in the mirror direction — the pin there was once a
CRLF-rendered hash that the LF Linux checkout could not verify.

So the entries exist. This file makes them impossible to forget: the dogfood test discovers
sworn documents DYNAMICALLY, so a sworn document added tomorrow without a .gitattributes
entry would be byte-pinned and EOL-translated at the same time, and nobody would learn until
someone ran the suite on Windows.

LOAD-BEARING: test_every_sworn_artifact_is_eol_pinned.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def check_attr(path: Path) -> str:
    """The `text` attribute git resolves for a path. `-text` reports as 'unset'."""
    r = subprocess.run(["git", "check-attr", "text", "--", str(path.relative_to(ROOT))],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # "path: text: unset" | "set" | "unspecified"
    return r.stdout.strip().rsplit(":", 1)[-1].strip() if r.returncode == 0 else "error"


def sworn_artifacts():
    """Every sworn sidecar, the document it pins, and its receipt."""
    out = []
    for side in sorted(ROOT.rglob("*.sworn.json")):
        if ".git" in side.parts:
            continue
        stem = str(side)[: -len(".sworn.json")]
        out.append(side)
        for suffix in (".md", ".sworn-receipt.json"):
            p = Path(stem + suffix)
            if p.exists():
                out.append(p)
    return out


def test_there_is_at_least_one_sworn_artifact():
    assert sworn_artifacts(), "no sworn artifacts found — has the discovery glob drifted?"


@pytest.mark.parametrize("path", sworn_artifacts(), ids=lambda p: p.name)
def test_every_sworn_artifact_is_eol_pinned(path):
    """LOAD-BEARING. A byte-pinned file that git is allowed to translate is a test that
    passes on one platform and fails on another, for reasons unrelated to the code."""
    attr = check_attr(path)
    assert attr == "unset", (
        f"{path.relative_to(ROOT)} is byte-pinned by a sworn sidecar but git reports "
        f"text={attr!r}. Add a '-text' entry for it in .gitattributes, then delete the "
        f"working copy and `git checkout HEAD -- <path>` so it is rewritten with the "
        f"stored bytes. Without this the sidecar round-trip fails on a CRLF checkout.")


@pytest.mark.skipif(sys.platform != "win32", reason="only a CRLF checkout can show this")
@pytest.mark.parametrize("path", sworn_artifacts(), ids=lambda p: p.name)
def test_the_working_copy_actually_has_no_carriage_returns(path):
    """The attribute is the rule; this is the observation. Both, because a correct
    .gitattributes on a worktree checked out before it was added still yields CRLF."""
    assert b"\r" not in path.read_bytes(), (
        f"{path.relative_to(ROOT)} contains CR bytes despite its -text attribute — this "
        f"worktree was checked out before the attribute existed. Delete the file and "
        f"`git checkout HEAD -- <path>` to restore the stored bytes.")
