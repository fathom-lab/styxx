# -*- coding: utf-8 -*-
"""THE SEAL (PREREG section 0.1; P21): the guard every read of a body, CI record, log or artifact
passes through.

Before the PREREG is frozen no one reads CI conclusions, check runs, statuses or job logs of any
pull request opened on or after 2026-11-02, nor classifies a claim in one. The collector refuses,
by the name SEAL_GUARD, any body, check, status, Actions, log or artifact read for a pull request
created at or after WINDOW_START until a file SEALED exists that names the PREREG's signing commit
and every check below holds.

The SEALED file (written by the lead after the signing pull request opened, never by the collector):

    prereg_commit: <40 hex>                 required: the signing commit
    freeze_anchor: 2026-11-06T12:00:00Z     required: the server time the pull request opened
    prereg_path: papers/tests-pass-ci/...   optional; default tpc_const.PREREG_REPO_PATH

The guard opens only when all of these hold (P21), each refused by name otherwise:

 1. SEALED names a 40-hex prereg_commit and a readable freeze_anchor that is not in the future;
 2. after a `git fetch origin` in the lab's checkout (best effort; its outcome is recorded), the
    commit is an ancestor of origin/main: a commit on a side branch or in a worktree that shares
    the object store does not open the seal;
 3. the PREREG exists at that commit (read with `git show`, nothing checked out), begins with
    "# PREREG", every line of it that begins with "**Signature.**" begins "**Signature.** Signed" (and
    there is one), and no lock-table row (a line beginning "|") still reads "TO-FILL at the signing
    commit"; prose that quotes these markers mid-line is not read;
 4. the lock table's collector row at that commit holds exactly one sha256, and it equals the
    combined digest of the code that is running (`code_digests()["combined"]`).

It records the blob's sha256, the fetch outcome and both code digests, so every run log names the
sealed bytes and the code that read under them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from tpc_const import PREREG_REPO_PATH, SEALED_FILE, STYXX_REPO, UTC, WINDOW_START, iso, parse_utc

HERE = Path(__file__).resolve().parent


class SealRefused(Exception):
    """SEAL_GUARD: a sealed read was asked for and the seal is not open."""

    code = "SEAL_GUARD"


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
SIGNATURE_LINE = re.compile(r"^\*\*Signature\.\*\*.*$", re.MULTILINE)
SIGNED = "**Signature.** Signed"
UNSIGNED = "**Signature.** Unsigned"
TO_FILL_AT_SIGNING = "TO-FILL at the signing commit"


def signature_lines(text: str) -> list:
    """The lines that begin with "**Signature.**": the signature paragraph's opening line. Prose that quotes the
    markers mid-line (P21 does) is not read."""
    return SIGNATURE_LINE.findall(text)


def signed(text: str) -> bool:
    lines = signature_lines(text)
    return bool(lines) and all(ln.startswith(SIGNED) and not ln.startswith(UNSIGNED) for ln in lines)


def to_fill_rows(text: str) -> list:
    """Lock-table rows (lines that begin with "|") still marked TO-FILL at the signing commit."""
    return [ln for ln in text.splitlines() if ln.startswith("|") and TO_FILL_AT_SIGNING in ln]
# the lock table's row (section 14) begins "| the collector: "; section 0.2's "| the collector's window dry run" row,
# which also carries a sha256, does not match
COLLECTOR_ROW = re.compile(r"^\| the collector: [^\n]*$", re.MULTILINE)


def code_digests(root: Path = HERE) -> dict:
    """sha256 of every file of the collector's code: the top-level *.py files and the vendored
    testreceipt files (not tests, checks, README or caches), and a combined digest over the sorted
    lines "<relative path> <sha256>\\n". The combined digest is what the lock table's collector
    row holds at the signing commit (PREREG section 14)."""
    files = sorted(list(root.glob("*.py")) + list((root / "vendor" / "testreceipt").glob("*")))
    out = {}
    for f in files:
        if f.is_file() and f.suffix in (".py", "") and "__pycache__" not in f.parts:
            out[f.relative_to(root).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    combined = hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(out.items())).encode("utf-8")).hexdigest()
    return {"combined": combined, "files": out}


def parse_sealed(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _git(repo: Path, args: list, timeout: int = 60):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", "-C", str(repo)] + args, capture_output=True, timeout=timeout, env=env)


def git_blob(repo: Path, commit: str, path: str) -> Optional[bytes]:
    try:
        r = _git(repo, ["show", f"{commit}:{path}"])
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def git_fetch(repo: Path) -> str:
    """`git fetch origin`, read-only; returns "ok" or why not. A failed fetch does not open the seal:
    the ancestry check below then reads the origin/main the checkout already has."""
    try:
        r = _git(repo, ["fetch", "--quiet", "origin"], timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"fetch failed: {type(e).__name__}"
    return "ok" if r.returncode == 0 else f"fetch failed: exit {r.returncode}"


def git_on_main(repo: Path, commit: str) -> bool:
    try:
        r = _git(repo, ["merge-base", "--is-ancestor", commit, "origin/main"])
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def collector_row_digests(text: str) -> list:
    """The sha256 digests on the lock table's collector row; [] when there is not exactly one such row."""
    rows = COLLECTOR_ROW.findall(text)
    return HEX64.findall(rows[0]) if len(rows) == 1 else []


class SealGuard:
    def __init__(self, sealed_file: Path = SEALED_FILE, repo: Path = STYXX_REPO,
                 now: Callable[[], dt.datetime] = lambda: dt.datetime.now(UTC),
                 blob_reader: Callable[[Path, str, str], Optional[bytes]] = git_blob,
                 fetch: Callable[[Path], str] = git_fetch,
                 on_main: Callable[[Path, str], bool] = git_on_main,
                 running_code: Optional[Callable[[], str]] = None) -> None:
        self.sealed_file = Path(sealed_file)
        self.repo = Path(repo)
        self._now = now
        self._blob = blob_reader
        self._fetch = fetch
        self._on_main = on_main
        self._code = running_code or (lambda: code_digests()["combined"])
        self.state = self._evaluate()

    def _evaluate(self) -> dict:
        st: dict = {"open": False, "sealed_file": str(self.sealed_file)}
        if not self.sealed_file.exists():
            st["why"] = "no SEALED file: the PREREG is not frozen"
            return st
        fields = parse_sealed(self.sealed_file.read_bytes().decode("utf-8", "replace"))
        commit = (fields.get("prereg_commit") or "").lower()
        path = fields.get("prereg_path") or PREREG_REPO_PATH
        st.update({"prereg_commit": commit, "prereg_path": path})
        if not HEX40.match(commit):
            st["why"] = "SEALED names no 40-hex prereg_commit"
            return st
        anchor = fields.get("freeze_anchor")
        if not anchor:
            st["why"] = "SEALED names no freeze_anchor (the server time the signing pull request opened)"
            return st
        try:
            t = parse_utc(anchor)
        except ValueError:
            st["why"] = f"freeze_anchor unreadable: {anchor!r}"
            return st
        st["freeze_anchor"] = iso(t)
        if self._now() < t:
            st["why"] = f"before the freeze anchor {iso(t)}"
            return st
        st["fetch"] = self._fetch(self.repo)
        if not self._on_main(self.repo, commit):
            st["why"] = f"{commit} is not an ancestor of origin/main in {self.repo} ({st['fetch']})"
            return st
        blob = self._blob(self.repo, commit, path)
        if blob is None:
            st["why"] = f"{commit}:{path} not found in {self.repo}"
            return st
        st["prereg_blob_sha256"] = hashlib.sha256(blob).hexdigest()
        text = blob.decode("utf-8", "replace")
        if not text.startswith("# PREREG"):
            st["why"] = "the file at that commit is not a PREREG"
            return st
        if not signed(text):
            st["why"] = "the PREREG at that commit carries no positive '**Signature.** Signed' line"
            return st
        if to_fill_rows(text):
            st["why"] = (f"the PREREG at that commit still has {len(to_fill_rows(text))} lock rows marked TO-FILL at "
                         "the signing commit")
            return st
        sealed_code = collector_row_digests(text)
        running = self._code()
        st["collector_sealed"] = sealed_code
        st["collector_running"] = running
        if len(sealed_code) != 1:
            st["why"] = f"the lock table's collector row holds {len(sealed_code)} sha256 digests, not one"
            return st
        if sealed_code[0] != running:
            st["why"] = f"the running collector {running[:12]}... is not the sealed one {sealed_code[0][:12]}..."
            return st
        st["open"] = True
        st["why"] = "sealed"
        return st

    @property
    def open(self) -> bool:
        return bool(self.state.get("open"))

    @staticmethod
    def in_window(created_at: Optional[str]) -> bool:
        """Unknown creation times are treated as inside the window."""
        if not created_at:
            return True
        return parse_utc(created_at) >= WINDOW_START

    def check(self, created_at: Optional[str], what: str) -> None:
        """Raise SealRefused unless this pull request may have its body, CI or logs read."""
        if self.in_window(created_at) and not self.open:
            raise SealRefused(f"SEAL_GUARD: refused {what} for a pull request created {created_at}: "
                              f"{self.state.get('why')}")
