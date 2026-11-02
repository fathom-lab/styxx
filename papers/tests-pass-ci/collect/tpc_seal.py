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
    prereg_path: papers/tests-pass-ci/...   optional; if given, it must be tpc_const.PREREG_REPO_PATH

The guard opens only when all of these hold (P21), each refused by name otherwise:

 1. SEALED names a 40-hex prereg_commit, a readable freeze_anchor that is not in the future, and no
    prereg_path other than this PREREG's;
 2. `git fetch origin` in the lab's checkout succeeds (a failed fetch keeps the seal shut: origin/main
    would then be the checkout's own ref, which anyone on the box can move), and the commit is an
    ancestor of origin/main: a commit on a side branch or in a worktree that shares the object store
    does not open the seal;
 3. the PREREG exists at that commit (read with `git show`, nothing checked out), begins with
    "# PREREG", every line of it that begins with "**Signature.**" begins "**Signature.** Signed" (and
    there is one), and no lock-table row (a line beginning "|") still reads "TO-FILL at the signing
    commit"; prose that quotes these markers mid-line is not read;
 4. each lock-table row filled at the signing commit (SIGNING_ROWS: the collector's code digest and
    P2's decoy set digest) is there exactly once and holds exactly one sha256: a reworded, "pending"
    or deleted row keeps the seal shut;
 5. the collector row's sha256 equals the combined digest of the code that is running
    (`code_digests()["combined"]`).

The SEALED file sits in a folder agents can write to; what opens the seal is the signed, filled PREREG
at a commit on the fetched origin/main and the running code's digest, none of which a written file
can supply. It records the blob's sha256, the fetch outcome, both code digests and P2's decoy set
digest, and keeps the blob's text (`signed_prereg`), so the packet build and the scorer read the lock
table the guard read, not a working copy.
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
COLLECTOR_ROW_PREFIX = "| the collector: "
COLLECTOR_ROW = re.compile(r"^\| the collector: [^\n]*$", re.MULTILINE)
P2_SET_ROW_PREFIX = "| decoy set digest of packet P2 "
# the lock-table rows filled at the signing commit: each must be there once and hold exactly one sha256
SIGNING_ROWS = (COLLECTOR_ROW_PREFIX, P2_SET_ROW_PREFIX)


def lock_row_digests(text: str, prefix: str) -> Optional[list]:
    """The sha256 digests on the one lock-table row that begins with `prefix`; None unless exactly one row does."""
    rows = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    return HEX64.findall(rows[0]) if len(rows) == 1 else None


def signing_row_problems(text: str) -> list:
    """Why the rows filled at the signing commit are not filled: a row missing or doubled, or not holding exactly one
    sha256 (a reworded "TO-FILL at signing", a "pending", a deleted row)."""
    out = []
    for prefix in SIGNING_ROWS:
        found = lock_row_digests(text, prefix)
        if found is None:
            n = sum(1 for ln in text.splitlines() if ln.startswith(prefix))
            out.append(f"{n} lock rows begin {prefix.strip()!r}, not one")
        elif len(found) != 1:
            out.append(f"the lock row {prefix.strip()!r} holds {len(found)} sha256 digests, not one")
    return out


def p2_set_digest(text: str) -> Optional[str]:
    """The sha256 P2's decoy set row holds, or None unless there is one such row holding exactly one."""
    found = lock_row_digests(text, P2_SET_ROW_PREFIX)
    return found[0] if found is not None and len(found) == 1 else None


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
        self._text: Optional[str] = None
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
        if path != PREREG_REPO_PATH:
            st["why"] = f"SEALED names prereg_path {path!r}, not this PREREG's {PREREG_REPO_PATH}"
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
        if st["fetch"] != "ok":
            st["why"] = f"git fetch origin did not succeed in {self.repo} ({st['fetch']}): origin/main is not read"
            return st
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
        unfilled = signing_row_problems(text)
        if unfilled:
            st["why"] = "the lock rows filled at the signing commit are not filled: " + "; ".join(unfilled)
            return st
        sealed_code = collector_row_digests(text)
        running = self._code()
        st["collector_sealed"] = sealed_code
        st["collector_running"] = running
        st["p2_decoy_set_sealed"] = p2_set_digest(text)
        if len(sealed_code) != 1:
            st["why"] = f"the lock table's collector row holds {len(sealed_code)} sha256 digests, not one"
            return st
        if sealed_code[0] != running:
            st["why"] = f"the running collector {running[:12]}... is not the sealed one {sealed_code[0][:12]}..."
            return st
        self._text = text
        st["open"] = True
        st["why"] = "sealed"
        return st

    @property
    def open(self) -> bool:
        return bool(self.state.get("open"))

    def signed_prereg(self, working_file: Optional[Path] = None) -> str:
        """The PREREG text the open guard read at the signing commit, for the lock table's rows (P2's decoy set digest
        above all). With `working_file`, refused unless that file's bytes are the signed blob's: a checkout whose
        PREREG differs from the signed one must not make the build or the scorer accept another set."""
        if not self.open or self._text is None:
            raise SealRefused(f"SEAL_GUARD: the signed PREREG is read only through an open seal: {self.state.get('why')}")
        if working_file is not None:
            got = hashlib.sha256(Path(working_file).read_bytes()).hexdigest()
            if got != self.state["prereg_blob_sha256"]:
                raise SealRefused(f"SEAL_GUARD: {working_file} ({got[:12]}...) is not the signed PREREG blob "
                                  f"({self.state['prereg_blob_sha256'][:12]}...)")
        return self._text

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
