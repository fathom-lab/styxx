"""styxx.receipt_binding — where a certificate's receipt bytes are: in the working tree, and in git history.

SPEC: ``papers/closed-model-frontier/SPEC_oath_receipt_binding_2026_09_04.md`` (frozen before this
module was written). This is the one module that talks to git on the certificate's behalf (rule
R7). It has no dependency on ``styxx.certify``; ``certify_doc`` calls :func:`bind_at_mint` to fill
the certificate's ``receipt_binding`` block (R1), and ``styxx.corpus_audit`` calls
:func:`classify_certificate` and :func:`sworn_bytes_at_issue` to fill the five cells (R3) and
re-derive over the sworn bytes (R4). Every failure degrades to ``None`` / ``unrecoverable`` with a
reason; nothing here raises into a verdict.

Why it exists, in one sentence: an OATH certificate recorded each receipt's digest and then bound
by *basename*, so a receipt regenerated in place silently invalidated every certificate citing it
(three documents in two days, 2026-08-31 → 2026-09-01), and the audit could not tell *the receipt
moved* from *the certificate is wrong*.

Content identity is modulo newlines everywhere (``content_sha256``): the corpus's recorded digests
are CRLF hashes taken on Windows over LF blobs, and ``corpus_audit._receipt_sha_matches`` and
``charon._content_sha256`` already compare that way. The record always says which normalisation
matched, so the weakening is visible per citation.

Nothing in this module is a verdict. It never decides that a certificate is right or wrong; it
says where the bytes are.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

SCHEMA = "styxx-oath/receipt-binding/v1"
CONTENT_RULE = "sha256 over bytes with CRLF normalised to LF"
CELLS = ("same", "at_issue", "elsewhere", "unbacked", "unrecoverable")

__all__ = ["SCHEMA", "CONTENT_RULE", "CELLS", "content_sha256", "raw_sha256", "git_blob_id",
           "match_normalisation", "Repo", "RepoUnavailable", "bind_at_mint",
           "classify_certificate", "sworn_bytes_at_issue"]


# ---------------------------------------------------------------- digests

def raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def content_sha256(raw: bytes) -> str:
    """sha256 of *raw* with CRLF normalised to LF — the binding that survives a checkout."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def git_blob_id(raw: bytes) -> str:
    """The id git would give these exact bytes as a blob (sha1 over ``blob <len>\\0`` + bytes)."""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(raw))
    h.update(raw)
    return h.hexdigest()


def match_normalisation(raw: bytes, recorded: str) -> Optional[str]:
    """Which newline reading of *raw* hashes to *recorded*: ``raw``, ``lf``, ``crlf`` or ``None``.

    Same three readings as ``corpus_audit._receipt_sha_matches``; this returns the name instead of
    a bool so every record can say how it matched.
    """
    if hashlib.sha256(raw).hexdigest() == recorded:
        return "raw"
    lf = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(lf).hexdigest() == recorded:
        return "lf"
    if hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == recorded:
        return "crlf"
    return None


# ---------------------------------------------------------------- the repository

class RepoUnavailable(Exception):
    """git is absent, the path is not inside a repository, or the clone is shallow (R5)."""


class Repo:
    """A thin, subprocess-only view of one git repository. No GitPython, no dulwich (R7)."""

    def __init__(self, path: Path):
        self.git = shutil.which("git")
        if not self.git:
            raise RepoUnavailable("git not on PATH")
        start = path if path.is_dir() else path.parent
        try:
            top = self._run(["rev-parse", "--show-toplevel"], cwd=start)
        except RepoUnavailable:
            raise RepoUnavailable(f"{start} is not inside a git repository")
        self.top = Path(top.decode("utf-8", "replace").strip()).resolve()
        self._tree_cache: dict = {}
        self._catfile = None

    # -- plumbing
    def _run(self, args: list, cwd: Optional[Path] = None, ok_codes=(0,)) -> bytes:
        try:
            p = subprocess.run([self.git, *args], cwd=str(cwd or self.top),
                               capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RepoUnavailable(f"git {args[0]} failed: {e}")
        if p.returncode not in ok_codes:
            raise RepoUnavailable(f"git {' '.join(args)} exited {p.returncode}: "
                                  f"{p.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return p.stdout

    def rel(self, path: Path) -> str:
        return Path(os.path.relpath(Path(path).resolve(), self.top)).as_posix()

    @property
    def shallow(self) -> bool:
        return self._run(["rev-parse", "--is-shallow-repository"]).strip() == b"true"

    def head(self) -> Optional[str]:
        try:
            return self._run(["rev-parse", "--verify", "-q", "HEAD"]).decode().strip() or None
        except RepoUnavailable:
            return None

    def tree_blobs(self, commit: str, paths: list) -> dict:
        """``{relpath: blob}`` for the given paths at *commit*, one call. Missing paths are absent."""
        if not paths:
            return {}
        out = self._run(["ls-tree", "-z", commit, "--", *paths], ok_codes=(0, 128))
        found = {}
        for rec in out.split(b"\0"):
            if not rec:
                continue
            meta, _, name = rec.partition(b"\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == b"blob":
                found[name.decode("utf-8", "replace")] = parts[2].decode()
        return found

    def tree(self, commit: str) -> dict:
        """``{relpath: blob}`` for the whole tree at *commit* (cached: the corpus shares a few
        issuing commits, and a basename scan must not cost one process per candidate)."""
        if commit not in self._tree_cache:
            out = self._run(["ls-tree", "-r", "-z", commit])
            tree = {}
            for rec in out.split(b"\0"):
                if not rec:
                    continue
                meta, _, name = rec.partition(b"\t")
                parts = meta.split()
                if len(parts) == 3 and parts[1] == b"blob":
                    tree[name.decode("utf-8", "replace")] = parts[2].decode()
            self._tree_cache[commit] = tree
        return self._tree_cache[commit]

    def tree_paths(self, commit: str) -> list:
        return list(self.tree(commit))

    def blob_at(self, commit: str, relpath: str) -> Optional[str]:
        if commit in self._tree_cache:
            return self._tree_cache[commit].get(relpath)
        return self.tree_blobs(commit, [relpath]).get(relpath)

    def cat(self, blob: str) -> bytes:
        """Bytes of one blob, through a persistent ``cat-file --batch`` so a census is not one
        process per receipt."""
        if self._catfile is None or self._catfile.poll() is not None:
            self._catfile = subprocess.Popen([self.git, "cat-file", "--batch"], cwd=str(self.top),
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
        p = self._catfile
        p.stdin.write(blob.encode() + b"\n")
        p.stdin.flush()
        header = p.stdout.readline()
        if not header or header.endswith(b" missing\n"):
            raise RepoUnavailable(f"blob {blob} missing")
        size = int(header.split()[2])
        data = p.stdout.read(size)
        p.stdout.read(1)   # the trailing newline the batch protocol appends
        return data

    def close(self) -> None:
        if self._catfile is not None and self._catfile.poll() is None:
            try:
                self._catfile.stdin.close()
                self._catfile.wait(timeout=10)
            except Exception:
                self._catfile.kill()
        self._catfile = None

    def is_clean_tracked(self, relpath: str) -> bool:
        """Tracked and unmodified in the working tree — the precondition for an issuing commit."""
        st = self._run(["status", "--porcelain", "--untracked-files=all", "--", relpath])
        return st.strip() == b""

    def last_commit_touching(self, relpath: str) -> Optional[str]:
        out = self._run(["log", "-1", "--format=%H", "--", relpath]).decode().strip()
        return out or None

    def commits_touching_basename(self, name: str) -> list:
        """``[(commit, relpath), ...]`` newest first, for every commit in HEAD's history that
        touched any path whose basename is *name*. Basename search only: a renamed receipt is
        invisible here, which the SPEC records as owed."""
        out = self._run(["log", "--format=%x00%H", "--name-only", "--",
                         f":(glob)**/{name}", name], ok_codes=(0, 128))
        hits, commit = [], None
        for line in out.decode("utf-8", "replace").splitlines():
            if line.startswith("\0"):
                commit = line[1:].strip()
            elif line.strip() and commit and line.strip().rsplit("/", 1)[-1] == name:
                hits.append((commit, line.strip()))
        return hits

    def relation(self, commit: str, issuing: str) -> str:
        """``before`` if *commit* is an ancestor of *issuing*, ``after`` if the reverse, else
        ``unrelated``."""
        if commit == issuing:
            return "same_commit"
        if self._is_ancestor(commit, issuing):
            return "before"
        if self._is_ancestor(issuing, commit):
            return "after"
        return "unrelated"

    def _is_ancestor(self, a: str, b: str) -> bool:
        p = subprocess.run([self.git, "merge-base", "--is-ancestor", a, b], cwd=str(self.top),
                           capture_output=True)
        return p.returncode == 0


# ---------------------------------------------------------------- R1: at mint

def bind_at_mint(receipt_paths: list) -> dict:
    """The certificate's ``receipt_binding`` block. Never raises: without git, ``head`` is null,
    every ``blob`` is null and every ``committed`` is false — the certificate does not vouch for
    what it could not see."""
    block = {"schema": SCHEMA, "content_rule": CONTENT_RULE, "head": None,
             "all_receipts_committed": False, "receipts": []}
    paths = [Path(p) for p in receipt_paths]
    repo = None
    reason = None
    try:
        repo = Repo(paths[0].parent) if paths else None
    except RepoUnavailable as e:
        reason = str(e)
    head = repo.head() if repo else None
    head_blobs: dict = {}
    rels: dict = {}
    if repo and head:
        for p in paths:
            try:
                rels[p] = repo.rel(p)
            except ValueError:
                rels[p] = None
        try:
            head_blobs = repo.tree_blobs(head, [r for r in rels.values() if r])
        except RepoUnavailable as e:
            reason, head = str(e), None
    for p in paths:
        raw = p.read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        rec = {"name": p.name, "path": rels.get(p) if head else None,
               "raw_sha256": raw_sha256(raw), "content_sha256": content_sha256(raw),
               "blob": None, "committed": False}
        committed_blob = head_blobs.get(rels.get(p) or "")
        if committed_blob and committed_blob in {git_blob_id(raw), git_blob_id(lf),
                                                 git_blob_id(lf.replace(b"\n", b"\r\n"))}:
            rec["blob"] = committed_blob
            rec["committed"] = True
        block["receipts"].append(rec)
    block["head"] = head
    block["all_receipts_committed"] = bool(paths) and all(r["committed"] for r in block["receipts"])
    if reason:
        block["note"] = f"no repository at mint: {reason}"
    if repo:
        repo.close()
    return block


# ---------------------------------------------------------------- R3: the five cells

def _blob_matches(repo: Repo, blob: str, digest: str) -> Optional[str]:
    try:
        return match_normalisation(repo.cat(blob), digest)
    except RepoUnavailable:
        return None


def classify_citation(repo: Repo, issuing: Optional[str], name: str, digest: str,
                      working: Optional[Path]) -> dict:
    """One receipt citation → one cell (SPEC R3), with the evidence that decided it."""
    rec = {"name": name, "digest": digest, "cell": None, "normalisation": None,
           "path": None, "commit": None, "blob": None, "at_issue_too": None}
    if issuing is None:
        rec.update(cell="unrecoverable", reason="no issuing commit")
        return rec
    # candidates at the issuing commit: every path with this basename
    at_issue = []
    for p in repo.tree_paths(issuing):
        if p.rsplit("/", 1)[-1] == name:
            blob = repo.blob_at(issuing, p)
            norm = _blob_matches(repo, blob, digest) if blob else None
            if norm:
                at_issue.append((p, blob, norm))
    if working is not None and working.exists():
        norm = match_normalisation(working.read_bytes(), digest)
        if norm:
            rec.update(cell="same", normalisation=norm, path=str(working),
                       at_issue_too=bool(at_issue))
            if at_issue:
                # the working-tree reading and the committed blob's reading are recorded
                # separately: on a Windows checkout the first is `raw` and the second `crlf`
                rec.update(commit=issuing, blob=at_issue[0][1], blob_path=at_issue[0][0],
                           blob_normalisation=at_issue[0][2])
            return rec
    if at_issue:
        p, blob, norm = at_issue[0]
        rec.update(cell="at_issue", normalisation=norm, path=p, commit=issuing, blob=blob,
                   candidates=len(at_issue))
        return rec
    # elsewhere in HEAD's history, under any path with this basename
    matches = []
    seen = set()
    for commit, p in repo.commits_touching_basename(name):
        if commit == issuing:
            continue
        blob = repo.blob_at(commit, p)
        if not blob or blob in seen:
            continue
        seen.add(blob)
        norm = _blob_matches(repo, blob, digest)
        if norm:
            matches.append({"commit": commit, "path": p, "blob": blob, "normalisation": norm,
                            "relation": repo.relation(commit, issuing)})
    if matches:
        rels = {m["relation"] for m in matches}
        first = matches[0]
        rec.update(cell="elsewhere", normalisation=first["normalisation"], path=first["path"],
                   commit=first["commit"], blob=first["blob"],
                   relation=("after" if "after" in rels else
                             "before" if "before" in rels else "unrelated"),
                   matches=matches[:8])
        return rec
    rec.update(cell="unbacked", searched_commits=len(seen))
    return rec


def classify_certificate(repo: Repo, cert_path: Path, cert: dict, resolved: dict) -> dict:
    """Every citation of one certificate → its cell. *resolved* maps receipt name → the working
    tree path ``corpus_audit._resolve_receipts`` found (absent names resolve nowhere)."""
    rel = repo.rel(cert_path)
    out = {"certificate": rel, "issuing_commit": None, "citations": [], "cells": {c: 0 for c in CELLS}}
    issuing = None
    try:
        if repo.is_clean_tracked(rel):
            issuing = repo.last_commit_touching(rel)
        else:
            out["unrecoverable_reason"] = "certificate untracked or modified in the working tree"
    except RepoUnavailable as e:
        out["unrecoverable_reason"] = str(e)
    out["issuing_commit"] = issuing
    for name, digest in cert.get("receipts_sha256", {}).items():
        c = classify_citation(repo, issuing, name, digest, resolved.get(name))
        out["citations"].append(c)
        out["cells"][c["cell"]] += 1
    return out


# ---------------------------------------------------------------- R4: the sworn bytes

def sworn_bytes_at_issue(repo: Repo, cert_path: Path, classified: dict,
                         resolved: dict) -> Optional[dict]:
    """``{"document": bytes, "receipts": {name: bytes}}`` — the bytes the certificate swore to,
    when every citation is ``same`` or ``at_issue`` and the document exists at the issuing commit.
    Otherwise ``None``: the audit records ``stands_over_sworn_bytes: null`` and never guesses."""
    issuing = classified.get("issuing_commit")
    if not issuing:
        return None
    if any(c["cell"] not in ("same", "at_issue") for c in classified["citations"]):
        return None
    doc_rel = repo.rel(cert_path.with_name(cert_path.name.replace(".certificate.json", ".md")))
    doc_blob = repo.blob_at(issuing, doc_rel)
    if not doc_blob:
        return None
    receipts = {}
    for c in classified["citations"]:
        if c.get("blob"):
            receipts[c["name"]] = repo.cat(c["blob"])
        else:
            wp = resolved.get(c["name"])
            if wp is None or not Path(wp).exists():
                return None
            receipts[c["name"]] = Path(wp).read_bytes()
    return {"document": repo.cat(doc_blob), "receipts": receipts, "commit": issuing}
