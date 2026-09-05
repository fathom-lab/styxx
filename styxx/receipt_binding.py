"""styxx.receipt_binding — where a certificate's receipt bytes are: in the working tree, and in git history.

SPEC: ``papers/closed-model-frontier/SPEC_oath_receipt_binding_2026_09_04.md`` (frozen before this
module was written; a dated ERRATA after the adversarial pass names what the battery changed).
This is the one module that talks to git on the certificate's behalf (rule R7). It has no
dependency on ``styxx.certify``; ``certify_doc`` calls :func:`bind_at_mint` to fill the
certificate's ``receipt_binding`` block (R1), and ``styxx.corpus_audit`` calls
:func:`classify_certificate` and :func:`sworn_bytes_at_issue` to fill the five cells (R3) and
re-derive over the sworn bytes (R4). Every failure degrades to ``None`` / ``unrecoverable`` with a
reason; nothing here raises into a verdict.

Why it exists, in one sentence: an OATH certificate recorded each receipt's digest and then bound
by *basename*, so a receipt regenerated in place silently invalidated every certificate citing it
(twice in two days, 2026-08-31 → 2026-09-01, and once in June), and the audit could not tell *the
receipt moved* from *the certificate is wrong*.

Content identity is modulo newlines everywhere (``content_sha256``): the corpus's recorded digests
are, with six exceptions, CRLF hashes taken on Windows over LF blobs, and
``corpus_audit._receipt_sha_matches`` and ``charon._content_sha256`` already compare that way. The
record always says which reading matched, so the weakening is visible per citation. A certificate
that carries its own ``receipt_binding`` block is matched on ``content_sha256`` first
(reading ``content``), which is the only reading that survives a receipt with mixed newlines.

What the battery taught this module (ATTACKS_receipt_binding_battery_2026_09_05.md): history is
searched with ``--full-history -m`` because a merge that is TREESAME to its first parent hides the
receipt's other parent; paths come back ``-z`` because ``core.quotepath`` octal-escapes anything
non-ASCII; glob metacharacters in a basename are neutralised before they reach a pathspec; basenames
compare case-insensitively when the repository says ``core.ignorecase``; a citation whose sworn
bytes still sit in the working tree *outside the audit root* reads ``same`` with that path, not
``at_issue``; and the document is checked at the issuing commit against ``document_sha256`` before
any verdict is re-derived over it, because a certificate stands over its sworn bytes only if the
document is one of them.

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
DOCUMENT_CELLS = ("same", "at_issue", "moved", "unrecoverable")

__all__ = ["SCHEMA", "CONTENT_RULE", "CELLS", "DOCUMENT_CELLS", "content_sha256", "raw_sha256",
           "git_blob_id", "match_normalisation", "Repo", "RepoUnavailable", "bind_at_mint",
           "classify_citation", "classify_certificate", "sworn_bytes_at_issue"]


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


def match_normalisation(raw: bytes, recorded: str, content: Optional[str] = None) -> Optional[str]:
    """Which reading of *raw* hashes to the recorded digest: ``content`` (the certificate's own
    ``content_sha256``, when it carries one), ``raw``, ``lf``, ``crlf`` — or ``None``.

    Same three newline readings as ``corpus_audit._receipt_sha_matches``; this returns the name
    instead of a bool so every record can say how it matched. The ``content`` reading is tried
    first because it is the only one a receipt with MIXED newlines can satisfy.
    """
    if content and content_sha256(raw) == content:
        return "content"
    if hashlib.sha256(raw).hexdigest() == recorded:
        return "raw"
    lf = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(lf).hexdigest() == recorded:
        return "lf"
    if hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == recorded:
        return "crlf"
    return None


def _glob_escape(name: str) -> str:
    """A basename made safe for a ``:(glob)`` pathspec: every metacharacter becomes ``?`` (one
    character), and the Python-side basename comparison rejects any over-match."""
    return "".join("?" if ch in "[]*?\\" else ch for ch in name)


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


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
        self._ignorecase: Optional[bool] = None

    # -- plumbing
    def _run(self, args: list, cwd: Optional[Path] = None, ok_codes=(0,)) -> bytes:
        try:
            p = subprocess.run([self.git, *args], cwd=str(cwd or self.top),
                               capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RepoUnavailable(f"git {args[0]} failed: {e}")
        if p.returncode not in ok_codes:
            raise RepoUnavailable(f"git {' '.join(args[:3])} exited {p.returncode}: "
                                  f"{p.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return p.stdout

    def rel(self, path: Path) -> str:
        """Repository-relative posix path; ValueError when *path* is on another drive."""
        return Path(os.path.relpath(Path(path).resolve(), self.top)).as_posix()

    def rel_or_none(self, path: Path) -> Optional[str]:
        try:
            r = self.rel(path)
        except ValueError:
            return None
        return None if r.startswith("..") else r

    @property
    def shallow(self) -> bool:
        return self._run(["rev-parse", "--is-shallow-repository"]).strip() == b"true"

    @property
    def ignorecase(self) -> bool:
        if self._ignorecase is None:
            out = self._run(["config", "--get", "core.ignorecase"], ok_codes=(0, 1))
            self._ignorecase = out.strip().lower() == b"true"
        return self._ignorecase

    def names_match(self, a: str, b: str) -> bool:
        return a.casefold() == b.casefold() if self.ignorecase else a == b

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

    def paths_named(self, commit: str, name: str) -> list:
        """Every path at *commit* whose basename is *name* (case-folded when the repository
        ignores case)."""
        return [p for p in self.tree(commit) if self.names_match(_basename(p), name)]

    def blob_at(self, commit: str, relpath: str) -> Optional[str]:
        if commit in self._tree_cache:
            return self._tree_cache[commit].get(relpath)
        return self.tree_blobs(commit, [relpath]).get(relpath)

    def cat(self, blob: str) -> bytes:
        """Bytes of one blob, through a persistent ``cat-file --batch`` (binary pipes) so a census
        is not one process per receipt."""
        if self._catfile is None or self._catfile.poll() is not None:
            self._catfile = subprocess.Popen([self.git, "cat-file", "--batch"], cwd=str(self.top),
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
        p = self._catfile
        try:
            p.stdin.write(blob.encode() + b"\n")
            p.stdin.flush()
            header = p.stdout.readline()
        except (OSError, ValueError) as e:
            self.close()
            raise RepoUnavailable(f"cat-file batch failed: {e}")
        if not header or header.rstrip(b"\n").endswith(b" missing"):
            raise RepoUnavailable(f"blob {blob} missing")
        parts = header.split()
        if len(parts) != 3:
            self.close()
            raise RepoUnavailable(f"cat-file batch header unreadable: {header[:60]!r}")
        size = int(parts[2])
        data = bytearray()
        while len(data) < size:
            chunk = p.stdout.read(size - len(data))
            if not chunk:
                self.close()
                raise RepoUnavailable(f"blob {blob} truncated at {len(data)} of {size} bytes")
            data.extend(chunk)
        p.stdout.read(1)   # the trailing newline the batch protocol appends
        return bytes(data)

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
        touched any path whose basename is *name* — merges included on every parent
        (``--full-history -m``), paths unquoted (``-z``). Basename search only: a receipt renamed
        before its first commit is invisible here, which the SPEC records as owed."""
        spec = _glob_escape(name)
        out = self._run(["log", "-z", "--full-history", "-m", "--format=%x01%H", "--name-only",
                         "--", f":(glob)**/{spec}", spec], ok_codes=(0, 128))
        hits, commit, seen = [], None, set()
        for rec in out.split(b"\0"):
            if not rec:
                continue
            if rec.startswith(b"\x01"):
                commit = rec[1:].decode("ascii", "replace").strip()
                continue
            path = rec.lstrip(b"\n").decode("utf-8", "replace").strip()
            if path and commit and self.names_match(_basename(path), name):
                if (commit, path) not in seen:
                    seen.add((commit, path))
                    hits.append((commit, path))
        return hits

    def relation(self, commit: str, issuing: str) -> str:
        """``before`` if *commit* is an ancestor of *issuing*, ``after`` if the reverse,
        ``unrelated`` when neither (a parallel branch merged later)."""
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

def bind_at_mint(receipt_paths: list, near: Optional[Path] = None) -> dict:
    """The certificate's ``receipt_binding`` block. Never raises: without git, ``head`` is null
    with the reason, every ``blob`` is null and every ``committed`` is false — the certificate does
    not vouch for what it could not see. *near* is where to look for the repository when there are
    no receipts (the document's directory), so ``head`` is filled even then."""
    block = {"schema": SCHEMA, "content_rule": CONTENT_RULE, "head": None,
             "all_receipts_committed": False, "receipts": []}
    paths = [Path(p) for p in receipt_paths]
    repo = None
    reason = None
    try:
        anchor = paths[0].parent if paths else (Path(near) if near is not None else None)
        repo = Repo(anchor) if anchor is not None else None
        if repo is None:
            reason = "no receipts and no document directory to locate a repository"
    except RepoUnavailable as e:
        reason = str(e)
    head = repo.head() if repo else None
    head_blobs: dict = {}
    rels: dict = {}
    if repo and head:
        for p in paths:
            rels[p] = repo.rel_or_none(p)
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
    if not paths:
        block["note"] = "no receipts"
    if reason:
        block["note"] = f"no repository at mint: {reason}"
    if repo:
        repo.close()
    return block


# ---------------------------------------------------------------- R3: the five cells

def _blob_matches(repo: Repo, blob: str, digest: str, content: Optional[str]) -> Optional[str]:
    try:
        return match_normalisation(repo.cat(blob), digest, content)
    except RepoUnavailable:
        return None


def classify_citation(repo: Repo, issuing: Optional[str], name: str, digest: str,
                      working: Optional[Path], head: Optional[str] = None,
                      content: Optional[str] = None, hint_path: Optional[str] = None) -> dict:
    """One receipt citation → one cell (SPEC R3, as amended by the ERRATA), with the evidence
    that decided it. *working* is the path R2 resolved under the audit root (or None);
    *content* and *hint_path* come from the certificate's own binding block when it has one."""
    rec = {"name": name, "digest": digest, "cell": None, "normalisation": None,
           "path": None, "commit": None, "blob": None, "at_issue_too": None}
    if issuing is None:
        rec.update(cell="unrecoverable", reason="no issuing commit")
        return rec
    # candidates at the issuing commit: the recorded path first, then every path with this basename
    at_issue = []
    cands = ([hint_path] if hint_path and hint_path in repo.tree(issuing) else []) + \
        [p for p in repo.paths_named(issuing, name) if p != hint_path]
    for p in cands:
        blob = repo.blob_at(issuing, p)
        norm = _blob_matches(repo, blob, digest, content) if blob else None
        if norm:
            at_issue.append((p, blob, norm))

    def _same(path_rel: str, norm: str, note: Optional[str] = None) -> dict:
        rec.update(cell="same", normalisation=norm, path=path_rel, at_issue_too=bool(at_issue))
        if at_issue:
            # the working-tree reading and the committed blob's reading are recorded
            # separately: on a Windows checkout the first is `raw` and the second `crlf`
            rec.update(commit=issuing, blob=at_issue[0][1], blob_path=at_issue[0][0],
                       blob_normalisation=at_issue[0][2])
        if note:
            rec["note"] = note
        return rec

    if working is not None and Path(working).exists():
        norm = match_normalisation(Path(working).read_bytes(), digest, content)
        if norm:
            return _same(repo.rel_or_none(Path(working)) or str(working), norm)
    # the sworn bytes may still sit in the working tree OUTSIDE the audit root (a synthesis
    # citing another arc's receipt): that is `same`, not `at_issue`
    head = head or repo.head()
    if head:
        for p in repo.paths_named(head, name):
            wp = repo.top / p
            try:
                norm = match_normalisation(wp.read_bytes(), digest, content) if wp.is_file() else None
            except OSError:
                norm = None
            if norm:
                return _same(p, norm, note="resolved outside the audit root")
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
        norm = _blob_matches(repo, blob, digest, content)
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
    rec.update(cell="unbacked", searched_blobs=len(seen))
    return rec


def _classify_document(repo: Repo, cert_path: Path, cert: dict, issuing: Optional[str]) -> dict:
    """Where the DOCUMENT's sworn bytes are (``document_sha256`` is the LF text hash certify
    took). ``same``: the working document beside the certificate matches; ``at_issue``: a blob
    at the issuing commit matches (by the certificate's path-derived name, or by the
    certificate's own ``document`` field); ``moved``: neither; ``unrecoverable``: no issuing
    commit or no recorded digest."""
    out = {"cell": None, "path": None, "blob": None, "normalisation": None, "at_issue_too": None}
    digest = cert.get("document_sha256")
    if not digest:
        out.update(cell="unrecoverable", reason="no document_sha256 recorded")
        return out
    working = cert_path.with_name(cert_path.name.replace(".certificate.json", ".md"))
    at_issue = []
    if issuing:
        names = [working.name]
        if cert.get("document") and cert["document"] != working.name:
            names.append(cert["document"])
        seen = set()
        for n in names:
            for p in repo.paths_named(issuing, n):
                if p in seen:
                    continue
                seen.add(p)
                blob = repo.blob_at(issuing, p)
                norm = _blob_matches(repo, blob, digest, None) if blob else None
                if norm:
                    at_issue.append((p, blob, norm))
    if working.exists():
        norm = match_normalisation(working.read_bytes(), digest)
        if norm:
            out.update(cell="same", normalisation=norm,
                       path=repo.rel_or_none(working) or str(working), at_issue_too=bool(at_issue))
            if at_issue:
                out.update(commit=issuing, blob=at_issue[0][1], blob_path=at_issue[0][0])
            return out
    if issuing is None:
        out.update(cell="unrecoverable", reason="no issuing commit")
        return out
    if at_issue:
        p, blob, norm = at_issue[0]
        out.update(cell="at_issue", normalisation=norm, path=p, blob=blob, commit=issuing)
        return out
    out.update(cell="moved", reason="neither the working document nor any document of that "
                                    "name at the issuing commit hashes to document_sha256")
    return out


def classify_certificate(repo: Repo, cert_path: Path, cert: dict, resolved: dict) -> dict:
    """Every citation of one certificate → its cell, plus the document's. *resolved* maps
    receipt name → the working tree path ``corpus_audit._resolve_receipts`` found under the audit
    root (absent names resolve nowhere)."""
    rel = repo.rel_or_none(cert_path)
    out = {"certificate": rel or str(cert_path), "issuing_commit": None, "citations": [],
           "cells": {c: 0 for c in CELLS}, "document": None}
    issuing = None
    if rel is None:
        out["unrecoverable_reason"] = "certificate outside the repository"
    else:
        try:
            if repo.is_clean_tracked(rel):
                issuing = repo.last_commit_touching(rel)
                if issuing is None:
                    out["unrecoverable_reason"] = "certificate untracked"
            else:
                out["unrecoverable_reason"] = "certificate untracked or modified in the working tree"
        except RepoUnavailable as e:
            out["unrecoverable_reason"] = str(e)
    out["issuing_commit"] = issuing
    hints = {}
    for r in (cert.get("receipt_binding") or {}).get("receipts", []) or []:
        if isinstance(r, dict) and r.get("name"):
            hints[r["name"]] = (r.get("path"), r.get("content_sha256"))
    head = repo.head()
    for name, digest in cert.get("receipts_sha256", {}).items():
        hint_path, content = hints.get(name, (None, None))
        c = classify_citation(repo, issuing, name, digest, resolved.get(name), head=head,
                              content=content, hint_path=hint_path)
        out["citations"].append(c)
        out["cells"][c["cell"]] += 1
    out["document"] = _classify_document(repo, cert_path, cert, issuing)
    return out


# ---------------------------------------------------------------- R4: the sworn bytes

def sworn_bytes_at_issue(repo: Repo, cert_path: Path, classified: dict, resolved: dict):
    """``({"document": bytes, "receipts": {name: bytes}, "commit": I(C)}, None)`` — the bytes the
    certificate swore to — when every citation's blob is known (``same``, ``at_issue`` or
    ``elsewhere``) and the document's sworn bytes are recoverable. Otherwise ``(None, reason)``:
    the audit records ``stands_over_sworn_bytes: null`` with the reason and never guesses."""
    issuing = classified.get("issuing_commit")
    if not issuing:
        return None, "no issuing commit"
    cits = classified.get("citations") or []
    if not cits:
        return None, "no citations"
    bad = [c["cell"] for c in cits if c["cell"] not in ("same", "at_issue", "elsewhere")]
    if bad:
        return None, f"a citation is {sorted(set(bad))[0]}"
    doc = classified.get("document") or {}
    if doc.get("cell") == "same":
        wp = Path(doc["path"]) if os.path.isabs(str(doc["path"])) else repo.top / doc["path"]
        try:
            doc_bytes = wp.read_bytes()
        except OSError:
            return None, "document unreadable"
    elif doc.get("cell") == "at_issue":
        doc_bytes = repo.cat(doc["blob"])
    elif doc.get("cell") == "moved":
        return None, "document at issuing commit is not the sworn document"
    else:
        return None, "document unrecoverable"
    receipts = {}
    for c in cits:
        if c.get("blob"):
            receipts[c["name"]] = repo.cat(c["blob"])
        else:
            wp = resolved.get(c["name"])
            if wp is None and c.get("path"):
                wp = repo.top / c["path"]
            if wp is None or not Path(wp).exists():
                return None, f"sworn bytes of {c['name']} unreadable"
            receipts[c["name"]] = Path(wp).read_bytes()
    return {"document": doc_bytes, "receipts": receipts, "commit": issuing}, None
