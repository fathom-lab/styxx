# -*- coding: utf-8 -*-
"""styxx.prereg_ci.git: the git layer of prereg-ci.

Every reading here is a function of git objects at named commits, read with plumbing. Nothing reads
a working copy and no command here talks to a remote; in a partial clone git itself fetches a
missing blob from the clone's promisor remote, which a full clone never needs. The readings
(DESIGN 3.2(b), 3.2(d), 3.4):

* a revision resolves to a full commit id (``rev-parse --verify <rev>^{commit}``), or COMMIT_ABSENT;
* the parent-one chain of a revision is walked from ``rev-list --parents`` (parent index 0), so no
  flag of git's names the walk;
* the landing commit L of a path is the oldest commit on the default branch's parent-one chain
  whose diff against its parent one touches the path (``diff-tree --stdin`` over the chain's pairs);
  the seal bytes are the path's blob at L;
* the add commit A is the oldest commit of the path's history whose diff adds it: the rule
  ``styxx.protocol`` reads with ``--diff-filter=A``, without ``--follow`` (the design's RV1);
* the order commit is the oldest commit, among L and its ancestors (HEAD and its ancestors, for a
  PREREG that arrives in the same pull request as its receipt, C2), at which the path's decision
  digest already was the one at L; the file-blob reading (the design's RV2 comparison) does the same
  on the blob id. "Oldest" is git's date order, which lists no commit before its children; where
  the value was written on branches that do not reach each other, every minimal holder is listed
  too. A decision changed and changed back orders from the commit that wrote it before the change;
  the blob history between seal and receipt (git's full history of the path) prints the interval;
* strict ancestry is ``merge-base --is-ancestor`` with a commit never strictly before itself; the
  blob history of a path between two commits; the paths a pull request adds, changes, deletes or
  renames, read from the merge base of BASE and HEAD, as git's three-dot diff reads them;
* receipts: R1's record rule, the resolver of 3.2(d) (``PREREG_AMBIGUOUS``, never a guess), the
  recorded digest of PLAN K2, the receipts a pull request writes (C1: every path it leaves holding
  a receipt, whatever the path held before, with R the oldest commit of the pull request at which
  the path holds one), and the immutability of the receipts it found committed
  (``RECEIPT_REWRITTEN``);
* sealed decisions (C0): a fence edited in place, a deleted PREREG and a renamed one are
  ``DECISION_EDITED_AFTER_SEAL`` (fail closed); prose edited around an unchanged fence is
  ``PROSE_AMENDED_IN_PLACE``, a NOTE, or a FAIL under ``immutability = "file"``. This is the refusal
  path protection gives, kept so one tool covers it (DESIGN 3.4);
* the seals manifest (3.2(b), step B2): its lines as a revision holds them; every ``blob_sha256``
  against the blob at its commit, which the history the manifest is read in must hold; the
  append-only rule, for a pull request (merge base to HEAD) and along the parent-one chain; the
  ``seen`` null of an imported record (PLAN K10); a prereg line held to the default seal's
  decision (``MANIFEST_SEAL_NOT_LANDED``); the decision in effect for a receipt, with
  ``AMENDED_AFTER_RECEIPT`` and ``AMENDED_AFTER_DATA`` (C4); and the lines the ``seal`` verb writes
  (3.3, PLAN K9).

History is read as the commit objects record it. Every command runs with replacement off
(``--no-replace-objects`` and ``GIT_NO_REPLACE_OBJECTS``, the rule of ``styxx.sworn.GitTree``), with
``GIT_GRAFT_FILE`` pointed at the null device (git honours ``info/grafts`` with replacement off),
and with the commit-graph cache off, so a parent is the one its commit object names. A shallow clone
is ``SHALLOW_CLONE`` (UNCHECKABLE) for every reading that walks history; bytes at a named commit
still read. A blob over ``styxx.sworn.MAX_RECEIPT_BYTES`` is never read (``BLOB_TOO_LARGE``).

Nothing here is public API: DESIGN 3.1 names ``core.rescore``, ``.compare``, ``.licences`` and the
CLI, and the CLI is this module's caller. Every module-level name is private.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from styxx.prereg_ci import core as _core
from styxx.sworn import MAX_RECEIPT_BYTES as _RECEIPT_CAP

# the kind of each code: UNCHECKABLE is exit 3, INPUT exit 2, FAIL refuses, NOTE is printed
_KIND = {
    "COMMIT_ABSENT": "UNCHECKABLE", "SHALLOW_CLONE": "UNCHECKABLE", "GIT_UNAVAILABLE": "UNCHECKABLE",
    "BLOB_TOO_LARGE": "UNCHECKABLE",
    "NOT_A_REPOSITORY": "INPUT", "NOT_A_BLOB": "INPUT", "PATH_UNREADABLE": "INPUT", "NO_MERGE_BASE": "INPUT",
    "PREREG_ABSENT": "INPUT", "PREREG_AMBIGUOUS": "INPUT", "RECEIPT_ABSENT": "INPUT", "RECEIPT_AMBIGUOUS": "INPUT",
    "RECORDED_DIGEST_AMBIGUOUS": "INPUT",
    "RECEIPT_UNREADABLE": "NOTE", "RECEIPT_NOT_A_FILE": "NOTE",
    "RECEIPT_REWRITTEN": "FAIL", "DECISION_EDITED_AFTER_SEAL": "FAIL",
    "PROSE_AMENDED_IN_PLACE": "NOTE",
    # the seals manifest (step B2)
    "MANIFEST_INVALID": "INPUT", "NOT_COMMITTED": "INPUT",
    "MANIFEST_SEEN_MISSING": "FAIL", "MANIFEST_BLOB_DIFFERS": "FAIL", "MANIFEST_COMMIT_UNREACHABLE": "FAIL",
    "MANIFEST_REWRITTEN": "FAIL",
    # a manifest prereg line against the default seal of its path (the B2 review, case A)
    "MANIFEST_SEAL_NOT_LANDED": "FAIL", "MANIFEST_SEAL_PROSE_DIFFERS": "NOTE",
}

_MAX_BLOB_BYTES = _RECEIPT_CAP           # read at call time, so a test may lower it
_GIT_TIMEOUT_S = 300
# Suffixes match whatever their case (a receipt.JSON is read, a PREREG.MD's fence is held). Nothing
# is left out by default: a repository-relative prefix in ``exclude`` (styxx's census can name
# conformance/, whose vectors hold gates digests inside their inputs) is configuration, and the
# paths it leaves out of C1 are listed for the report's not_examined.
_HEX = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_NEEDLE = '"gates_sha256"'
# git's own environment that would point a command at another repository, ref namespace or shallow
# list, or read its pathspecs another way than the one each call names
_DROP_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_SHALLOW_FILE",
             "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS",
             "GIT_REPLACE_REF_BASE")
_GLOBAL = ("--no-replace-objects", "-c", "core.commitGraph=false", "-c", "color.ui=false", "-c",
           "core.quotepath=false")
_POLICIES = ("fail", "note")
_IMMUTABILITY = ("gates", "file")
_HOW = {"D": "deleted", "M": "changed in place", "T": "changed to another file type"}
_WRITES = ("A", "R", "M", "T")           # the changes that leave a path holding new bytes at HEAD
_MANIFEST = "prereg-ci.seals.jsonl"      # DESIGN 3.2(b): the seals manifest, at the work tree's top


class _Refusal(Exception):
    """A reading this layer does not give: ``code``, its ``kind``, ``detail``, and for an
    ambiguous name the ``candidates``."""

    def __init__(self, code: str, detail: str, candidates: Sequence[str] = ()):
        super().__init__(f"{code}: {detail}")
        self.code, self.kind, self.detail = code, _KIND[code], detail
        self.candidates = tuple(candidates)

    def finding(self, path: Optional[str] = None) -> Dict[str, Any]:
        return {"code": self.code, "kind": self.kind, "detail": self.detail, "path": path}


def _env() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() not in _DROP_ENV}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_GRAFT_FILE=os.devnull, GIT_TERMINAL_PROMPT="0",
               GIT_OPTIONAL_LOCKS="0")
    return env


def _plain_path(path: str) -> str:
    """A path git can be asked about one line at a time, or PATH_UNREADABLE."""
    if not isinstance(path, str) or not path or any(ch in path for ch in "\n\r\0"):
        raise _Refusal("PATH_UNREADABLE", f"{path!r} is not a path this layer asks git about one line at a time")
    return path


class _Batch:
    """One long-lived ``git cat-file`` process: one object name a line in, one answer out."""

    def __init__(self, argv: List[str], env: Dict[str, str]):
        try:
            self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, env=env)
        except (OSError, ValueError) as e:
            raise _Refusal("GIT_UNAVAILABLE", f"git cat-file could not start: {e}")

    def _line(self, name: str) -> bytes:
        try:
            self.p.stdin.write(name.encode("utf-8", "surrogateescape") + b"\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        except (OSError, ValueError) as e:
            raise _Refusal("GIT_UNAVAILABLE", f"git cat-file stopped answering: {e}")
        if not line.endswith(b"\n"):
            raise _Refusal("GIT_UNAVAILABLE", "git cat-file ended before it answered")
        return line[:-1]

    def header(self, name: str) -> Optional[Tuple[str, str, int]]:
        """(oid, type, size) of the object *name* names, or None where it names none."""
        line = self._line(name)
        if line.endswith((b" missing", b" ambiguous")):
            return None
        parts = line.split(b" ")
        if len(parts) != 3 or not parts[2].isdigit() or not _HEX.fullmatch(parts[0].decode("ascii", "replace")):
            raise _Refusal("GIT_UNAVAILABLE", f"git cat-file answered {line[:200]!r}")
        return parts[0].decode("ascii"), parts[1].decode("ascii"), int(parts[2])

    def contents(self, oid: str) -> bytes:
        head = self.header(oid)
        if head is None:
            raise _Refusal("GIT_UNAVAILABLE", f"object {oid} is not in the object store")
        size = head[2]
        try:
            data = self.p.stdout.read(size)
            tail = self.p.stdout.read(1)
        except (OSError, ValueError) as e:
            raise _Refusal("GIT_UNAVAILABLE", f"git cat-file stopped mid-object: {e}")
        if len(data) != size or tail != b"\n":
            raise _Refusal("GIT_UNAVAILABLE", f"git cat-file sent {len(data)} of {size} bytes of {oid}")
        return data

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()
        finally:
            if self.p.stdout is not None:
                self.p.stdout.close()


class _Paths(frozenset):
    """A tree's blob paths, with the basename index the resolver reads."""

    def __new__(cls, paths: Iterable[str]):
        self = super().__new__(cls, paths)
        by: Dict[str, List[str]] = {}
        for p in sorted(self):
            by.setdefault(p.rsplit("/", 1)[-1], []).append(p)
        self.by_name = {k: tuple(v) for k, v in by.items()}
        return self


class _Repo:
    """A repository read through git plumbing; a context manager, since it keeps two ``cat-file``
    processes open for the walks. Readings are kept by commit id, which names immutable objects;
    a revision name is resolved once, so a ``_Repo`` reads a repository whose refs do not move
    while it is open (one check, one census). *root* may be any directory inside the work tree:
    every command runs at the work tree's top, so a path is always repository-relative."""

    def __init__(self, root: Any):
        self.root = os.fspath(root)
        self._top: Optional[str] = None
        self._env = _env()
        self._shallow: Optional[bool] = None
        self._commits: Dict[str, str] = {}
        self._parents: Dict[str, Tuple[str, ...]] = {}
        self._chains: Dict[str, List[str]] = {}
        self._trees: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
        self._orders: Dict[str, Tuple[str, ...]] = {}
        self._views: Dict[str, _Paths] = {}
        self._digests: Dict[str, Optional[str]] = {}
        self._check: Optional[_Batch] = None
        self._cat: Optional[_Batch] = None
        self._memo: Dict[Any, Any] = {}

    # -- process plumbing ------------------------------------------------------------------------

    def __enter__(self) -> "_Repo":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:                           # noqa: BLE001 - interpreter shutdown
            pass

    def close(self) -> None:
        for b in (self._check, self._cat):
            if b is not None:
                b.close()
        self._check = self._cat = None

    def _where(self) -> str:
        """The directory every command runs in: the work tree's top (``rev-parse --show-toplevel``),
        so git never reads a pathspec relative to a subdirectory *root* names; a bare repository or
        a git directory as given, since git reads no prefix there. NOT_A_REPOSITORY otherwise."""
        if self._top is None:
            def ask(*flags: str) -> Tuple[int, List[str]]:
                try:
                    r = subprocess.run(["git", *_GLOBAL, "-C", self.root, "rev-parse", *flags], capture_output=True,
                                       env=self._env, check=False, timeout=_GIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    raise _Refusal("GIT_UNAVAILABLE", f"git rev-parse ran past {_GIT_TIMEOUT_S} s")
                except (OSError, ValueError) as e:
                    raise _Refusal("GIT_UNAVAILABLE", f"git could not run: {e}")
                return r.returncode, r.stdout.decode("utf-8", "surrogateescape").splitlines()

            rc, flags = ask("--is-bare-repository", "--is-inside-git-dir")
            if rc != 0 or len(flags) != 2 or any(f not in ("true", "false") for f in flags):
                raise _Refusal("NOT_A_REPOSITORY", f"{self.root} is not inside a git repository git will read")
            if "true" in flags:
                self._top = self.root
            else:
                rc, top = ask("--show-toplevel")
                if rc != 0 or len(top) != 1 or not top[0]:
                    raise _Refusal("NOT_A_REPOSITORY", f"git names no work tree top for {self.root}")
                self._top = top[0]
        return self._top

    def _argv(self, *args: str, literal: bool = True) -> List[str]:
        return ["git", *_GLOBAL, *(("--literal-pathspecs",) if literal else ()), "-C", self._where(), *args]

    def _run(self, *args: str, stdin: Optional[bytes] = None, literal: bool = True) -> Tuple[int, bytes, bytes]:
        try:
            r = subprocess.run(self._argv(*args, literal=literal), input=stdin, capture_output=True,
                               env=self._env, check=False, timeout=_GIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise _Refusal("GIT_UNAVAILABLE", f"git {args[0]} ran past {_GIT_TIMEOUT_S} s")
        except (OSError, ValueError) as e:
            raise _Refusal("GIT_UNAVAILABLE", f"git could not run: {e}")
        return r.returncode, r.stdout, r.stderr

    def _ok(self, *args: str, stdin: Optional[bytes] = None, literal: bool = True) -> bytes:
        rc, out, err = self._run(*args, stdin=stdin, literal=literal)
        if rc != 0:
            raise _Refusal("GIT_UNAVAILABLE", f"git {args[0]} exited {rc}: "
                                              f"{err.decode('utf-8', 'replace').strip()[:300]}")
        return out

    def _batch_check(self) -> _Batch:
        if self._check is None:
            self._check = _Batch(self._argv("cat-file", "--batch-check"), self._env)
        return self._check

    def _batch_cat(self) -> _Batch:
        if self._cat is None:
            self._cat = _Batch(self._argv("cat-file", "--batch"), self._env)
        return self._cat

    # -- revisions and history --------------------------------------------------------------------

    def commit(self, rev: Any) -> str:
        """The full id of the commit *rev* names, or COMMIT_ABSENT."""
        if not isinstance(rev, str) or not rev or any(ch in rev for ch in "\n\r\0"):
            raise _Refusal("COMMIT_ABSENT", f"{rev!r} names no commit")
        if rev not in self._commits:
            rc, out, _ = self._run("rev-parse", "--verify", "--quiet", "--end-of-options", rev + "^{commit}")
            oid = out.decode("ascii", "replace").strip()
            if rc != 0 or not _HEX.fullmatch(oid):
                raise _Refusal("COMMIT_ABSENT", f"{rev!r} names no commit in {self.root}")
            self._commits[rev] = oid
        return self._commits[rev]

    def is_shallow(self) -> bool:
        if self._shallow is None:
            out = self._ok("rev-parse", "--is-shallow-repository").strip()
            if out not in (b"true", b"false"):
                raise _Refusal("GIT_UNAVAILABLE", f"git rev-parse answered {out[:80]!r} on shallowness")
            self._shallow = out == b"true"
        return self._shallow

    def _full(self) -> None:
        if self.is_shallow():
            raise _Refusal("SHALLOW_CLONE", "the clone is shallow: its history stops at a cut boundary, so the "
                                            "landing commit, the order commit and ancestry are not read")

    def _parents_of(self, commit: str) -> Tuple[str, ...]:
        if commit not in self._parents:
            for line in self._ok("rev-list", "--parents", commit).decode("ascii").split("\n"):
                ids = line.split()
                if ids:
                    self._parents.setdefault(ids[0], tuple(ids[1:]))
            if commit not in self._parents:
                raise _Refusal("COMMIT_ABSENT", f"{commit} is not in the history git lists")
        return self._parents[commit]

    def parents(self, rev: str) -> Tuple[str, ...]:
        """The parents *rev*'s commit object names, in order."""
        self._full()
        return self._parents_of(self.commit(rev))

    def chain(self, rev: str) -> List[str]:
        """*rev*'s parent-one chain, newest to oldest: parent index 0 of each commit, to a root."""
        self._full()
        tip = self.commit(rev)
        if tip not in self._chains:
            out: List[str] = []
            cur: Optional[str] = tip
            while cur is not None:
                out.append(cur)
                ps = self._parents_of(cur)
                cur = ps[0] if ps else None
            self._chains[tip] = out
        return list(self._chains[tip])

    def is_strict_ancestor(self, a: str, b: str) -> bool:
        """Whether *a* is an ancestor of *b* and not *b* itself (C3)."""
        self._full()
        a, b = self.commit(a), self.commit(b)
        if a == b:
            return False
        if ("anc", a, b) in self._memo:
            return self._memo[("anc", a, b)]
        rc, _, err = self._run("merge-base", "--is-ancestor", a, b)
        if rc in (0, 1):
            self._memo[("anc", a, b)] = rc == 0
            return rc == 0
        raise _Refusal("COMMIT_ABSENT", f"git merge-base exited {rc}: {err.decode('utf-8', 'replace').strip()[:300]}")

    def merge_base(self, base: str, head: str) -> str:
        self._full()
        base, head = self.commit(base), self.commit(head)
        rc, out, err = self._run("merge-base", base, head)
        oid = out.decode("ascii", "replace").strip()
        if rc == 1 and not oid:
            raise _Refusal("NO_MERGE_BASE", f"{base} and {head} share no history")
        if rc != 0 or not _HEX.fullmatch(oid):
            raise _Refusal("COMMIT_ABSENT", f"git merge-base exited {rc}: {err.decode('utf-8', 'replace').strip()[:300]}")
        return oid

    # -- trees and blobs ------------------------------------------------------------------------

    def _tree(self, commit: str) -> Dict[str, Tuple[str, str, str]]:
        if commit not in self._trees:
            entries: Dict[str, Tuple[str, str, str]] = {}
            order: List[str] = []
            for ent in self._ok("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
                if not ent:
                    continue
                meta, _, raw = ent.partition(b"\t")
                mode, typ, oid = meta.decode("ascii").split()
                path = raw.decode("utf-8", "surrogateescape")
                entries[path] = (mode, typ, oid)
                if typ == "blob":
                    order.append(path)
            self._trees[commit] = entries
            self._orders[commit] = tuple(order)
        return self._trees[commit]

    def paths(self, rev: str) -> Tuple[str, ...]:
        """Every blob path of *rev*'s tree, in git's tree order."""
        c = self.commit(rev)
        self._tree(c)
        return self._orders[c]

    def _view(self, rev: str) -> _Paths:
        c = self.commit(rev)
        if c not in self._views:
            self._views[c] = _Paths(self.paths(c))
        return self._views[c]

    def _entry(self, commit: str, path: str) -> Optional[Tuple[str, str, str]]:
        """(mode, type, oid) of the entry at *path*; a directory reads as its tree entry."""
        hit = self._trees.get(commit, {}).get(path)
        if hit is not None:
            return hit
        out = self._ok("ls-tree", "-z", "--full-tree", commit, "--", path)
        for ent in out.split(b"\0"):
            meta, _, raw = ent.partition(b"\t")
            if raw.decode("utf-8", "surrogateescape") == path:
                mode, typ, oid = meta.decode("ascii").split()
                return mode, typ, oid
        return None

    def _blob_id(self, commit: str, path: str) -> Optional[str]:
        head = self._batch_check().header(f"{commit}:{_plain_path(path)}")
        return head[0] if head is not None and head[1] == "blob" else None

    def blob_id(self, rev: str, path: str) -> Optional[str]:
        """The blob id at *path* in *rev*'s tree, or None where no blob is there."""
        return self._blob_id(self.commit(rev), path)

    def _contents(self, oid: str, path: str) -> bytes:
        head = self._batch_check().header(oid)
        if head is None:
            raise _Refusal("GIT_UNAVAILABLE", f"blob {oid} ({path}) is not in the object store")
        if head[2] > _MAX_BLOB_BYTES:
            raise _Refusal("BLOB_TOO_LARGE", f"{path} is {head[2]} bytes, over the {_MAX_BLOB_BYTES} this layer reads")
        return self._batch_cat().contents(oid)

    def read(self, rev: str, path: str) -> Optional[bytes]:
        """The bytes of the regular file at *path* in *rev*'s tree; None where no entry is there.
        A symlink or a submodule is NOT_A_BLOB, as ``styxx.sworn.GitTree.blob`` reads them."""
        c = self.commit(rev)
        e = self._entry(c, _plain_path(path))
        if e is None:
            return None
        mode, typ, oid = e
        if typ != "blob" or mode == "120000":
            raise _Refusal("NOT_A_BLOB", f"{path} at {c[:12]} is a {'symlink' if mode == '120000' else typ}, "
                                         f"not a regular file")
        return self._contents(oid, path)

    def _digest(self, oid: str, path: str) -> Optional[str]:
        if oid not in self._digests:
            self._digests[oid] = _core._decision(self._contents(oid, path))["gates_sha256"]
        return self._digests[oid]

    def _key(self, commit: str, path: str, key: str) -> Optional[str]:
        oid = self._blob_id(commit, path)
        if oid is None or key == "blob":
            return oid
        return self._digest(oid, path)

    # -- the seal rules -------------------------------------------------------------------------

    def landing(self, paths: Iterable[str], rev: str) -> Dict[str, Optional[str]]:
        """Each path's landing commit on *rev*'s parent-one chain (DESIGN 3.2(b)), or None for a
        path no commit of the chain touches. A pathspec also matches the files under a directory of
        that name, which ``diff-tree`` names too; a token is a commit header only when it is a chain
        commit's id, and a path that was not asked about is passed over."""
        chain = self.chain(rev)
        tip = chain[0]
        asked = list(dict.fromkeys(_plain_path(p) for p in paths))
        for p in asked:
            if _HEX.fullmatch(p):
                raise _Refusal("PATH_UNREADABLE", f"{p!r} reads like a commit id in diff-tree's output")
        wanted = [p for p in asked if ("landing", tip, p) not in self._memo]
        found: Dict[str, Optional[str]] = {p: None for p in wanted}
        on_chain = set(chain)
        feed = "".join(f"{c} {self._parents_of(c)[0]}\n" if self._parents_of(c) else f"{c}\n"
                       for c in chain) if wanted else ""
        for chunk in _chunks(wanted):
            want = {p.encode("utf-8", "surrogateescape"): p for p in chunk}
            out = self._ok("diff-tree", "--stdin", "-r", "-z", "--name-only", "--root", "--", *chunk,
                           stdin=feed.encode("ascii"))
            current: Optional[str] = None
            for tok in out.split(b"\0"):
                if not tok:
                    continue
                if tok in want:
                    if current is None:
                        raise _Refusal("GIT_UNAVAILABLE", "git diff-tree named a path before a commit")
                    found[want[tok]] = current          # the chain runs newest to oldest; the last one stays
                    continue
                text = tok.decode("utf-8", "surrogateescape")
                if _HEX.fullmatch(text):
                    if text not in on_chain:
                        raise _Refusal("GIT_UNAVAILABLE", f"git diff-tree named {text[:60]!r}, not a chain commit")
                    current = text
                elif current is None:
                    raise _Refusal("GIT_UNAVAILABLE", f"git diff-tree named {text[:60]!r} before a commit")
                # else: a file under a directory an asked path names (asked paths never read as ids)
        for p, c in found.items():
            self._memo[("landing", tip, p)] = c
        return {p: self._memo[("landing", tip, p)] for p in asked}

    def add_commit(self, path: str, rev: str) -> Optional[str]:
        """The oldest commit of *path*'s history from *rev* whose diff adds the path: git's
        ``--diff-filter=A`` reading (a merge shows no diff there), or None."""
        self._full()
        tip = self.commit(rev)
        memo = ("add", tip, _plain_path(path))
        if memo not in self._memo:
            self._parents_of(tip)
            self._memo[memo] = None
            for c in reversed(self._ok("rev-list", tip, "--", path).decode("ascii").split()):
                ps = self._parents_of(c)
                if len(ps) > 1 or self._blob_id(c, path) is None:
                    continue
                if not ps or self._blob_id(ps[0], path) is None:
                    self._memo[memo] = c
                    break
        return self._memo[memo]

    def order_commit(self, path: str, start: str, key: str = "decision") -> Optional[str]:
        """The order commit of DESIGN 3.2(b) from *start* (L, or HEAD for a PREREG not yet landed):
        the oldest commit, among *start* and its ancestors, at which the path's decision digest
        (``key="decision"``) or blob id (``key="blob"``) already was the one at *start*. None where
        *start* holds no such value (no blob, or no decision the scanner reads).

        The oldest holder is a commit that wrote the value, so it differs at the path from every
        parent and ``rev-list --full-history`` lists it, on whichever branch it sits; "oldest" is
        git's date order (commit date, with no commit listed before all of its children), as the
        design's RV2 reads it with ``git log``, so the order commit never has a holder among its
        ancestors, whatever dates the author set. Holders on branches that do not reach each other
        are each oldest on their own side: ``order_holders`` lists every one."""
        held = self._holding(path, start, key)
        return held[-1] if held else None

    def order_holders(self, path: str, start: str, key: str = "decision") -> Tuple[str, ...]:
        """Every minimal holder of the value at *start*: the commits among *start* and its
        ancestors holding it with no other holder among their own ancestors, newest to oldest in
        git's date order; the order commit is the last. Two or more mean the value was written on
        branches that do not reach each other (the same plan committed twice), where "oldest" is
        a partial order: C3 can ask whether any of them is strictly before R, or print each."""
        return tuple(_minimal(self, self._holding(path, start, key)))

    def _holding(self, path: str, start: str, key: str) -> Tuple[str, ...]:
        if key not in ("decision", "blob"):
            raise ValueError(f"key is 'decision' or 'blob', not {key!r}")
        self._full()
        c = self.commit(start)
        memo = ("held", c, _plain_path(path), key)
        if memo not in self._memo:
            want = self._key(c, path, key)
            held: List[str] = []
            if want is not None:
                self._parents_of(c)
                held = self._held(path, c, lambda x: self._key(x, path, key) == want)
            self._memo[memo] = tuple(held)
        return self._memo[memo]

    def _held(self, path: str, start: str, holds: Any, stop: Optional[str] = None) -> List[str]:
        """Every commit git lists for *path* from *start*, *stop* and its ancestors left out, at
        which ``holds(commit)`` is true, newest to oldest in git's date order (no commit before all of
        its children); ``[start]`` where git lists none and *start* holds, else ``[]``. No other
        holder listed is an ancestor of the last one."""
        argv = ["rev-list", "--full-history", "--date-order", start, *(("--not", stop) if stop else ()),
                "--", _plain_path(path)]
        listed = [x for x in self._ok(*argv).decode("ascii").split() if holds(x)]
        return listed or ([start] if holds(start) else [])

    def blob_history(self, path: str, a: str, b: str) -> List[Tuple[str, Optional[str]]]:
        """(commit, blob id) for *a* and every commit from *b* back to *a* whose diff against one of
        its parents touches *path* (a merge that takes the path unchanged from every parent is left
        out), oldest to newest with no commit before its parents. DESIGN 3.2(e) prints it because an
        edit and revert on a branch never reaches a per-PR check (RED_TEAM m8), so the history is
        git's full one (``--full-history``): default simplification follows one TREESAME parent of a
        merge and drops the side branch that edited and reverted the path."""
        self._full()
        a, b = self.commit(a), self.commit(b)
        listed = self._ok("rev-list", "--full-history", "--date-order", b, "--not", a + "^@", "--",
                          _plain_path(path)).decode("ascii").split()
        commits = [a] + [c for c in reversed(listed) if c != a]
        return [(c, self._blob_id(c, path)) for c in commits]

    def renamed_from(self, path: str, rev: str) -> Optional[str]:
        """The path *path* was renamed from in *rev*'s diff against its parent one, or None."""
        self._full()
        c = self.commit(rev)
        ps = self._parents_of(c)
        if not ps:
            return None
        for ch in _name_status(self._ok("diff-tree", "-r", "-z", "-M", "--name-status", ps[0], c)):
            if ch["status"] == "R" and ch["path"] == path:
                return ch["old"]
        return None

    def changes(self, base: str, head: str) -> Dict[str, Any]:
        """What a pull request changes: the tree diff from the merge base of *base* and *head* to
        *head*, with renames named. ``{base, head, merge_base, changes}``; each change is
        ``{status, path, old}`` with status A, M, T, D or R (``old`` None for A)."""
        self._full()
        b, h = self.commit(base), self.commit(head)
        mb = self.merge_base(b, h)
        out = self._ok("diff-tree", "-r", "-z", "-M", "--name-status", mb, h)
        return {"base": b, "head": h, "merge_base": mb, "changes": _name_status(out)}

    def grep(self, rev: str, needle: str, glob: str) -> List[str]:
        """Paths under *rev*'s tree matching *glob* whose bytes hold *needle*, in tree order."""
        c = self.commit(rev)
        rc, out, err = self._run("grep", "-l", "-z", "-F", "--no-textconv", "-e", needle, c, "--", glob, literal=False)
        if rc == 1:
            return []
        if rc != 0:
            raise _Refusal("GIT_UNAVAILABLE", f"git grep exited {rc}: {err.decode('utf-8', 'replace').strip()[:300]}")
        prefix = (c + ":").encode("ascii")
        return [tok[len(prefix):].decode("utf-8", "surrogateescape") for tok in out.split(b"\0")
                if tok.startswith(prefix)]


def _minimal(repo: _Repo, held: Sequence[str]) -> List[str]:
    """The commits of *held* (newest to oldest in git's date order, so an ancestor is never listed
    before its descendant) that have no other commit of *held* among their ancestors."""
    return [x for i, x in enumerate(held) if not any(repo.is_strict_ancestor(o, x) for o in held[i + 1:])]


def _chunks(paths: Sequence[str], budget: int = 6000) -> List[List[str]]:
    """Pathspec groups that keep a command line short on every platform."""
    out: List[List[str]] = []
    cur: List[str] = []
    size = 0
    for p in paths:
        if cur and (size + len(p) + 1 > budget or len(cur) >= 200):
            out.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p) + 1
    if cur:
        out.append(cur)
    return out


def _name_status(raw: bytes) -> List[Dict[str, Any]]:
    toks = [t.decode("utf-8", "surrogateescape") for t in raw.split(b"\0")]
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(toks) and toks[i]:
        status = toks[i][:1]
        if status in ("R", "C"):
            out.append({"status": status, "path": toks[i + 2], "old": toks[i + 1]})
            i += 3
        else:
            path = toks[i + 1]
            out.append({"status": status, "path": path, "old": None if status == "A" else path})
            i += 2
    return out


# ---- receipts: the resolver, the records, the recorded digest ----------------------------------------

def _resolve(name: Any, near: str, paths: Any, what: str = "PREREG") -> str:
    """DESIGN 3.2(d)'s resolver. A value holding a ``/`` is a repository-relative path; a bare
    basename is the file of that name in *near* (the receipt's own directory), else the one file of
    that name in *paths* (the tree the receipt is read at). Two candidates are ``<what>_AMBIGUOUS``,
    never a guess; anything else that names no file is ``<what>_ABSENT``."""
    absent, ambiguous = f"{what}_ABSENT", f"{what}_AMBIGUOUS"
    if not isinstance(name, str) or not name:
        raise _Refusal(absent, f"the record names no {what} ({name!r})")
    if any(ch in name for ch in "\\\0\n\r"):
        raise _Refusal(absent, f"{name!r} is neither a repository-relative path nor a bare basename")
    if "/" in name:
        parts = name.split("/")
        if name.startswith("/") or re.match(r"[A-Za-z]:", name) or any(p in ("", ".", "..") for p in parts):
            raise _Refusal(absent, f"{name!r} is not a repository-relative path")
        if name in paths:
            return name
        raise _Refusal(absent, f"{name} is not in the tree")
    if name in (".", ".."):
        raise _Refusal(absent, f"{name!r} names no file")
    here = posixpath.join(near, name) if near else name
    if here in paths:
        return here
    by_name = getattr(paths, "by_name", None)
    cands = by_name.get(name, ()) if by_name is not None else tuple(sorted(p for p in paths
                                                                          if p.rsplit("/", 1)[-1] == name))
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise _Refusal(absent, f"no file named {name} is in the tree")
    raise _Refusal(ambiguous, f"{len(cands)} files are named {name}: {', '.join(cands)}", candidates=cands)


def _receipt_kind(obj: Any) -> Optional[str]:
    """``"receipt"`` for an object naming its PREREG by a top-level ``prereg`` string (3.2(d)),
    ``"record"`` for one that records a gates digest (R1's rule: a top-level ``gates_sha256``, or a
    ``protocol`` entry holding one, as ``styxx.seal`` writes), else None."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("prereg"), str):
        return "receipt"
    if "gates_sha256" in obj:
        return "record"
    entries = obj.get("protocol")
    if isinstance(entries, list) and any(isinstance(e, dict) and "gates_sha256" in e for e in entries):
        return "record"
    return None


def _json(data: bytes) -> Any:
    """A receipt as the conformance set reads one: UTF-8, then ``json.loads``."""
    return json.loads(data.decode("utf-8"))


def _fields(obj: Dict[str, Any], verdict_key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if verdict_key in obj:
        out["verdict"] = obj[verdict_key]
    if "gates" in obj:
        out["gates"] = obj["gates"]
    out["gates_sha256"] = obj.get("gates_sha256")
    return out


def _excluded(path: str, exclude: Sequence[str]) -> bool:
    return any(path.startswith(x) for x in exclude)


def _suffix(path: str, suffixes: Sequence[str]) -> bool:
    """Whether *path* ends with one of *suffixes*, case aside."""
    low = path.lower()
    return any(low.endswith(s.lower()) for s in suffixes)


def _not_a_file(e: _Refusal, path: str, rev: str) -> Dict[str, Any]:
    """The NOTE for a ``.json`` path that is a symlink or a submodule: it holds no receipt this
    layer reads, and it does not stop the check (C0 skips one at the base the same way). Any other
    refusal is raised again."""
    if e.code != "NOT_A_BLOB":
        raise e
    return _Refusal("RECEIPT_NOT_A_FILE", f"{path} at {rev[:12]} is not a regular file, so no receipt is read "
                                          f"from it").finding(path)


def _records(repo: _Repo, rev: str, exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Every committed record naming a gates digest at *rev* (R1's rule): a JSON file whose
    top-level object holds ``gates_sha256``, or one entry per ``protocol`` item holding it. Each is
    ``{record, pointer, own, dir, prereg, result, prereg_commit, recorded}``, with ``recorded`` the
    ``verdict`` (``frozen_verdict`` in a protocol entry), ``gates`` and ``gates_sha256`` it states.
    A file that holds the key and is not JSON is listed with a ``refusal`` (a NOTE); git grep reads
    regular files only, and a path that read as anything else would be a NOTE too. A ``.json``
    suffix is read whatever its case; paths under *exclude* are not read."""
    c = repo.commit(rev)
    out: List[Dict[str, Any]] = []
    for path in repo.grep(c, _NEEDLE, ":(icase)*.json"):
        if _excluded(path, exclude):
            continue
        base = {"record": path, "dir": posixpath.dirname(path)}
        try:
            d = _json(repo.read(c, path) or b"")
        except _Refusal as e:
            out.append({**base, "pointer": "", "own": True, "refusal": _not_a_file(e, path, c)})
            continue
        except (UnicodeDecodeError, ValueError) as e:
            out.append({**base, "pointer": "", "own": True,
                        "refusal": _Refusal("RECEIPT_UNREADABLE", f"{path} is not JSON: {e}").finding(path)})
            continue
        if isinstance(d, dict) and isinstance(d.get("protocol"), list):
            for i, entry in enumerate(d["protocol"]):
                if isinstance(entry, dict) and "gates_sha256" in entry:
                    out.append({**base, "pointer": f"/protocol/{i}", "own": False, "prereg": entry.get("prereg"),
                                "result": entry.get("result"), "prereg_commit": entry.get("prereg_commit"),
                                "recorded": _fields(entry, "frozen_verdict")})
        elif isinstance(d, dict) and "gates_sha256" in d:
            out.append({**base, "pointer": "", "own": True, "prereg": d.get("prereg"), "result": path,
                        "prereg_commit": d.get("prereg_commit"), "recorded": _fields(d, "verdict")})
    return out


def _pairs(repo: _Repo, rev: str, exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """``_records`` with each record's receipt and PREREG resolved in *rev*'s tree: ``receipt_path``
    (the record itself, or its ``result`` resolved from the record's directory) and ``prereg_path``
    (resolved from the receipt's directory), or a ``refusal``."""
    tree = repo._view(rev)
    out = []
    for rec in _records(repo, rev, exclude):
        rec = dict(rec)
        if "refusal" not in rec:
            try:
                rp = rec["record"] if rec["own"] else _resolve(rec["result"], rec["dir"], tree, what="RECEIPT")
                pp = _resolve(rec["prereg"], posixpath.dirname(rp), tree, what="PREREG")
                rec.update(receipt_path=rp, prereg_path=pp)
            except _Refusal as e:
                rec["refusal"] = e.finding(rec["record"])
        out.append(rec)
    return out


def _sidecars(repo: _Repo, rev: str, exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """Every ``protocol`` entry holding a gates digest in every ``*.seal.json`` of *rev*'s tree,
    by the receipt its ``result`` resolves to. ``{by, ambiguous, unreadable, not_files}``: the
    sidecars that are not JSON, and the symlinks and submodules, are listed and not read."""
    c = repo.commit(rev)
    key = ("sidecars", c, tuple(exclude))
    if key not in repo._memo:
        tree = repo._view(c)
        by: Dict[str, List[Dict[str, Any]]] = {}
        ambiguous: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []
        unreadable: List[str] = []
        not_files: List[str] = []
        for path in repo.paths(c):
            if not _suffix(path, (".seal.json",)) or _excluded(path, exclude):
                continue
            try:
                d = _json(repo.read(c, path) or b"")
            except _Refusal as e:
                _not_a_file(e, path, c)
                not_files.append(path)
                continue
            except (UnicodeDecodeError, ValueError):
                unreadable.append(path)
                continue
            if not isinstance(d, dict) or not isinstance(d.get("protocol"), list):
                continue
            for i, entry in enumerate(d["protocol"]):
                if not isinstance(entry, dict) or "gates_sha256" not in entry:
                    continue
                rec = {**_fields(entry, "frozen_verdict"), "source": f"{path}#/protocol/{i}"}
                try:
                    target = _resolve(entry.get("result"), posixpath.dirname(path), tree, what="RECEIPT")
                except _Refusal as e:
                    if e.code == "RECEIPT_AMBIGUOUS":
                        ambiguous.append((e.candidates, rec))
                    continue
                by.setdefault(target, []).append(rec)
        repo._memo[key] = {"by": by, "ambiguous": ambiguous, "unreadable": unreadable, "not_files": not_files}
    return repo._memo[key]


def _recorded(repo: _Repo, rev: str, receipt_path: str, receipt: Any,
              exclude: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """The digest a receipt was scored under (DESIGN 3.2(d), PLAN K2): its own ``gates_sha256``,
    else every ``protocol`` entry of every ``*.seal.json`` in *rev*'s tree whose ``result`` resolves
    to the receipt. Readings that disagree (digest, verdict or gates), or an entry whose result is
    ambiguous between this receipt and another file, are ``RECORDED_DIGEST_AMBIGUOUS``; none is None
    (printed as absent). Returns ``{gates_sha256, verdict?, gates?, source}``."""
    if isinstance(receipt, dict) and receipt.get("gates_sha256") is not None:
        return {**_fields(receipt, "verdict"), "source": receipt_path}
    side = _sidecars(repo, rev, exclude)
    for cands, rec in side["ambiguous"]:
        if receipt_path in cands:
            raise _Refusal("RECORDED_DIGEST_AMBIGUOUS", f"{rec['source']} names a result that resolves to "
                                                        f"{', '.join(cands)}, one of them {receipt_path}")
    found = list(side["by"].get(receipt_path, ()))
    readings = {json.dumps([f["gates_sha256"], f.get("verdict"), f.get("gates")], sort_keys=True) for f in found}
    if len(readings) > 1:
        raise _Refusal("RECORDED_DIGEST_AMBIGUOUS", f"{receipt_path} is recorded {len(readings)} ways: " + "; ".join(
            f"{f['source']} gates {f['gates_sha256']} verdict {f.get('verdict')}" for f in found))
    return found[0] if found else None


# ---- the seal of one PREREG (C2) ----------------------------------------------------------------------

def _default_start(repo: _Repo, path: str, base: str, head: Optional[str] = None
                   ) -> Tuple[Optional[str], str, str]:
    """``(L, start, add_rev)`` for the default seal of *path*: its landing commit L on *base*'s
    parent-one chain, read there; else, for a PREREG that arrives in the pull request (C2), no L and
    *head*'s bytes. PREREG_ABSENT otherwise."""
    b = repo.commit(base)
    landing = repo.landing([path], b)[path]
    if landing is not None:
        return landing, landing, b
    if head is not None and repo.blob_id(head, path) is not None:
        h = repo.commit(head)
        return None, h, h
    where = f"on {base}'s parent-one chain" + ("" if head is None else f" or in {head}")
    raise _Refusal("PREREG_ABSENT", f"{path} is not {where}")


def _seal(repo: _Repo, path: str, base: str, head: Optional[str] = None) -> Dict[str, Any]:
    """The default seal of *path* (DESIGN 3.2(b)): its landing commit L on *base*'s parent-one chain
    and the blob there, the order commit on the decision digest and on the file blob, every minimal
    holder of the decision (``order_holders``), the add commit, the decision the seal bytes hold,
    and the path L renamed it from, if any. A PREREG not on the chain that is in *head* arrives in
    the pull request (C2): no L, orders read from *head*, the add commit from *head*'s history.
    Neither is PREREG_ABSENT."""
    landing, start, add_rev = _default_start(repo, path, base, head)
    data = repo.read(start, path)
    if data is None:
        raise _Refusal("PREREG_ABSENT", f"{path} is not a file at {start[:12]}")
    decision = _core._decision(data)
    return {"path": path, "landing": landing, "landed": landing is not None, "add": repo.add_commit(path, add_rev),
            "order": repo.order_commit(path, start, "decision"), "order_blob": repo.order_commit(path, start, "blob"),
            "order_holders": repo.order_holders(path, start, "decision"),
            "blob": repo.blob_id(start, path), "blob_sha256": hashlib.sha256(data).hexdigest(), "bytes": data,
            "form": decision["form"], "gates_sha256": decision["gates_sha256"], "code": decision["code"],
            "detail": decision["detail"], "renamed_from": repo.renamed_from(path, landing) if landing else None}


# ---- what a pull request does to receipts and sealed decisions (C0, C1) -------------------------------

def _holds_receipt(repo: _Repo, commit: str, path: str) -> bool:
    """Whether *path* at *commit* is a regular file whose JSON names a PREREG (3.2(d))."""
    try:
        return _receipt_kind(_json(repo.read(commit, path) or b"")) == "receipt"
    except _Refusal as e:
        if e.code == "NOT_A_BLOB":
            return False                            # a symlink or a submodule holds no receipt
        raise
    except (UnicodeDecodeError, ValueError):
        return False


def _receipts_written(repo: _Repo, base: str, head: str,
                      exclude: Sequence[str] = ()) -> Dict[str, List[Any]]:
    """C1's receipts (DESIGN 3.4): every ``.json`` path the pull request leaves holding a receipt at
    HEAD whose change is an add, a rename's new path, an edit in place or a type change (A, R, M,
    T), whatever the path held at the merge base: nothing, a placeholder, bytes that are not JSON,
    or a receipt (which ``_receipts_rewritten`` also reads, so under ``receipt_rewrite = "note"`` the
    new bytes are still scored).

    Each is ``{path, status, old, receipt_commit, receipt_holders, receipt}``. ``receipt_commit`` is
    R, the oldest commit in merge_base..HEAD at which the path holds a receipt (git's date order), so
    a path used before the pull request does not move R outside it; ``receipt_holders`` is every
    such commit with no other among its ancestors, R the last (two or more where branches of the
    pull request that do not reach each other each wrote one).

    A changed ``.json`` that is not JSON at HEAD is a ``RECEIPT_UNREADABLE`` NOTE, since no receipt
    can be read from it; one that is a symlink or a submodule at HEAD is a ``RECEIPT_NOT_A_FILE``
    NOTE. A ``.json`` suffix is read whatever its case. Every changed ``.json`` path (new or old)
    under *exclude* is left out here and in ``_receipts_rewritten`` and listed in ``excluded``, for
    the report's not_examined. Returns ``{receipts, notes, excluded}``."""
    ch = repo.changes(base, head)
    mb, h = ch["merge_base"], ch["head"]
    receipts, notes, excluded = [], [], set()
    for x in ch["changes"]:
        path = x["path"]
        excluded.update(p for p in (path, x["old"]) if p and _suffix(p, (".json",)) and _excluded(p, exclude))
        if x["status"] not in _WRITES or not _suffix(path, (".json",)) or _excluded(path, exclude):
            continue
        try:
            obj = _json(repo.read(h, path) or b"")
        except _Refusal as e:
            notes.append(_not_a_file(e, path, h))
            continue
        except (UnicodeDecodeError, ValueError) as e:
            notes.append(_Refusal("RECEIPT_UNREADABLE", f"{path} is not JSON: {e}").finding(path))
            continue
        if _receipt_kind(obj) != "receipt":
            continue
        held = repo._held(path, h, lambda c, p=path: _holds_receipt(repo, c, p), stop=mb)
        receipts.append({"path": path, "status": x["status"], "old": x["old"], "receipt_commit": held[-1],
                         "receipt_holders": tuple(_minimal(repo, held)), "receipt": obj})
    return {"receipts": receipts, "notes": notes, "excluded": sorted(excluded)}


def _receipts_rewritten(repo: _Repo, base: str, head: str, policy: str = "fail",
                        exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """RECEIPT_REWRITTEN (DESIGN 3.2(d), C1) for every receipt or record committed at the merge
    base that the pull request changes, deletes or renames; a re-run writes a new path. FAIL under
    ``receipt_rewrite = "fail"`` (the default), a NOTE under ``"note"``."""
    if policy not in _POLICIES:
        raise ValueError(f"receipt_rewrite is one of {_POLICIES}, not {policy!r}")
    ch = repo.changes(base, head)
    mb = ch["merge_base"]
    out = []
    for x in ch["changes"]:
        old = x["old"]
        if old is None or not _suffix(old, (".json",)) or _excluded(old, exclude):
            continue
        try:
            obj = _json(repo.read(mb, old) or b"")
        except _Refusal as e:
            _not_a_file(e, old, mb)
            continue                                # a symlink or a submodule at the base held no receipt
        except (UnicodeDecodeError, ValueError):
            continue                                # no receipt was ever read from bytes that are not JSON
        if _receipt_kind(obj) is None:
            continue
        how = f"renamed to {x['path']}" if x["status"] == "R" else _HOW.get(x["status"], f"changed ({x['status']})")
        if x["status"] == "M":
            was, now = repo._entry(mb, old), repo._entry(ch["head"], x["path"])
            if was is not None and now is not None and was[2] == now[2] and was[0] != now[0]:
                how = f"changed in file mode only ({was[0]} to {now[0]}; its bytes are identical)"
        out.append({"code": "RECEIPT_REWRITTEN", "kind": "FAIL" if policy == "fail" else "NOTE", "path": old,
                    "detail": f"{old}, committed at {mb[:12]}, is {how} in this pull request; a re-run writes a "
                              f"new path (receipt_rewrite = {policy!r})"})
    return out


def _decisions_moved(repo: _Repo, base: str, head: str, immutability: str = "gates",
                     suffixes: Sequence[str] = (".md",)) -> List[Dict[str, Any]]:
    """C0 for every file whose name ends with one of *suffixes* (case aside, so ``PREREG.MD`` is
    read) that holds a decision at the merge base and that the pull request changes, deletes or
    renames: a deleted or renamed one, or one whose fence moved, is
    DECISION_EDITED_AFTER_SEAL; prose moved around an unchanged fence is PROSE_AMENDED_IN_PLACE,
    a NOTE under ``immutability = "gates"`` (the default) and a FAIL under ``"file"``."""
    if immutability not in _IMMUTABILITY:
        raise ValueError(f"immutability is one of {_IMMUTABILITY}, not {immutability!r}")
    ch = repo.changes(base, head)
    mb, h = ch["merge_base"], ch["head"]
    out = []
    for x in ch["changes"]:
        old = x["old"]
        if old is None or not _suffix(old, suffixes):
            continue
        try:
            before = _core._decision(repo.read(mb, old) or b"")
        except _Refusal as e:
            if e.code == "NOT_A_BLOB":
                continue                            # a symlink at the base holds no decision
            raise
        if before["form"] is None:
            continue
        sealed = before["gates_sha256"]
        moved = None
        if x["status"] == "D":
            moved = "is deleted"
        elif x["status"] == "R":
            moved = f"is renamed to {x['path']}"
        else:
            try:
                after = _core._decision(repo.read(h, old) or b"")
            except _Refusal as e:
                if e.code != "NOT_A_BLOB":
                    raise
                moved = "is no longer a regular file"
            else:
                if after["gates_sha256"] != sealed:
                    moved = ("holds no fence the scanner reads" if after["form"] is None
                             else f"holds the fence {after['gates_sha256']}")
        if moved is not None:
            out.append({"code": "DECISION_EDITED_AFTER_SEAL", "kind": "FAIL", "path": old,
                        "detail": f"{old} held the sealed fence {sealed} at {mb[:12]}; at HEAD it {moved}. A "
                                  f"sealed decision changes through an amendment file (fail closed)"})
        else:
            out.append({"code": "PROSE_AMENDED_IN_PLACE", "kind": "FAIL" if immutability == "file" else "NOTE",
                        "path": old, "detail": f"{old}: prose outside the fence {sealed} changed in place "
                                               f"(immutability = {immutability!r})"})
    return out


# ---- the seals manifest (DESIGN 3.2(b); step B2) -------------------------------------------------------

def _manifest_parse(data: bytes, path: str, where: str) -> List[Dict[str, Any]]:
    """``core._manifest`` over bytes read at *where*, its refusal raised as this layer's."""
    try:
        return _core._manifest(data)
    except _core._ManifestError as e:
        at = f", line {e.line}" if e.line is not None else ""
        raise _Refusal(e.code, f"{path} at {where}{at}: {e.detail}")


def _manifest_at(repo: _Repo, rev: str, path: str = _MANIFEST) -> Optional[List[Dict[str, Any]]]:
    """The manifest's lines as *rev*'s tree holds them, or None where it holds no manifest. A
    manifest path that is a symlink or a submodule is MANIFEST_INVALID."""
    c = repo.commit(rev)
    try:
        data = repo.read(c, path)
    except _Refusal as e:
        if e.code != "NOT_A_BLOB":
            raise
        raise _Refusal("MANIFEST_INVALID", f"{path} at {c[:12]} is not a regular file")
    return None if data is None else _manifest_parse(data, path, c[:12])


def _line(code: str, detail: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    return {"code": code, "kind": _KIND[code], "detail": detail, "path": entry["path"], "id": entry["id"]}


def _manifest_blobs(repo: _Repo, entries: Sequence[Dict[str, Any]], rev: str) -> List[Dict[str, Any]]:
    """DESIGN 3.2(b): every ``blob_sha256`` equals ``git show <commit>:<path> | sha256sum``, the
    sha256 of the blob as git stores it (never a working copy, never newlines translated), else
    ``MANIFEST_BLOB_DIFFERS``. A line's commit is *rev* or an ancestor of it, so every clone of the
    history the manifest is read in holds its bytes; else ``MANIFEST_COMMIT_UNREACHABLE``. A commit
    the object store lacks is COMMIT_ABSENT (UNCHECKABLE)."""
    tip = repo.commit(rev)
    out = []
    for e in entries:
        c = repo.commit(e["commit"])
        if c != tip and not repo.is_strict_ancestor(c, tip):
            out.append(_line("MANIFEST_COMMIT_UNREACHABLE", f"line {e['id']}: {c[:12]} is not in the history of "
                             f"{tip[:12]}, so a clone of that history holds no bytes for it", e))
            continue
        try:
            data = repo.read(c, e["path"])
        except _Refusal as r:
            if r.code != "NOT_A_BLOB":
                raise
            data = None
        if data is None:
            out.append(_line("MANIFEST_BLOB_DIFFERS", f"line {e['id']}: {e['path']} at {c[:12]} is no regular file, "
                             f"so no bytes there match blob_sha256 {e['blob_sha256']}", e))
            continue
        got = hashlib.sha256(data).hexdigest()
        if got != e["blob_sha256"]:
            out.append(_line("MANIFEST_BLOB_DIFFERS", f"line {e['id']}: git show {c[:12]}:{e['path']} | sha256sum "
                             f"gives {got}; the line records {e['blob_sha256']}", e))
    return out


def _not_appended(old: bytes, new: bytes) -> Optional[str]:
    """Why *new* is not *old* with lines appended, or None."""
    if new.startswith(old) and (not old or old.endswith(b"\n")):
        return None
    was, now = old.splitlines(keepends=True), new.splitlines(keepends=True)
    for i, (x, y) in enumerate(zip(was, now)):
        if x != y:
            return f"line {i + 1} is not the line the manifest held there"
    return f"it holds {len(now)} of the {len(was)} lines the manifest held"


def _manifest_append_only(repo: _Repo, base: str, head: str, path: str = _MANIFEST) -> List[Dict[str, Any]]:
    """DESIGN 3.2(b): the manifest is append-only. For a pull request, the bytes HEAD holds start
    with the bytes the merge base held (a manifest the pull request adds has nothing to keep); a
    manifest deleted, renamed away or made something other than a regular file is
    ``MANIFEST_REWRITTEN`` too."""
    ch = repo.changes(base, head)
    mb, h = ch["merge_base"], ch["head"]
    out = []

    def rewritten(how: str) -> None:
        out.append({"code": "MANIFEST_REWRITTEN", "kind": _KIND["MANIFEST_REWRITTEN"], "path": path, "id": None,
                    "detail": f"{path}, committed at {mb[:12]}, {how} in this pull request; the manifest is "
                              f"append-only, so a line is corrected by a new line"})

    for x in ch["changes"]:
        if x["old"] == path and x["path"] != path:
            rewritten("is deleted" if x["status"] == "D" else f"is renamed to {x['path']}")
        elif x["path"] == path:
            try:
                old = repo.read(mb, path)
                new = repo.read(h, path)
            except _Refusal as e:
                if e.code != "NOT_A_BLOB":
                    raise
                rewritten("is no longer a regular file")
                continue
            why = None if old is None else _not_appended(old, new or b"")
            if why is not None:
                rewritten(f"is not appended to: {why}")
    return out


def _manifest_chain(repo: _Repo, rev: str, path: str = _MANIFEST) -> List[Dict[str, Any]]:
    """Append-only along *rev*'s parent-one chain (the default branch's history, which a census
    reads): every chain commit whose manifest differs from its parent one's holds the parent's
    bytes with lines appended, else ``MANIFEST_REWRITTEN`` naming the commit."""
    out = []
    for c in reversed(repo.chain(rev)):
        ps = repo._parents_of(c)
        now, was = repo._blob_id(c, path), (repo._blob_id(ps[0], path) if ps else None)
        if now == was or was is None:
            continue
        why = "deleted it" if now is None else _not_appended(repo._contents(was, path), repo._contents(now, path))
        if why is not None:
            out.append({"code": "MANIFEST_REWRITTEN", "kind": _KIND["MANIFEST_REWRITTEN"], "path": path, "id": None,
                        "detail": f"{c[:12]} on the parent-one chain of {rev} rewrote {path}: {why}"})
    return out


def _manifest_seen(repo: _Repo, entries: Sequence[Dict[str, Any]], rev: str,
                   path: str = _MANIFEST) -> List[Dict[str, Any]]:
    """PLAN K10. An amendment or withdrawn line may say ``seen`` null only as part of the record
    the manifest imported when it began: the manifest held the same line at the commit that added
    it (git's add commit of *path* in *rev*'s history), and the line's commit is strictly before
    that commit. Any other null is ``MANIFEST_SEEN_MISSING`` (DESIGN 3.2(b): "a new amendment line
    without seen is refused"); a missing key is refused by the parser."""
    nulls = [e for e in entries if e["kind"] != "prereg" and e["seen"] is None]
    if not nulls:
        return []
    added = repo.add_commit(path, rev)
    imported: Dict[str, Dict[str, Any]] = {}
    if added is not None:
        try:
            imported = {e["id"]: e for e in _manifest_parse(repo.read(added, path) or b"", path, added[:12])}
        except _Refusal:
            imported = {}
    out = []
    for e in nulls:
        if added is None:
            why = f"{path} has no add commit in the history of {rev}"
        elif imported.get(e["id"]) != e:
            why = f"the line was not in {path} when {added[:12]} added it"
        elif not repo.is_strict_ancestor(e["commit"], added):
            why = f"its commit {e['commit'][:12]} is not strictly before {added[:12]}, which added {path}"
        else:
            continue
        out.append(_line("MANIFEST_SEEN_MISSING", f"line {e['id']} ({e['path']} at {e['commit'][:12]}): an amendment "
                         f"line states what had been seen; seen is null and {why}, so it is no record the manifest "
                         f"imported", e))
    return out


# ---- the decision in effect for one receipt (DESIGN 3.4 C2, C4) -----------------------------------------

def _bytes_of(repo: _Repo, entry: Dict[str, Any], rev: str) -> bytes:
    """A manifest line's bytes, refused unless they are what the line records."""
    found = _manifest_blobs(repo, [entry], rev)
    if found:
        raise _Refusal(found[0]["code"], found[0]["detail"])
    return repo.read(entry["commit"], entry["path"]) or b""


def _label(reading: Dict[str, Any]) -> str:
    return (f"the fence gates {reading['gates_sha256']}" if reading["form"] == "gates"
            else f"no decision the scanner reads ({reading['code']})")


def _seal_finding(repo: _Repo, line: Dict[str, Any], base: str, head: str) -> Optional[Dict[str, Any]]:
    """A manifest prereg line against the default seal of its path (DESIGN 3.2(b), C2; the B2
    review, case A): the landing rule on *base*'s parent-one chain, else, for a PREREG that arrives
    in the pull request, HEAD's bytes. The manifest overrides the default seal's commit and bytes,
    never its decision: a line whose bytes hold another decision (another fence, or none where the
    default holds one) is ``MANIFEST_SEAL_NOT_LANDED``, a FAIL, since a decision changed after it
    is an amendment line, with ``seen``, which C4 reads (``AMENDED_AFTER_DATA``). A line whose bytes
    hold the default's decision with other prose is ``MANIFEST_SEAL_PROSE_DIFFERS``, a NOTE. None
    where the line's bytes are the default seal's bytes."""
    path = line["path"]
    landing, start, _ = _default_start(repo, path, base, head)
    ref = repo.read(start, path) or b""
    mine = _bytes_of(repo, line, head)
    if mine == ref:
        return None
    a, d = _core._decision(mine), _core._decision(ref)
    where = (f"the default seal is its landing commit {landing[:12]} on the parent-one chain of {base}" if landing
             else f"{path} has not landed on the parent-one chain of {base}, so the default seal is its bytes at "
                  f"{start[:12]}, the head of the pull request (C2)")
    said = f"line {line['id']} seals {path} at {line['commit'][:12]}"
    if (a["form"], a["gates_sha256"], a["code"]) != (d["form"], d["gates_sha256"], d["code"]):
        return _line("MANIFEST_SEAL_NOT_LANDED", (
            f"{said}, holding {_label(a)}; {where}, holding {_label(d)}. A prereg line overrides the default seal's "
            f"commit and bytes, never its decision: a decision changed after it is an amendment line, with seen, "
            f"which AMENDED_AFTER_DATA reads. The default seal governs"), line)
    return _line("MANIFEST_SEAL_PROSE_DIFFERS", (
        f"{said} (blob sha256 {line['blob_sha256']}); {where} (blob sha256 {hashlib.sha256(ref).hexdigest()}). "
        f"Both hold {_label(d)}; the bytes differ outside the fence, and the line's bytes are the seal bytes"), line)


def _manifest_landed(repo: _Repo, entries: Sequence[Dict[str, Any]], base: str, head: str) -> List[Dict[str, Any]]:
    """``_seal_finding`` for every prereg line of *entries* (the manifest as *head* holds it), in
    file order: the manifest-wide reading of case A, which a pull request that adds a receipt
    nowhere still gets."""
    found = (_seal_finding(repo, e, base, head) for e in entries if e["kind"] == "prereg")
    return [f for f in found if f is not None]


def _line_seal(repo: _Repo, line: Dict[str, Any], base: str, head: str) -> Dict[str, Any]:
    """The seal a manifest line records, read from its own bytes, with the landing commit on *base*'s
    chain printed beside it (DESIGN 3.2(b): both are printed). No reading against the default seal."""
    path = line["path"]
    data = _bytes_of(repo, line, head)
    c, b = repo.commit(line["commit"]), repo.commit(base)
    landing = repo.landing([path], b)[path]
    decision = _core._decision(data)
    return {"path": path, "manifest_id": line["id"], "commit": c, "landing": landing, "landed": landing is not None,
            "add": repo.add_commit(path, b if landing is not None else head),
            "order": repo.order_commit(path, c, "decision"), "order_blob": repo.order_commit(path, c, "blob"),
            "order_holders": repo.order_holders(path, c, "decision"), "blob": repo.blob_id(c, path),
            "blob_sha256": line["blob_sha256"], "bytes": data, "form": decision["form"],
            "gates_sha256": decision["gates_sha256"], "code": decision["code"], "detail": decision["detail"],
            "renamed_from": repo.renamed_from(path, landing) if landing else None, "seen": line["seen"],
            "refused": None, "findings": []}


def _sealed(repo: _Repo, path: str, base: str, head: str,
            entries: Optional[Sequence[Dict[str, Any]]]) -> Tuple[Dict[str, Any], List[Tuple[Any, Any, bytes]]]:
    """The seal of *path* (C2: its manifest line, else the landing rule) and the lineage of that
    line with each line's decision and bytes. A manifest seal carries the landing commit beside it,
    printed (DESIGN 3.2(b): both are printed), and its reading against the default seal
    (``_seal_finding``) in ``findings``. A line refused there (``MANIFEST_SEAL_NOT_LANDED``) does not
    govern: the default seal does, under the line's id (its amendments still hang on it), and the
    refused line's commit and decision are kept in ``refused``. ``seen`` is the line's (a prereg
    line may state what had been seen when it was sealed; null is not stated), None for no line."""
    line = next((e for e in entries or () if e["kind"] == "prereg" and e["path"] == path), None)
    if line is None:
        seal = dict(_seal(repo, path, base, head))
        seal.update(manifest_id=None, commit=seal["landing"] or repo.commit(head), seen=None, refused=None,
                    findings=[])
        return seal, []
    seal = _line_seal(repo, line, base, head)
    found = _seal_finding(repo, line, base, head)
    if found is not None and found["kind"] == "FAIL":
        kept = dict(_seal(repo, path, base, head))
        kept.update(manifest_id=line["id"], commit=kept["landing"] or repo.commit(head), seen=line["seen"],
                    refused={k: seal[k] for k in ("commit", "blob_sha256", "form", "gates_sha256", "code")})
        seal = kept
    seal["findings"] = [found] if found is not None else []
    chain = []
    for e in _core._lineage(entries or (), line["id"]):
        b_e = _bytes_of(repo, e, head)
        chain.append((e, _core._decision(b_e), b_e))
    return seal, chain


def _reading(repo: _Repo, seal: Dict[str, Any], chain: Sequence[Tuple[Any, Any, bytes]], r: str) -> Dict[str, Any]:
    before = {e["id"]: repo.is_strict_ancestor(e["commit"], r) for e, _, _ in chain}
    readings = _core._amendment_readings(_core._decision(seal["bytes"]), chain)
    return _core._in_effect({"id": seal["manifest_id"], "gates_sha256": seal["gates_sha256"], "bytes": seal["bytes"]},
                            readings, before)


def _receipts_of(repo: _Repo, prereg_path: str, revs: Sequence[str],
                 exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Every receipt (3.2(d)) in the trees of *revs* whose ``prereg`` resolves to *prereg_path*,
    by path (a path in more than one tree is read at the earliest-listed rev that holds it), each
    ``{path, receipt, receipt_commit, receipt_holders, rev}``: R is the oldest commit of that rev's
    history at which the path holds a receipt."""
    out: Dict[str, Dict[str, Any]] = {}
    for rev in revs:
        c = repo.commit(rev)
        tree = repo._view(c)
        for path in repo.grep(c, '"prereg"', ":(icase)*.json"):
            if path in out or _excluded(path, exclude):
                continue
            try:
                obj = _json(repo.read(c, path) or b"")
            except _Refusal as e:
                if e.code == "NOT_A_BLOB":
                    continue
                raise
            except (UnicodeDecodeError, ValueError):
                continue
            if _receipt_kind(obj) != "receipt":
                continue
            try:
                if _resolve(obj["prereg"], posixpath.dirname(path), tree) != prereg_path:
                    continue
            except _Refusal:
                continue
            held = repo._held(path, c, lambda x, p=path: _holds_receipt(repo, x, p))
            out[path] = {"path": path, "receipt": obj, "receipt_commit": held[-1],
                         "receipt_holders": tuple(_minimal(repo, held)), "rev": c}
    return [out[p] for p in sorted(out)]


def _effect(repo: _Repo, prereg_path: str, receipt_path: str, receipt_commit: str, base: str, head: str,
            entries: Optional[Sequence[Dict[str, Any]]] = None, *, amend_after_data: str = "report",
            exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """The decision in effect for one receipt at R (DESIGN 3.4 C4), with the seal it starts from.

    *entries* is the manifest as HEAD holds it (``_manifest_at``), or None. The seal is the
    PREREG's manifest line, else the landing rule on *base*'s chain (C2); the decision is the seal's
    fence replaced by the latest own amendment line strictly before R (``core._in_effect``).
    Findings: the seal line's reading against the default seal (``MANIFEST_SEAL_NOT_LANDED``,
    ``MANIFEST_SEAL_PROSE_DIFFERS``), ``MANIFEST_DECISION_DIFFERS`` for a line its bytes do not bear out,
    ``AMENDED_AFTER_RECEIPT`` for each amendment line not before R, and, when an own amendment
    governs, ``AMENDED_AFTER_DATA`` for every other receipt of the PREREG (in HEAD's tree or in the
    tree at the amendment's commit) whose R is strictly before the amendment's commit, re-scored
    under its own decision in effect (``earlier``). A manifest line whose bytes are not what it
    records refuses the reading (``MANIFEST_BLOB_DIFFERS``, ``MANIFEST_COMMIT_UNREACHABLE``).

    Returns ``{seal, effect, findings, earlier}``; ``effect["bytes"]`` is what ``core.rescore``
    scores and ``effect["sealed"]`` the digests C6(i) reads."""
    if amend_after_data not in _core._AMEND_AFTER_DATA:
        raise ValueError(f"amend_after_data is one of {_core._AMEND_AFTER_DATA}, not {amend_after_data!r}")
    r = repo.commit(receipt_commit)
    seal, chain = _sealed(repo, prereg_path, base, head, entries)
    effect = _reading(repo, seal, chain, r)
    findings = list(seal["findings"]) + list(effect["findings"])
    earlier: List[Dict[str, Any]] = []
    gov = effect["source"]
    if gov is not None and gov != seal["manifest_id"]:
        amendment = next(e for e, _, _ in chain if e["id"] == gov)
        for x in _receipts_of(repo, prereg_path, [head, amendment["commit"]], exclude):
            r0 = x["receipt_commit"]
            if x["path"] == receipt_path or not repo.is_strict_ancestor(r0, amendment["commit"]):
                continue
            under = _reading(repo, seal, chain, r0)
            scored = _core.rescore(under["bytes"], x["receipt"])
            earlier.append({"path": x["path"], "commit": r0, "before_amendment": True, "source": under["source"],
                            "verdict": scored["verdict"], "code": scored["code"]})
        findings += _core._amended_after_data(effect, earlier, amend_after_data)
    return {"seal": seal, "effect": effect, "findings": findings, "earlier": earlier}


# ---- the lines ``seal`` writes (DESIGN 3.3; PLAN K9) --------------------------------------------------

def _next_number(entries: Sequence[Dict[str, Any]]) -> str:
    nums = [int(m.group(0)) for m in (re.match(r"[0-9]+", e["id"]) for e in entries) if m]
    return str(max(nums) + 1 if nums else 1)


def _amendment_id(entries: Sequence[Dict[str, Any]], root: str) -> str:
    """The root line's id with the next letter (SEALS' 3b, 6b, 7b), then ``<root>.<n>``."""
    used = {e["id"] for e in entries}
    for ch in "bcdefghijklmnopqrstuvwxyz":
        if root + ch not in used:
            return root + ch
    n = 2
    while f"{root}.{n}" in used:
        n += 1
    return f"{root}.{n}"


def _root_of(entries: Sequence[Dict[str, Any]], line_id: str) -> Dict[str, Any]:
    by = {e["id"]: e for e in entries}
    e = by[line_id]
    while e["kind"] != "prereg":
        e = by[e["amends"]]
    return e


def _sealed_line(repo: _Repo, entries: Sequence[Dict[str, Any]], path: str, head: str, commit: Optional[str],
                 kind: str, amends: Optional[str] = None, seen: Optional[str] = None) -> Dict[str, Any]:
    if commit is None:
        c = repo.landing([path], head)[path]
        if c is None:
            raise _Refusal("NOT_COMMITTED", f"{path} is on no commit of HEAD's parent-one chain; commit it, or name "
                                            f"the commit that holds it with --commit")
    else:
        c = repo.commit(commit)
        if c != head and not repo.is_strict_ancestor(c, head):
            raise _Refusal("NOT_COMMITTED", f"{c[:12]} is not in HEAD's history, so a clone of it holds no bytes "
                                            f"there")
    data = repo.read(c, path)
    if data is None:
        raise _Refusal("NOT_COMMITTED", f"{path} is not a file at {c[:12]}")
    if kind == "prereg":
        line_id, decision = _next_number(entries), "own"
        landing, start, _ = _default_start(repo, path, head, head)
        a, d = _core._decision(data), _core._decision(repo.read(start, path) or b"")
        if (a["form"], a["gates_sha256"], a["code"]) != (d["form"], d["gates_sha256"], d["code"]):
            raise _Refusal("MANIFEST_SEAL_NOT_LANDED", (
                f"{path} at {c[:12]} holds {_label(a)}; it landed at {(landing or start)[:12]} on HEAD's parent-one chain "
                f"holding {_label(d)}, and check refuses a prereg line whose decision is not the landed one. A "
                f"decision changed after landing is an amendment line: seal {path} --amends ID --seen LEVEL "
                f"--commit {c[:12]}"))
    else:
        reading = _core._decision(data)
        if reading["form"] is None and reading["code"] != "NOT_GATED":
            raise _Refusal("MANIFEST_INVALID", f"{path} at {c[:12]} holds a fence the scanner refuses: "
                                               f"{reading['detail']}")
        root = _root_of(entries, amends)
        digest = _core._decision(repo.read(root["commit"], root["path"]) or b"")["gates_sha256"]
        for e in _core._lineage(entries, root["id"]):
            if e["kind"] == "amendment" and e["decision"] == "own":
                digest = _core._decision(repo.read(e["commit"], e["path"]) or b"")["gates_sha256"]
        decision = "own" if reading["form"] is not None and reading["gates_sha256"] != digest else "inherits"
        line_id = _amendment_id(entries, root["id"])
    return {"id": line_id, "kind": kind, "path": path, "commit": c, "blob_sha256": hashlib.sha256(data).hexdigest(),
            "amends": amends, "decision": decision, "seen": seen, "time": []}


def _seal_lines(repo: _Repo, entries: Sequence[Dict[str, Any]], path: str, head: str = "HEAD",
                commit: Optional[str] = None, amends: Optional[str] = None,
                seen: Optional[str] = None) -> List[Dict[str, Any]]:
    """The manifest line(s) ``seal`` writes for *path* (repository-relative), read at *commit* or,
    by default, at the path's landing commit on HEAD's parent-one chain.

    Without *amends*: one prereg line (a path is sealed once). With *amends* (PLAN K9): a line id
    of *entries*, else a repository-relative path, which names that path's newest line, or, where
    the path has none, gets its default-seal line (the landing rule) written before the amendment.
    An amendment line's ``decision`` is read from its bytes: ``own`` when it holds a fence other than
    the one in effect for its lineage, else ``inherits``; ``seen`` is required."""
    h = repo.commit(head)
    _plain_path(path)
    if amends is None:
        held = next((e for e in entries if e["kind"] == "prereg" and e["path"] == path), None)
        if held is not None:
            raise _Refusal("MANIFEST_INVALID", f"{path} is sealed already, by line {held['id']}; a later seal of it "
                                               f"is an amendment (--amends {held['id']})")
        return [_sealed_line(repo, entries, path, h, commit, "prereg", seen=seen)]
    if seen is None:
        raise _Refusal("MANIFEST_SEEN_MISSING", "an amendment line states what had been seen: name --seen, one of "
                                                + ", ".join(repr(s) for s in _core._SEEN))
    new: List[Dict[str, Any]] = []
    target = next((e for e in entries if e["id"] == amends), None)
    if target is None:
        named = [e for e in entries if e["path"] == amends]
        if named:
            target = named[-1]
        else:
            target = _sealed_line(repo, entries, amends, h, None, "prereg")
            new.append(target)
    if target["kind"] == "withdrawn":
        raise _Refusal("MANIFEST_INVALID", f"line {target['id']} is withdrawn, and nothing amends a withdrawn line")
    new.append(_sealed_line(repo, [*entries, *new], path, h, commit, "amendment", amends=target["id"], seen=seen))
    return new
