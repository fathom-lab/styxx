# -*- coding: utf-8 -*-
"""The recorder: a pytest plugin that turns every call into ``styxx.sworn`` into one record.

Built to ``papers/sworn/SPEC_sworn_conformance_vectors_v01_2026_09_05.md`` (C2, C3, C5, C10).
This is where the ambient world enters: the clock is pinned, git is read, the environment is
written. ``styxx.sworn`` never imports it. Loaded as ``-p conformance.sworn.recorder`` by
``gen_vectors.py``; appends one JSON line per top-level call to ``$SWORN_RECORDER_OUT``.

Nothing here changes a verdict. Every wrapper calls the original with the same arguments and
re-raises what it raised; a call made from inside another wrapped call (``verify_receipt`` calling
``verify``, ``verify`` calling ``load_sidecar`` and ``Manifest.from_dict``) is passed through
unrecorded, so a vector is one entrypoint the test itself called.

An input the record cannot carry is written as a ``skip`` line with its reason (C5): a manifest
holding a value no JSON text can represent, a tree that is the repository this set lives in or
whose embeddable blobs exceed one mebibyte, an object that does not survive a JSON round trip.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from styxx import sworn

from conformance.sworn import CLOCK

EMBED_CAP = 1024 * 1024
ROOT = Path(sworn.__file__).resolve().parents[1]
_HEX = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")

# C3: the clock and the git dates are pinned at generation. Pinning changes no verdict
# (tests/test_sworn.py::TestDoctrine::test_the_verdict_is_a_function_of_bytes_not_of_cwd_or_clock);
# it makes every minted_at, captured_at and fixture commit id the same on every run.
sworn._now = lambda: CLOCK
os.environ["GIT_AUTHOR_DATE"] = CLOCK
os.environ["GIT_COMMITTER_DATE"] = CLOCK

_OUT = os.environ.get("SWORN_RECORDER_OUT")
_DEPTH = [0]
_ORIGINAL = {
    "verify": sworn.verify,
    "to_sidecar": sworn.to_sidecar,
    "load_sidecar": sworn.load_sidecar,
    "verify_receipt": sworn.verify_receipt,
}
_ORIGINAL_FROM_DICT = sworn.Manifest.__dict__["from_dict"]


class Unrepresentable(Exception):
    """An input or outcome the set cannot carry; listed under index.unvectored.skipped."""


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _source() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]


def _same(a: Any, b: Any) -> bool:
    """Type-strict structural equality: a tuple is not a list, 1 is not True, 1 is not 1.0."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _why_unrepresentable(e: BaseException) -> str:
    """The lab's word for why, never the interpreter's.

    A CPython exception message is not stable across versions — py3.9 through py3.11 refused this
    set while py3.12 accepted it, on nothing but the prose inside two `skipped` entries. The set
    pins what the format decides, so the reason is classified from the exception type here and the
    message is left out.
    """
    if isinstance(e, UnicodeEncodeError):
        return "carries text that is not encodable as UTF-8 (a lone surrogate)"
    if isinstance(e, ValueError):
        return "carries a value no canonical serialisation can hold (NaN or an infinity)"
    if isinstance(e, TypeError):
        return "carries a value JSON has no type for"
    return "is not JSON-representable"


def _jsonable(obj: Any, label: str) -> Any:
    """The object, if a JSON text can carry it and parse back to the same thing; else refused."""
    try:
        text = json.dumps(obj, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise Unrepresentable("%s %s" % (label, _why_unrepresentable(e)))
    if not _same(json.loads(text), obj):
        raise Unrepresentable("%s does not survive a JSON round trip" % label)
    return obj


def _manifest_input(m: Any) -> Optional[dict]:
    """C2: ``Manifest.core()`` plus the declared digest when the object carried one, so that
    ``intact()`` is reproduced exactly, a tampered digest included. Never ``to_dict()``."""
    if m is None:
        return None
    if not isinstance(m, sworn.Manifest):
        raise Unrepresentable("manifest is a %s, not a Manifest" % type(m).__name__)
    obj = m.core()
    declared = getattr(m, "declared_digest", None)
    if declared is not None:
        obj = dict(obj)
        obj["digest"] = declared
    return _jsonable(obj, "manifest")


def _git(repo: Any, *args: str):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    return r.returncode, r.stdout


def _entries_from_git(repo: Any, commit: str) -> Dict[str, dict]:
    rc, out = _git(repo, "ls-tree", "-r", "-t", "-l", "-z", "--full-tree", commit)
    if rc != 0:
        raise Unrepresentable("git ls-tree failed at %s" % commit[:12])
    entries: Dict[str, dict] = {}
    total = 0
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        data = None
        if kind == b"blob" and mode != b"120000":
            rc, data = _git(repo, "cat-file", "blob", oid.decode("ascii"))
            if rc != 0:
                raise Unrepresentable("git cat-file failed for %s" % path.decode("utf-8", "replace"))
            total += len(data)
            if total > EMBED_CAP:
                raise Unrepresentable("the tree's embeddable blobs exceed one mebibyte (C5)")
        entries[path.decode("utf-8")] = {
            "mode": mode.decode("ascii"),
            "size": int(size) if size.isdigit() else None,
            "sha256": sworn._sha256(data) if data is not None else None,
            "bytes": _b64(data) if data is not None else None,
        }
    return entries


def _tree_input(tree: Any, handle_commit: Any) -> Optional[dict]:
    """C2/C10: a snapshot with modes, taken AFTER the call at the commit the verdict resolved
    against (the handle's commit as ``verify()`` left it), with the handle's own commit as it was
    before the call recorded beside it."""
    if tree is None:
        return None
    effective = getattr(tree, "commit", None)
    if isinstance(tree, sworn.MemoryTree):
        snap = sworn.SnapshotTree.from_memory(tree)
        entries = {p: dict(e, bytes=_b64(e["bytes"])) for p, e in snap.entries.items()}
        snapshot_commit = effective if isinstance(effective, str) and _HEX.fullmatch(effective) else None
    elif isinstance(tree, sworn.SnapshotTree):
        entries = {p: dict(e, bytes=_b64(e["bytes"]) if e.get("bytes") is not None else None)
                   for p, e in tree.entries.items()}
        snapshot_commit = tree.snapshot_commit
    elif isinstance(tree, sworn.GitTree):
        repo = tree.repo
        if repo is not None and Path(repo).resolve() == ROOT:
            raise Unrepresentable("the tree is the repository this set lives in (C5)")
        entries, snapshot_commit = {}, None
        if repo is not None and isinstance(effective, str) and _HEX.fullmatch(effective):
            rc, out = _git(repo, "cat-file", "-t", effective)
            if rc == 0 and out.strip() == b"commit":
                snapshot_commit = effective
                entries = _entries_from_git(repo, effective)
    else:
        raise Unrepresentable("tree handle of an unknown type %s" % type(tree).__name__)
    total = sum(len(e["bytes"] or "") for e in entries.values())
    if total * 3 // 4 > EMBED_CAP:
        raise Unrepresentable("the tree's embeddable blobs exceed one mebibyte (C5)")
    return {"snapshot_commit": _jsonable(snapshot_commit, "snapshot_commit"),
            "handle_commit": _jsonable(handle_commit, "handle_commit"), "entries": entries}


def _document_input(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, (bytes, bytearray)):
        raise Unrepresentable("document is a %s, not bytes" % type(raw).__name__)
    return _b64(bytes(raw))


def _write(record: dict) -> None:
    if _OUT is None:
        return
    with open(_OUT, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")


def _skip(where: str, why: str) -> None:
    _write({"source": _source(), "where": where, "skip": why})


def _floor_and_observer(core: dict):
    cov = core.get("coverage") or {}
    floor = {"sworn_total": cov.get("sworn_total"), "narrative_sentences": cov.get("narrative_sentences"),
             "sentence_share": cov.get("sentence_share")}
    observer = {"diff_claim_sentences": cov.get("diff_claim_sentences"),
                "diff_claim_share": cov.get("diff_claim_share"),
                "unsworn_claims": len(cov.get("unsworn_claims") or []),
                "claimdetect_version": cov.get("claimdetect_version")}
    return floor, observer


def _core_outcome(core: dict) -> dict:
    portable = {k: v for k, v in core.items() if k not in ("verifier", "coverage")}
    try:
        text = sworn._jcs(portable)
        text.encode("utf-8")
    except (TypeError, ValueError) as e:
        raise Unrepresentable("the core is not JCS-serialisable: %s"
                              % _why_unrepresentable(e))
    floor, observer = _floor_and_observer(core)
    return {"outcome": "core", "core": portable, "core_jcs": text, "floor": floor, "observer": observer}


def _refused(where: str, e: SystemExit) -> dict:
    return {"outcome": "refused", "where": where, "message": str(e.code)}


# --------------------------------------------------------------------------- the wrappers

def _guarded(fn):
    def run(*args, **kwargs):
        if _DEPTH[0] or _OUT is None:
            return fn.original(*args, **kwargs)
        _DEPTH[0] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _DEPTH[0] -= 1
    run.original = fn.original
    run.__name__ = fn.original.__name__
    run.__doc__ = fn.original.__doc__
    return run


def _rec_verify(raw=None, sidecar=None, *, name="", manifest=None, tree=None, commit=None):
    original = _ORIGINAL["verify"]
    where = "verify"
    handle_commit = getattr(tree, "commit", None) if tree is not None else None
    pre: Dict[str, Any] = {}
    problem: Optional[str] = None
    try:
        pre["name"] = _jsonable(name, "name")
        pre["commit"] = _jsonable(commit, "commit")
        pre["document"] = _document_input(raw)
        pre["sidecar"] = _jsonable(sidecar, "sidecar")
        pre["manifest"] = _manifest_input(manifest)
    except Unrepresentable as e:
        problem = str(e)
    try:
        core = original(raw, sidecar, name=name, manifest=manifest, tree=tree, commit=commit)
    except SystemExit as e:
        refused = _refused(where, e)
        _finish(where, "sidecar" if sidecar is not None else "inline", pre, problem, tree, handle_commit,
                lambda: refused)
        raise
    _finish(where, "sidecar" if sidecar is not None else "inline", pre, problem, tree, handle_commit,
            lambda: _core_outcome(core))
    return core


_rec_verify.original = _ORIGINAL["verify"]


def _finish(where, mode, pre, problem, tree, handle_commit, outcome_fn) -> None:
    if problem is None:
        try:
            pre["tree"] = _tree_input(tree, handle_commit)
            outcome = outcome_fn()
        except Unrepresentable as e:
            problem = str(e)
    if problem is not None:
        _skip(where, problem)
        return
    _write({"source": _source(), "where": where, "mode": mode, "inputs": pre, "outcome": outcome})


def _rec_to_sidecar(raw, name, commit=None, manifest=None):
    original = _ORIGINAL["to_sidecar"]
    where = "to_sidecar"
    pre: Dict[str, Any] = {}
    problem: Optional[str] = None
    try:
        pre["name"] = _jsonable(name, "name")
        pre["commit"] = _jsonable(commit, "commit")
        pre["document"] = _document_input(raw)
        pre["manifest"] = _manifest_input(manifest)
    except Unrepresentable as e:
        problem = str(e)
    try:
        side = original(raw, name, commit, manifest)
    except SystemExit as e:
        refused = _refused(where, e)
        _finish(where, "canon", pre, problem, None, None, lambda: refused)
        raise

    def outcome():
        obj = _jsonable(side, "sidecar")
        try:
            text = sworn._jcs(obj)
            text.encode("utf-8")
        except (TypeError, ValueError) as e:
            raise Unrepresentable("the sidecar is not JCS-serialisable: %s"
                                  % _why_unrepresentable(e))
        return {"outcome": "sidecar", "sidecar": obj, "sidecar_jcs": text}

    _finish(where, "canon", pre, problem, None, None, outcome)
    return side


_rec_to_sidecar.original = _ORIGINAL["to_sidecar"]


def _rec_load_sidecar(obj):
    original = _ORIGINAL["load_sidecar"]
    where = "load_sidecar"
    pre: Dict[str, Any] = {}
    problem: Optional[str] = None
    try:
        pre["sidecar"] = _jsonable(obj, "sidecar")
    except Unrepresentable as e:
        problem = str(e)
    try:
        out = original(obj)
    except SystemExit as e:
        refused = _refused(where, e)
        _finish(where, "load", pre, problem, None, None, lambda: refused)
        raise
    _finish(where, "load", pre, problem, None, None, lambda: {"outcome": "accepted"})
    return out


_rec_load_sidecar.original = _ORIGINAL["load_sidecar"]


def _rec_verify_receipt(receipt, raw=None, sidecar=None, *, manifest=None, tree=None):
    original = _ORIGINAL["verify_receipt"]
    where = "verify_receipt"
    handle_commit = getattr(tree, "commit", None) if tree is not None else None
    pre: Dict[str, Any] = {}
    problem: Optional[str] = None
    try:
        pre["receipt"] = _jsonable(receipt, "receipt")
        pre["document"] = _document_input(raw)
        pre["sidecar"] = _jsonable(sidecar, "sidecar")
        pre["manifest"] = _manifest_input(manifest)
    except Unrepresentable as e:
        problem = str(e)
    result = original(receipt, raw, sidecar, manifest=manifest, tree=tree)

    def outcome():
        check = {k: result.get(k) for k in ("status", "digest_match", "verdict_reproduces")}
        observer = {k: result.get(k) for k in ("coverage_reproduces", "same_verifier_build")}
        return {"outcome": "check", "check": _jsonable(check, "check"), "observer": observer}

    _finish(where, "receipt_check", pre, problem, tree, handle_commit, outcome)
    return result


_rec_verify_receipt.original = _ORIGINAL["verify_receipt"]


def _rec_from_dict(cls, d):
    where = "Manifest.from_dict"
    pre: Dict[str, Any] = {}
    problem: Optional[str] = None
    try:
        pre["manifest"] = _jsonable(d, "manifest")
    except Unrepresentable as e:
        problem = str(e)
    try:
        m = _ORIGINAL_FROM_DICT.__func__(cls, d)
    except SystemExit as e:
        refused = _refused(where, e)
        _finish(where, "manifest", pre, problem, None, None, lambda: refused)
        raise

    def outcome():
        state, rung = m.rung_status()
        return {"outcome": "manifest",
                "manifest": _jsonable({"digest": m.digest_or_none(), "spec": m.spec,
                                       "rung_status": [state, rung], "intact": bool(m.intact())},
                                      "manifest outcome")}

    _finish(where, "manifest", pre, problem, None, None, outcome)
    return m


_rec_from_dict.original = _ORIGINAL_FROM_DICT.__func__


def _guarded_from_dict(cls, d):
    if _DEPTH[0] or _OUT is None:
        return _ORIGINAL_FROM_DICT.__func__(cls, d)
    _DEPTH[0] += 1
    try:
        return _rec_from_dict(cls, d)
    finally:
        _DEPTH[0] -= 1


WRAPPERS = {
    "verify": _guarded(_rec_verify),
    "to_sidecar": _guarded(_rec_to_sidecar),
    "load_sidecar": _guarded(_rec_load_sidecar),
    "verify_receipt": _guarded(_rec_verify_receipt),
}

# Patch at import: a plugin named by -p is imported before any test module, so a test module's
# `from styxx.sworn import verify` binds the wrapper. The collection hook below is the fallback.
for _name, _wrapper in WRAPPERS.items():
    setattr(sworn, _name, _wrapper)
sworn.Manifest.from_dict = classmethod(_guarded_from_dict)


def pytest_collection_modifyitems(session, config, items) -> None:
    seen: List[Any] = []
    for item in items:
        module = getattr(item, "module", None)
        if module is None or any(module is m for m in seen):
            continue
        seen.append(module)
        for name, wrapper in WRAPPERS.items():
            if getattr(module, name, None) is _ORIGINAL[name]:
                setattr(module, name, wrapper)
