# -*- coding: utf-8 -*-
"""Replay one conformance vector through ``styxx.sworn``.

Built to ``papers/sworn/SPEC_sworn_conformance_vectors_v01_2026_09_05.md``. A function of the
vector and its blobs: no clock, no git, no filesystem. ``gen_vectors.py`` runs it over every vector
before the set is written (C6) and ``tests/test_sworn_conformance.py`` runs it over the committed
set. A second verifier does the same thing in its own language; this module is the reference for
what "the same thing" is, entrypoint by entrypoint.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional, Tuple

from styxx import sworn
from styxx.sworn import Manifest, SnapshotTree, load_sidecar, render, to_sidecar, verify, verify_receipt

PORTABLE_EXCLUDED = ("verifier", "coverage")


class BadBlob(Exception):
    """A blob that does not hash to its key, or a reference the store does not hold."""


def blob_bytes(blobs: Dict[str, str], sha: str) -> bytes:
    if sha not in blobs:
        raise BadBlob("blob %s is not in the store" % sha)
    data = base64.b64decode(blobs[sha], validate=True)
    if sworn._sha256(data) != sha:
        raise BadBlob("blob %s does not hash to its key" % sha)
    return data


def blob_json(blobs: Dict[str, str], sha: str) -> Any:
    return json.loads(blob_bytes(blobs, sha).decode("utf-8"))


def portable(core: dict) -> dict:
    """The core a second verifier reproduces: the verdict minus the build and the observer (C1)."""
    return {k: v for k, v in core.items() if k not in PORTABLE_EXCLUDED}


def core_sha256(core: dict) -> str:
    return sworn._sha256(sworn._jcs(portable(core)).encode("utf-8"))


def floor_of(core: dict) -> dict:
    cov = core.get("coverage") or {}
    return {"sworn_total": cov.get("sworn_total"), "narrative_sentences": cov.get("narrative_sentences"),
            "sentence_share": cov.get("sentence_share")}


def build_tree(blobs: Dict[str, str], t: Optional[dict]) -> Optional[SnapshotTree]:
    if t is None:
        return None
    entries = {}
    for path, e in t["entries"].items():
        data = blob_bytes(blobs, e["sha256"]) if e.get("sha256") else None
        entries[path] = {"mode": e["mode"], "size": e.get("size"), "sha256": e.get("sha256"), "bytes": data}
    return SnapshotTree(entries, t["snapshot_commit"], commit=t["handle_commit"])


def build_manifest(blobs: Dict[str, str], sha: Optional[str]) -> Optional[Manifest]:
    if sha is None:
        return None
    return Manifest.from_dict(blob_json(blobs, sha))


def _core_diff(got: dict, want_blob: Optional[dict]) -> str:
    if want_blob is None:
        return "core_sha256 differs and no core blob was found to diff against"
    for k in sorted(set(got) | set(want_blob)):
        if sworn._jcs(got.get(k)) != sworn._jcs(want_blob.get(k)):
            if k == "spans":
                for i, (a, b) in enumerate(zip(got.get("spans") or [], want_blob.get("spans") or [])):
                    if sworn._jcs(a) != sworn._jcs(b):
                        return "spans[%d] differs: got %s, vector has %s" % (i, sworn._jcs(a)[:300], sworn._jcs(b)[:300])
                return "spans differ in length: got %d, vector has %d" % (len(got.get("spans") or []), len(want_blob.get("spans") or []))
            return "%s differs: got %s, vector has %s" % (k, sworn._jcs(got.get(k))[:300], sworn._jcs(want_blob.get(k))[:300])
    return "core_sha256 differs but every key compares equal under jcs"


def _expect_core(vec: dict, blobs: Dict[str, str], core: dict) -> Tuple[str, str]:
    expect = vec["expect"]
    got = core_sha256(core)
    if got != expect["core_sha256"]:
        want = None
        if expect["core_sha256"] in blobs:
            want = json.loads(blob_bytes(blobs, expect["core_sha256"]).decode("utf-8"))
        return "fail", _core_diff(portable(core), want)
    if floor_of(core) != expect["floor"]:
        return "fail", "floor differs: got %r, vector has %r" % (floor_of(core), expect["floor"])
    return "pass", "core"


def _expect_refusal(vec: dict, where: str, e: SystemExit) -> Tuple[str, str]:
    expect = vec["expect"]
    if expect["outcome"] != "refused":
        return "fail", "%s refused (%s) where the vector expects %s" % (where, str(e.code)[:200], expect["outcome"])
    r = expect["refusal"]
    if r["where"] != where:
        return "fail", "refused in %s, vector says %s" % (where, r["where"])
    if r["match"] not in str(e.code):
        return "fail", "refusal %r does not contain %r" % (str(e.code)[:200], r["match"])
    return "pass", "refused " + r["code"]


def replay_vector(vec: dict, blobs: Dict[str, str]) -> Tuple[str, str]:
    """('pass' | 'fail' | 'skip', detail). Skips only a vector whose ``requires`` names ``git``."""
    if "git" in vec.get("requires", []):
        return "skip", "requires a live git object store"
    mode = vec["mode"]
    inputs = vec["inputs"]
    expect = vec["expect"]
    try:
        if mode in ("inline", "sidecar", "receipt_check"):
            tree = build_tree(blobs, inputs.get("tree"))
        else:
            tree = None
        if mode == "manifest":
            manifest = None
        else:
            try:
                manifest = build_manifest(blobs, inputs.get("manifest"))
            except SystemExit as e:
                return "fail", "the vector's manifest input is refused by Manifest.from_dict: %s" % str(e.code)[:200]
        doc = blob_bytes(blobs, inputs["document"]) if inputs.get("document") else None
        side = blob_json(blobs, inputs["sidecar"]) if inputs.get("sidecar") else None
    except BadBlob as e:
        return "fail", str(e)

    if mode == "inline":
        try:
            core = verify(doc, name=inputs["name"], manifest=manifest, tree=tree, commit=inputs["commit"])
        except SystemExit as e:
            return _expect_refusal(vec, "verify", e)
        if expect["outcome"] != "core":
            return "fail", "verify returned a core where the vector expects %s" % expect["outcome"]
        return _expect_core(vec, blobs, core)

    if mode == "sidecar":
        try:
            core = verify(sidecar=side, name=inputs["name"], manifest=manifest, tree=tree, commit=inputs["commit"])
        except SystemExit as e:
            return _expect_refusal(vec, "verify", e)
        if expect["outcome"] != "core":
            return "fail", "verify returned a core where the vector expects %s" % expect["outcome"]
        return _expect_core(vec, blobs, core)

    if mode == "canon":
        try:
            out = to_sidecar(doc, inputs["name"], inputs["commit"], manifest)
        except SystemExit as e:
            return _expect_refusal(vec, "to_sidecar", e)
        if expect["outcome"] != "sidecar":
            return "fail", "to_sidecar returned a sidecar where the vector expects %s" % expect["outcome"]
        got = sworn._sha256(sworn._jcs(out).encode("utf-8"))
        if got != expect["sidecar_sha256"]:
            return "fail", "sidecar_sha256 differs: got %s, vector has %s" % (got, expect["sidecar_sha256"])
        stored = blob_json(blobs, expect["sidecar"])
        if sworn._jcs(stored) != sworn._jcs(out):
            return "fail", "the stored sidecar blob does not equal the sidecar produced"
        if render(out) != doc:
            return "fail", "render(sidecar) is not byte-identical to the document"
        return "pass", "sidecar"

    if mode == "load":
        try:
            load_sidecar(side)
        except SystemExit as e:
            return _expect_refusal(vec, "load_sidecar", e)
        if expect["outcome"] != "accepted":
            return "fail", "load_sidecar accepted where the vector expects %s" % expect["outcome"]
        return "pass", "accepted"

    if mode == "manifest":
        try:
            m = Manifest.from_dict(blob_json(blobs, inputs["manifest"]))
        except SystemExit as e:
            return _expect_refusal(vec, "Manifest.from_dict", e)
        if expect["outcome"] != "manifest":
            return "fail", "Manifest.from_dict loaded where the vector expects %s" % expect["outcome"]
        state, rung = m.rung_status()
        got = {"digest": m.digest_or_none(), "spec": m.spec, "rung_status": [state, rung], "intact": bool(m.intact())}
        if got != expect["manifest"]:
            return "fail", "manifest outcome differs: got %r, vector has %r" % (got, expect["manifest"])
        return "pass", "manifest"

    if mode == "receipt_check":
        receipt = blob_json(blobs, inputs["receipt"])
        result = verify_receipt(receipt, doc, sidecar=side, manifest=manifest, tree=tree)
        got = {k: result.get(k) for k in ("status", "digest_match", "verdict_reproduces")}
        if expect["outcome"] != "check":
            return "fail", "verify_receipt returned where the vector expects %s" % expect["outcome"]
        if got != expect["check"]:
            return "fail", "check differs: got %r, vector has %r (%s)" % (got, expect["check"], result.get("note", "")[:200])
        return "pass", "check"

    return "fail", "unknown mode %r" % mode
