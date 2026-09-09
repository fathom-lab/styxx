# -*- coding: utf-8 -*-
"""The recorder: a pytest plugin that turns every call into a `styxx.v8` entrypoint into a record.

Built to `styxx/v8/INTERFACES_layer2.md` section 10. Loaded as `-p conformance.v8.recorder` by
`gen_vectors.py`; appends one JSON line per recorded call to `$STYXX_V8_RECORDER_OUT`. Nothing
under `styxx/v8/` imports this module, and nothing here changes a verdict: every wrapper calls the
original with the arguments it was given and returns or re-raises exactly what it produced.

What is recorded, and what is not:

* One vector is one call into one entrypoint, with the ARGUMENTS as JSON and the outcome the
  implementation produced. A call made from inside another call of the SAME family is passed
  through unrecorded (a `verify_sth` inside `verify_inclusion`), so a family never carries its own
  internal recursion. A call into a DIFFERENT family is recorded (`distances.exact` inside
  `floor.pairwise`, `cert.check` inside `verify.diff`): those are the same function of the same
  arguments and a second implementation owes the same answer.
* An argument the record cannot carry is written as a `skip` line with its reason, and
  `gen_vectors.py` lists it under `index.unvectored.skipped`: a cert holding NaN, an infinity or an
  integer past 2**53 (no canonical bytes exist for it), or an argument whose canonical bytes do not
  parse back to what was passed.
* Calls made from a property test are NOT recorded, and every property test is named under
  `index.unvectored.sources`. A hypothesis example is a function of the hypothesis version and its
  database, so vectors recorded from one would move the set digest on a dependency upgrade with no
  change to `styxx.v8`. The set carries the cases the tests chose.

Numbers cross the record through RFC 8785, which renders `-1.0` as `-1`; a reader gets an integer
where Python had a float. Values are therefore compared by value across that round trip, with
`True` never equal to `1`. No entrypoint in the set distinguishes `1` from `1.0`.

`verify.diff`'s resolver is the one argument that is not data. It is carried as its KIND plus, for
a resolver that answers ids, the answers it gave: `{"kind": "none"}`, `{"kind": "unsupported"}`, or
`{"kind": "resolving", "answers": {id: cert sha256}}`. The kind is in the record because dropping
it collided two calls that behave differently — a `diff` with no resolver and a `diff` whose
resolver is refused before it is ever asked anything both looked like "resolver: null", and the
same vector id then carried two outcomes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from styxx.v8 import cert as certmod
from styxx.v8 import distances as distmod
from styxx.v8 import floor as floormod
from styxx.v8 import log as logmod
from styxx.v8 import merkle as merklemod
from styxx.v8 import verify as verifymod
from styxx.v8.jcs import canonical_bytes

from conformance.v8 import (
    BLOB_CAP,
    ENTRYPOINTS,
    REASON_KINDS,
    REASON_PREFIXES,
    RECORDER_OUT,
    reason_kind,
)

__all__ = ["REASON_KINDS", "REASON_PREFIXES", "reason_kind", "WRAPPERS", "Unrepresentable"]

_OUT = os.environ.get(RECORDER_OUT)
_DEPTH: Dict[str, int] = {family: 0 for family in set(ENTRYPOINTS.values())}
_PROPERTY_TESTS: set = set()


class Unrepresentable(Exception):
    """An argument the set cannot carry; listed under index.unvectored.skipped."""


# --------------------------------------------------------------------------- serialisation


def _same(a: Any, b: Any) -> bool:
    """Structural equality across a canonical round trip.

    Type-strict except for numbers: JCS renders `1.0` as `1`, so an int and a float of equal
    value are the same value here. `True` is never `1` — bool is checked first.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(a.keys()) == sorted(b.keys()) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    return a == b


def _why(exc: BaseException) -> str:
    """The lab's word for why, never the interpreter's: a CPython message is not stable."""
    if isinstance(exc, ValueError):
        return "holds a value no canonical serialisation can carry (NaN, an infinity, a lone surrogate, or an integer past 2**53)"
    if isinstance(exc, TypeError):
        return "holds a value JSON has no type for"
    return "is not representable as JSON"


def _jsonable(obj: Any, label: str) -> Any:
    """`obj`, if canonical bytes exist for it and parse back to the same value; else refused."""
    try:
        data = canonical_bytes(obj)
    except (TypeError, ValueError) as exc:
        raise Unrepresentable("%s %s" % (label, _why(exc)))
    back = json.loads(data.decode("utf-8"))
    if not _same(back, obj):
        raise Unrepresentable("%s does not survive a canonical round trip" % label)
    return obj


def _blob(obj: Any, label: str, store: Dict[str, str]) -> str:
    """Register `obj` as a blob and return its sha256. The blob's bytes are its canonical bytes."""
    _jsonable(obj, label)
    data = canonical_bytes(obj)
    if len(data) > BLOB_CAP:
        raise Unrepresentable("%s is %d bytes, past the blob cap" % (label, len(data)))
    sha = hashlib.sha256(data).hexdigest()
    store[sha] = base64.b64encode(data).decode("ascii")
    return sha


def _hex(raw: Any, label: str) -> str:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise Unrepresentable("%s is a %s, not bytes" % (label, type(raw).__name__))
    return bytes(raw).hex()


def _hexes(seq: Any, label: str) -> List[str]:
    try:
        items = list(seq)
    except TypeError:
        raise Unrepresentable("%s is not iterable" % label)
    return [_hex(x, "%s[%d]" % (label, i)) for i, x in enumerate(items)]


def _int(n: Any, label: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise Unrepresentable("%s is a %s, not an int" % (label, type(n).__name__))
    return n


# --------------------------------------------------------------------------- the record file


def _source() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]


def _write(record: dict) -> None:
    if _OUT is None:
        return
    with open(_OUT, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")


def _skip(entrypoint: str, why: str) -> None:
    _write({"source": _source(), "entrypoint": entrypoint, "skip": why})


def _emit(entrypoint: str, inputs: dict, blobs: Dict[str, str], expect: dict) -> None:
    _write(
        {
            "source": _source(),
            "family": ENTRYPOINTS[entrypoint],
            "entrypoint": entrypoint,
            "inputs": inputs,
            "blobs": blobs,
            "expect": expect,
        }
    )


def _refused(exc: BaseException) -> dict:
    """A refusal pins the exception TYPE and nothing else: a message tail is not a contract."""
    return {"outcome": "refused", "error": type(exc).__name__}


# --------------------------------------------------------------------------- the wrapper engine


class _Recorded:
    """One wrapped entrypoint. `prepare(args, kwargs)` -> (inputs, blobs); `describe(value)` ->
    expect. Both may raise `Unrepresentable`, which turns the call into a skip line."""

    def __init__(self, module, attr: str, entrypoint: str, prepare, describe):
        self.module = module
        self.attr = attr
        self.entrypoint = entrypoint
        self.family = ENTRYPOINTS[entrypoint]
        self.prepare = prepare
        self.describe = describe
        self.original = getattr(module, attr)

    def __call__(self, *args, **kwargs):
        if _OUT is None or _DEPTH[self.family] or _source() in _PROPERTY_TESTS:
            return self.original(*args, **kwargs)
        _DEPTH[self.family] += 1
        try:
            problem: Optional[str] = None
            inputs: Any = None
            blobs: Dict[str, str] = {}
            try:
                inputs, blobs = self.prepare(args, kwargs)
            except Unrepresentable as exc:
                problem = str(exc)
            try:
                value = self.original(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - a refusal is an outcome
                if problem is None:
                    _emit(self.entrypoint, inputs, blobs, _refused(exc))
                else:
                    _skip(self.entrypoint, problem)
                raise
            if problem is None:
                try:
                    expect = self.describe(value, blobs)
                except Unrepresentable as exc:
                    problem = str(exc)
                else:
                    _emit(self.entrypoint, inputs, blobs, expect)
            if problem is not None:
                _skip(self.entrypoint, problem)
            return value
        finally:
            _DEPTH[self.family] -= 1


# --------------------------------------------------------------------------- cert


def _prepare_check(args, kwargs):
    cert = kwargs["cert"] if "cert" in kwargs else args[0]
    blobs: Dict[str, str] = {}
    if not isinstance(cert, dict):
        raise Unrepresentable("check was called on a %s, not a dict" % type(cert).__name__)
    return {"cert": _blob(cert, "cert", blobs)}, blobs


def _kind(reason: str) -> str:
    """Classify one `cert.check` reason. Refuses to carry a reason outside the taxonomy."""
    kind = reason_kind(reason)
    if kind is None:
        raise Unrepresentable(
            "cert.check emitted a reason the taxonomy does not classify: %r" % reason
        )
    return kind


def _describe_check(value, _blobs):
    return {
        "outcome": "check",
        "ok": bool(value.ok),
        "type": value.type,
        "id": value.id,
        "reason_kinds": sorted(_kind(r) for r in value.reasons),
    }


# --------------------------------------------------------------------------- merkle


def _prepare_merkle_root(args, kwargs):
    leaves = kwargs["leaf_hashes"] if "leaf_hashes" in kwargs else args[0]
    return {"op": "root", "leaves": _hexes(leaves, "leaves")}, {}


def _describe_merkle_root(value, _blobs):
    return {"outcome": "root", "root": _hex(value, "root")}


def _prepare_inclusion_proof(args, kwargs):
    leaves = kwargs["leaf_hashes"] if "leaf_hashes" in kwargs else args[0]
    index = kwargs["index"] if "index" in kwargs else args[1]
    size = kwargs.get("tree_size", args[2] if len(args) > 2 else None)
    return (
        {
            "op": "inclusion_proof",
            "leaves": _hexes(leaves, "leaves"),
            "index": _int(index, "index"),
            "tree_size": None if size is None else _int(size, "tree_size"),
        },
        {},
    )


def _describe_proof(value, _blobs):
    return {"outcome": "proof", "path": _hexes(value, "path")}


def _prepare_consistency_proof(args, kwargs):
    leaves = kwargs["leaf_hashes"] if "leaf_hashes" in kwargs else args[0]
    first = kwargs["first"] if "first" in kwargs else args[1]
    second = kwargs.get("second", args[2] if len(args) > 2 else None)
    return (
        {
            "op": "consistency_proof",
            "leaves": _hexes(leaves, "leaves"),
            "first": _int(first, "first"),
            "second": None if second is None else _int(second, "second"),
        },
        {},
    )


def _positional(args, kwargs, names):
    out = []
    for i, name in enumerate(names):
        if name in kwargs:
            out.append(kwargs[name])
        elif len(args) > i:
            out.append(args[i])
        else:
            raise Unrepresentable("argument %s was not passed" % name)
    return out


def _prepare_merkle_verify_inclusion(args, kwargs):
    leaf, index, size, proof, root = _positional(
        args, kwargs, ("leaf_hash", "index", "tree_size", "proof", "expected_root")
    )
    return (
        {
            "op": "verify_inclusion",
            "leaf_hash": _hex(leaf, "leaf_hash"),
            "index": _int(index, "index"),
            "tree_size": _int(size, "tree_size"),
            "proof": _hexes(proof, "proof"),
            "expected_root": _hex(root, "expected_root"),
        },
        {},
    )


def _prepare_merkle_verify_consistency(args, kwargs):
    first, second, first_root, second_root, proof = _positional(
        args, kwargs, ("first", "second", "first_root", "second_root", "proof")
    )
    return (
        {
            "op": "verify_consistency",
            "first": _int(first, "first"),
            "second": _int(second, "second"),
            "first_root": _hex(first_root, "first_root"),
            "second_root": _hex(second_root, "second_root"),
            "proof": _hexes(proof, "proof"),
        },
        {},
    )


def _describe_bool(value, _blobs):
    if not isinstance(value, bool):
        raise Unrepresentable("expected a bool, got a %s" % type(value).__name__)
    return {"outcome": "verified", "ok": value}


# --------------------------------------------------------------------------- sth


def _prepare_verify_sth(args, kwargs):
    sth, public = _positional(args, kwargs, ("sth", "log_public"))
    blobs: Dict[str, str] = {}
    return (
        {"op": "verify_sth", "sth": _blob(sth, "sth", blobs), "log_public": _hex(public, "log_public")},
        blobs,
    )


def _prepare_verify_inclusion(args, kwargs):
    proof, sth, public = _positional(args, kwargs, ("proof", "sth", "log_public"))
    blobs: Dict[str, str] = {}
    return (
        {
            "op": "verify_inclusion",
            "proof": _blob(proof, "proof", blobs),
            "sth": _blob(sth, "sth", blobs),
            "log_public": _hex(public, "log_public"),
        },
        blobs,
    )


def _prepare_verify_consistency(args, kwargs):
    sth_m, sth_n, proof, public = _positional(args, kwargs, ("sth_m", "sth_n", "proof", "log_public"))
    blobs: Dict[str, str] = {}
    return (
        {
            "op": "verify_consistency",
            "sth_m": _blob(sth_m, "sth_m", blobs),
            "sth_n": _blob(sth_n, "sth_n", blobs),
            "proof": _blob(proof, "proof", blobs),
            "log_public": _hex(public, "log_public"),
        },
        blobs,
    )


def _describe_ok_reason(value, _blobs):
    ok, reason = value
    return {"outcome": "verified", "ok": bool(ok), "reason": str(reason)}


# --------------------------------------------------------------------------- distances


def _distance_prepare(fn: str, names):
    def prepare(args, kwargs):
        values = _positional(args, kwargs, names)
        blobs: Dict[str, str] = {}
        inputs: dict = {"fn": fn, "a": _blob(values[0], "a", blobs), "b": _blob(values[1], "b", blobs)}
        if fn == "exact":
            inputs["roles"] = _jsonable(_roles_json(values[2]), "roles")
        if fn == "lens":
            inputs["n_layers"] = _int(values[2], "n_layers")
        return inputs, blobs

    return prepare


def _roles_json(roles: Any) -> Any:
    """`roles` may be a mapping or the battery's items list; both are JSON already."""
    if isinstance(roles, dict):
        return {str(k): v for k, v in roles.items()}
    return list(roles)


def _describe_number(value, _blobs):
    if isinstance(value, tuple):
        distance, flips = value
        return {"outcome": "value", "value": float(distance), "anchor_flips": _int(flips, "anchor_flips")}
    return {"outcome": "value", "value": float(value)}


# --------------------------------------------------------------------------- floor


def _prepare_pairwise(args, kwargs):
    runs = kwargs["run_bodies"] if "run_bodies" in kwargs else args[0]
    channel = kwargs["channel"] if "channel" in kwargs else (args[1] if len(args) > 1 else None)
    roles = kwargs.get("roles")
    blobs: Dict[str, str] = {}
    if not isinstance(channel, str):
        raise Unrepresentable("channel is a %s, not a str" % type(channel).__name__)
    return (
        {
            "op": "pairwise",
            "runs": _blob(list(runs), "run_bodies", blobs),
            "channel": channel,
            "roles": None if roles is None else _jsonable(_roles_json(roles), "roles"),
        },
        blobs,
    )


def _describe_block(value, _blobs):
    if value is None:
        return {"outcome": "block", "block": None}
    return {"outcome": "block", "block": _jsonable(dict(value), "floor block")}


def _prepare_decide(args, kwargs):
    distance, floor_value, confirmation = _positional(args, kwargs, ("distance", "floor", "confirmation"))
    return (
        {
            "distance": _jsonable(distance, "distance"),
            "floor": _jsonable(floor_value, "floor"),
            "confirmation": _jsonable(confirmation, "confirmation"),
            "covered": _jsonable(kwargs.get("covered", True), "covered"),
            "skew": _jsonable(kwargs.get("skew", False), "skew"),
        },
        {},
    )


def _describe_verdict(value, _blobs):
    return {"outcome": "verdict", "verdict": str(value)}


# --------------------------------------------------------------------------- exit


class _ResolverProxy:
    """Records every id `verify.diff` looks up and what the caller's resolver answered.

    A plain callable: `verify._resolve` dispatches on Mapping, then on `find`/`cert`, then on
    callability, so this lands in the callable branch and returns exactly what the real resolver
    would have. `verify._resolve` is the same dispatch the entrypoint performs on the original.
    """

    def __init__(self, inner):
        self.inner = inner
        self.seen: Dict[str, Any] = {}

    def __call__(self, cert_id):
        got = verifymod._resolve(self.inner, cert_id)
        self.seen[cert_id] = got
        return got


def resolver_kind(resolver: Any) -> str:
    """Which branch of `verify._resolve` this argument lands in, decided without calling it.

    `None` performs no resolution; a mapping, a `find`/`cert` pair and a plain callable all
    answer ids; anything else is refused with a TypeError. The kind is part of the record because
    `None` and a resolver that answers nothing are different arguments with different behaviour,
    and a call that refuses before any lookup carries no answers to tell them apart by.
    """
    if resolver is None:
        return "none"
    if isinstance(resolver, Mapping):
        return "resolving"
    find, get = getattr(resolver, "find", None), getattr(resolver, "cert", None)
    if (callable(find) and callable(get)) or callable(resolver):
        return "resolving"
    return "unsupported"


def _prepare_diff(args, kwargs):
    cert_a, cert_b, resolver = _positional(args, kwargs, ("cert_a", "cert_b", "resolver"))
    blobs: Dict[str, str] = {}
    inputs = {
        "a": _blob(cert_a, "cert_a", blobs) if isinstance(cert_a, dict) else None,
        "b": _blob(cert_b, "cert_b", blobs) if isinstance(cert_b, dict) else None,
    }
    if inputs["a"] is None or inputs["b"] is None:
        raise Unrepresentable("diff was called on something other than two cert dicts")
    kind = resolver_kind(resolver)
    # `answers` is filled in by _RecordedDiff once the call has made its lookups.
    inputs["resolver"] = {"kind": kind, "answers": {}} if kind == "resolving" else {"kind": kind}
    return inputs, blobs


def _describe_outcome(value, blobs):
    body = _jsonable(value.result_body, "result body")
    sha = _blob(body, "result body", blobs)
    return {
        "outcome": "outcome",
        "exit_code": _int(value.exit_code, "exit_code"),
        "verdict": str(value.verdict),
        "mismatched": list(value.mismatched),
        "result_sha256": sha,
    }


class _RecordedDiff(_Recorded):
    """`verify.diff` needs the resolver's answers in the record, so it substitutes a proxy."""

    def __call__(self, *args, **kwargs):
        if _OUT is None or _DEPTH[self.family] or _source() in _PROPERTY_TESTS:
            return self.original(*args, **kwargs)
        try:
            cert_a, cert_b, resolver = _positional(args, kwargs, ("cert_a", "cert_b", "resolver"))
        except Unrepresentable as exc:
            _skip(self.entrypoint, str(exc))
            return self.original(*args, **kwargs)
        proxy = _ResolverProxy(resolver) if resolver is not None else None
        _DEPTH[self.family] += 1
        try:
            problem: Optional[str] = None
            inputs: Any = None
            blobs: Dict[str, str] = {}
            try:
                inputs, blobs = _prepare_diff((cert_a, cert_b, resolver), {})
            except Unrepresentable as exc:
                problem = str(exc)
            def answers() -> None:
                """Fold the proxy's answers into the record. A refusal keeps the partial map:
                the replay makes the same lookups in the same order and meets the same refusal."""
                if problem is not None or proxy is None:
                    return
                if inputs["resolver"]["kind"] != "resolving":
                    return
                resolved: Dict[str, Any] = {}
                for cert_id, got in sorted(proxy.seen.items()):
                    if isinstance(got, dict):
                        resolved[cert_id] = _blob(got, "resolved %s" % cert_id, blobs)
                inputs["resolver"]["answers"] = resolved

            try:
                value = self.original(cert_a, cert_b, proxy)
            except Exception as exc:  # noqa: BLE001
                if problem is None:
                    try:
                        answers()
                    except Unrepresentable as inner:
                        _skip(self.entrypoint, str(inner))
                    else:
                        _emit(self.entrypoint, inputs, blobs, _refused(exc))
                else:
                    _skip(self.entrypoint, problem)
                raise
            if problem is None:
                try:
                    answers()
                    expect = _describe_outcome(value, blobs)
                except Unrepresentable as exc:
                    problem = str(exc)
                else:
                    _emit(self.entrypoint, inputs, blobs, expect)
            if problem is not None:
                _skip(self.entrypoint, problem)
            return value
        finally:
            _DEPTH[self.family] -= 1


# --------------------------------------------------------------------------- installation

WRAPPERS: List[_Recorded] = [
    _Recorded(certmod, "check", "cert.check", _prepare_check, _describe_check),
    _Recorded(merklemod, "root", "merkle.root", _prepare_merkle_root, _describe_merkle_root),
    _Recorded(merklemod, "inclusion_proof", "merkle.inclusion_proof", _prepare_inclusion_proof, _describe_proof),
    _Recorded(merklemod, "consistency_proof", "merkle.consistency_proof", _prepare_consistency_proof, _describe_proof),
    _Recorded(merklemod, "verify_inclusion", "merkle.verify_inclusion", _prepare_merkle_verify_inclusion, _describe_bool),
    _Recorded(merklemod, "verify_consistency", "merkle.verify_consistency", _prepare_merkle_verify_consistency, _describe_bool),
    _Recorded(logmod, "verify_sth", "log.verify_sth", _prepare_verify_sth, _describe_ok_reason),
    _Recorded(logmod, "verify_inclusion", "log.verify_inclusion", _prepare_verify_inclusion, _describe_ok_reason),
    _Recorded(logmod, "verify_consistency", "log.verify_consistency", _prepare_verify_consistency, _describe_ok_reason),
    _Recorded(distmod, "exact", "distances.exact", _distance_prepare("exact", ("a_items", "b_items", "roles")), _describe_number),
    _Recorded(distmod, "seqlp", "distances.seqlp", _distance_prepare("seqlp", ("a_items", "b_items")), _describe_number),
    _Recorded(distmod, "topk", "distances.topk", _distance_prepare("topk", ("a_items", "b_items")), _describe_number),
    _Recorded(distmod, "resid", "distances.resid", _distance_prepare("resid", ("a_profiles", "b_profiles")), _describe_number),
    _Recorded(distmod, "lens", "distances.lens", _distance_prepare("lens", ("a_layers", "b_layers", "n_layers")), _describe_number),
    _Recorded(floormod, "pairwise", "floor.pairwise", _prepare_pairwise, _describe_block),
    _Recorded(floormod, "decide", "floor.decide", _prepare_decide, _describe_verdict),
    _RecordedDiff(verifymod, "diff", "verify.diff", _prepare_diff, _describe_outcome),
]

for _w in WRAPPERS:
    setattr(_w.module, _w.attr, _w)


def pytest_collection_modifyitems(session, config, items) -> None:
    """Rebind a test module's direct-name imports, and name the property tests.

    A plugin named by `-p` is imported before any test module, so `from styxx.v8.distances import
    exact` already binds the wrapper. This is the fallback for a module imported earlier, and it is
    where the hypothesis-driven tests are collected: their calls are passed through unrecorded.
    """
    seen: List[Any] = []
    for item in items:
        function = getattr(item, "function", None)
        if function is not None and getattr(function, "is_hypothesis_test", False):
            if item.nodeid not in _PROPERTY_TESTS:
                _PROPERTY_TESTS.add(item.nodeid)
                _write(
                    {
                        "source": item.nodeid,
                        "entrypoint": "*",
                        "skip": "a property test: its examples are a function of the hypothesis "
                        "version and database, not of styxx.v8",
                    }
                )
        module = getattr(item, "module", None)
        if module is None or any(module is m for m in seen):
            continue
        seen.append(module)
        for wrapper in WRAPPERS:
            if getattr(module, wrapper.attr, None) is wrapper.original:
                setattr(module, wrapper.attr, wrapper)
