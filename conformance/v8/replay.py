# -*- coding: utf-8 -*-
"""Replay one conformance vector through the `styxx.v8` Python API.

Built to `styxx/v8/INTERFACES_layer2.md` section 10. This module is the reference for what a
second implementation does with the set, entrypoint by entrypoint: it holds no expectations of its
own, it rebuilds the arguments a vector carries, calls the entrypoint the vector names, and returns
the same `expect` shape the recorder wrote. `tests/test_v8_conformance.py` compares the two.

Nothing here imports `recorder.py`: importing the recorder installs its wrappers into `styxx.v8`,
and a replay must call the shipped functions, not a wrapper around them. The two share only the
`cert.check` reason taxonomy, which lives in `conformance/v8/__init__.py`.

Comparison is by canonical bytes, never by `==` on the parsed objects. RFC 8785 renders `1.0` as
`1`, so a float that made a round trip through the set comes back an integer; comparing the
canonical bytes of both sides puts them on the same footing. No entrypoint in the set distinguishes
`1` from `1.0` -- `cert.check` refuses a cert that would need the difference, and the distances are
compared after `rounded`.

Usage:
    python conformance/v8/replay.py                 # replay the committed set, print per family
    python conformance/v8/replay.py --id <hex>      # replay one vector and print both sides
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.v8 import cert as certmod  # noqa: E402
from styxx.v8 import distances as distmod  # noqa: E402
from styxx.v8 import floor as floormod  # noqa: E402
from styxx.v8 import log as logmod  # noqa: E402
from styxx.v8 import merkle as merklemod  # noqa: E402
from styxx.v8 import verify as verifymod  # noqa: E402
from styxx.v8.jcs import canonical_bytes  # noqa: E402

from conformance.v8 import ENTRYPOINTS, FAMILIES, reason_kind  # noqa: E402


class ReplayError(Exception):
    """The vector cannot be replayed: a blob is missing, an argument is malformed, or the
    implementation produced something the record has no shape for. Never a disagreement --
    a disagreement is a passing replay whose `expect` differs."""


class _UnsupportedResolver:
    """Not a mapping, not callable, no `find`/`cert`: `verify._resolve` refuses it.

    A second implementation reproduces a `resolver.kind == "unsupported"` vector by passing
    whatever its own dispatch refuses; the vector pins that such an argument is refused, not
    which Python type was used to get there.
    """


UNSUPPORTED_RESOLVER = _UnsupportedResolver()


# --------------------------------------------------------------------------- the set on disk


def vector_id(family: str, inputs: Any) -> str:
    """`sha256(canonical_bytes({"family", "inputs"}))`, the address of a vector."""
    return hashlib.sha256(canonical_bytes({"family": family, "inputs": inputs})).hexdigest()


def load_blobs(directory: Path = HERE) -> Dict[str, str]:
    path = directory / "blobs.json"
    if not path.exists():
        raise ReplayError("no blob store at %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(directory: Path = HERE) -> dict:
    path = directory / "index.json"
    if not path.exists():
        raise ReplayError("no index at %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_vectors(directory: Path = HERE) -> Dict[str, List[dict]]:
    """`{family: [vector, ...]}` for every family file present, in `FAMILIES` order."""
    out: Dict[str, List[dict]] = {}
    for family in FAMILIES:
        path = directory / "vectors" / ("%s.json" % family)
        if path.exists():
            out[family] = json.loads(path.read_text(encoding="utf-8"))
    return out


def blob_bytes(blobs: Mapping[str, str], sha: Any) -> bytes:
    """The bytes a vector names, checked against the key they are stored under."""
    if not isinstance(sha, str):
        raise ReplayError("a blob reference must be a sha256 string, got %s" % type(sha).__name__)
    if sha not in blobs:
        raise ReplayError("the set names a blob it does not carry: %s" % sha)
    try:
        raw = base64.b64decode(blobs[sha], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReplayError("blob %s is not base64: %s" % (sha, exc))
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise ReplayError("blob %s hashes to %s" % (sha, got))
    return raw


def blob_value(blobs: Mapping[str, str], sha: Any) -> Any:
    """The JSON value a blob holds. Blob bytes are canonical bytes, so this round trips."""
    raw = blob_bytes(blobs, sha)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReplayError("blob %s does not parse as JSON: %s" % (sha, exc))


# --------------------------------------------------------------------------- argument rebuilding


def _need(inputs: Mapping, name: str) -> Any:
    if name not in inputs:
        raise ReplayError("the vector does not carry %r" % name)
    return inputs[name]


def _raw(value: Any, what: str) -> bytes:
    if not isinstance(value, str):
        raise ReplayError("%s must be hex, got %s" % (what, type(value).__name__))
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ReplayError("%s is not hex: %s" % (what, exc))


def _raws(value: Any, what: str) -> List[bytes]:
    if not isinstance(value, list):
        raise ReplayError("%s must be a list of hex strings" % what)
    return [_raw(v, "%s[%d]" % (what, i)) for i, v in enumerate(value)]


def _resolver(spec: Any, blobs: Mapping[str, str]) -> Any:
    """Rebuild `verify.diff`'s third argument from its recorded kind and answers."""
    if not isinstance(spec, Mapping) or "kind" not in spec:
        raise ReplayError("the vector carries no resolver kind")
    kind = spec["kind"]
    if kind == "none":
        return None
    if kind == "unsupported":
        return UNSUPPORTED_RESOLVER
    if kind == "resolving":
        answers = spec.get("answers") or {}
        if not isinstance(answers, Mapping):
            raise ReplayError("resolver answers must be a map of id -> blob")
        return {cert_id: blob_value(blobs, sha) for cert_id, sha in answers.items()}
    raise ReplayError("unknown resolver kind %r" % (kind,))


# --------------------------------------------------------------------------- outcome shapes


def _describe_check(outcome: Any) -> dict:
    kinds = []
    for reason in outcome.reasons:
        kind = reason_kind(reason)
        if kind is None:
            raise ReplayError(
                "cert.check emitted a reason the taxonomy does not classify: %r" % reason
            )
        kinds.append(kind)
    return {
        "outcome": "check",
        "ok": bool(outcome.ok),
        "type": outcome.type,
        "id": outcome.id,
        "reason_kinds": sorted(kinds),
    }


def _describe_number(value: Any) -> dict:
    if isinstance(value, tuple):
        distance, flips = value
        return {"outcome": "value", "value": float(distance), "anchor_flips": int(flips)}
    return {"outcome": "value", "value": float(value)}


def _describe_ok_reason(value: Any) -> dict:
    ok, reason = value
    return {"outcome": "verified", "ok": bool(ok), "reason": str(reason)}


def _describe_bool(value: Any) -> dict:
    if not isinstance(value, bool):
        raise ReplayError("expected a bool, got a %s" % type(value).__name__)
    return {"outcome": "verified", "ok": value}


def _describe_block(value: Any) -> dict:
    return {"outcome": "block", "block": None if value is None else dict(value)}


def _describe_outcome(value: Any) -> dict:
    body = value.result_body
    return {
        "outcome": "outcome",
        "exit_code": int(value.exit_code),
        "verdict": str(value.verdict),
        "mismatched": list(value.mismatched),
        "result_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
    }


# --------------------------------------------------------------------------- the entrypoints

_Runner = Callable[[Mapping, Mapping[str, str]], dict]


def _run_cert_check(inputs, blobs):
    return _describe_check(certmod.check(blob_value(blobs, _need(inputs, "cert"))))


def _run_merkle_root(inputs, blobs):
    return {
        "outcome": "root",
        "root": merklemod.root(_raws(_need(inputs, "leaves"), "leaves")).hex(),
    }


def _run_merkle_inclusion_proof(inputs, blobs):
    path = merklemod.inclusion_proof(
        _raws(_need(inputs, "leaves"), "leaves"),
        _need(inputs, "index"),
        _need(inputs, "tree_size"),
    )
    return {"outcome": "proof", "path": [h.hex() for h in path]}


def _run_merkle_consistency_proof(inputs, blobs):
    path = merklemod.consistency_proof(
        _raws(_need(inputs, "leaves"), "leaves"),
        _need(inputs, "first"),
        _need(inputs, "second"),
    )
    return {"outcome": "proof", "path": [h.hex() for h in path]}


def _run_merkle_verify_inclusion(inputs, blobs):
    return _describe_bool(
        merklemod.verify_inclusion(
            _raw(_need(inputs, "leaf_hash"), "leaf_hash"),
            _need(inputs, "index"),
            _need(inputs, "tree_size"),
            _raws(_need(inputs, "proof"), "proof"),
            _raw(_need(inputs, "expected_root"), "expected_root"),
        )
    )


def _run_merkle_verify_consistency(inputs, blobs):
    return _describe_bool(
        merklemod.verify_consistency(
            _need(inputs, "first"),
            _need(inputs, "second"),
            _raw(_need(inputs, "first_root"), "first_root"),
            _raw(_need(inputs, "second_root"), "second_root"),
            _raws(_need(inputs, "proof"), "proof"),
        )
    )


def _run_log_verify_sth(inputs, blobs):
    return _describe_ok_reason(
        logmod.verify_sth(
            blob_value(blobs, _need(inputs, "sth")),
            _raw(_need(inputs, "log_public"), "log_public"),
        )
    )


def _run_log_verify_inclusion(inputs, blobs):
    return _describe_ok_reason(
        logmod.verify_inclusion(
            blob_value(blobs, _need(inputs, "proof")),
            blob_value(blobs, _need(inputs, "sth")),
            _raw(_need(inputs, "log_public"), "log_public"),
        )
    )


def _run_log_verify_consistency(inputs, blobs):
    return _describe_ok_reason(
        logmod.verify_consistency(
            blob_value(blobs, _need(inputs, "sth_m")),
            blob_value(blobs, _need(inputs, "sth_n")),
            blob_value(blobs, _need(inputs, "proof")),
            _raw(_need(inputs, "log_public"), "log_public"),
        )
    )


def _run_distance(inputs, blobs):
    fn = _need(inputs, "fn")
    a = blob_value(blobs, _need(inputs, "a"))
    b = blob_value(blobs, _need(inputs, "b"))
    if fn == "exact":
        return _describe_number(distmod.exact(a, b, _need(inputs, "roles")))
    if fn == "seqlp":
        return _describe_number(distmod.seqlp(a, b))
    if fn == "topk":
        return _describe_number(distmod.topk(a, b))
    if fn == "resid":
        return _describe_number(distmod.resid(a, b))
    if fn == "lens":
        return _describe_number(distmod.lens(a, b, _need(inputs, "n_layers")))
    raise ReplayError("unknown distance function %r" % (fn,))


def _run_floor_pairwise(inputs, blobs):
    return _describe_block(
        floormod.pairwise(
            blob_value(blobs, _need(inputs, "runs")),
            _need(inputs, "channel"),
            roles=inputs.get("roles"),
        )
    )


def _run_floor_decide(inputs, blobs):
    return {
        "outcome": "verdict",
        "verdict": str(
            floormod.decide(
                _need(inputs, "distance"),
                _need(inputs, "floor"),
                _need(inputs, "confirmation"),
                covered=_need(inputs, "covered"),
                skew=_need(inputs, "skew"),
            )
        ),
    }


def _run_verify_diff(inputs, blobs):
    return _describe_outcome(
        verifymod.diff(
            blob_value(blobs, _need(inputs, "a")),
            blob_value(blobs, _need(inputs, "b")),
            _resolver(_need(inputs, "resolver"), blobs),
        )
    )


RUNNERS: Dict[str, _Runner] = {
    "cert.check": _run_cert_check,
    "merkle.root": _run_merkle_root,
    "merkle.inclusion_proof": _run_merkle_inclusion_proof,
    "merkle.consistency_proof": _run_merkle_consistency_proof,
    "merkle.verify_inclusion": _run_merkle_verify_inclusion,
    "merkle.verify_consistency": _run_merkle_verify_consistency,
    "log.verify_sth": _run_log_verify_sth,
    "log.verify_inclusion": _run_log_verify_inclusion,
    "log.verify_consistency": _run_log_verify_consistency,
    "distances.exact": _run_distance,
    "distances.seqlp": _run_distance,
    "distances.topk": _run_distance,
    "distances.resid": _run_distance,
    "distances.lens": _run_distance,
    "floor.pairwise": _run_floor_pairwise,
    "floor.decide": _run_floor_decide,
    "verify.diff": _run_verify_diff,
}

assert set(RUNNERS) == set(ENTRYPOINTS), "a wrapper exists with no replay, or the other way round"


#: Which fields of a vector hold a blob key, per entrypoint. Stated rather than sniffed: a merkle
#: leaf hash, a log public key and a cert id are all 64 hex characters, and a heuristic that
#: guessed by shape would call every one of them a blob.
BLOB_FIELDS: Dict[str, Tuple[str, ...]] = {
    "cert.check": ("cert",),
    "merkle.root": (),
    "merkle.inclusion_proof": (),
    "merkle.consistency_proof": (),
    "merkle.verify_inclusion": (),
    "merkle.verify_consistency": (),
    "log.verify_sth": ("sth",),
    "log.verify_inclusion": ("proof", "sth"),
    "log.verify_consistency": ("sth_m", "sth_n", "proof"),
    "distances.exact": ("a", "b"),
    "distances.seqlp": ("a", "b"),
    "distances.topk": ("a", "b"),
    "distances.resid": ("a", "b"),
    "distances.lens": ("a", "b"),
    "floor.pairwise": ("runs",),
    "floor.decide": (),
    "verify.diff": ("a", "b"),
}

assert set(BLOB_FIELDS) == set(ENTRYPOINTS)


def blob_refs(vector: Mapping) -> set:
    """Every blob key one vector names, by the entrypoint's own argument shape.

    `verify.diff` adds two beyond its arguments: the certs its resolver answered with, and the
    result body its outcome digests. The body is carried so a moved result can be diffed key by
    key rather than reported as two different hashes.
    """
    entrypoint = vector.get("entrypoint")
    if entrypoint not in BLOB_FIELDS:
        raise ReplayError("no blob shape for entrypoint %r" % (entrypoint,))
    inputs = vector.get("inputs") or {}
    found = {inputs[field] for field in BLOB_FIELDS[entrypoint] if field in inputs}
    if entrypoint == "verify.diff":
        resolver = inputs.get("resolver") or {}
        found |= set((resolver.get("answers") or {}).values())
        expect = vector.get("expect") or {}
        if "result_sha256" in expect:
            found.add(expect["result_sha256"])
    return {sha for sha in found if isinstance(sha, str)}


# --------------------------------------------------------------------------- replay and compare


def replay(vector: Mapping, blobs: Mapping[str, str]) -> dict:
    """Run one vector's entrypoint on its inputs; return the `expect` the run produced.

    A refusal is an outcome: an entrypoint that raises produces `{"outcome": "refused", "error":
    <exception type name>}`, the same shape the recorder wrote. `ReplayError` is not an outcome
    and is never caught here -- it means the vector, not the implementation, is wrong.
    """
    entrypoint = vector.get("entrypoint")
    runner = RUNNERS.get(entrypoint)
    if runner is None:
        raise ReplayError("no replay for entrypoint %r" % (entrypoint,))
    inputs = vector.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ReplayError("the vector carries no inputs object")
    try:
        return runner(inputs, blobs)
    except ReplayError:
        raise
    except Exception as exc:  # noqa: BLE001 -- a refusal is an outcome
        return {"outcome": "refused", "error": type(exc).__name__}


def agrees(expect: Any, got: Any) -> bool:
    """Canonical-bytes equality: the comparison a second implementation is held to."""
    return canonical_bytes(expect) == canonical_bytes(got)


def difference(expect: Any, got: Any) -> Optional[str]:
    """`None` when the two agree, else a one-line account of the first key that differs."""
    if agrees(expect, got):
        return None
    if isinstance(expect, Mapping) and isinstance(got, Mapping):
        for key in sorted(set(expect) | set(got)):
            if key not in expect:
                return "the replay produced %s, the vector has no such key" % key
            if key not in got:
                return "the vector expects %s, the replay produced no such key" % key
            if not agrees(expect[key], got[key]):
                return "%s: expected %s, replayed %s" % (
                    key,
                    json.dumps(expect[key], sort_keys=True)[:200],
                    json.dumps(got[key], sort_keys=True)[:200],
                )
    return "expected %s, replayed %s" % (
        json.dumps(expect, sort_keys=True)[:200],
        json.dumps(got, sort_keys=True)[:200],
    )


def check_vector(vector: Mapping, blobs: Mapping[str, str]) -> Optional[str]:
    """`None` when the vector reproduces, else why it did not.

    Three things are checked, in order: the stored id is the address of the stored family and
    inputs, every blob the vector names hashes to its key (`blob_bytes` does that on the way in),
    and the entrypoint reproduces `expect`.
    """
    family, inputs = vector.get("family"), vector.get("inputs")
    address = vector_id(family, inputs)
    if address != vector.get("id"):
        return "id %s is not the address of its own family and inputs (%s)" % (
            vector.get("id"),
            address,
        )
    got = replay(vector, blobs)
    return difference(vector.get("expect"), got)


def replay_set(
    vectors: Mapping[str, Iterable[Mapping]], blobs: Mapping[str, str]
) -> Tuple[Dict[str, dict], List[Tuple[str, str, str]]]:
    """Replay every vector. Returns per-family counts and the list of (family, id, why) failures."""
    counts: Dict[str, dict] = {}
    failures: List[Tuple[str, str, str]] = []
    for family, family_vectors in vectors.items():
        passed = failed = 0
        for vector in family_vectors:
            try:
                why = check_vector(vector, blobs)
            except ReplayError as exc:
                why = "the vector cannot be replayed: %s" % exc
            if why is None:
                passed += 1
            else:
                failed += 1
                failures.append((family, str(vector.get("id")), why))
        counts[family] = {"vectors": passed + failed, "passed": passed, "failed": failed}
    return counts, failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="replay the styxx.v8 conformance set")
    parser.add_argument("--dir", default=str(HERE), help="the set to replay")
    parser.add_argument("--id", help="replay one vector and print both sides")
    args = parser.parse_args(argv)

    directory = Path(args.dir).resolve()
    blobs = load_blobs(directory)
    vectors = load_vectors(directory)

    if args.id:
        for family, family_vectors in vectors.items():
            for vector in family_vectors:
                if vector.get("id") == args.id:
                    got = replay(vector, blobs)
                    print(json.dumps({"family": family, "vector": vector, "replayed": got},
                                     indent=1, sort_keys=True))
                    return 0 if agrees(vector.get("expect"), got) else 1
        print("no vector with id %s" % args.id, file=sys.stderr)
        return 2

    counts, failures = replay_set(vectors, blobs)
    for family in FAMILIES:
        if family in counts:
            row = counts[family]
            print("%-10s %5d vectors  %5d pass  %5d fail"
                  % (family, row["vectors"], row["passed"], row["failed"]))
    for family, vid, why in failures:
        print("  FAIL %s %s: %s" % (family, vid, why), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
