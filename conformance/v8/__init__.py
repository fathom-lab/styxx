"""The styxx.v8 conformance set: vectors, the recorder that produced them, and their calibration.

Built to `styxx/v8/INTERFACES_layer2.md` section 10. Not part of the `styxx` package: nothing
under `styxx/v8/` imports anything here, and this package imports `styxx.v8` only.

The set is generated (`gen_vectors.py`), replayed (`replay.py`, `tests/test_v8_conformance.py`),
and calibrated (`mutation_catalogue.json` + `mutation_coverage.py`). The lab's rule applies:
an agreement number is worth nothing without a measurement of the instrument's detection power,
so the vectors ship with a mutation catalogue and the miss list is a deliverable.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: The families in the order section 10 names them.
FAMILIES = ("cert", "merkle", "sth", "distance", "floor", "decide", "exit")

#: The test files the recorder runs over.
SOURCES = (
    "tests/test_v8_cert.py",
    "tests/test_v8_log.py",
    "tests/test_v8_distances.py",
    "tests/test_v8_floor.py",
    "tests/test_v8_verify.py",
)

#: Entrypoint name -> family. A wrapper exists for exactly these.
ENTRYPOINTS = {
    "cert.check": "cert",
    "merkle.root": "merkle",
    "merkle.inclusion_proof": "merkle",
    "merkle.consistency_proof": "merkle",
    "merkle.verify_inclusion": "merkle",
    "merkle.verify_consistency": "merkle",
    "log.verify_sth": "sth",
    "log.verify_inclusion": "sth",
    "log.verify_consistency": "sth",
    "distances.exact": "distance",
    "distances.seqlp": "distance",
    "distances.topk": "distance",
    "distances.resid": "distance",
    "distances.lens": "distance",
    "floor.pairwise": "floor",
    "floor.decide": "decide",
    "verify.diff": "exit",
}

#: The environment variable the recorder appends its JSON lines to.
RECORDER_OUT = "STYXX_V8_RECORDER_OUT"

#: A blob larger than this is not carried; the call is listed under index.unvectored.skipped.
BLOB_CAP = 1024 * 1024

SCHEMA = "styxx.v8.conformance/v1"

#: The `cert.check` reason taxonomy, by prefix. A vector pins the KINDS a cert produces, never a
#: reason's tail: the tail carries interpreter text and file paths. Lives here rather than in
#: `recorder.py` because importing the recorder installs its wrappers into `styxx.v8`; the replay
#: and the tests need the taxonomy and must not take that side effect.
REASON_PREFIXES = (
    ("schema[", "schema"),
    ("version: ", "version"),
    ("id: canonical bytes unavailable", "id-uncomputable"),
    ("id: does not recompute", "id-mismatch"),
    ("issuer.key: missing", "key-missing"),
    ("issuer.key: ", "key-bad"),
    ("sig: missing", "sig-missing"),
    ("sig: does not decode", "sig-undecodable"),
    ("sig: not verifiable", "sig-unverifiable"),
    ("sig: does not verify", "sig-bad"),
    ("materials: ", "materials"),
    ("refs: embedded id", "refs-missing"),
    ("refs: role", "refs-role"),
    ("number: ", "number"),
    ("cert: not a JSON object", "not-an-object"),
)

#: Every kind the table can produce, sorted. `index.unvectored.reason_kinds` names the ones no
#: vector reaches.
REASON_KINDS = tuple(sorted({kind for _prefix, kind in REASON_PREFIXES}))


def reason_kind(reason: str):
    """Classify one `cert.check` reason, or ``None`` when it escapes the taxonomy.

    A caller decides what an unclassified reason means: the recorder refuses to carry the call,
    the replay refuses to compare it. Neither guesses.
    """
    for prefix, kind in REASON_PREFIXES:
        if reason.startswith(prefix):
            return kind
    return None


#: How `verify.diff`'s third argument is carried. `styxx.v8.verify._resolve` dispatches on None,
#: then Mapping, then find/cert, then callable, and refuses anything else; a record therefore holds
#: the KIND plus, for a resolver that answers, the id -> cert map it actually answered with.
RESOLVER_KINDS = ("none", "resolving", "unsupported")
