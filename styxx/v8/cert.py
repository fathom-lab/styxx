"""styxx.v8.cert — the styxx/8.0 cert envelope (spec sections 2, 2.1-2.5, 7.2, Appendix A.1).

What this module decides:

* ``digest_bytes`` / ``compute_id`` — ``D = sha256(JCS(cert minus "id" and "sig"))``,
  ``id = "sha256:" + hex(D)``.
* ``sign`` — Ed25519 over ``keys.tagged(CERT_TAG, D)``; refuses a seed whose public key is
  not the cert's ``issuer.key``.
* ``check`` — every failure collected, in the contract's order: schema (envelope, then the
  per-type file), id recompute, issuer key decoding and point validation (non-canonical and
  small-order encodings rejected, section 2.1), signature, recipe materials against their
  hashes (section 2.3), every embedded cert id present in ``refs`` (A-09), ref roles allowed for
  the type, integers within +-2**53, no NaN or infinity.
* ``schema_errors`` — JSON Schema Draft 2020-12 over ``schema/*.json``.
* ``seal_commitment`` — the sealed-prereg commitment of section 7.2.
* ``version_ok`` — section 2.5.
* ``identity_fields`` / ``recipe_core`` / ``comparable`` / ``skew_fields`` — the keys of
  sections 2.2 and 2.3.
* ``challenge_validity`` — section 9 rule 1, the two-cert predicate a client computes over a
  challenge's ``target`` and ``own`` (see its docstring).

Nothing here reads the log; ``check`` is a function of the cert's bytes alone. Whether a ref
resolves is the log's question.

Decisions
---------

**A body a synthetic runner produced carries ``synthetic: true``, and ``comparable`` refuses to
cross that line.** ``--runner mock`` shipped in the CLI (and is the DEFAULT when ``--runner`` is
omitted), and ``MockRunner`` reported back whatever subject it was handed, so the section 6
subject guard was true by construction and an attacker signed a section 9 challenge against a
published canonical carrying distances from a model that was never loaded — passing the guard at
mint AND at append (``papers/v8/challenge_and_attack_2026_09_09``, C-MOCK; the C2 entry is the
same hole reached with the real runner, and is repaired in ``styxx/v8/runner.py``).

Two rules were available: refuse to sign anything a synthetic runner produced, or mark every
cert it produces in the SIGNED bytes. The marker governs, for a reason about this codebase and
not about elegance — the mock IS the exercise surface, and a tool that cannot mint a toy ladder
end to end needs a second, unmarked minting path for its own tests, which is the hole again with
an extra door. So the mock keeps minting and every artifact it touches says so.

WHERE the marker lives is the load-bearing part. It is a **body** member. The body is the
measurement — the per-item digests, the per-channel distances, the floor — and "these numbers
came from a hash, not from a model" is a fact about the measurement, not about the model that
was requested. The subject was considered and rejected: a subject is what the caller asked for,
and stamping it there would say a synthetic run had a different MODEL rather than no model.
Every cert type carries it identically (``is_synthetic``): a fingerprint body, a verify result
body, a section 9 challenge body.

``comparable`` reads it, which is what makes one member reach every place that matters, with no
new predicate written for any of them: section 9 rule 1 (``challenge_validity`` IS
``comparable``, so the check runs at the client, at mint and at append), the section 5.5
baseline rule (``Log.previous_comparable``), ``verify --ref``'s guard and ``verify --diff``. A
synthetic fingerprint therefore cannot be the baseline for a measured one, cannot be diffed
against one, and cannot challenge one. The entry is ``body.synthetic`` and it is NEVER softened
to ``cross-subject:``: ``precision`` and ``revision`` are softened because two precisions of one
model are still that model, and a fabrication is not a precision.

**The marker is re-derived, not trusted (the mock oracle).** The paragraph above used to end by
conceding that an issuer holding the key can hand-sign a body with the member omitted, and that
the marker was therefore a property of the tooling and not of the format. The third adversarial
pass took the concession: ``schema/fingerprint.json`` did not require ``synthetic``, so deleting
one key from a mock's body and re-signing produced an unmarked fabrication that ``comparable``
read as a measurement (C-MOCK-2). Requiring the key in the schema does not repair that -- a
schema can say the member is present and cannot say it is TRUE, and an issuer that will delete a
key will write ``false`` into it.

What repairs it is that ``MockRunner`` computes from a hash whose inputs the cert carries.
``runner.mock_derived_items`` re-derives the mock's ``token_ids_sha256`` for each item from this
cert's own ``subject``, ``recipe`` core and ``item_id``s -- trying every variant list
``MockRunner._variants`` can build, since which one an item took is runner configuration no cert
records -- and ``check`` refuses a fingerprint that reproduces them without carrying the marker.
The numbers say whose they are. This is the same move the log made on the floor (``log.py``,
"the floor is not a claim"): stop asking the issuer for a fact that is a function of bytes
already present.

What it still does NOT do, stated rather than implied. It catches the mock, not fabrication in
general: a body from some OTHER hash, or hand-typed digits, reproduces no derivation here and is
refused by nothing in this file. ``synthetic`` on a verify result or a section 9 challenge body
is still the issuer's word, because those bodies carry no items to re-derive -- what anchors
them is the fingerprint they name, which this does check. And the oracle is an accusation only
when it MATCHES: a subject or recipe it cannot read yields no items and no reason, because "I
could not check" must never read as "this is a model".

**A single-valued role is named once.** ``Log._check_challenge_subject`` built ``by_role[role]``
in a loop, so a challenge carrying ``refs = [target, own=<fp16>, own=<bf16>]`` gave the log the
LAST ``own`` while a reader walking ``refs`` in order stops at the earlier one — one cert, two
readings of section 9 rule 1, and the attacker chose which reading each party got (same paper,
C-DUPREF).

Refusing duplicates only on a challenge would close that construction and leave the shape.
Section 2.1's ``refs`` is a list of ``{role, id}`` pairs that every reader in this codebase
turns into a mapping from role to cert — ``verify._roles_of``, ``Log._check_challenge_subject``,
the JavaScript verifier — and a mapping built from a list with a repeated key has no defined
value; which one you get is which loop you wrote. So duplicates are refused for every role
except the three that are plural by construction: ``run`` (the R−1 floor runs of section 5.1),
and ``result``/``robustness`` (a promotion's evidence, section 7.2). The argument the other way
is that a repeated role is a schema question and ``check`` should stay narrow; it loses because
the schema cannot express "except run, result and robustness" per type without restating the
role table a third time, and because the defect was found in a reader, not in a schema.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from styxx.v8 import keys
from styxx.v8 import runner as runnermod
from styxx.v8.consts import CERT_TAG, ID_RE, REF_ROLES, SCHEMA_VERSION, SEAL_TAG, TYPES
from styxx.v8.jcs import MAX_SAFE_INT, canonical_bytes

__all__ = [
    "CertCheck",
    "CROSS_SUBJECT_FIELDS",
    "ENVELOPE_KEYS",
    "IDENTITY_FIELDS",
    "MATERIAL_HASHES",
    "NON_REF_HASH_PATHS",
    "RECIPE_CORE_FIELDS",
    "RESTATED_ID_PATHS",
    "MULTI_ROLES",
    "ROLES_BY_TYPE",
    "ROLE_TYPES",
    "ROLE_TYPES_BY_CERT_TYPE",
    "SKEW_FIELDS",
    "SYNTHETIC",
    "challenge_validity",
    "check",
    "comparable",
    "duplicate_roles",
    "is_synthetic",
    "role_type",
    "compute_id",
    "digest_bytes",
    "embedded_ids",
    "evidence_id",
    "identity_fields",
    "material_hash",
    "public_key_reason",
    "recipe_core",
    "refs",
    "schema_errors",
    "schema_names",
    "seal_commitment",
    "sign",
    "skew_fields",
    "version_ok",
]

# ----------------------------------------------------------------- vocabulary

ENVELOPE_KEYS = ("styxx", "type", "id", "created", "issuer", "subject", "recipe", "body", "refs", "sig")
_HASHED_KEYS = ("styxx", "type", "created", "issuer", "subject", "recipe", "body", "refs")
_CONTENT_KEYS = ("subject", "recipe", "body")

# recipe.materials.<field> -> recipe.<field>_sha256 (section 2.3, Appendix A.1)
MATERIAL_HASHES = (
    ("chat_template", "chat_template_sha256"),
    ("system_prompt", "system_prompt_sha256"),
    ("env_lock", "env_lock_sha256"),
)

# Paths (slash-joined, from the cert root) whose "sha256:<hex>" value is a content hash the
# spec defines, not a cert id: a blob address (section 3.1), a seal commitment (section 7.2), and
# the log a cert binds itself to (section 8.1's `log_id`, the hash of the raw log public key).
# They are excluded from the embedded-id rule in ``check``; ``embedded_ids`` stays literal.
#
# ``body/log_hint/log_id`` is the THIRD member, and A.1 currently says "there is no third
# exemption and none is inferable from what a field means". That sentence has to move for L7's
# binding to be legal, and the owed edit is named here rather than made quietly: a `log_id` is a
# hash of 32 bytes of public key, it names no cert, and requiring it to appear in ``refs`` and
# resolve to a logged cert would make the field unusable by construction.
NON_REF_HASH_PATHS = frozenset({"body/items_blob", "body/commitment", "body/log_hint/log_id"})

# Paths holding a cert id RESTATED from a cert an existing ref already names, as part of a
# self-report the log checks for equality against that cert. A-09 exists so that a cert cannot
# depend on another cert without saying so in ``refs``; these create no dependency the refs do
# not already carry -- ``body.recipe_core`` on a section 9 challenge is a copy of the ``own``
# fingerprint's ``recipe_core`` (``Log._check_challenge_self_report`` refuses a copy that
# disagrees), and ``body.observed_recipe_core`` on a verify result is the recipe of the cert its
# ``ref`` names. Adding a ``battery`` ref instead would put a role on a challenge that section 9
# does not give it; excluding the two paths, by name, is the narrower move.
RESTATED_ID_PATHS = frozenset({"body/recipe_core/battery", "body/observed_recipe_core/battery"})

# Which cert type a ref of each role must resolve to (section 2.1, last bullet). ``None``
# means "the referencing cert's own type" (``previous``); ``"*"`` means any type
# (``target`` is a fingerprint for a challenge and a challenge for a response result).
ROLE_TYPES: dict[str, Optional[str]] = {
    "battery": "battery",
    "run": "fingerprint",
    "selected_against": "fingerprint",
    "pool": "battery",
    "prereg": "prereg",
    "result": "result",
    "robustness": "result",
    "target": "*",
    "own": "fingerprint",
    "parent": "action",
    "fingerprint": "fingerprint",
    "sealed": "prereg",
    "previous": None,
    "sensitivity": "result",
    "noise_plan": "prereg",
}

# Roles a cert of each type may carry.
ROLES_BY_TYPE: dict[str, frozenset[str]] = {
    "fingerprint": frozenset({"battery", "run", "previous", "noise_plan", "sensitivity", "prereg", "pool"}),
    "battery": frozenset({"selected_against", "pool", "previous", "battery"}),
    "prereg": frozenset({"sealed", "previous", "battery", "fingerprint", "prereg"}),
    "result": frozenset(
        {
            "prereg", "target", "own", "robustness", "result", "fingerprint", "battery",
            "noise_plan", "run", "sensitivity", "previous",
        }
    ),
    "promotion": frozenset({"prereg", "result", "robustness"}),
    "action": frozenset({"fingerprint", "parent"}),
    "challenge": frozenset({"target", "own"}),
    "sublog": frozenset({"previous", "fingerprint"}),
}

# S_identity per subject kind (section 2.2).
IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "weights": (
        "hf_repo", "revision", "weights_sha256", "config_sha256", "tokenizer_sha256",
        "generation_config_sha256", "precision",
    ),
    "alias": ("provider", "alias", "region"),
}
RECIPE_CORE_FIELDS = ("battery", "decoding", "chat_template_sha256", "system_prompt_sha256")
SKEW_FIELDS = ("harness.version", "env_lock_sha256")
CROSS_SUBJECT_FIELDS = ("precision", "revision")

# The synthetic marker (see the module docstring, "Decisions"). A body whose numbers a synthetic
# runner produced carries ``synthetic: true``; a body a model produced does not carry the key at
# all. ``comparable`` reads it as a hard difference, so no comparison, no floor, no baseline and
# no challenge ever crosses the line between a measurement and a fabrication.
SYNTHETIC = "synthetic"

# Roles that may appear more than once in ``refs``. Every other role names one cert, and a refs
# list carrying it twice has no defined value for a reader (see "Decisions").
MULTI_ROLES = frozenset({"run", "result", "robustness"})

# Where the type a role must resolve to depends on the REFERENCING cert's type. ``ROLE_TYPES``
# keeps ``target`` at ``"*"`` because a ``result`` of kind ``response`` names a challenge; on a
# challenge cert section 9 fixes the target to the fingerprint being challenged.
ROLE_TYPES_BY_CERT_TYPE: dict[str, dict[str, str]] = {
    "challenge": {"target": "fingerprint"},
    # `previous` is None in ROLE_TYPES -- "a cert of my own type" -- which is right on a
    # fingerprint, where it names the baseline this one replaces (section 5.5). A verify RESULT
    # carrying the section 5.5 disclosure embeds that same baseline's id in its body, so section
    # 2.1 makes it a ref the result must carry, and the cert it resolves to is a FINGERPRINT, not
    # another result. Without this override the log refused such a result at append, naming a
    # `previous ref ... resolves to a fingerprint cert, not a result cert`.
    "result": {"previous": "fingerprint"},
}


def role_type(cert_type: Optional[str], role: str) -> Optional[str]:
    """The cert type a ``role`` ref must resolve to, for a cert of ``cert_type``.

    ``None`` means "the referencing cert's own type" (``previous``); ``"*"`` means any type.
    This is ``ROLE_TYPES`` with the per-type overrides of ``ROLE_TYPES_BY_CERT_TYPE`` applied,
    and it is the only function a resolver should ask.
    """
    override = ROLE_TYPES_BY_CERT_TYPE.get(cert_type or "")
    if override is not None and role in override:
        return override[role]
    return ROLE_TYPES.get(role, "*")


def is_synthetic(cert: Any) -> bool:
    """True when this cert's body carries the synthetic marker.

    A cert of any type: a fingerprint body, a verify result body and a section 9 challenge body
    all carry the marker the same way, because the question is always the same one -- did a
    model produce these numbers.

    The rule is **present and not ``false``**, not ``is True``, and it fails closed on purpose:
    the schema pins the member to ``true`` where it is declared, so any other value is hostile
    input, and a hostile ``synthetic: null`` or ``synthetic: "no"`` must not read as a
    measurement. ``challengeValidity`` in ``styxx/_data/v8_verify.js`` implements the same rule,
    so the two implementations agree on hostile bytes and not only on well-formed ones.
    """
    body = cert.get("body") if isinstance(cert, dict) else None
    if not isinstance(body, dict) or SYNTHETIC not in body:
        return False
    return body[SYNTHETIC] is not False


def duplicate_roles(cert: dict) -> list[str]:
    """Roles this cert's ``refs`` name more than once, excluding ``MULTI_ROLES``. Sorted."""
    counts: dict[str, int] = {}
    for role, _ in refs(cert):
        counts[role] = counts.get(role, 0) + 1
    return sorted(r for r, n in counts.items() if n > 1 and r not in MULTI_ROLES)

_ID_PATTERN = re.compile(ID_RE)
_VERSION_PATTERN = re.compile(r"^([0-9]+)\.([0-9]+)$")
_SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
_MESSAGE_LIMIT = 240


@dataclass
class CertCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    type: Optional[str] = None
    id: Optional[str] = None


# ----------------------------------------------------------------- digest / id / sign

def _core(cert: dict) -> dict:
    return {k: v for k, v in cert.items() if k not in ("id", "sig")}


def digest_bytes(cert: dict) -> bytes:
    """Raw 32 bytes: ``sha256(canonical_bytes(cert minus "id" and "sig"))``.

    Raises TypeError / ValueError from ``canonical_bytes`` when the cert holds a value
    outside the JCS domain (NaN, infinity, an int beyond +-2**53, a non-JSON type).
    """
    if not isinstance(cert, dict):
        raise TypeError(f"cert must be a dict, got {type(cert).__name__}")
    return hashlib.sha256(canonical_bytes(_core(cert))).digest()


def compute_id(cert: dict) -> str:
    """``"sha256:" + hex(digest_bytes(cert))``."""
    return "sha256:" + digest_bytes(cert).hex()


def evidence_id(cert: dict) -> str:
    """Informative: ``"sha256:" + hex(sha256(JCS({styxx, type, subject, recipe, body})))``.

    Two issuers of the same evidence share this value; they never share an ``id``.
    Refs may not use it.
    """
    # GATED S1-01: recommendation implemented; operator may reverse.
    if not isinstance(cert, dict):
        raise TypeError(f"cert must be a dict, got {type(cert).__name__}")
    part = {k: cert[k] for k in ("styxx", "type", "subject", "recipe", "body") if k in cert}
    return "sha256:" + hashlib.sha256(canonical_bytes(part)).hexdigest()


def sign(cert: dict, private_seed: bytes) -> dict:
    """Return a NEW dict with ``id`` and ``sig`` set; the input is not modified.

    ``sig`` is Ed25519 over ``keys.tagged(CERT_TAG, digest_bytes(cert))``. ValueError when
    ``issuer.key`` is not the encoding of the seed's public key.
    """
    if not isinstance(cert, dict):
        raise TypeError(f"cert must be a dict, got {type(cert).__name__}")
    expected_key = keys.encode_public(keys.public_from_private(private_seed))
    issuer = cert.get("issuer")
    actual_key = issuer.get("key") if isinstance(issuer, dict) else None
    if actual_key != expected_key:
        raise ValueError("issuer.key is not the public key of the signing seed; refusing to sign")
    core = copy.deepcopy(_core(cert))
    digest = hashlib.sha256(canonical_bytes(core)).digest()
    signature = keys.sign(private_seed, keys.tagged(CERT_TAG, digest))
    out: dict[str, Any] = {}
    for key in ENVELOPE_KEYS:
        if key == "id":
            out["id"] = "sha256:" + digest.hex()
        elif key == "sig":
            out["sig"] = keys.encode_signature(signature)
        elif key in core:
            out[key] = core[key]
    for key, value in core.items():  # unknown keys travel along; the schema refuses them
        if key not in out:
            out[key] = value
    return out


# ----------------------------------------------------------------- schema

def schema_names() -> tuple[str, ...]:
    """The schema files present, by stem, sorted."""
    return tuple(sorted(p.stem for p in _SCHEMA_DIR.glob("*.json")))


@lru_cache(maxsize=None)
def _schemas() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(_SCHEMA_DIR.glob("*.json")):
        with open(path, "rb") as fh:
            data = fh.read()
        if data.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"schema {path.name} carries a BOM")
        schema = json.loads(data.decode("utf-8"))
        if schema.get("$id") != f"urn:styxx:v8:schema:{path.stem}":
            raise ValueError(f"schema {path.name} must declare $id urn:styxx:v8:schema:{path.stem}")
        Draft202012Validator.check_schema(schema)
        out[path.stem] = schema
    return out


@lru_cache(maxsize=None)
def _registry() -> Registry:
    resources = [
        (schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012))
        for schema in _schemas().values()
    ]
    return Registry().with_resources(resources)


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(_schemas()[name], registry=_registry())


def _error_lines(name: str, instance: Any) -> list[str]:
    lines: list[str] = []
    errors = list(_validator(name).iter_errors(instance))
    for err in sorted(errors, key=lambda e: ([str(p) for p in e.absolute_path], e.message)):
        chosen = err
        if err.context:
            deeper = best_match(err.context)
            if deeper is not None:
                chosen = deeper
        path = "/".join(str(p) for p in chosen.absolute_path) or "$"
        message = chosen.message
        if len(message) > _MESSAGE_LIMIT:
            message = message[:_MESSAGE_LIMIT] + "..."
        lines.append(f"schema[{name}]: {path}: {message}")
    return lines


def schema_errors(cert: dict) -> list[str]:
    """Draft 2020-12 errors: the envelope schema, then ``schema/<type>.json``."""
    if not isinstance(cert, dict):
        return ["schema[envelope]: $: cert is not a JSON object"]
    lines = _error_lines("envelope", cert)
    ctype = cert.get("type")
    if isinstance(ctype, str) and ctype in TYPES:
        lines.extend(_error_lines(ctype, cert))
    return lines


# ----------------------------------------------------------------- public-key validation

_P = 2**255 - 19
_D = (-121665 * pow(121666, -1, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)
_IDENTITY = (0, 1)


def _decode_point(raw: bytes) -> Optional[tuple[int, int]]:
    """RFC 8032 section 5.1.3 decoding; None when the encoding does not decode."""
    y = int.from_bytes(raw, "little")
    sign_bit = y >> 255
    y &= (1 << 255) - 1
    if y >= _P:
        return None
    u = (y * y - 1) % _P
    v = (_D * y * y + 1) % _P
    x2 = u * pow(v, -1, _P) % _P
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    if x == 0 and sign_bit:
        return None
    if (x & 1) != sign_bit:
        x = _P - x
    return (x, y)


def _add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    t = _D * x1 * x2 * y1 * y2 % _P
    x3 = (x1 * y2 + x2 * y1) * pow(1 + t, -1, _P) % _P
    y3 = (y1 * y2 + x1 * x2) * pow(1 - t, -1, _P) % _P
    return (x3, y3)


def _is_small_order(point: tuple[int, int]) -> bool:
    p = point
    for _ in range(3):  # [8]P
        p = _add(p, p)
    return p == _IDENTITY


def public_key_reason(encoded: str) -> Optional[str]:
    """None when ``encoded`` is an acceptable issuer key; else the reason it is not.

    Rejected (section 2.1): a wrong prefix, padding, a wrong length, a non-canonical
    base64url ending (``keys.decode_public``); a non-canonical point encoding (y >= p); an
    encoding RFC 8032 section 5.1.3 does not decode; a small-order point (the identity, the
    order-2 point, the order-4 and order-8 points).
    """
    try:
        raw = keys.decode_public(encoded)
    except (TypeError, ValueError) as exc:
        return f"does not decode: {exc}"
    y = int.from_bytes(raw, "little") & ((1 << 255) - 1)
    if y >= _P:
        return "non-canonical point encoding (y >= p)"
    point = _decode_point(raw)
    if point is None:
        return "not a point on the curve (RFC 8032 section 5.1.3)"
    if _is_small_order(point):
        return "small-order point"
    return None


# ----------------------------------------------------------------- walks

def _walk(obj: Any, path: str) -> Iterator[tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}/{i}")


def _walk_ids(cert: dict) -> Iterator[tuple[str, str]]:
    for key in _CONTENT_KEYS:
        for path, value in _walk(cert.get(key), key):
            if isinstance(value, str) and _ID_PATTERN.match(value):
                yield path, value


def embedded_ids(cert: dict) -> set[str]:
    """Every value matching ``ID_RE`` anywhere under subject/recipe/body (recursive).

    ``RESTATED_ID_PATHS`` are excluded, so this stays the same set ``check``'s A-09 pass uses
    and a caller that builds refs from it produces a cert ``check`` will accept.
    """
    if not isinstance(cert, dict):
        return set()
    return {
        value for path, value in _walk_ids(cert) if path not in RESTATED_ID_PATHS
    }


def refs(cert: dict) -> list[tuple[str, str]]:
    """``[(role, id)]`` in cert order; malformed entries are skipped (the schema names them)."""
    out: list[tuple[str, str]] = []
    entries = cert.get("refs") if isinstance(cert, dict) else None
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("role"), str) and isinstance(entry.get("id"), str):
                out.append((entry["role"], entry["id"]))
    return out


def _numeric_reasons(obj: Any) -> list[str]:
    out: list[str] = []
    for path, value in _walk(obj, "$"):
        if isinstance(value, bool) or value is None or isinstance(value, (str, dict, list)):
            continue
        if isinstance(value, int):
            if value > MAX_SAFE_INT or value < -MAX_SAFE_INT:
                out.append(f"number: {path}: integer {value} is outside +-2**53")
        elif isinstance(value, float):
            if math.isnan(value):
                out.append(f"number: {path}: NaN has no JSON representation")
            elif math.isinf(value):
                out.append(f"number: {path}: infinity has no JSON representation")
        else:
            out.append(f"number: {path}: {type(value).__name__} is not a JSON value")
    return out


# ----------------------------------------------------------------- materials

def material_hash(text: str) -> str:
    """Appendix A.1: ``sha256(UTF-8 of the string)`` as 64 lowercase hex, no prefix."""
    if not isinstance(text, str):
        raise TypeError(f"material must be str, got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _material_reasons(recipe: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(recipe, dict) or not recipe:
        return out
    materials = recipe.get("materials")
    if not isinstance(materials, dict):
        return out  # the schema already names the missing block
    for source, hashed in MATERIAL_HASHES:
        text = materials.get(source)
        declared = recipe.get(hashed)
        if not isinstance(text, str) or not isinstance(declared, str):
            continue
        if material_hash(text) != declared:
            out.append(f"materials: recipe.{hashed} does not match sha256(materials.{source})")
    return out


# ----------------------------------------------------------------- version

def _parse_version(s: Any) -> Optional[tuple[int, int]]:
    if not isinstance(s, str):
        return None
    m = _VERSION_PATTERN.match(s)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)))


def version_ok(cert: dict, own: str = SCHEMA_VERSION) -> bool:
    """Section 2.5: the cert's ``styxx`` major.minor is <= ``own``; malformed is not ok."""
    theirs = _parse_version(cert.get("styxx") if isinstance(cert, dict) else None)
    mine = _parse_version(own)
    return theirs is not None and mine is not None and theirs <= mine


# ----------------------------------------------------------------- check

def check(cert: dict) -> CertCheck:
    """Every failure, collected in the contract's order. ``ok`` iff ``reasons`` is empty."""
    if not isinstance(cert, dict):
        return CertCheck(ok=False, reasons=["cert: not a JSON object"], type=None, id=None)
    ctype = cert.get("type") if isinstance(cert.get("type"), str) else None
    cid = cert.get("id") if isinstance(cert.get("id"), str) else None
    reasons: list[str] = []

    # 1. schema (envelope + per-type) and the version rule
    reasons.extend(schema_errors(cert))
    if isinstance(cert.get("styxx"), str) and not version_ok(cert):
        reasons.append(f"version: styxx {cert['styxx']!r} is not accepted by a {SCHEMA_VERSION} verifier")

    # 2. id recompute
    digest: Optional[bytes] = None
    try:
        digest = digest_bytes(cert)
    except (TypeError, ValueError) as exc:
        reasons.append(f"id: canonical bytes unavailable: {exc}")
    if digest is not None:
        expected = "sha256:" + digest.hex()
        if cid != expected:
            reasons.append(f"id: does not recompute (cert carries {cid!r}, bytes give {expected})")

    # 3. issuer key
    public: Optional[bytes] = None
    issuer = cert.get("issuer")
    key_s = issuer.get("key") if isinstance(issuer, dict) else None
    if not isinstance(key_s, str):
        reasons.append("issuer.key: missing")
    else:
        why = public_key_reason(key_s)
        if why is not None:
            reasons.append(f"issuer.key: {why}")
        else:
            public = keys.decode_public(key_s)

    # 4. signature over the tagged digest
    sig_s = cert.get("sig")
    signature: Optional[bytes] = None
    if not isinstance(sig_s, str):
        reasons.append("sig: missing")
    else:
        try:
            signature = keys.decode_signature(sig_s)
        except (TypeError, ValueError) as exc:
            reasons.append(f"sig: does not decode: {exc}")
    if signature is not None:
        if public is None or digest is None:
            reasons.append("sig: not verifiable without a valid issuer key and digest")
        elif not keys.verify(public, keys.tagged(CERT_TAG, digest), signature):
            reasons.append("sig: does not verify against issuer.key over the tagged digest")

    # 5. materials
    reasons.extend(_material_reasons(cert.get("recipe")))

    # 6. embedded ids appear in refs
    ref_ids = {rid for _, rid in refs(cert)}
    for path, value in _walk_ids(cert):
        if path in NON_REF_HASH_PATHS or path in RESTATED_ID_PATHS:
            continue
        if value not in ref_ids:
            reasons.append(f"refs: embedded id {value} at {path} is not in refs")

    # 7. roles valid for the type
    if ctype in ROLES_BY_TYPE:
        allowed = ROLES_BY_TYPE[ctype]
        for role, rid in refs(cert):
            if role in REF_ROLES and role not in allowed:
                reasons.append(f"refs: role {role!r} is not allowed on a {ctype} cert")

    # 7b. a single-valued role is named once (see the module docstring, "Decisions")
    for role in duplicate_roles(cert):
        reasons.append(
            f"refs: role {role!r} is named more than once; only {sorted(MULTI_ROLES)} may "
            "repeat, and a reader given two certs under one role has no rule for choosing "
            "between them"
        )

    # 8. numbers
    reasons.extend(_numeric_reasons(cert))

    # 9. the synthetic marker is not the issuer's to omit (see "Decisions", the mock oracle)
    reasons.extend(_synthetic_reasons(cert, ctype))

    return CertCheck(ok=not reasons, reasons=reasons, type=ctype, id=cid)


def _synthetic_reasons(cert: dict, ctype: Optional[str]) -> list[str]:
    """Refuse an unmarked fingerprint whose numbers ``MockRunner`` produces (C-MOCK-2).

    Only fingerprints: they are the certs that carry ``body.items``, which is where the mock's
    output lands and what every downstream number is computed from. A body already carrying the
    marker is not re-derived -- it has already said what it is.
    """
    if ctype != "fingerprint" or is_synthetic(cert):
        return []
    body, subject, recipe = cert.get("body"), cert.get("subject"), cert.get("recipe")
    if not (isinstance(body, dict) and isinstance(subject, dict) and isinstance(recipe, dict)):
        return []
    named = runnermod.mock_derived_items(body, subject, recipe)
    if not named:
        return []
    shown = named[:3]
    return [
        f"body: items {shown}{' and %d more' % (len(named) - 3) if len(named) > 3 else ''} carry "
        f"the token_ids_sha256 that {runnermod.MockRunner.__name__} derives from this cert's own "
        f"subject, recipe core and item ids, and the body does not carry {SYNTHETIC!r}; a body "
        "computed from a hash is not a measurement, and the marker is re-derived here rather "
        "than taken on the issuer's word (C-MOCK-2)"
    ]


# ----------------------------------------------------------------- seal

def seal_commitment(salt32: bytes, body_to_reveal: dict) -> str:
    """Section 7.2: ``"sha256:" + hex(sha256(SEAL_TAG || 0x00 || salt || JCS(body)))``."""
    if isinstance(salt32, (bytearray, memoryview)):
        salt32 = bytes(salt32)
    if not isinstance(salt32, bytes):
        raise TypeError(f"salt must be bytes, got {type(salt32).__name__}")
    if len(salt32) != 32:
        raise ValueError(f"salt must be 32 bytes, got {len(salt32)}")
    if not isinstance(body_to_reveal, dict):
        raise TypeError("body_to_reveal must be a dict")
    preimage = SEAL_TAG.encode("ascii") + b"\x00" + salt32 + canonical_bytes(body_to_reveal)
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


# ----------------------------------------------------------------- comparability keys

def identity_fields(subject: dict) -> dict:
    """Section 2.2 S_identity per kind, plus ``kind`` itself; missing fields read as None."""
    kind = subject.get("kind") if isinstance(subject, dict) else None
    out: dict[str, Any] = {"kind": kind if isinstance(kind, str) else None}
    for name in IDENTITY_FIELDS.get(out["kind"], ()):
        out[name] = copy.deepcopy(subject.get(name))
    return out


def recipe_core(recipe: dict) -> dict:
    """Section 2.3 recipe_core: battery, decoding, chat_template_sha256, system_prompt_sha256."""
    src = recipe if isinstance(recipe, dict) else {}
    return {name: copy.deepcopy(src.get(name)) for name in RECIPE_CORE_FIELDS}


def comparable(a: dict, b: dict) -> list[str]:
    """Mismatched key names between two certs; empty means comparable.

    Entries are ``subject.<field>`` and ``recipe.<field>``. A difference in ``precision`` or
    ``revision`` is returned as ``cross-subject:<field>`` instead (section 2.3): two certs whose
    only entries are cross-subject are comparable and labelled cross-subject; any other entry
    means exit 3.

    ``body.synthetic`` is returned when one side's distances came from a synthetic runner
    and the other's did not (see the module docstring, "Decisions"). It is never softened to
    ``cross-subject:``: a fabricated distance and a measured one are not two precisions of one
    subject, and nothing that reads this function may treat them as comparable.
    """
    out: list[str] = []
    if is_synthetic(a) != is_synthetic(b):
        out.append(f"body.{SYNTHETIC}")
    ia = identity_fields(a.get("subject", {}) if isinstance(a, dict) else {})
    ib = identity_fields(b.get("subject", {}) if isinstance(b, dict) else {})
    names = ["kind"] + [n for n in list(ia) + list(ib) if n != "kind"]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if ia.get(name) != ib.get(name):
            if name in CROSS_SUBJECT_FIELDS:
                out.append(f"cross-subject:{name}")
            else:
                out.append(f"subject.{name}")
    ra = recipe_core(a.get("recipe", {}) if isinstance(a, dict) else {})
    rb = recipe_core(b.get("recipe", {}) if isinstance(b, dict) else {})
    for name in RECIPE_CORE_FIELDS:
        if ra[name] != rb[name]:
            out.append(f"recipe.{name}")
    return out


def challenge_validity(target: dict, own: dict) -> list[str]:
    """Section 9 rule 1: why ``own`` is not a challenge to ``target``; empty means it is one.

    The rule, quoted: "A challenge is valid iff the challenger's fingerprint is comparable
    (section 2.3) AND has the same subject identity (every ``S_identity`` field equal); clients
    compute this from the two certs — nothing in the body asserts it. No match, no challenge."

    That is exactly ``comparable(target, own) == []``: ``comparable`` already walks every
    ``S_identity`` field of section 2.2 and every ``recipe_core`` field of section 2.3, and the
    two fields it softens to ``cross-subject:`` — ``precision`` and ``revision`` — are
    ``S_identity`` fields, so section 9 counts them. A cross-subject pair is a legitimate
    ``verify --diff``; it is not a challenge, because the target's floor was measured on the
    target's subject and says nothing about a different one (section 5.3).

    Returns ``comparable``'s own entries (``subject.<field>``, ``cross-subject:<field>``,
    ``recipe.<field>``) so a caller can name what differs. This function reads two certs and
    nothing else — no log, no runner, no environment — which is what makes it the same
    computation for a client, for a mint and for an append.

    ``environment`` is deliberately not here: hardware, runtime and region are what a challenge
    is *for* (section 9), and coverage, not validity, is where they land (section 5.4).
    """
    return comparable(target if isinstance(target, dict) else {}, own if isinstance(own, dict) else {})


def _dotted(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def skew_fields(a: dict, b: dict) -> list[str]:
    """The skew-gating fields (section 2.3) that differ: harness.version, env_lock_sha256."""
    ra = a.get("recipe", {}) if isinstance(a, dict) else {}
    rb = b.get("recipe", {}) if isinstance(b, dict) else {}
    return [name for name in SKEW_FIELDS if _dotted(ra, name) != _dotted(rb, name)]
