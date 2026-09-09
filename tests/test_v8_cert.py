"""Tests for styxx.v8.cert -- the styxx/8.0 cert envelope.

Contract: `styxx/v8/INTERFACES_layer2.md` section 1. Spec: `papers/v8/SPEC_v8_v0.2_draft.md`
sections 2, 2.1-2.5, 7.2 and Appendix A.1.

What is pinned here:

1. `sign` -> `check` round trips for every one of the eight types with minimal valid bodies,
   and for every `kind` the battery, prereg and result schemas define.
2. One negative case per failure `check` can report, with a classifier over the reason strings
   (`reason_kind`) and a completeness assertion: the negative table's declared reasons are
   exactly the taxonomy, and no reason `check` emits escapes the classifier.
3. The two content-hash paths (`body/items_blob`, `body/commitment`) are exempt from the
   embedded-id rule and every other embedded id is not.
4. `seal_commitment` against a hand-built preimage, and a sealed prereg with its reveal.
5. `identity_fields` / `recipe_core` / `comparable` / `skew_fields`, including the cross-subject
   case where only `precision` or `revision` differs.

Nothing here reads the log: `check` is a function of the cert's bytes, and whether a ref
resolves is the log's question (section 8).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from styxx.v8 import cert, keys
from styxx.v8.consts import (
    BATTERY_KINDS,
    CERT_TAG,
    ID_RE,
    PREREG_KINDS,
    REF_ROLES,
    RESULT_KINDS,
    SCHEMA_VERSION,
    SEAL_TAG,
    TYPES,
)
from styxx.v8.jcs import canonical_bytes

from tests import v8_fixtures as F  # tests/ is a package; import it by its package path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "styxx" / "v8" / "schema"


# ============================================================ reason taxonomy

SCHEMA = "schema"
VERSION = "version"
ID_UNCOMPUTABLE = "id-uncomputable"
ID_MISMATCH = "id-mismatch"
KEY_MISSING = "key-missing"
KEY_BAD = "key-bad"
SIG_MISSING = "sig-missing"
SIG_UNDECODABLE = "sig-undecodable"
SIG_UNVERIFIABLE = "sig-unverifiable"
SIG_BAD = "sig-bad"
MATERIALS = "materials"
REFS_MISSING = "refs-missing"
REFS_ROLE = "refs-role"
NUMBER = "number"
NOT_OBJECT = "not-object"

TAXONOMY = frozenset({
    SCHEMA, VERSION, ID_UNCOMPUTABLE, ID_MISMATCH, KEY_MISSING, KEY_BAD, SIG_MISSING,
    SIG_UNDECODABLE, SIG_UNVERIFIABLE, SIG_BAD, MATERIALS, REFS_MISSING, REFS_ROLE, NUMBER,
    NOT_OBJECT,
})


def reason_kind(reason: str) -> str:
    """Classify one `check` reason. Raises when a reason escapes the taxonomy.

    An unclassified reason is a finding, not a nuisance: it means `check` grew a failure mode
    that no negative test covers.
    """
    table = [
        ("schema[", SCHEMA),
        ("version: ", VERSION),
        ("id: canonical bytes unavailable", ID_UNCOMPUTABLE),
        ("id: does not recompute", ID_MISMATCH),
        ("issuer.key: missing", KEY_MISSING),
        ("issuer.key: ", KEY_BAD),
        ("sig: missing", SIG_MISSING),
        ("sig: does not decode", SIG_UNDECODABLE),
        ("sig: not verifiable", SIG_UNVERIFIABLE),
        ("sig: does not verify", SIG_BAD),
        ("materials: ", MATERIALS),
        ("refs: embedded id", REFS_MISSING),
        ("refs: role", REFS_ROLE),
        ("number: ", NUMBER),
        ("cert: not a JSON object", NOT_OBJECT),
    ]
    for prefix, kind in table:
        if reason.startswith(prefix):
            return kind
    raise AssertionError(f"cert.check emitted a reason no test classifies: {reason!r}")


def kinds_of(result) -> set[str]:
    return {reason_kind(r) for r in result.reasons}


# ============================================================ helpers

def force_envelope(core: dict, *, seed: bytes | None = None) -> dict:
    """Attach `id` and `sig` without `sign`'s issuer check.

    `sign` refuses a seed that is not the issuer's, and it cannot run at all on a core whose
    canonical bytes do not exist (NaN, an out-of-range int). These cases still have to reach
    `check`, so the envelope is closed by hand: the recomputed id when there is one, an
    all-zero placeholder id when there is not, and a signature that decodes but does not verify.
    """
    out = dict(core)
    try:
        out["id"] = cert.compute_id(core)
    except (TypeError, ValueError):
        out["id"] = "sha256:" + "0" * 64
    if seed is None:
        out["sig"] = keys.encode_signature(bytes(64))
    else:
        digest = bytes.fromhex(out["id"].split(":", 1)[1])
        out["sig"] = keys.encode_signature(keys.sign(seed, keys.tagged(CERT_TAG, digest)))
    return out


def signed_fingerprint() -> dict:
    return F.make_cert("fingerprint")


# ============================================================ 1. round trips

@pytest.mark.parametrize("cert_type", TYPES)
def test_round_trip_sign_then_check(cert_type):
    c = F.make_cert(cert_type)
    result = cert.check(c)
    assert result.reasons == []
    assert result.ok is True
    assert result.type == cert_type
    assert result.id == c["id"]
    assert re.match(ID_RE, c["id"])
    assert c["styxx"] == SCHEMA_VERSION


def test_the_round_trip_covers_all_eight_types():
    """The parametrisation above is the whole type set, not a subset of it."""
    assert len(TYPES) == 8
    assert set(TYPES) == {
        "fingerprint", "battery", "prereg", "result", "promotion", "action", "challenge", "sublog",
    }


@pytest.mark.parametrize("kind", BATTERY_KINDS)
def test_battery_round_trips_every_kind(kind):
    if kind == "canary-v1":
        c = F.make_cert(
            "battery",
            body=F.canary_battery_body(),
            recipe=F.recipe(battery=F.POOL_ID),
            refs=[{"role": "pool", "id": F.POOL_ID},
                  {"role": "selected_against", "id": F.FINGERPRINT_ID}],
        )
    else:
        c = F.make_cert("battery", body=F.battery_body(kind))
    assert cert.check(c).reasons == []


@pytest.mark.parametrize("kind", PREREG_KINDS)
def test_prereg_round_trips_every_kind(kind):
    # A noise-plan names R (section 5.1 step 1); schema/prereg.json requires it on that kind
    # since A-NORUNS, so the minimal body of THAT kind carries it and a study's does not.
    body = {"kind": kind, "runs": 5} if kind == "noise-plan" else {"kind": kind}
    c = F.make_cert("prereg", body=body)
    assert cert.check(c).reasons == []


def _result_body(kind: str) -> dict:
    if kind in ("confirmatory", "pilot", "robustness", "sensitivity"):
        return {"kind": kind, "deviations": []}
    if kind == "verify":
        return F.verify_result_body()
    if kind == "response":
        return {
            "kind": "response",
            "challenge": F.CHALLENGE_ID,
            "disposition": "new_floor",
            "detail": "the floor was measured on one box and did not cover this one",
        }
    if kind == "document":
        return {
            "kind": "document",
            "path": "papers/v8/SPEC_v8_v0.2_draft.md",
            "commit": "0" * 40,
            "git_blob_sha256": "1" * 40,
            "eol": "lf",
        }
    raise AssertionError(f"no body for result kind {kind!r}")


@pytest.mark.parametrize("kind", RESULT_KINDS)
def test_result_round_trips_every_kind(kind):
    body = _result_body(kind)
    refs = [{"role": "target", "id": F.CHALLENGE_ID}] if kind == "response" else []
    c = F.make_cert("result", body=body, refs=refs)
    assert cert.check(c).reasons == []


def test_alias_subject_without_logprobs_round_trips():
    """A black-box tier fingerprint: alias subject, seqlp and topk absent (section 2.2)."""
    c = F.make_cert(
        "fingerprint",
        subject=F.alias_subject(),
        body=F.fingerprint_body(logprobs=False, tier="black-box"),
    )
    assert cert.check(c).reasons == []
    assert c["body"]["channels"]["seqlp"]["present"] is False


def test_redacted_fingerprint_round_trips_without_output_text():
    c = F.make_cert("fingerprint", body=F.fingerprint_body(redacted=True))
    assert cert.check(c).reasons == []
    assert all("output_text" not in i for i in c["body"]["items"])
    assert all(len(i["output_sha256"]) == 64 for i in c["body"]["items"])


def test_fingerprint_with_floor_and_sensitivity_round_trips():
    refs = [{"role": "battery", "id": F.BATTERY_ID},
            {"role": "noise_plan", "id": F.NOISE_PLAN_ID},
            {"role": "sensitivity", "id": F.SENSITIVITY_ID},
            {"role": "previous", "id": F.PREV_FINGERPRINT_ID}]
    refs += [{"role": "run", "id": rid} for rid in F.RUN_IDS]
    c = F.make_cert(
        "fingerprint",
        body=F.fingerprint_body(
            items_blob=F.ITEMS_BLOB_ID,
            noise_floor=F.noise_floor_block(),
            sensitivity=F.SENSITIVITY_ID,
        ),
        refs=refs,
    )
    assert cert.check(c).reasons == []


# ============================================================ 2. id, signature, digest

def test_sign_returns_a_new_dict_and_does_not_mutate_the_input():
    core = F.unsigned("prereg")
    before = copy.deepcopy(core)
    out = cert.sign(core, F.keypair()[0])
    assert core == before
    assert "id" not in core and "sig" not in core
    assert out is not core


def test_sign_emits_the_envelope_keys_in_order():
    c = F.make_cert("prereg")
    assert list(c) == list(cert.ENVELOPE_KEYS)


def test_sign_refuses_a_seed_that_is_not_the_issuer_key():
    core = F.unsigned("prereg")
    other_seed = F.keypair("other")[0]
    with pytest.raises(ValueError, match="not the public key of the signing seed"):
        cert.sign(core, other_seed)


def test_id_is_the_digest_of_the_core_without_id_and_sig():
    c = F.make_cert("prereg")
    core = {k: v for k, v in c.items() if k not in ("id", "sig")}
    assert c["id"] == "sha256:" + hashlib.sha256(canonical_bytes(core)).hexdigest()
    assert cert.compute_id(c) == c["id"]
    assert cert.digest_bytes(c) == bytes.fromhex(c["id"].split(":", 1)[1])
    assert len(cert.digest_bytes(c)) == 32


def test_digest_ignores_the_id_and_sig_fields_themselves():
    c = F.make_cert("prereg")
    d = cert.digest_bytes(c)
    mutated = dict(c, id="sha256:" + "0" * 64, sig=keys.encode_signature(bytes(64)))
    assert cert.digest_bytes(mutated) == d


def test_digest_is_insensitive_to_python_key_insertion_order():
    c = F.make_cert("prereg")
    shuffled = {k: c[k] for k in sorted(c, reverse=True)}
    assert cert.digest_bytes(shuffled) == cert.digest_bytes(c)


def test_the_signature_is_over_the_tagged_digest():
    c = F.make_cert("prereg")
    seed, public = F.keypair()
    signature = keys.decode_signature(c["sig"])
    digest = cert.digest_bytes(c)
    assert keys.verify(public, keys.tagged(CERT_TAG, digest), signature)
    # The tag is what stops a cert signature standing as a tree head or a seal (section 2.1).
    assert not keys.verify(public, keys.tagged("styxx.v8/sth/1", digest), signature)
    assert not keys.verify(public, digest, signature)


def test_signing_is_deterministic_so_fixture_certs_are_reproducible():
    assert F.make_cert("prereg") == F.make_cert("prereg")


def test_evidence_id_is_shared_by_two_issuers_of_the_same_evidence():
    a = F.make_cert("fingerprint")
    b = F.make_cert("fingerprint", issuer_label="other")
    assert cert.evidence_id(a) == cert.evidence_id(b)
    assert a["id"] != b["id"]
    assert re.match(ID_RE, cert.evidence_id(a))


def test_evidence_id_changes_with_the_body():
    a = F.make_cert("fingerprint")
    b = F.make_cert("fingerprint", body=F.fingerprint_body(run_index=1))
    assert cert.evidence_id(a) != cert.evidence_id(b)


# ============================================================ 3. one negative per reason

def _tampered_byte():
    c = signed_fingerprint()
    t = copy.deepcopy(c)
    t["body"]["items"][0]["n_generated"] += 1
    return t


def _wrong_key():
    """A cert whose id recomputes but whose signature is another issuer's, over other bytes."""
    a = F.make_cert("fingerprint")
    b = F.make_cert("fingerprint", issuer_label="other")
    return dict(b, sig=a["sig"])


def _small_order_key():
    core = F.unsigned("fingerprint")
    core["issuer"] = {"name": F.ISSUER_NAME, "key": keys.encode_public(bytes(32))}
    return force_envelope(core)


def _key_with_padding():
    core = F.unsigned("fingerprint")
    core["issuer"] = {"name": F.ISSUER_NAME, "key": F.public_key() + "="}
    return force_envelope(core)


def _key_not_on_the_curve():
    # y = 2 does not yield an x on the curve (RFC 8032 section 5.1.3 x recovery fails).
    core = F.unsigned("fingerprint")
    core["issuer"] = {"name": F.ISSUER_NAME, "key": keys.encode_public((2).to_bytes(32, "little"))}
    return force_envelope(core)


def _issuer_key_missing():
    core = F.unsigned("fingerprint")
    core["issuer"] = {"name": F.ISSUER_NAME}
    return force_envelope(core)


def _sig_missing():
    c = signed_fingerprint()
    del c["sig"]
    return c


def _sig_undecodable():
    c = signed_fingerprint()
    c["sig"] = "ed25519:not+valid+base64url"
    return c


def _missing_ref():
    return F.make_cert("fingerprint", refs=[])


def _ref_role_invalid_for_the_type():
    return F.make_cert(
        "fingerprint",
        refs=[{"role": "battery", "id": F.BATTERY_ID}, {"role": "parent", "id": F.PARENT_ID}],
    )


def _material_mismatch(field: str):
    def build():
        return F.make_cert("fingerprint", recipe=F.recipe(**{field: "0" * 64}))
    return build


def _future_version(value: str):
    def build():
        return F.make_cert("fingerprint", styxx=value)
    return build


def _nan():
    core = F.unsigned("fingerprint")
    core["body"]["items"][0]["seq_logprob"] = float("nan")
    return force_envelope(core)


def _infinity():
    core = F.unsigned("fingerprint")
    core["body"]["items"][0]["seq_logprob"] = float("inf")
    return force_envelope(core)


def _int_out_of_range():
    core = F.unsigned("fingerprint")
    core["body"]["items"][0]["token_ids"] = [2**53 + 1]
    return force_envelope(core)


def _bad_created():
    return F.make_cert("fingerprint", created="2026-09-08 18:00:00Z")


def _id_replaced():
    c = signed_fingerprint()
    c["id"] = "sha256:" + "0" * 64
    return c


def _public_is_not_an_envelope_field():
    return F.make_cert("fingerprint", public=True)


def _role_outside_the_enum():
    return F.make_cert(
        "fingerprint",
        refs=[{"role": "battery", "id": F.BATTERY_ID}, {"role": "bogus", "id": F.PARENT_ID}],
    )


def _not_a_json_object():
    return "sha256:" + "0" * 64


NEGATIVE_CASES = [
    ("tampered byte", _tampered_byte, {ID_MISMATCH, SIG_BAD}),
    ("wrong key", _wrong_key, {SIG_BAD}),
    ("small-order issuer key", _small_order_key, {KEY_BAD, SIG_UNVERIFIABLE}),
    ("issuer key with base64 padding", _key_with_padding, {KEY_BAD}),
    ("issuer key not on the curve", _key_not_on_the_curve, {KEY_BAD}),
    ("issuer key missing", _issuer_key_missing, {KEY_MISSING, SCHEMA}),
    ("signature missing", _sig_missing, {SIG_MISSING, SCHEMA}),
    ("signature undecodable", _sig_undecodable, {SIG_UNDECODABLE, SCHEMA}),
    ("embedded id absent from refs", _missing_ref, {REFS_MISSING}),
    ("ref role invalid for the type", _ref_role_invalid_for_the_type, {REFS_ROLE}),
    ("chat_template hash mismatch", _material_mismatch("chat_template_sha256"), {MATERIALS}),
    ("system_prompt hash mismatch", _material_mismatch("system_prompt_sha256"), {MATERIALS}),
    ("env_lock hash mismatch", _material_mismatch("env_lock_sha256"), {MATERIALS}),
    ("future schema major", _future_version("9.0"), {VERSION}),
    ("future schema minor", _future_version("8.1"), {VERSION}),
    ("NaN", _nan, {ID_UNCOMPUTABLE, NUMBER, SIG_UNVERIFIABLE}),
    ("infinity", _infinity, {ID_UNCOMPUTABLE, NUMBER, SIG_UNVERIFIABLE}),
    ("integer beyond 2**53", _int_out_of_range, {ID_UNCOMPUTABLE, NUMBER, SIG_UNVERIFIABLE}),
    ("created is not RFC 3339 Z", _bad_created, {SCHEMA}),
    ("id replaced", _id_replaced, {ID_MISMATCH}),
    ("public is not an envelope field", _public_is_not_an_envelope_field, {SCHEMA}),
    ("ref role outside the enum", _role_outside_the_enum, {SCHEMA}),
    ("not a JSON object", _not_a_json_object, {NOT_OBJECT}),
]


@pytest.mark.parametrize(
    "label,build,expected",
    NEGATIVE_CASES,
    ids=[c[0].replace(" ", "-") for c in NEGATIVE_CASES],
)
def test_one_negative_per_check_reason(label, build, expected):
    result = cert.check(build())
    assert result.ok is False
    assert result.reasons, f"{label}: check reported no reason"
    observed = kinds_of(result)          # also asserts every reason classifies
    assert expected <= observed, f"{label}: missing {sorted(expected - observed)}"


def test_the_negative_table_covers_the_whole_reason_taxonomy():
    """Every failure `check` can report has a negative case above, and no case is dead weight."""
    declared = set()
    for _, _, expected in NEGATIVE_CASES:
        declared |= expected
    assert declared == set(TAXONOMY)


def test_ok_is_true_exactly_when_reasons_is_empty():
    good = cert.check(signed_fingerprint())
    assert good.ok is True and good.reasons == []
    bad = cert.check(_tampered_byte())
    assert bad.ok is False and bad.reasons != []


def test_check_collects_every_failure_rather_than_stopping_at_the_first_one():
    """Four independent faults in one cert produce four kinds of reason (contract section 1)."""
    core = F.unsigned("fingerprint", styxx="9.0", created="yesterday", refs=[])
    core["recipe"] = F.recipe(env_lock_sha256="0" * 64)
    c = cert.sign(core, F.keypair()[0])
    observed = kinds_of(cert.check(c))
    assert {SCHEMA, VERSION, MATERIALS, REFS_MISSING} <= observed


def test_a_lower_schema_version_is_accepted():
    assert cert.check(F.make_cert("fingerprint", styxx="7.9")).reasons == []


# ============================================================ 4. per-type schema violations

TYPE_VIOLATIONS = {
    "fingerprint": ("body/tier removed", lambda c: c["body"].pop("tier")),
    "battery": ("body/kind outside the enum", lambda c: c["body"].__setitem__("kind", "canary-v9")),
    "prereg": ("sealed without a commitment", lambda c: c["body"].__setitem__("sealed", True)),
    "result": ("verify body without its fields", lambda c: c["body"].__setitem__("kind", "verify")),
    "promotion": ("body/scope removed", lambda c: c["body"].pop("scope")),
    "action": ("action_kind outside the enum", lambda c: c["body"].__setitem__("action_kind", "sneeze")),
    "challenge": ("coverage outside the enum", lambda c: c["body"].__setitem__("coverage", "elsewhere")),
    "sublog": ("body/root_hash removed", lambda c: c["body"].pop("root_hash")),
}


@pytest.mark.parametrize("cert_type", TYPES)
def test_one_schema_violation_per_type(cert_type):
    label, mutate = TYPE_VIOLATIONS[cert_type]
    core = F.unsigned(cert_type)
    mutate(core)
    c = cert.sign(core, F.keypair()[0])
    result = cert.check(c)
    assert result.ok is False, f"{cert_type}: {label} was accepted"
    named = [r for r in result.reasons if r.startswith(f"schema[{cert_type}]")]
    assert named, f"{cert_type}: {label} was not reported by its own schema: {result.reasons}"


def test_every_type_has_a_schema_violation_case():
    assert set(TYPE_VIOLATIONS) == set(TYPES)


def test_canary_selection_fields_are_forbidden_outside_canary_v1():
    """schema/battery.json: the selection fields are required iff the kind is canary-v1."""
    body = F.battery_body("pool-v1")
    body["params"] = {"n": 3}
    c = F.make_cert("battery", body=body)
    assert cert.check(c).ok is False

    body = F.canary_battery_body()
    body.pop("excluded")
    c = F.make_cert(
        "battery",
        body=body,
        recipe=F.recipe(battery=F.POOL_ID),
        refs=[{"role": "pool", "id": F.POOL_ID},
              {"role": "selected_against", "id": F.FINGERPRINT_ID}],
    )
    assert cert.check(c).ok is False


def test_a_canary_battery_needs_a_selected_against_ref():
    c = F.make_cert(
        "battery",
        body=F.canary_battery_body(),
        recipe=F.recipe(battery=F.POOL_ID),
        refs=[{"role": "pool", "id": F.POOL_ID}],
    )
    reasons = cert.check(c).reasons
    assert any(r.startswith("schema[battery]") for r in reasons), reasons


def test_challenge_requires_both_a_target_and_an_own_ref():
    c = F.make_cert("challenge", refs=[{"role": "target", "id": F.TARGET_ID}])
    reasons = cert.check(c).reasons
    assert any(r.startswith("schema[challenge]") for r in reasons), reasons


def test_schema_errors_reports_the_envelope_before_the_type():
    core = F.unsigned("fingerprint", created="yesterday")
    core["body"].pop("tier")
    c = cert.sign(core, F.keypair()[0])
    lines = cert.schema_errors(c)
    envelope = [i for i, line in enumerate(lines) if line.startswith("schema[envelope]")]
    per_type = [i for i, line in enumerate(lines) if line.startswith("schema[fingerprint]")]
    assert envelope and per_type
    assert max(envelope) < min(per_type)


def test_schema_errors_on_a_non_object():
    assert cert.schema_errors("nope") == ["schema[envelope]: $: cert is not a JSON object"]


def test_a_schema_file_exists_for_the_envelope_and_every_type():
    names = set(cert.schema_names())
    assert {"envelope", "common", "subject", "recipe"} <= names
    assert set(TYPES) <= names


def test_schema_files_are_lf_utf8_without_a_bom():
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), path.name
        assert b"\r\n" not in data, path.name
        schema = json.loads(data.decode("utf-8"))
        assert schema["$id"] == f"urn:styxx:v8:schema:{path.stem}"


# ============================================================ 5. embedded ids and refs

def test_every_embedded_id_outside_the_content_hash_paths_needs_a_ref():
    c = F.make_cert("fingerprint", body=F.fingerprint_body(sensitivity=F.SENSITIVITY_ID))
    reasons = cert.check(c).reasons
    assert [r for r in reasons if reason_kind(r) == REFS_MISSING]
    assert F.SENSITIVITY_ID in " ".join(reasons)


def test_items_blob_is_a_blob_address_and_not_a_ref():
    """`body.items_blob` addresses stored bytes (section 3.1), so no ref answers for it."""
    c = F.make_cert("fingerprint", body=F.fingerprint_body(items_blob=F.ITEMS_BLOB_ID))
    assert cert.check(c).reasons == []
    assert F.ITEMS_BLOB_ID not in {rid for _, rid in cert.refs(c)}


def test_a_seal_commitment_is_a_commitment_and_not_a_ref():
    """`body.commitment` is sha256 over a salted body (section 7.2), not a cert id."""
    c = F.make_cert("prereg", body=F.sealed_prereg_body())
    assert cert.check(c).reasons == []
    assert cert.refs(c) == []


def test_embedded_ids_stays_literal():
    """`embedded_ids` reports every ID_RE value; the exemptions live in `check`, not here."""
    c = F.make_cert("fingerprint", body=F.fingerprint_body(items_blob=F.ITEMS_BLOB_ID))
    found = cert.embedded_ids(c)
    assert F.ITEMS_BLOB_ID in found
    assert F.BATTERY_ID in found
    assert c["id"] not in found          # the envelope's own id is not "embedded"
    assert cert.embedded_ids("nope") == set()


def test_refs_returns_role_id_pairs_in_cert_order():
    refs = [{"role": "battery", "id": F.BATTERY_ID}, {"role": "run", "id": F.RUN_IDS[0]}]
    c = F.make_cert("fingerprint", refs=refs)
    assert cert.refs(c) == [("battery", F.BATTERY_ID), ("run", F.RUN_IDS[0])]


def test_refs_skips_malformed_entries_and_leaves_them_to_the_schema():
    c = {"refs": [{"role": "battery", "id": F.BATTERY_ID}, {"role": 7, "id": F.POOL_ID}, "nope", {}]}
    assert cert.refs(c) == [("battery", F.BATTERY_ID)]
    assert cert.refs({}) == []
    assert cert.refs({"refs": "nope"}) == []


def test_the_role_tables_cover_every_role_in_the_contract():
    assert set(cert.ROLE_TYPES) == set(REF_ROLES)
    union = set()
    for roles in cert.ROLES_BY_TYPE.values():
        union |= set(roles)
    assert union <= set(REF_ROLES)
    assert set(cert.ROLES_BY_TYPE) == set(TYPES)


@pytest.mark.parametrize("role", sorted(cert.ROLES_BY_TYPE["fingerprint"]))
def test_a_role_allowed_on_a_fingerprint_is_accepted(role):
    # The battery ref is the ladder's base and is what answers `recipe.battery`. For
    # role == "battery" it IS the ref under test, named once: naming it twice is its own refusal
    # (`cert.duplicate_roles`, C-DUPREF of papers/v8/challenge_and_attack_2026_09_09), and naming
    # a different id once leaves `recipe.battery` unanswered, which is a refs-missing refusal.
    # Neither is what this test is about, so the battery case names the recipe's own battery.
    rid = F.BATTERY_ID if role == "battery" else F.fake_id("role-" + role)
    base = [] if role == "battery" else [{"role": "battery", "id": F.BATTERY_ID}]
    c = F.make_cert("fingerprint", refs=base + [{"role": role, "id": rid}])
    assert [r for r in cert.check(c).reasons if reason_kind(r) == REFS_ROLE] == []


# ============================================================ 6. issuer key validation

def test_public_key_reason_accepts_a_real_key_and_names_every_refusal():
    assert cert.public_key_reason(F.public_key()) is None
    cases = {
        "padding": F.public_key() + "=",
        "prefix": "ed25519x:" + "A" * 43,
        "length": "ed25519:" + "A" * 42,
        "non-canonical y": keys.encode_public((2**255 - 19).to_bytes(32, "little")),
        "small order": keys.encode_public(bytes(32)),
        "off curve": keys.encode_public((2).to_bytes(32, "little")),
    }
    for label, encoded in cases.items():
        why = cert.public_key_reason(encoded)
        assert isinstance(why, str) and why, label
    assert "small-order" in cert.public_key_reason(keys.encode_public(bytes(32)))


def test_a_small_order_key_cannot_be_rescued_by_a_matching_signature():
    """Under a small-order key one string verifies over many messages, so `check` refuses the
    key itself and never reports the signature as the thing that went wrong (section 2.1)."""
    c = _small_order_key()
    reasons = cert.check(c).reasons
    assert [r for r in reasons if reason_kind(r) == KEY_BAD]
    assert not [r for r in reasons if reason_kind(r) == SIG_BAD]


# ============================================================ 7. version_ok (section 2.5)

@pytest.mark.parametrize("value,expected", [
    ("8.0", True),
    ("7.9", True),
    ("0.1", True),
    ("8.1", False),
    ("8.10", False),
    ("9.0", False),
    ("10.0", False),
    ("8", False),
    ("8.0.1", False),
    ("v8.0", False),
    ("", False),
])
def test_version_ok(value, expected):
    assert cert.version_ok({"styxx": value}) is expected


def test_version_ok_against_another_verifier_version():
    assert cert.version_ok({"styxx": "8.1"}, own="8.2") is True
    assert cert.version_ok({"styxx": "8.3"}, own="8.2") is False
    assert cert.version_ok({"styxx": "8.0"}, own="nonsense") is False


def test_version_ok_on_a_missing_or_non_string_field():
    assert cert.version_ok({}) is False
    assert cert.version_ok({"styxx": 8.0}) is False
    assert cert.version_ok("nope") is False


# ============================================================ 8. seal (section 7.2)

def test_seal_commitment_matches_the_hand_built_preimage():
    body = F.reveal_prereg_body()
    preimage = SEAL_TAG.encode("ascii") + b"\x00" + F.SALT + canonical_bytes(body)
    assert cert.seal_commitment(F.SALT, body) == "sha256:" + hashlib.sha256(preimage).hexdigest()
    assert SEAL_TAG == "styxx.v8/seal/1"


def test_seal_commitment_is_deterministic_and_prefixed():
    body = F.reveal_prereg_body()
    once = cert.seal_commitment(F.SALT, body)
    assert once == cert.seal_commitment(F.SALT, copy.deepcopy(body))
    assert re.match(ID_RE, once)


def test_seal_commitment_moves_with_the_salt_and_with_the_body():
    body = F.reveal_prereg_body()
    base = cert.seal_commitment(F.SALT, body)
    assert cert.seal_commitment(bytes(32), body) != base
    assert cert.seal_commitment(F.SALT, dict(body, kind="noise-plan")) != base
    other = copy.deepcopy(body)
    other["hypotheses"][0]["direction"] = "less"
    assert cert.seal_commitment(F.SALT, other) != base


def test_seal_commitment_refuses_a_salt_that_is_not_32_bytes():
    body = F.reveal_prereg_body()
    for bad in (b"", bytes(31), bytes(33)):
        with pytest.raises(ValueError, match="32 bytes"):
            cert.seal_commitment(bad, body)
    with pytest.raises(TypeError):
        cert.seal_commitment("x" * 32, body)
    with pytest.raises(TypeError):
        cert.seal_commitment(F.SALT, ["not", "a", "body"])


def test_seal_commitment_accepts_a_bytearray_salt():
    body = F.reveal_prereg_body()
    assert cert.seal_commitment(bytearray(F.SALT), body) == cert.seal_commitment(F.SALT, body)


def test_the_seal_tag_separates_a_commitment_from_a_cert_digest():
    """The same 32 bytes under the cert tag and the seal tag are different preimages."""
    body = {"kind": "study"}
    salted = SEAL_TAG.encode("ascii") + b"\x00" + F.SALT + canonical_bytes(body)
    certish = CERT_TAG.encode("ascii") + b"\x00" + F.SALT + canonical_bytes(body)
    assert salted != certish
    assert SEAL_TAG != CERT_TAG


def test_a_sealed_prereg_and_its_reveal():
    """Section 7.2: the reveal names the sealed cert and carries the salt and the full body."""
    revealed = F.reveal_prereg_body()
    sealed = F.make_cert("prereg", body=F.sealed_prereg_body(revealed))
    assert cert.check(sealed).reasons == []
    assert sealed["body"]["sealed"] is True

    reveal = F.make_cert(
        "prereg",
        body=dict(revealed, salt=F.salt_b64()),
        refs=[{"role": "sealed", "id": sealed["id"]}],
    )
    assert cert.check(reveal).reasons == []
    assert cert.refs(reveal) == [("sealed", sealed["id"])]

    # A validator recomputes the commitment from the revealed body and the salt.
    salt = _decode_salt(reveal["body"]["salt"])
    recomputed = cert.seal_commitment(salt, {k: v for k, v in reveal["body"].items() if k != "salt"})
    assert recomputed == sealed["body"]["commitment"]


def test_a_reveal_that_does_not_match_its_commitment_is_detectable():
    revealed = F.reveal_prereg_body()
    sealed = F.make_cert("prereg", body=F.sealed_prereg_body(revealed))
    tampered = copy.deepcopy(revealed)
    tampered["hypotheses"][0]["direction"] = "less"
    assert cert.seal_commitment(F.SALT, tampered) != sealed["body"]["commitment"]


def _decode_salt(text: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def test_the_fixture_salt_encodes_as_43_base64url_characters():
    assert len(F.salt_b64()) == 43
    assert _decode_salt(F.salt_b64()) == F.SALT
    assert re.match(r"^[A-Za-z0-9_-]{43}$", F.salt_b64())


# ============================================================ 9. comparability keys

def test_identity_fields_for_a_weights_subject():
    got = cert.identity_fields(F.weights_subject())
    assert list(got) == [
        "kind", "hf_repo", "revision", "weights_sha256", "config_sha256", "tokenizer_sha256",
        "generation_config_sha256", "precision",
    ]
    assert "environment" not in got
    assert "model_family" not in got


def test_identity_fields_for_an_alias_subject():
    got = cert.identity_fields(F.alias_subject(observed_model_id="acme-large-2026-09", observed_at=F.CREATED))
    assert list(got) == ["kind", "provider", "alias", "region"]
    assert "observed_model_id" not in got
    assert "observed_at" not in got


def test_identity_fields_on_an_empty_or_unknown_subject():
    assert cert.identity_fields({}) == {"kind": None}
    assert cert.identity_fields({"kind": "mystery"}) == {"kind": "mystery"}
    assert cert.identity_fields("nope") == {"kind": None}


def test_identity_fields_returns_a_copy():
    subject = F.weights_subject()
    got = cert.identity_fields(subject)
    got["hf_repo"] = "someone/else"
    assert subject["hf_repo"] == F.WEIGHTS_SUBJECT["hf_repo"]


def test_recipe_core_is_exactly_the_four_fields():
    got = cert.recipe_core(F.recipe())
    assert list(got) == ["battery", "decoding", "chat_template_sha256", "system_prompt_sha256"]
    assert got["battery"] == F.BATTERY_ID
    assert cert.recipe_core({}) == {k: None for k in cert.RECIPE_CORE_FIELDS}
    assert cert.recipe_core("nope") == {k: None for k in cert.RECIPE_CORE_FIELDS}


def test_recipe_core_excludes_the_skew_and_material_fields():
    got = cert.recipe_core(F.recipe())
    for name in ("harness", "env_lock_sha256", "materials", "sae"):
        assert name not in got


def _pair(subject_b=None, recipe_b=None):
    a = F.make_cert("fingerprint")
    b = F.make_cert(
        "fingerprint",
        subject=subject_b if subject_b is not None else F.weights_subject(),
        recipe=recipe_b if recipe_b is not None else F.recipe(),
    )
    return a, b


def test_comparable_is_empty_for_two_certs_of_the_same_subject_and_recipe():
    a, b = _pair()
    assert cert.comparable(a, b) == []
    assert cert.comparable(a, a) == []


def test_comparable_names_an_identity_difference():
    a, b = _pair(subject_b=F.weights_subject(weights_sha256="9" * 64))
    assert cert.comparable(a, b) == ["subject.weights_sha256"]


def test_comparable_names_a_recipe_core_difference():
    a, b = _pair(recipe_b=F.recipe(decoding=F.decoding(batch_size=8)))
    assert cert.comparable(a, b) == ["recipe.decoding"]
    a, b = _pair(recipe_b=F.recipe(battery=F.POOL_ID))
    assert cert.comparable(a, b) == ["recipe.battery"]


def test_comparable_ignores_the_environment_and_the_harness():
    subject = F.weights_subject()
    subject["environment"] = {
        "runtime": {"framework": "transformers", "version": "5.0.0", "backend": "torch 2.9.0+cu128"},
        "hardware": {"gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "driver": "580.00", "count": 1},
    }
    a, b = _pair(subject_b=subject, recipe_b=F.recipe(
        harness={"name": "styxx", "version": "8.0.1", "commit": "1" * 40},
        env_lock_sha256=F.sha256_text("a different lockfile"),
    ))
    assert cert.comparable(a, b) == []


def test_comparable_labels_a_precision_difference_as_cross_subject():
    """Section 2.3: a `--diff` may cross precision; it is labelled, not refused."""
    a, b = _pair(subject_b=F.weights_subject(precision="fp16"))
    assert cert.comparable(a, b) == ["cross-subject:precision"]


def test_comparable_labels_a_revision_difference_as_cross_subject():
    a, b = _pair(subject_b=F.weights_subject(revision="0" * 40))
    assert cert.comparable(a, b) == ["cross-subject:revision"]


def test_a_cross_subject_pair_carries_no_plain_mismatch():
    a, b = _pair(subject_b=F.weights_subject(precision="fp16", revision="0" * 40))
    entries = cert.comparable(a, b)
    assert sorted(entries) == ["cross-subject:precision", "cross-subject:revision"]
    assert not [e for e in entries if not e.startswith("cross-subject:")]


def test_a_cross_subject_label_does_not_hide_a_real_mismatch():
    a, b = _pair(subject_b=F.weights_subject(precision="fp16", weights_sha256="9" * 64))
    entries = cert.comparable(a, b)
    assert "cross-subject:precision" in entries
    assert "subject.weights_sha256" in entries


def test_a_challenge_is_valid_only_against_its_own_subject():
    """Section 9 rule 1, the defect C1 of papers/v8/challenge_and_attack_2026_09_09.

    An fp16 fingerprint filed as the `own` half of a challenge against a bf16 target was taken
    by every layer. `comparable` already knew; nothing asked it.
    """
    a, b = _pair(subject_b=F.weights_subject(precision="fp16"))
    assert cert.challenge_validity(a, b) == ["cross-subject:precision"]
    assert cert.challenge_validity(a, a) == []
    # a cross-subject pair is a legitimate --diff and is NOT a challenge: section 9 wants every
    # S_identity field equal, and precision is one (section 2.2).
    assert cert.comparable(a, b) == cert.challenge_validity(a, b)


def test_a_challenge_from_a_different_recipe_or_a_different_battery_is_not_a_challenge():
    a, b = _pair(recipe_b=F.recipe(battery=F.POOL_ID))
    assert cert.challenge_validity(a, b) == ["recipe.battery"]
    a, b = _pair(subject_b=F.weights_subject(weights_sha256="9" * 64))
    assert cert.challenge_validity(a, b) == ["subject.weights_sha256"]


def test_challenge_validity_ignores_the_environment_a_challenge_exists_to_differ_on():
    """Hardware, driver and runtime are what a challenge is FOR (section 9); coverage, not
    validity, is where they land (section 5.4)."""
    subject = F.weights_subject()
    subject["environment"] = {
        "runtime": {"framework": "transformers", "version": "5.0.0", "backend": "torch 2.9"},
        "hardware": {"gpu": "NVIDIA RTX 4090", "driver": "999.99", "count": 1},
    }
    a, b = _pair(subject_b=subject)
    assert cert.challenge_validity(a, b) == []


def test_challenge_validity_never_raises_on_hostile_input():
    assert cert.challenge_validity({}, {}) == []
    assert cert.challenge_validity(None, None) == []
    assert cert.challenge_validity({"subject": "not a dict"}, {"subject": []}) == []


def test_comparable_across_subject_kinds_names_the_kind():
    a, b = _pair(subject_b=F.alias_subject())
    entries = cert.comparable(a, b)
    assert "subject.kind" in entries
    assert "subject.provider" in entries
    assert [e for e in entries if not e.startswith("cross-subject:")] != []


def test_comparable_between_two_alias_subjects():
    a = F.make_cert("fingerprint", subject=F.alias_subject())
    b = F.make_cert("fingerprint", subject=F.alias_subject(region="us"))
    assert cert.comparable(a, b) == ["subject.region"]
    c = F.make_cert("fingerprint", subject=F.alias_subject(observed_model_id="acme-large-2026-09"))
    assert cert.comparable(a, c) == []


def test_skew_fields_names_the_two_gating_fields():
    assert cert.SKEW_FIELDS == ("harness.version", "env_lock_sha256")
    a, b = _pair()
    assert cert.skew_fields(a, b) == []

    a, b = _pair(recipe_b=F.recipe(harness={"name": "styxx", "version": "8.0.1", "commit": "1" * 40}))
    assert cert.skew_fields(a, b) == ["harness.version"]

    a, b = _pair(recipe_b=F.recipe(env_lock_sha256=F.sha256_text("another lockfile")))
    assert cert.skew_fields(a, b) == ["env_lock_sha256"]

    a, b = _pair(recipe_b=F.recipe(
        harness={"name": "styxx", "version": "8.0.1", "commit": "1" * 40},
        env_lock_sha256=F.sha256_text("another lockfile"),
    ))
    assert cert.skew_fields(a, b) == ["harness.version", "env_lock_sha256"]


def test_skew_and_comparability_are_separate_questions():
    """A verifier change is skew, never a comparability failure (section 2.3)."""
    a, b = _pair(recipe_b=F.recipe(harness={"name": "styxx", "version": "9.9.9", "commit": "2" * 40}))
    assert cert.comparable(a, b) == []
    assert cert.skew_fields(a, b) == ["harness.version"]


def test_skew_fields_on_certs_without_recipes():
    a = F.make_cert("prereg")
    b = F.make_cert("prereg")
    assert cert.skew_fields(a, b) == []
    assert cert.skew_fields("nope", b) == []


# ============================================================ 10. material hashes (A.1)

def test_material_hash_is_sha256_of_the_utf8_bytes():
    assert cert.material_hash("") == hashlib.sha256(b"").hexdigest()
    assert cert.material_hash("euro sign €") == hashlib.sha256("euro sign €".encode("utf-8")).hexdigest()
    assert len(cert.material_hash("x")) == 64
    with pytest.raises(TypeError):
        cert.material_hash(b"already bytes")


def test_the_fixture_recipe_hashes_match_its_materials():
    r = F.recipe()
    for source, hashed in cert.MATERIAL_HASHES:
        assert r[hashed] == cert.material_hash(r["materials"][source])


def test_a_material_hash_is_checked_even_when_the_text_is_empty():
    r = F.recipe(system_prompt_sha256=cert.material_hash("not empty"))
    c = F.make_cert("fingerprint", recipe=r)
    assert [x for x in cert.check(c).reasons if reason_kind(x) == MATERIALS]


def test_materials_are_not_checked_when_the_recipe_is_absent():
    assert cert.check(F.make_cert("prereg")).reasons == []


# ============================================================ 11. hygiene

def test_no_claim_word_in_the_module_or_the_schemas():
    """Contract global rule: never `immutable`, `tamper-proof`, `self-verifying`, `first`."""
    banned = re.compile(r"immutable|tamper[- ]proof|self[- ]verifying|\bfirst\b", re.IGNORECASE)
    targets = [ROOT / "styxx" / "v8" / "cert.py", Path(__file__).parent / "v8_fixtures.py"]
    targets += sorted(SCHEMA_DIR.glob("*.json"))
    for path in targets:
        text = path.read_bytes().decode("utf-8")
        found = banned.findall(text)
        assert not found, f"{path.name} carries claim words: {sorted(set(found))}"


def test_cert_imports_without_torch():
    import subprocess
    import sys
    code = "import styxx.v8.cert, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_the_fixture_module_builds_a_valid_cert_for_every_type():
    """`v8_fixtures` is imported by every later v8 test file; a broken default breaks them all."""
    for cert_type in TYPES:
        assert cert.check(F.make_cert(cert_type)).ok is True, cert_type
        core = F.make_cert(cert_type, signed=False)
        assert "id" not in core and "sig" not in core


def test_the_fixture_keys_are_distinct_and_stable():
    a_seed, a_pub = F.keypair()
    b_seed, b_pub = F.keypair("other")
    assert a_seed != b_seed and a_pub != b_pub
    assert len(a_seed) == 32 and len(a_pub) == 32
    assert F.keypair() == (a_seed, a_pub)
    assert cert.public_key_reason(F.public_key()) is None
    assert cert.public_key_reason(F.public_key("other")) is None
