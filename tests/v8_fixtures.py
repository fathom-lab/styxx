"""Shared cert fixtures for the styxx.v8 test files.

Contract: `styxx/v8/INTERFACES_layer2.md` section 1 ("a fixture module `tests/v8_fixtures.py`
exporting `make_cert(type, **overrides)`, `keypair()`, and minimal valid subjects/recipes for
weights and alias -- every later test file imports it").

Everything here is DETERMINISTIC: the keys are derived from a label with SHA-256, the placeholder
cert ids are derived from a label, and `CREATED` is fixed. Two runs of the same test produce
byte-identical certs, which is what the conformance recorder (contract section 10) needs.

The stable names, all importable from this module:

    keypair(label="issuer")     -> (private_seed_32, public_32)
    public_key(label="issuer")  -> "ed25519:<43 base64url chars>"
    issuer(label="issuer")      -> {"name": ..., "key": ...}
    fake_id(label)              -> "sha256:<64 hex>"        (a placeholder cert id)
    sha256_text(text)           -> "<64 hex>"               (Appendix A.1 material hash)

    ISSUER_NAME, CREATED, SALT
    CHAT_TEMPLATE, SYSTEM_PROMPT, ENV_LOCK
    WEIGHTS_SUBJECT, ALIAS_SUBJECT, RECIPE          (module constants; never mutate them --
                                                     the builder functions hand out deep copies)
    weights_subject(**overrides), alias_subject(**overrides), recipe(**overrides)
    decoding(**overrides)

    BATTERY_ID, POOL_ID, FINGERPRINT_ID, PREV_FINGERPRINT_ID, RUN_IDS, PREREG_ID,
    NOISE_PLAN_ID, RESULT_ID, SENSITIVITY_ID, TARGET_ID, OWN_ID, PARENT_ID, SEALED_ID,
    CHALLENGE_ID, ITEMS_BLOB_ID

    battery_items(n=3), fingerprint_items(n=3), canary_items(n=3)
    minimal_body(cert_type), default_refs(cert_type), default_subject/-_recipe(cert_type)
    unsigned(cert_type, **overrides)
    make_cert(cert_type, **overrides)               -> a SIGNED cert that `cert.check` accepts

    fingerprint_body(...), battery_body(kind=...), canary_battery_body(...),
    canary_battery_refs(...),
    sealed_prereg_body(...), reveal_prereg_body(...), verify_result_body(...)

`make_cert` takes three keyword-only controls beside the envelope overrides:

    seed=<32 bytes>      sign with this seed instead of the issuer label's
    issuer_label="..."   which deterministic key pair to issue under
    signed=False         return the unsigned core (no "id", no "sig")

Every other keyword is written straight into the envelope, so `make_cert("fingerprint",
created="not-a-time")` and `make_cert("prereg", body={...})` both do what they look like, and
`make_cert("fingerprint", public=True)` can build the cert the envelope schema must refuse.
"""
from __future__ import annotations

import base64
import copy
import hashlib
from typing import Any, Optional

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as fpmod
from styxx.v8 import floor as floormod
from styxx.v8 import keys

__all__ = [
    "ALIAS_SUBJECT",
    "BATTERY_ID",
    "CHALLENGE_ID",
    "CHAT_TEMPLATE",
    "CREATED",
    "ENV_LOCK",
    "FINGERPRINT_ID",
    "ISSUER_NAME",
    "ITEMS_BLOB_ID",
    "NOISE_PLAN_ID",
    "OWN_ID",
    "PARENT_ID",
    "POOL_ID",
    "PREREG_ID",
    "PREV_FINGERPRINT_ID",
    "RECIPE",
    "RESULT_ID",
    "RUN_IDS",
    "SALT",
    "SEALED_ID",
    "SENSITIVITY_ID",
    "SYSTEM_PROMPT",
    "TARGET_ID",
    "WEIGHTS_SUBJECT",
    "alias_subject",
    "battery_body",
    "battery_items",
    "canary_battery_body",
    "canary_battery_refs",
    "canary_items",
    "decoding",
    "default_recipe",
    "default_refs",
    "default_subject",
    "fake_id",
    "fingerprint_body",
    "fingerprint_items",
    "issuer",
    "keypair",
    "make_cert",
    "minimal_body",
    "public_key",
    "recipe",
    "reveal_prereg_body",
    "salt_b64",
    "sealed_prereg_body",
    "sha256_text",
    "unsigned",
    "verify_result_body",
    "weights_subject",
]

ISSUER_NAME = "fathom lab"
CREATED = "2026-09-08T18:00:00Z"

# Types whose per-type schema requires a populated subject (contract section 1).
SUBJECT_TYPES = ("fingerprint", "battery")

# Types whose per-type schema requires a populated recipe. A battery is NOT one of them: a
# pool-v1 or fixed-v1 battery is the root of the dependency ladder and carries no recipe at all
# (spec section 4.5). Only a canary-v1 battery names a battery in its recipe, and the tests that
# build one pass `recipe=recipe(battery=...)` explicitly.
RECIPE_TYPES = ("fingerprint",)


# ----------------------------------------------------------------- deterministic material

def _seed_for(label: str) -> bytes:
    """A 32-byte Ed25519 seed derived from a label. Any 32 bytes is a valid seed (RFC 8032)."""
    return hashlib.sha256(("styxx.v8/test-key/" + label).encode("utf-8")).digest()


def keypair(label: str = "issuer") -> tuple[bytes, bytes]:
    """``(private_seed_32, public_32)`` for ``label`` -- the same pair on every run."""
    seed = _seed_for(label)
    return seed, keys.public_from_private(seed)


def public_key(label: str = "issuer") -> str:
    """``keys.encode_public`` of ``label``'s public key."""
    return keys.encode_public(keypair(label)[1])


def issuer(label: str = "issuer", name: str = ISSUER_NAME) -> dict:
    """An envelope ``issuer`` block for ``label``."""
    return {"name": name, "key": public_key(label)}


def fake_id(label: str) -> str:
    """A placeholder cert id: ``"sha256:" + sha256(label)``. Not the id of any real cert."""
    return "sha256:" + hashlib.sha256(("styxx.v8/test-id/" + label).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Appendix A.1: ``sha256(UTF-8 of the string)`` as 64 lowercase hex, no prefix."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


BATTERY_ID = fake_id("battery")
POOL_ID = fake_id("pool")
FINGERPRINT_ID = fake_id("fingerprint")
PREV_FINGERPRINT_ID = fake_id("fingerprint-previous")
RUN_IDS = [fake_id(f"run-{k}") for k in range(5)]
PREREG_ID = fake_id("prereg")
NOISE_PLAN_ID = fake_id("noise-plan")
RESULT_ID = fake_id("result")
SENSITIVITY_ID = fake_id("sensitivity-result")
TARGET_ID = fake_id("target-fingerprint")
OWN_ID = fake_id("own-fingerprint")
PARENT_ID = fake_id("parent-action")
SEALED_ID = fake_id("sealed-prereg")
CHALLENGE_ID = fake_id("challenge")
ITEMS_BLOB_ID = fake_id("items-blob")

# 32 bytes from a fixed generator, so the sealed-prereg commitment is reproducible.
SALT = bytes(range(32))


def salt_b64() -> str:
    """``SALT`` as the 43-character unpadded base64url the prereg schema requires."""
    return base64.urlsafe_b64encode(SALT).decode("ascii").rstrip("=")


# ----------------------------------------------------------------- subjects (section 2.2)

CHAT_TEMPLATE = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
SYSTEM_PROMPT = ""
ENV_LOCK = "styxx==8.0.0\ntransformers==4.57.3\ntorch==2.5.1\n"

WEIGHTS_SUBJECT: dict[str, Any] = {
    "kind": "weights",
    "model_family": "qwen2.5",
    "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "weights_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "tokenizer_sha256": "c" * 64,
    "generation_config_sha256": "d" * 64,
    "precision": "bf16",
    "environment": {
        "runtime": {"framework": "transformers", "version": "4.57.3", "backend": "torch 2.5.1+cpu"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
    },
}

ALIAS_SUBJECT: dict[str, Any] = {
    "kind": "alias",
    "model_family": "unknown",
    "provider": "acme",
    "alias": "acme-large",
    "region": "eu",
    # An alias run still happens somewhere: the client's runtime, and the box it dialled out from,
    # which computes no forward pass and therefore holds no card of its own. `schema/subject.json`
    # leaves `environment` optional on the alias branch and `schema/fingerprint.json` now requires
    # it on any fingerprint's subject (ENV-ABSENT) -- a fingerprint IS the record of a run and
    # section 2.2 has the environment of a run recorded because it was observed, so an alias
    # fingerprint carrying none observed none, and its floor's `not_covered` was empty, which
    # section 5.4 reads as covering every environment there is.
    "environment": {
        "runtime": {"framework": "http", "version": "1.1", "backend": "acme-api"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
    },
}


def weights_subject(**overrides: Any) -> dict:
    """A minimal valid ``kind: weights`` subject; ``overrides`` replace top-level fields."""
    out = copy.deepcopy(WEIGHTS_SUBJECT)
    out.update(copy.deepcopy(overrides))
    return out


def alias_subject(**overrides: Any) -> dict:
    """A minimal valid ``kind: alias`` subject; ``overrides`` replace top-level fields."""
    out = copy.deepcopy(ALIAS_SUBJECT)
    out.update(copy.deepcopy(overrides))
    return out


# ----------------------------------------------------------------- recipe (section 2.3)

DECODING: dict[str, Any] = {
    "temperature": 0,
    "top_p": 1.0,
    "max_new_tokens": 64,
    "stop": ["\n"],
    "seed": 7,
    "batch_size": 1,
    "padding_side": "left",
}

RECIPE: dict[str, Any] = {
    "battery": BATTERY_ID,
    "decoding": copy.deepcopy(DECODING),
    "materials": {
        "chat_template": CHAT_TEMPLATE,
        "chat_template_source": "tokenizer_config.json@7ae5576",
        "system_prompt": SYSTEM_PROMPT,
        "env_lock": ENV_LOCK,
    },
    # Derived from materials (Appendix A.1); `cert.check` recomputes all three.
    "chat_template_sha256": sha256_text(CHAT_TEMPLATE),
    "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
    "env_lock_sha256": sha256_text(ENV_LOCK),
    "harness": {"name": "styxx", "version": "8.0.0", "commit": "0" * 40},
}


def decoding(**overrides: Any) -> dict:
    """A minimal valid ``recipe.decoding`` block."""
    out = copy.deepcopy(DECODING)
    out.update(copy.deepcopy(overrides))
    return out


def recipe(**overrides: Any) -> dict:
    """A minimal valid recipe whose three ``*_sha256`` fields match its materials.

    ``overrides`` replace top-level recipe keys. Passing ``materials=`` alone does NOT
    recompute the hashes -- that is deliberate, so a test can build the mismatch on purpose.
    """
    out = copy.deepcopy(RECIPE)
    out.update(copy.deepcopy(overrides))
    return out


# ----------------------------------------------------------------- item helpers

def battery_items(n: int = 3, *, family: str = "recall", role: str = "item") -> list[dict]:
    """``n`` minimal battery items (schema/battery.json ``$defs/item``)."""
    out = []
    for k in range(n):
        prompt = f"prompt {k}"
        out.append({
            "item_id": f"i{k:02d}",
            "prompt_sha256": sha256_text(prompt),
            "prompt_text": prompt,
            "family": family,
            "role": role,
        })
    return out


def canary_items(n: int = 3) -> list[dict]:
    """``n`` canary-v1 items: the score block schema/battery.json requires under canary-v1."""
    out = battery_items(n, role="canary")
    for k, item in enumerate(out):
        item["role"] = "anchor" if k == 0 else "canary"
        item["score"] = round(0.1 * (k + 1), 9)
        item["margin"] = round(2.0 - 0.25 * k, 9)
        item["flip1"] = 0.0
        item["flip2"] = 0.0
        item["flip3"] = round(0.05 * k, 9)
    return out


def fingerprint_items(n: int = 3, *, redacted: bool = False, logprobs: bool = True) -> list[dict]:
    """``n`` minimal fingerprint items (schema/fingerprint.json ``$defs/item``, section 3.1)."""
    out = []
    for k in range(n):
        token_ids = [100 + k, 200 + k, 300 + k]
        text = f"answer {k}"
        item: dict[str, Any] = {
            "item_id": f"i{k:02d}",
            "token_ids": token_ids,
            "token_ids_sha256": sha256_text(str(token_ids)),
            "output_sha256": sha256_text(text),
            "n_generated": len(token_ids),
        }
        if not redacted:
            item["output_text"] = text
        if logprobs:
            item["seq_logprob"] = round(-1.5 - 0.25 * k, 9)
            item["topk"] = [{"pos": 0, "ids": [100 + k, 101 + k], "lps": [-0.1, -2.3]}]
        else:
            item["seq_logprob"] = None
            item["topk"] = None
        out.append(item)
    return out


def _item_order_sha256(items: list[dict]) -> str:
    return sha256_text("\n".join(i["item_id"] for i in items))


# ----------------------------------------------------------------- bodies

def fingerprint_body(
    *,
    n: int = 3,
    run_index: int = 0,
    batch_size: int = 1,
    redacted: bool = False,
    logprobs: bool = True,
    tier: str = "white-box",
    items_blob: Optional[str] = None,
    noise_floor: Optional[dict] = None,
    sensitivity: Optional[str] = None,
) -> dict:
    """A fingerprint body (section 3.1). Minimal by default; the flags add the optional blocks."""
    items = fingerprint_items(n, redacted=redacted, logprobs=logprobs)
    body: dict[str, Any] = {
        "run_index": run_index,
        "nuisance": {"batch_size": batch_size, "item_order_sha256": _item_order_sha256(items)},
        "items": items,
        "channels": {
            "exact": {"hash": sha256_text("exact")},
            "seqlp": {"present": logprobs},
            "topk": {"present": logprobs},
        },
        "redacted": redacted,
        "tier": tier,
    }
    if items_blob is not None:
        body["items_blob"] = items_blob
    if noise_floor is not None:
        body["noise_floor"] = noise_floor
    if sensitivity is not None:
        body["sensitivity"] = sensitivity
    return body


def noise_floor_block(
    *,
    plan: str = NOISE_PLAN_ID,
    runs: Optional[list[str]] = None,
    bodies: Optional[list[dict]] = None,
    plan_body: Optional[dict] = None,
) -> dict:
    """A ``body.noise_floor`` block (section 5); every id in it needs a ref.

    ``bodies`` are the run bodies the floor is over, the canonical at index 0 — the order
    ``fingerprint.attach_floor`` uses and ``Log.append`` re-derives in. Pass them whenever the
    cert will be appended to a log: since the log recomputes ``per_channel`` from the certs
    ``runs`` names and refuses a disagreement, a block whose numbers were made up is a block no
    log takes. Without ``bodies`` the numbers are a placeholder for ``cert.check``-only tests.

    ``plan_body`` is the noise plan's own body, and it plays the same role for ``covers`` and
    ``not_covered`` that ``bodies`` plays for ``per_channel``: the log derives both lists from the
    plan's ``nuisance`` and ``environment`` and refuses a disagreement (A-COVER,
    ``Log._check_floor_covers_match_the_plan``), so a block a log will take is a block whose two
    coverage lists are the plan's. Without it the lists are a placeholder, like the numbers.
    """
    block = {
        "plan": plan,
        "runs": list(RUN_IDS) if runs is None else list(runs),
        "covers": ["runtime.version"],
        "not_covered": ["hardware.gpu"],
        "per_channel": {"exact": {"floor": 0.0, "runs": 5, "pairs": 10, "alpha_single": 1 / 11}},
    }
    if plan_body is not None:
        block["covers"], block["not_covered"] = floormod.plan_coverage(plan_body)
    if bodies is not None:
        roles = {
            item["item_id"]: item.get("role", "item") for item in bodies[0].get("items", [])
        }
        per = floormod.floors(list(bodies), roles=roles or None)
        block["per_channel"] = {c: b for c, b in per.items() if b is not None}
        # Section 5.7's overall size, the same four keys `fingerprint.attach_floor` writes. The
        # anchor requires them (A-OPTIONAL: they used to be re-derived only when offered, so a
        # floor with them deleted appended with an empty disagreement list), which means a block
        # a log will take is a block that carries them.
        block.update(fpmod._overall_size(list(bodies), block["per_channel"], roles))
    return block


def battery_body(kind: str = "pool-v1", *, n: int = 3, source: str = "styxx-bench@v8") -> dict:
    """A minimal ``pool-v1`` or ``fixed-v1`` battery body (section 4.5).

    Neither carries the selection fields; schema/battery.json forbids them outside canary-v1.
    Both are ROOTS: paired with the default empty recipe and empty refs they build a battery an
    empty log accepts at index 0.
    """
    body: dict[str, Any] = {
        "kind": kind,
        "items": battery_items(n),
        "families": ["recall"],
        "redacted": False,
    }
    if kind == "fixed-v1":
        body["source"] = source
    return body


def canary_battery_body(*, n: int = 3) -> dict:
    """A minimal ``canary-v1`` battery body: params and excluded are required under that kind."""
    return {
        "kind": "canary-v1",
        "items": canary_items(n),
        "families": ["recall"],
        "redacted": False,
        "pool_sha256": sha256_text("pool"),
        "pool_size": 84,
        "params": {
            "n": n,
            "k": 1,
            "tau": 1.0,
            "max_family_share": 0.25,
            "perm_seed": 7,
            "k_anchors_actual": 1,
            "sensitivity_after_exclusion": 0.0,
        },
        "excluded": [{"item_id": "i99", "flip2": 1.0}],
    }


def canary_battery_refs(pool_id: str = POOL_ID, reference_id: str = FINGERPRINT_ID) -> list[dict]:
    """The refs a canary-v1 battery needs (section 4.5): the pool it was drawn from, twice by
    role, and the reference fingerprint it was selected against.

    ``recipe.battery`` is the pool, so the ``battery`` role carries ``pool_id`` as well -- an
    embedded id has to appear in refs (section 2.1). Pair this with
    ``recipe=recipe(battery=pool_id)``.
    """
    return [
        {"role": "battery", "id": pool_id},
        {"role": "pool", "id": pool_id},
        {"role": "selected_against", "id": reference_id},
    ]


def sealed_prereg_body(revealed: Optional[dict] = None) -> dict:
    """A sealed prereg body (section 7.2): ``sealed`` true, the commitment over ``revealed``."""
    return {
        "kind": "study",
        "sealed": True,
        "commitment": certmod.seal_commitment(SALT, revealed if revealed is not None else reveal_prereg_body()),
    }


def reveal_prereg_body() -> dict:
    """The COMPLETE prereg body a sealed prereg commits to; the reveal carries it plus the salt."""
    return {
        "kind": "study",
        "sealed": False,
        "hypotheses": [{"id": "H1", "direction": "greater", "endpoint": "auroc"}],
        "grader": {"kind": "gold-labels", "id": "styxx-bench@v8"},
    }


def verify_result_body(*, overall: str = "same", ref: Optional[str] = None) -> dict:
    """A ``kind: verify`` result body (section 6.1)."""
    return {
        "kind": "verify",
        "ref": ref,
        "per_channel": {"exact": {"distance": 0.0, "floor": 0.0, "ratio": None, "verdict": "same"}},
        "overall": overall,
        "floor_owner": "A",
        "coverage": "within",
        "skipped_channels": [],
        "rounding": 9,
    }


_MINIMAL_BODIES: dict[str, Any] = {
    "fingerprint": fingerprint_body,
    "battery": battery_body,
    # A noise-plan names R (schema/prereg.json requires `runs` on this kind since A-NORUNS; the
    # default is the R the floor fixtures below rest on, so the minimal prereg is a plan a log
    # will take rather than a document shaped like one).
    "prereg": lambda: {"kind": "noise-plan", "runs": 5},
    "result": lambda: {"kind": "pilot", "deviations": []},
    "promotion": lambda: {
        "instrument": "depth",
        "version": "1.2.0",
        "code_sha256": sha256_text("depth.py"),
        "scope": {"model_family": "qwen2.5", "task_family": "recall"},
        "claim": "a scope-limited sentence with no numeral in it",
    },
    "action": lambda: {
        "seq": 0,
        "context_sha256": sha256_text("context"),
        "readings": {"certified": {}, "lab": {"depth": 7.5}},
        "action_sha256": sha256_text("action"),
        "action_kind": "message",
        "ts": CREATED,
    },
    # `subject` and `recipe_core` are the challenger's own report of what produced the
    # distances (spec section 9 body, C3 of papers/v8/challenge_and_attack_2026_09_09): a
    # challenge that says nothing about what was run is not one.
    "challenge": lambda: {
        "per_channel": {"exact": {"distance": 0.125, "target_floor": 0.0}},
        "coverage": "within",
        "environment": {"runtime": {"framework": "transformers", "version": "4.57.3", "backend": "torch 2.5.1+cpu"}},
        "subject": {
            k: v for k, v in weights_subject().items()
            if k in ("kind", "hf_repo", "revision", "weights_sha256", "config_sha256",
                     "tokenizer_sha256", "generation_config_sha256", "precision")
        },
        "recipe_core": {
            k: copy.deepcopy(recipe().get(k))
            for k in ("battery", "decoding", "chat_template_sha256", "system_prompt_sha256")
        },
    },
    "sublog": lambda: {
        "sublog_id": sha256_text("sublog"),
        "tree_size": 2,
        "root_hash": sha256_text("root"),
        "prev_tree_size": 1,
        "prev_root_hash": sha256_text("prev-root"),
        "consistency_proof": [sha256_text("proof-0")],
        "count_since_prev": 1,
        "entries_sha256": sha256_text("entries"),
    },
}


def minimal_body(cert_type: str) -> dict:
    """The minimal body its per-type schema accepts, for any of the eight types."""
    if cert_type not in _MINIMAL_BODIES:
        raise KeyError(f"no minimal body for cert type {cert_type!r}")
    return _MINIMAL_BODIES[cert_type]()


_DEFAULT_REFS: dict[str, Any] = {
    # recipe.battery is an embedded id, so every cert carrying a recipe needs the matching ref.
    "fingerprint": lambda: [{"role": "battery", "id": BATTERY_ID}],
    # A root battery (pool-v1 / fixed-v1) has no recipe and therefore no ref to anything: it is
    # an entry an empty log accepts at index 0 (spec section 4.5).
    "battery": lambda: [],
    "prereg": lambda: [],
    "result": lambda: [],
    "promotion": lambda: [],
    "action": lambda: [],
    # schema/challenge.json requires both roles.
    "challenge": lambda: [{"role": "target", "id": TARGET_ID}, {"role": "own", "id": OWN_ID}],
    "sublog": lambda: [],
}


def default_refs(cert_type: str) -> list[dict]:
    """The refs a minimal cert of ``cert_type`` needs to satisfy `cert.check`."""
    if cert_type not in _DEFAULT_REFS:
        raise KeyError(f"no default refs for cert type {cert_type!r}")
    return _DEFAULT_REFS[cert_type]()


def default_subject(cert_type: str) -> dict:
    """A weights subject for the two types that require one; ``{}`` for the rest (section 2)."""
    return weights_subject() if cert_type in SUBJECT_TYPES else {}


def default_recipe(cert_type: str) -> dict:
    """A full recipe for the one type that requires one; ``{}`` for the rest (sections 2, 4.5).

    A default battery cert is a ROOT battery: empty recipe, no refs. A canary-v1 battery is not
    a root and its test passes ``recipe=recipe(battery=<pool id>)`` with the matching refs.
    """
    return recipe() if cert_type in RECIPE_TYPES else {}


# ----------------------------------------------------------------- the cert builders

def unsigned(cert_type: str, **overrides: Any) -> dict:
    """The cert core -- everything but ``id`` and ``sig``. ``overrides`` replace envelope keys."""
    core: dict[str, Any] = {
        "styxx": "8.0",
        "type": cert_type,
        "created": CREATED,
        "issuer": issuer(overrides.pop("issuer_label", "issuer")),
        "subject": default_subject(cert_type),
        "recipe": default_recipe(cert_type),
        "body": minimal_body(cert_type),
        "refs": default_refs(cert_type),
    }
    core.update(copy.deepcopy(overrides))
    return core


def make_cert(
    cert_type: str,
    *,
    seed: Optional[bytes] = None,
    issuer_label: str = "issuer",
    signed: bool = True,
    **overrides: Any,
) -> dict:
    """A signed cert of ``cert_type`` that `cert.check` accepts when nothing is overridden.

    ``seed`` signs with a different key without touching ``issuer`` -- use it to build the
    wrong-key case. ``signed=False`` returns the unsigned core.
    """
    core = unsigned(cert_type, issuer_label=issuer_label, **overrides)
    if not signed:
        return core
    return certmod.sign(core, _seed_for(issuer_label) if seed is None else seed)
