"""styxx.v8.fingerprint -- fingerprint bodies, the Appendix A.3 exact hash, floors attached.

Contract: ``styxx/v8/INTERFACES_layer2.md`` section 6.  Spec: v0.2 draft section 3 (body and
channels), section 5 (the floor block), Appendix A.3 (order and the exact hash).

What this module is.  A *fingerprint body* is the per-run record of section 3.1: one item
record per battery item, the channel blocks that say which channels this run actually carries,
the nuisance settings the run was made under, and -- on the canonical run (``run_index`` 0) --
the noise-floor block computed from the R runs of section 5.1.  Nothing here talks to a model:
``run_fingerprint`` drives a ``Runner`` (``styxx.v8.runner``), everything else is a pure
function of dicts.

Appendix A.3, implemented literally::

    token_ids_sha256      = sha256(UTF-8(JCS(ids)))                 # per item, 64 lowercase hex
    channels.exact.hash   = sha256(concat of the raw 32-byte token_ids_sha256 digests,
                                   items in item_id ascending order (UTF-8 byte order))

so the battery hash does not depend on the order the battery was run in, nor on the decoder's
text rendering.

Decisions this module makes where the contract or the spec leaves room (each is pinned by a
test in ``tests/test_v8_fingerprint.py``):

* **Token ids stay in the clear under redaction.**  ``redacted=True`` omits ``output_text`` and
  nothing else: ``token_ids`` are the preimage of ``token_ids_sha256`` and of the exact channel,
  so a redacted fingerprint that dropped them could not be checked by a stranger at all.  What
  redaction buys is that the model's text is not republished; the ids remain.
* **Body items are stored in the RUN order**, not in A.3 order.  ``nuisance.item_order_sha256``
  is the receipt for that order, and the white-box per-item lists (``resid.profile``,
  ``lens.converge_layer``) are positional in the same order (``styxx.v8.floor`` re-keys them by
  the body's own item order).  ``exact_hash`` sorts internally, so the hash is unaffected.
* **``item_order_sha256`` = ``sha256(UTF-8(JCS([item_id, ...])))``** over the ids in the run
  order.  Neither the spec nor the contract pins the preimage; this one hashes bytes this
  module constructed, uses the same JCS the rest of v8 uses, and cannot collide across
  different orders.  Use ``order_sha256`` rather than re-deriving it.
* **A channel is present only when every item carries it.**  One item with ``seq_logprob is
  None`` makes ``seqlp`` absent for the whole run (section 3.2: absent, never zero-filled).
* **``tier`` follows ``subject.kind``** (contract section 6): ``weights`` -> ``white-box``,
  ``alias`` -> ``black-box``, whether or not this particular run carried the white-box
  channels.  ``white_box=`` on an alias subject raises rather than producing a black-box body
  with a ``resid`` block.
* **Every item record carries a ``role``**, copied from the battery item (default ``"item"``).
  Section 3.1 does not show the field; Appendix B needs it (``exact`` scores ``item``/``canary``
  and counts ``anchor`` flips separately) and ``styxx.v8.floor`` reads it off the run bodies, so
  a fingerprint that dropped it would force every consumer back to the battery cert.
* **The results must cover the battery exactly** -- same set of ``item_id`` values, no
  duplicates.  A fingerprint over a subset is a fingerprint of a different battery.
* **A nuisance block that contradicts the run is refused.**  ``batch_size`` must equal
  ``recipe.decoding.batch_size`` and ``item_order_sha256`` must equal the order actually run;
  both are filled in when absent.
* **``build`` refuses a ``recipe.battery`` that is not the battery cert's ``id``** when both are
  present: the recipe names the battery, and a body built against another one is not
  recipe-complete.
* **A redacted battery is flagged, not refused.**  ``body.redacted_battery`` is set to ``True``
  when the battery body says ``redacted``; section 4.5 makes such a fingerprint non-verifiable
  and ``verify --ref`` exits 3 on it.  ``run_fingerprint`` still refuses it earlier, because it
  cannot run prompts it does not have.
* **``attach_floor`` returns a new body** (deep copy) and refuses a body whose ``run_index`` is
  not 0 (section 5.1: the canonical fingerprint carries the floor), a ``run_ids`` list whose
  length is neither ``len(run_bodies)`` nor ``len(run_bodies) - 1``, a duplicate run id, a
  ``covers``/``not_covered`` overlap, and a run set on which no channel is present everywhere.
* **``run_ids`` may be one short, and that is the section 5.1 step 5 shape.**  The append order
  there logs the R-1 non-canonical runs and then the canonical fingerprint, so at the moment the
  canonical body is built the canonical run has no cert of its own to name -- it *is* the cert
  being built, and a cert cannot reference itself (section 2.1).  A caller in that shape passes
  R-1 ids for R bodies and they are taken to be ``run_bodies[1:]``; a reader recomputes the floor
  from the canonical body's own items plus the R-1 referenced runs.  Passing R ids for R bodies
  (every run logged separately, the canonical a further entry) is still accepted.
* **``noise_floor.alpha_overall`` is measured, never asserted (section 5.7).**  For each of the R
  runs, its standardized maximum channel distance to the reference run is ``max`` over the
  evaluated channels of ``d_c / floor_c``; a channel whose ``floor_c`` is 0 contributes no ratio
  and counts as an exceedance whenever ``d_c > 0``, which is the rule section 5.7 states and the
  reason a zero floor never reaches a division.  ``alpha_overall`` is the fraction of the R runs
  that exceed, ``standardization`` and ``alpha_overall_method`` record the rule and the
  procedure, and ``standardized_max`` lists the per-run numbers so a reader re-derives the
  fraction from the same run certs the floor came from.
* **``run_fingerprint`` refuses a runner that runs another subject.**  Before any prompt is sent
  it asks the runner what it will actually run (``runner.reported_identity``) and compares the
  S_identity fields against the ``subject`` the body will record; a difference raises and a
  runner that does not report raises ``SubjectUnavailable``.  A fingerprint body carries
  ``subject``, so a body a runner did not produce under that identity is a false record at the
  moment of minting -- the same defect ``verify --ref`` had (C2 of
  ``papers/v8/challenge_and_attack_2026_09_09``), caught where the claim is made.  Advisory
  fields (``model_family``, ``environment``) are not compared: they are not identity.
* ``exact_hash`` on an empty item list raises: a run with no items has no battery hash.
* **The plan is applied, not described.**  ``plan_run_settings`` turns a noise plan plus a recipe
  into the R runs it commits to: the assignment each run takes, the recipe that run actually runs
  under (the plan's ``batch_size``/``padding_side`` written into ``decoding``), its item order,
  and the ``nuisance`` record naming the assignment.  The enumeration rule is stated on
  ``assignment_sequence``; a factor no runner can set is a refusal that names the factor, never a
  run at the default.  The first real floor declared ``batch_size`` and ``item_order``, ran all
  five runs at batch 1, and got a floor of 0.0 on every channel against which every later
  difference exceeded (`papers/v8/vacuous_floor_2026_09_09/`): the plan was a description the runs
  ignored, and this is the half of the repair that lives in the runner.

Pure CPU; importing this module does not import torch.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from styxx.v8 import floor as floormod
from styxx.v8 import runner as runnermod
from styxx.v8.consts import ID_RE, ROUND_PLACES
from styxx.v8.jcs import canonical_bytes, sha256_hex

__all__ = [
    "A3_ORDER",
    "A3_ORDER_NAMES",
    "APPLICABLE_FACTORS",
    "ORDER_FACTORS",
    "assignment_sequence",
    "attach_floor",
    "battery_body",
    "battery_item_map",
    "build",
    "exact_hash",
    "execution_state",
    "execution_states_for",
    "item_record",
    "order_for_value",
    "order_sha256",
    "plan_factors",
    "plan_run_settings",
    "run_fingerprint",
]

A3_ORDER = "item_id ascending (UTF-8 byte order)"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID_PATTERN = re.compile(ID_RE)
_TIER_BY_KIND = {"weights": "white-box", "alias": "black-box"}
_WHITE_BOX_CHANNELS = ("resid", "lens")


# --------------------------------------------------------------------------- small guards

def _mapping(what: str, value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a dict, got {type(value).__name__}")
    return dict(value)


def _sequence(what: str, value: Any) -> list:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a list, got {type(value).__name__}")
    return list(value)


def _str(what: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _bool(what: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{what} must be a bool, got {type(value).__name__}")
    return value


def _int(what: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}")
    return value


def _number(what: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number, got {type(value).__name__}")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return f


def _cert_id(what: str, value: Any) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ValueError(f"{what} must be 'sha256:<64 lowercase hex>', got {value!r}")
    return value


def _hex_digest(what: str, value: Any) -> bytes:
    """A 64-hex digest, with or without the ``sha256:`` prefix, as raw bytes (Appendix A.1)."""
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, got {type(value).__name__}")
    text = value[7:] if value.startswith("sha256:") else value
    if not _HEX64.match(text):
        raise ValueError(f"{what} must be 64 lowercase hex characters, got {value!r}")
    return bytes.fromhex(text)


def _id_key(item_id: str) -> bytes:
    """A.3 order key: the UTF-8 bytes of the item id."""
    return item_id.encode("utf-8")


def _token_ids(what: str, value: Any) -> list[int]:
    ids = _sequence(what, value)
    out: list[int] = []
    for k, t in enumerate(ids):
        out.append(_int(f"{what}[{k}]", t, minimum=0))
    return out


def order_sha256(item_ids: Sequence[str]) -> str:
    """``sha256(UTF-8(JCS([item_id, ...])))`` over the ids in the order given, 64 hex.

    The preimage is bytes this module constructs, never text read from a file.  A duplicate id
    is refused: an order that names an item twice is not an order.
    """
    ids = [_str(f"item_ids[{k}]", v) for k, v in enumerate(_sequence("item_ids", item_ids))]
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids repeats an id; an order names every item once")
    return sha256_hex(canonical_bytes(ids))


# --------------------------------------------------------------------------- item records

def _topk_record(value: Any, iid: str) -> Optional[list[dict]]:
    """Validate and copy an ItemResult ``topk`` list (section 3.2), or pass ``None`` through."""
    if value is None:
        return None
    entries = _sequence(f"item {iid!r}: topk", value)
    out: list[dict] = []
    seen: set[int] = set()
    for k, entry in enumerate(entries):
        where = f"item {iid!r}: topk[{k}]"
        block = _mapping(where, entry)
        pos = _int(f"{where}.pos", block.get("pos"), minimum=0)
        if pos in seen:
            raise ValueError(f"{where}.pos repeats position {pos}")
        seen.add(pos)
        ids = _token_ids(f"{where}.ids", block.get("ids"))
        lps_raw = _sequence(f"{where}.lps", block.get("lps"))
        lps = [_number(f"{where}.lps[{j}]", v) for j, v in enumerate(lps_raw)]
        if len(ids) != len(lps):
            raise ValueError(f"{where}: {len(ids)} ids against {len(lps)} log-probs")
        if not ids:
            raise ValueError(f"{where}: an empty top-k vector is not a reading")
        out.append({"pos": pos, "ids": ids, "lps": lps})
    return out


def item_record(r: Mapping[str, Any], redacted: bool) -> dict:
    """One ``body.items`` record (section 3.1) from an ``ItemResult`` (section 3).

    ``token_ids_sha256 = sha256(UTF-8(JCS(token_ids)))`` and
    ``output_sha256 = sha256(UTF-8(output_text))``, both 64 lowercase hex (Appendix A.1/A.3).
    ``output_text`` is omitted when ``redacted`` is true; ``token_ids`` are kept either way --
    they are the preimage of the checksum and of the exact channel, and a redacted fingerprint
    without them could not be checked at all.  ``n_generated``, ``seq_logprob`` and ``topk`` are
    carried through as they are (``None`` for a provider that returns no log-probs), and
    ``prefix_token_ids`` when the runner supplied it (section 3.2, weights subjects).
    """
    redacted = _bool("redacted", redacted)
    rec = _mapping("ItemResult", r)
    iid = _str("ItemResult.item_id", rec.get("item_id"))
    ids = _token_ids(f"item {iid!r}: token_ids", rec.get("token_ids"))
    text = rec.get("output_text")
    if not isinstance(text, str):
        raise TypeError(f"item {iid!r}: output_text must be a string, got {type(text).__name__}")
    n_generated = _int(f"item {iid!r}: n_generated", rec.get("n_generated"), minimum=0)
    if n_generated != len(ids):
        raise ValueError(
            f"item {iid!r}: n_generated {n_generated} does not match {len(ids)} token ids"
        )
    seq_lp = rec.get("seq_logprob")
    if seq_lp is not None:
        seq_lp = _number(f"item {iid!r}: seq_logprob", seq_lp)
    out: dict[str, Any] = {
        "item_id": iid,
        "token_ids": ids,
        "token_ids_sha256": sha256_hex(canonical_bytes(ids)),
        "output_sha256": sha256_hex(text.encode("utf-8")),
        "n_generated": n_generated,
        "seq_logprob": seq_lp,
        "topk": _topk_record(rec.get("topk"), iid),
    }
    if not redacted:
        out["output_text"] = text
    prefix = rec.get("prefix_token_ids")
    if prefix is not None:
        out["prefix_token_ids"] = _token_ids(f"item {iid!r}: prefix_token_ids", prefix)
    return out


def _record_digest(rec: Mapping[str, Any], iid: str) -> bytes:
    """The raw 32-byte A.3 digest of one item record.

    From ``token_ids`` when present (the spec defines the digest, so computing it is licensed),
    else from ``token_ids_sha256``.  A record carrying both must agree with itself.
    """
    ids = rec.get("token_ids")
    stored = rec.get("token_ids_sha256")
    computed = None
    if ids is not None:
        computed = canonical_bytes(_token_ids(f"item {iid!r}: token_ids", ids))
        computed = hashlib.sha256(computed).digest()
    if stored is None:
        if computed is None:
            raise ValueError(f"item {iid!r}: neither token_ids nor token_ids_sha256")
        return computed
    given = _hex_digest(f"item {iid!r}: token_ids_sha256", stored)
    if computed is not None and given != computed:
        raise ValueError(
            f"item {iid!r}: token_ids_sha256 {given.hex()} is not the digest of its own "
            f"token_ids ({computed.hex()})"
        )
    return given


def exact_hash(items: Sequence[Mapping[str, Any]]) -> str:
    """Appendix A.3: sha256 over the concatenated raw 32-byte digests, items in A.3 order.

    ``items`` are ``body.items`` records; the order they are given in does not matter (they are
    sorted by ``item_id``), which is what makes the battery hash independent of the run order.
    Returns 64 lowercase hex, no prefix (the schema's ``channels.exact.hash``).
    """
    records = _sequence("items", items)
    if not records:
        raise ValueError("exact_hash needs at least one item; a run with no items has no hash")
    digests: list[tuple[bytes, bytes]] = []
    seen: set[str] = set()
    for k, rec in enumerate(records):
        block = _mapping(f"items[{k}]", rec)
        iid = _str(f"items[{k}].item_id", block.get("item_id"))
        if iid in seen:
            raise ValueError(f"items repeats item_id {iid!r}")
        seen.add(iid)
        digests.append((_id_key(iid), _record_digest(block, iid)))
    digests.sort(key=lambda pair: pair[0])
    return sha256_hex(b"".join(d for _, d in digests))


# --------------------------------------------------------------------------- battery access

def battery_body(battery_cert: Mapping[str, Any]) -> dict:
    """The battery *body* from a battery cert or from a bare body (section 4.5)."""
    cert = _mapping("battery_cert", battery_cert)
    if isinstance(cert.get("body"), Mapping):
        return dict(cert["body"])
    if isinstance(cert.get("items"), Sequence):
        return cert
    raise ValueError("battery_cert carries neither a body nor an items list")


def battery_item_map(battery_cert: Mapping[str, Any]) -> dict[str, dict]:
    """``{item_id: item}`` from a battery cert or body; duplicates are refused."""
    body = battery_body(battery_cert)
    items = _sequence("battery body items", body.get("items"))
    if not items:
        raise ValueError("battery body has no items")
    out: dict[str, dict] = {}
    for k, item in enumerate(items):
        block = _mapping(f"battery item {k}", item)
        iid = _str(f"battery item {k}: item_id", block.get("item_id"))
        if iid in out:
            raise ValueError(f"battery repeats item_id {iid!r}")
        out[iid] = block
    return out


def _battery_id(battery_cert: Mapping[str, Any]) -> Optional[str]:
    cid = battery_cert.get("id") if isinstance(battery_cert, Mapping) else None
    return cid if isinstance(cid, str) else None


# --------------------------------------------------------------------------- white box

def _white_box_channels(white_box: Mapping[str, Any], n_items: int) -> dict[str, dict]:
    block = _mapping("white_box", white_box)
    unknown = [k for k in block if k not in _WHITE_BOX_CHANNELS]
    if unknown:
        raise ValueError(f"white_box may only carry {list(_WHITE_BOX_CHANNELS)}, got {unknown}")
    out: dict[str, dict] = {}
    if "resid" in block:
        resid = _mapping("white_box.resid", block["resid"])
        n_layers = _int("white_box.resid.n_layers", resid.get("n_layers"), minimum=1)
        profiles_raw = _sequence("white_box.resid.profile", resid.get("profile"))
        if len(profiles_raw) != n_items:
            raise ValueError(
                f"white_box.resid.profile has {len(profiles_raw)} entries for {n_items} items"
            )
        profiles: list[list[float]] = []
        for k, vec in enumerate(profiles_raw):
            values = _sequence(f"white_box.resid.profile[{k}]", vec)
            if len(values) != n_layers:
                raise ValueError(
                    f"white_box.resid.profile[{k}] has {len(values)} layers, n_layers is {n_layers}"
                )
            profiles.append([_number(f"white_box.resid.profile[{k}][{j}]", v) for j, v in enumerate(values)])
        mean = resid.get("mean")
        if mean is None:
            mean = [
                round(math.fsum(p[j] for p in profiles) / len(profiles), ROUND_PLACES)
                for j in range(n_layers)
            ]
        else:
            values = _sequence("white_box.resid.mean", mean)
            if len(values) != n_layers:
                raise ValueError("white_box.resid.mean must have one entry per layer")
            mean = [_number(f"white_box.resid.mean[{j}]", v) for j, v in enumerate(values)]
        out["resid"] = {"present": True, "n_layers": n_layers, "profile": profiles, "mean": mean}
    if "lens" in block:
        lens = _mapping("white_box.lens", block["lens"])
        n_layers = _int("white_box.lens.n_layers", lens.get("n_layers"), minimum=1)
        layers_raw = _sequence("white_box.lens.converge_layer", lens.get("converge_layer"))
        if len(layers_raw) != n_items:
            raise ValueError(
                f"white_box.lens.converge_layer has {len(layers_raw)} entries for {n_items} items"
            )
        layers: list[int] = []
        for k, v in enumerate(layers_raw):
            layer = _int(f"white_box.lens.converge_layer[{k}]", v, minimum=0)
            if layer > n_layers:
                raise ValueError(
                    f"white_box.lens.converge_layer[{k}] is {layer}, above n_layers {n_layers}"
                )
            layers.append(layer)
        mean = lens.get("mean")
        if mean is None:
            mean = round(math.fsum(layers) / len(layers), ROUND_PLACES)
        else:
            mean = _number("white_box.lens.mean", mean)
        out["lens"] = {"present": True, "n_layers": n_layers, "converge_layer": layers, "mean": mean}
    if not out:
        raise ValueError("white_box carries no channel")
    return out


# --------------------------------------------------------------------------- the body

def build(
    subject: Mapping[str, Any],
    recipe: Mapping[str, Any],
    battery_cert: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    run_index: int,
    nuisance: Mapping[str, Any],
    redacted: bool = False,
    white_box: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The fingerprint BODY of section 3.1 for one run.

    ``results`` are the runner's ``ItemResult`` records **in the order they were run**; the body
    stores them in that order and ``nuisance.item_order_sha256`` is its receipt.  Channels carry
    present/absent flags (section 3.2: a channel absent on any item is absent for the run, never
    zero-filled), ``tier`` follows ``subject.kind``, and every item record carries the battery's
    ``role`` so Appendix B can score ``item``/``canary`` and count anchor flips separately.
    """
    subj = _mapping("subject", subject)
    rec = _mapping("recipe", recipe)
    kind = subj.get("kind")
    if kind not in _TIER_BY_KIND:
        raise ValueError(f"subject.kind must be 'weights' or 'alias', got {kind!r}")
    redacted = _bool("redacted", redacted)
    run_index = _int("run_index", run_index, minimum=0)

    bcert = _mapping("battery_cert", battery_cert)
    bbody = battery_body(bcert)
    bitems = battery_item_map(bcert)
    battery_id = _battery_id(bcert)
    recipe_battery = rec.get("battery")
    if battery_id is not None and isinstance(recipe_battery, str) and recipe_battery != battery_id:
        raise ValueError(
            f"recipe.battery {recipe_battery} is not the battery cert's id {battery_id}"
        )

    rows = _sequence("results", results)
    if not rows:
        raise ValueError("results is empty; a fingerprint covers its battery")
    records: list[dict] = []
    order: list[str] = []
    for k, row in enumerate(rows):
        record = item_record(_mapping(f"results[{k}]", row), redacted)
        iid = record["item_id"]
        if iid not in bitems:
            raise ValueError(f"results[{k}]: item {iid!r} is not in the battery")
        role = bitems[iid].get("role", "item")
        if not isinstance(role, str) or not role:
            raise ValueError(f"battery item {iid!r} has a non-string role {role!r}")
        record["role"] = role
        records.append(record)
        order.append(iid)
    if len(set(order)) != len(order):
        raise ValueError("results repeats an item_id")
    missing = sorted(set(bitems) - set(order))
    if missing:
        raise ValueError(f"results does not cover the battery; missing {missing}")

    if white_box is not None and kind != "weights":
        raise ValueError("the white-box channels need a weights subject (section 3.2)")

    channels: dict[str, Any] = {"exact": {"hash": exact_hash(records)}}
    channels["seqlp"] = {"present": all(r["seq_logprob"] is not None for r in records)}
    channels["topk"] = {"present": all(r["topk"] is not None for r in records)}
    if white_box is not None:
        channels.update(_white_box_channels(white_box, len(records)))

    nuis = _mapping("nuisance", nuisance)
    decoding = rec.get("decoding")
    batch_size = 1
    if isinstance(decoding, Mapping) and "batch_size" in decoding:
        batch_size = _int("recipe.decoding.batch_size", decoding.get("batch_size"), minimum=1)
    if "batch_size" not in nuis:
        nuis["batch_size"] = batch_size
    elif _int("nuisance.batch_size", nuis["batch_size"], minimum=1) != batch_size:
        raise ValueError(
            f"nuisance.batch_size {nuis['batch_size']} contradicts recipe.decoding.batch_size "
            f"{batch_size}"
        )
    computed_order = order_sha256(order)
    if "item_order_sha256" not in nuis:
        nuis["item_order_sha256"] = computed_order
    elif nuis["item_order_sha256"] != computed_order:
        raise ValueError(
            "nuisance.item_order_sha256 does not match the order the items were run in"
        )

    body: dict[str, Any] = {
        "run_index": run_index,
        "nuisance": nuis,
        "items": records,
        "channels": channels,
        "redacted": redacted,
        "tier": _TIER_BY_KIND[kind],
    }
    if bbody.get("redacted") is True:
        # Section 4.5: prompts withheld -> every fingerprint on this battery is non-verifiable.
        body["redacted_battery"] = True
    return body


# --------------------------------------------------------------------------- the floor block

def _names(what: str, value: Any) -> list[str]:
    names = _sequence(what, value)
    out = [_str(f"{what}[{k}]", v) for k, v in enumerate(names)]
    if len(set(out)) != len(out):
        raise ValueError(f"{what} repeats a name: {out!r}")
    return out


def attach_floor(
    canonical_body: Mapping[str, Any],
    run_bodies: Sequence[Mapping[str, Any]],
    plan_id: str,
    run_ids: Sequence[str],
    covers: Sequence[str],
    not_covered: Sequence[str],
) -> dict:
    """A NEW canonical body carrying ``noise_floor`` (section 5.1 step 3, section 5.4).

    ``run_bodies`` are the R run bodies of the plan (the canonical run among them) and
    ``run_ids`` their logged cert ids, in the same order.  Per-channel blocks come from
    ``styxx.v8.floor.floors`` -- ``{floor, distances, runs, pairs, alpha_single}`` -- and only
    channels present on every run appear; a channel absent anywhere has no floor and is left
    out, which is what makes ``verify`` say ``inconclusive`` rather than ``same``.  The exact
    channel's floor is computed with the roles the canonical body carries, so anchor flips do
    not move it.

    ``run_ids`` may also be ``len(run_bodies) - 1`` long, in which case they are the logged ids
    of ``run_bodies[1:]``: that is the section 5.1 step 5 append order, where the canonical run's
    only cert is the one this body goes into.  The block also carries the section 5.7 overall
    size (``alpha_overall`` with its ``standardization`` and ``alpha_overall_method``).
    """
    body = copy.deepcopy(_mapping("canonical_body", canonical_body))
    if body.get("run_index") != 0:
        raise ValueError(
            f"only the canonical run carries a floor (run_index 0), got {body.get('run_index')!r}"
        )
    items = _sequence("canonical_body.items", body.get("items"))
    roles: dict[str, str] = {}
    for k, item in enumerate(items):
        block = _mapping(f"canonical_body.items[{k}]", item)
        iid = _str(f"canonical_body.items[{k}].item_id", block.get("item_id"))
        role = block.get("role", "item")
        if not isinstance(role, str) or not role:
            raise ValueError(f"canonical_body.items[{k}] has a non-string role {role!r}")
        roles[iid] = role

    bodies = _sequence("run_bodies", run_bodies)
    ids = [_cert_id(f"run_ids[{k}]", v) for k, v in enumerate(_sequence("run_ids", run_ids))]
    if len(ids) not in (len(bodies), len(bodies) - 1):
        raise ValueError(f"{len(ids)} run ids for {len(bodies)} run bodies")
    if len(set(ids)) != len(ids):
        raise ValueError("run_ids repeats a cert id")
    plan = _cert_id("plan_id", plan_id)
    covers_l = _names("covers", covers)
    not_covered_l = _names("not_covered", not_covered)
    both = sorted(set(covers_l) & set(not_covered_l))
    if both:
        raise ValueError(f"{both} are listed as both covered and not covered")

    per = floormod.floors(bodies, roles=roles)
    per_channel = {ch: block for ch, block in per.items() if block is not None}
    if not per_channel:
        raise ValueError("no channel is present on every run; there is no floor to attach")

    body["noise_floor"] = {
        "plan": plan,
        "runs": ids,
        "covers": covers_l,
        "not_covered": not_covered_l,
        "per_channel": per_channel,
    }
    body["noise_floor"].update(_overall_size(bodies, per_channel, roles))
    return body


# The two sentences of section 5.7 that a reader needs beside the number, stored in the cert so
# the fraction is re-derivable from the same run certs the floor came from.  Neither is a
# threshold this module chose: the standardization and the zero-floor rule are quoted from the
# section, and nothing here invents a width for a floor the runs measured as 0.
STANDARDIZATION = (
    "max over the evaluated channels of d_c / floor_c, d_c the distance from the reference run; "
    "a channel whose floor_c is 0 contributes no ratio and is an exceedance whenever d_c > 0"
)
ALPHA_OVERALL_METHOD = (
    "section 5.7: over the R floor runs, each run's standardized maximum channel distance to the "
    "reference run is computed under 'standardization' and alpha_overall is the fraction of those "
    "runs that exceed the floor, over the channel set actually evaluated; no new runs are used"
)


def _round(x: float) -> float:
    """``ROUND_PLACES`` decimals, the rounding section 5.2 compares distances under."""
    return round(float(x), ROUND_PLACES) + 0.0


def _overall_size(
    bodies: Sequence[Mapping[str, Any]],
    per_channel: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, str],
) -> dict:
    """Section 5.7's ``alpha_overall`` and the two strings that make it re-derivable.

    A channel whose measured floor is 0 is never divided by: section 5.7 counts it as an
    exceedance whenever the distance is above 0, and that rule -- not a bare 0, and not a width
    this module made up -- is what governs a floor over runs that agreed exactly.
    """
    channels = sorted(per_channel)
    reference = bodies[0]
    per_run: list[dict] = []
    exceeded_count = 0
    for k, body in enumerate(bodies):
        ratios: list[float] = []
        zero_floor_exceedance = False
        for channel in channels:
            floor_c = float(per_channel[channel]["floor"])
            pair = floormod.pairwise([reference, body], channel, roles=roles)
            distance = float(pair["distances"][0])
            if floor_c > 0:
                ratios.append(_round(distance / floor_c))
            elif distance > 0:
                zero_floor_exceedance = True
        standardized = None if zero_floor_exceedance else _round(max(ratios) if ratios else 0.0)
        exceeds = zero_floor_exceedance or (standardized is not None and standardized > 1.0)
        exceeded_count += 1 if exceeds else 0
        per_run.append(
            {"run": k, "standardized_max": standardized, "exceeds": bool(exceeds)}
        )
    return {
        "alpha_overall": _round(exceeded_count / len(bodies)),
        "alpha_overall_method": ALPHA_OVERALL_METHOD,
        "standardization": STANDARDIZATION,
        "standardized_max": per_run,
    }


# --------------------------------------------------------------------------- the plan, applied

# The nuisance factors a runner can actually apply.  Section 5.1 step 2 varies "item order and,
# where the plan says so, `execution.batch_size`, physical GPU and driver"; of those, the two a
# process on one box can set are the item order and the decoding execution keys.  A plan naming
# anything else is refused at mint (`plan_run_settings`), never run at the default -- a factor
# declared and silently held fixed is the defect this section exists to stop.
ORDER_FACTORS = ("item_order", "order")
APPLICABLE_FACTORS = ("batch_size", "item_order", "order", "padding_side")

# The value names that mean "A.3 order" (item_id ascending).  Section 5.1 step 2 fixes the
# reference run as batch 1 in A.3 order, and run 0 takes the first value of every factor, so the
# order factor's FIRST declared value has to be one of these and no later value may be.
A3_ORDER_NAMES = frozenset({"a3", "a.3", "canonical", "identity", "reference"})

_PADDING_SIDES = ("left", "right")
_ORDER_TAG = b"styxx.v8/plan/order/1"


def order_for_value(item_ids: Sequence[str], value: str) -> list[str]:
    """The item order a non-A.3 plan value names: a permutation keyed by the value's own text.

    Deterministic and re-derivable by anyone holding the plan and the battery, which is what
    makes ``nuisance.item_order_sha256`` a checkable receipt rather than a number only the issuer
    could have produced.  A derivation that lands back on A.3 order is swapped at the first two
    positions, so a run whose assignment says "not the canonical order" never runs the canonical
    order.
    """
    ids = [_str(f"item_ids[{k}]", v) for k, v in enumerate(_sequence("item_ids", item_ids))]
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids repeats an id")
    text = _str("value", value).encode("utf-8")
    out = sorted(
        ids,
        key=lambda i: hashlib.sha256(_ORDER_TAG + b"\x00" + text + b"\x00" + i.encode("utf-8")).digest(),
    )
    a3 = sorted(ids, key=_id_key)
    if out == a3 and len(out) >= 2:
        out[0], out[1] = out[1], out[0]
    return out


def execution_state(cert: Mapping[str, Any]) -> tuple:
    """The execution a fingerprint cert records, as a value two runs can be compared on.

    A run's *label* is what its ``body.nuisance`` says.  Its *execution* is the set of forward
    passes it actually issued, and that is what a noise floor measures across.  This function
    derives the second from the bytes of one cert, so that two runs wearing different labels
    cannot be counted as two measurements when they were one.

    THE STATE, exactly::

        ("batches", <the multiset of batches>, <padding side or None>)

    * A **batch** is the tuple of ``item_id`` values computed in one forward pass, in the order
      they sit inside it.  It is derived, not read: ``body.items`` are stored in run order
      (this module's rule) and ``runner_hf`` batches them with ``range(0, len(items),
      batch_size)``, so the batches are ``items`` chunked by ``body.nuisance.batch_size``.
      A declared batch size the item count coerces -- 8, 16 and 32 over eight items -- yields
      the same chunking and therefore the same state, which is the point.
    * The batches are held as a **multiset** (sorted tuple), not a sequence: two runs that
      issued the same passes in a different sequence computed the same forward passes.  Within a
      batch the order is kept, because position inside a batch is padding and reduction order
      and is part of the pass.  The consequence, stated so it can be disagreed with: at
      ``batch_size`` 1 every item is its own pass and permuting the item order does not change
      the state at all.  A lab whose numbers move when independent single-item passes are issued
      in a different sequence is claiming cross-pass state on its accelerator; that is a claim
      about the box, and the way to put it in a floor is a factor that changes the batches.
    * **Padding side** is carried only when some batch holds two or more items.  A batch of one
      is padded to its own length and emits no pad token, so ``padding_side`` names nothing.

    When ``body.items`` is absent -- a redacted run carrying only ``items_blob`` -- the items
    cannot be chunked and the state falls back to ``("recorded", item_order_sha256, batch_size,
    padding)``: the labels, which is all such a cert offers.  Two redacted runs therefore
    separate on their recorded order hash, and a caller that needs the derived state has to
    resolve the blob.  This limit is real and is not hidden: it is why the check in
    ``styxx.v8.log`` names the state kind in its refusal.
    """
    c = _mapping("cert", cert)
    body = c.get("body")
    body = body if isinstance(body, Mapping) else {}
    nuisance = body.get("nuisance")
    nuisance = nuisance if isinstance(nuisance, Mapping) else {}
    batch = nuisance.get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        batch = None

    recipe = c.get("recipe")
    recipe = recipe if isinstance(recipe, Mapping) else {}
    decoding = recipe.get("decoding")
    decoding = decoding if isinstance(decoding, Mapping) else {}
    padding = decoding.get("padding_side")
    if not isinstance(padding, str):
        padding = nuisance.get("padding_side")
        padding = padding if isinstance(padding, str) else None

    items = body.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or batch is None:
        order = nuisance.get("item_order_sha256")
        return ("recorded", order if isinstance(order, str) else None, batch, padding)

    ids = _run_order(body)
    if ids is None:
        return ("recorded", nuisance.get("item_order_sha256"), batch, padding)
    return _state_of(ids, batch, padding)


def _run_order(body: Mapping[str, Any]) -> Optional[list[str]]:
    """The ``item_id`` values in the order the run stored them, or ``None`` if unreadable."""
    items = body.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return None
    out: list[str] = []
    for entry in items:
        if not isinstance(entry, Mapping):
            return None
        item_id = entry.get("item_id")
        if not isinstance(item_id, str):
            return None
        out.append(item_id)
    return out


def _state_of(ids: Sequence[str], batch: int, padding: Optional[str]) -> tuple:
    """``("batches", <multiset of batches>, <padding side or None>)`` -- see ``execution_state``."""
    batches = tuple(sorted(tuple(ids[k:k + batch]) for k in range(0, len(ids), batch)))
    return ("batches", batches, padding if any(len(b) > 1 for b in batches) else None)


def execution_states_for(
    cert: Mapping[str, Any], factor: str, values: Sequence[Any]
) -> Optional[set]:
    """The execution states ``factor``'s declared values would produce **at this run's own
    configuration** -- everything else the run recorded held where the run recorded it.

    This is the counterfactual that says whether a declared factor can move anything at all
    where it was declared.  ``{one state}`` means it cannot: the plan named a knob that, on this
    battery at this batch size, turns nothing.  ``None`` means the question is not answerable
    from the bytes -- the run carries no readable items, or the factor is one no process here can
    apply (``gpu``, ``region``, a physical driver), in which case the caller must say so rather
    than pretend either answer.

    The counterfactual is re-derivable by anyone holding the plan and the battery, because the
    order a plan value names is ``order_for_value`` and the batching is
    ``range(0, len(items), batch_size)``.  It is the same arithmetic ``plan_run_settings`` does
    at mint, asked backwards.
    """
    c = _mapping("cert", cert)
    body = c.get("body")
    body = body if isinstance(body, Mapping) else {}
    ids = _run_order(body)
    if ids is None:
        return None

    nuisance = body.get("nuisance")
    nuisance = nuisance if isinstance(nuisance, Mapping) else {}
    batch = nuisance.get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        return None
    recipe = c.get("recipe")
    recipe = recipe if isinstance(recipe, Mapping) else {}
    decoding = recipe.get("decoding")
    decoding = decoding if isinstance(decoding, Mapping) else {}
    padding = decoding.get("padding_side")
    if not isinstance(padding, str):
        padding = nuisance.get("padding_side")
        padding = padding if isinstance(padding, str) else None

    out: set = set()
    if factor in ORDER_FACTORS:
        a3 = sorted(ids, key=_id_key)
        for value in values:
            text = value if isinstance(value, str) else str(value)
            try:
                order = a3 if text.strip().lower() in A3_ORDER_NAMES else order_for_value(a3, text)
            except (ValueError, TypeError):
                return None
            out.add(_state_of(order, batch, padding))
        return out
    if factor == "batch_size":
        for value in values:
            try:
                size = int(str(value))
            except (ValueError, TypeError):
                return None
            if size < 1:
                return None
            out.add(_state_of(ids, size, padding))
        return out
    if factor == "padding_side":
        for value in values:
            out.add(_state_of(ids, batch, value if isinstance(value, str) else str(value)))
        return out
    return None


def plan_factors(plan_body: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """The plan's declared ``[(factor, values)]`` (section 5.1 step 1), factor-sorted."""
    body = _mapping("plan_body", plan_body)
    blocks = body.get("nuisance")
    if blocks is None:
        return []
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for k, block in enumerate(_sequence("plan.nuisance", blocks)):
        b = _mapping(f"plan.nuisance[{k}]", block)
        factor = _str(f"plan.nuisance[{k}].factor", b.get("factor"))
        values = [
            _str(f"plan.nuisance[{k}].values[{j}]", v)
            for j, v in enumerate(_sequence(f"plan.nuisance[{k}].values", b.get("values")))
        ]
        if not values:
            raise ValueError(f"plan.nuisance[{k}]: factor {factor!r} has no values")
        if len(set(values)) != len(values):
            raise ValueError(f"plan.nuisance[{k}]: factor {factor!r} repeats a value")
        if factor in seen:
            raise ValueError(f"plan.nuisance: factor {factor!r} is declared more than once")
        seen.add(factor)
        out.append((factor, values))
    return sorted(out, key=lambda pair: pair[0])


def assignment_sequence(
    factors: Sequence[tuple[str, Sequence[str]]], runs: int
) -> list[dict[str, str]]:
    """Run 0..R-1 -> the factor assignment each run takes (the enumeration rule of this module).

    THE RULE, stated once so a reader can re-derive every run's settings from the plan alone:

    1. Factors are taken in ``factor`` name order, values in the order the plan lists them.
    2. The **diagonal head** comes first.  With ``D = max(len(values))``, head entry ``k`` takes
       ``values_i[k % len(values_i)]`` for every factor ``i``, so run 0 is the plan's first value
       everywhere (section 5.1 step 2's reference run) and run 1 already differs in *every*
       factor that declares two or more values.  Head entries are pairwise distinct: two would
       need ``lcm(len(values_i))`` to divide their gap, and every gap is smaller than ``D``,
       which is at most that lcm.
    3. The **tail** is the rest of the cartesian product in lexicographic order (first factor
       slowest), skipping what the head already used.  Head + tail is therefore a permutation of
       the whole product: the first ``min(R, N)`` runs take pairwise different assignments.
    4. Run ``k`` takes entry ``k % N``.  When ``R`` exceeds the number ``N`` of assignments the
       plan's own value lists can form, assignments repeat -- ``R`` distinct assignments do not
       exist and the enumeration says so by cycling rather than by inventing values.

    Why not the plain odometer: with the last factor fastest, a plan of two three-valued factors
    run five times never leaves the first factor's first value, which is exactly the shape of the
    defect (`papers/v8/vacuous_floor_2026_09_09/`) -- a factor declared, and never varied.
    """
    runs = _int("runs", runs, minimum=1)
    pairs = [(_str("factor", f), [_str("value", v) for v in vals]) for f, vals in factors]
    if not pairs:
        return [{} for _ in range(runs)]
    names = [f for f, _ in pairs]
    values = [v for _, v in pairs]
    head = [tuple(vals[k % len(vals)] for vals in values) for k in range(max(len(v) for v in values))]
    used = set(head)
    tail = [combo for combo in itertools.product(*values) if combo not in used]
    ordered = head + tail
    return [dict(zip(names, ordered[k % len(ordered)])) for k in range(runs)]


def plan_run_settings(
    plan_body: Mapping[str, Any],
    runs: int,
    recipe: Mapping[str, Any],
    item_ids: Sequence[str],
) -> list[dict]:
    """What each of the R runs of a plan actually runs: recipe, item order, nuisance record.

    Returns one ``{"assignment", "recipe", "order", "nuisance"}`` per run index.  ``recipe`` is
    the run's own recipe (the plan's ``batch_size``/``padding_side`` written into ``decoding``,
    so the cert records the execution it ran under rather than the one it was launched with),
    ``order`` is the item order (``None`` = A.3 order) and ``nuisance`` is the assignment as it
    goes into ``body.nuisance``, which is what makes the plan checkable against the runs.

    Refusals, all at mint time and all naming the factor:

    * a declared factor no runner can apply (``gpu``, ``region``, ``time_of_day``, ...);
    * both ``item_order`` and ``order`` declared -- one factor under two names;
    * an order factor whose first value does not name A.3 order, or whose later values do
      (section 5.1 step 2: the reference run is batch 1 in A.3 order);
    * an order factor whose values cannot be realized as distinct orders on this battery;
    * a ``batch_size`` or ``padding_side`` whose first value is not the recipe's own -- run 0 is
      the recipe as written, and a plan that renames it has moved the reference run;
    * a ``batch_size`` value that is not a positive integer, a ``padding_side`` outside
      ``left``/``right``.
    """
    factors = plan_factors(plan_body)
    runs = _int("runs", runs, minimum=1)
    rec = _mapping("recipe", recipe)
    ids = [_str(f"item_ids[{k}]", v) for k, v in enumerate(_sequence("item_ids", item_ids))]
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids repeats an id")
    a3 = sorted(ids, key=_id_key)

    decoding = rec.get("decoding")
    decoding = decoding if isinstance(decoding, Mapping) else {}
    base_batch = decoding.get("batch_size", 1)
    base_batch = _int("recipe.decoding.batch_size", base_batch, minimum=1)
    base_padding = decoding.get("padding_side", "left")

    declared = [f for f, _ in factors]
    if len([f for f in declared if f in ORDER_FACTORS]) > 1:
        raise ValueError(
            "the plan declares " + " and ".join(f for f in declared if f in ORDER_FACTORS)
            + ": one item-order factor under two names cannot be applied as two"
        )

    orders: dict[str, Optional[list[str]]] = {}
    batches: dict[str, int] = {}
    paddings: dict[str, str] = {}
    for factor, values in factors:
        if factor in ORDER_FACTORS:
            if values[0].strip().lower() not in A3_ORDER_NAMES:
                raise ValueError(
                    f"nuisance factor {factor!r}: the first value is {values[0]!r}, which does not "
                    f"name A.3 order (one of {sorted(A3_ORDER_NAMES)}); run 0 takes the first value "
                    "of every factor and section 5.1 step 2 fixes it as batch 1 in A.3 order"
                )
            for v in values[1:]:
                if v.strip().lower() in A3_ORDER_NAMES:
                    raise ValueError(
                        f"nuisance factor {factor!r}: value {v!r} names A.3 order a second time; "
                        "two names for one order is a factor that varies on paper only"
                    )
            realized: dict[str, Optional[list[str]]] = {values[0]: None}
            seen_orders = {tuple(a3): values[0]}
            for v in values[1:]:
                perm = order_for_value(a3, v)
                key = tuple(perm)
                if key in seen_orders:
                    raise ValueError(
                        f"nuisance factor {factor!r}: values {seen_orders[key]!r} and {v!r} realize "
                        f"the same order on a battery of {len(a3)} items, so the plan cannot be "
                        "applied as declared"
                    )
                seen_orders[key] = v
                realized[v] = perm
            orders = realized
        elif factor == "batch_size":
            sizes: dict[str, int] = {}
            for v in values:
                try:
                    size = int(v)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"nuisance factor 'batch_size': value {v!r} is not an integer"
                    ) from None
                if size < 1:
                    raise ValueError(f"nuisance factor 'batch_size': value {v!r} is not positive")
                sizes[v] = size
            if sizes[values[0]] != base_batch:
                raise ValueError(
                    f"nuisance factor 'batch_size': the first value is {values[0]!r} while the "
                    f"recipe runs at {base_batch}; run 0 is the recipe as written (section 5.1 "
                    "step 2), so the plan's first value is the recipe's own batch size"
                )
            batches = sizes
        elif factor == "padding_side":
            for v in values:
                if v not in _PADDING_SIDES:
                    raise ValueError(
                        f"nuisance factor 'padding_side': value {v!r} is not "
                        f"{' or '.join(_PADDING_SIDES)}"
                    )
            if values[0] != base_padding:
                raise ValueError(
                    f"nuisance factor 'padding_side': the first value is {values[0]!r} while the "
                    f"recipe runs {base_padding!r}; run 0 is the recipe as written (section 5.1 "
                    "step 2)"
                )
            paddings = {v: v for v in values}
        else:
            raise ValueError(
                f"the runner cannot apply nuisance factor {factor!r}: this process can set "
                f"{', '.join(APPLICABLE_FACTORS)} and nothing else, and a declared factor run at "
                "the default is a plan the runs did not honour (section 5.1 step 2). Drop it from "
                "the plan, or measure it with a runner that can vary it."
            )

    out: list[dict] = []
    for assignment in assignment_sequence(factors, runs):
        run_recipe = copy.deepcopy(dict(rec))
        nuis: dict[str, Any] = {}
        order: Optional[list[str]] = None
        for factor, value in assignment.items():
            if factor in ORDER_FACTORS:
                order = list(orders[value]) if orders.get(value) is not None else None
                nuis[factor] = value
            elif factor == "batch_size":
                run_decoding = dict(run_recipe.get("decoding") or {})
                run_decoding["batch_size"] = batches[value]
                run_recipe["decoding"] = run_decoding
                nuis["batch_size"] = batches[value]
            else:  # padding_side
                run_decoding = dict(run_recipe.get("decoding") or {})
                run_decoding["padding_side"] = paddings[value]
                run_recipe["decoding"] = run_decoding
                nuis["padding_side"] = paddings[value]
        out.append(
            {"assignment": dict(assignment), "recipe": run_recipe, "order": order, "nuisance": nuis}
        )
    return out


# --------------------------------------------------------------------------- driving a runner

def run_fingerprint(
    runner: Any,
    subject: Mapping[str, Any],
    recipe: Mapping[str, Any],
    battery_cert: Mapping[str, Any],
    *,
    run_index: int,
    nuisance: Mapping[str, Any],
    order: Optional[Sequence[str]] = None,
    redacted: bool = False,
) -> dict:
    """Run one fingerprint: order the items, call the runner, return the body.

    ``order`` is the item order for this run -- ``None`` means A.3 order (``item_id``
    ascending), anything else must be a permutation of the battery's item ids (the nuisance
    factor of section 5.1).  ``nuisance.gpu``/``nuisance.driver`` are filled from
    ``runner.environment()`` when the caller did not set them (section 3.1).

    Refuses before running when ``runner.subject(subject)`` names a different S_identity than
    ``subject`` does (``ValueError``), and when the runner does not report one at all
    (``runner.SubjectUnavailable``): the body this returns carries ``subject``, and a body may
    not name weights that did not produce it.
    """
    bitems = battery_item_map(battery_cert)
    if order is None:
        ids = sorted(bitems, key=_id_key)
    else:
        ids = [_str(f"order[{k}]", v) for k, v in enumerate(_sequence("order", order))]
        if len(set(ids)) != len(ids):
            raise ValueError("order repeats an item_id")
        if set(ids) != set(bitems):
            raise ValueError("order must be a permutation of the battery's item ids")

    # The same guard `verify --ref` applies, applied where the claim is MADE: a fingerprint body
    # records `subject`, so a runner that will run other weights must not be allowed to produce
    # one.  `SubjectUnavailable` (a runner that will not say) propagates -- the CLI turns it into
    # exit 5, unavailable, not a cert.
    reported = runnermod.reported_identity(runner, subject)
    declared = runnermod.subject_identity(subject)
    differs = [name for name, value in reported.items() if declared.get(name) != value]
    if differs:
        raise ValueError(
            "the runner runs a different subject than this fingerprint names: "
            + ", ".join(
                f"{name} {declared.get(name)!r} != {reported[name]!r}" for name in differs
            )
            + " -- a cert may not record a subject that was not run (section 2.2)"
        )

    run_items: list[dict] = []
    for iid in ids:
        prompt = bitems[iid].get("prompt_text")
        if not isinstance(prompt, str):
            raise ValueError(
                f"battery item {iid!r} carries no prompt_text; a redacted battery cannot be run"
            )
        run_items.append({"item_id": iid, "prompt_text": prompt})

    results = runner.run(run_items, dict(recipe), dict(subject))
    rows = _sequence("runner results", results)
    if [_mapping(f"runner results[{k}]", r).get("item_id") for k, r in enumerate(rows)] != ids:
        raise ValueError("the runner did not return one result per item in the order given")

    nuis = _mapping("nuisance", nuisance)
    if "gpu" not in nuis or "driver" not in nuis:
        env = runner.environment()
        hardware = env.get("hardware") if isinstance(env, Mapping) else None
        if isinstance(hardware, Mapping):
            nuis.setdefault("gpu", hardware.get("gpu"))
            nuis.setdefault("driver", hardware.get("driver"))

    body = build(
        subject,
        recipe,
        battery_cert,
        rows,
        run_index=run_index,
        nuisance=nuis,
        redacted=redacted,
    )
    # The C-MOCK marker (``styxx/v8/cert.py``, "Decisions"): a runner that computes from a hash
    # instead of from a model labels the numbers it produced, in the bytes that get signed. It
    # is stamped HERE rather than in the CLI because this is the one place a fingerprint body is
    # built from a runner, so no minting path can reach a body without passing it.
    if runnermod.is_synthetic(runner):
        body[runnermod.SYNTHETIC] = True
    return body
