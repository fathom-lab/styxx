"""styxx.v8.sweep -- the delta sweep record (spec section 4.2, Appendix C).

Appendix C scores an item from four passes over the same pool:

* the **reference** pass -- batch 1, A.3 order, the subject's own precision, one device;
* **delta1** -- the same pool at other precisions (bf16 -> fp16 -> int8-bnb -> nf4-bnb);
* **delta2** -- the enumerated nuisance configurations (batch size, item order, device);
* **delta4** -- optional semantically-null template jitter.

``run_sweep`` runs all of them and returns one JSON-serializable *sweep record*.  It computes
nothing: ``styxx.v8.battery.score`` reads the record and produces Appendix C's numbers, and
``styxx.v8.battery.select`` turns those into a canary-v1 battery body.  Splitting it this way
keeps the expensive part (model passes) separate from the arithmetic, so the arithmetic can be
tested on hand-built records and a recorded sweep can be re-scored with different parameters.

The record shape (contract section 7)::

    {
      "reference": { item_id: ItemResult },
      "delta1":    { precision: { item_id: ItemResult } },
      "delta2":    [ { "config": {...}, "results": { item_id: ItemResult } } ],
      "delta4":    [ { "config": {...}, "results": { item_id: ItemResult } } ],
      "params":    { ... what the sweep was }
    }

Determinism: the item order is derived from the item ids alone (A.3 for the canonical order,
a sha256-keyed permutation for ``order: "perm"``), so two runs of ``run_sweep`` against the
same runner factory produce byte-identical records.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from styxx.v8.jcs import canonical_bytes, sha256_hex

__all__ = [
    "RECORD_TAG",
    "REFERENCE_BATCH_SIZE",
    "a3_order",
    "item_order_sha256",
    "pool_sha256",
    "read_record",
    "run_sweep",
    "write_record",
]

# Recorded in ``params.record`` so a reader can tell what it is holding.
RECORD_TAG = "styxx.v8/sweep/1"

# Section 4.2: the reference configuration is (batch 1, A.3 order, device 0, reference
# precision, reference runtime).
REFERENCE_BATCH_SIZE = 1

_ORDERS = ("canonical", "perm")
_DELTA2_KEYS = frozenset({"batch_size", "order", "perm_seed", "device"})
_DELTA4_KEYS = frozenset({"id", "prompt_prefix", "prompt_suffix"})


# --------------------------------------------------------------------------- pool handling

def _utf8(s: str) -> bytes:
    return s.encode("utf-8")


def _pool(pool_items: Any) -> list[dict]:
    """Normalize the pool to ``[{item_id, prompt_text, family?}]``; refuse anything else."""
    if isinstance(pool_items, Mapping) or not isinstance(pool_items, Sequence):
        raise ValueError("pool_items must be a list of {item_id, prompt_text}")
    out: list[dict] = []
    seen: set[str] = set()
    for i, it in enumerate(pool_items):
        if not isinstance(it, Mapping):
            raise ValueError(f"pool_items[{i}] is not a mapping")
        item_id = it.get("item_id")
        prompt = it.get("prompt_text")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"pool_items[{i}]: item_id must be a non-empty string")
        if not isinstance(prompt, str):
            raise ValueError(f"pool_items[{i}] ({item_id!r}): prompt_text must be a string")
        if item_id in seen:
            raise ValueError(f"pool_items: duplicate item_id {item_id!r}")
        seen.add(item_id)
        rec = {"item_id": item_id, "prompt_text": prompt}
        if "family" in it:
            rec["family"] = it["family"]
        out.append(rec)
    if not out:
        raise ValueError("pool_items is empty")
    return out


def a3_order(items: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Appendix A.3: items ordered by ``item_id`` ascending in UTF-8 byte order."""
    return [dict(it) for it in sorted(items, key=lambda it: _utf8(str(it["item_id"])))]


def _permuted(items: Sequence[Mapping[str, Any]], perm_seed: int) -> list[dict]:
    """A deterministic permutation of the A.3 order keyed by ``perm_seed``.

    The sort key is ``sha256(JCS([perm_seed, item_id]))``, so the permutation depends only on
    the seed and the item ids -- never on the order the caller happened to pass.
    """
    base = a3_order(items)

    def key(it: Mapping[str, Any]) -> bytes:
        return hashlib.sha256(canonical_bytes([perm_seed, str(it["item_id"])])).digest()

    return sorted(base, key=key)


def item_order_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """``sha256(UTF-8("\\n".join(item ids in the given order)))`` -- section 3.1 ``nuisance``."""
    return sha256_hex(_utf8("\n".join(str(it["item_id"]) for it in items)))


def pool_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """The pool digest recorded in a canary-v1 body (section 4.5), 64 hex, no prefix.

    Preimage: JCS of the A.3-ordered list of ``{item_id, prompt_sha256, family}``, with
    ``prompt_sha256 = sha256(UTF-8(prompt_text))`` and ``family`` null when the pool file
    carries none.  Prompt text is hashed rather than embedded so the digest is stable under
    the record's own formatting.
    """
    rows = [
        {
            "item_id": str(it["item_id"]),
            "prompt_sha256": sha256_hex(_utf8(str(it.get("prompt_text", "")))),
            "family": it.get("family"),
        }
        for it in a3_order(items)
    ]
    return sha256_hex(canonical_bytes(rows))


# --------------------------------------------------------------------------- configurations

def _recipe_at(recipe: Mapping[str, Any], batch_size: int) -> dict:
    r = copy.deepcopy(dict(recipe))
    decoding = dict(r.get("decoding") or {})
    decoding["batch_size"] = batch_size
    r["decoding"] = decoding
    return r


def _delta2_config(i: int, cfg: Any) -> dict:
    if not isinstance(cfg, Mapping):
        raise ValueError(f"delta2[{i}] is not a mapping")
    unknown = set(cfg) - _DELTA2_KEYS
    if unknown:
        raise ValueError(f"delta2[{i}]: unknown keys {sorted(unknown)}")
    bs = cfg.get("batch_size")
    if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
        raise ValueError(f"delta2[{i}]: batch_size must be an int >= 1, got {bs!r}")
    order = cfg.get("order", "canonical")
    if order not in _ORDERS:
        raise ValueError(f"delta2[{i}]: order must be one of {list(_ORDERS)}, got {order!r}")
    out: dict[str, Any] = {"batch_size": bs, "order": order}
    if order == "perm":
        seed = cfg.get("perm_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"delta2[{i}]: order 'perm' needs an integer perm_seed")
        out["perm_seed"] = seed
    elif "perm_seed" in cfg:
        raise ValueError(f"delta2[{i}]: perm_seed is meaningless with order 'canonical'")
    if "device" in cfg:
        out["device"] = cfg["device"]
    return out


def _delta4_config(i: int, cfg: Any) -> dict:
    if not isinstance(cfg, Mapping):
        raise ValueError(f"delta4[{i}] is not a mapping")
    unknown = set(cfg) - _DELTA4_KEYS
    if unknown:
        raise ValueError(f"delta4[{i}]: unknown keys {sorted(unknown)}")
    out: dict[str, Any] = {"id": str(cfg.get("id", f"jitter-{i}"))}
    for key in ("prompt_prefix", "prompt_suffix"):
        if key in cfg:
            if not isinstance(cfg[key], str):
                raise ValueError(f"delta4[{i}]: {key} must be a string")
            out[key] = cfg[key]
    if "prompt_prefix" not in out and "prompt_suffix" not in out:
        raise ValueError(f"delta4[{i}]: needs prompt_prefix or prompt_suffix")
    return out


def _jittered(items: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[dict]:
    prefix = cfg.get("prompt_prefix", "")
    suffix = cfg.get("prompt_suffix", "")
    out = []
    for it in items:
        rec = dict(it)
        rec["prompt_text"] = prefix + str(it["prompt_text"]) + suffix
        out.append(rec)
    return out


# --------------------------------------------------------------------------- running

def _results_by_id(runner: Any, items: list[dict], recipe: Mapping[str, Any], subject: Mapping[str, Any]) -> dict:
    """Run ``items`` in the order given and key the results by ``item_id``.

    The runner returns results in the order it was handed the items; keying by id here is what
    makes the record order-independent, which is exactly what the delta2 family needs -- an
    order change must show up as a changed *output*, never as a changed record layout.
    """
    payload = [{"item_id": it["item_id"], "prompt_text": it["prompt_text"]} for it in items]
    results = runner.run(payload, dict(recipe), dict(subject))
    if len(results) != len(payload):
        raise ValueError(f"runner returned {len(results)} results for {len(payload)} items")
    out: dict[str, dict] = {}
    for given, got in zip(payload, results):
        rec = dict(got)
        if rec.get("item_id") != given["item_id"]:
            raise ValueError(
                f"runner returned results out of order: expected {given['item_id']!r}, "
                f"got {rec.get('item_id')!r}"
            )
        out[given["item_id"]] = rec
    return out


def run_sweep(
    runner_factory: Callable[[Any], Any],
    pool_items: Any,
    recipe: Mapping[str, Any],
    subject: Mapping[str, Any],
    *,
    delta1: list[str],
    delta2: list[dict],
    delta4: list[dict] = [],
) -> dict:
    """Run the delta sweep of section 4.2 and return the sweep record.

    ``runner_factory(precision)`` returns a ``Runner`` for that precision; it is called once per
    distinct precision and the result is reused.  The reference pass uses the subject's own
    ``precision``.

    Raises ``ValueError`` on a malformed pool, a malformed delta configuration, a delta1 entry
    equal to the reference precision (its flip1 contribution would be zero by construction and
    would deflate every score), a duplicate delta1 precision, or a runner that returns results
    out of order.
    """
    items = _pool(pool_items)
    ordered = a3_order(items)
    ref_precision = subject.get("precision")

    d1_list = list(delta1 or [])
    if d1_list and ref_precision is None:
        raise ValueError("delta1 needs a subject carrying a reference precision (kind 'weights')")
    seen_p: set[str] = set()
    for p in d1_list:
        if not isinstance(p, str) or not p:
            raise ValueError(f"delta1: precision must be a non-empty string, got {p!r}")
        if p == ref_precision:
            raise ValueError(f"delta1: {p!r} is the reference precision, not a perturbation")
        if p in seen_p:
            raise ValueError(f"delta1: duplicate precision {p!r}")
        seen_p.add(p)

    d2_configs = [_delta2_config(i, c) for i, c in enumerate(delta2 or [])]
    d4_configs = [_delta4_config(i, c) for i, c in enumerate(delta4 or [])]

    runners: dict[Any, Any] = {}

    def runner_for(precision: Any) -> Any:
        if precision not in runners:
            runners[precision] = runner_factory(precision)
        return runners[precision]

    ref_recipe = _recipe_at(recipe, REFERENCE_BATCH_SIZE)
    ref_runner = runner_for(ref_precision)
    reference = _results_by_id(ref_runner, ordered, ref_recipe, subject)

    d1_results: dict[str, dict] = {}
    for p in d1_list:
        subject_p = dict(subject)
        subject_p["precision"] = p
        d1_results[p] = _results_by_id(runner_for(p), ordered, ref_recipe, subject_p)

    d2_results: list[dict] = []
    for cfg in d2_configs:
        run_items = ordered if cfg["order"] == "canonical" else _permuted(ordered, cfg["perm_seed"])
        d2_results.append({
            "config": dict(cfg),
            "order_sha256": item_order_sha256(run_items),
            "results": _results_by_id(ref_runner, run_items, _recipe_at(recipe, cfg["batch_size"]), subject),
        })

    d4_results: list[dict] = []
    for cfg in d4_configs:
        d4_results.append({
            "config": dict(cfg),
            "results": _results_by_id(ref_runner, _jittered(ordered, cfg), ref_recipe, subject),
        })

    params = {
        "record": RECORD_TAG,
        "reference": {
            "batch_size": REFERENCE_BATCH_SIZE,
            "order": "canonical",
            "precision": ref_precision,
        },
        "delta1": list(d1_list),
        "delta2": [dict(c) for c in d2_configs],
        "delta4": [dict(c) for c in d4_configs],
        "delta4_run": bool(d4_configs),
        "pool_size": len(ordered),
        "pool_sha256": pool_sha256(ordered),
        "item_order_sha256": item_order_sha256(ordered),
    }

    return {
        "reference": reference,
        "delta1": d1_results,
        "delta2": d2_results,
        "delta4": d4_results,
        "params": params,
    }


# --------------------------------------------------------------------------- storage

def write_record(record: Mapping[str, Any], path: Any) -> Path:
    """Write the record as UTF-8 JSON with LF endings and no BOM; returns the path."""
    p = Path(path)
    text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


def read_record(path: Any) -> dict:
    """Read a record written by ``write_record``."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        record = json.load(fh)
    if not isinstance(record, dict):
        raise ValueError("sweep record must be a JSON object")
    return record
