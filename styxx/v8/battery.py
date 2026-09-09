"""styxx.v8.battery -- battery cert bodies and canary-v1 selection (spec section 4, Appendix C).

Three kinds of battery body live here:

* ``pool_v1(items)``      -- the candidate set a canary selection draws from;
* ``fixed_v1(items, source=...)`` -- the public, model-agnostic battery;
* ``select(...)``         -- canary-v1, selected against one weights subject.

``score`` implements Appendix C verbatim over a sweep record (``styxx.v8.sweep``); ``select``
implements section 4.4 over those scores; ``validate_body`` is the structural check that the
JSON schema cannot express on its own (the share rule, the exclusion rule, the anchor rule).

Two readings the spec leaves open, both recorded in ``params`` so a reader can see them:

* Section 4.4 step 5 says the battery is the top-N union the anchors.  An item that qualifies
  for both takes the ANCHOR role, because Appendix B counts anchors separately from items and
  an item cannot be counted twice.  ``params.n_actual`` is the number of role ``canary`` items
  that survived, ``params.anchor_overlap`` how many were taken by the anchor role.
* ``zero(i)`` (Appendix C) is evaluated on the STORED, rounded numbers, the same way section
  5.2 compares distances after rounding.  ``flip3 = 1 - exp(stay)`` is a float; on a confident
  item ``exp(stay)`` is 1.0 exactly and flip3 is 0.0, but a value of 4e-10 is the same claim
  and rounding is what makes the two agree.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from styxx.v8.consts import BATTERY_KINDS, FAMILIES, ROUND_PLACES
from styxx.v8.distances import rounded
from styxx.v8.jcs import sha256_hex
from styxx.v8.sweep import a3_order, pool_sha256

__all__ = [
    "ANCHOR_K",
    "SELECTION_ITEM_FIELDS",
    "W_FLIP",
    "W_MARGIN",
    "fixed_v1",
    "pool_v1",
    "score",
    "select",
    "validate_body",
]

# Appendix C: s(i) = W_FLIP * (flip1 + flip3) / 2 + W_MARGIN * exp(-margin / tau).
W_FLIP = 0.6
W_MARGIN = 0.4

# Section 4.4 step 5: K = 64 by default; ``select`` takes it as ``k``.
ANCHOR_K = 64

TIE_BREAK = "margin asc, item_id asc"

# The per-item fields that exist only under canary-v1 (schema/battery.json enforces the same).
SELECTION_ITEM_FIELDS = ("score", "margin", "margin_by_position", "flip1", "flip2", "flip3", "flip4")

_ITEM_ROLES = ("item", "canary", "anchor")
_ZERO = 10.0 ** (-ROUND_PLACES) / 2.0  # anything that rounds to 0.0 at ROUND_PLACES places


def _utf8(s: str) -> bytes:
    return s.encode("utf-8")


def _is_zero(x: float) -> bool:
    return abs(float(x)) < _ZERO


# --------------------------------------------------------------------------- Appendix C

def _token_ids(rec: Any, where: str) -> tuple[int, ...]:
    if not isinstance(rec, Mapping):
        raise ValueError(f"{where}: item result must be a mapping")
    ids = rec.get("token_ids")
    if ids is None or isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise ValueError(f"{where}: token_ids must be a list of integers")
    for t in ids:
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError(f"{where}: token_ids must be integers")
    return tuple(int(t) for t in ids)


def _finite(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{where}: expected a number, got {x!r}")
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{where}: expected a finite number, got {x!r}")
    return v


def _variant_results(record: Mapping[str, Any]) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]], list[tuple[str, dict]]]:
    """``(delta1, delta2, delta4)`` each as ``[(label, {item_id: ItemResult})]``."""
    d1_raw = record.get("delta1")
    if d1_raw is None:
        d1_raw = {}
    if not isinstance(d1_raw, Mapping):
        raise ValueError("record.delta1 must be an object keyed by precision")
    d1 = [(str(p), d1_raw[p]) for p in sorted(d1_raw)]

    def as_list(key: str) -> list[tuple[str, dict]]:
        raw = record.get(key)
        if raw is None:
            raw = []
        if isinstance(raw, Mapping) or isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError(f"record.{key} must be a list of {{config, results}}")
        out = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, Mapping) or "results" not in entry:
                raise ValueError(f"record.{key}[{i}] needs a 'results' object")
            out.append((f"{key}[{i}]", entry["results"]))
        return out

    return d1, as_list("delta2"), as_list("delta4")


def _flip(item_id: str, ref_ids: tuple[int, ...], variants: list[tuple[str, dict]], family: str) -> float | None:
    """Appendix C: the fraction of variants whose greedy token ids differ from the reference."""
    if not variants:
        return None
    differ = 0
    for label, results in variants:
        if not isinstance(results, Mapping):
            raise ValueError(f"{family} {label}: results must be an object keyed by item_id")
        if item_id not in results:
            raise ValueError(f"{family} {label}: item {item_id!r} is missing")
        if _token_ids(results[item_id], f"{family} {label}[{item_id!r}]") != ref_ids:
            differ += 1
    return differ / len(variants)


def score(record: dict, tau: float = 1.0) -> dict[str, dict]:
    """Appendix C scores for every item in the sweep record's reference pass.

    Returns ``{item_id: {margin, margin_by_position, stay, flip1, flip2, flip3, flip4, s}}``.
    ``flip4`` is ``None`` when delta4 was not run.  A delta family with no variants yields a
    flip fraction of ``0.0`` -- an unrun family cannot flip anything, and ``params`` on the
    record (and on the battery body ``select`` builds) records how many variants each fraction
    rests on, so a zero from evidence and a zero from absence stay distinguishable.

    Raises ``ValueError`` when the reference pass carries no log-probs (``margin_by_position``
    or ``stay`` is ``None``): canary-v1 is selected against a weights subject and Appendix C
    has no fallback without them.
    """
    if not isinstance(record, Mapping):
        raise ValueError("sweep record must be a mapping")
    t = _finite(tau, "tau")
    if t <= 0:
        raise ValueError(f"tau must be > 0, got {tau!r}")
    reference = record.get("reference")
    if not isinstance(reference, Mapping) or not reference:
        raise ValueError("record.reference must be a non-empty object keyed by item_id")

    d1, d2, d4 = _variant_results(record)

    out: dict[str, dict] = {}
    for item_id in sorted(reference, key=_utf8):
        ref = reference[item_id]
        ref_ids = _token_ids(ref, f"reference[{item_id!r}]")

        mbp = ref.get("margin_by_position") if isinstance(ref, Mapping) else None
        if mbp is None or isinstance(mbp, (str, bytes)) or not isinstance(mbp, Sequence) or not mbp:
            raise ValueError(
                f"reference[{item_id!r}]: margin_by_position is required by Appendix C "
                "(a subject without log-probs cannot carry a canary selection)"
            )
        positions = [_finite(v, f"reference[{item_id!r}].margin_by_position") for v in mbp]
        margin = min(positions)

        stay = ref.get("stay")
        if stay is None:
            raise ValueError(f"reference[{item_id!r}]: stay is required by Appendix C")
        stay_v = _finite(stay, f"reference[{item_id!r}].stay")
        # flip3 = 1 - exp(stay); stay <= 0 by construction, so flip3 lands in [0, 1).
        flip3 = min(1.0, max(0.0, 1.0 - math.exp(stay_v)))

        flip1 = _flip(item_id, ref_ids, d1, "delta1")
        flip2 = _flip(item_id, ref_ids, d2, "delta2")
        flip4 = _flip(item_id, ref_ids, d4, "delta4")
        f1 = 0.0 if flip1 is None else flip1
        f2 = 0.0 if flip2 is None else flip2

        s = W_FLIP * (f1 + flip3) / 2.0 + W_MARGIN * math.exp(-margin / t)

        out[item_id] = {
            "margin": rounded(margin),
            "margin_by_position": [rounded(v) for v in positions],
            "stay": rounded(stay_v),
            "flip1": rounded(f1),
            "flip2": rounded(f2),
            "flip3": rounded(flip3),
            "flip4": None if flip4 is None else rounded(flip4),
            "s": rounded(s),
        }
    return out


# --------------------------------------------------------------------------- item helpers

def _norm_pool_item(i: int, it: Any, *, need_family: bool) -> dict:
    if not isinstance(it, Mapping):
        raise ValueError(f"items[{i}] is not a mapping")
    item_id = it.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"items[{i}]: item_id must be a non-empty string")
    prompt = it.get("prompt_text")
    if not isinstance(prompt, str):
        raise ValueError(f"items[{i}] ({item_id!r}): prompt_text must be a string")
    family = it.get("family")
    if need_family or family is not None:
        if family not in FAMILIES:
            raise ValueError(f"items[{i}] ({item_id!r}): family must be one of {list(FAMILIES)}, got {family!r}")
    computed = sha256_hex(_utf8(prompt))
    given = it.get("prompt_sha256")
    if given is not None and given != computed:
        raise ValueError(f"items[{i}] ({item_id!r}): prompt_sha256 does not match prompt_text")
    return {
        "item_id": item_id,
        "prompt_sha256": computed,
        "prompt_text": prompt,
        "family": family,
        "role": "item",
    }


def _norm_pool(items: Any, *, need_family: bool) -> list[dict]:
    if isinstance(items, Mapping) or not isinstance(items, Sequence):
        raise ValueError("items must be a list")
    out = [_norm_pool_item(i, it, need_family=need_family) for i, it in enumerate(items)]
    if not out:
        raise ValueError("items is empty")
    seen: set[str] = set()
    for rec in out:
        if rec["item_id"] in seen:
            raise ValueError(f"items: duplicate item_id {rec['item_id']!r}")
        seen.add(rec["item_id"])
    return a3_order(out)


def _families_present(items: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {it.get("family") for it in items}
    return [f for f in FAMILIES if f in present]


# --------------------------------------------------------------------------- bodies

def pool_v1(items: Any) -> dict:
    """A ``pool-v1`` battery body (section 4.5): every candidate in the clear, role ``item``."""
    norm = _norm_pool(items, need_family=True)
    return {
        "kind": "pool-v1",
        "pool_sha256": pool_sha256(norm),
        "pool_size": len(norm),
        "families": _families_present(norm),
        "items": [dict(it) for it in norm],
        "redacted": False,
    }


def fixed_v1(items: list[dict], *, source: str) -> dict:
    """A ``fixed-v1`` battery body (section 4.5): public, model-agnostic, no selection fields."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string naming where the items came from")
    norm = _norm_pool(items, need_family=True)
    return {
        "kind": "fixed-v1",
        "source": source,
        "families": _families_present(norm),
        "items": [dict(it) for it in norm],
        "redacted": False,
    }


def select(
    pool_items: Any,
    scores: Mapping[str, Mapping[str, Any]],
    *,
    n: int,
    k: int,
    max_family_share: float = 0.25,
    tau: float = 1.0,
    perm_seed: int,
) -> dict:
    """Section 4.4: exclude, rank, stratify, take top-N, add anchors -> a canary-v1 body.

    ``pool_items`` is the candidate set (``{item_id, prompt_text, family}``); ``scores`` is what
    ``score`` returned for it.  Every pool item needs a score and every score needs a pool item.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an int >= 1, got {n!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"k must be an int >= 0, got {k!r}")
    if isinstance(perm_seed, bool) or not isinstance(perm_seed, int):
        raise ValueError(f"perm_seed must be an int, got {perm_seed!r}")
    share = _finite(max_family_share, "max_family_share")
    if not 0.0 < share <= 1.0:
        raise ValueError(f"max_family_share must be in (0, 1], got {max_family_share!r}")
    if not isinstance(scores, Mapping):
        raise ValueError("scores must be a mapping of item_id to the Appendix C block")

    pool = _norm_pool(pool_items, need_family=True)
    by_id = {it["item_id"]: it for it in pool}
    missing = sorted(set(by_id) - set(scores), key=_utf8)
    if missing:
        raise ValueError(f"scores: no score for pool items {missing[:5]}")
    extra = sorted(set(scores) - set(by_id), key=_utf8)
    if extra:
        raise ValueError(f"scores: scored items that are not in the pool {extra[:5]}")

    delta4_run = any(scores[i].get("flip4") is not None for i in by_id)

    # 1. Exclude every item with flip2 > 0 (section 4.4 step 1).
    excluded: list[dict] = []
    eligible: list[str] = []
    for item_id in sorted(by_id, key=_utf8):
        flip2 = _finite(scores[item_id].get("flip2", 0.0), f"scores[{item_id!r}].flip2")
        if _is_zero(flip2):
            eligible.append(item_id)
        else:
            excluded.append({"item_id": item_id, "flip2": rounded(flip2)})

    # 2. Rank by s desc; ties by margin asc then item_id asc.
    def rank_key(item_id: str) -> tuple:
        blk = scores[item_id]
        return (-_finite(blk["s"], f"scores[{item_id!r}].s"),
                _finite(blk["margin"], f"scores[{item_id!r}].margin"),
                _utf8(item_id))

    ranked = sorted(eligible, key=rank_key)

    # 3./4. Stratify by family under the share cap, then fill the shortfall by rank.
    cap = max(1, int(math.floor(n * share)))
    counts: dict[str, int] = {}
    chosen: list[str] = []
    taken: set[str] = set()
    for item_id in ranked:
        if len(chosen) >= n:
            break
        fam = by_id[item_id]["family"]
        if counts.get(fam, 0) >= cap:
            continue
        counts[fam] = counts.get(fam, 0) + 1
        chosen.append(item_id)
        taken.add(item_id)
    stratified = len(chosen)
    for item_id in ranked:
        if len(chosen) >= n:
            break
        if item_id in taken:
            continue
        counts[by_id[item_id]["family"]] = counts.get(by_id[item_id]["family"], 0) + 1
        chosen.append(item_id)
        taken.add(item_id)
    shortfall = len(chosen) - stratified

    # 5. Anchors: the k largest-margin items among zero(i).  zero(i) is read on the stored
    #    rounded numbers; excluded items cannot qualify because zero(i) requires flip2 == 0.
    def is_zero_item(item_id: str) -> bool:
        blk = scores[item_id]
        if not (_is_zero(blk.get("flip1", 0.0)) and _is_zero(blk.get("flip2", 0.0))
                and _is_zero(blk.get("flip3", 0.0))):
            return False
        f4 = blk.get("flip4")
        return f4 is None or _is_zero(f4)

    zero_items = [i for i in eligible if is_zero_item(i)]
    zero_items.sort(key=lambda i: (-_finite(scores[i]["margin"], f"scores[{i!r}].margin"), _utf8(i)))
    anchors = zero_items[:k]
    anchor_set = set(anchors)
    overlap = len(anchor_set & taken)

    selected = sorted(set(chosen) | anchor_set, key=_utf8)
    items: list[dict] = []
    for item_id in selected:
        blk = scores[item_id]
        rec = dict(by_id[item_id])
        rec["role"] = "anchor" if item_id in anchor_set else "canary"
        rec["score"] = rounded(blk["s"])
        rec["margin"] = rounded(blk["margin"])
        rec["margin_by_position"] = [rounded(v) for v in blk.get("margin_by_position", [])]
        rec["flip1"] = rounded(blk.get("flip1", 0.0))
        rec["flip2"] = rounded(blk.get("flip2", 0.0))
        rec["flip3"] = rounded(blk.get("flip3", 0.0))
        if delta4_run:
            f4 = blk.get("flip4")
            rec["flip4"] = None if f4 is None else rounded(f4)
        items.append(rec)

    canaries = [it for it in items if it["role"] == "canary"]
    # The probe's finding, made a number: what fraction of the SELECTED canaries still moves
    # under a precision change once the flip2 exclusion has run.  On the probe's battery every
    # precision-sensitive item was also batch-sensitive, so this is 0.0 and the battery carries
    # no measured precision sensitivity at all.
    sensitivity = 0.0
    if canaries:
        sensitivity = math.fsum(it["flip1"] for it in canaries) / len(canaries)

    params = {
        "tau": rounded(tau),
        "w_flip": W_FLIP,
        "w_margin": W_MARGIN,
        "n": n,
        "k_anchors": k,
        "k_anchors_actual": len(anchors),
        "max_family_share": share,
        "family_cap": cap,
        "family_counts": {f: counts[f] for f in FAMILIES if f in counts},
        "shortfall": shortfall,
        "n_actual": len(canaries),
        "anchor_overlap": overlap,
        "perm_seed": perm_seed,
        "delta4_run": delta4_run,
        "tie_break": TIE_BREAK,
        "sensitivity_after_exclusion": rounded(sensitivity),
    }

    return {
        "kind": "canary-v1",
        "pool_sha256": pool_sha256(pool),
        "pool_size": len(pool),
        "params": params,
        "families": _families_present(pool),
        "items": items,
        "excluded": excluded,
        "redacted": False,
    }


# --------------------------------------------------------------------------- validation

def validate_body(body: dict) -> list[str]:
    """Structural check of a battery body; ``[]`` means it holds.

    Reason strings begin with one of ``body: ``, ``kind: ``, ``items: ``, ``item[<id>]: ``,
    ``params: ``, ``excluded: ``, ``families: ``, ``share: ``, ``anchors: ``.  Match on the
    prefix, never on the tail.
    """
    reasons: list[str] = []
    if not isinstance(body, Mapping):
        return ["body: not a JSON object"]

    kind = body.get("kind")
    if kind not in BATTERY_KINDS:
        reasons.append(f"kind: must be one of {list(BATTERY_KINDS)}, got {kind!r}")
    canary = kind == "canary-v1"

    raw_items = body.get("items")
    if isinstance(raw_items, Mapping) or not isinstance(raw_items, Sequence):
        reasons.append("items: must be a list")
        return reasons
    if not raw_items:
        reasons.append("items: must not be empty")

    seen: set[str] = set()
    present_families: set[str] = set()
    counts: dict[str, int] = {}
    anchors = 0
    ordered_ids: list[str] = []
    for i, it in enumerate(raw_items):
        if not isinstance(it, Mapping):
            reasons.append(f"items: entry {i} is not an object")
            continue
        item_id = it.get("item_id")
        label = item_id if isinstance(item_id, str) and item_id else f"#{i}"
        if not isinstance(item_id, str) or not item_id:
            reasons.append(f"item[{label}]: item_id must be a non-empty string")
            continue
        if item_id in seen:
            reasons.append(f"item[{label}]: duplicate item_id")
        seen.add(item_id)
        ordered_ids.append(item_id)

        prompt = it.get("prompt_text")
        if not isinstance(prompt, str):
            reasons.append(f"item[{label}]: prompt_text must be a string")
        elif it.get("prompt_sha256") != sha256_hex(_utf8(prompt)):
            reasons.append(f"item[{label}]: prompt_sha256 does not match prompt_text")

        family = it.get("family")
        if family not in FAMILIES:
            reasons.append(f"item[{label}]: family must be one of {list(FAMILIES)}, got {family!r}")
        else:
            present_families.add(family)

        role = it.get("role")
        if role not in _ITEM_ROLES:
            reasons.append(f"item[{label}]: role must be one of {list(_ITEM_ROLES)}, got {role!r}")
        elif canary and role == "item":
            reasons.append(f"item[{label}]: role 'item' is not allowed under canary-v1")
        elif not canary and role != "item":
            reasons.append(f"item[{label}]: role {role!r} is only allowed under canary-v1")

        if canary:
            for field in ("score", "margin", "flip1", "flip2", "flip3"):
                if field not in it:
                    reasons.append(f"item[{label}]: canary-v1 requires {field}")
            flip2 = it.get("flip2")
            if isinstance(flip2, (int, float)) and not isinstance(flip2, bool) and not _is_zero(flip2):
                reasons.append(f"item[{label}]: flip2 must be 0 in the battery (section 4.4 step 1)")
            if role == "anchor":
                anchors += 1
                for field in ("flip1", "flip2", "flip3", "flip4"):
                    v = it.get(field)
                    if v is None:
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and not _is_zero(v):
                        reasons.append(f"item[{label}]: an anchor needs {field} == 0 (Appendix C zero(i))")
            elif role == "canary" and family in FAMILIES:
                counts[family] = counts.get(family, 0) + 1
        else:
            for field in SELECTION_ITEM_FIELDS:
                if field in it:
                    reasons.append(f"item[{label}]: {field} is only allowed under canary-v1")

    if ordered_ids != sorted(ordered_ids, key=_utf8):
        reasons.append("items: must be in A.3 order (item_id ascending, UTF-8 byte order)")

    families = body.get("families")
    if isinstance(families, Mapping) or not isinstance(families, Sequence):
        reasons.append("families: must be a list")
    else:
        bad = [f for f in families if f not in FAMILIES]
        if bad:
            reasons.append(f"families: not in the enum {bad}")
        undeclared = sorted(present_families - set(families))
        if undeclared:
            reasons.append(f"families: items use families the body does not declare {undeclared}")

    if body.get("redacted") not in (True, False):
        reasons.append("body: redacted must be a boolean")

    params = body.get("params")
    excluded = body.get("excluded")
    params_ok = isinstance(params, Mapping)
    if canary:
        if not params_ok:
            reasons.append("params: canary-v1 requires a params object")
        else:
            for field in ("tau", "w_flip", "w_margin", "n", "k_anchors", "k_anchors_actual",
                          "max_family_share", "perm_seed", "tie_break", "sensitivity_after_exclusion"):
                if field not in params:
                    reasons.append(f"params: canary-v1 requires {field}")
            if params.get("k_anchors_actual") != anchors:
                reasons.append(
                    f"anchors: params.k_anchors_actual is {params.get('k_anchors_actual')!r} "
                    f"but the body carries {anchors}"
                )
            k_max = params.get("k_anchors")
            if isinstance(k_max, int) and not isinstance(k_max, bool) and anchors > k_max:
                reasons.append(f"anchors: {anchors} anchors exceed k_anchors {k_max}")

        if isinstance(excluded, Mapping) or not isinstance(excluded, Sequence):
            reasons.append("excluded: canary-v1 requires a list")
        else:
            for j, ex in enumerate(excluded):
                if not isinstance(ex, Mapping) or not isinstance(ex.get("item_id"), str):
                    reasons.append(f"excluded: entry {j} needs an item_id")
                    continue
                f2 = ex.get("flip2")
                if not isinstance(f2, (int, float)) or isinstance(f2, bool) or _is_zero(f2):
                    reasons.append(f"excluded: {ex['item_id']} needs flip2 > 0")
                if ex["item_id"] in seen:
                    reasons.append(f"excluded: {ex['item_id']} is also in items")

        # The share rule (section 4.4 step 3).  The cap is over the selected canaries; the
        # shortfall fill is the one thing allowed to exceed it, and only by what it recorded.
        if isinstance(params, Mapping):
            n_target = params.get("n")
            share = params.get("max_family_share")
            shortfall = params.get("shortfall", 0)
            if (isinstance(n_target, int) and not isinstance(n_target, bool)
                    and isinstance(share, (int, float)) and not isinstance(share, bool)
                    and isinstance(shortfall, int) and not isinstance(shortfall, bool)):
                cap = max(1, int(math.floor(n_target * share)))
                for fam, count in sorted(counts.items()):
                    if count > cap + max(0, shortfall):
                        reasons.append(
                            f"share: family {fam!r} holds {count} of {n_target} canaries, "
                            f"over the cap {cap} with shortfall {shortfall}"
                        )
    else:
        if params is not None:
            reasons.append("params: only canary-v1 carries a params block")
        if excluded is not None:
            reasons.append("excluded: only canary-v1 carries an excluded list")

    return reasons
