"""styxx.v8.sampled_challenge — a challenge that re-runs k of the battery's n items.

Why this module exists
----------------------
``papers/v8/THE_BOUNDARY_2026_09_09.md`` concludes that every check the log performs on bytes
one party wrote can only ask whether that party contradicted itself, and that the challenge --
a SECOND party running the same battery and putting their own bytes in the log -- is the only
mechanism in the system that introduces a byte the issuer did not write.  That mechanism has
never been run.  A challenge that costs a full battery re-run is a challenge nobody performs,
so this module makes being a second party cost ``k/n`` of one.

The comparison is between per-item digests that the issuer's fingerprint cert ALREADY records
(``items[i].output_sha256`` and ``items[i].token_ids_sha256``, both present on all 320 items of
the five published fingerprints under ``papers/v8/first_verdict_2026_09_09/``).  A challenger
re-runs k items and reports, per item, its own two digests.  Nothing is estimated that the
issuer did not already commit to.

The selection must not be choosable or predictable by the issuer, or the issuer fabricates
exactly the items nobody will sample.  Section "Binding" below states exactly whose choice each
mode removes and whose it does not.  Read ``binding_report`` before quoting a challenge's k.

Scope, stated once and enforced by the names in this file
--------------------------------------------------------
A sampled challenge speaks to the ``exact`` channel of ONE fingerprint cert's items: did this
subject, on this battery, under this recipe, emit these tokens for these prompts.

It says NOTHING about:

* **the noise floor.**  A floor is a function of R runs (spec section 5.1); one challenger run
  measures no floor and this module never computes one, never compares a sampled distance to
  ``target_floor``, and never emits ``coverage``.  A reader who wants "is this difference inside
  the machine's own variability" needs the challenger's own multi-run floor, which is a full
  ``prereg noise-plan`` + R runs and is not what this module makes cheap.
* **the ``seqlp``, ``topk``, ``resid`` and ``lens`` channels.**  Those are numeric and their
  Appendix B distances are means over per-item quantities; a sampled version is constructible
  the same way but is not built here, because each carries its own tolerance question and the
  digest comparison carries none.
* **the n - k items not sampled**, beyond the inference in ``detection_hypergeometric``, which
  is a statement about a sampling procedure and not about any particular unsampled item.

What one sampled challenge with zero disagreements establishes is in ``ESTABLISHES``; what it
does not is in ``DOES_NOT_ESTABLISH``.  Both are strings in this module so a caller can print
them beside a verdict rather than paraphrase them.

Decisions
---------
- **Sampling is without replacement** (a keyed ranking of the n item ids, first k taken), so
  the honest power is hypergeometric and strictly above the ``1 - (1-f)^k`` figure that a
  with-replacement design gives.  Both are computed here; the with-replacement number is kept
  because it is the one that bounds the grinding cost of a colluding challenger.
- **The own cert is a PARTIAL fingerprint** carrying k of n items.  It carries ``body.sample``,
  the same object the challenge carries, so that a reader holding it alone cannot mistake it
  for a fingerprint of the battery.  This is a new forgery surface and it is not closed here:
  a k-item cert used as a ``previous``, as a floor ``run``, or as a ``--ref`` baseline would be
  a partial measurement wearing a full measurement's type, and the schema and the log are the
  only places that can refuse it.  See ``LOG_SIDE_OWED``.
- **No signing happens in this module.**  It builds bodies and computes predicates over bytes.
  Minting, appending and signature verification live in ``cert``/``log``/``cli``.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from styxx.v8 import jcs

try:  # CONTRACT FALLBACK: consts is frozen, but keep the module importable alone.
    from styxx.v8.consts import ROUND_PLACES
except ImportError:  # pragma: no cover
    ROUND_PLACES = 9

__all__ = [
    "SAMPLE_TAG",
    "RANK_TAG",
    "MODES",
    "MIN_NONCE_BYTES",
    "ESTABLISHES",
    "DOES_NOT_ESTABLISH",
    "LOG_SIDE_OWED",
    "selector",
    "selector_digest",
    "select_items",
    "target_digests",
    "compare",
    "sample_block",
    "challenge_body",
    "verify_sample",
    "binding_report",
    "detection_with_replacement",
    "detection_hypergeometric",
    "k_for_power",
    "expected_grinding_trials",
    "power_table",
]

# The two domain-separated tags this module introduces.  Style and framing follow
# ``styxx.v8.keys.tagged`` and ``styxx.v8.fingerprint._ORDER_TAG``: an ASCII tag, a NUL, then
# the bytes.  Neither preimage can collide with a cert id ("styxx.v8/cert/1"), a tree head
# ("styxx.v8/sth/1"), a seal, a sweep record or a plan item order.
SAMPLE_TAG = "styxx.v8/challenge/sample/1"
RANK_TAG = "styxx.v8/challenge/rank/1"

_RANK_PREFIX = RANK_TAG.encode("ascii") + b"\x00"

#: ``head`` -- the sample is fixed by a signed tree head that already commits to the target
#: cert, plus the challenger's key and a challenger-chosen nonce.
#: ``commit-then-head`` -- the challenger first logs (target, k, nonce, key); the sample is
#: fixed by the FIRST head whose ``tree_size`` exceeds that commitment's index.
MODES = ("head", "commit-then-head")

MIN_NONCE_BYTES = 16

_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")

ESTABLISHES = (
    "A sampled challenge with k items, zero disagreements and a verified selection establishes "
    "that a second key ran k of the n battery items under the target's recipe and observed the "
    "output digests the target cert records for exactly those k items, where the k were fixed "
    "by a tree head the target cert was already inside. It is evidence against the hypothesis "
    "that a fraction f of the target's items were fabricated, at strength "
    "detection_hypergeometric(n, round(f*n), k) -- 88 percent at n=64, f=0.1, k=20 -- and at no "
    "strength at all against a challenger who colluded, since a colluding challenger can report "
    "digests it never computed."
)

DOES_NOT_ESTABLISH = (
    "It does not establish anything about the target's noise floor, which is a function of R "
    "runs and cannot be measured by one; nor about the seqlp, topk, resid or lens channels, "
    "which this module does not sample; nor about the n-k items not drawn, except through the "
    "stated sampling inference; nor that the target's subject is what its cert names, which is "
    "the runner-reported-identity check and is a separate predicate; nor that a disagreement is "
    "drift rather than the machine's own variability, which is exactly the question a floor "
    "answers and this challenge does not ask."
)

LOG_SIDE_OWED = (
    "A partial fingerprint (body.items shorter than the battery) must be refused as a floor "
    "run, as a `previous`, and as a `--ref` baseline; the log must check that a "
    "`commit-then-head` sample used the FIRST head above its commitment index, and must check "
    "that the head named by the sample has tree_size greater than the target's own leaf index. "
    "None of those is implemented in this module and none can be: they are predicates over the "
    "log's own entry order, and this module holds no log."
)


# ------------------------------------------------------------------ argument checks

def _mapping(name: str, value: Any) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    return value


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return value


def _ids(name: str, value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be a sequence of item ids, got {type(value).__name__}")
    out = [_str(f"{name}[{i}]", v) for i, v in enumerate(value)]
    if len(set(out)) != len(out):
        raise ValueError(f"{name} repeats an item id")
    return out


# ------------------------------------------------------------------ selection

def selector(
    *,
    log_id: str,
    tree_size: int,
    root_hash: str,
    target: str,
    challenger_key: str,
    nonce: str,
    k: int,
    n: int,
    mode: str = "head",
    commit_index: int | None = None,
) -> dict:
    """The object whose canonical bytes fix the sample.  Every member is a commitment.

    ``log_id``, ``tree_size`` and ``root_hash`` come from a signed tree head; ``target`` is the
    challenged fingerprint's cert id, which that head must already commit to (the caller checks
    the leaf index against ``tree_size`` -- this module holds no log).  ``challenger_key`` is
    the challenger's issuer key in wire form, ``nonce`` is lowercase hex of at least
    ``MIN_NONCE_BYTES`` bytes, and ``k``/``n`` are the sample and battery sizes.

    Neither party can choose the sample alone: the issuer signed the target before the root
    existed, and (in ``commit-then-head``) the challenger committed key and nonce before the
    deciding root existed.  ``binding_report`` states what each mode leaves open.
    """
    mode = _str("mode", mode)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    n = _int("n", n)
    k = _int("k", k)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n ({k} not in [1, {n}])")
    size = _int("tree_size", tree_size)
    if size < 1:
        raise ValueError("tree_size must be at least 1")
    nonce = _str("nonce", nonce).lower()
    if len(nonce) % 2 or not _HEX_RE.match(nonce):
        raise ValueError("nonce must be an even-length lowercase hex string")
    if len(nonce) < 2 * MIN_NONCE_BYTES:
        raise ValueError(f"nonce must be at least {MIN_NONCE_BYTES} bytes ({2 * MIN_NONCE_BYTES} hex chars)")
    for name, value in (("log_id", log_id), ("root_hash", root_hash), ("target", target)):
        if not _ID_RE.match(_str(name, value)):
            raise ValueError(f"{name} must look like sha256:<64 hex>, got {value!r}")
    challenger_key = _str("challenger_key", challenger_key)
    if not challenger_key:
        raise ValueError("challenger_key must not be empty")

    out = {
        "challenger_key": challenger_key,
        "k": k,
        "log_id": log_id,
        "method": SAMPLE_TAG,
        "mode": mode,
        "n": n,
        "nonce": nonce,
        "root_hash": root_hash,
        "target": target,
        "tree_size": size,
    }
    if mode == "commit-then-head":
        idx = _int("commit_index", commit_index)
        if idx < 0:
            raise ValueError("commit_index must be non-negative")
        if size <= idx:
            raise ValueError(
                f"the deciding head must come after the commitment: tree_size {size} <= commit_index {idx}"
            )
        out["commit_index"] = idx
    elif commit_index is not None:
        raise ValueError("commit_index belongs only to mode 'commit-then-head'")
    return out


def selector_digest(sel: Mapping) -> str:
    """``sha256:`` + the SHA-256 of the selector's JCS bytes -- the sample's public name."""
    return "sha256:" + jcs.digest(dict(_mapping("selector", sel)))


def _seed(sel: Mapping) -> bytes:
    """The ranking seed: the selector's JCS bytes WITHOUT ``k``.

    Dropping ``k`` makes the samples nested -- k and k+1 share a prefix -- and that is a
    security property, not a convenience.  With ``k`` inside the seed, each k gives an
    independent permutation, so a colluding challenger holding the fabricated list computes all
    n rankings and publishes whichever k's prefix is clean: 64 free draws at n = 64, on top of
    nonce grinding.  With ``k`` outside, there is ONE ranking per (head, key, nonce, target) and
    the challenger's only remaining freedom over it is the prefix length, which is ``k`` itself
    and is the number a reader already discounts by (``detection_hypergeometric``).  A test in
    ``tests/test_v8_sampled_challenge.py`` pins the nesting; this comment is why it is there.

    ``k`` is still inside the selector and therefore inside ``selector_id``, so the claimed
    sample size remains a commitment -- it just does not move the draw.
    """
    return hashlib.sha256(jcs.canonical_bytes({k: v for k, v in sel.items() if k != "k"})).digest()


def select_items(item_ids: Sequence[str], sel: Mapping) -> list[str]:
    """The k item ids the selector draws, in the order the ranking gives.

    Sampling is WITHOUT replacement: the n ids are ranked by
    ``sha256(RANK_TAG || 0x00 || seed || 0x00 || item_id)`` and the first k are taken, ties (a
    SHA-256 collision) broken by the id itself so the function is total.  Re-derivable by
    anyone holding the battery's item ids and the selector, which is what makes the sample a
    checkable receipt rather than a list only the challenger could have produced.  The seed
    excludes ``k`` (see ``_seed``), so the draw is one ranking and k is a prefix length.
    """
    sel = _mapping("selector", sel)
    ids = _ids("item_ids", item_ids)
    n, k = _int("selector.n", sel.get("n")), _int("selector.k", sel.get("k"))
    if len(ids) != n:
        raise ValueError(f"selector.n is {n} but {len(ids)} item ids were given")
    if not 1 <= k <= n:
        raise ValueError(f"selector.k must satisfy 1 <= k <= n ({k} not in [1, {n}])")
    seed = _seed(sel)
    ranked = sorted(
        ids,
        key=lambda i: (hashlib.sha256(_RANK_PREFIX + seed + b"\x00" + i.encode("utf-8")).digest(), i),
    )
    return ranked[:k]


# ------------------------------------------------------------------ the comparison

def target_digests(target_cert: Mapping) -> dict[str, dict[str, Any]]:
    """``{item_id: {output_sha256, token_ids_sha256, n_generated}}`` from a fingerprint cert.

    Raises ValueError when an item carries neither digest -- a cert that records no per-item
    digest cannot be sampled and saying so is better than sampling nothing.
    """
    cert = _mapping("target_cert", target_cert)
    body = _mapping("target_cert.body", cert.get("body", {}))
    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("target cert carries no body.items (redacted certs are not sampleable here)")
    out: dict[str, dict[str, Any]] = {}
    for pos, item in enumerate(items):
        it = _mapping(f"body.items[{pos}]", item)
        iid = _str(f"body.items[{pos}].item_id", it.get("item_id"))
        if iid in out:
            raise ValueError(f"target cert repeats item_id {iid}")
        o, t = it.get("output_sha256"), it.get("token_ids_sha256")
        if not isinstance(o, str) or not isinstance(t, str):
            raise ValueError(f"body.items[{pos}] ({iid}) carries no output_sha256/token_ids_sha256")
        out[iid] = {"output_sha256": o, "token_ids_sha256": t, "n_generated": it.get("n_generated")}
    return out


def compare(sampled: Sequence[str], target_cert: Mapping, own_items: Mapping[str, Mapping]) -> list[dict]:
    """One result row per sampled item, in sample order.

    ``own_items`` maps item_id to the challenger's own ``{output_sha256, token_ids_sha256}``.
    A row agrees only when BOTH digests match: the text digest alone would let a tokenizer
    difference pass as agreement, and the token digest alone would let a decoding difference.
    """
    tgt = target_digests(target_cert)
    own = _mapping("own_items", own_items)
    rows: list[dict] = []
    for iid in _ids("sampled", sampled):
        if iid not in tgt:
            raise ValueError(f"sampled item {iid} is not in the target cert")
        if iid not in own:
            raise ValueError(f"sampled item {iid} has no own result")
        mine = _mapping(f"own_items[{iid}]", own[iid])
        o_t, t_t = tgt[iid]["output_sha256"], tgt[iid]["token_ids_sha256"]
        o_o = _str(f"own_items[{iid}].output_sha256", mine.get("output_sha256"))
        t_o = _str(f"own_items[{iid}].token_ids_sha256", mine.get("token_ids_sha256"))
        rows.append(
            {
                "agree": o_t == o_o and t_t == t_o,
                "item_id": iid,
                "own_output_sha256": o_o,
                "own_token_ids_sha256": t_o,
                "target_output_sha256": o_t,
                "target_token_ids_sha256": t_t,
            }
        )
    return rows


def sample_block(sel: Mapping, rows: Sequence[Mapping]) -> dict:
    """``body.sample`` -- the selector, the drawn ids, the rows, and the two derived counts."""
    sel = dict(_mapping("selector", sel))
    rows = [dict(_mapping(f"rows[{i}]", r)) for i, r in enumerate(rows)]
    ids = [_str(f"rows[{i}].item_id", r.get("item_id")) for i, r in enumerate(rows)]
    if len(ids) != sel.get("k"):
        raise ValueError(f"selector.k is {sel.get('k')} but {len(ids)} rows were given")
    n_dis = sum(1 for r in rows if r.get("agree") is not True)
    return {
        "disagreements": n_dis,
        "item_ids": ids,
        "results": rows,
        "sampled_exact_distance": round(n_dis / len(ids), ROUND_PLACES),
        "selector": sel,
        "selector_id": selector_digest(sel),
    }


def challenge_body(
    sample: Mapping,
    *,
    environment: Mapping,
    subject: Mapping,
    recipe_core: Mapping,
    synthetic: bool,
    note: str | None = None,
) -> dict:
    """A section 9 challenge body carrying ``sample`` instead of ``per_channel``/``coverage``.

    ``per_channel`` and ``coverage`` are deliberately ABSENT.  Both are floor-relative -- a
    ``target_floor`` per channel and a within/beyond classification against the floor's
    ``covers`` -- and a one-run k-item reproduction has measured no floor.  Emitting them with
    the target's own floor copied in would let a reader read a floor comparison out of bytes
    that contain none, which is the shape of defect the boundary paper's round 4 describes.
    The scope is instead carried explicitly in ``sample_of``.
    """
    body = {
        "environment": dict(_mapping("environment", environment)),
        "recipe_core": dict(_mapping("recipe_core", recipe_core)),
        "sample": dict(_mapping("sample", sample)),
        "sample_of": "exact",
        "subject": dict(_mapping("subject", subject)),
        "synthetic": bool(synthetic),
    }
    if note is not None:
        body["note"] = _str("note", note)
    return body


# ------------------------------------------------------------------ the predicate

def verify_sample(
    challenge: Mapping,
    target_cert: Mapping,
    *,
    sth: Mapping,
    battery_item_ids: Sequence[str],
    target_index: int | None = None,
) -> list[str]:
    """Why this is not a valid sampled challenge to ``target_cert``.  Empty means it is one.

    Computed from bytes the caller holds: the challenge cert, the target cert, a signed tree
    head, the battery's item ids, and (when known) the target's leaf index.  Signature checking
    is NOT done here -- ``cert.check`` and ``log`` own that, and a caller that skips them holds
    an unsigned claim whatever this function returns.

    The eleven conditions, in the order they are reported:

    1. the cert is a ``challenge`` and its body carries ``sample`` with ``method`` = SAMPLE_TAG;
    2. the sample's head fields are the given head's, member for member;
    3. the head commits to the target: ``target_index < tree_size`` (skipped, and reported as
       skipped, when the caller passes no index);
    4. the sample names the target the cert's ``target`` ref names;
    5. the selector's ``challenger_key`` is the cert's own issuer key;
    6. ``n`` is the battery's item count and the target cert covers exactly that battery;
    7. the drawn ids ARE ``select_items(battery_item_ids, selector)``, in order;
    8. every row quotes the target's own recorded digests for its item;
    9. every row's ``agree`` follows from its four digests;
    10. ``disagreements`` and ``sampled_exact_distance`` follow from the rows;
    11. ``selector_id`` is the selector's digest, and a synthetic challenge does not challenge a
        measured target.
    """
    ch = _mapping("challenge", challenge)
    reasons: list[str] = []

    if ch.get("type") != "challenge":
        reasons.append(f"cert type is {ch.get('type')!r}, not 'challenge'")
    body = ch.get("body")
    if not isinstance(body, Mapping):
        return reasons + ["challenge carries no body"]
    sample = body.get("sample")
    if not isinstance(sample, Mapping):
        return reasons + ["body.sample is absent; this is not a sampled challenge"]
    sel = sample.get("selector")
    if not isinstance(sel, Mapping):
        return reasons + ["body.sample.selector is absent"]
    if sel.get("method") != SAMPLE_TAG:
        reasons.append(f"selector.method is {sel.get('method')!r}, not {SAMPLE_TAG!r}")
    if sel.get("mode") not in MODES:
        reasons.append(f"selector.mode is {sel.get('mode')!r}, not one of {MODES}")

    head = _mapping("sth", sth)
    for key in ("log_id", "tree_size", "root_hash"):
        if sel.get(key) != head.get(key):
            reasons.append(f"selector.{key} {sel.get(key)!r} is not the head's {head.get(key)!r}")

    size = head.get("tree_size")
    if target_index is None:
        reasons.append(
            "UNCHECKED: no target leaf index was given, so 'the head already commits to the "
            "target' was not verified; without it the issuer may have minted the target after "
            "seeing the root"
        )
    elif not isinstance(size, int) or isinstance(target_index, bool) or not isinstance(target_index, int):
        reasons.append("target_index and the head's tree_size must both be ints")
    elif not 0 <= target_index < size:
        reasons.append(
            f"the head does not commit to the target: leaf index {target_index} not in [0, {size})"
        )

    tid = target_cert.get("id") if isinstance(target_cert, Mapping) else None
    if sel.get("target") != tid:
        reasons.append(f"selector.target {sel.get('target')!r} is not the target cert id {tid!r}")
    refs = ch.get("refs")
    named = [r.get("id") for r in refs if isinstance(r, Mapping) and r.get("role") == "target"] if isinstance(refs, list) else []
    if named != [tid]:
        reasons.append(f"the challenge's target refs are {named!r}, not exactly [{tid!r}]")

    issuer = ch.get("issuer")
    own_key = issuer.get("key") if isinstance(issuer, Mapping) else None
    if sel.get("challenger_key") != own_key:
        reasons.append(
            f"selector.challenger_key {sel.get('challenger_key')!r} is not the cert's issuer key {own_key!r}"
        )

    try:
        ids = _ids("battery_item_ids", battery_item_ids)
    except (TypeError, ValueError) as e:
        return reasons + [f"battery_item_ids is unusable: {e}"]
    if sel.get("n") != len(ids):
        reasons.append(f"selector.n is {sel.get('n')!r} but the battery has {len(ids)} items")
    try:
        tgt = target_digests(target_cert)
    except (TypeError, ValueError) as e:
        return reasons + [f"target cert is not sampleable: {e}"]
    if set(tgt) != set(ids):
        missing, extra = sorted(set(ids) - set(tgt))[:3], sorted(set(tgt) - set(ids))[:3]
        reasons.append(f"the target cert's items are not the battery's (missing {missing}, extra {extra})")

    drawn = sample.get("item_ids")
    try:
        derived = select_items(ids, sel)
    except (TypeError, ValueError) as e:
        return reasons + [f"the selector does not derive a sample: {e}"]
    if drawn != derived:
        reasons.append(
            f"the drawn ids are not the derived sample (derived {derived[:4]}..., cert says "
            f"{drawn[:4] if isinstance(drawn, list) else drawn!r}...)"
        )

    rows = sample.get("results")
    if not isinstance(rows, list):
        return reasons + ["body.sample.results is not a list"]
    if [r.get("item_id") if isinstance(r, Mapping) else None for r in rows] != derived:
        reasons.append("body.sample.results does not carry one row per drawn item, in sample order")
    else:
        n_dis = 0
        for pos, row in enumerate(rows):
            iid = row["item_id"]
            rec = tgt.get(iid, {})
            for key, field in (("target_output_sha256", "output_sha256"), ("target_token_ids_sha256", "token_ids_sha256")):
                if row.get(key) != rec.get(field):
                    reasons.append(
                        f"results[{pos}] ({iid}) misquotes the target's {field}: {row.get(key)!r} "
                        f"is not {rec.get(field)!r}"
                    )
            agrees = (
                row.get("own_output_sha256") == row.get("target_output_sha256")
                and row.get("own_token_ids_sha256") == row.get("target_token_ids_sha256")
            )
            if row.get("agree") is not agrees:
                reasons.append(f"results[{pos}] ({iid}) claims agree={row.get('agree')!r} but the digests say {agrees}")
            if not agrees:
                n_dis += 1
        if sample.get("disagreements") != n_dis:
            reasons.append(f"body.sample.disagreements is {sample.get('disagreements')!r}, the rows give {n_dis}")
        want = round(n_dis / len(rows), ROUND_PLACES) if rows else None
        if sample.get("sampled_exact_distance") != want:
            reasons.append(
                f"body.sample.sampled_exact_distance is {sample.get('sampled_exact_distance')!r}, the rows give {want}"
            )

    try:
        want_id = selector_digest(sel)
    except (TypeError, ValueError) as e:  # pragma: no cover - selector already validated above
        want_id, _ = None, reasons.append(f"selector is not serializable: {e}")
    if want_id is not None and sample.get("selector_id") != want_id:
        reasons.append(f"body.sample.selector_id is {sample.get('selector_id')!r}, the selector digests to {want_id}")

    t_body = target_cert.get("body") if isinstance(target_cert, Mapping) else None
    t_syn = bool(t_body.get("synthetic")) if isinstance(t_body, Mapping) else False
    if bool(body.get("synthetic")) != t_syn:
        reasons.append(
            f"synthetic mismatch: the challenge is synthetic={bool(body.get('synthetic'))} and the "
            f"target is synthetic={t_syn}; a run holding no model neither challenges nor is challenged "
            "by one that does"
        )
    if body.get("sample_of") != "exact":
        reasons.append(f"body.sample_of is {body.get('sample_of')!r}; this module samples the 'exact' channel only")
    return reasons


def binding_report(sel: Mapping) -> dict:
    """Whose choice this selector removes, whose it does not, and what it costs to grind.

    The honest reading of the two modes, and the reason ``head`` is not enough on its own:

    * **``head``** removes the ISSUER's choice.  The target cert is inside the tree at
      ``tree_size``, so its bytes were fixed before ``root_hash`` existed and the issuer could
      not have aimed its fabrication at a known sample.  It does NOT remove the CHALLENGER's:
      ``nonce`` is free and ``challenger_key`` is free (an ed25519 key costs nothing to make),
      so a challenger colluding with the issuer redraws until the sample misses every
      fabricated item.  ``expected_trials`` is that cost, and at f=0.1, k=20 it is about 8
      draws -- which is not a defence.
    * **``commit-then-head``** removes both, to the extent the log operator is a third party:
      key, nonce, target and k are logged at ``commit_index``, and the deciding root is the
      first head above it, which the challenger did not know when committing.  Two residues
      remain and neither is closed by any predicate over these bytes: the log operator can
      choose what to append and when, and therefore what the next root is; and a challenger who
      dislikes the sample can simply never publish the challenge, so unanswered commitments are
      the disclosure that matters and a challenger's commitment-to-challenge ratio is the
      number a reader should want.

    Both modes leave the largest residue untouched: **a colluding challenger can report digests
    it never computed.** Sampling changes the cost of a challenge, not the trust model. The
    quantity that survives collusion is the count of DISTINCT keys, in a reader's own trust
    file, that have challenged a cert -- the same threshold section 9's Disputed rule already
    carries, and it is a client-side judgment, not a log predicate.
    """
    sel = _mapping("selector", sel)
    mode = sel.get("mode")
    common = [
        "a colluding challenger can report digests it never computed; nothing over these bytes reaches that",
        "a challenger who dislikes the drawn sample can decline to publish, so abandoned samples are invisible unless commitments are logged",
    ]
    if mode == "commit-then-head":
        return {
            "mode": mode,
            "removes_issuer_choice": True,
            "removes_challenger_choice": True,
            "residue": common + [
                "the log operator chooses what is appended and when, and therefore what the deciding root is",
                "'first head above the commitment index' is a predicate over log order and is not checked here",
            ],
        }
    return {
        "mode": mode,
        "removes_issuer_choice": True,
        "removes_challenger_choice": False,
        "residue": common + [
            "nonce and challenger_key are both free, so a colluding challenger regrinds the sample until it misses every fabricated item",
        ],
    }


# ------------------------------------------------------------------ detection power

def detection_with_replacement(f: float, k: int) -> float:
    """``1 - (1-f)^k`` -- k independent draws against a fabricated fraction f.

    This is the WEAKER of the two numbers and is kept for one reason: it is the probability a
    single grinding attempt fails, so ``1/(1-f)^k`` is what a colluding challenger pays.
    """
    f = float(f)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"f must be in [0, 1], got {f}")
    k = _int("k", k)
    if k < 0:
        raise ValueError("k must be non-negative")
    return 1.0 - (1.0 - f) ** k


def detection_hypergeometric(n: int, m: int, k: int) -> float:
    """``1 - C(n-m, k)/C(n, k)`` -- k of n drawn without replacement, m of them fabricated.

    This is what ``select_items`` actually does.  Exact rational arithmetic, then one float
    division, so no product of small floats is formed.
    """
    n, m, k = _int("n", n), _int("m", m), _int("k", k)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= m <= n:
        raise ValueError(f"m must satisfy 0 <= m <= n ({m} not in [0, {n}])")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n ({k} not in [0, {n}])")
    if m == 0 or k == 0:
        return 0.0
    if k > n - m:
        return 1.0
    miss = math.comb(n - m, k)
    total = math.comb(n, k)
    return 1.0 - miss / total


def k_for_power(n: int, m: int, power: float = 0.95) -> int | None:
    """The smallest k with ``detection_hypergeometric(n, m, k) >= power``; None if none exists."""
    n, m = _int("n", n), _int("m", m)
    power = float(power)
    if not 0.0 <= power <= 1.0:
        raise ValueError(f"power must be in [0, 1], got {power}")
    for k in range(0, n + 1):
        if detection_hypergeometric(n, m, k) >= power:
            return k
    return None


def expected_grinding_trials(f: float, k: int) -> float:
    """``1/(1-f)^k`` -- expected redraws before a colluding challenger's sample misses every
    fabricated item.  ``inf`` when f == 1.  This is the number that makes mode ``head``
    insufficient on its own."""
    f = float(f)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"f must be in [0, 1], got {f}")
    k = _int("k", k)
    if k < 0:
        raise ValueError("k must be non-negative")
    p_miss = (1.0 - f) ** k
    return math.inf if p_miss == 0.0 else 1.0 / p_miss


def power_table(
    n: int = 64,
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    fs: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5),
    power: float = 0.95,
) -> dict:
    """Detection power for every (k, f), plus the k each f needs for ``power``.

    ``m = round(f*n)`` is the integer count of fabricated items, and ``f_effective = m/n`` is
    reported beside the nominal f because at n = 64 they differ: f = 0.01 asks for 0.64 items
    and gets 1.
    """
    n = _int("n", n)
    ks = [_int("k", k) for k in ks]
    fs = [float(f) for f in fs]
    rows = []
    for f in fs:
        m = int(round(f * n))
        need = k_for_power(n, m, power)
        rows.append(
            {
                "f": f,
                "f_effective": m / n,
                "hyper": {k: detection_hypergeometric(n, m, k) for k in ks},
                "k_for_power": need,
                "m": m,
                "with_replacement": {k: detection_with_replacement(f, k) for k in ks},
            }
        )
    return {"ks": ks, "n": n, "power": power, "rows": rows}
