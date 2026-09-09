"""Appendix B distance functions for styxx.v8 (spec v0.2, one per channel).

Every mean here is over items in A.3 order (``item_id`` ascending, byte order of the UTF-8
string) with exact summation (``math.fsum``).  ``verify`` and the noise-floor procedure call
exactly these functions, so a floor and a distance are always in the same units.

Item alignment.  ``a_items`` / ``b_items`` are the fingerprint ``body.items`` records of §3.1
(dicts carrying ``item_id`` and, per channel, ``token_ids`` or ``token_ids_sha256``,
``seq_logprob``, ``topk``).  A mapping ``{item_id: record}`` is accepted too.  Both sides must
carry the same set of ``item_id`` values; an item missing on either side raises ``ValueError``
(a channel-level absence -- ``seq_logprob`` / ``topk`` being ``None`` -- is the caller's
"channel absent" case and raises ``ValueError`` here as well, because a distance over a
channel that is not there is not a number).

Decisions this module makes where Appendix B leaves room (recorded so a second implementation
can match them):

* ``topk`` is one pooled mean over every (item, position) pair, not a mean of per-item means:
  the sum of all per-position L1 terms, in ``item_id`` then ``pos`` order, divided by the
  number of positions counted.  A position present on one side only contributes ``5 * 20``
  and is counted in the denominator.
* ``resid`` clamps each per-item JSD at 0 after the fact: the quantity is non-negative
  mathematically, and the guard only removes a last-bit negative that finite arithmetic can
  produce for nearly equal profiles.  A profile that is all zeros after clamping cannot be
  L1-normalized and raises ``ValueError``.
* ``exact`` with no item of role ``item``/``canary`` raises ``ValueError`` (the ratio is
  undefined); ``anchor`` items are counted only under ``anchor_flips``.

Decisions (hostile-review repairs, 2026-09-08; each refuses rather than guesses):

* ``exact`` compares items by the A.3 digest ``sha256(UTF-8(JCS(ids)))``.  When a record
  carries ``token_ids`` the digest is computed from them here (``styxx.v8.jcs.digest``); when
  it carries only ``token_ids_sha256`` that value is used.  So one side with ids and the other
  with only a digest compare fine -- the spec defines the digest, computing it is licensed.
  A record carrying both must agree with itself: a ``token_ids_sha256`` that is not the digest
  of its own ``token_ids`` raises ``ValueError`` (a malformed record yields no number).
* ``token_ids_sha256`` must be 64 lowercase hex (A.1), optionally with the ``sha256:`` prefix
  (the §3.1 sketch shows it bare, A.1 writes hashes prefixed; both name the same digest and are
  compared on the hex).  Anything else -- uppercase, wrong length, another algorithm -- raises
  ``ValueError`` instead of silently counting as a mismatch.
* Token ids (``token_ids`` and ``topk.ids``) must be non-negative ints: a vocabulary index.
* ``exact`` returns the literal ``1 - matching / scored`` of Appendix B, not
  ``(scored - matching) / scored``: for 1 of 3 matching the two differ in the last bit
  (0.6666666666666667 vs 0.6666666666666666).  Both round to the same 9 places; a second
  implementation must use the literal form to reproduce the unrounded conformance numbers.
* ``topk`` positions must lie in ``0..TOPK_MAX_POS`` (§3.2: positions ``0..min(7, n-1)``) and a
  position's vector has at most ``TOPK_K`` tokens; wider or later entries are not the channel
  and raise ``ValueError``.  A log-prob above 0 is not refused (no formula depends on it).
* ``lens`` ``converge_layer`` values must be ints in ``0..n_layers`` inclusive; a fractional,
  negative or out-of-range layer is not a layer index and raises ``ValueError`` (it would also
  let the distance exceed 1).
* ``resid`` uses ``p * log(2p / (p + q))`` for each KL term.  It is the same rational value as
  ``p * log(p / m)`` with ``m = (p + q) / 2`` (both are one correctly-rounded division of
  exactly represented operands, so the double is identical), but it does not divide by zero
  when the halving underflows for a subnormal ``p`` with ``q = 0`` (found by hypothesis).
* ``resid`` / ``lens`` refuse a list mixing ``{item_id, ...}`` records with bare values, and
  refuse a bare-positional side against a keyed side; one form per call, the same on both.
* ``rounded`` refuses a non-number, a bool or a non-finite value (``ValueError``) instead of
  returning ``nan``/``inf`` -- a rounded distance that is not a number is not a distance.
* An int too large for a double (``10**400``) raises ``ValueError('not finite')``, not
  ``OverflowError``.
* ``roles`` keys/ids must be ``str``; an int key is not coerced.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from styxx.v8.jcs import digest as _jcs_digest

try:  # consts.py is written by the cert agent; the value is pinned by the contract.
    from styxx.v8.consts import ROUND_PLACES
except ImportError:  # pragma: no cover - consts.py not present yet
    ROUND_PLACES = 9

__all__ = ["exact", "seqlp", "topk", "resid", "lens", "rounded", "ROUND_PLACES"]

ABSENT_LP = -20.0            # log-prob assigned to a token absent from one side's top-k
TOPK_K = 5                   # top-k width the channel is defined over
TOPK_MAX_POS = 7             # §3.2: positions 0..min(7, n_generated-1)
ONE_SIDED_POSITION = float(TOPK_K) * 20.0   # 100.0: the maximum per-position distance

SCORED_ROLES = frozenset({"item", "canary"})
ANCHOR_ROLE = "anchor"

_DIGEST_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


# --------------------------------------------------------------------------- helpers

def _id_key(item_id: str) -> bytes:
    """A.3 order: byte order of the UTF-8 string."""
    return item_id.encode("utf-8")


def _by_id(items: Any, side: str) -> dict[str, Any]:
    """Index records by ``item_id``; refuse duplicates and records without an id."""
    if isinstance(items, Mapping):
        out: dict[str, Any] = {}
        for k, v in items.items():
            if not isinstance(k, str):
                raise ValueError(f"{side}: item_id must be a str, got {type(k).__name__}")
            out[k] = v
        return out
    out = {}
    for rec in items:
        if not isinstance(rec, Mapping) or "item_id" not in rec:
            raise ValueError(f"{side}: every item needs an item_id")
        iid = rec["item_id"]
        if not isinstance(iid, str):
            raise ValueError(f"{side}: item_id must be a str, got {type(iid).__name__}")
        if iid in out:
            raise ValueError(f"{side}: duplicate item_id {iid!r}")
        out[iid] = rec
    return out


def _aligned(a_items: Any, b_items: Any) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Both sides indexed by id, plus the shared ids in A.3 order.  Any id on one side only
    is a missing item and raises."""
    a = _by_id(a_items, "a")
    b = _by_id(b_items, "b")
    only_a = sorted(set(a) - set(b), key=_id_key)
    only_b = sorted(set(b) - set(a), key=_id_key)
    if only_a or only_b:
        raise ValueError(
            f"items not aligned: missing on b {only_a}, missing on a {only_b}"
        )
    return sorted(a, key=_id_key), a, b


def _field(rec: Any, name: str, side: str, iid: str) -> Any:
    if isinstance(rec, Mapping):
        if name not in rec:
            raise ValueError(f"{side}[{iid!r}]: no {name}")
        return rec[name]
    raise ValueError(f"{side}[{iid!r}]: item record must be a mapping")


def _finite(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{where}: not a number ({type(x).__name__})")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{where}: not finite")
    return x


def _mean(terms: Sequence[float], what: str) -> float:
    if not terms:
        raise ValueError(f"{what}: mean over zero items is undefined")
    return math.fsum(terms) / len(terms)


def rounded(x: float) -> float:
    """§5.2: distances are compared after rounding both sides to ROUND_PLACES decimals."""
    return round(float(x), ROUND_PLACES)


# --------------------------------------------------------------------------- exact

def _roles_map(roles: Any) -> dict[str, str]:
    """Accept ``{item_id: role}`` or the battery's ``items`` list (``{item_id, role, ...}``)."""
    if isinstance(roles, Mapping):
        out = {str(k): v for k, v in roles.items()}
    else:
        out = {}
        for rec in roles:
            if not isinstance(rec, Mapping) or "item_id" not in rec or "role" not in rec:
                raise ValueError("roles: every entry needs item_id and role")
            if rec["item_id"] in out:
                raise ValueError(f"roles: duplicate item_id {rec['item_id']!r}")
            out[rec["item_id"]] = rec["role"]
    for iid, role in out.items():
        if role not in SCORED_ROLES and role != ANCHOR_ROLE:
            raise ValueError(f"roles[{iid!r}]: unknown role {role!r}")
    return out


def _ids_key(rec: Any, side: str, iid: str) -> Any:
    """The comparison key for the exact channel: the token id list when present, else the
    A.3 digest ``token_ids_sha256`` (which is sha256(JCS(ids)) and so equal iff ids are)."""
    if not isinstance(rec, Mapping):
        raise ValueError(f"{side}[{iid!r}]: item record must be a mapping")
    if "token_ids" in rec and rec["token_ids"] is not None:
        ids = rec["token_ids"]
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise ValueError(f"{side}[{iid!r}]: token_ids must be a list")
        for t in ids:
            if isinstance(t, bool) or not isinstance(t, int):
                raise ValueError(f"{side}[{iid!r}]: token_ids must be integers")
        return ("ids", tuple(int(t) for t in ids))
    if "token_ids_sha256" in rec and rec["token_ids_sha256"] is not None:
        return ("sha", str(rec["token_ids_sha256"]))
    raise ValueError(f"{side}[{iid!r}]: neither token_ids nor token_ids_sha256")


def exact(a_items: Any, b_items: Any, roles: Any) -> tuple[float, int]:
    """Appendix B ``exact``: ``1 - matching/items`` over items of role ``item`` or ``canary``;
    anchor mismatches are returned separately as ``anchor_flips``.

    ``roles`` maps ``item_id -> role`` (a dict, or the battery cert's ``items`` list).  Every
    id in ``roles`` must be on both sides and every id on either side must have a role.
    """
    ids, a, b = _aligned(a_items, b_items)
    role_of = _roles_map(roles)
    missing_role = [i for i in ids if i not in role_of]
    missing_item = sorted((i for i in role_of if i not in a), key=_id_key)
    if missing_role:
        raise ValueError(f"items without a role: {missing_role}")
    if missing_item:
        raise ValueError(f"battery items missing from both sides: {missing_item}")
    scored = 0
    matching = 0
    anchor_flips = 0
    for iid in ids:
        ka = _ids_key(a[iid], "a", iid)
        kb = _ids_key(b[iid], "b", iid)
        if ka[0] != kb[0]:
            raise ValueError(
                f"[{iid!r}]: one side carries token_ids and the other only a digest; "
                "compare like with like"
            )
        same = ka == kb
        if role_of[iid] == ANCHOR_ROLE:
            if not same:
                anchor_flips += 1
        else:
            scored += 1
            if same:
                matching += 1
    if scored == 0:
        raise ValueError("exact: no item of role item|canary; the ratio is undefined")
    return 1.0 - matching / scored, anchor_flips


# --------------------------------------------------------------------------- seqlp

def seqlp(a_items: Any, b_items: Any) -> float:
    """Appendix B ``seqlp``: mean over items of ``|delta seq_logprob|`` (fsum, A.3 order)."""
    ids, a, b = _aligned(a_items, b_items)
    terms: list[float] = []
    for iid in ids:
        la = _field(a[iid], "seq_logprob", "a", iid)
        lb = _field(b[iid], "seq_logprob", "b", iid)
        if la is None or lb is None:
            raise ValueError(f"[{iid!r}]: seq_logprob absent (channel absent, not zero)")
        terms.append(abs(_finite(la, f"a[{iid!r}].seq_logprob")
                         - _finite(lb, f"b[{iid!r}].seq_logprob")))
    return _mean(terms, "seqlp")


# --------------------------------------------------------------------------- topk

def _positions(rec: Any, side: str, iid: str) -> dict[int, dict[int, float]]:
    """``topk`` list of ``{pos, ids, lps}`` -> ``{pos: {token_id: lp}}``."""
    tk = _field(rec, "topk", side, iid)
    if tk is None:
        raise ValueError(f"[{iid!r}]: topk absent (channel absent, not zero)")
    out: dict[int, dict[int, float]] = {}
    for entry in tk:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{side}[{iid!r}].topk: entries must be mappings")
        for k in ("pos", "ids", "lps"):
            if k not in entry:
                raise ValueError(f"{side}[{iid!r}].topk: entry without {k}")
        pos = entry["pos"]
        if isinstance(pos, bool) or not isinstance(pos, int) or pos < 0:
            raise ValueError(f"{side}[{iid!r}].topk: pos must be a non-negative int")
        if pos in out:
            raise ValueError(f"{side}[{iid!r}].topk: duplicate pos {pos}")
        tids, lps = entry["ids"], entry["lps"]
        if len(tids) != len(lps):
            raise ValueError(f"{side}[{iid!r}].topk[pos={pos}]: ids and lps lengths differ")
        vec: dict[int, float] = {}
        for t, lp in zip(tids, lps):
            if isinstance(t, bool) or not isinstance(t, int):
                raise ValueError(f"{side}[{iid!r}].topk[pos={pos}]: token ids must be ints")
            if t in vec:
                raise ValueError(f"{side}[{iid!r}].topk[pos={pos}]: duplicate token id {t}")
            vec[t] = _finite(lp, f"{side}[{iid!r}].topk[pos={pos}]")
        out[pos] = vec
    return out


def topk(a_items: Any, b_items: Any) -> float:
    """Appendix B ``topk``: union-of-vocab L1 per position with ``-20`` for a token absent
    from one side; a position present on one side only contributes ``5 * 20 = 100`` and is
    counted.  One pooled mean over every (item, position) pair, items in A.3 order and
    positions ascending within an item, exact summation."""
    ids, a, b = _aligned(a_items, b_items)
    terms: list[float] = []
    for iid in ids:
        pa = _positions(a[iid], "a", iid)
        pb = _positions(b[iid], "b", iid)
        for pos in sorted(set(pa) | set(pb)):
            if pos not in pa or pos not in pb:
                terms.append(ONE_SIDED_POSITION)
                continue
            va, vb = pa[pos], pb[pos]
            l1 = [abs(va.get(t, ABSENT_LP) - vb.get(t, ABSENT_LP)) for t in sorted(set(va) | set(vb))]
            terms.append(math.fsum(l1))
    return _mean(terms, "topk")


# --------------------------------------------------------------------------- resid

def _profiles(profiles: Any, side: str) -> dict[str, Any] | list[Any]:
    """Accept ``{item_id: vector}``, ``[{item_id, profile}]`` or a bare list of vectors
    (positional, the shape §3.1 stores under ``channels.resid.profile``)."""
    if isinstance(profiles, Mapping):
        return _by_id(profiles, side)
    seq = list(profiles)
    if seq and all(isinstance(p, Mapping) for p in seq):
        return _by_id(seq, side)
    return seq


def _normalized(vec: Any, where: str) -> list[float]:
    """Clamp negatives to 0, then L1-normalize."""
    if isinstance(vec, Mapping):
        if "profile" not in vec:
            raise ValueError(f"{where}: no profile")
        vec = vec["profile"]
    if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
        raise ValueError(f"{where}: profile must be a list of numbers")
    clamped = [max(0.0, _finite(x, where)) for x in vec]
    total = math.fsum(clamped)
    if total <= 0.0:
        raise ValueError(f"{where}: profile is all zero after clamping; cannot normalize")
    return [x / total for x in clamped]


def _jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """0.5 KL(p||m) + 0.5 KL(q||m), m = (p+q)/2, natural log, 0 log 0 = 0, no smoothing.

    The per-term ratio is written 2*x / (x + y) rather than x / ((x + y) / 2): the two
    are the same number, but halving the sum underflows to 0.0 when x is the smallest
    subnormal and y is 0, and the division then raises where the exact ratio is 2. The
    denominator is zero only when x and y are both zero, which the 0 log 0 = 0 guard
    skips, so no term is ever dropped and nothing is smoothed. Repaired 2026-09-08
    after the module's own hypothesis test found p = [5e-324, 1.0], q = [0.0, 1.0].
    """
    kl_p = math.fsum(
        pi * math.log(2.0 * pi / (pi + qi)) for pi, qi in zip(p, q) if pi > 0.0
    )
    kl_q = math.fsum(
        qi * math.log(2.0 * qi / (pi + qi)) for pi, qi in zip(p, q) if qi > 0.0
    )
    return max(0.0, 0.5 * kl_p + 0.5 * kl_q)


def resid(a_profiles: Any, b_profiles: Any) -> float:
    """Appendix B ``resid``: mean over items of the Jensen-Shannon divergence in nats between
    the L1-normalized, clamped per-layer profiles.  Profiles are aligned by ``item_id`` when
    ids are given, positionally when bare vectors are given; an item count or a vector length
    mismatch raises ``ValueError``."""
    pa = _profiles(a_profiles, "a")
    pb = _profiles(b_profiles, "b")
    if isinstance(pa, dict) != isinstance(pb, dict):
        raise ValueError("profiles: one side keyed by item_id, the other positional")
    if isinstance(pa, dict):
        ids, a, b = _aligned(pa, pb)
        pairs = [(a[i], b[i], repr(i)) for i in ids]
    else:
        if len(pa) != len(pb):
            raise ValueError(f"profiles: item count differs ({len(pa)} vs {len(pb)})")
        pairs = [(x, y, str(k)) for k, (x, y) in enumerate(zip(pa, pb))]
    terms: list[float] = []
    for va, vb, tag in pairs:
        p = _normalized(va, f"a[{tag}]")
        q = _normalized(vb, f"b[{tag}]")
        if len(p) != len(q):
            raise ValueError(f"[{tag}]: profile length differs ({len(p)} vs {len(q)})")
        terms.append(_jsd(p, q))
    return _mean(terms, "resid")


# --------------------------------------------------------------------------- lens

def _layers(layers: Any, side: str) -> dict[str, Any] | list[Any]:
    """Accept ``{item_id: layer}``, ``[{item_id, converge_layer}]`` or a bare list of layers
    (positional, the shape §3.1 stores under ``channels.lens.converge_layer``)."""
    if isinstance(layers, Mapping):
        return _by_id(layers, side)
    seq = list(layers)
    if seq and all(isinstance(p, Mapping) for p in seq):
        return _by_id(seq, side)
    return seq


def _layer_value(v: Any, where: str) -> float:
    if isinstance(v, Mapping):
        if "converge_layer" not in v:
            raise ValueError(f"{where}: no converge_layer")
        v = v["converge_layer"]
    if v is None:
        raise ValueError(f"{where}: converge_layer absent")
    return _finite(v, where)


def lens(a_layers: Any, b_layers: Any, n_layers: int) -> float:
    """Appendix B ``lens``: mean over items of ``|delta converge_layer| / L`` with ``L`` the
    reference cert's ``n_layers``.  Whether the two certs' ``n_layers`` agree is the caller's
    check (they get ``inconclusive`` when they differ); this function takes the one ``L``."""
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers <= 0:
        raise ValueError("n_layers must be a positive int")
    la = _layers(a_layers, "a")
    lb = _layers(b_layers, "b")
    if isinstance(la, dict) != isinstance(lb, dict):
        raise ValueError("layers: one side keyed by item_id, the other positional")
    if isinstance(la, dict):
        ids, a, b = _aligned(la, lb)
        pairs = [(a[i], b[i], repr(i)) for i in ids]
    else:
        if len(la) != len(lb):
            raise ValueError(f"layers: item count differs ({len(la)} vs {len(lb)})")
        pairs = [(x, y, str(k)) for k, (x, y) in enumerate(zip(la, lb))]
    terms = [
        abs(_layer_value(x, f"a[{tag}]") - _layer_value(y, f"b[{tag}]")) / n_layers
        for x, y, tag in pairs
    ]
    return _mean(terms, "lens")
