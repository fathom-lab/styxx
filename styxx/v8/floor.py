"""styxx.v8.floor — the noise floor and the drift decision (spec v0.2 §5, §5.2, §6).

Contract: styxx/v8/INTERFACES_layer2.md section 4.

The floor is an empirical null: R same-subject runs under a logged nuisance plan, every
pairwise distance computed with the Appendix B function of the channel, ``floor = max``.
``alpha_single = 1/(pairs+1)`` is the nominal probability under exchangeability that a
fresh same-model run exceeds that max.  The decision compares a rounded distance to a
rounded floor (``ROUND_PLACES`` = 9) and the overall verdict follows the §5.2 precedence.

Pure CPU; nothing here touches a model runtime.

Decisions (where the contract or the spec leaves room; each is pinned by a test):

* ``decide(distance=None, floor=<number>, ...)`` is ``"inconclusive"``: the lens channel is
  inconclusive when the two certs' ``n_layers`` differ (Appendix B) and the caller has no
  number to pass.  ``floor is None`` is still checked before anything else.  Every numeric
  argument that is not None is validated before any verdict is issued — a NaN, an infinity, a
  negative number or a non-number raises even when the verdict would not have looked at it,
  because a distance that is not a distance must never reach a result cert.  ``covered``,
  ``skew`` and ``sensitivity_present`` must be real ``bool`` values (a truthy string is not a
  receipt).
* ``overall({}, ...)`` (no shared channel at all) is ``("inconclusive", 2)``: nothing was
  compared, so it cannot be ``same``.
* ``overall`` refuses (``ValueError``) a channel name outside ``CHANNELS``, a channel listed
  both under ``per_channel`` and ``skipped``, a verdict outside ``CHANNEL_VERDICTS``, and an
  ``identity_diff`` entry outside ``IDENTITY_FIELDS`` (``precision`` and ``revision`` are the
  ``cross-subject`` label of §6, never an ``identity`` verdict).  The identity verdict names
  its fields in ``IDENTITY_FIELDS`` order, so the string does not depend on the caller's order.
* ``floors()`` returns every channel in ``CHANNELS`` as a key, with ``None`` for a channel
  absent on any run (contract type ``dict[str, dict | None]``).
* A channel block's ``present`` flag, when it exists, must be a ``bool``; anything else is a
  malformed body and raises rather than counting as present.
* ``pairwise`` accepts an optional keyword ``roles`` (``{item_id: role}`` or a battery items
  list) so the caller can pass the battery cert's roles and the floor on ``exact`` is computed
  over exactly the items ``verify`` scores.  Without it, roles come from the run bodies' items
  (``role`` per item, default ``"item"``) and every run must agree on every item's role — a
  role that differs between runs would make the floor depend on the order the runs were given
  in, so it is refused.
* ``resid`` profiles and ``lens`` converge layers are positional per item in §3.1; here they
  are re-keyed by each run's own ``items`` order (``item_id``) before the Appendix B function
  aligns them, so two runs whose items were stored in different orders still compare the
  same item with itself.  A profile/layer list whose length differs from the item count is
  refused, as are differing or non-positive ``n_layers``.
* Two run bodies carrying the same ``run_index`` are refused: a floor over one run counted
  twice is a floor over nothing.  Bodies without ``run_index`` are accepted (the contract
  does not require it on the input).
* The overall verdict when every shared channel is ``same`` or ``same (transient)`` is
  ``"same"`` (exit 0) with a sensitivity receipt and ``"same (sensitivity unmeasured)"``
  (exit 2) without; the per-channel transient detail lives only in ``per_channel``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from styxx.v8 import distances as D

# --------------------------------------------------------------------------- constants
# consts.py is written by the `cert` agent.  Until it lands the values below are the
# frozen contract's; when it exists it wins.
try:  # pragma: no cover - exercised whichever branch the tree is in
    from styxx.v8.consts import CHANNELS, EXIT, ROUND_PLACES
except ImportError:  # CONTRACT FALLBACK: delete once styxx/v8/consts.py exists
    CHANNELS = ("exact", "seqlp", "topk", "resid", "lens")
    ROUND_PLACES = 9
    EXIT = {
        "same": 0,
        "drift": 1,
        "identity": 1,
        "inconclusive": 2,
        "skew": 2,
        "beyond-floor-coverage": 2,
        "sensitivity-unmeasured": 2,
        "mismatch": 3,
        "invalid": 4,
        "unavailable": 5,
    }

IDENTITY_FIELDS = (
    "weights_sha256",
    "tokenizer_sha256",
    "config_sha256",
    "generation_config_sha256",
)

# The verdict vocabulary a channel may carry (§5.2 + GATED S5-02 below).
CHANNEL_VERDICTS = (
    "same",
    "same (transient)",
    "exceeds_floor",
    "drift",
    "skew",
    "inconclusive",
    "beyond-floor-coverage",
)

# Exit code per channel-level verdict (§6 table).  exceeds_floor is an unconfirmed
# exceedance and exits like inconclusive (a --diff without a confirmation step).
_CHANNEL_EXIT = {
    "same": EXIT["same"],
    "same (transient)": EXIT["same"],
    "exceeds_floor": EXIT["inconclusive"],
    "drift": EXIT["drift"],
    "skew": EXIT["skew"],
    "inconclusive": EXIT["inconclusive"],
    "beyond-floor-coverage": EXIT["beyond-floor-coverage"],
}


# --------------------------------------------------------------------------- rounding
def _rounded(x: float) -> float:
    """§5.2 rounding, shared with the distance functions."""
    return D.rounded(x)


def _finite(name: str, x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0 (Appendix B distances are non-negative), got {x!r}")
    return x


def _flag(name: str, x: Any) -> bool:
    if not isinstance(x, bool):
        raise TypeError(f"{name} must be a bool, got {type(x).__name__}")
    return x


# --------------------------------------------------------------------------- channel access
def _channel_block(body: Mapping, channel: str) -> dict | None:
    """The body's channel block when the channel is present, else None (§3.2: absent, never zero)."""
    channels = body.get("channels")
    if not isinstance(channels, Mapping):
        return None
    block = channels.get(channel)
    if not isinstance(block, Mapping):
        return None
    if "present" in block:
        present = block["present"]
        if not isinstance(present, bool):
            raise ValueError(
                f"channels.{channel}.present must be a bool, got {type(present).__name__}"
            )
        if not present:
            return None
    return dict(block)


def _items(body: Mapping) -> list[dict]:
    items = body.get("items")
    if not isinstance(items, list):
        raise ValueError("run body has no items list")
    for it in items:
        if not isinstance(it, Mapping) or not isinstance(it.get("item_id"), str):
            raise ValueError("every run item needs a str item_id")
    return items


def _ids_in_order(body: Mapping) -> list[str]:
    """The run's item ids in the order the body stores them (the positional order of the
    white-box channels' per-item lists).  Duplicates are refused."""
    ids = [it["item_id"] for it in _items(body)]
    if len(set(ids)) != len(ids):
        raise ValueError("run body carries a duplicate item_id")
    return ids


def _roles_from_runs(run_bodies: Sequence[Mapping]) -> dict[str, str]:
    """Roles from the run bodies' items (``role`` per item, default ``"item"``).  Every run
    must agree on every item's role; a disagreement would make the floor depend on the
    order the runs were given in."""
    roles: dict[str, str] = {}
    for k, body in enumerate(run_bodies):
        for it in _items(body):
            iid = it["item_id"]
            role = it.get("role", "item")
            if not isinstance(role, str):
                raise ValueError(f"run {k}: item {iid!r} has a non-str role")
            if iid in roles and roles[iid] != role:
                raise ValueError(
                    f"run {k}: item {iid!r} has role {role!r} but an earlier run says {roles[iid]!r}"
                )
            roles[iid] = role
    return roles


def _roles_arg(roles: Any) -> dict[str, str]:
    """A caller-supplied ``roles``: ``{item_id: role}`` or the battery cert's items list."""
    if isinstance(roles, Mapping):
        out = {}
        for k, v in roles.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("roles must map str item_id -> str role")
            out[k] = v
        return out
    out = {}
    for rec in roles:
        if not isinstance(rec, Mapping) or not isinstance(rec.get("item_id"), str) or not isinstance(rec.get("role"), str):
            raise ValueError("roles: every entry needs a str item_id and a str role")
        if rec["item_id"] in out:
            raise ValueError(f"roles: duplicate item_id {rec['item_id']!r}")
        out[rec["item_id"]] = rec["role"]
    return out


def _n_layers(block: Mapping, channel: str) -> int:
    n = block.get("n_layers")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"channels.{channel}.n_layers must be a positive int, got {n!r}")
    return n


def _per_item(body: Mapping, block: Mapping, channel: str, key: str) -> dict[str, Any]:
    """Re-key a positional per-item list under ``channels.<channel>.<key>`` by the run's own
    item order, so the Appendix B function aligns the two runs by item_id."""
    values = block.get(key)
    if not isinstance(values, list):
        raise ValueError(f"channels.{channel}.{key} must be a list (one entry per item)")
    ids = _ids_in_order(body)
    if len(values) != len(ids):
        raise ValueError(
            f"channels.{channel}.{key} has {len(values)} entries for {len(ids)} items"
        )
    return dict(zip(ids, values))


# --------------------------------------------------------------------------- one distance
def _distance(a: Mapping, b: Mapping, channel: str, roles: Mapping[str, str] | None = None) -> float:
    """Appendix B distance between two run bodies on one channel (both must carry it)."""
    if channel == "exact":
        a_items, b_items = _items(a), _items(b)
        if roles is None:
            roles = _roles_from_runs([a, b])
        return float(D.exact(a_items, b_items, roles)[0])
    if channel == "seqlp":
        return float(D.seqlp(_items(a), _items(b)))
    if channel == "topk":
        return float(D.topk(_items(a), _items(b)))
    if channel == "resid":
        ca, cb = _channel_block(a, "resid"), _channel_block(b, "resid")
        if ca is None or cb is None:
            raise ValueError("resid absent on a run")
        na, nb = _n_layers(ca, "resid"), _n_layers(cb, "resid")
        if na != nb:
            raise ValueError(f"resid n_layers differ between runs ({na} vs {nb})")
        return float(D.resid(_per_item(a, ca, "resid", "profile"), _per_item(b, cb, "resid", "profile")))
    if channel == "lens":
        ca, cb = _channel_block(a, "lens"), _channel_block(b, "lens")
        if ca is None or cb is None:
            raise ValueError("lens absent on a run")
        na, nb = _n_layers(ca, "lens"), _n_layers(cb, "lens")
        if na != nb:
            raise ValueError(f"lens n_layers differ between runs ({na} vs {nb})")
        return float(
            D.lens(
                _per_item(a, ca, "lens", "converge_layer"),
                _per_item(b, cb, "lens", "converge_layer"),
                na,
            )
        )
    raise ValueError(f"unknown channel {channel!r}; channels are {CHANNELS}")


# --------------------------------------------------------------------------- §5.1 step 3
def _check_runs(run_bodies: Any) -> list[Mapping]:
    if isinstance(run_bodies, (str, bytes, Mapping)) or not isinstance(run_bodies, Sequence):
        raise TypeError("run_bodies must be a list of run bodies")
    bodies = list(run_bodies)
    if len(bodies) < 2:
        raise ValueError(f"a floor needs at least 2 runs, got {len(bodies)}")
    for k, body in enumerate(bodies):
        if not isinstance(body, Mapping):
            raise TypeError(f"run {k}: a run body must be a dict, got {type(body).__name__}")
    return bodies


def pairwise(run_bodies: list[dict], channel: str, *, roles: Any = None) -> dict | None:
    """All pairwise distances between R same-subject runs on one channel; floor = max.

    Returns ``{"floor", "distances", "runs", "pairs", "alpha_single"}`` or None when the
    channel is absent on any run.  R < 2 raises ValueError (one run has no pairs).
    Distances are rounded to ROUND_PLACES and listed in (i, j) order, i < j, over the runs
    as given; the floor is the max of the rounded list.  ``roles`` (optional, ``exact`` only)
    is the battery's ``{item_id: role}`` or items list; see the module docstring.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; channels are {CHANNELS}")
    bodies = _check_runs(run_bodies)
    runs = len(bodies)
    for body in bodies:
        if _channel_block(body, channel) is None:
            return None
    role_map = None
    if channel == "exact":
        role_map = _roles_arg(roles) if roles is not None else _roles_from_runs(bodies)
    distances: list[float] = []
    for i in range(runs):
        for j in range(i + 1, runs):
            d = _finite(f"distance({i},{j})", _distance(bodies[i], bodies[j], channel, role_map))
            distances.append(_rounded(d))
    pairs = len(distances)
    assert pairs == runs * (runs - 1) // 2
    return {
        "floor": _rounded(max(distances)),
        "distances": distances,
        "runs": runs,
        "pairs": pairs,
        "alpha_single": 1.0 / (pairs + 1),
    }


def floors(run_bodies: list[dict], *, roles: Any = None) -> dict[str, dict | None]:
    """Per-channel floor blocks; a channel absent on any run maps to None (never zero-filled)."""
    bodies = _check_runs(run_bodies)
    return {channel: pairwise(bodies, channel, roles=roles) for channel in CHANNELS}


# ------------------------------------------------------------------- §5.4 coverage vocabulary


def environment_paths(environment: Any, prefix: str = "") -> list[str]:
    """Every leaf of an environment block as a dotted path, sorted — §5.4's vocabulary.

    ``{"hardware": {"gpu": "cpu"}}`` -> ``["hardware.gpu"]``.  An empty mapping is a leaf of its
    own, so "the block is present and empty" and "the block is absent" stay different facts;
    that is ``runner.environment_leaves``'s rule and this function agrees with it deliberately,
    because ``verify`` resolves the names this produces against ``subject.environment`` with
    ``runner``'s reading.  Duplicated here rather than imported so this module keeps importing
    nothing but ``distances``.
    """
    out: list[str] = []
    if not isinstance(environment, Mapping):
        return out
    for key in sorted(k for k in environment if isinstance(k, str)):
        value = environment[key]
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.extend(environment_paths(value, path + "."))
        else:
            out.append(path)
    return out


def plan_coverage(plan_body: Any) -> tuple[list[str], list[str]]:
    """``(covers, not_covered)`` as §5.4 defines them, DERIVED from a noise plan's own bytes.

    §5.4: "``noise_floor.covers`` lists the nuisance factors the plan varied; ``not_covered``
    lists the environment fields it held fixed."  Both halves are therefore functions of two
    fields the plan signed before the runs — ``body.nuisance`` and ``body.environment`` — and
    neither is a third field an issuer may write beside them.  ``covers`` is the declared factor
    names; ``not_covered`` is every environment leaf the plan did not declare as a factor, by
    full dotted path or by its first segment (a plan that varies ``hardware`` covers
    ``hardware.gpu``).

    This is the whole of A-COVER.  ``covers`` decides whether an outside reproduction is binding
    or unreachable (§5.4, §9), so an issuer that picks it after the runs picks who may contradict
    the result: a canonical signed with ``covers`` naming the gpu, the driver and everything else
    and ``not_covered: []`` appended at exit 0 beside an honest floor, because nothing compared
    either list against the plan it names by id.  ``Log._check_floor_covers_match_the_plan``
    compares them, exactly as ``_check_floor_matches_its_runs`` compares the numbers.

    LIMIT, and it is the label class of ``papers/v8/THE_BOUNDARY_2026_09_09.md``: the plan's own
    ``nuisance`` and ``environment`` are still bytes the issuer wrote.  This moves the choice from
    after the runs to before them and pins it to a cert signed earlier; it does not make the plan
    true.
    """
    if not isinstance(plan_body, Mapping):
        return [], []
    covers: list[str] = []
    blocks = plan_body.get("nuisance")
    for block in blocks if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)) else []:
        if not isinstance(block, Mapping):
            continue
        factor = block.get("factor")
        if isinstance(factor, str) and factor and factor not in covers:
            covers.append(factor)
    covers.sort()
    held = set(covers)
    not_covered = [
        name
        for name in environment_paths(plan_body.get("environment"))
        if name not in held and name.split(".")[0] not in held
    ]
    return covers, sorted(not_covered)


# --------------------------------------------------------------------------- §5.2 per channel
def decide(
    distance: float | None,
    floor: float | None,
    confirmation: float | None,
    *,
    covered: bool = True,
    skew: bool = False,
) -> str:
    """The §5.2 per-channel verdict.  Both sides are rounded to ROUND_PLACES before comparing.

    Order: no floor → "inconclusive"; no distance → "inconclusive"; not covered →
    "beyond-floor-coverage"; d ≤ floor → "same"; d > floor with an instrument skew → "skew"
    (checked before drift); d > floor with no confirmation run → "exceeds_floor";
    confirmation ≤ floor → "same (transient)"; both > floor → "drift".
    Every argument is validated before any verdict is issued.
    """
    f = None if floor is None else _rounded(_finite("floor", floor))
    d = None if distance is None else _rounded(_finite("distance", distance))
    c = None if confirmation is None else _rounded(_finite("confirmation", confirmation))
    covered = _flag("covered", covered)
    skew = _flag("skew", skew)
    if f is None:
        return "inconclusive"
    if d is None:
        # The channel could not be measured against this floor (e.g. lens with differing
        # n_layers, Appendix B): a verdict that says nothing rather than a number.
        return "inconclusive"
    if not covered:
        return "beyond-floor-coverage"
    if d <= f:
        return "same"
    if skew:
        return "skew"
    if c is None:
        # GATED S5-02: recommendation implemented; operator may reverse.
        # Option A: an unconfirmed exceedance is `exceeds_floor`; `drift` is reserved for a
        # comparison that ran the confirmation step (verify --ref).
        return "exceeds_floor"
    if c <= f:
        return "same (transient)"
    return "drift"


# --------------------------------------------------------------------------- §5.2 overall
_RANK = ("drift", "skew", "exceeds_floor", "inconclusive", "beyond-floor-coverage")


def _names(name: str, value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a list of field names")
    out = []
    for v in value:
        if not isinstance(v, str) or not v:
            raise ValueError(f"{name} must name fields, got {v!r}")
        out.append(v)
    return out


def overall(
    per_channel: dict[str, str],
    *,
    identity_diff: list[str],
    skipped: list[str],
    sensitivity_present: bool,
) -> tuple[str, int]:
    """The overall verdict and exit code (§5.2 precedence, §6 exit table).

    identity > drift > skew > exceeds_floor > (inconclusive | beyond-floor-coverage) > same.
    `skipped` (channels present on one side only) never affects the code.  A `same` with no
    sensitivity receipt on record is "same (sensitivity unmeasured)" and exits 2 (§5.3).
    With no shared channel at all nothing was compared: "inconclusive", 2.
    """
    if not isinstance(per_channel, Mapping):
        raise TypeError("per_channel must be a dict of channel -> verdict")
    skipped_names = _names("skipped", skipped)
    identity = _names("identity_diff", identity_diff)
    sensitivity_present = _flag("sensitivity_present", sensitivity_present)
    for ch, v in per_channel.items():
        if ch not in CHANNELS:
            raise ValueError(f"unknown channel {ch!r}; channels are {CHANNELS}")
        if v not in CHANNEL_VERDICTS:
            raise ValueError(f"channel {ch!r} carries an unknown verdict {v!r}")
        if ch in skipped_names:
            raise ValueError(f"channel {ch!r} is both compared and skipped")
    for ch in skipped_names:
        if ch not in CHANNELS:
            raise ValueError(f"skipped names an unknown channel {ch!r}")
    bad = [f for f in identity if f not in IDENTITY_FIELDS]
    if bad:
        raise ValueError(f"identity_diff may only name {IDENTITY_FIELDS}, got {bad!r}")
    if len(set(identity)) != len(identity):
        raise ValueError(f"identity_diff repeats a field: {identity!r}")
    if identity:
        ordered = [f for f in IDENTITY_FIELDS if f in identity]
        return f"identity ({', '.join(ordered)})", EXIT["identity"]
    verdicts = set(per_channel.values())
    for v in _RANK:
        if v in verdicts:
            return v, _CHANNEL_EXIT[v]
    if not per_channel:
        return "inconclusive", EXIT["inconclusive"]
    # every shared channel is same or same (transient)
    if not sensitivity_present:
        return "same (sensitivity unmeasured)", EXIT["sensitivity-unmeasured"]
    return "same", EXIT["same"]


# --------------------------------------------------------------------------- §5.5 baseline gap
def _floor_of(body: Mapping) -> dict[str, float | None]:
    """``{channel: floor}`` read off ``body.noise_floor.per_channel``; absent -> None.

    A channel with no block, a block with no ``floor``, or a ``floor`` that is not a number has
    no floor here.  Nothing is invented: an absent floor stays None all the way to the report,
    where it prints as ``-`` (§3.2: absent, never zero).
    """
    out: dict[str, float | None] = {channel: None for channel in CHANNELS}
    nf = body.get("noise_floor")
    if not isinstance(nf, Mapping):
        return out
    per = nf.get("per_channel")
    if not isinstance(per, Mapping):
        return out
    for channel, block in per.items():
        if channel not in CHANNELS or not isinstance(block, Mapping):
            continue
        value = block.get("floor")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[channel] = _rounded(float(value))
    return out


def _present(body: Mapping, channel: str) -> bool:
    try:
        return _channel_block(body, channel) is not None
    except ValueError:
        return False


def baseline_gap(
    previous_body: Mapping,
    body: Mapping,
    *,
    roles: Any = None,
) -> dict:
    """How far a new canonical fingerprint sits from the baseline it replaces (§5.5).

    **This is disclosure, not prevention.**  §5.5 lets a second canonical fingerprint for one
    subject append when it carries a ``previous`` ref (or is another run under the same logged
    noise plan), and choosing a new baseline is a legitimate act -- a re-run on repaired
    hardware, a corrected battery, a genuinely better measurement.  Nothing here refuses that
    choice and nothing here should.  What the choice must not be is *silent*: the adversary's
    BASELINE-CHOICE attack is not that a second baseline exists, it is that a reader comparing
    against the new one is never told a different one was on record, nor how far the two are
    apart.  This function computes that distance, from two certs the log already holds, so the
    number travels beside the verdict instead of having to be reconstructed by a reader who
    already suspects something.

    **What it cannot do, and this is the label class.**  The gap says how far the two baselines
    are apart.  It cannot say which of them measured the subject, because both were written by
    the same party and a relabelled computation is byte-indistinguishable from a computation
    (``papers/v8/THE_BOUNDARY_2026_09_09.md``, class two).  An issuer who wants a favourable
    baseline and does not mind the disclosure can still publish one; an issuer who fabricates
    the *runs* under the new baseline produces a gap that is honest arithmetic over dishonest
    bytes.  BASELINE-CHOICE remains in the label class where the relabel itself is concerned.
    What moves out of it is only the announcement.

    ``previous_body`` is the run body of the baseline already on record, ``body`` the run body of
    the fingerprint that would replace it.  The yardstick is the PREVIOUS baseline's floor: it is
    the number a reader had already accepted before the choice was made, and scaling the gap by
    the new baseline's own floor would let the same act that moves the baseline also move the
    ruler.  The new baseline's floors travel beside it under ``floor_new`` so both are visible.

    Returns::

        {
          "floor_owner": "previous",
          "per_channel": {channel: {"distance", "floor", "ratio", "exceeds_floor"[, "note"]}},
          "floor_new": {channel: floor},          # channels the new baseline gives a floor
          "skipped_channels": [...],              # present on one side only -- never compared
          "channels_exceeding_floor": [...],      # sorted, CHANNELS order
          "max_ratio": float | None,              # the largest gap/floor over the channels
        }

    A channel present on one side only is skipped, exactly as §5.2 skips it in a comparison; a
    channel whose distance will not compute carries ``distance: None`` and a ``note`` naming the
    exception, and never a fabricated zero.  Nothing here raises on a hostile body: a gap that
    could not be computed is a gap reported as uncomputed.
    """
    if not isinstance(previous_body, Mapping):
        raise TypeError("previous_body must be a run body dict")
    if not isinstance(body, Mapping):
        raise TypeError("body must be a run body dict")

    floors_prev = _floor_of(previous_body)
    floors_new = _floor_of(body)
    per: dict[str, dict] = {}
    skipped: list[str] = []
    exceeding: list[str] = []
    ratios: list[float] = []

    for channel in CHANNELS:
        in_prev, in_new = _present(previous_body, channel), _present(body, channel)
        if in_prev != in_new:
            skipped.append(channel)
            continue
        if not in_prev:
            continue
        note: str | None = None
        distance: float | None
        try:
            distance = _rounded(
                _finite(
                    f"gap({channel})",
                    _distance(previous_body, body, channel, _roles_arg(roles) if roles is not None else None),
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            distance, note = None, f"{type(exc).__name__}: {exc}"
        f = floors_prev.get(channel)
        ratio = None
        if distance is not None and f is not None and f > 0.0:
            ratio = _rounded(distance / f)
            ratios.append(ratio)
        exceeds: bool | None = None
        if distance is not None and f is not None:
            exceeds = distance > f
            if exceeds:
                exceeding.append(channel)
        block: dict[str, Any] = {
            "distance": distance,
            "floor": f,
            "ratio": ratio,
            "exceeds_floor": exceeds,
        }
        if note is not None:
            block["note"] = note
        per[channel] = block

    return {
        "floor_owner": "previous",
        "per_channel": per,
        "floor_new": {c: v for c, v in floors_new.items() if v is not None},
        "skipped_channels": skipped,
        "channels_exceeding_floor": exceeding,
        "max_ratio": max(ratios) if ratios else None,
    }


__all__ = [
    "CHANNEL_VERDICTS",
    "IDENTITY_FIELDS",
    "baseline_gap",
    "decide",
    "floors",
    "overall",
    "pairwise",
]
