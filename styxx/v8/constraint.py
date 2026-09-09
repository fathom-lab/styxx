"""styxx.v8.constraint — how much prior logged material could contradict this claim.

Context: ``papers/v8/THE_BOUNDARY_2026_09_09.md`` and the receipts under
``papers/v8/class_two_empty_2026_09_09/``.

That document opened with a partition of surviving defects into a class reachable by a check over
logged bytes and a class "reachable by nothing" — fields whose forged value is byte-indistinguishable
from the honest one. The second class had four members. All four were reclassified as reachable, and
every one of the four errors was the same error: *the argument examined one certificate instead of
the log the certificate sits in*. Another logged cert already carried information a forged value
would contradict, and nothing compared them.

What is left after that is not a list of unreachable fields. It is one unreachable act: **the first
claim about anything**. A subject nobody has measured has no prior logged cert to contradict, so
every consistency check on it is vacuous whatever it reports. Consistency accrues; it cannot be
bootstrapped.

A verdict is therefore incomplete without the quantity this module computes: **how many prior
entries were in a position to contradict it**. Zero for a first claim, growing with that subject's
history. Nothing in this package computed it before.

What this module is
-------------------

``census(cert, prior)`` takes one cert and the entries that precede it, and returns, per
cross-cert predicate, the count of prior entries that predicate could have run against. It is a
census, not a check: it does not run the predicates and it never says a cert is wrong. It says how
much a check would have had to work with.

The five predicates are the ones this system has or should have. Two exist in ``log.py`` today;
two are named in THE_BOUNDARY as reachable-but-unimplemented; one is a consequence of the same
bytes. Each is listed with what it asks and, in ``blocked_by``, what stops it running here.

* ``floor_agreement`` — same ``weights_sha256`` and same battery: the floor's own distances and
  the per-run channel values. This is the member-1 predicate. On the published log entry 6 records
  ten pairwise distances whose same-batch pairs are all exactly 0 and whose different-batch pairs
  are all strictly positive, so a later floor claiming batch 1 against batch 8 separates nothing
  contradicts a cert already on file.
* ``snapshot_agreement`` — same ``(hf_repo, revision)``: the Appendix A.2 content hashes beside
  them must agree, in both directions. This is the member-3 predicate.
* ``schedule`` — the noise plan this cert names must fix which assignment each run takes, so the
  reference run is not chosen after the plan is logged. This is the member-2 predicate. The
  published plan fixes the factors, their levels and the count, and carries no schedule at all,
  so this predicate reports ``usable = 0`` on it and names why.
* ``determinism`` — same subject at a batch level the log has already recorded: the channel values
  must reproduce. Falls out of the same bytes as ``floor_agreement`` and is separate because it
  binds a single run rather than a floor.
* ``issuer_history`` — how many entries this key signed before. The weakest of the five and
  reported as such: it constrains a party only against itself.

Own material is excluded, and that is the point
-----------------------------------------------

A cert's own inputs cannot constrain it. ``own_material`` is the cert's id, every id in its
``refs``, its ``recipe.battery``, and its floor's ``plan`` and ``runs``. An entry in that set is
counted under ``own`` and never under ``available``.

This is not a refinement. ``Log.previous_comparable`` does not make the exclusion, and the
consequence is recorded in THE_BOUNDARY: on this lab's own published verdict the lookup for the
canonical fingerprint returns entry 5, which is that canonical's own floor run 4. Measuring a
claim against material it produced is not constraint, and a number that counts it overstates.

``independent`` is stricter still: available entries signed by a **different issuer key**. A
second party's bytes are the only thing that can constrain a first claim, and on a single-issuer
log this column is 0 everywhere by construction.

What this reaches, and what it does not
---------------------------------------

It reaches a count of prior entries in scope, derived from stored bytes alone. It does **not**
reach whether those entries actually agree with this one — that is the predicate's job, not the
census's — and a large count is not evidence of correctness. A log with a thousand entries by one
issuer who has been consistently wrong from the start yields a large census and constrains
nothing a stranger cares about, which is why ``independent`` is reported separately and why the
disclosure line prints it.

It also does not reach suppression. A party who commits to a favourable schedule before running is
contradicted by nothing, so the schedule predicate would pass. THE_BOUNDARY files that residue
under member 3's caveat and this module inherits it unchanged.

No styxx import
---------------

This module's own imports are ``json``, ``pathlib`` and ``typing`` — nothing from ``styxx``, and
in particular none of ``log``, ``cert``, ``verify`` or ``cli``. It is a pure function of the JSON
a log stores. (Importing it by its dotted name still executes ``styxx/__init__.py``, which pulls
in the rest of the package on its own account; that is the package's behaviour, not this
module's. Copy the file out and it runs on stdlib alone, which is the standing the
``class_two_empty_2026_09_09`` receipts have.)
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "PREDICATES",
    "own_material",
    "census",
    "disclosure_line",
    "read_log_entries",
    "census_log",
]

# --------------------------------------------------------------------------- the roster

PREDICATES: tuple[tuple[str, str], ...] = (
    (
        "floor_agreement",
        "across entries sharing a weights_sha256 and a battery, the floor's distances and the "
        "per-run channel values must be consistent (THE_BOUNDARY member 1)",
    ),
    (
        "snapshot_agreement",
        "across the entries of one log, one (hf_repo, revision) names one set of Appendix A.2 "
        "content hashes, and one set of content hashes names one (hf_repo, revision) "
        "(THE_BOUNDARY member 3)",
    ),
    (
        "schedule",
        "the runs' factor assignments are the logged noise plan's schedule, in order, so the "
        "reference run is fixed before the data exists (THE_BOUNDARY member 2)",
    ),
    (
        "determinism",
        "a run at a batch level this subject has already been logged at must reproduce the "
        "recorded channel values exactly",
    ),
    (
        "issuer_history",
        "how many entries this issuer key has signed before; constrains a party only against "
        "itself",
    ),
)

# Appendix A.2: the content hashes taken over the snapshot's actual bytes.
A2_HASHES = (
    "weights_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "generation_config_sha256",
)

# Keys that would let a plan fix which assignment each run takes. None of them is in the v0.2
# prereg body; the census reports their absence rather than assuming a schedule exists.
SCHEDULE_KEYS = ("schedule", "assignments", "runs_spec", "reference")


# --------------------------------------------------------------------------- field access
#
# Every accessor below tolerates a malformed cert by returning None. A census is a disclosure
# printed beside a verdict; it must not raise on bytes the verifier has already refused.


def _mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _body(cert: Mapping) -> dict:
    return _mapping(cert.get("body"))


def _subject(cert: Mapping) -> dict:
    return _mapping(cert.get("subject"))


def _recipe(cert: Mapping) -> dict:
    return _mapping(cert.get("recipe"))


def _cert_id(cert: Mapping) -> Optional[str]:
    value = cert.get("id")
    return value if isinstance(value, str) else None


def _issuer_key(cert: Mapping) -> Optional[str]:
    value = _mapping(cert.get("issuer")).get("key")
    return value if isinstance(value, str) else None


def _ref_ids(cert: Mapping, role: Optional[str] = None) -> list[str]:
    refs = cert.get("refs")
    out: list[str] = []
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        return out
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        if role is not None and ref.get("role") != role:
            continue
        value = ref.get("id")
        if isinstance(value, str):
            out.append(value)
    return out


def _battery_id(cert: Mapping) -> Optional[str]:
    """The battery this cert is about: its ``recipe.battery``, or its own id if it IS one."""
    if cert.get("type") == "battery":
        return _cert_id(cert)
    value = _recipe(cert).get("battery")
    if isinstance(value, str):
        return value
    refs = _ref_ids(cert, "battery")
    return refs[0] if refs else None


def _floor_block(cert: Mapping) -> dict:
    return _mapping(_body(cert).get("noise_floor"))


def _plan_id(cert: Mapping) -> Optional[str]:
    """The noise plan this cert rests on: its ``noise_plan`` ref, else the floor's ``plan``."""
    refs = _ref_ids(cert, "noise_plan")
    if refs:
        return refs[0]
    value = _floor_block(cert).get("plan")
    return value if isinstance(value, str) else None


def _is_plan(cert: Mapping) -> bool:
    return cert.get("type") == "prereg"


def _plan_has_schedule(cert: Mapping) -> bool:
    body = _body(cert)
    return any(key in body for key in SCHEDULE_KEYS)


def _channels(cert: Mapping) -> Optional[dict]:
    value = _body(cert).get("channels")
    return dict(value) if isinstance(value, Mapping) else None


def _batch_level(cert: Mapping) -> Optional[Any]:
    """The batch size this cert's run was executed at.

    The nuisance assignment is preferred over the recipe because it is the value the run reports
    for itself; the recipe is the fallback for a cert carrying no nuisance block. When the two
    disagree that is a defect ``log.py`` already checks for (``_check_floor_labels_match_the_recipe``)
    and this module does not re-litigate it.
    """
    nuisance = _body(cert).get("nuisance")
    if isinstance(nuisance, Mapping) and "batch_size" in nuisance:
        value = nuisance.get("batch_size")
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
    decoding = _mapping(_recipe(cert).get("decoding"))
    value = decoding.get("batch_size")
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value)
    return None


def _repo_revision(cert: Mapping) -> Optional[tuple[str, str]]:
    subject = _subject(cert)
    repo, revision = subject.get("hf_repo"), subject.get("revision")
    if isinstance(repo, str) and isinstance(revision, str):
        return (repo, revision)
    return None


def _weights(cert: Mapping) -> Optional[str]:
    value = _subject(cert).get("weights_sha256")
    return value if isinstance(value, str) else None


def _has_a2(cert: Mapping) -> bool:
    subject = _subject(cert)
    return any(isinstance(subject.get(name), str) for name in A2_HASHES)


# --------------------------------------------------------------------------- own material


def own_material(cert: Mapping) -> list[str]:
    """Every cert id this cert is, names, or rests on — sorted, deduplicated.

    The cert's own id, every id under ``refs`` whatever its role, ``recipe.battery``, and the
    floor block's ``plan`` and ``runs``. Material in this set is the cert's own input and cannot
    corroborate it; ``census`` counts it under ``own`` and never under ``available``.
    """
    ids: set[str] = set()
    own = _cert_id(cert)
    if own:
        ids.add(own)
    ids.update(_ref_ids(cert))
    battery = _recipe(cert).get("battery")
    if isinstance(battery, str):
        ids.add(battery)
    floor = _floor_block(cert)
    plan = floor.get("plan")
    if isinstance(plan, str):
        ids.add(plan)
    runs = floor.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
        ids.update(r for r in runs if isinstance(r, str))
    return sorted(ids)


# --------------------------------------------------------------------------- the census


def _entry_label(cert: Mapping, index: Optional[int]) -> dict:
    label: dict[str, Any] = {"id": _cert_id(cert), "type": cert.get("type")}
    if index is not None:
        label["index"] = index
    return label


def _predicate_result(
    key: str,
    asks: str,
    *,
    applicable: bool,
    scope: Any,
    matched: list[tuple[Optional[int], Mapping]],
    own_ids: set[str],
    usable_test,
    own_issuer: Optional[str],
    blocked_by: list[str],
) -> dict:
    own_hits = [(i, c) for i, c in matched if _cert_id(c) in own_ids]
    available = [(i, c) for i, c in matched if _cert_id(c) not in own_ids]
    usable = [(i, c) for i, c in available if usable_test(c)]
    independent = [
        (i, c)
        for i, c in available
        if own_issuer is None or (_issuer_key(c) is not None and _issuer_key(c) != own_issuer)
    ]
    if not applicable:
        verdict = "not-applicable"
    elif usable:
        verdict = "constrained"
    else:
        verdict = "unconstrained"
    if applicable and available and not usable and not blocked_by:
        blocked_by = [
            "prior entries are in scope but none carries the fields this predicate reads"
        ]
    return {
        "asks": asks,
        "applicable": applicable,
        "scope": scope,
        "matched": len(matched),
        "own": len(own_hits),
        "available": len(available),
        "usable": len(usable),
        "independent": len(independent),
        # `entries` names the USABLE entries, not every entry in scope. An entry that matches the
        # scope but carries none of the fields the predicate reads constrains nothing, and listing
        # it beside the ones that do would overstate the census in the column a reader skims.
        "entries": [_entry_label(c, i) for i, c in usable],
        "in_scope_not_usable": len(available) - len(usable),
        "own_entries": [_entry_label(c, i) for i, c in own_hits],
        "verdict": verdict,
        "blocked_by": blocked_by,
    }


def census(
    cert: Mapping,
    prior: Iterable[Mapping] | Iterable[tuple[Optional[int], Mapping]],
    *,
    index: Optional[int] = None,
) -> dict:
    """The census of prior entries that could contradict ``cert``.

    ``prior`` is the entries strictly preceding ``cert`` in its log, either as bare certs or as
    ``(index, cert)`` pairs. Order is irrelevant; nothing here depends on it.

    The returned dict carries one block per entry of ``PREDICATES`` under ``predicates``, plus the
    roll-up: ``constraining_entries`` is the size of the union of every predicate's ``usable`` set,
    ``independent_entries`` the part of that union signed by another key, and ``first_claim`` is
    ``True`` exactly when ``constraining_entries`` is 0 — the case in which every consistency
    check on this cert is vacuous whatever it reports.
    """
    pairs: list[tuple[Optional[int], Mapping]] = []
    for item in prior:
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], Mapping)
            and not isinstance(item[0], Mapping)
        ):
            pairs.append((item[0], item[1]))
        elif isinstance(item, Mapping):
            pairs.append((None, item))
    own_id = _cert_id(cert)
    pairs = [(i, c) for i, c in pairs if own_id is None or _cert_id(c) != own_id]

    own_ids = set(own_material(cert))
    own_issuer = _issuer_key(cert)
    blocks: dict[str, dict] = {}
    usable_union: dict[str, Mapping] = {}

    def record(key: str, block: dict, usable_pairs: list[tuple[Optional[int], Mapping]]) -> None:
        blocks[key] = block
        for _, c in usable_pairs:
            cid = _cert_id(c)
            if cid:
                usable_union[cid] = c

    # --- 1. floor_agreement: same weights_sha256 + same battery -------------------------------
    weights, battery = _weights(cert), _battery_id(cert)
    applicable = bool(weights) and bool(battery) and (
        bool(_floor_block(cert)) or _channels(cert) is not None
    )
    matched = [
        (i, c)
        for i, c in pairs
        if weights is not None
        and battery is not None
        and _weights(c) == weights
        and _battery_id(c) == battery
    ]

    def _floor_usable(c: Mapping) -> bool:
        return bool(_floor_block(c)) or _channels(c) is not None

    blocked: list[str] = []
    if not applicable:
        if not weights:
            blocked.append("this cert's subject carries no weights_sha256")
        if not battery:
            blocked.append("this cert names no battery")
        if not (_floor_block(cert) or _channels(cert) is not None):
            blocked.append("this cert carries neither a noise_floor nor channel values")
    block = _predicate_result(
        "floor_agreement",
        PREDICATES[0][1],
        applicable=applicable,
        scope={"weights_sha256": weights, "battery": battery},
        matched=matched,
        own_ids=own_ids,
        usable_test=_floor_usable,
        own_issuer=own_issuer,
        blocked_by=blocked,
    )
    record(
        "floor_agreement",
        block,
        [(i, c) for i, c in matched if _cert_id(c) not in own_ids and _floor_usable(c)],
    )

    # --- 2. snapshot_agreement: same (hf_repo, revision) ---------------------------------------
    pair = _repo_revision(cert)
    applicable = pair is not None and _has_a2(cert)
    matched = [(i, c) for i, c in pairs if pair is not None and _repo_revision(c) == pair]
    blocked = []
    if pair is None:
        blocked.append("this cert's subject carries no (hf_repo, revision)")
    elif not _has_a2(cert):
        blocked.append("this cert's subject carries none of the Appendix A.2 content hashes")
    block = _predicate_result(
        "snapshot_agreement",
        PREDICATES[1][1],
        applicable=applicable,
        scope={"hf_repo": pair[0] if pair else None, "revision": pair[1] if pair else None},
        matched=matched,
        own_ids=own_ids,
        usable_test=_has_a2,
        own_issuer=own_issuer,
        blocked_by=blocked,
    )
    record(
        "snapshot_agreement",
        block,
        [(i, c) for i, c in matched if _cert_id(c) not in own_ids and _has_a2(c)],
    )

    # --- 3. schedule: the noise plan this cert names --------------------------------------------
    #
    # The plan is own material for every cert that names it, so `available` is 0 by construction
    # for the runs and the floor under one plan. The number that carries the finding is `usable`
    # over the matched plans: a plan that fixes no schedule constrains no reference choice even
    # when it is on file. Both are reported.
    plan = _plan_id(cert)
    applicable = plan is not None
    matched = [(i, c) for i, c in pairs if plan is not None and _cert_id(c) == plan]
    plans_without_schedule = [
        (i, c) for i, c in matched if _is_plan(c) and not _plan_has_schedule(c)
    ]
    blocked = []
    if plan is None:
        blocked.append("this cert names no noise plan")
    elif not matched:
        blocked.append("the plan this cert names is not among the prior entries")
    if plans_without_schedule:
        blocked.append(
            "the plan on file fixes the factors, their levels and the run count, and names no "
            "schedule: it does not fix which assignment each run takes, so the reference run is "
            "chosen after the plan is logged"
        )
    block = _predicate_result(
        "schedule",
        PREDICATES[2][1],
        applicable=applicable,
        scope={"plan": plan},
        matched=matched,
        own_ids=own_ids,
        usable_test=_plan_has_schedule,
        own_issuer=own_issuer,
        blocked_by=blocked,
    )
    # A plan is the cert's own material, so it is reported under `own`; the schedule check is
    # still worth its own count, because a plan that carried a schedule would bind the issuer to
    # a commitment made before the data existed even though the issuer wrote it.
    block["plans_on_file"] = len(matched)
    block["plans_carrying_a_schedule"] = sum(1 for _, c in matched if _plan_has_schedule(c))
    record(
        "schedule",
        block,
        [(i, c) for i, c in matched if _cert_id(c) not in own_ids and _plan_has_schedule(c)],
    )

    # --- 4. determinism: same subject, same batch level -----------------------------------------
    batch = _batch_level(cert)
    applicable = bool(weights) and bool(battery) and batch is not None and _channels(cert) is not None
    matched = [
        (i, c)
        for i, c in pairs
        if weights is not None
        and battery is not None
        and batch is not None
        and _weights(c) == weights
        and _battery_id(c) == battery
        and _batch_level(c) == batch
    ]

    def _det_usable(c: Mapping) -> bool:
        return _channels(c) is not None

    blocked = []
    if batch is None:
        blocked.append("this cert records no batch level")
    elif _channels(cert) is None:
        blocked.append("this cert carries no channel values to reproduce")
    block = _predicate_result(
        "determinism",
        PREDICATES[3][1],
        applicable=applicable,
        scope={"weights_sha256": weights, "battery": battery, "batch_size": batch},
        matched=matched,
        own_ids=own_ids,
        usable_test=_det_usable,
        own_issuer=own_issuer,
        blocked_by=blocked,
    )
    record(
        "determinism",
        block,
        [(i, c) for i, c in matched if _cert_id(c) not in own_ids and _det_usable(c)],
    )

    # --- 5. issuer_history ----------------------------------------------------------------------
    applicable = own_issuer is not None
    matched = [(i, c) for i, c in pairs if own_issuer is not None and _issuer_key(c) == own_issuer]
    blocked = ["this predicate constrains a party only against itself"] if applicable else [
        "this cert names no issuer key"
    ]
    block = _predicate_result(
        "issuer_history",
        PREDICATES[4][1],
        applicable=applicable,
        scope={"issuer": own_issuer},
        matched=matched,
        own_ids=own_ids,
        usable_test=lambda c: True,
        own_issuer=own_issuer,
        blocked_by=blocked,
    )
    # `independent` is 0 here by definition: the scope IS the issuer's own key. Stated rather
    # than left for the reader to notice.
    block["independent"] = 0
    record(
        "issuer_history",
        block,
        [(i, c) for i, c in matched if _cert_id(c) not in own_ids],
    )

    independent = [
        c
        for c in usable_union.values()
        if own_issuer is None or (_issuer_key(c) is not None and _issuer_key(c) != own_issuer)
    ]
    out = {
        "cert": _entry_label(cert, index),
        "prior_entries": len(pairs),
        "own_material": sorted(own_ids),
        "own_material_present": sum(1 for _, c in pairs if _cert_id(c) in own_ids),
        "issuer": own_issuer,
        "predicates": blocks,
        "constraining_entries": len(usable_union),
        "independent_entries": len(independent),
        "first_claim": len(usable_union) == 0,
    }
    out["verdict"] = "unconstrained" if out["first_claim"] else "constrained"
    out["disclosure"] = disclosure_line(out)
    return out


def disclosure_line(result: Mapping) -> str:
    """The one line this census exists to put beside a verdict."""
    n = result.get("constraining_entries")
    ind = result.get("independent_entries")
    prior = result.get("prior_entries")
    own = result.get("own_material_present")
    if n == 0:
        return (
            f"constraint: 0 of {prior} prior entries could contradict this claim "
            f"({own} are its own material); first claim, every consistency check on it is vacuous"
        )
    return (
        f"constraint: {n} of {prior} prior entries could contradict this claim "
        f"({ind} signed by another key, {own} are its own material)"
    )


# --------------------------------------------------------------------------- reading a log


def read_log_entries(root: str | Path) -> list[tuple[int, dict]]:
    """Every stored entry under ``<root>/entries``, as ``(index, cert)`` in index order.

    Reads bytes and parses JSON; it does not verify a signature, an id or a tree head. A census is
    a disclosure computed alongside verification, not a substitute for it — run
    ``python -m styxx.v8 log verify-cert`` for that.
    """
    base = Path(root)
    entries_dir = base / "entries" if (base / "entries").is_dir() else base
    out: list[tuple[int, dict]] = []
    for path in sorted(entries_dir.glob("*/*.json")) + sorted(entries_dir.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        try:
            cert = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cert, dict):
            continue
        try:
            index = int(path.stem)
        except ValueError:
            index = len(out)
        out.append((index, cert))
    out.sort(key=lambda t: t[0])
    return out


def census_log(root: str | Path) -> dict:
    """The census of every entry in the log at ``root``, each against the entries below it."""
    entries = read_log_entries(root)
    per_entry = [
        census(cert, entries[:position], index=index)
        for position, (index, cert) in enumerate(entries)
    ]
    return {
        "root": str(Path(root)),
        "entries": len(entries),
        "per_entry": per_entry,
        "unconstrained_entries": sum(1 for r in per_entry if r["first_claim"]),
        "max_constraining_entries": max((r["constraining_entries"] for r in per_entry), default=0),
        "independent_entries_anywhere": max(
            (r["independent_entries"] for r in per_entry), default=0
        ),
    }
