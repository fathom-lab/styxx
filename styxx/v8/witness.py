"""styxx.v8.witness — a mirror that remembers, and the exact statement of what it cannot reach.

``log.mirror`` is a pure function of one directory: it copies a log, re-derives every claim the
bytes make, and returns. It holds nothing between calls. So a reader who mirrored a seven-entry
log on Monday and mirrors the same location on Tuesday, after the operator deleted two entries and
every tree head and re-sealed at five, gets ``verified: True, entries: 5, tamper: [],
misbehaviour: []`` — a five-entry log and a truncated seven-entry log are the same bytes, and
nothing in the Monday run survived to say otherwise.

That is L11, reproduced in this session on a seven-entry log built by the fixtures — not on
`papers/v8/first_verdict_2026_09_09/log` itself, whose log key this process does not hold; what
was checked against the published bytes is that they are the unwitnessed case the rest of this
docstring is about. ``mirror`` already knows how to catch it — hand it the Monday head as
``pinned_sth`` and it says
``sth <pinned>: covers 7 entries, the mirror holds 5`` and clears ``verified`` — so the missing
piece was never the predicate. It was **persistence**: nothing kept the head.

``watch()`` is that persistence and nothing more. It calls ``log.mirror``, then compares the copy
against every head this store has seen before, and writes what it saw back. What it buys is
stated exactly in "What this reaches" below; what it does not buy is stated at greater length in
"What no internal check reaches", because that half is the deliverable and it is not repairable
here.

The store
---------

One JSON file, keyed by ``log_id``. Per log it holds every distinct head ever observed (as
``(tree_size, root_hash)``), each with the mirror's own clock at first and last sight and a
``source`` of ``observed`` (copied out of the operator's own ``sth/``) or ``pinned`` (handed in
through ``pinned_sth``, i.e. obtained through a channel the operator does not control); the
sha256 of ``keys/issuers.json`` and the policy derived from it; and a run count. Nothing in it is
signed by anybody. It is the witness's own notebook.

**The store must live where the log operator cannot write.** If the operator can edit it, every
sentence below is void — the retained head is then as forgeable as the head it is compared
against, and this module has moved the trust boundary nowhere. That condition is not checkable
from inside this process and is therefore a deployment requirement, stated here rather than
enforced.

What this reaches
-----------------

Each item is a predicate over bytes: the copy on disk today, against bytes this store recorded on
an earlier run. Every one was demonstrated as a test in ``tests/test_v8_witness.py``.

* **retracted** — a retained head covers more entries than the copy now holds. This is L11: the
  log published N and now holds M < N. It clears ``verified``.
* **rewritten** — a retained head's tree_size is within the copy, and the copy's entries root to
  something else at that size. History moved under a head that was published. Clears ``verified``.
* **inconsistent** — the RFC 6962 consistency proof from a retained head to the copy's largest
  head does not verify. Clears ``verified``. **It reaches nothing on its own, and that is written
  down rather than dressed up**: every construction found for it is also caught by *rewritten*
  (the retained size roots differently) or by ``mirror``'s own "the entries do not reproduce the
  signed root" (the current head is forged), because a tree whose retained root reproduces and
  whose current root reproduces is one tree and its consistency proof verifies by construction.
  It is kept as a second arithmetic path over the same bytes, on the ground that a disagreement
  between the two would itself be a finding; it is not counted as an attack this module closes.
* **withdrawn** — a retained head the operator no longer publishes in ``sth/``, whose root the
  entries still reproduce. A DISCLOSURE, not an accusation: nothing the entries say is
  contradicted, and the witness still holds the head, so nothing is lost to *this* reader. What is
  gone is the operator's own publication of it, which is a fact about the ending (DURABILITY §4.4)
  and is owed to a reader who arrives later. It does not clear ``verified``.
* **admission_changed** — ``keys/issuers.json``, or the policy derived from it, is not what this
  store last saw. This is the reachable half of L10a and L10c′ and it is only a half: see below.
  A DISCLOSURE. An operator adding an issuer honestly produces it, and no predicate over an
  unsigned file distinguishes that operator from the attacker who wrote the same bytes.
* **late_admitted** — an entry above the last witnessed tree size, signed by a key that was NOT in
  the roster this store held at that observation. A DISCLOSURE for the same reason: rosters grow
  honestly. What it adds over *admission_changed* is the entry index — it names which leaves rest
  on the roster edit rather than only reporting that one happened.

The three that clear ``verified`` are the three where the bytes contradict bytes. The three that
do not are the three where an honest operator produces the same report as a hostile one, and this
lab already has the receipt for what an accuser that fires on honest artifacts costs: EXTERNAL-1
measured a path-claim accusation at 0.23 precision on external agents' pull requests and DISABLED
the class. A checker that refuses everything is as useless as one that refuses nothing.

L10a and L10c′: moved, not closed, and which is which
-----------------------------------------------------

Another agent moved the admission policy into a logged cert (``Log.policy_cert``, C-12 option
(a)). The question this module was asked is whether that closes L10a (append one object to
``keys/issuers.json``) and L10c′ (write the eighteen bytes ``{"policy": "open"}``) **on an
unwitnessed log**. Both halves were run in this session; the answer is not the same on both sides
of that word.

* On a **witnessed** log — one carrying its own signed ``policy`` statement inside the tree —
  both are CLOSED. The rogue append is refused (``policy: contradicted``), and ``mirror`` reports
  ``verified: False`` naming the contradiction. Measured, both edits, both refusals.
* On an **UNWITNESSED** log — one with no policy statement in the tree, which is exactly what
  `papers/v8/first_verdict_2026_09_09/log` is (``policy_cert()`` returns ``None``,
  ``issuer_policy()`` reports ``source: "file", witnessed: false``) — both are OPEN, unchanged.
  Measured: with the roster object appended, a rogue key appends at exit 0 and ``mirror`` reports
  ``verified: True, tamper: [], metadata: []``; with the eighteen-byte marker, the same.

So the repair MOVED the defence rather than closing it: it put the defence behind a statement the
published log does not carry. That is not nothing — ``issuer_policy`` prints ``witnessed: false``,
so a reader of a mirror is told which of the two logs they are holding — but it is a disclosure,
not a defence, and this module's ``admission_changed`` is the only other thing available: it
detects that the file MOVED since a prior observation, and cannot say who moved it or whether the
move was legitimate. A first-time reader of an unwitnessed log gets neither.

What no internal check reaches, and why
---------------------------------------

This is the part that is not repairable and is the reason the module exists in this shape.

**A first observation establishes nothing about the past.** Every check above compares today's
bytes to bytes this store recorded earlier. On the first run there is nothing to compare to, so
``watch`` on a log this store has never seen returns exactly what ``mirror`` returns, with
``witnessed: false`` and ``basis: "first-observation"``. A log truncated before anyone mirrored it
is internally consistent, correctly signed, roots correctly at every size, and is indistinguishable
from a log that was always that size. There is no predicate over its bytes that separates the two,
because the two are the same bytes: the operator wrote the entries, wrote the tree heads over
them, and holds the key that signs both. This is DURABILITY §3.1 and §3.2 and THE_BOUNDARY's
general form in one instance — *a check on bytes an issuer wrote can only ask whether that party
contradicted itself, and a party that does not contradict itself is not caught by asking.*

**What the store adds is retroactive detection, not a date.** A retained head proves that THIS
witness saw those bytes before it saw these. It does not prove when, to anyone who does not
already trust the witness's own clock, because ``first_seen`` is a string this process wrote. Two
witnesses' notebooks agreeing is better evidence than one and is still two assertions. The pin
that cannot be made after the fact is DURABILITY §3.1's, and it is not made here.

**So the property that requires an external witness, stated exactly.** *That a log of tree size N
existed at a stated time* — and therefore that a log now of size M < N was truncated, or that a
roster now admitting key K did not admit it then — requires a head recorded by a party that is not
the log operator, at a time that party can establish independently. Two constructions supply it
and neither is in this module: a dated pin outside the operator's control (§8.5 anchoring, which
§15.5 defers), or a second party who mirrors and publishes heads (§8.4/§8.6, whose count today is
zero — `REPLICATIONS.md` carries one row and it records no outside replication). ``watch`` is the
machinery that makes the second one worth running: without persistence a mirror had nothing to
publish and nothing to compare, and with it the only remaining input is a party that is not the
operator. It supplies no such party.

**And the honest bound on ``pinned``.** A head handed in through ``pinned_sth`` is external in
provenance and still undated: it says the operator signed those bytes, not when. ``basis:
"external-pin"`` is therefore a statement about where the head came from, not about time, and
``watch`` never prints a date it did not get from its own clock.

Proposed spec text
------------------

Offered for §8.4 and §8.6 rather than written into them; the spec is another agent's file this
round. It restates DURABILITY amendment 4 and extends it in the one direction this module makes
available.

    §8.4, added to a mirror's duties: A mirror SHALL retain every tree head it has verified,
    together with the time it first observed the head and whether the head was copied from the
    log or obtained through a channel the log operator does not control. On every later run a
    mirror SHALL compare the log against every retained head, and SHALL report:
    (a) a retained head covering more entries than the log now holds;
    (b) a retained head whose tree size the log now roots differently;
    (c) a retained head for which no consistency proof to the log's current head verifies.
    Each of (a), (b) and (c) is operator misbehaviour and refuses the mirror's verified state.
    A mirror SHALL also report, as disclosures that do not refuse: a retained head the log no
    longer publishes; and any change to the log's admission policy or to keys/issuers.json since
    a prior run, naming every entry appended since that run under a key the retained roster did
    not admit.

    §8.4, and repeated wherever a mirror prints verified: A mirror that holds no head from an
    earlier run and no externally obtained head has checked internal consistency only, and its
    report SHALL say so under its own key rather than as prose. Truncation of a log before its
    first observation is not detectable from the log's bytes by this or any other check the
    specification defines, because the operator writes the entries and the heads over them. This
    is a permanent property of a single-key log with no external pin and belongs beside the
    verdict, not in a limits section.

    §8.6, added to "Does not defend against": truncation, or a roster edit, on a log no second
    party has ever mirrored and no dated pin covers. The remedy is §8.5 anchoring or a second
    mirror, and until one of the two exists the log establishes internal consistency and nothing
    more (§1.5).

Decisions
---------

**``log.mirror`` is called, not reimplemented, and ``log.py`` is not edited.** Every predicate
here is new state around an unchanged function. The module also stays out of
``conformance/v8/mutation_coverage.json``'s pinned set, which holds a sha256 of ``styxx/v8/log.py``
and whose receipt is one agent's to regenerate this round and not this one's.

**The store is never repaired and never pruned by this module.** A head once recorded stays
recorded, because the whole value of the notebook is that it is not rewritten by the party it is
evidence about — the same argument ``mirror`` makes for not refreshing a stale ``meta.json`` in
the copy. A witness that dropped a head it could no longer reproduce would report agreement over
bytes that never agreed.

**A head that no longer verifies under the pinned key is reported and kept.** It means either the
store was edited or the caller passed a different ``--pin``; both are facts about the witness's
own side, so they land under ``store`` rather than under any accusation against the log.

**A store that will not parse is rebuilt, and the rebuild leaves a scar.** There is no way to keep
watching from an unreadable notebook, so ``load_store`` returns an empty one — and carries the
reason forward under ``unreadable``, which is then written into the new file. A store whose
history was lost says so forever, rather than presenting as a witness that simply started today.
That is the same asymmetry ``mirror`` draws between a metadata file that is OLD and one that is
WRONG: the loss is disclosed under its own name and accuses the log of nothing.

**``observed_at`` is a parameter.** The mirror's clock is an assertion (section 8.1's rule for the
STH timestamp, applied to the only other clock in the system), so a test may set it and the report
never treats it as more than a label.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from styxx.v8 import keys
from styxx.v8 import log as logmod

__all__ = [
    "SCHEMA",
    "head_key",
    "load_store",
    "save_store",
    "watch",
]

SCHEMA = "styxx.v8.witness/v1"

# The standing disclosure, printed with every report whatever it found. THE_BOUNDARY's practical
# difference between its two classes is what a reader should DO about them: class one is a
# backlog, class two "belongs in the output beside the verdict, not in a limits section, because
# no future release closes it". These three sentences are that, for this check.
UNREACHABLE = (
    "a log truncated before this store's first observation is byte-indistinguishable from a log "
    "that was always that size: the operator wrote the entries and signed the heads over them",
    "first_seen is this process's own clock and dates nothing to anyone who does not already "
    "trust this witness; the pin that cannot be made retroactively is not made here",
    "an unsigned keys/issuers.json that MOVED is reportable and an unsigned keys/issuers.json "
    "that was always wrong is not; on an unwitnessed log the roster is the operator's assertion",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def head_key(head: dict) -> str:
    """The identity of a head in the store: its size and the root it signs over."""
    return "%d:%s" % (head.get("tree_size", -1), head.get("root_hash", ""))


def _empty_store() -> dict:
    return {"schema": SCHEMA, "logs": {}}


def load_store(path) -> dict:
    """The notebook at ``path``, or an empty one. Never raises: a corrupt store is an empty one
    with the reason kept, because a witness that crashed on its own file would report nothing
    about the log it was pointed at."""
    p = Path(path)
    if not p.is_file():
        return _empty_store()
    try:
        obj = json.loads(p.read_bytes().decode("utf-8"))
    except Exception as exc:
        store = _empty_store()
        store["unreadable"] = f"{type(exc).__name__}: {exc}"
        return store
    if not isinstance(obj, dict) or not isinstance(obj.get("logs"), dict):
        store = _empty_store()
        store["unreadable"] = "the store is not a %s object" % SCHEMA
        return store
    obj.setdefault("schema", SCHEMA)
    return obj


def save_store(path, store: dict) -> None:
    """Write the notebook: UTF-8, LF, no BOM, sorted keys, one trailing newline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(store, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    p.write_bytes(text.encode("utf-8"))


def _file_sha256(path: Path) -> Optional[str]:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _policy_summary(policy: Any) -> dict:
    """The part of ``issuer_policy`` a witness compares across runs: what it admits and on whose
    authority. The roster's key list is kept because that is the object L10a edits."""
    if not isinstance(policy, dict):
        return {"policy": "unreadable", "source": None, "witnessed": False, "keys": []}
    roster = policy.get("issuers")
    return {
        "policy": policy.get("policy"),
        "source": policy.get("source"),
        "witnessed": bool(policy.get("witnessed")),
        "keys": sorted(
            str(e.get("key"))
            for e in roster
            if isinstance(e, dict)
        ) if isinstance(roster, list) else [],
    }


def watch(
    src,
    dst,
    pinned_public: bytes,
    store_path,
    *,
    pinned_sth: Optional[dict] = None,
    observed_at: Optional[str] = None,
) -> dict:
    """Mirror ``src`` into ``dst`` and compare it against every head ``store_path`` has seen.

    Never raises. Returns the ``mirror`` report under ``"mirror"`` plus this module's own keys;
    ``verified`` is the mirror's verified AND nothing in ``retracted``, ``rewritten`` or
    ``inconsistent``. The store is written back before returning, so the run that CATCHES a
    truncation is also the run that records the state it caught it in.
    """
    observed_at = observed_at or _now()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "observed_at": observed_at,
        "log_id": None,
        "witnessed": False,
        "basis": "first-observation",
        "runs": 0,
        "first_seen": observed_at,
        "heads_held": 0,
        "heads_compared": 0,
        "external_pins": 0,
        "retracted": [],
        "rewritten": [],
        "inconsistent": [],
        "withdrawn": [],
        "admission_changed": [],
        "late_admitted": [],
        "store": [],
        "unreachable": list(UNREACHABLE),
        "verified": False,
        "mirror": None,
    }

    mirror_report = logmod.mirror(src, dst, pinned_public, pinned_sth)
    report["mirror"] = mirror_report

    store = load_store(store_path)
    if store.get("unreadable"):
        report["store"].append("store: %s; treated as empty" % store["unreadable"])

    try:
        log = logmod.Log(dst)
        log_id = log.log_id()
    except Exception as exc:
        report["store"].append(f"mirror copy unreadable: {type(exc).__name__}: {exc}")
        report["verified"] = bool(mirror_report.get("verified"))
        return report
    report["log_id"] = log_id

    record = store["logs"].get(log_id)
    if not isinstance(record, dict):
        record = {
            "log_public": None,
            "first_seen": observed_at,
            "last_seen": observed_at,
            "runs": 0,
            "heads": {},
            "admission": [],
        }
    record.setdefault("heads", {})
    record.setdefault("admission", [])
    held: dict[str, dict] = {
        k: v for k, v in record["heads"].items() if isinstance(v, dict)
    }
    report["heads_held"] = len(held)
    report["witnessed"] = bool(held)
    report["first_seen"] = record.get("first_seen", observed_at)
    report["runs"] = int(record.get("runs", 0)) + 1
    report["external_pins"] = sum(1 for h in held.values() if h.get("source") == "pinned")

    try:
        size = log.size()
    except Exception as exc:
        report["store"].append(f"entries unreadable: {type(exc).__name__}: {exc}")
        size = None

    # --------------------------------------------------------------- the retained heads
    local_sizes: set[int] = set()
    current: list[dict] = []
    try:
        for head in log.sths():
            ok, _reason = logmod.verify_sth(head, pinned_public)
            if ok:
                current.append(head)
                local_sizes.add(head["tree_size"])
    except Exception as exc:
        report["store"].append(f"sth/ unreadable: {type(exc).__name__}: {exc}")
    pin_usable = False
    if pinned_sth is not None:
        ok, reason = logmod.verify_sth(pinned_sth, pinned_public)
        if ok:
            current.append(pinned_sth)
            pin_usable = True
            report["external_pins"] += 1
        else:
            # A pin that does not verify is not a pin. `mirror` has already put it under
            # misbehaviour; here it must not be allowed to upgrade `basis`.
            report["store"].append(f"pinned sth: {reason}")

    largest = max(current, key=lambda h: h["tree_size"], default=None)

    for key in sorted(held, key=lambda k: (held[k].get("tree_size", -1), k)):
        head = held[key].get("sth")
        if not isinstance(head, dict):
            report["store"].append(f"retained head {key}: no sth recorded; skipped")
            continue
        ok, reason = logmod.verify_sth(head, pinned_public)
        if not ok:
            # The witness's own side, not the log's: either the store was edited or --pin moved.
            report["store"].append(f"retained head {key}: {reason}; not compared")
            continue
        report["heads_compared"] += 1
        tree_size = head["tree_size"]
        source = held[key].get("source", "observed")
        seen = held[key].get("first_seen", "?")
        if size is None:
            continue
        if tree_size > size:
            report["retracted"].append(
                "head %d (%s, %s at %s): covers %d entries, the log now holds %d -- the log "
                "published entries it no longer holds (L11)"
                % (tree_size, head["root_hash"], source, seen, tree_size, size)
            )
            continue
        try:
            local_root = "sha256:" + log.root(tree_size).hex()
        except Exception as exc:
            report["rewritten"].append(
                "head %d (%s at %s): root not computable: %s: %s"
                % (tree_size, source, seen, type(exc).__name__, exc)
            )
            continue
        if local_root != head["root_hash"]:
            report["rewritten"].append(
                "head %d (%s at %s): the entries now root to %s, this witness recorded %s -- "
                "history moved under a head that was published"
                % (tree_size, source, seen, local_root, head["root_hash"])
            )
            continue
        if tree_size not in local_sizes:
            report["withdrawn"].append(
                "head %d (%s at %s): the entries still reproduce it and the log no longer "
                "publishes it in sth/ -- a fact about the ending, not a contradiction"
                % (tree_size, source, seen)
            )
        if largest is not None and largest["tree_size"] > tree_size:
            try:
                proof = log.consistency(tree_size, largest["tree_size"])
            except Exception as exc:
                report["inconsistent"].append(
                    "head %d -> %d: no proof: %s: %s"
                    % (tree_size, largest["tree_size"], type(exc).__name__, exc)
                )
                continue
            proof["first_root"] = head["root_hash"]
            proof["second_root"] = largest["root_hash"]
            ok, reason = logmod.verify_consistency(head, largest, proof, pinned_public)
            if not ok:
                report["inconsistent"].append(
                    "head %d -> %d: %s" % (tree_size, largest["tree_size"], reason)
                )

    # --------------------------------------------------------------- admission, across runs
    try:
        policy = log.issuer_policy()
    except Exception as exc:
        policy = {"policy": "unreadable", "reason": f"{type(exc).__name__}: {exc}"}
    summary = _policy_summary(policy)
    issuers_sha = _file_sha256(Path(dst) / "keys" / "issuers.json")
    observation = {
        "file_sha256": issuers_sha,
        "policy": summary,
        "first_seen": observed_at,
        "last_seen": observed_at,
    }
    previous = record["admission"][-1] if record["admission"] else None
    if isinstance(previous, dict) and (
        previous.get("file_sha256") != issuers_sha or previous.get("policy") != summary
    ):
        report["admission_changed"].append(
            "keys/issuers.json: this witness recorded %s (policy %s, %d roster key(s), source %s) "
            "at %s and now reads %s (policy %s, %d roster key(s), source %s). The file is unsigned "
            "and outside the tree; a change is reported and cannot be attributed%s"
            % (
                previous.get("file_sha256"),
                previous.get("policy", {}).get("policy"),
                len(previous.get("policy", {}).get("keys", [])),
                previous.get("policy", {}).get("source"),
                previous.get("first_seen"),
                issuers_sha,
                summary["policy"],
                len(summary["keys"]),
                summary["source"],
                "" if summary["witnessed"] else
                " -- and this log states no policy inside its own tree, so nothing else can "
                "attribute it either (L10a, L10c')",
            )
        )
        # Which leaves rest on the edit. A roster grows honestly, so this names entries and
        # accuses nobody.
        last_size = max(
            (h.get("tree_size", 0) for h in held.values() if isinstance(h, dict)), default=0
        )
        old_keys = set(previous.get("policy", {}).get("keys", []))
        old_policy = previous.get("policy", {}).get("policy")
        if size is not None and old_policy == "roster":
            for index in range(last_size, size):
                try:
                    key = log.cert(index).get("issuer", {}).get("key")
                except Exception:
                    continue
                if key is not None and key not in old_keys:
                    report["late_admitted"].append(
                        "entry %d is signed by %s, which was not in the roster this witness held "
                        "at tree size %d" % (index, key, last_size)
                    )

    # --------------------------------------------------------------- write the notebook back
    for head in current:
        key = head_key(head)
        source = "pinned" if (pinned_sth is not None and head == pinned_sth) else "observed"
        entry = held.get(key)
        if isinstance(entry, dict):
            entry["last_seen"] = observed_at
            if entry.get("source") != "pinned":
                entry["source"] = source
        else:
            held[key] = {
                "sth": head,
                "tree_size": head["tree_size"],
                "root_hash": head["root_hash"],
                "source": source,
                "first_seen": observed_at,
                "last_seen": observed_at,
            }
    if previous is None:
        record["admission"].append(observation)
    elif previous.get("file_sha256") == issuers_sha and previous.get("policy") == summary:
        previous["last_seen"] = observed_at
    else:
        record["admission"].append(observation)
    record["heads"] = held
    try:
        record["log_public"] = keys.encode_public(bytes(pinned_public))
    except Exception:
        record["log_public"] = None
    record["last_seen"] = observed_at
    record["runs"] = report["runs"]
    record.setdefault("first_seen", observed_at)
    store["logs"][log_id] = record
    try:
        save_store(store_path, store)
    except Exception as exc:
        report["store"].append(f"store not written: {type(exc).__name__}: {exc}")

    if report["external_pins"] or pin_usable:
        report["basis"] = "external-pin"
    elif report["witnessed"]:
        report["basis"] = "observed-history"
    else:
        report["basis"] = "first-observation"

    report["verified"] = bool(
        mirror_report.get("verified")
        and not report["retracted"]
        and not report["rewritten"]
        and not report["inconsistent"]
    )
    return report
