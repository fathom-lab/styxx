"""Four log-integrity defects, each pinned as the refusal it now earns.

All four are predicates over certs the log already holds — class one of
``papers/v8/THE_BOUNDARY_2026_09_09.md``'s partition, the class the boundary paper calls "a
backlog, and all of them work". Every test below is an attack that was run and passed, written
back as the refusal.

* **A-META** — the floor census was written into ``<index>.meta.json``, outside the Merkle tree
  and signed by nothing, and ``verify_entry`` compared exactly two fields of it (``id`` and
  ``index``). On the published log the attacker rewrote the census in place — ``executions``
  3 -> 5, ``pairs_same_execution`` 2 -> 0, ``all_channels_zero`` true -> false — and
  ``verify_entry`` returned ``ok``, the root was unchanged because no leaf moved, and ``mirror``
  reported ``verified: True, tamper: []``. ``Log.floor_census`` re-derives the whole block
  exactly and nothing called it.
* **L10c** — ``Log.issuers()`` returned ``None`` when ``keys/issuers.json`` was absent and append
  step 2 read ``if roster is not None``. The control is two lines: with a roster present a rogue
  key is refused; delete the file and the same rogue key appends.
* **L7** — the seven published entries replayed verbatim into a log built on a DIFFERENT key.
  All seven appended, byte-identical, same Merkle root, different ``log_id``, so "this cert is in
  the log" was not a checkable statement.
* **L6** — nothing compared the noise plan's log index against its runs'. Section 7.2 says the
  log index is the proof of order; a plan appended after the runs it governs preregisters
  nothing.

Two limitations are pinned here as tests rather than hidden, because a repair that could not be
made is a result: ``test_an_unbound_cert_still_replays_verbatim`` and
``test_the_append_time_metadata_fields_are_not_re_derivable``.

Nothing here skips.
"""
from __future__ import annotations

import copy
import json

import pytest

from styxx.v8 import cli as climod
from styxx.v8 import floor as floormod
from styxx.v8.jcs import canonical_bytes
from styxx.v8.log import (
    APPEND_TIME_META_KEYS,
    OPEN_POLICY,
    AppendRefused,
    Log,
    mirror,
    verify_entry,
)
from tests import v8_fixtures as F

ISSUER_SEED, ISSUER_PUB = F.keypair("issuer")
OTHER_SEED, OTHER_PUB = F.keypair("other")
LOG_SEED, LOG_PUB = F.keypair("log-key")
SECOND_LOG_SEED, SECOND_LOG_PUB = F.keypair("second-log-key")

TS = "2026-09-09T12:00:00Z"

# Run 0 .. run 4 batch sizes; the plan declares batch_size 8|4, so the runs really do differ in
# execution and the floor is a floor of more than one computation.
SIZES = (8, 4, 8, 4, 8)
ENVIRONMENT = copy.deepcopy(F.WEIGHTS_SUBJECT["environment"])


# ----------------------------------------------------------------- helpers

def roster(*labels: str) -> list[dict]:
    return [
        {
            "name": F.ISSUER_NAME,
            "key": F.public_key(label),
            "from_index": 0,
            "retired_at_index": None,
        }
        for label in (labels or ("issuer",))
    ]


def fresh(root, *labels: str, public: bytes = LOG_PUB) -> Log:
    return Log.init(root, public, roster(*labels))


def plan_cert(*, runs: int = 5) -> dict:
    """A noise-plan prereg fixing R, one derivable factor, and the environment section 5.4 needs."""
    return F.make_cert(
        "prereg",
        body={
            "kind": "noise-plan",
            "runs": runs,
            "nuisance": [{"factor": "batch_size", "values": ["8", "4"]}],
            "environment": copy.deepcopy(ENVIRONMENT),
        },
    )


def _run_cert(battery: dict, k: int, refs: list[dict]) -> dict:
    return F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=SIZES[k])),
        refs=[{"role": "battery", "id": battery["id"]}] + refs,
        body=F.fingerprint_body(n=8, run_index=k, batch_size=SIZES[k]),
    )


def _canonical(battery: dict, plan: dict, runs: list[dict], refs_extra=()) -> dict:
    """The canonical fingerprint over ``runs``, with the coverage lists section 5.4 derives."""
    body = F.fingerprint_body(n=8, run_index=0, batch_size=SIZES[0])
    block = F.noise_floor_block(
        plan=plan["id"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    block["covers"], block["not_covered"] = floormod.plan_coverage(plan["body"])
    body["noise_floor"] = block
    return F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=SIZES[0])),
        refs=(
            [{"role": "battery", "id": battery["id"]}]
            + [{"role": "noise_plan", "id": plan["id"]}]
            + [{"role": "run", "id": c["id"]} for c in runs]
            + [dict(r) for r in refs_extra]
        ),
        body=body,
    )


def ladder(root) -> dict:
    """battery, plan, four run certs, the canonical carrying the floor — appended in 5.1 step 5's order."""
    log = fresh(root)
    battery = F.make_cert("battery", body=F.battery_body("fixed-v1", n=8))
    log.append(battery)
    plan = plan_cert()
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    runs = [_run_cert(battery, k, plan_ref) for k in range(1, 5)]
    for cert in runs:
        log.append(cert)
    canonical = _canonical(battery, plan, runs)
    index = log.append(canonical)
    return {
        "log": log,
        "battery": battery,
        "plan": plan,
        "runs": runs,
        "canonical": canonical,
        "index": index,
    }


def rewrite_meta(log: Log, index: int, **fields) -> None:
    """Edit ``<index>.meta.json`` in place — the attacker's move, in one line."""
    path = log.meta_path(index)
    meta = json.loads(path.read_bytes().decode("utf-8"))
    for key, value in fields.items():
        meta[key] = value
    path.write_bytes((json.dumps(meta, sort_keys=True, indent=2) + "\n").encode("utf-8"))


# ================================================================= A-META


def test_the_honest_ladder_appends_and_every_entry_verifies(tmp_path):
    """The control. Every refusal below differs from this by one edited file or one moved cert."""
    built = ladder(tmp_path / "log")
    log = built["log"]
    assert log.size() == 7  # battery, plan, four runs, the canonical
    for index in range(log.size()):
        ok, why = verify_entry(log, index)
        assert ok, why
        assert log.meta_disagreement(index) == []
    census = log.meta(built["index"])["floor"]
    assert census == log.derived_meta(built["index"])["floor"]
    assert census["executions"] == 2 and census["pairs_same_execution"] == 4


def test_a_rewritten_floor_census_is_caught_and_the_root_never_moved(tmp_path):
    """A-META, exactly as it was run: the three census fields the attacker edited.

    The Merkle root is asserted UNCHANGED on purpose. That is the whole shape of the finding —
    the census lives outside the tree, so no leaf moves, no proof breaks, and every check that
    looks at the tree keeps saying yes. What has to change is who computes the census.
    """
    built = ladder(tmp_path / "log")
    log, index = built["log"], built["index"]
    before = log.root()

    rewrite_meta(log, index, floor={
        **log.meta(index)["floor"],
        "executions": 5,
        "pairs_same_execution": 0,
        "all_channels_zero": False,
    })

    assert log.root() == before  # no leaf moved; this is the point of the attack
    reasons = log.meta_disagreement(index)
    assert len(reasons) == 1
    assert "metadata says floor =" in reasons[0]
    assert "'executions': 5" in reasons[0]
    assert "'executions': 2" in reasons[0]  # the derived value is named beside the stored one
    ok, why = verify_entry(log, index)
    assert not ok
    assert "metadata says floor" in why


def test_mirror_reports_the_census_disagreement_and_is_not_verified(tmp_path):
    """The published log's ``mirror`` said ``verified: True, tamper: []`` over this exact edit."""
    built = ladder(tmp_path / "log")
    log, index = built["log"], built["index"]
    sth = log.sth(LOG_SEED, TS)
    rewrite_meta(log, index, floor={**log.meta(index)["floor"], "pairs_same_execution": 0})

    report = mirror(log.path, tmp_path / "dst", LOG_PUB, pinned_sth=sth)
    assert report["verified"] is False
    assert any("pairs_same_execution" in line for line in report["metadata"])
    assert any(f"entry {index}" in line for line in report["tamper"])


def test_a_census_beside_an_entry_that_has_no_floor_is_a_disagreement(tmp_path):
    """The other direction: a census invented for a cert whose bytes carry no ``noise_floor``.

    The derivation gives none, so a file that carries one is reporting a measurement the entries
    do not contain — which is what an unsigned file outside the tree can always do, and what
    re-deriving on read is for.
    """
    built = ladder(tmp_path / "log")
    log = built["log"]
    run_index = log.find(built["runs"][0]["id"])
    rewrite_meta(log, run_index, floor={"executions": 5, "pairs_same_execution": 0})

    reasons = log.meta_disagreement(run_index)
    assert len(reasons) == 1
    assert "carries 'floor'" in reasons[0] and "derive none" in reasons[0]
    assert verify_entry(log, run_index)[0] is False


def test_public_is_derived_from_the_certs_and_not_from_the_neighbouring_metadata(tmp_path):
    """``public`` (section 2.6) used to be read off other entries' metadata files.

    ``_public_for`` walked ``refs`` and consulted ``meta(at)["public"]`` at each one, so a single
    edited file flipped the transitive answer for every cert stacked on it — and ``public`` gates
    section 4.5's redaction refusal at append. It is computed from the certs now.
    """
    log = fresh(tmp_path / "log")
    pool = F.make_cert("battery", body=F.battery_body("pool-v1"))
    log.append(pool)
    secret = F.make_cert(
        "battery", body={**F.battery_body("fixed-v1", n=3), "redacted": True}
    )
    log.append(secret)
    assert log.meta(1)["public"] is False

    rewrite_meta(log, 1, public=True)  # the operator says the redacted battery is public
    assert log.derived_meta(1)["public"] is False
    assert verify_entry(log, 1)[0] is False

    # and the redaction rule still refuses, because it no longer asks the file
    later = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=secret["id"]),
        refs=[{"role": "battery", "id": secret["id"]}],
        body=F.fingerprint_body(),
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(later)
    assert exc.value.reason.startswith("redaction:")


def test_metadata_cannot_point_a_cert_id_at_another_leaf(tmp_path):
    """``_id_map`` preferred the metadata's ``id`` and fell back to the entry only when it was absent.

    So an edited file re-pointed an id at another index, and every ref resolution, floor
    recomputation and challenge check that runs through ``find`` followed it. It reads the cert.
    """
    built = ladder(tmp_path / "log")
    log = built["log"]
    battery_id = built["battery"]["id"]
    assert log.find(battery_id) == 0

    rewrite_meta(log, 0, id=built["plan"]["id"])
    log._id_map_cache = None
    assert log.find(battery_id) == 0
    assert log.find(built["plan"]["id"]) == 1
    assert verify_entry(log, 0)[0] is False


def test_the_append_time_metadata_fields_are_not_re_derivable(tmp_path):
    """A LIMITATION, pinned rather than hidden.

    ``baseline_gap`` is computed from the log as it stood at append: it resolves the previous
    comparable fingerprint by scanning the whole log, so an entry that had no comparable
    predecessor when it was appended acquires one as soon as a later comparable fingerprint lands.
    Re-deriving it on read therefore gives a different and strictly later answer than the file
    records, and ``meta_disagreement`` passes over it. Those fields are exactly as rewritable as
    the census was before this repair. Closing it needs the derivation to be index-relative, which
    is a change to the field's own definition and not to the checker in front of it.
    """
    built = ladder(tmp_path / "log")
    log = built["log"]
    assert APPEND_TIME_META_KEYS == ("baseline_gap",)
    for index in range(log.size()):
        stored = log.meta(index)
        if "baseline_gap" not in stored:
            continue
        rewrite_meta(log, index, baseline_gap={"previous_index": 999})
        assert log.meta_disagreement(index) == []  # the hole, stated
        assert verify_entry(log, index)[0] is True
        break


# ================================================================= L10c


def test_a_log_with_no_admission_policy_refuses_every_append(tmp_path):
    """L10c, and it is the control the attacker ran, both halves.

    With a roster on file a rogue key is refused. Delete the file and the SAME key used to append
    at exit 0, because ``if roster is not None`` reads a missing configuration as a permission.
    """
    log = fresh(tmp_path / "log")
    rogue = F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1"))
    with pytest.raises(AppendRefused) as exc:
        log.append(rogue)
    assert exc.value.reason.startswith("issuer:")
    assert "not in the roster" in exc.value.reason

    (log.keys_dir / "issuers.json").unlink()
    assert log.issuer_policy()["policy"] == "absent"
    assert log.issuers() is None
    with pytest.raises(AppendRefused) as exc:
        log.append(rogue)
    assert exc.value.reason.startswith("issuer:")
    assert "states no admission policy" in exc.value.reason
    assert log.size() == 0


def test_an_empty_roster_admits_nobody_and_that_is_a_policy(tmp_path):
    """An empty array is not the same statement as a missing file, and never was.

    ``log init`` with no ``--issuer`` writes ``[]``: a log that admits nobody. That is a decision
    an operator can make and defend. "The file is not there" is not.
    """
    log = Log.init(tmp_path / "log", LOG_PUB, [])
    assert log.issuer_policy() == {"policy": "roster", "issuers": []}
    with pytest.raises(AppendRefused) as exc:
        log.append(F.make_cert("battery", body=F.battery_body("pool-v1")))
    assert "not in the roster" in exc.value.reason


def test_an_intentionally_open_log_is_a_marker_and_admits_any_key(tmp_path):
    """What an open log looks like now: an explicit marker with a reason, not an absent file."""
    log = Log.init(
        tmp_path / "log",
        LOG_PUB,
        {"policy": OPEN_POLICY, "reason": "v1 open submission, section 8.3"},
    )
    assert log.issuer_policy() == {
        "policy": OPEN_POLICY,
        "reason": "v1 open submission, section 8.3",
    }
    assert log.issuers() is None  # an open log has no roster, and that is not an admission rule
    assert log.append(
        F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1"))
    ) == 0

    report = mirror(log.path, tmp_path / "dst", LOG_PUB)
    assert report["issuer_policy"]["policy"] == OPEN_POLICY
    # an open log is a disclosure, not an accusation: EXTERNAL-1's lesson about predicates that
    # fire on a configuration rather than on a fact
    assert report["misbehaviour"] == []


def test_a_policy_this_version_cannot_enforce_refuses_rather_than_admits(tmp_path):
    """Every branch decides. An unreadable or unknown policy is not a quiet pass."""
    log = fresh(tmp_path / "log")
    (log.keys_dir / "issuers.json").write_bytes(b'{"policy": "allow-listed-later"}')
    assert log.issuer_policy()["policy"] == "unknown"
    with pytest.raises(AppendRefused) as exc:
        log.append(F.make_cert("battery", body=F.battery_body("pool-v1")))
    assert "states no admission policy" in exc.value.reason

    (log.keys_dir / "issuers.json").write_bytes(b"{not json")
    assert log.issuer_policy()["policy"] == "unreadable"
    with pytest.raises(AppendRefused):
        log.append(F.make_cert("battery", body=F.battery_body("pool-v1")))


def test_init_refuses_a_policy_it_could_not_enforce(tmp_path):
    with pytest.raises(ValueError):
        Log.init(tmp_path / "log", LOG_PUB, {"policy": "maybe"})


def test_the_cli_can_only_reach_an_open_log_by_asking_for_one(tmp_path):
    """``log init`` writes a policy on every path; ``--open-issuers`` is the only open one.

    Without it and without ``--issuer`` the log gets an empty roster and admits nobody, which is
    the safe reading of "the operator named no keys". What is no longer reachable through any
    flag is a log with no policy file at all.
    """
    pem = tmp_path / "log.pem"
    assert climod.run(["key", "generate", "--out", str(pem)])[0] == 0

    code, payload = climod.run(
        ["log", "init", "--log", str(tmp_path / "closed"), "--key", str(pem)]
    )
    assert code == 0 and payload["issuer_policy"] == "roster"
    assert Log(tmp_path / "closed").issuer_policy() == {"policy": "roster", "issuers": []}

    code, payload = climod.run(
        ["log", "init", "--log", str(tmp_path / "open"), "--key", str(pem), "--open-issuers"]
    )
    assert code == 0 and payload["issuer_policy"] == OPEN_POLICY
    assert Log(tmp_path / "open").issuer_policy()["policy"] == OPEN_POLICY

    code, payload = climod.run(
        [
            "log", "init", "--log", str(tmp_path / "both"), "--key", str(pem),
            "--open-issuers", "--issuer", f"lab={F.public_key('issuer')}",
        ]
    )
    assert code != 0
    assert "two different admission policies" in payload["error"]


# ================================================================= L7


def bound(cert_type: str, log_id: str, **over) -> dict:
    """A cert carrying the section 8.1 binding in its signed body."""
    body = dict(over.pop("body", F.battery_body("pool-v1")))
    body["log_hint"] = {"log_id": log_id}
    return F.make_cert(cert_type, body=body, **over)


def test_a_cert_that_names_its_log_appends_to_that_log(tmp_path):
    """The control: the binding is a fact about this log, so this log takes it."""
    log = fresh(tmp_path / "log")
    cert = bound("battery", log.log_id())
    assert log.append(cert) == 0
    assert verify_entry(log, 0)[0] is True
    report = mirror(log.path, tmp_path / "dst", LOG_PUB)
    assert report["log_binding"] == {"bound": 1, "unbound": 0}


def test_a_bound_cert_replayed_into_a_log_on_another_key_is_refused(tmp_path):
    """L7. The seven published entries were replayed verbatim into a log on a different key.

    Byte-identical entries, the same Merkle root, a different ``log_id`` — so "this cert is in
    the log" named no log. A bound cert is now refused by every log but the one it names.
    """
    first = fresh(tmp_path / "first")
    cert = bound("battery", first.log_id())
    first.append(cert)

    second = fresh(tmp_path / "second", public=SECOND_LOG_PUB)
    assert second.log_id() != first.log_id()
    with pytest.raises(AppendRefused) as exc:
        second.append(copy.deepcopy(cert))
    assert exc.value.reason.startswith("log_hint:")
    assert first.log_id() in exc.value.reason and second.log_id() in exc.value.reason
    assert second.size() == 0


def test_a_bound_entry_seated_in_the_wrong_log_is_caught_on_read(tmp_path):
    """The same predicate where the gate never ran: a clone, or a log that pre-dates the rule.

    The entry bytes are written straight into the second log's directory with correct derived
    metadata, so nothing else about it is wrong — the id recomputes, the signature verifies, the
    leaf is the hash of these bytes. Only the binding is false.
    """
    first = fresh(tmp_path / "first")
    cert = bound("battery", first.log_id())
    first.append(cert)

    second = fresh(tmp_path / "second", public=SECOND_LOG_PUB)
    second.entry_path(0).parent.mkdir(parents=True, exist_ok=True)
    second.entry_path(0).write_bytes(canonical_bytes(cert))
    second.meta_path(0).write_bytes(
        (
            json.dumps(
                {
                    "index": 0,
                    "id": cert["id"],
                    "type": "battery",
                    "public": True,
                    "appended_at": TS,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    )
    second._id_map_cache = None
    assert second.meta_disagreement(0) == []  # everything else about the entry is in order
    ok, why = verify_entry(second, 0)
    assert not ok
    assert "log_hint names log" in why


def test_a_malformed_binding_is_refused_rather_than_ignored(tmp_path):
    log = fresh(tmp_path / "log")
    for value in ("not-an-object", {"log_id": "sha256:zz"}, {"locations": ["mirror"]}):
        body = dict(F.battery_body("pool-v1"))
        body["log_hint"] = value
        with pytest.raises(AppendRefused) as exc:
            log.append(F.make_cert("battery", body=body))
        assert exc.value.reason.startswith("log_hint:")


def test_an_unbound_cert_still_replays_verbatim(tmp_path):
    """A LIMITATION, pinned rather than hidden.

    ``log_hint`` is optional, because requiring it would refuse every cert already signed and
    because a cert cannot name a log that does not exist yet. So an unbound cert is exactly as
    replayable as it was before this repair, and the honest statement of what L7 buys is "a cert
    MAY be bound, and a bound cert is refused by the wrong log". ``mirror`` reports the count, so
    a reader can see how much of a corpus is bound.
    """
    first = fresh(tmp_path / "first")
    plain = F.make_cert("battery", body=F.battery_body("pool-v1"))
    first.append(plain)

    second = fresh(tmp_path / "second", public=SECOND_LOG_PUB)
    assert second.append(copy.deepcopy(plain)) == 0
    assert first.root() == second.root()  # identical leaves under two different log_ids
    assert first.log_id() != second.log_id()

    report = mirror(second.path, tmp_path / "dst", SECOND_LOG_PUB)
    assert report["log_binding"] == {"bound": 0, "unbound": 1}


# ================================================================= L6


def test_a_noise_plan_appended_after_its_own_runs_is_refused(tmp_path):
    """L6. Section 7.2: the log index is the proof of order, and nothing compared the two.

    Section 5.1 step 5 (ii) has each run reference the plan under role ``noise_plan``, and refs
    resolve backwards, so a run carrying that ref cannot precede its plan. A run cert that omits
    the ref carries no such constraint: chain the runs by ``previous`` to satisfy section 5.5,
    append them, mint the plan afterwards, then the canonical that names both. The floor then
    rests on a preregistration written once the numbers were in.
    """
    log = fresh(tmp_path / "log")
    battery = F.make_cert("battery", body=F.battery_body("fixed-v1", n=8))
    log.append(battery)

    runs: list[dict] = []
    for k in range(1, 5):
        extra = [{"role": "previous", "id": runs[-1]["id"]}] if runs else []
        cert = _run_cert(battery, k, extra)
        log.append(cert)
        runs.append(cert)

    plan = plan_cert()
    plan_at = log.append(plan)
    assert plan_at > log.find(runs[0]["id"])  # the plan is ABOVE the runs it governs

    canonical = _canonical(
        battery, plan, runs, refs_extra=[{"role": "previous", "id": runs[-1]["id"]}]
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert reason.startswith("floor:")
    assert "the log index is the proof of order" in reason
    assert f"index {plan_at}" in reason
    assert "preregisters nothing" in reason


def test_the_ordered_ladder_passes_the_same_predicate(tmp_path):
    """The control, run as the predicate itself so it cannot pass for another reason."""
    built = ladder(tmp_path / "log")
    log, canonical = built["log"], built["canonical"]
    assert log.find(built["plan"]["id"]) < min(log.find(c["id"]) for c in built["runs"])
    log._check_floor_plan_precedes_its_runs(canonical)  # returns: no refusal to make


def test_the_order_rule_does_not_claim_the_plan_was_written_first(tmp_path):
    """What L6 does NOT establish, kept in front of a reader.

    Section 8.6 already says the log cannot tell a late-logged prereg from an early one. The index
    orders the APPENDS, not the computations, so a party that ran everything first and then
    appended the plan, the runs and the canonical in section 5.1 step 5's order is not caught by
    this rule and cannot be by any rule over these bytes. What is refused is narrower and exact:
    a floor resting on runs the log accepted before it accepted their plan.
    """
    built = ladder(tmp_path / "log")
    log = built["log"]
    plan_at = log.find(built["plan"]["id"])
    assert all(plan_at < log.find(c["id"]) for c in built["runs"])
    assert log.meta(plan_at)["appended_at"] <= log.meta(built["index"])["appended_at"]
