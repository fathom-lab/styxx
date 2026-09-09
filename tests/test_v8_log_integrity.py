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
* **A-META-ABSENT** — that repair then ran ONE predicate over TWO facts: a key the file does not
  carry was treated exactly like a key it carries wrongly. Metadata written before a derived
  field existed carries nothing for it, so ``mirror`` printed the lab's own untouched published
  log (`papers/v8/first_verdict_2026_09_09/log`) under ``tamper`` with ``verified: False``.
  ABSENT is stale and reported; CONTRADICTED is tamper and refused. EXTERNAL-1 is the receipt for
  what an accuser that fires on honest artifacts costs: 0.23 precision, class disabled.
* **A-META-GAP** — that repair then exempted one key from the comparison BY NAME. ``baseline_gap``
  resolved the previous comparable fingerprint by scanning the whole log, so it answered later on
  read than at append and could not be compared; ``meta_disagreement`` skipped it, and a key the
  checker skips is exactly as rewritable as the census was. The field is index-relative now and
  the exemption is empty.
* **L10c** — ``Log.issuers()`` returned ``None`` when ``keys/issuers.json`` was absent and append
  step 2 read ``if roster is not None``. The control is two lines: with a roster present a rogue
  key is refused; delete the file and the same rogue key appends.
* **L10c residue** — that repair moved the fail-open rather than closing it: ``rm`` was refused
  and ``echo`` was not. ``{"policy": "open"}`` is eighteen bytes into an unsigned file outside the
  tree. The policy can now be a signed entry, the tree beats the file, and a file that contradicts
  the log's own statement refuses every append.
* **L7** — the seven published entries replayed verbatim into a log built on a DIFFERENT key.
  All seven appended, byte-identical, same Merkle root, different ``log_id``, so "this cert is in
  the log" was not a checkable statement.
* **L7 residue** — the enforcement was complete and NOTHING MINTED THE FIELD, so no cert this
  system produced was bound and the replay worked on every one of them. ``--bind-log`` is the
  mint.
* **L6** — nothing compared the noise plan's log index against its runs'. Section 7.2 says the
  log index is the proof of order; a plan appended after the runs it governs preregisters
  nothing.

Two limitations are pinned here as tests rather than hidden, because a repair that could not be
made is a result: ``test_an_unbound_cert_still_replays_verbatim`` (an UNBOUND cert is exactly as
replayable as it was, and the mint does not change that for certs already signed) and
``test_an_unwitnessed_open_marker_still_admits_every_key`` (a log whose tree states no policy
falls back to the file, and requiring the statement is [OPERATOR-GATED] at §8.2).

Nothing here skips.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from styxx.v8 import cli as climod
from styxx.v8 import floor as floormod
from styxx.v8 import keys as keysmod
from styxx.v8.jcs import canonical_bytes
from styxx.v8.log import (
    APPEND_TIME_META_KEYS,
    DERIVED_META_KEYS,
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
    assert log.stale_metadata(run_index) == []  # it states a value; it is not silent


# ================================================================= A-META-ABSENT


def test_absent_is_stale_and_contradicted_is_tamper(tmp_path):
    """BOTH halves of the split, on ONE entry, one edit apart (A-META-ABSENT).

    The A-META repair ran one predicate over two different facts: a key the file does not carry
    was reported exactly like a key it carries wrongly. A metadata file written before a derived
    field existed carries nothing for that field, so the day after the repair landed it accused
    the lab's own untouched published log of tampering.

    The distinction, pinned here in the only way that is convincing — the same file, the same
    entry, the same key, once deleted and once falsified:

    * ABSENT: the derivation is authoritative, the file is a stale cache, and nothing the entries
      say is contradicted. Not tamper, does not refuse, does not clear ``verified``.
    * CONTRADICTED: the file asserts a census the bytes refute. That is the attack, and it stays
      a refusal.
    """
    built = ladder(tmp_path / "log")
    log, index = built["log"], built["index"]
    honest = log.derived_meta(index)["floor"]

    # --- ABSENT: the file predates ``floor_census``, which is the published log's real state.
    stored = json.loads(log.meta_path(index).read_bytes().decode("utf-8"))
    del stored["floor"]
    log.meta_path(index).write_bytes(
        (json.dumps(stored, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )

    assert log.meta_disagreement(index) == []
    stale = log.stale_metadata(index)
    assert len(stale) == 1
    assert "carries no 'floor'" in stale[0] and "stale and not tamper" in stale[0]
    ok, why = verify_entry(log, index)
    assert ok, why

    report = mirror(log.path, tmp_path / "dst-absent", LOG_PUB, pinned_sth=log.sth(LOG_SEED, TS))
    assert report["verified"] is True
    assert report["tamper"] == [] and report["metadata"] == []
    assert any("carries no 'floor'" in line for line in report["stale_metadata"])

    # --- CONTRADICTED: the same key, now stating the adversary's census. One edit apart.
    rewrite_meta(log, index, floor={
        **honest,
        "executions": 5,
        "pairs_same_execution": 0,
        "all_channels_zero": not honest["all_channels_zero"],
    })

    assert log.stale_metadata(index) == []
    reasons = log.meta_disagreement(index)
    assert len(reasons) == 1 and "metadata says floor =" in reasons[0]
    assert verify_entry(log, index)[0] is False

    report = mirror(log.path, tmp_path / "dst-wrong", LOG_PUB, pinned_sth=log.sth(LOG_SEED, TS))
    assert report["verified"] is False
    assert report["stale_metadata"] == []
    assert any("executions" in line for line in report["metadata"])
    assert any(f"entry {index}" in line for line in report["tamper"])


def test_the_published_log_mirrors_verified_with_its_census_reported_stale(tmp_path):
    """THE REGRESSION, on the artifact that was accused: no fixture, the real published bytes.

    `papers/v8/first_verdict_2026_09_09/log` is untouched, published and pushed to a public
    branch. It was minted before ``floor_census`` existed, so none of its seven metadata files
    carries a ``floor`` key, and the A-META repair printed entry 6 under ``tamper`` with
    ``verified: False``.

    This lab has a receipt for what that costs: EXTERNAL-1 measured a path-claim accusation at
    0.23 precision on external agents' pull requests and disabled the class. An accuser that
    fires on honest artifacts is not strict, it is broken — and here the accusation is
    machine-readable and lands on someone else's artifact.
    """
    root = Path(__file__).resolve().parents[1] / "papers/v8/first_verdict_2026_09_09/log"
    assert root.is_dir(), f"the published log is committed at {root} and must not move"

    published = Log(root)
    pinned = keysmod.load_public(root / "keys/log.pub")
    report = mirror(root, tmp_path / "published-mirror", pinned)

    assert report["verified"] is True, report["tamper"] + report["metadata"]
    assert report["tamper"] == []
    assert report["metadata"] == []
    assert any("entry 6" in line and "'floor'" in line for line in report["stale_metadata"])
    for index in range(published.size()):
        ok, why = verify_entry(published, index)
        assert ok, why


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


def _first_gap_index(log: Log) -> int:
    for index in range(log.size()):
        if "baseline_gap" in log.meta(index):
            return index
    raise AssertionError("no entry of this ladder carries a baseline_gap; the attack has no target")


def test_a_rewritten_baseline_gap_is_caught_like_any_other_derived_field(tmp_path):
    """A-META-GAP. The field that was exempt from the check BY NAME, now derived and compared.

    The exemption was real and its reason was true: ``baseline_gap`` resolved the previous
    comparable fingerprint by scanning the WHOLE log, so an entry with no comparable predecessor
    at append acquired one as soon as a later comparable fingerprint landed, and re-deriving on
    read gave a strictly later answer than the file recorded. Comparing it then would have accused
    honest logs. But a key ``meta_disagreement`` skips by name is exactly as rewritable as the
    census was before A-META — an unsigned file outside the tree, no leaf moving, no root
    changing, ``verify_entry`` returning ``ok``.

    The repair is to the FIELD: ``Log.baseline_gap`` is index-relative, so the answer is the one
    the appending index produced and does not move when the log grows. The exemption is therefore
    empty and the field is checked like every other.
    """
    built = ladder(tmp_path / "log")
    log = built["log"]
    assert APPEND_TIME_META_KEYS == ()  # nothing is exempt from the derivation any more
    assert "baseline_gap" in DERIVED_META_KEYS

    index = _first_gap_index(log)
    before = log.root()
    honest = log.meta(index)["baseline_gap"]
    assert log.derived_meta(index)["baseline_gap"] == honest  # the control

    rewrite_meta(log, index, baseline_gap={**honest, "previous_index": 999, "announced": True})

    assert log.root() == before  # no leaf moved; the file is outside the tree, as it always was
    reasons = log.meta_disagreement(index)
    assert len(reasons) == 1
    assert "metadata says baseline_gap =" in reasons[0]
    assert "'previous_index': 999" in reasons[0]
    ok, why = verify_entry(log, index)
    assert not ok
    assert "baseline_gap" in why

    report = mirror(log.path, tmp_path / "dst", LOG_PUB, pinned_sth=log.sth(LOG_SEED, TS))
    assert report["verified"] is False
    assert any("baseline_gap" in line for line in report["metadata"])


def test_the_baseline_gap_a_reader_derives_is_the_one_the_appending_index_produced(tmp_path):
    """WHY the exemption existed, run as the case that used to break it.

    Entry ``i`` is appended when no comparable fingerprint precedes it, so its gap names the
    baseline as of index ``i``. A later comparable fingerprint then lands. Under the whole-log
    scan, re-deriving entry ``i``'s gap picked up that later entry and answered with a baseline
    that did not exist when ``i`` was appended — a moving number, which is why no checker could
    compare it. Bounded below ``i``'s own index it does not move, and that is what makes the
    field checkable rather than the checker lenient.
    """
    built = ladder(tmp_path / "log")
    log = built["log"]
    index = _first_gap_index(log)
    at_append = log.meta(index)["baseline_gap"]
    assert at_append["previous_index"] < index

    # the log grows past that entry: another comparable fingerprint, appended later
    later = _run_cert(built["battery"], 1, [{"role": "previous", "id": built["canonical"]["id"]}])
    grew = log.append(later)
    assert grew > index

    assert log.derived_meta(index)["baseline_gap"] == at_append
    assert log.derived_meta(index)["baseline_gap"]["previous_index"] < index
    assert log.meta_disagreement(index) == []
    for i in range(log.size()):
        ok, why = verify_entry(log, i)
        assert ok, why


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
    assert log.issuer_policy() == {
        "policy": "roster",
        "issuers": [],
        "source": "file",       # nothing in the tree states a policy, so the file is all there is
        "witnessed": False,
    }
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
        "source": "file",
        "witnessed": False,     # the residue: eighteen unsigned bytes, and this says so
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
    assert Log(tmp_path / "closed").issuer_policy() == {
        "policy": "roster", "issuers": [], "source": "file", "witnessed": False,
    }

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


# ================================================================= L10c residue: the eighteen bytes


def policy_statement(log: Log, stated: dict, *, log_id: str = None, label: str = "log-key") -> dict:
    """The log's own admission policy as a signed entry — C-12 option (a), in one helper.

    A ``result`` of kind ``policy``, signed by the LOG's key (``label``), bound to the log it
    speaks for. ``label`` and ``log_id`` are parameters so the two forgeries — someone else's key,
    another log's id — are one argument away from the honest cert.
    """
    return F.make_cert(
        "result",
        issuer_label=label,
        subject={},
        recipe={},
        refs=[],
        body={
            "kind": "policy",
            "issuer_policy": stated,
            "log_hint": {"log_id": log_id if log_id is not None else log.log_id()},
        },
    )


def test_a_log_can_state_its_admission_policy_inside_its_own_tree(tmp_path):
    """The control for the residue repair: the policy as an entry, not as eighteen unsigned bytes.

    ``keys/issuers.json`` is outside the Merkle tree and signed by nothing, so "this log admits
    anybody" was a sentence anyone with write access could write. Signed by the log's own key,
    bound to this log and appended, it has an index, a leaf hash and a signature — and
    ``issuer_policy`` reads it in preference to the file.
    """
    log = fresh(tmp_path / "log")
    statement = policy_statement(log, {"policy": "roster", "issuers": roster()})
    at = log.append(statement)  # the log's own key is not in the roster; the bootstrap admits it
    assert at == 0

    policy = log.issuer_policy()
    assert policy["policy"] == "roster"
    assert policy["source"] == "cert" and policy["witnessed"] is True
    assert policy["cert_index"] == 0 and policy["cert_id"] == statement["id"]

    # THE TREE GOVERNS. Delete the file — L10c's original attack, which refuses every append on a
    # log that states nothing. Here the policy is an entry, so deleting the cache changes nothing:
    # an absent file contradicts no statement (the A-META-ABSENT reading), and the roster still
    # decides.
    (log.keys_dir / "issuers.json").unlink()
    assert log._file_policy()["policy"] == "absent"
    assert log.issuer_policy()["policy"] == "roster"
    assert log.issuer_policy()["source"] == "cert"

    assert log.append(F.make_cert("battery", body=F.battery_body("pool-v1"))) == 1
    with pytest.raises(AppendRefused) as exc:
        log.append(F.make_cert("battery", issuer_label="other", body=F.battery_body("fixed-v1", n=3)))
    assert "not in the roster" in exc.value.reason
    assert log.size() == 2
    for index in range(log.size()):
        ok, why = verify_entry(log, index)
        assert ok, why


def test_an_open_log_can_say_so_inside_the_tree_and_then_it_admits_anybody(tmp_path):
    """The open case as a STATEMENT: "this log admits anybody" with an index and a signature.

    The difference from the marker is not what it permits — both admit every key — but who can
    say it. The file is eighteen bytes anyone with write access can produce; this is the log's own
    key, bound to this log, at a leaf a reader can cite.
    """
    log = Log.init(
        tmp_path / "log", LOG_PUB, {"policy": OPEN_POLICY, "reason": "v1 open submission"}
    )
    statement = policy_statement(
        log, {"policy": OPEN_POLICY, "reason": "v1 open submission"}
    )
    assert log.append(statement) == 0
    policy = log.issuer_policy()
    assert policy["policy"] == OPEN_POLICY and policy["witnessed"] is True
    assert policy["reason"] == "v1 open submission"
    assert log.append(
        F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1"))
    ) == 1
    report = mirror(log.path, tmp_path / "dst", LOG_PUB)
    assert report["issuer_policy"]["source"] == "cert"
    assert report["verified"] is True and report["misbehaviour"] == []


def test_eighteen_bytes_written_into_issuers_json_now_refuse_every_append(tmp_path):
    """THE RESIDUE, run as the attack and pinned as the refusal.

    The first L10c repair moved the fail-open rather than closing it: deleting the file was
    refused, and ``{"policy": "open"}`` — eighteen bytes into an unsigned file outside the tree —
    admitted every key instead. No predicate over an unsigned file tells the operator who wrote it
    from the attacker who did.

    With the policy stated inside the tree, that edit is not a configuration change: it is a file
    contradicting the log's own signature, so the log appends NOTHING until they agree. Refusing
    everything is the only safe reading — the file is the half anyone with write access can
    produce, so the disagreement is evidence about the directory rather than a choice between two
    policies.
    """
    log = fresh(tmp_path / "log")
    log.append(policy_statement(log, {"policy": "roster", "issuers": roster()}))
    assert log.issuer_policy()["policy"] == "roster"
    honest = F.make_cert("battery", body=F.battery_body("pool-v1"))
    assert log.append(honest) == 1

    (log.keys_dir / "issuers.json").write_bytes(b'{"policy": "open"}')
    assert len(b'{"policy": "open"}') == 18

    policy = log.issuer_policy()
    assert policy["policy"] == "contradicted"
    assert policy["cert_index"] == 0
    assert "'open'" in policy["reason"] and "'roster'" in policy["reason"]

    rogue = F.make_cert("battery", issuer_label="other", body=F.battery_body("fixed-v1", n=3))
    with pytest.raises(AppendRefused) as exc:
        log.append(rogue)
    assert exc.value.reason.startswith("issuer:")
    assert "contradicts" in exc.value.reason or "appends nothing" in exc.value.reason
    assert log.size() == 2

    # the roster's own issuer is refused too: this is not "fall back to the stricter policy",
    # it is "this directory disagrees with itself and nothing goes in"
    with pytest.raises(AppendRefused):
        log.append(F.make_cert("battery", body=F.battery_body("fixed-v1", n=4)))

    report = mirror(log.path, tmp_path / "dst", LOG_PUB, pinned_sth=log.sth(LOG_SEED, TS))
    assert report["verified"] is False
    assert any("issuers:" in line for line in report["metadata"])
    assert report["tamper"] == []  # no leaf moved; the entries are exactly as they were


def test_a_policy_statement_signed_by_another_key_is_not_this_logs_policy(tmp_path):
    """The forgery the bootstrap must not admit: the shape without the log's signature.

    ``_is_own_policy_statement`` grants one append that does not consult the policy, so the clause
    that matters is who signed it. An issuer minting the same body is an issuer making a claim
    about a log it does not hold, and the roster decides it like anything else.
    """
    log = fresh(tmp_path / "log")
    forged = policy_statement(log, {"policy": OPEN_POLICY}, label="other")
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert "not in the roster" in exc.value.reason

    # even seated by an admitted issuer, it is not the log's policy: the key is wrong
    seated = policy_statement(log, {"policy": OPEN_POLICY}, label="issuer")
    assert log.append(seated) == 0
    assert log.policy_cert() is None
    assert log.issuer_policy()["source"] == "file"
    with pytest.raises(AppendRefused):
        log.append(F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1")))


def test_a_policy_statement_naming_another_log_governs_neither(tmp_path):
    """L7 pointed at the admission rule: "admits anybody" may not be minted once and replayed.

    A statement that did not name its own log would open every log it was appended to, which is
    the replay attack aimed at the gate instead of at a measurement. It is refused at append by
    ``_check_log_binding``, and a copy of it seated in a clone is not read as that clone's policy.
    """
    first = fresh(tmp_path / "first")
    # `second` is OPEN, so the roster cannot be what refuses this: the binding is.
    second = Log.init(tmp_path / "second", SECOND_LOG_PUB, {"policy": OPEN_POLICY})
    lifted = policy_statement(first, {"policy": OPEN_POLICY}, log_id=first.log_id())
    with pytest.raises(AppendRefused) as exc:
        second.append(lifted)
    assert exc.value.reason.startswith("log_hint:")

    # and written straight into the clone's directory, it still is not the clone's policy
    second.entry_path(0).parent.mkdir(parents=True, exist_ok=True)
    second.entry_path(0).write_bytes(canonical_bytes(lifted))
    second._id_map_cache = None
    assert second.policy_cert() is None
    assert second.issuer_policy()["source"] == "file"  # the clone's file, not the lifted statement
    assert second.issuer_policy()["witnessed"] is False


def test_an_unwitnessed_open_marker_still_admits_every_key(tmp_path):
    """[OPERATOR-GATED], pinned as a test rather than described in a comment.

    A log whose tree states no policy falls back to the file, and there the eighteen-byte marker
    is exactly as strong as it was: it admits every key and nothing distinguishes the operator who
    wrote it from anyone else who could write that directory. Refusing it — making the logged
    statement mandatory — would refuse every append to every open log already running, and it
    moves section 8.2's layout, which is the operator's signature to give and not this module's.

    What IS closed without their signature: the witnessed form exists, the tree beats the file,
    and a file contradicting the tree refuses everything
    (``test_eighteen_bytes_written_into_issuers_json_now_refuse_every_append``). What is not:
    this.
    """
    log = fresh(tmp_path / "log")
    (log.keys_dir / "issuers.json").write_bytes(b'{"policy": "open"}')
    policy = log.issuer_policy()
    assert policy["policy"] == OPEN_POLICY
    assert policy["source"] == "file" and policy["witnessed"] is False
    assert log.append(
        F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1"))
    ) == 0

    report = mirror(log.path, tmp_path / "dst", LOG_PUB)
    assert report["issuer_policy"]["witnessed"] is False
    assert report["verified"] is True  # a configuration is not an accusation (EXTERNAL-1)


def test_the_cli_mints_the_policy_statement_only_when_it_holds_the_log_key(tmp_path):
    """``log init --witness-policy``: the mint, and the half of it that needs the operator.

    The statement is signed by the LOG's key. With ``--key`` this process holds it and writes the
    statement at index 0; with ``--pub`` the operator keeps that key elsewhere, and inventing one
    is not something this command can do for them.
    """
    pem = tmp_path / "log.pem"
    assert climod.run(["key", "generate", "--out", str(pem)])[0] == 0

    code, payload = climod.run([
        "log", "init", "--log", str(tmp_path / "witnessed"), "--key", str(pem),
        "--open-issuers", "--witness-policy",
    ])
    assert code == 0, payload
    assert payload["issuer_policy"] == OPEN_POLICY
    assert payload["policy_source"] == "cert"
    assert payload["policy_cert"] and payload["size"] == 1

    log = Log(tmp_path / "witnessed")
    assert log.policy_cert()[0] == 0
    assert verify_entry(log, 0)[0] is True

    pub = tmp_path / "log.pub"
    pub.write_bytes((tmp_path / "witnessed" / "keys" / "log.pub").read_bytes())
    code, payload = climod.run([
        "log", "init", "--log", str(tmp_path / "unwitnessed"), "--pub", str(pub),
        "--open-issuers", "--witness-policy",
    ])
    assert code != 0
    assert "--witness-policy needs --key" in payload["error"]


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


def test_the_cli_mints_the_binding_and_the_replay_is_then_refused(tmp_path):
    """L7's other half: enforcement was complete and NOTHING MINTED THE FIELD.

    Every rule above was already in place — refused at append, refused on read, counted by
    ``mirror`` — and no CLI path wrote ``body.log_hint``, so no cert produced by this system was
    bound and the replay went on working on every one of them. A gate nothing ever passes through
    is not a closed attack; it is an unreachable branch.

    ``--bind-log`` is the mint, on the commands that sign a cert. The same replay that succeeds on
    the unbound cert below is refused on the bound one, and the only difference between the two
    invocations is the flag.
    """
    pem = tmp_path / "issuer.pem"
    code, key_payload = climod.run(["key", "generate", "--out", str(pem)])
    assert code == 0, key_payload
    public = key_payload["public"]

    log_pem = tmp_path / "log.pem"
    assert climod.run(["key", "generate", "--out", str(log_pem)])[0] == 0
    home = tmp_path / "home"
    code, payload = climod.run(
        ["log", "init", "--log", str(home), "--key", str(log_pem), "--issuer", f"lab={public}"]
    )
    assert code == 0, payload
    home_id = payload["log_id"]

    pool = tmp_path / "pool.json"
    pool.write_bytes(json.dumps(F.battery_items(4)).encode("utf-8"))
    spec = tmp_path / "spec.json"
    spec.write_bytes(json.dumps({
        "subject": copy.deepcopy(F.WEIGHTS_SUBJECT), "recipe": {},
    }).encode("utf-8"))

    def mint(out: Path, *extra: str) -> dict:
        code, payload = climod.run([
            "battery", "pool", "--source", str(pool), "--key", str(pem),
            "--subject", str(spec), "--created", F.CREATED, "--out", str(out), *extra,
        ])
        assert code == 0, payload
        return payload

    unbound = mint(tmp_path / "unbound.json")
    bound_payload = mint(tmp_path / "bound.json", "--bind-log", str(home))

    assert unbound["log_hint"] is None
    assert bound_payload["log_hint"] == {"log_id": home_id}
    assert bound_payload["cert"]["body"]["log_hint"] == {"log_id": home_id}

    # both belong in the log they were minted for
    for path in (tmp_path / "unbound.json", tmp_path / "bound.json"):
        code, appended = climod.run(["log", "append", str(path), "--log", str(home)])
        assert code == 0, appended
    assert climod.run(["log", "append", str(tmp_path / "bound.json"), "--log", str(home)])[0] != 0

    # THE REPLAY, verbatim, into a log on a different key
    other = tmp_path / "other"
    code, payload = climod.run(
        ["log", "init", "--log", str(other), "--key", str(log_pem), "--open-issuers"]
    )
    assert code == 0, payload
    Log(other).keys_dir.joinpath("log.pub").write_bytes(
        keysmod.encode_public(SECOND_LOG_PUB).encode("utf-8")
    )
    assert Log(other).log_id() != home_id

    code, appended = climod.run(["log", "append", str(tmp_path / "unbound.json"), "--log", str(other)])
    assert code == 0, appended            # the limitation, unchanged: an unbound cert replays
    assert appended["bound"] is False

    code, refused = climod.run(["log", "append", str(tmp_path / "bound.json"), "--log", str(other)])
    assert code != 0
    assert "log_hint:" in refused["error"] and home_id in refused["error"]
    assert Log(other).size() == 1


def test_log_append_can_require_a_binding_the_mint_did_not_write(tmp_path):
    """The switch for an operator who has decided their log takes bound certs only.

    It is the INVOCATION's rule and not the log's, and that limit is the point of the test: a log
    that wants bound certs cannot say so on disk, so an operator who forgets the flag once appends
    an unbound cert and nothing afterwards records that they meant otherwise. Making it the log's
    own rule needs the policy statement of ``Log.policy_cert`` to carry it, and that is not landed.
    """
    pem = tmp_path / "log.pem"
    assert climod.run(["key", "generate", "--out", str(pem)])[0] == 0
    root = tmp_path / "log"
    code, payload = climod.run(
        ["log", "init", "--log", str(root), "--key", str(pem), "--open-issuers"]
    )
    assert code == 0, payload

    plain = tmp_path / "plain.json"
    plain.write_bytes(canonical_bytes(F.make_cert("battery", body=F.battery_body("pool-v1"))))
    code, refused = climod.run(
        ["log", "append", str(plain), "--log", str(root), "--require-binding"]
    )
    assert code != 0
    assert "carries no body.log_hint" in refused["error"]
    assert Log(root).size() == 0

    bound_cert = bound("battery", payload["log_id"])
    path = tmp_path / "bound.json"
    path.write_bytes(canonical_bytes(bound_cert))
    code, appended = climod.run(
        ["log", "append", str(path), "--log", str(root), "--require-binding"]
    )
    assert code == 0, appended
    assert appended["bound"] is True


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
