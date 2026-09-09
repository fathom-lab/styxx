"""L11 and the roster forgeries: what a mirror with a memory reaches, and what nothing reaches.

Every test below is an attack that was RUN in this session against the code as it stood, written
back either as the refusal it now earns or as the limitation it still is. Both halves are here
deliberately: a repair that could not be made is a result, and this file is the receipt for the
half that could not.

* **L11** — delete entries 5 and 6 and every tree head from a copy of a sealed seven-entry log,
  re-seal at five with the log key, and ``log.mirror`` reports ``verified: True, entries: 5,
  tamper: [], misbehaviour: []``. Reproduced here as
  ``test_the_L11_truncation_is_invisible_to_a_stateless_mirror``. ``mirror`` already holds the
  predicate that catches it — it says ``sth <pinned>: covers 7 entries, the mirror holds 5`` the
  moment it is handed the seven-entry head — so what was missing was never the check. It was that
  nothing kept the head between runs.
* **The reachable half** — ``witness.watch`` keeps it. A store that observed the seven-entry head
  on Monday reports ``retracted`` on Tuesday and clears ``verified``. Also ``rewritten`` (a head's
  size roots differently now) and ``inconsistent`` (no consistency proof from a retained head).
* **The boundary** — ``test_a_first_observation_of_a_truncated_log_is_indistinguishable`` builds
  a five-entry log and a truncated seven-entry log and asserts that a first-run ``watch`` of each
  returns the same verdict. It is the one test in this file that pins an ATTACK AS UNCLOSED, and
  it is the point of the module: the operator writes the entries and signs the heads over them,
  so a reader who holds no earlier head has nothing to compare and no predicate to run.
* **L10a / L10c′ on an UNWITNESSED log** — another agent moved the admission policy into a logged
  cert. On a log carrying that statement, appending a roster object or writing the eighteen bytes
  ``{"policy": "open"}`` is refused and ``mirror`` says ``verified: False``. On a log NOT carrying
  it — which `papers/v8/first_verdict_2026_09_09/log` is, pinned here against the published bytes
  — both still admit a rogue key at exit 0 with ``mirror`` reporting ``verified: True``. The
  repair MOVED those two; it did not close them. Both halves are pinned.
* **The reachable half of those** — a witness that saw the roster before reports
  ``admission_changed``, and names the entries resting on the edit under ``late_admitted``. Both
  are DISCLOSURES that do not clear ``verified``, because an honest operator adding an issuer
  produces the same bytes, and EXTERNAL-1 is this lab's receipt for what an accuser that fires on
  honest artifacts costs.

Nothing here skips.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from styxx.v8 import cli as climod
from styxx.v8 import keys as keysmod
from styxx.v8 import witness as witnessmod
from styxx.v8.log import AppendRefused, Log, mirror
from styxx.v8.witness import load_store, watch
from tests import v8_fixtures as F

LOG_SEED, LOG_PUB = F.keypair("log-key")

MON = "2026-09-09T12:05:00Z"
TUE = "2026-09-10T09:05:00Z"
SEAL_MON = "2026-09-09T12:00:00Z"
SEAL_TUE = "2026-09-10T09:00:00Z"

PUBLISHED = Path(__file__).resolve().parents[1] / "papers/v8/first_verdict_2026_09_09/log"


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


def battery(n: int, *, issuer_label: str = "issuer") -> dict:
    """A root battery: pool-v1 at n == 1, fixed-v1 above it, so each ``n`` is a distinct cert
    and none of them needs a ref to resolve."""
    kind = "pool-v1" if n == 1 else "fixed-v1"
    return F.make_cert("battery", body=F.battery_body(kind, n=n), issuer_label=issuer_label)


def sealed_log(root: Path, entries: int = 7, *, timestamp: str = SEAL_MON) -> Log:
    log = Log.init(root, LOG_PUB, roster("issuer"))
    for k in range(entries):
        log.append(battery(k + 1))
    log.sth(LOG_SEED, timestamp)
    return log


def truncate(src: Path, dst: Path, keep: int, *, timestamp: str = SEAL_TUE) -> Log:
    """THE L11 ATTACK: drop the entries above ``keep``, drop every head, re-seal with the log
    key. Everything the operator needs, and the operator holds all of it."""
    shutil.copytree(src, dst)
    for path in sorted((dst / "entries").rglob("*.json")):
        index = int(path.name.split(".")[0])
        if index >= keep:
            path.unlink()
    for path in (dst / "sth").glob("*.json"):
        path.unlink()
    log = Log(dst)
    log.sth(LOG_SEED, timestamp)
    return log


# ----------------------------------------------------------------- L11, the attack itself

def test_the_L11_truncation_is_invisible_to_a_stateless_mirror(tmp_path):
    """The demonstrated attack, reproduced before anything repairs it."""
    sealed_log(tmp_path / "src")
    truncate(tmp_path / "src", tmp_path / "bad", keep=5)

    report = mirror(tmp_path / "bad", tmp_path / "dst", LOG_PUB, None)

    assert report["entries"] == 5
    assert report["verified"] is True
    assert report["tamper"] == []
    assert report["misbehaviour"] == []
    assert report["unpublished"] == []


def test_mirror_already_catches_it_when_it_is_handed_the_earlier_head(tmp_path):
    """The predicate was never missing. Persistence was."""
    log = sealed_log(tmp_path / "src")
    head7 = log.latest_sth()
    truncate(tmp_path / "src", tmp_path / "bad", keep=5)

    report = mirror(tmp_path / "bad", tmp_path / "dst", LOG_PUB, head7)

    assert report["verified"] is False
    assert any("covers 7 entries, the mirror holds 5" in line for line in report["tamper"])


# ----------------------------------------------------------------- the reachable half

def test_a_witness_that_saw_the_log_on_monday_reports_the_truncation_on_tuesday(tmp_path):
    store = tmp_path / "witness.json"
    sealed_log(tmp_path / "src")

    monday = watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)
    assert monday["verified"] is True
    assert monday["witnessed"] is False
    assert monday["basis"] == "first-observation"
    assert monday["heads_held"] == 0

    truncate(tmp_path / "src", tmp_path / "bad", keep=5)
    tuesday = watch(tmp_path / "bad", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert tuesday["mirror"]["verified"] is True, "the stateless half still sees nothing"
    assert tuesday["mirror"]["tamper"] == []
    assert tuesday["verified"] is False
    assert tuesday["witnessed"] is True
    assert tuesday["basis"] == "observed-history"
    assert tuesday["heads_compared"] == 1
    assert len(tuesday["retracted"]) == 1
    line = tuesday["retracted"][0]
    assert "covers 7 entries, the log now holds 5" in line
    assert "L11" in line


def test_an_honest_extension_is_not_an_accusation(tmp_path):
    """The control. A checker that fires on a growing log is a broken checker (EXTERNAL-1)."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    log.append(battery(8))
    log.sth(LOG_SEED, SEAL_TUE)
    later = watch(tmp_path / "src", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert later["verified"] is True
    assert later["retracted"] == []
    assert later["rewritten"] == []
    assert later["inconsistent"] == []
    assert later["withdrawn"] == []
    assert later["heads_compared"] == 1


def test_history_rewritten_under_a_published_head_is_reported(tmp_path):
    """Same tree size, different entries, re-sealed. The stateless mirror re-roots the new
    entries, gets the new head, and agrees with itself."""
    store = tmp_path / "witness.json"
    sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    bad = tmp_path / "bad"
    shutil.copytree(tmp_path / "src", bad)
    for path in sorted((bad / "entries").rglob("*.json")):
        if int(path.name.split(".")[0]) >= 3:
            path.unlink()
    for path in (bad / "sth").glob("*.json"):
        path.unlink()
    rewritten = Log(bad)
    for n in (100, 101, 102, 103):
        rewritten.append(battery(n))
    rewritten.sth(LOG_SEED, SEAL_TUE)
    assert rewritten.size() == 7

    report = watch(bad, tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert report["mirror"]["verified"] is True, "seven entries rooting to their own head"
    assert report["verified"] is False
    assert len(report["rewritten"]) == 1
    assert "the entries now root to" in report["rewritten"][0]


def test_the_consistency_path_fires_on_a_forged_current_head_and_reaches_nothing_alone(tmp_path):
    """``inconsistent`` is a SECOND arithmetic path, not a second attack. The only construction
    found that reaches it is a forged current head, and ``mirror`` catches that on its own — so
    this test pins the branch as exercised AND pins that it adds no coverage of its own."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    log.append(battery(8))
    core = {
        "log_id": log.log_id(),
        "tree_size": 8,
        "root_hash": "sha256:" + "ab" * 32,  # a root the entries do not produce
        "timestamp": SEAL_TUE,
    }
    import hashlib

    from styxx.v8.consts import STH_TAG
    from styxx.v8.jcs import canonical_bytes

    digest = hashlib.sha256(canonical_bytes(core)).digest()
    forged = dict(core)
    forged["sig"] = keysmod.encode_signature(
        keysmod.sign(LOG_SEED, keysmod.tagged(STH_TAG, digest))
    )
    (log.path / "sth" / "000000000008.json").write_text(json.dumps(forged), encoding="utf-8")

    report = watch(tmp_path / "src", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert len(report["inconsistent"]) == 1
    assert "head 7 -> 8" in report["inconsistent"][0]
    assert "does not extend the first" in report["inconsistent"][0]
    assert report["verified"] is False
    # and the stateless half already had it, which is the point of the sentence in the docstring
    assert any(
        "do not reproduce the signed root" in line
        for line in report["mirror"]["misbehaviour"] + report["mirror"]["tamper"]
    )


def test_a_head_the_operator_stopped_publishing_is_a_disclosure_not_an_accusation(tmp_path):
    """The entries still reproduce it, so nothing the log says is contradicted. What is gone is
    the operator's own publication of it, and the witness still holds the head."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    log.append(battery(8))
    log.sth(LOG_SEED, SEAL_TUE)
    (log.path / "sth" / "000000000007.json").unlink()

    report = watch(tmp_path / "src", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert len(report["withdrawn"]) == 1
    assert "no longer publishes it in sth/" in report["withdrawn"][0]
    assert report["retracted"] == [] and report["rewritten"] == []
    assert report["verified"] is True, "a disclosure does not clear verified"


def test_the_store_survives_a_run_and_carries_the_head_that_caught_it(tmp_path):
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    held = load_store(store)["logs"][log.log_id()]
    assert held["runs"] == 1
    assert held["first_seen"] == MON
    keys = list(held["heads"])
    assert len(keys) == 1 and keys[0].startswith("7:sha256:")
    assert held["heads"][keys[0]]["source"] == "observed"
    assert held["heads"][keys[0]]["sth"] == log.latest_sth()

    truncate(tmp_path / "src", tmp_path / "bad", keep=5)
    watch(tmp_path / "bad", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    after = load_store(store)["logs"][log.log_id()]
    assert after["runs"] == 2
    assert len(after["heads"]) == 2, "a head once recorded is never dropped"
    assert any(k.startswith("7:") for k in after["heads"])


def test_a_head_obtained_from_outside_is_labelled_as_such(tmp_path):
    """``pinned`` is a statement about PROVENANCE and never about time: the head says the
    operator signed those bytes, not when."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    head7 = log.latest_sth()
    truncate(tmp_path / "src", tmp_path / "bad", keep=5)

    report = watch(
        tmp_path / "bad", tmp_path / "m1", LOG_PUB, store, pinned_sth=head7, observed_at=MON
    )

    assert report["basis"] == "external-pin"
    assert report["verified"] is False
    assert any("covers 7 entries" in line for line in report["mirror"]["tamper"])
    held = load_store(store)["logs"][log.log_id()]
    assert any(v["source"] == "pinned" for v in held["heads"].values())


def test_a_pin_that_does_not_verify_does_not_upgrade_the_basis(tmp_path):
    """``basis`` is what the report is RESTING on. A head that fails its own signature check is
    not a channel outside the operator, so it must not read as one."""
    log = sealed_log(tmp_path / "src")
    forged = dict(log.latest_sth())
    forged["sig"] = "ed25519:" + "A" * 86

    report = watch(
        tmp_path / "src", tmp_path / "m1", LOG_PUB, tmp_path / "w.json",
        pinned_sth=forged, observed_at=MON,
    )

    assert report["basis"] == "first-observation"
    assert report["external_pins"] == 0
    assert any("pinned sth" in line for line in report["store"])


def test_the_store_is_never_the_reason_the_log_is_accused(tmp_path):
    """A retained head that no longer verifies is a fact about the WITNESS's own side."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    raw = json.loads(store.read_text(encoding="utf-8"))
    record = raw["logs"][log.log_id()]
    key = next(iter(record["heads"]))
    record["heads"][key]["sth"]["sig"] = "ed25519:" + "A" * 86
    store.write_text(json.dumps(raw), encoding="utf-8")

    report = watch(tmp_path / "src", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert report["heads_compared"] == 0
    assert any("retained head" in line for line in report["store"])
    assert report["retracted"] == [] and report["rewritten"] == []
    assert report["verified"] is True


def test_a_corrupt_store_is_an_empty_one_and_the_log_is_still_mirrored(tmp_path):
    store = tmp_path / "witness.json"
    store.write_text("{not json", encoding="utf-8")
    sealed_log(tmp_path / "src")

    report = watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    assert report["witnessed"] is False
    assert any("treated as empty" in line for line in report["store"])
    assert report["mirror"]["entries"] == 7
    assert report["verified"] is True
    # the rebuild leaves a scar: a store whose history was lost says so, forever
    assert "unreadable" in json.loads(store.read_text(encoding="utf-8"))


def test_watch_never_raises_on_a_hostile_directory(tmp_path):
    # An absent log is an empty one and makes no claim to break (``mirror``'s own rule).
    report = watch(tmp_path / "nope", tmp_path / "dst", LOG_PUB, tmp_path / "w.json")
    assert report["mirror"]["entries"] == 0
    assert report["witnessed"] is False
    assert report["retracted"] == []

    junk = tmp_path / "junk"
    (junk / "entries" / "000000").mkdir(parents=True)
    (junk / "entries" / "000000" / "00000000.json").write_bytes(b"not json at all")
    (junk / "sth").mkdir(parents=True)
    (junk / "sth" / "000000000001.json").write_bytes(b"{")
    report = watch(junk, tmp_path / "dst-junk", LOG_PUB, tmp_path / "w2.json")
    assert report["verified"] is False
    assert report["mirror"]["verified"] is False


# ----------------------------------------------------------------- THE BOUNDARY

def test_a_first_observation_of_a_truncated_log_is_indistinguishable(tmp_path):
    """PINNED AS UNCLOSED. Two logs, one honest at five entries and one truncated from seven,
    each mirrored by a witness that has never seen it before. The reports are the same verdict,
    and no predicate over the bytes separates them, because the operator wrote the entries and
    signed the heads over them.

    This is DURABILITY section 3.1 and 3.2 in one instance and THE_BOUNDARY's general form:
    a check on bytes an issuer wrote can only ask whether that party contradicted itself.
    """
    honest = sealed_log(tmp_path / "honest", entries=5, timestamp=SEAL_TUE)
    sealed_log(tmp_path / "seven")
    truncate(tmp_path / "seven", tmp_path / "cut", keep=5, timestamp=SEAL_TUE)

    a = watch(honest.path, tmp_path / "ma", LOG_PUB, tmp_path / "wa.json", observed_at=TUE)
    b = watch(tmp_path / "cut", tmp_path / "mb", LOG_PUB, tmp_path / "wb.json", observed_at=TUE)

    for report in (a, b):
        assert report["verified"] is True
        assert report["witnessed"] is False
        assert report["basis"] == "first-observation"
        assert report["retracted"] == []
        assert report["mirror"]["entries"] == 5
        assert report["mirror"]["tamper"] == []
        assert report["mirror"]["misbehaviour"] == []

    shape = ("entries", "sths", "verified", "tamper", "misbehaviour", "unpublished")
    assert {k: a["mirror"][k] for k in shape} == {k: b["mirror"][k] for k in shape}
    # And the two logs are the same bytes wherever they overlap.
    assert Log(tmp_path / "cut").root() == honest.root()


def test_the_report_states_what_it_cannot_reach_beside_the_verdict(tmp_path):
    """THE_BOUNDARY: a class-two defect is a DISCLOSURE and belongs in the output beside the
    verdict, not in a limits section, because no future release closes it."""
    sealed_log(tmp_path / "src")
    report = watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, tmp_path / "w.json")

    assert report["unreachable"] == list(witnessmod.UNREACHABLE)
    assert any("byte-indistinguishable" in line for line in report["unreachable"])
    assert any("dates nothing" in line for line in report["unreachable"])


def test_first_seen_is_this_witnesss_own_clock_and_dates_nothing(tmp_path):
    """The store's timestamp is a parameter of the process that wrote it, so a witness can be
    handed any date. It is a label, never a pin: DURABILITY section 3.1's construction is not
    made here and cannot be made retroactively."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src")
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at="1999-01-01T00:00:00Z")

    held = load_store(store)["logs"][log.log_id()]
    assert held["first_seen"] == "1999-01-01T00:00:00Z"
    assert all(h["first_seen"] == "1999-01-01T00:00:00Z" for h in held["heads"].values())


# ----------------------------------------------------------------- L10a and L10c', both halves

def rogue_append(log: Log, n: int) -> str:
    try:
        return "appended at %d" % log.append(battery(n, issuer_label="rogue"))
    except AppendRefused as exc:
        return "refused: " + exc.reason


def add_roster_object(root: Path) -> None:
    """L10a: one object appended to an unsigned array."""
    path = root / "keys" / "issuers.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    current.append(
        {
            "name": "rogue",
            "key": F.public_key("rogue"),
            "from_index": 0,
            "retired_at_index": None,
        }
    )
    path.write_text(json.dumps(current), encoding="utf-8")


def write_open_marker(root: Path) -> None:
    """L10c': eighteen bytes."""
    payload = '{"policy": "open"}'
    assert len(payload.encode("utf-8")) == 18
    (root / "keys" / "issuers.json").write_text(payload, encoding="utf-8")


@pytest.mark.parametrize("edit", [add_roster_object, write_open_marker],
                         ids=["L10a-roster-object", "L10c-open-marker"])
def test_on_an_unwitnessed_log_both_forgeries_still_admit_a_rogue_key(tmp_path, edit):
    """PINNED AS UNCLOSED, and it is the answer to "does the logged policy close these?".

    It does not, on a log that carries no policy statement. The defence moved behind a statement
    this log does not have.
    """
    log = sealed_log(tmp_path / "src", entries=2)
    assert log.policy_cert() is None
    assert log.issuer_policy()["witnessed"] is False
    assert rogue_append(log, 50).startswith("refused:"), "the control, before the edit"

    edit(log.path)
    after = Log(log.path)
    assert rogue_append(after, 50).startswith("appended at")

    after.sth(LOG_SEED, SEAL_TUE)
    report = mirror(after.path, tmp_path / "dst", LOG_PUB, after.latest_sth())
    assert report["verified"] is True
    assert report["tamper"] == [] and report["metadata"] == []
    assert report["issuer_policy"]["witnessed"] is False


@pytest.mark.parametrize("edit", [add_roster_object, write_open_marker],
                         ids=["L10a-roster-object", "L10c-open-marker"])
def test_on_a_witnessed_log_both_forgeries_are_refused(tmp_path, edit):
    """The other half of the same sentence: the repair works where the statement exists."""
    root = tmp_path / "witnessed"
    keypath = tmp_path / "log.pem"
    keysmod.save_private_pem(LOG_SEED, keypath, overwrite=True)
    code, _payload = climod.run(
        [
            "log", "init", "--log", str(root), "--key", str(keypath),
            "--issuer", F.public_key("issuer"), "--witness-policy",
        ]
    )
    assert code == 0
    log = Log(root)
    assert log.policy_cert() is not None
    assert log.issuer_policy()["witnessed"] is True

    edit(root)
    after = Log(root)
    assert after.issuer_policy()["policy"] == "contradicted"
    assert rogue_append(after, 50).startswith("refused:")

    report = mirror(root, tmp_path / "dst", LOG_PUB, after.latest_sth())
    assert report["verified"] is False
    assert any("keys/issuers.json" in line and "L10c" in line for line in report["metadata"])


def test_the_published_log_is_the_unwitnessed_case(tmp_path):
    """No fixture: the real published bytes. `papers/v8/first_verdict_2026_09_09/log` carries no
    policy statement, so it is the log the two tests above describe."""
    assert PUBLISHED.is_dir(), f"the published log is committed at {PUBLISHED} and must not move"
    log = Log(PUBLISHED)
    assert log.policy_cert() is None
    policy = log.issuer_policy()
    assert policy["source"] == "file"
    assert policy["witnessed"] is False


@pytest.mark.parametrize("edit", [add_roster_object, write_open_marker],
                         ids=["L10a-roster-object", "L10c-open-marker"])
def test_a_witness_that_saw_the_roster_before_reports_that_it_moved(tmp_path, edit):
    """The reachable half, and only a half: it reports the CHANGE and cannot attribute it."""
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src", entries=2)
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)

    edit(log.path)
    after = Log(log.path)
    after.append(battery(50, issuer_label="rogue"))
    after.sth(LOG_SEED, SEAL_TUE)

    report = watch(tmp_path / "src", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert report["mirror"]["verified"] is True, "the stateless half sees nothing"
    assert len(report["admission_changed"]) == 1
    line = report["admission_changed"][0]
    assert "keys/issuers.json" in line and "cannot be attributed" in line
    assert "L10a, L10c'" in line, "an unwitnessed log has nothing else to attribute it either"
    assert report["late_admitted"] == [
        "entry 2 is signed by %s, which was not in the roster this witness held at tree size 2"
        % F.public_key("rogue")
    ]
    assert report["verified"] is True, (
        "a roster grows honestly and the same bytes are produced either way; this is a "
        "disclosure and an accusation would be EXTERNAL-1's 0.23 precision again"
    )


def test_an_unchanged_roster_is_reported_as_nothing(tmp_path):
    store = tmp_path / "witness.json"
    log = sealed_log(tmp_path / "src", entries=2)
    watch(tmp_path / "src", tmp_path / "m1", LOG_PUB, store, observed_at=MON)
    log.append(battery(9))
    log.sth(LOG_SEED, SEAL_TUE)

    report = watch(tmp_path / "src", tmp_path / "m2", LOG_PUB, store, observed_at=TUE)

    assert report["admission_changed"] == []
    assert report["late_admitted"] == []
    held = load_store(store)["logs"][log.log_id()]
    assert len(held["admission"]) == 1, "one unchanged observation, its last_seen moved"
    assert held["admission"][0]["last_seen"] == TUE


# ----------------------------------------------------------------- the CLI

def test_log_watch_exits_zero_on_a_first_observation_and_non_zero_on_the_truncation(tmp_path):
    store = tmp_path / "witness.json"
    sealed_log(tmp_path / "src")
    pin = tmp_path / "src" / "keys" / "log.pub"

    code, payload = climod.run(
        [
            "log", "watch", "--log", str(tmp_path / "src"), "--to", str(tmp_path / "m1"),
            "--store", str(store), "--pin", str(pin), "--observed-at", MON,
        ]
    )
    assert code == 0
    assert payload["command"] == "log watch"
    assert payload["report"]["basis"] == "first-observation"

    truncate(tmp_path / "src", tmp_path / "bad", keep=5)
    code, payload = climod.run(
        [
            "log", "watch", "--log", str(tmp_path / "bad"), "--to", str(tmp_path / "m2"),
            "--store", str(store), "--pin", str(pin), "--observed-at", TUE,
        ]
    )
    assert code != 0
    assert payload["report"]["retracted"], payload["report"]
    assert payload["report"]["unreachable"]


def test_log_watch_needs_a_store(tmp_path):
    sealed_log(tmp_path / "src")
    code, payload = climod.run(
        ["log", "watch", "--log", str(tmp_path / "src"), "--to", str(tmp_path / "m1")]
    )
    assert code != 0
    assert "store" in json.dumps(payload)
