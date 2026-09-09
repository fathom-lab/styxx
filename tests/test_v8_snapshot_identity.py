"""M3 — one (hf_repo, revision) names one snapshot, and one snapshot names one (hf_repo, revision).

``papers/v8/THE_BOUNDARY_2026_09_09.md`` published a roster of defects it called unreachable by
any check over bytes an issuer wrote, and member 3 was *"the repository and revision, read off a
directory name"*. Nothing in the log witnessed which directory the runner opened, and a fabricated
``hf_repo``/``revision`` is byte-indistinguishable from an honest one, so the member looked like
the label class.

A fifth adversarial pass refuted it by the same test that had already reclassified the environment
member one pass earlier: **another logged cert already carries the information that would catch
it, and nothing compares them.** Beside ``hf_repo`` and ``revision`` in the same subject sit the
four Appendix A.2 hashes, taken over the snapshot's actual bytes. Both halves were demonstrated
appending, and both are refused here:

* the same revision under two ``weights_sha256`` — one name over two snapshots. **Refused**: one
  revision of one repository is one set of files, so two certs hashing it two ways cannot both be
  right, and appending the second makes the log state two incompatible facts about one directory.
* the same ``weights_sha256`` under two revisions — one snapshot wearing two names. **Disclosed,
  not refused**, and the asymmetry is a decision this file pins rather than assumes. A repository
  commit touching only files outside the A.2 list — a README, a licence — gives two honest
  revisions one identical quadruple, so a refusal here fires on an honest artifact, which is the
  defect EXTERNAL-1 measured at 0.23 precision before that class was disabled. It goes into the
  entry's metadata as ``snapshot_aliases`` instead, and whether it ever becomes a refusal is the
  operator gate the spec draft calls B-XLOG (A-62). What that costs is stated where it is paid:
  the demonstrated attack, one ``weights_sha256`` under a second revision, still appends.

**What this does not close, which is most of the member and is pinned at the bottom of this file.**
It catches an issuer contradicting itself across one log. Rename the directory once, mint every
cert from that runner, and every entry tells the same lie consistently: one name, one snapshot, no
disagreement, and ``snapshot_disagreement`` returns ``[]``. The hashes are over the bytes the
runner read; nothing here says those bytes came from the repository the subject names, because
nothing in this log fetched them. Closing that needs a second party who fetches the named revision
and logs its hashes — the reproduction leg, not a predicate. Member 3 moves from "no check reaches
this" to "this check reaches the inconsistent case", which is exactly what the environment member
got, and by the standard the lab applied to that one it belongs in class one.

``precision`` takes no part. It is a load-time cast and changes no content hash: this lab's own
published bf16 and fp16 subjects carry identical A.2 hashes under one (hf_repo, revision), and a
predicate that put ``precision`` on the hash side would refuse that honest pair. The test that
pins it reads those two published files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from styxx.v8.log import AppendRefused, Log
from tests import v8_fixtures as F

LOG_SEED, LOG_PUB = F.keypair("log-key")
PUBLISHED = Path(__file__).resolve().parents[1] / "papers/v8/first_verdict_2026_09_09"

OTHER_REVISION = "0" * 40
OTHER_WEIGHTS = "b" * 64


def _roster() -> list[dict]:
    return [{
        "name": F.ISSUER_NAME,
        "key": F.public_key("issuer"),
        "from_index": 0,
        "retired_at_index": None,
    }]


def _fresh(root) -> Log:
    return Log.init(root, LOG_PUB, _roster())


def _fp(battery_id: str, subject: dict, **body_kw) -> dict:
    return F.make_cert(
        "fingerprint",
        subject=subject,
        recipe=F.recipe(battery=battery_id),
        refs=[{"role": "battery", "id": battery_id}],
        body=F.fingerprint_body(**body_kw),
    )


def _log_with_battery(tmp_path, subject=None):
    log = _fresh(tmp_path / "log")
    battery = F.make_cert(
        "battery",
        body=F.battery_body("pool-v1"),
        **({"subject": subject} if subject is not None else {}),
    )
    log.append(battery)
    return log, battery


# ----------------------------------------------------------------- the two halves


def test_one_snapshot_under_two_revisions_is_disclosed_and_not_refused(tmp_path):
    """M3a, as demonstrated: the same ``weights_sha256`` appending under a second revision.

    It still appends, and that is the decision rather than an oversight — see the module docstring
    and ``Log.snapshot_aliases``. What changes is that the entry now names the cert it shares its
    bytes with, so a reader deciding whether a revision was fabricated has the two ids.
    """
    log, battery = _log_with_battery(tmp_path)
    at_first = log.append(_fp(battery["id"], F.weights_subject()))

    same_bytes_new_name = _fp(
        battery["id"], F.weights_subject(revision=OTHER_REVISION), run_index=1
    )
    at = log.append(same_bytes_new_name)          # NOT refused
    aliases = log.meta(at)["snapshot_aliases"]
    assert {a["index"] for a in aliases} == {0, at_first}   # the battery carries the subject too
    entry = [a for a in aliases if a["index"] == at_first][0]
    assert entry["half"] == "one-snapshot-two-names"
    assert entry["fields"] == ["revision"]
    assert entry["mine"] == [OTHER_REVISION]
    assert entry["theirs"] == [F.WEIGHTS_SUBJECT["revision"]]
    assert log.derived_meta(at)["snapshot_aliases"] == aliases   # re-derivable, A-META


def test_the_disclosure_is_absent_rather_than_empty_when_there_is_nothing_to_say(tmp_path):
    """A key present on every entry is a key readers stop reading (and a wider tamper surface)."""
    log, battery = _log_with_battery(tmp_path)
    at = log.append(_fp(battery["id"], F.weights_subject()))
    assert log.snapshot_aliases(log.cert(at)) is None
    assert "snapshot_aliases" not in log.meta(at)
    assert "snapshot_aliases" not in log.derived_meta(at)


def test_one_revision_over_two_snapshots_is_refused(tmp_path):
    """M3b, as demonstrated: the same revision appending over a different ``weights_sha256``."""
    log, battery = _log_with_battery(tmp_path)
    log.append(_fp(battery["id"], F.weights_subject()))

    same_name_new_bytes = _fp(
        battery["id"], F.weights_subject(weights_sha256=OTHER_WEIGHTS), run_index=1
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(same_name_new_bytes)
    reason = exc.value.reason
    assert reason.startswith("subject:")
    assert "names the same (hf_repo, revision)" in reason
    assert "['weights_sha256']" in reason
    assert OTHER_WEIGHTS in reason and F.WEIGHTS_SUBJECT["weights_sha256"] in reason
    assert "(M3, section 2.2)" in reason


def test_the_same_repository_under_two_names_is_disclosed(tmp_path):
    """The other way M3a arrives: one set of files under two ``hf_repo`` strings.

    A mirror of one repository under a second org name is the honest version of this, which is the
    same reason this half discloses instead of refusing.
    """
    log, battery = _log_with_battery(tmp_path)
    log.append(_fp(battery["id"], F.weights_subject()))
    at = log.append(_fp(battery["id"], F.weights_subject(hf_repo="acme/mirror"), run_index=1))
    aliases = log.meta(at)["snapshot_aliases"]
    assert all(a["fields"] == ["hf_repo"] for a in aliases)
    assert all(a["mine"] == ["acme/mirror"] for a in aliases)


@pytest.mark.parametrize(
    "field", ["config_sha256", "tokenizer_sha256", "generation_config_sha256"]
)
def test_every_a2_hash_binds_the_name_not_only_the_weights(tmp_path, field):
    """All four A.2 hashes, not a chosen one: a config swap under one revision is the same lie."""
    log, battery = _log_with_battery(tmp_path)
    log.append(_fp(battery["id"], F.weights_subject()))
    with pytest.raises(AppendRefused) as exc:
        log.append(_fp(battery["id"], F.weights_subject(**{field: "f" * 64}), run_index=1))
    assert f"['{field}']" in exc.value.reason


# ----------------------------------------------------------------- what still appends


def test_two_precisions_of_one_snapshot_append(tmp_path):
    """The honest pair the predicate must not refuse, taken from the published subjects.

    ``precision`` is a load-time cast. The lab's own bf16 and fp16 subject files are byte-identical
    but for that one field — same repo, same revision, same four A.2 hashes — and a predicate that
    counted ``precision`` as part of the snapshot would refuse the second one.
    """
    bf16 = json.loads((PUBLISHED / "subject_bf16.json").read_text(encoding="utf-8"))
    fp16 = json.loads((PUBLISHED / "subject_fp16.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in bf16.items() if k != "precision"} == {
        k: v for k, v in fp16.items() if k != "precision"
    }
    assert bf16["precision"] != fp16["precision"]

    log, battery = _log_with_battery(tmp_path, subject=bf16)
    log.append(_fp(battery["id"], bf16))
    assert log.snapshot_disagreement(_fp(battery["id"], fp16, run_index=1)) == []
    assert log.append(_fp(battery["id"], fp16, run_index=1)) == 2


def test_two_revisions_with_two_snapshots_append(tmp_path):
    """A second revision that really is a second set of files: two names, two snapshots, no lie."""
    log, battery = _log_with_battery(tmp_path)
    log.append(_fp(battery["id"], F.weights_subject()))
    second = F.weights_subject(revision=OTHER_REVISION, weights_sha256=OTHER_WEIGHTS)
    assert log.append(_fp(battery["id"], second, run_index=1)) == 2


def test_a_second_model_appends(tmp_path):
    """Two different models in one log: different names, different hashes, nothing compared."""
    log, battery = _log_with_battery(tmp_path)
    log.append(_fp(battery["id"], F.weights_subject()))
    other = F.weights_subject(
        hf_repo="acme/other-model",
        revision="c" * 40,
        weights_sha256="1" * 64,
        config_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        generation_config_sha256="4" * 64,
    )
    assert log.append(_fp(battery["id"], other, run_index=1)) == 2


def test_an_alias_subject_is_out_of_reach_and_says_so(tmp_path):
    """An alias subject carries no snapshot at all — no repo, no revision, no A.2 hashes.

    The predicate is over the weights branch of section 2.2 and there is nothing here to compare;
    what an alias endpoint served is unreachable by anything this log holds, which is a different
    open member and not this one.
    """
    log = _fresh(tmp_path / "log")
    battery = F.make_cert("battery", subject=F.alias_subject(), body=F.battery_body("pool-v1"))
    log.append(battery)
    alias_fp = F.make_cert(
        "fingerprint",
        subject=F.alias_subject(),
        recipe=F.recipe(battery=battery["id"]),
        refs=[{"role": "battery", "id": battery["id"]}],
        body=F.fingerprint_body(),
    )
    assert log.snapshot_disagreement(alias_fp) == []
    assert log.append(alias_fp) == 1


# ----------------------------------------------------------------- the predicate as a reader runs it


def test_the_predicate_is_public_and_names_both_certs(tmp_path):
    """``snapshot_disagreement`` is a reader's function, like ``floor_census`` (section 8.2).

    Everything the refusal states is in the log's own bytes, so a reader who never appends
    anything can run the same predicate over a log they were handed.
    """
    log, battery = _log_with_battery(tmp_path)
    at = log.append(_fp(battery["id"], F.weights_subject()))
    liar = _fp(battery["id"], F.weights_subject(weights_sha256=OTHER_WEIGHTS), run_index=1)

    found = log.snapshot_disagreement(liar)
    assert [f["half"] for f in found] == ["one-name-two-snapshots"] * len(found)
    assert {f["index"] for f in found} == {0, at}   # the battery carries the subject too
    entry = [f for f in found if f["index"] == at][0]
    assert entry["id"] == log.cert(at)["id"]
    assert entry["fields"] == ["weights_sha256"]
    assert entry["mine"] == [OTHER_WEIGHTS]
    assert entry["theirs"] == [F.WEIGHTS_SUBJECT["weights_sha256"]]


def test_the_predicate_is_bounded_by_the_entry_index(tmp_path):
    """A cert the log already holds is compared against what preceded IT, not against everything.

    Same rule as ``Log.baseline_gap`` after A-META-GAP: an answer a reader re-derives must be the
    answer the appending index produced, or it moves when a later entry lands.
    """
    log, battery = _log_with_battery(tmp_path)
    first = _fp(battery["id"], F.weights_subject())
    log.append(first)
    assert log.snapshot_disagreement(first) == []      # nothing before it contradicts it
    assert log.snapshot_disagreement(log.cert(0)) == []


def test_the_published_log_contradicts_itself_about_nothing():
    """The lab's own published verdict, entry by entry: seven certs, one snapshot, no disagreement.

    A predicate that refuses honest artifacts is the defect EXTERNAL-1 measured at 0.23 precision,
    so this runs the new rule over the bytes it must never fire on.
    """
    log = Log(PUBLISHED / "log")
    for index in log.indices():
        assert log.snapshot_disagreement(log.cert(index)) == []


def test_a_uniform_misnaming_is_not_caught_and_this_is_the_limit(tmp_path):
    """THE LIMIT, pinned as a passing test rather than left in prose.

    Rename the directory once and mint every cert from that runner: every entry names
    ``acme/not-the-real-repo`` at revision ``deadbeef...`` over one consistent set of A.2 hashes.
    There is no contradiction in the log, ``snapshot_disagreement`` returns ``[]`` on every entry,
    and all of it appends. The predicate reaches an issuer that contradicts itself and reaches
    nothing else; only a second party fetching the named revision closes the rest.
    """
    lie = {"hf_repo": "acme/not-the-real-repo", "revision": "deadbeef" * 5}
    log, battery = _log_with_battery(tmp_path, subject=F.weights_subject(**lie))
    previous = None
    for k in range(3):
        cert = F.make_cert(
            "fingerprint",
            subject=F.weights_subject(**lie),
            recipe=F.recipe(battery=battery["id"]),
            refs=(
                [{"role": "battery", "id": battery["id"]}]
                + ([{"role": "previous", "id": previous}] if previous else [])
            ),
            body=F.fingerprint_body(run_index=k),
        )
        assert log.snapshot_disagreement(cert) == []
        log.append(cert)          # section 5.5 wants the ref; M3 wants nothing at all
        previous = cert["id"]
    assert log.size() == 4
    for index in log.indices():
        assert log.snapshot_disagreement(log.cert(index)) == []
