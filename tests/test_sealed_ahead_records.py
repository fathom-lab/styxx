# -*- coding: utf-8 -*-
"""papers/sealed-ahead/records_2026_10_12: the stopped build's records hold the bytes its NOTE names.

The records were copied byte for byte from a separate local repository
(NOTE_sealed_ahead_build_stopped_2026_10_12.md, section 7). A receipt is history too: one regenerated
in place silently breaks every sentence that cites its sha256, so any edit here fails a test. The
battery's commit is re-derived from the committed bytes; its stated date is checked only for
agreement with those bytes, never as a time anyone else can verify (NOTE section 5).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SA = ROOT / "papers" / "sealed-ahead"
REC = SA / "records_2026_10_12"
NOTE = SA / "NOTE_sealed_ahead_build_stopped_2026_10_12.md"

RECORDS = {
    "BATTERY.md": "022cbe90392bd797d65eecb4cda02c1d60b7e7d7f987b90290468e76090398fb",
    "BATTERY_vs_build.md": "d2025e4e913797b9817fce1006912c340388c617cbe9f3d4bd5ff4e7650f79a9",
    "SIZING.md": "0b3e9c77da4c37c7512b9a26904a0d97264bc4b34210df1c26acd694bd882b08",
    "battery_commit_6faeff16.txt": "0df975a3cd1dd140416971780180d4ee904a053953cd75a196baea8c062199c7",
    "hour_name_check_2026_10_12.json": "797f8c7d5bd434c410b1071f3e874bd26e0dafd2c255cb2b2075821c51742b95",
    "sizing/params_provisional_2026_09_28.json": "ce19d3b372d654c3fcfe345da7b2fced63c8b133df76292c9e3e9afaee7042eb",
    "sizing/sizing.py": "809604ef34ebe17d6593a80de5d7a0c1d6644a7a91f6bf924cdbb9982414fb39",
    "sizing/sizing_plan_2026_09_28.json": "1b7ad7e3bdbc6907667abf23e2516c54bf6cfee32a672b46607a4f15ef88735b",
    "sizing/stranger_pool_2026_10_12.json": "740a1b544b5981e21c89e1ceb909c7ffce354d075d30e7265ff955400f02c1f5",
    "sizing_derived/published/pool_meta.json": "92c5ba6a6d0cbe0d791ac5c06d09d5f7b9293031d32119dc31904dd42021666f",
    "sizing_week_2026_09_28.json": "1503f12ed98da2ae18a79b4c45d60191e6de7d905e37f51c1f1bff4e5afdfade",
}

BATTERY_BLOB = "44d45c30cb4c06c536019963aa4a3a7e4363eb5c"
BATTERY_TREE = "4b49d6d9bca6cb0538a04ce529a2b58cac769153"
BATTERY_COMMIT = "6faeff16c71192385b5881628ab63c0933271487"
BATTERY_STATED = " 1791811103 -0400"          # 2026-10-12 09:18:23 -0400, the committer's own clock


def _object_id(kind: bytes, body: bytes) -> str:
    """The git object id (sha1) of a blob, tree or commit body."""
    return hashlib.sha1(b"%s %d\x00" % (kind, len(body)) + body).hexdigest()


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _json(rel: str):
    return json.loads((REC / rel).read_bytes().decode("utf-8"))


@pytest.mark.parametrize("rel", sorted(RECORDS))
def test_each_record_holds_the_bytes_the_note_names(rel):
    assert hashlib.sha256((REC / rel).read_bytes()).hexdigest() == RECORDS[rel], (
        f"{rel} no longer holds the bytes committed as a record; a record is never regenerated in place")
    assert RECORDS[rel] in NOTE.read_text(encoding="utf-8"), f"the NOTE does not name {rel} by its sha256"


def test_the_records_folder_holds_nothing_else():
    found = sorted(p.relative_to(REC).as_posix() for p in REC.rglob("*") if p.is_file())
    assert found == sorted(RECORDS)


def test_the_records_are_never_eol_translated():
    for rel in sorted(RECORDS):
        path = (REC / rel).relative_to(ROOT).as_posix()
        r = subprocess.run(["git", "check-attr", "text", "--", path], cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0 and r.stdout.strip().endswith(": unset"), (
            f"{path} is not -text in .gitattributes; a CRLF checkout would change bytes named by sha256")


def test_the_battery_commit_names_these_bytes_and_states_its_own_date():
    blob = _object_id(b"blob", (REC / "BATTERY.md").read_bytes())
    assert blob == BATTERY_BLOB
    tree = _object_id(b"tree", b"100644 BATTERY.md\x00" + bytes.fromhex(blob))
    assert tree == BATTERY_TREE
    raw = (REC / "battery_commit_6faeff16.txt").read_bytes()
    assert _object_id(b"commit", raw) == BATTERY_COMMIT
    header = raw.split(b"\n\n", 1)[0].decode("utf-8").splitlines()
    assert header[0] == "tree " + BATTERY_TREE
    assert not [h for h in header if h.startswith("parent ")], "the battery's commit is its repository's root"
    stamped = [h for h in header if h.startswith(("author ", "committer "))]
    assert len(stamped) == 2 and all(h.endswith(BATTERY_STATED) for h in stamped)


def test_the_builder_code_is_the_code_the_sizing_week_ran():
    builder = _json("sizing/sizing_plan_2026_09_28.json")["builder"]
    for name, digest in sorted(builder["modules_lf_sha256"].items()):
        assert _lf_sha256(SA / name) == digest, f"papers/sealed-ahead/{name} moved since the sizing week ran it"
    assert _lf_sha256(SA / "identity_table.json") == builder["identity_table_lf_sha256"]


def test_the_sizing_receipts_agree_with_each_other():
    week = _json("sizing_week_2026_09_28.json")
    stranger = _json("sizing/stranger_pool_2026_10_12.json")
    meta = _json("sizing_derived/published/pool_meta.json")
    params = _json("sizing/params_provisional_2026_09_28.json")
    assert week["script"]["lf_sha256"] == RECORDS["sizing/sizing.py"]
    assert week["plan"]["lf_sha256"] == RECORDS["sizing/sizing_plan_2026_09_28.json"]
    canon = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(canon).hexdigest() == week["params_sha256"] == meta["params_sha256"]
    assert stranger["stage"] == "pool" and stranger["verdict"] == "AGREES" and all(stranger["checks"].values())
    pool_sha = week["population"]["pool_sha256"]
    assert stranger["pool_sha256"] == stranger["published_pool_sha256"] == meta["pool_sha256"] == pool_sha
    assert stranger["records"] == meta["records"] == week["population"]["pool_records"] == 18098
    sample = week["sample"]
    assert sample["final_outcomes"] == 300 and sample["unfetched"] == 0
    assert sample["accusations"] == 0 and sample["by_outcome"] == {
        "EMPTY_BODY": 2, "GATED": 190, "HEAD_MOVED": 100, "PR_GONE": 8}
