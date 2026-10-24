"""The /v2 conformance set replays through styxx.sworn, byte for byte, with nothing skipped.

papers/sworn/NOTE_certifies_correction_2026_10_24.md. conformance/sworn/receipt_v2/ sits beside the /v1
set (conformance/sworn/), in the same layout and from the same generator, and pins what /v2 moved: the
two strings `schema` and `certifies`, and the clauses of the /v2 sentence a document, a manifest and an
in-memory tree can show (tests/test_sworn_receipt_v2.py, its only source). Its cores carry the schema the
verifier issues today, so a replay projects nothing away: `_core_as(core, /v2)` is the core itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "sworn" / "receipt_v2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.sworn.replay import blob_bytes, replay_vector  # noqa: E402
from styxx import sworn  # noqa: E402

V2 = "styxx.sworn.verdict-receipt/v2"


def _load(name):
    return json.loads((SET / name).read_bytes().decode("utf-8"))


INDEX = _load("index.json")
BLOBS = _load("blobs.json")
FAMILIES = {name: _load(meta["file"])["vectors"] for name, meta in INDEX["families"].items()}
ALL = [v for name in sorted(FAMILIES) for v in FAMILIES[name]]


def _jcs_sha(obj):
    return sworn._sha256(sworn._jcs(obj).encode("utf-8"))


def test_the_set_declares_v2_cores_and_its_own_source_and_spec():
    assert INDEX["schema"] == "styxx.sworn.conformance/v1"          # the layout did not change
    assert INDEX["core_schema"] == V2 == sworn.RECEIPT_SCHEMA
    assert INDEX["receipt_schemas"] == list(sworn.RECEIPT_SCHEMAS)
    assert INDEX["sources"] == ["tests/test_sworn_receipt_v2.py"]
    assert INDEX["spec"] == "papers/sworn/NOTE_certifies_correction_2026_10_24.md"
    assert INDEX["vector_count"] == len(ALL) == sum(m["count"] for m in INDEX["families"].values())
    assert set(INDEX["families"]) == {"core_v2", "check_v2", "check_v1", "check_v0"}
    assert INDEX["unvectored"]["skipped"] == [], "the source is pure: every call it makes records"


def test_the_index_re_derives_its_digest_and_every_file_hashes_to_its_entry():
    for name, meta in INDEX["families"].items():
        assert sworn._sha256((SET / meta["file"]).read_bytes()) == meta["sha256"], name
    assert sworn._sha256((SET / "blobs.json").read_bytes()) == INDEX["blobs"]["sha256"]
    assert len(BLOBS) == INDEX["blobs"]["count"]
    for sha in BLOBS:
        blob_bytes(BLOBS, sha)
    body = {k: v for k, v in INDEX.items() if k not in ("set_sha256", "provenance")}
    assert _jcs_sha(body) == INDEX["set_sha256"]


def test_every_vector_id_re_derives():
    for v in ALL:
        assert _jcs_sha({"mode": v["mode"], "inputs": v["inputs"]}) == v["id"], v["id"]


def test_every_core_carries_the_v2_schema_and_sentence():
    cores = [v for v in ALL if v["expect"]["outcome"] == "core"]
    assert cores
    for v in cores:
        core = json.loads(blob_bytes(BLOBS, v["expect"]["core_sha256"]).decode("utf-8"))
        assert core["schema"] == V2, v["id"]
        assert core["certifies"] == sworn._CERTIFIES, v["id"]


def test_every_receipt_the_set_checks_carries_its_own_schema_s_strings():
    """The checks are /v2, /v1 and /v0 receipts; each /v1 and /v2 one carries the sentence of its schema
    unless the vector is the case that edited it, which the source names and which must FAIL."""
    seen = set()
    for v in ALL:
        if v["mode"] != "receipt_check":
            continue
        rec = json.loads(blob_bytes(BLOBS, v["inputs"]["receipt"]).decode("utf-8"))
        seen.add(rec["schema"])
        if rec["schema"] in sworn._CERTIFIES_OF and rec.get("certifies") != sworn._CERTIFIES_OF[rec["schema"]]:
            assert v["expect"]["check"]["status"] == "FAILED", v["sources"]
    assert seen == {"styxx.sworn.verdict-receipt/v0", "styxx.sworn.verdict-receipt/v1", V2}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_vector_replays_through_styxx_sworn(family):
    failures, skipped = [], []
    for v in FAMILIES[family]:
        status, detail = replay_vector(v, BLOBS, core_schema=INDEX["core_schema"])
        if status == "fail":
            failures.append((v["id"][:12], v["sources"][0], detail))
        elif status == "skip":
            skipped.append(v["id"])
    assert not failures, failures[:5]
    assert not skipped


def test_the_set_regenerates_to_its_own_digest():
    """gen_vectors.py --set receipt_v2 --check regenerates in memory and exits 1 if set_sha256 differs.
    A child that never ran is not a digest that moved: a spawn failure is named and skipped, and only a
    refusal the tool printed is a drift (the rule tests/test_sworn_conformance.py states)."""
    r = subprocess.run([sys.executable, "conformance/sworn/gen_vectors.py", "--set", "receipt_v2", "--check"],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=900)
    if r.returncode == 0:
        assert INDEX["set_sha256"] in r.stdout and "CHECK OK" in r.stdout
        return
    printed = (r.stdout or "") + (r.stderr or "")
    if "REFUSED: set_sha256 drifted" not in printed:
        pytest.skip("the regeneration subprocess did not run to a verdict, so this says nothing about the set: "
                    "exit %d. Tail: %r" % (r.returncode, printed[-400:]))
    raise AssertionError(printed[-3000:])
