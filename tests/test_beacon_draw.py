# -*- coding: utf-8 -*-
"""The beacon draw, wired end to end: a draw returns its record, a fingerprint carries it and refuses a
record copied onto other items, two fingerprints compare only under the same draw, the cert digests it,
the observatory loads it back from the file, and the runner refuses to call a beacon-drawn run the
sealed experiment (the 2026-09-13 PREREG froze the hand set). Before 2026-09-13 the draw was wired into
nothing: no fingerprint, cert or runner carried the pool hash or the beacon."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from styxx import beacon
from styxx import checksum as ck
from styxx.observatory import Observatory, _load_fp_file
from tests.test_checksum import _scripted_probe

B1 = "a" * 64
B2 = "b" * 64


def test_draw_returns_the_items_and_a_record_that_names_them():
    items, rec = beacon.draw(B1, 12)
    assert rec["schema"] == "styxx.beacon/draw/v0"
    assert rec["n"] == 12 and rec["beacon"] == B1 and rec["pool_sha256"] == beacon.pool_sha256()
    assert rec["pool_size"] == len(beacon.POOL)
    assert rec["canary_sha256"] == ck.canary_sha256(items)
    assert [c[0] for c in items] == [c[0] for c in beacon.select(B1, 12)]


def test_the_cli_prints_the_canary_hash_a_prereg_can_cite():
    r = subprocess.run([sys.executable, "-m", "styxx.beacon", B1, "5"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["canary_sha256"] == ck.canary_sha256(beacon.select(B1, 5)) and len(d["ids"]) == 5


def test_a_fingerprint_carries_its_draw_and_refuses_a_record_copied_onto_other_items():
    items, rec = beacon.draw(B1, 12)
    fp = ck.fingerprint(_scripted_probe(1), "m", canaries=items, tokenizer_id="t", draw=rec)
    assert fp.draw == rec and fp.to_json()["draw"] == rec
    other, _ = beacon.draw(B2, 12)
    with pytest.raises(ValueError):
        ck.fingerprint(_scripted_probe(1), "m", canaries=other, tokenizer_id="t", draw=rec)
    with pytest.raises(ValueError):
        ck.fingerprint(_scripted_probe(1), "m", canaries=items, tokenizer_id="t", draw={"canary_sha256": rec["canary_sha256"]})


def test_a_record_whose_beacon_or_pool_lies_is_refused_even_when_its_canary_hash_matches():
    # the canary hash names the items; only re-running the beacon proves the record MADE them
    items, rec = beacon.draw(B1, 12)
    for lie in ({"beacon": B2}, {"pool_sha256": "0" * 64}, {"n": 11}):
        forged = {**rec, **lie}
        with pytest.raises(ValueError):
            ck.fingerprint(_scripted_probe(1), "m", canaries=items, tokenizer_id="t", draw=forged)
    with pytest.raises(ValueError):
        Observatory("unused", "m", canaries=items, tokenizer_id="t", draw={**rec, "beacon": B2})


def test_fingerprints_compare_only_under_the_same_draw():
    items, rec = beacon.draw(B1, 12)
    a = ck.fingerprint(_scripted_probe(1), "m", canaries=items, tokenizer_id="t", draw=rec)
    b = ck.fingerprint(_scripted_probe(1), "m-again", canaries=items, tokenizer_id="t", draw=rec)
    assert ck.distance(a, b, n_boot=50).verdict == "SAME"
    hand = ck.fingerprint(_scripted_probe(1), "m-hand", canaries=items, tokenizer_id="t")   # same items, no record
    with pytest.raises(ValueError):
        ck.distance(a, hand, n_boot=50)
    other, rec2 = beacon.draw(B2, 12)
    c = ck.fingerprint(_scripted_probe(1), "m", canaries=other, tokenizer_id="t", draw=rec2)
    with pytest.raises(ValueError):
        ck.distance(a, c, n_boot=50)


def test_the_cert_digests_the_draw():
    items, rec = beacon.draw(B1, 12)
    a = ck.fingerprint(_scripted_probe(1), "m", canaries=items, tokenizer_id="t", draw=rec)
    b = ck.fingerprint(_scripted_probe(2), "n", canaries=items, tokenizer_id="t", draw=rec)
    c = ck.cert(a, b, ck.distance(a, b, n_boot=50))
    assert c["draw"] == rec
    x = ck.fingerprint(_scripted_probe(1), "m", canaries=items, tokenizer_id="t")
    y = ck.fingerprint(_scripted_probe(2), "n", canaries=items, tokenizer_id="t")
    assert ck.cert(x, y, ck.distance(x, y, n_boot=50))["digest"] != c["digest"]


def test_the_observatory_carries_and_reloads_the_draw(tmp_path):
    items, rec = beacon.draw(B1, 12)
    obs = Observatory(str(tmp_path), "m", canaries=items, tokenizer_id="t", draw=rec)
    e1 = obs.observe(_scripted_probe(1), when="2026-09-13T00:00:00Z")
    e2 = obs.observe(_scripted_probe(1), when="2026-09-14T00:00:00Z")
    assert _load_fp_file(str(tmp_path / e1["fingerprint"])).draw == rec
    assert e2["vs_baseline"]["verdict"] == "SAME"
    assert Observatory.verify(str(tmp_path), expect_entries=2)["ok"]
    with pytest.raises(ValueError):
        Observatory(str(tmp_path / "x"), "m", canaries=beacon.select(B2, 12), tokenizer_id="t", draw=rec)


def test_the_runner_refuses_to_call_a_beacon_drawn_run_the_sealed_experiment():
    r = subprocess.run([sys.executable, "papers/checksum/run_deploy_quant.py", "--beacon", B1],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "not the sealed experiment" in (r.stdout + r.stderr)


def test_the_beacon_draw_prereg_requires_the_beacon_and_a_well_formed_one():
    # the 2026-09-14 PREREG's run draws its canaries from the seal's block hash: no beacon, no run —
    # decided before torch is imported, so this holds on a box without a model stack
    r = subprocess.run([sys.executable, "papers/checksum/run_deploy_quant.py", "--prereg", "beacon_draw"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "draws its canaries from the block hash" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, "papers/checksum/run_deploy_quant.py", "--prereg", "beacon_draw", "--beacon", "xyz"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "64 lowercase hex" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, "papers/checksum/run_deploy_quant.py", "--prereg", "not_a_prereg"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0


def test_the_pool_is_the_one_the_beacon_draw_prereg_froze():
    # PREREG_checksum_beacon_draw_2026_09_14 froze the pool by hash and size; an edit to the pool
    # (a new template family, a fixed typo in a hand item) would silently make every draw under
    # the sealed beacon a different experiment. The PREREG's bytes are frozen; the pool must stay.
    import re
    doc = open("papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.md", encoding="utf-8").read()
    frozen_hash = re.search(r"pool sha256\s+<sworn[^>]*>`([0-9a-f]{64})`</sworn>", doc).group(1)
    frozen_size = int(re.search(r"<sworn[^>]*pool_size[^>]*>(\d+)</sworn> items", doc).group(1))
    assert beacon.pool_sha256() == frozen_hash, "styxx.beacon.POOL is not the pool the beacon-draw PREREG froze"
    assert len(beacon.POOL) == frozen_size
