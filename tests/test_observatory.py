# -*- coding: utf-8 -*-
"""styxx.observatory — baseline fixed until a reasoned rebaseline; chain, bytes, coefficients and verdicts
re-derived; a forged verdict, a truncated log and an empty reason are caught (v1, 2026-09-13: the red team
showed v0's verify() accepted all three)."""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np
import pytest

from styxx import checksum as ck
from styxx.observatory import Observatory, _line_hash
from tests.test_checksum import _scripted_probe


def test_an_observatory_names_its_tokenization():
    with pytest.raises(ValueError):
        Observatory("unused", "scripted-model")


def _three_days(tmp_path):
    obs = Observatory(str(tmp_path), "scripted-model", tokenizer_id="scripted")
    d1 = obs.observe(_scripted_probe(1), n_null=2, when="2026-09-13T00:00:00Z")
    d2 = obs.observe(_scripted_probe(1), n_null=2, when="2026-09-14T00:00:00Z")
    d3 = obs.observe(_scripted_probe(1, perturb=0.2), n_null=2, when="2026-09-15T00:00:00Z", note="perturbed")
    return obs, d1, d2, d3


def test_three_days_same_same_drift_and_a_verified_chain(tmp_path):
    obs, d1, d2, d3 = _three_days(tmp_path)
    assert d1["kind"] == "baseline"
    assert d2["vs_baseline"]["verdict"] == "SAME" and d2["vs_previous"]["verdict"] == "SAME"
    assert d3["vs_baseline"]["verdict"] == "DRIFT" and d3["vs_previous"]["verdict"] == "DRIFT"
    v = Observatory.verify(str(tmp_path), expect_head=d3["entry_hash"], expect_entries=3)
    assert v["ok"], v
    assert v["entries"] == 3 and "head pin" in v["checks"] and "plate bytes" in v["not_checked"]
    assert obs.baseline()["seq"] == 1
    obs.rebaseline("accepted the new weights after review")
    assert obs.baseline()["seq"] == 4
    s = Observatory.status(str(tmp_path))
    assert "DRIFT" in s and "taken" in s and "perturbed" in s


def test_every_line_carries_the_label_and_the_clock_and_both_floors(tmp_path):
    obs, d1, d2, d3 = _three_days(tmp_path)
    for d in (d1, d2, d3):
        assert d["when"].startswith("2026-09-1") and d["taken"] != d["when"]
        assert d["floor_applied_nats"] == max(d["null_floor_nats"], ck.RESOLUTION_NATS)


def test_the_logged_coefficients_and_verdicts_are_re_derivable_from_the_files(tmp_path):
    # v0 hashed the in-memory array and the logged hash never matched the file; v1 hashes what is on disk
    obs, d1, d2, d3 = _three_days(tmp_path)
    from styxx.geoplate import coefficients, coefficients_sha256
    from styxx.observatory import _load_fp_file
    for d in (d1, d2, d3):
        fp = _load_fp_file(str(tmp_path / d["fingerprint"]))
        assert coefficients_sha256(coefficients(fp.rdm)) == d["coefficients_sha256"]


def _rehash_forward(path):
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    prev = "0" * 64
    for e in lines:
        e["prev"] = prev
        body = {k: v for k, v in e.items() if k != "entry_hash"}
        e["entry_hash"] = _line_hash(prev, body)
        prev = e["entry_hash"]
    path.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in lines), encoding="utf-8")
    return prev


def test_a_forward_rehashed_forgery_of_a_verdict_is_caught(tmp_path):
    obs, d1, d2, d3 = _three_days(tmp_path)
    p = tmp_path / "log.jsonl"
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines[2]["vs_baseline"]["verdict"] = "SAME"
    lines[2]["vs_baseline"]["mean_abs_nats"] = 0.0
    p.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in lines), encoding="utf-8")
    head = _rehash_forward(p)
    v = Observatory.verify(str(tmp_path))
    assert not v["ok"]
    assert (3, "vs_baseline") in v["problems"]
    assert not any(kind == "chain" for _, kind in v["problems"])   # the chain was rebuilt cleanly; the bytes still say no
    assert v["head"] == head


def test_a_truncated_log_is_a_valid_chain_until_the_head_or_the_count_is_pinned(tmp_path):
    obs, d1, d2, d3 = _three_days(tmp_path)
    p = tmp_path / "log.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    assert Observatory.verify(str(tmp_path))["ok"]                                    # the finding, stated
    v = Observatory.verify(str(tmp_path), expect_head=d3["entry_hash"])
    assert not v["ok"] and any(k == "head" for k, _ in v["problems"])
    v = Observatory.verify(str(tmp_path), expect_entries=3)
    assert not v["ok"] and any("entries" in str(what) for _, what in v["problems"])


def test_a_tampered_line_or_fingerprint_is_caught(tmp_path):
    obs, d1, d2, d3 = _three_days(tmp_path)
    p = tmp_path / "log.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"n_null":2', '"n_null":9')
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    v = Observatory.verify(str(tmp_path))
    assert not v["ok"] and any(kind == "chain" for _, kind in v["problems"])
    fp = tmp_path / d2["fingerprint"]
    fp.write_bytes(fp.read_bytes().replace(b"mean_lp", b"mean_LP", 1))
    v = Observatory.verify(str(tmp_path))
    assert (2, "fingerprint bytes") in v["problems"]


def test_an_empty_rebaseline_reason_is_refused(tmp_path):
    obs, *_ = _three_days(tmp_path)
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            obs.rebaseline(bad)
    assert obs.baseline()["seq"] == 1


def test_a_v0_log_fails_v1_verification_by_design(tmp_path):
    # the committed demo under papers/checksum/observatory_demo is a v0 log: no floor_applied, coefficients
    # hashed from memory. v1 names both, rather than passing it.
    v = Observatory.verify("papers/checksum/observatory_demo")
    assert not v["ok"]
    whats = {what for _, what in v["problems"]}
    assert "coefficients" in whats and any("v0" in str(w) for w in whats)


def _topk_of(seed, k=8, wobble=0.0):
    def fn(prompt):
        rng = np.random.default_rng((zlib.crc32(prompt.encode()) ^ seed) & 0xFFFFFFFF)
        toks = [" the", " a", " Paris", " one", " two", " blue", " yes", " no", " green", " red"][:k]
        lp = np.sort(rng.normal(-2, 1, size=k))[::-1] + wobble * np.random.default_rng(1).normal(size=k)
        return list(zip(toks, lp))
    return fn


def test_topk_fingerprint_is_comparable_with_itself_and_sees_drift():
    a = ck.fingerprint_topk(_topk_of(1), "api-model")
    b = ck.fingerprint_topk(_topk_of(1), "api-model")
    c = ck.fingerprint_topk(_topk_of(2), "api-model")
    assert a.kind == "topk" and a.k == 8
    assert ck.distance(a, b, n_boot=100).verdict == "SAME"
    assert ck.distance(a, c, n_boot=100).verdict == "DRIFT"


def test_topk_at_different_k_and_topk_versus_full_are_refused():
    a = ck.fingerprint_topk(_topk_of(1, k=8), "api-model")
    b = ck.fingerprint_topk(_topk_of(1, k=10), "api-model")
    with pytest.raises(ValueError):
        ck.distance(a, b, n_boot=50)
    full = ck.fingerprint(_scripted_probe(1), "local-model", tokenizer_id="topk")
    assert full.kind == "full"
    with pytest.raises(ValueError):
        ck.distance(full, a, n_boot=50)
