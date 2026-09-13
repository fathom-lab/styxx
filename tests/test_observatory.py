# -*- coding: utf-8 -*-
"""styxx.observatory — baseline fixed until a human rebaselines; chain and bytes verified; drift surfaces."""
from __future__ import annotations

import numpy as np

from styxx import checksum as ck
from styxx.observatory import Observatory
from tests.test_checksum import _scripted_probe


def test_three_days_same_same_drift_and_a_verified_chain(tmp_path):
    obs = Observatory(str(tmp_path), "scripted-model")
    d1 = obs.observe(_scripted_probe(1), n_null=2, when="2026-09-13T00:00:00Z")
    d2 = obs.observe(_scripted_probe(1), n_null=2, when="2026-09-14T00:00:00Z")
    d3 = obs.observe(_scripted_probe(1, perturb=0.2), n_null=2, when="2026-09-15T00:00:00Z")
    assert d1["kind"] == "baseline"
    assert d2["vs_baseline"]["verdict"] == "SAME" and d2["vs_previous"]["verdict"] == "SAME"
    assert d3["vs_baseline"]["verdict"] == "DRIFT" and d3["vs_previous"]["verdict"] == "DRIFT"
    v = Observatory.verify(str(tmp_path))
    assert v["ok"] and v["entries"] == 3
    # the baseline does not move on its own
    assert obs.baseline()["seq"] == 1
    obs.rebaseline("accepted the new weights after review")
    assert obs.baseline()["seq"] == 4
    assert "DRIFT" in Observatory.status(str(tmp_path))


def test_a_tampered_line_or_fingerprint_is_caught(tmp_path):
    obs = Observatory(str(tmp_path), "scripted-model")
    obs.observe(_scripted_probe(1), when="2026-09-13T00:00:00Z")
    obs.observe(_scripted_probe(1), when="2026-09-14T00:00:00Z")
    p = tmp_path / "log.jsonl"
    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace('"n_null":2', '"n_null":9')
    p.write_text("\n".join(lines) + "\n")
    v = Observatory.verify(str(tmp_path))
    assert not v["ok"] and any(kind == "chain" for _, kind in v["problems"])


def test_topk_fingerprint_is_comparable_with_itself_and_sees_drift():
    def topk_of(seed, wobble=0.0):
        def fn(prompt):
            rng = np.random.default_rng(abs(hash((seed, prompt))) % (2**32))
            toks = [" the", " a", " Paris", " one", " two", " blue", " yes", " no"]
            lp = np.sort(rng.normal(-2, 1, size=8))[::-1] + wobble * np.random.default_rng(1).normal(size=8)
            return list(zip(toks, lp))
        return fn
    a = ck.fingerprint_topk(topk_of(1), "api-model")
    b = ck.fingerprint_topk(topk_of(1), "api-model")
    c = ck.fingerprint_topk(topk_of(2), "api-model")
    assert ck.distance(a, b, n_boot=100).verdict == "SAME"
    assert ck.distance(a, c, n_boot=100).verdict == "DRIFT"
