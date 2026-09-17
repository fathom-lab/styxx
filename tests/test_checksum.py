# -*- coding: utf-8 -*-
"""styxx.checksum — the instrument's own positive controls, pinned before it touches a model.

Scripted probes with a known truth: identical probes must read SAME at distance exactly 0; a
small perturbation must read as a small drift and a different model as a large one, in that
order; a probe with no variation across items — in its log-probs OR in its belief geometry — must
read INCONCLUSIVE, never SAME; fingerprints on different canary sets, different tokenizers, no
tokenizer, different kinds or different items must refuse to compare; a floor that is not a finite
measurement is refused; a cert is strict JSON whose digest covers the written fingerprint hashes.
If any of these ever fails, no verdict the instrument gives on a real model means anything.

The scripted probe is keyed by a stable digest, not Python's per-process hash(): before 2026-09-13
every run of this file pinned different bytes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from styxx import checksum as fp


def _scripted_probe(seed: int, perturb: float = 0.0, V: int = 256):
    """A deterministic 'model': per-item behavior fixed by the seed; perturb adds a fixed-seed wobble."""
    def probe(prompt: str, continuation: str) -> fp.Probe:
        h = int.from_bytes(hashlib.sha256(f"{seed}|{prompt}|{continuation}".encode()).digest()[:4], "big")
        rng = np.random.default_rng(h)
        cont = list(rng.normal(-2.0, 0.8, size=3))
        beliefs = rng.normal(0, 1, size=V)
        if perturb:
            wr = np.random.default_rng(h ^ 0xABCDEF)
            cont = [c + perturb * wr.normal() for c in cont]
            beliefs = beliefs + perturb * wr.normal(size=V)
        return fp.Probe(cont_logprobs=cont, next_logprobs=beliefs)
    return probe


def _fp(seed, model_id="m", perturb=0.0, **kw):
    kw.setdefault("tokenizer_id", "tok")
    return fp.fingerprint(_scripted_probe(seed, perturb=perturb), model_id, **kw)


def test_same_probe_reads_same_at_distance_zero():
    a = _fp(1, "m1")
    b = _fp(1, "m1-again")
    d = fp.distance(a, b, n_boot=200)
    assert d.mean_abs_nats == 0.0
    assert d.verdict == "SAME"
    assert d.rdm_r == pytest.approx(1.0)
    assert d.floor_measured == fp.RESOLUTION_NATS and d.floor_effective == fp.RESOLUTION_NATS and d.seed == 20260913


def test_perturbation_is_smaller_than_a_different_model_and_both_read_drift():
    a = _fp(1, "m1")
    p = _fp(1, "m1-perturbed", perturb=0.05)
    o = _fp(2, "m2")
    dp = fp.distance(a, p, n_boot=200)
    do = fp.distance(a, o, n_boot=200)
    assert dp.verdict == "DRIFT" and do.verdict == "DRIFT"
    assert dp.mean_abs_nats < do.mean_abs_nats
    assert dp.rdm_r > 0.9 > do.rdm_r


def test_degenerate_probe_is_inconclusive_never_same():
    def flat(prompt, continuation):
        return fp.Probe(cont_logprobs=[-1.0, -1.0], next_logprobs=np.zeros(64))
    a = fp.fingerprint(flat, "flat-a", tokenizer_id="tok")
    b = fp.fingerprint(flat, "flat-b", tokenizer_id="tok")
    d = fp.distance(a, b, n_boot=50)
    assert d.verdict == "INCONCLUSIVE"


def test_flat_geometry_with_varying_logprobs_is_inconclusive_never_same():
    # v0 guarded mean_lp only: constant next-token beliefs with varying continuation log-probs read SAME
    def flat_geometry(prompt, continuation):
        h = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
        return fp.Probe(cont_logprobs=list(np.random.default_rng(h).normal(-2, 0.8, size=3)), next_logprobs=np.zeros(64))
    a = fp.fingerprint(flat_geometry, "fg-a", tokenizer_id="tok")
    b = fp.fingerprint(flat_geometry, "fg-b", tokenizer_id="tok")
    assert fp.distance(a, b, n_boot=50).verdict == "INCONCLUSIVE"


def test_different_canary_sets_refuse_to_compare():
    a = _fp(1, "m1")
    b = _fp(1, "m1", canaries=fp.CANARIES[:10])
    with pytest.raises(ValueError):
        fp.distance(a, b, n_boot=10)


def test_a_copied_canary_hash_over_different_items_is_still_refused():
    a = _fp(1, "m1")
    b = _fp(1, "m1")
    b.ids = list(reversed(b.ids))
    with pytest.raises(ValueError):
        fp.distance(a, b, n_boot=10)


def test_different_or_missing_tokenizers_refuse_to_compare():
    a = _fp(1, "m1", tokenizer_id="tok-a")
    b = _fp(1, "m1", tokenizer_id="tok-b")
    with pytest.raises(ValueError):
        fp.distance(a, b, n_boot=10)
    with pytest.raises(ValueError):
        fp.distance(_fp(1, tokenizer_id=""), _fp(1, tokenizer_id=""), n_boot=10)


def test_a_floor_that_is_not_a_finite_measurement_is_refused():
    a, p = _fp(1, "m1"), _fp(1, "m1-perturbed", perturb=0.05)
    for bad in (float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError):
            fp.distance(a, p, n_boot=50, floor_nats=bad)


def test_a_measured_null_floor_turns_a_small_drift_into_inconclusive_not_drift():
    a = _fp(1, "m1")
    p = _fp(1, "m1-perturbed", perturb=0.05)
    without = fp.distance(a, p, n_boot=200)
    assert without.verdict == "DRIFT"
    # a null floor as large as the drift itself: the verdict must lose its confidence, not keep it
    floor = fp.null_floor([a, p])
    with_floor = fp.distance(a, p, n_boot=200, floor_nats=floor)
    assert with_floor.verdict == "INCONCLUSIVE"
    assert with_floor.floor_measured == floor and with_floor.floor_effective == max(floor, fp.RESOLUTION_NATS)


def test_the_written_fingerprint_normalises_signed_zero_and_hashes_both_arrays():
    a = _fp(1, "m1")
    a.rdm = a.rdm.copy()
    a.rdm[0, 0] = -1e-17
    b = _fp(1, "m1")
    b.rdm = b.rdm.copy()
    b.rdm[0, 0] = 1e-17
    ja, jb = a.to_json(), b.to_json()
    assert ja["rdm_sha256"] == jb["rdm_sha256"]
    assert ja["rdm"][0][0] == 0.0 and "-0.0" not in json.dumps(ja["rdm"])
    assert ja["mean_lp_sha256"] == hashlib.sha256(json.dumps(ja["mean_lp"], separators=(",", ":")).encode()).hexdigest()
    assert ja["kind"] == "full" and ja["k"] == 0


def test_cert_digest_is_over_the_comparison_not_the_clock():
    a = _fp(1, "m1")
    b = _fp(3, "m3")
    d = fp.distance(a, b, n_boot=100)
    c1 = fp.cert(a, b, d)
    a.created = "1999-01-01T00:00:00Z"
    c2 = fp.cert(a, b, d)
    assert c1["digest"] == c2["digest"]


def test_cert_digest_covers_the_written_fingerprint_hashes_the_seed_and_the_floors():
    a, b = _fp(1, "m1"), _fp(3, "m3")
    x, y = _fp(2, "m1"), _fp(4, "m3")          # same model_id strings, different bytes
    d = fp.distance(a, b, n_boot=100)
    c = fp.cert(a, b, d)
    assert c["schema"] == "styxx.checksum/compare/v2"
    assert c["draw"] is None
    assert c["a"]["rdm_sha256"] == a.written_hashes()[0] and c["b"]["mean_lp_sha256"] == b.written_hashes()[1]
    assert c["digest"] != fp.cert(x, y, fp.distance(x, y, n_boot=100))["digest"]
    assert c["seed"] == 20260913 and c["floor_effective_nats"] == fp.RESOLUTION_NATS
    assert c["digest"] != fp.cert(a, b, fp.distance(a, b, n_boot=100, seed=7))["digest"]


def test_a_cert_is_strict_json_even_when_inconclusive():
    def flat(prompt, continuation):
        return fp.Probe(cont_logprobs=[-1.0, -1.0], next_logprobs=np.zeros(64))
    a = fp.fingerprint(flat, "flat-a", tokenizer_id="tok")
    b = fp.fingerprint(flat, "flat-b", tokenizer_id="tok")
    c = fp.cert(a, b, fp.distance(a, b, n_boot=50))
    text = json.dumps(c, allow_nan=False)                   # raises on NaN/Infinity
    def refuse(name):
        raise ValueError(name)
    back = json.loads(text, parse_constant=refuse)
    assert back["distance"]["verdict"] == "INCONCLUSIVE" and back["distance"]["rdm_r"] is None
    assert len(c["digest"]) == 64 and c["canary_sha256"] == fp.canary_sha256()
