# -*- coding: utf-8 -*-
"""styxx.checksum — the instrument's own positive controls, pinned before it touches a model.

Scripted probes with a known truth: identical probes must read SAME at distance exactly 0; a
small perturbation must read as a small drift and a different model as a large one, in that
order; a probe with no variation across items must read INCONCLUSIVE, never SAME; fingerprints
on different canary sets must refuse to compare. If any of these ever fails, no verdict the
instrument gives on a real model means anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from styxx import checksum as fp


def _scripted_probe(seed: int, perturb: float = 0.0, V: int = 256):
    """A deterministic 'model': per-item behavior fixed by the seed; perturb adds a fixed-seed wobble."""
    def probe(prompt: str, continuation: str) -> fp.Probe:
        h = abs(hash((seed, prompt, continuation))) % (2**32)
        rng = np.random.default_rng(h)
        cont = list(rng.normal(-2.0, 0.8, size=3))
        beliefs = rng.normal(0, 1, size=V)
        if perturb:
            wr = np.random.default_rng(h ^ 0xABCDEF)
            cont = [c + perturb * wr.normal() for c in cont]
            beliefs = beliefs + perturb * wr.normal(size=V)
        return fp.Probe(cont_logprobs=cont, next_logprobs=beliefs)
    return probe


def test_same_probe_reads_same_at_distance_zero():
    a = fp.fingerprint(_scripted_probe(1), "m1")
    b = fp.fingerprint(_scripted_probe(1), "m1-again")
    d = fp.distance(a, b, n_boot=200)
    assert d.mean_abs_nats == 0.0
    assert d.verdict == "SAME"
    assert d.rdm_r == pytest.approx(1.0)


def test_perturbation_is_smaller_than_a_different_model_and_both_read_drift():
    a = fp.fingerprint(_scripted_probe(1), "m1")
    p = fp.fingerprint(_scripted_probe(1, perturb=0.05), "m1-perturbed")
    o = fp.fingerprint(_scripted_probe(2), "m2")
    dp = fp.distance(a, p, n_boot=200)
    do = fp.distance(a, o, n_boot=200)
    assert dp.verdict == "DRIFT" and do.verdict == "DRIFT"
    assert dp.mean_abs_nats < do.mean_abs_nats
    assert dp.rdm_r > 0.9 > do.rdm_r


def test_degenerate_probe_is_inconclusive_never_same():
    def flat(prompt, continuation):
        return fp.Probe(cont_logprobs=[-1.0, -1.0], next_logprobs=np.zeros(64))
    a = fp.fingerprint(flat, "flat-a")
    b = fp.fingerprint(flat, "flat-b")
    d = fp.distance(a, b, n_boot=50)
    assert d.verdict == "INCONCLUSIVE"


def test_different_canary_sets_refuse_to_compare():
    a = fp.fingerprint(_scripted_probe(1), "m1")
    b = fp.fingerprint(_scripted_probe(1), "m1", canaries=fp.CANARIES[:10])
    with pytest.raises(ValueError):
        fp.distance(a, b, n_boot=10)


def test_cert_digest_is_over_the_body():
    a = fp.fingerprint(_scripted_probe(1), "m1")
    b = fp.fingerprint(_scripted_probe(3), "m3")
    c = fp.cert(a, b, fp.distance(a, b, n_boot=100))
    assert c["canary_sha256"] == fp.canary_sha256()
    assert len(c["digest"]) == 64
