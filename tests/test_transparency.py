# -*- coding: utf-8 -*-
"""Kill-gate for the Cognometric Transparency Log (RFC 6962 for attestations).

These tests ARE the pre-registered kill-gate from
scripts/dogfood/PREREG_transparency_log.md:

  K1 — inclusion soundness + completeness: EVERY real leaf's proof verifies; a
       tampered / non-member leaf FAILS.
  K2 (decisive) — consistency catches a rewrite: any edit / delete / reorder of
       the first m witnessed leaves makes the m->n consistency proof FAIL, while a
       pure append PASSES.
  K3 — cross-language agreement: Python and the JS verifier agree on the root and
       on every inclusion / consistency proof over the real log (Node tests skip
       if node is unavailable).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from styxx.transparency import (
    TransparencyLog,
    leaf_hash,
    verify_consistency,
    verify_inclusion,
)

REPO = Path(__file__).resolve().parents[1]
JS_VERIFIER = REPO / "web" / "styxx_verify.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")

SIZES = list(range(1, 34)) + [64, 100, 129]


def _log(n: int) -> TransparencyLog:
    return TransparencyLog([f"entry-{i:04d}" for i in range(n)])


# ---- K1: inclusion is sound AND complete ----------------------------------

def test_K1_every_leaf_inclusion_verifies():
    for n in SIZES:
        log = _log(n)
        for i in range(n):
            p = log.inclusion_proof(i)
            assert verify_inclusion(p), f"genuine inclusion failed n={n} i={i}"


def test_K1_tampered_leaf_fails():
    log = _log(20)
    for i in (0, 7, 19):
        p = log.inclusion_proof(i)
        bad = dict(p, leaf_hash="0" * 64)
        assert not verify_inclusion(bad), f"tampered leaf_hash verified n=20 i={i}"
        # wrong root must also fail
        assert not verify_inclusion(p, root="f" * 64)


def test_K1_nonmember_leaf_fails():
    log = _log(16)
    p = log.inclusion_proof(5)
    # claim a leaf that is not the one at index 5
    forged = dict(p, leaf_hash=leaf_hash("entry-9999"))
    assert not verify_inclusion(forged)


def test_K1_out_of_range_index_raises():
    log = _log(8)
    with pytest.raises(IndexError):
        log.inclusion_proof(8)
    with pytest.raises(IndexError):
        log.inclusion_proof(-1)


# ---- K2 (decisive): consistency catches any rewrite of witnessed history ----

def test_K2_genuine_append_passes():
    for n in SIZES:
        for m in range(0, n + 1):
            log = _log(n)
            witnessed = _log(m).root()
            c = log.consistency_proof(m)
            assert verify_consistency(c, first_root=witnessed), (
                f"genuine append rejected n={n} m={m}"
            )


def test_K2_edit_of_past_leaf_caught():
    n, m = 23, 15
    witnessed = _log(m).root()
    edited = _log(n)
    edited.entries[3] = "TAMPERED"
    c = edited.consistency_proof(m)
    assert not verify_consistency(c, first_root=witnessed), "edit of past leaf not caught"


def test_K2_delete_of_past_leaf_caught():
    n, m = 23, 15
    witnessed = _log(m).root()
    deleted = TransparencyLog([f"entry-{i:04d}" for i in range(n) if i != 3])
    c = deleted.consistency_proof(m)
    assert not verify_consistency(c, first_root=witnessed), "delete of past leaf not caught"


def test_K2_reorder_of_past_leaves_caught():
    n, m = 23, 15
    witnessed = _log(m).root()
    reordered = _log(n)
    reordered.entries[2], reordered.entries[5] = reordered.entries[5], reordered.entries[2]
    c = reordered.consistency_proof(m)
    assert not verify_consistency(c, first_root=witnessed), "reorder of past leaves not caught"


def test_K2_truncation_caught():
    # operator drops the most recent witnessed leaf (suppression of a bad receipt)
    n, m = 20, 12
    witnessed = _log(m).root()
    truncated = TransparencyLog([f"entry-{i:04d}" for i in range(n) if i != 11])
    c = truncated.consistency_proof(m)
    assert not verify_consistency(c, first_root=witnessed), "truncation not caught"


def test_K2_self_consistent_proof_verifies_without_witness():
    # the proof carries its own roots; without an external witness it self-checks
    log = _log(30)
    c = log.consistency_proof(18)
    assert verify_consistency(c)


# ---- tree head ------------------------------------------------------------

def test_tree_head_content_addressed_and_deterministic():
    log = _log(10)
    ts = "2026-05-28T00:00:00+00:00"
    a = log.tree_head(timestamp=ts)
    b = log.tree_head(timestamp=ts)
    assert a == b
    assert a["size"] == 10
    assert a["root"] == log.root()
    # the digest pins {size, root, timestamp}
    assert len(a["digest"]["value"]) == 64


def test_empty_log_root_is_sha256_empty():
    import hashlib
    assert TransparencyLog([]).root() == hashlib.sha256(b"").hexdigest()


# ---- K3: cross-language agreement (Python <-> JS) --------------------------

def _node(script: str, payload) -> str:
    full = (
        f"const v=require({json.dumps(str(JS_VERIFIER))});"
        "const I=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        + script
    )
    r = subprocess.run(
        [NODE, "-e", full], input=json.dumps(payload, ensure_ascii=False),
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


@needs_node
def test_K3_root_agreement_over_sizes():
    payload = {str(n): _log(n).root() for n in SIZES}
    script = (
        "let bad=0;"
        "for(const n of Object.keys(I)){const ls=[];"
        "for(let i=0;i<Number(n);i++)ls.push(v.leafHash('entry-'+String(i).padStart(4,'0')));"
        "if(v.merkleTreeHash(ls)!==I[n])bad++;}"
        "process.stdout.write(String(bad));"
    )
    assert _node(script, payload) == "0", "Python/JS root mismatch"


@needs_node
def test_K3_node_verifies_inclusion_and_catches_tamper():
    log = _log(29)
    proofs = [log.inclusion_proof(i) for i in range(29)]
    script = (
        "let bad=0;"
        "for(const p of I)if(!v.verifyInclusion(p))bad++;"
        "const t=Object.assign({},I[5]);t.leaf_hash='0'.repeat(64);"
        "if(v.verifyInclusion(t))bad++;"
        "process.stdout.write(String(bad));"
    )
    assert _node(script, proofs) == "0", "Node inclusion verify / tamper mismatch"


@needs_node
def test_K3_node_verifies_consistency_and_catches_rewrite():
    n, m = 26, 17
    witnessed = _log(m).root()
    good = _log(n).consistency_proof(m)
    edited = _log(n)
    edited.entries[4] = "TAMPERED"
    bad = edited.consistency_proof(m)
    payload = {"good": good, "bad": bad, "witnessed": witnessed}
    script = (
        "let bad=0;"
        "if(!v.verifyConsistency(I.good, I.witnessed))bad++;"
        "if(v.verifyConsistency(I.bad, I.witnessed))bad++;"
        "process.stdout.write(String(bad));"
    )
    assert _node(script, payload) == "0", "Node consistency verify / rewrite mismatch"


@needs_node
def test_K3_node_verify_dispatches_log_proofs():
    log = _log(12)
    incl = log.inclusion_proof(7)
    cons = log.consistency_proof(5)
    for proof in (incl, cons):
        script = "process.stdout.write(v.verify(I).ok ? 'OK':'FAIL');"
        assert _node(script, proof) == "OK"
