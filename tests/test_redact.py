# -*- coding: utf-8 -*-
"""Kill-gate for Redactable Cognometric Attestation (selective disclosure).

These tests ARE the pre-registered kill-gate from
scripts/dogfood/PREREG_redactable_attestation.md:

  K1 — disclosure soundness: a tampered value / wrong salt / wrong root FAILS;
       a genuine disclosure verifies. Python and JS agree.
  K2 (decisive) — confidentiality: undisclosed values (esp. the response text)
       do not appear in a disclosure, AND the salt is load-bearing — an unsalted
       small-domain leaf is brute-forceable, a salted one is not.
  K3 — additivity (legacy + portable digest unchanged; no existing regression)
       and cross-language agreement (Python <-> JS) on the root and every
       disclosure.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from styxx.attestation import _compute_digest, _compute_portable_digest, attest
from styxx.redact import (
    _leaf_hash,
    disclose,
    flatten,
    redactable_commit,
    verify_disclosure,
)

REPO = Path(__file__).resolve().parents[1]
JS_VERIFIER = REPO / "web" / "styxx_verify.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")


SECRET = "a long private response with confidential reasoning the agent will not reveal"
OBJ = {
    "report": SECRET,
    "summary": {"coverage": 1.0, "claims": 3, "passed": 2},
    "vitals": {"scores": {"sycophancy": 0.08, "overconfidence": 0.21}},
    "claims": [{"id": "C1", "verdict": "PASS"}, {"id": "C2", "verdict": "FAIL"}],
}


# ---- K1: disclosure soundness ---------------------------------------------

def test_K1_genuine_disclosure_verifies():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals/scores/sycophancy"])
    assert verify_disclosure(d)
    assert verify_disclosure(d, root=c["root"])


def test_K1_subtree_and_multi_field_disclosure():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals", "claims/0/verdict"])
    ptrs = {f["pointer"] for f in d["fields"]}
    assert ptrs == {"vitals/scores/sycophancy", "vitals/scores/overconfidence", "claims/0/verdict"}
    assert verify_disclosure(d, root=c["root"])


def test_K1_tampered_value_fails():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals/scores/sycophancy"])
    bad = copy.deepcopy(d)
    bad["fields"][0]["value"] = 0.999
    assert not verify_disclosure(bad, root=c["root"])


def test_K1_wrong_salt_fails():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals/scores/sycophancy"])
    bad = copy.deepcopy(d)
    bad["fields"][0]["salt"] = "00" * 32
    assert not verify_disclosure(bad, root=c["root"])


def test_K1_wrong_root_fails():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals/scores/sycophancy"])
    assert not verify_disclosure(d, root="f" * 64)


def test_K1_swapped_index_fails():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals"])
    bad = copy.deepcopy(d)
    bad["fields"][0]["leaf_index"], bad["fields"][1]["leaf_index"] = (
        bad["fields"][1]["leaf_index"], bad["fields"][0]["leaf_index"],
    )
    assert not verify_disclosure(bad, root=c["root"])


# ---- K2 (decisive): confidentiality + the salt is load-bearing -------------

def test_K2_undisclosed_text_not_in_disclosure():
    c = redactable_commit(OBJ)
    d = disclose(OBJ, c, ["vitals/scores/sycophancy"])
    blob = json.dumps(d, ensure_ascii=False)
    assert SECRET not in blob
    assert "report" not in {f["pointer"].split("/")[0] for f in d["fields"]}
    # the only revealed pointer/value is the one we chose
    assert [f["pointer"] for f in d["fields"]] == ["vitals/scores/sycophancy"]


def test_K2_salt_is_load_bearing_brute_force():
    # An attacker knows the small domain of a verdict field and the pointer.
    ptr = "claims/0/verdict"
    domain = ["PASS", "FAIL", "ERROR"]
    truth = "FAIL"

    # WITHOUT a salt, the leaf is recovered by hashing the domain candidates.
    unsalted = _leaf_hash("", ptr, truth)
    recovered = [v for v in domain if _leaf_hash("", ptr, v) == unsalted]
    assert recovered == [truth], "unsalted leaf should be brute-forceable"

    # WITH a 256-bit salt (unknown to the attacker), brute force over the domain
    # with an empty/guessed salt cannot recover the value.
    import secrets
    salt = secrets.token_hex(32)
    salted = _leaf_hash(salt, ptr, truth)
    recovered_salted = [v for v in domain if _leaf_hash("", ptr, v) == salted]
    assert recovered_salted == [], "salted leaf must NOT be brute-forceable without the salt"


def test_K2_same_value_different_salts_unlinkable():
    # Two commitments to the same object differ (salts are fresh), so equal
    # underlying values do not produce equal leaf hashes / roots.
    a = redactable_commit(OBJ)
    b = redactable_commit(OBJ)
    assert a["root"] != b["root"]
    assert a["salts"] != b["salts"]


# ---- K3: additivity + cross-language ---------------------------------------

def test_K3_additive_legacy_and_portable_unchanged():
    plain = attest("The version is 1.2.3.", REPO).artifact
    red = attest("The version is 1.2.3.", REPO, redactable=True).artifact
    # legacy + portable exclude the whole digest key, so they are identical
    assert _compute_digest(plain) == _compute_digest(red)
    assert _compute_portable_digest(plain) == _compute_portable_digest(red)
    # the public artifact carries only the root, never the salts
    assert "salts" not in red["digest"]["redactable"]
    assert "root" in red["digest"]["redactable"]


def test_K3_non_redactable_disclose_raises():
    a = attest("The version is 1.2.3.", REPO)
    with pytest.raises(ValueError):
        a.disclose(["summary"])


def test_flatten_is_pointer_sorted_and_canonical():
    leaves = flatten(OBJ)
    ptrs = [p for p, _ in leaves]
    assert ptrs == sorted(ptrs)
    # every scalar leaf is present, nested ones flattened
    assert "vitals/scores/sycophancy" in ptrs
    assert "claims/1/verdict" in ptrs


def _node_verify_disclosure(disclosure, root):
    script = (
        f"const v=require({json.dumps(str(JS_VERIFIER))});"
        "const o=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(v.verifyDisclosure(o.disclosure, o.root) ? 'OK':'FAIL');"
    )
    r = subprocess.run([NODE, "-e", script],
                       input=json.dumps({"disclosure": disclosure, "root": root}, ensure_ascii=False),
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


@needs_node
def test_K3_node_verifies_python_disclosure():
    a = attest("The version is 7.7.12.", REPO, prompt="Report.", vitals=True, redactable=True)
    root = a.artifact["digest"]["redactable"]["root"]
    d = a.disclose(["vitals/scores/sycophancy", "summary/coverage"])
    assert _node_verify_disclosure(d, root) == "OK"
    # Node rejects a tampered disclosure
    bad = copy.deepcopy(d)
    bad["fields"][0]["value"] = "tampered"
    assert _node_verify_disclosure(bad, root) == "FAIL"


@needs_node
def test_K3_node_dispatch_on_kind():
    a = attest("The version is 7.7.12.", REPO, prompt="Report.", vitals=True, redactable=True)
    d = a.disclose(["summary/coverage"])
    script = (
        f"const v=require({json.dumps(str(JS_VERIFIER))});"
        "const o=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(v.verify(o).ok ? 'OK':'FAIL');"
    )
    r = subprocess.run([NODE, "-e", script], input=json.dumps(d, ensure_ascii=False),
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "OK"
