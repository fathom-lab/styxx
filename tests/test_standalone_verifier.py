# -*- coding: utf-8 -*-
"""Cross-validation kill-gate for the standalone (zero-styxx-import) verifier.

These tests ARE the pre-registered kill-gate from
scripts/dogfood/PREREG_standalone_verifier.md:

  K1 (decisive) — cross-implementation digest agreement: the standalone
       verifier (scripts/styxx_verify_standalone.py, stdlib only) recomputes
       every per-attestation + chain-link digest and matches the library
       byte-for-byte, over a 4-shape corpus (plain / pinned / vitals / chain).
  K2 — tamper detection without styxx: flipped digest / broken chain -> FAIL.
  K3 — scope: the standalone verifier shares no code with styxx (asserted by
       inspecting its imports) and does not assert semantic properties.

P4 boundary: a re-sealed chain passes structure but is caught with
--expected-head.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
from pathlib import Path

import pytest

from styxx.attestation import attest, attest_chain

REPO = Path(__file__).resolve().parents[1]
STANDALONE = REPO / "scripts" / "styxx_verify_standalone.py"


def _load_standalone():
    spec = importlib.util.spec_from_file_location("styxx_verify_standalone", STANDALONE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SV = _load_standalone()


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


@pytest.fixture
def substrate(tmp_path):
    repo = tmp_path / "substrate"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    (repo / "notes.md").write_text(
        "the pipeline emits a receipt token\n", encoding="utf-8"
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return repo, head


REPORT = 'The version is 1.2.3.\nThe file notes.md contains "receipt token".\n'
VITALS_PROMPT = "Report on the version and the notes file."


# ---- K3: the standalone verifier imports nothing from styxx -----------------

def test_K3_standalone_imports_no_styxx():
    tree = ast.parse(STANDALONE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "styxx" not in imported, f"standalone must not import styxx; got {imported}"
    assert imported <= {"argparse", "hashlib", "json", "sys"}, imported


# ---- K1 (decisive): byte-for-byte digest agreement on the 4-shape corpus ----

def test_K1_plain_attestation_digest_agrees(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo).artifact
    assert SV.compute_digest(art) == art["digest"]["value"]


def test_K1_pinned_attestation_digest_agrees(substrate):
    repo, head = substrate
    art = attest(REPORT, repo, ref=head).artifact
    assert SV.compute_digest(art) == art["digest"]["value"]


def test_K1_vitals_attestation_digest_agrees(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo, prompt=VITALS_PROMPT, vitals=True).artifact
    assert art.get("vitals") is not None
    assert SV.compute_digest(art) == art["digest"]["value"]


def test_K1_chain_digests_agree(substrate):
    repo, head = substrate
    chain = attest_chain(
        [(REPORT, head, VITALS_PROMPT), (REPORT, head, VITALS_PROMPT)],
        repo, vitals=True,
    ).artifact
    prev = SV.CHAIN_GENESIS
    for link in chain["links"]:
        att = link["attestation"]
        att_digest = SV.compute_digest(att)
        assert att_digest == link["attestation_digest"]
        assert link["prev_chain_digest"] == prev
        assert link["chain_digest"] == SV.chain_digest(prev, att_digest)
        prev = link["chain_digest"]
    assert chain["head_chain_digest"] == prev


def test_K1_standalone_verify_chain_ok(substrate):
    repo, head = substrate
    chain = attest_chain([(REPORT, head, None)], repo).artifact
    out = []
    assert SV.verify_chain(chain, out, expected_head=chain["head_chain_digest"]) is True


# ---- K2: tamper detection using only the standalone verifier -----------------

def test_K2_flipped_digest_fails(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo).artifact
    art["digest"]["value"] = "0" * 64
    out = []
    assert SV.verify_attestation(art, out) is False


def test_K2_broken_chain_link_fails(substrate):
    repo, head = substrate
    chain = attest_chain([(REPORT, head, None), (REPORT, head, None)], repo).artifact
    chain["links"][1]["chain_digest"] = "0" * 64
    out = []
    assert SV.verify_chain(chain, out) is False


def test_K2_reordered_chain_fails(substrate):
    repo, head = substrate
    r2 = "The version is 1.2.3.\n"
    chain = attest_chain([(REPORT, head, None), (r2, head, None)], repo).artifact
    chain["links"][0], chain["links"][1] = chain["links"][1], chain["links"][0]
    out = []
    assert SV.verify_chain(chain, out) is False


# ---- P4: re-sealed chain passes structure, caught only by anchor -------------

def test_P4_resealed_chain_needs_anchor(substrate):
    repo, head = substrate
    chain = attest_chain([(REPORT, head, None)], repo).artifact
    good_head = chain["head_chain_digest"]
    # Tamper the substrate then fully re-seal the chain from scratch.
    link = chain["links"][0]
    link["attestation"]["report"] = "The version is 9.9.9.\n"
    att_digest = SV.compute_digest(link["attestation"])
    link["attestation"]["digest"]["value"] = att_digest
    link["attestation_digest"] = att_digest
    new_chain_digest = SV.chain_digest(SV.CHAIN_GENESIS, att_digest)
    link["chain_digest"] = new_chain_digest
    chain["head_chain_digest"] = new_chain_digest

    out = []
    # Structure alone: internally consistent -> passes (honest boundary).
    assert SV.verify_chain(chain, out) is True
    # Anchored to the original head: re-seal is caught.
    out2 = []
    assert SV.verify_chain(chain, out2, expected_head=good_head) is False


# ---- transparency-log proofs: standalone reimpl agrees with styxx ----------

def test_standalone_tlog_primitives_match_library():
    from styxx import transparency as T
    assert SV.leaf_hash("e0") == T.leaf_hash("e0")
    assert SV.node_hash("a" * 64, "b" * 64) == T.node_hash("a" * 64, "b" * 64)
    for n in (1, 2, 5, 13, 33):
        leaves = [SV.leaf_hash(f"e{i}") for i in range(n)]
        assert SV.merkle_tree_hash(leaves) == T.merkle_tree_hash(leaves)


def test_standalone_verifies_inclusion_and_catches_tamper():
    from styxx import transparency as T
    log = T.TransparencyLog([f"e{i}" for i in range(20)])
    for i in (0, 9, 19):
        out = []
        assert SV.verify_inclusion(log.inclusion_proof(i), out) is True
    bad = dict(log.inclusion_proof(7), leaf_hash="0" * 64)
    out = []
    assert SV.verify_inclusion(bad, out) is False


def test_standalone_consistency_catches_rewrite_against_witnessed_root():
    from styxx import transparency as T
    n, m = 23, 15
    witnessed = T.TransparencyLog([f"e{i}" for i in range(m)]).root()
    good = T.TransparencyLog([f"e{i}" for i in range(n)]).consistency_proof(m)
    out = []
    assert SV.verify_consistency(good, out, first_root=witnessed) is True
    edited = T.TransparencyLog([f"e{i}" for i in range(n)])
    edited.entries[3] = "TAMPERED"
    out = []
    assert SV.verify_consistency(edited.consistency_proof(m), out, first_root=witnessed) is False
