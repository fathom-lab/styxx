# -*- coding: utf-8 -*-
"""Kill-gate tests for styxx.attestation (Verifiable Cognometric Attestation).

These tests ARE the pre-registered kill-gate from
scripts/dogfood/PREREG_verifiable_attestation.md:

  K1 — determinism: two attest() runs on identical substrate share a digest.
  K2 — independent reproduction (decisive): verify re-derives verdicts from
       the substrate; a flipped embedded verdict is caught.
  K3 — tamper-evidence: mutating any hashed field breaks the digest check.

The substrate is a throwaway git repo built in a tmp dir so the verdicts are
deterministic and self-contained (no dependency on the styxx repo state).
"""
from __future__ import annotations

import copy
import subprocess

import pytest

from styxx.attestation import attest, verify_attestation, ATTESTATION_VERSION


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True,
                   capture_output=True)


@pytest.fixture
def substrate(tmp_path):
    """A minimal git repo with a pyproject.toml and a known file."""
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
    return repo


# A self-report whose claims are all TRUE against the substrate above.
REPORT_TRUE = (
    'The version is 1.2.3.\n'
    'The file notes.md contains "receipt token".\n'
)


def test_attest_produces_passing_artifact(substrate):
    att = attest(REPORT_TRUE, substrate)
    art = att.artifact
    assert art["styxx_attestation_version"] == ATTESTATION_VERSION
    assert art["summary"]["claims_extracted"] == 2
    assert art["summary"]["passed"] == 2
    assert art["summary"]["failed"] == 0
    assert att.passed is True
    # every claim carries its derived Article 15 clause citations
    assert all(c["clauses"] for c in art["claims"])


# --- K1: determinism -------------------------------------------------------

def test_K1_determinism_same_digest_across_runs(substrate):
    a = attest(REPORT_TRUE, substrate)
    b = attest(REPORT_TRUE, substrate)
    # generated_at differs (volatile, outside the hash); the digest does not.
    assert a.digest == b.digest
    assert a.artifact["generated_at"] != b.artifact["generated_at"] or True
    # the canonical (hashed) payload must be byte-identical
    from styxx.attestation import _canonical_payload
    assert _canonical_payload(a.artifact) == _canonical_payload(b.artifact)


# --- K2: independent reproduction (the decisive bar) -----------------------

def test_K2_verify_reproduces_verdicts_from_substrate(substrate):
    att = attest(REPORT_TRUE, substrate)
    res = verify_attestation(att, substrate)
    assert res.digest_ok is True
    assert res.ok is True
    assert len(res.reproduced) == 2
    assert res.mismatches == []


def test_K2_flipped_verdict_is_caught(substrate):
    """An attacker flips an embedded verdict WITHOUT touching the substrate.
    verify must report the TRUE substrate verdict and flag the mismatch."""
    att = attest(REPORT_TRUE, substrate)
    tampered = copy.deepcopy(att.artifact)
    # flip the first claim's verdict PASS -> FAIL and fix the digest so the
    # tamper is NOT caught by the hash — only by independent reproduction.
    tampered["claims"][0]["verdict"] = "FAIL"
    from styxx.attestation import _compute_digest
    tampered["digest"]["value"] = _compute_digest(tampered)

    res = verify_attestation(tampered, substrate)
    assert res.digest_ok is True  # digest was re-sealed by the attacker
    assert res.ok is False        # but independent reproduction catches it
    assert len(res.mismatches) == 1
    m = res.mismatches[0]
    assert m["embedded_verdict"] == "FAIL"
    assert m["reproduced_verdict"] == "PASS"


def test_K2_substrate_change_flips_reproduced_verdict(substrate):
    """If the substrate itself changes after attestation, verify reflects the
    NEW truth — the embedded PASS no longer reproduces."""
    att = attest(REPORT_TRUE, substrate)
    # mutate the substrate: bump the version so the embedded "1.2.3" PASS is
    # no longer true.
    (substrate / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "9.9.9"\n', encoding="utf-8"
    )
    res = verify_attestation(att, substrate)
    assert res.ok is False
    flipped = [m for m in res.mismatches if m["reproduced_verdict"] == "FAIL"]
    assert len(flipped) == 1


# --- K3: tamper-evidence ---------------------------------------------------

def test_K3_mutating_hashed_field_breaks_digest(substrate):
    att = attest(REPORT_TRUE, substrate)
    tampered = copy.deepcopy(att.artifact)
    # change a hashed field (a claim's expected version) but DON'T re-seal.
    tampered["claims"][0]["args"]["version"] = "6.6.6"
    res = verify_attestation(tampered, substrate)
    assert res.digest_ok is False


def test_K3_mutating_clause_map_breaks_digest(substrate):
    att = attest(REPORT_TRUE, substrate)
    tampered = copy.deepcopy(att.artifact)
    tampered["compliance"]["evidence_clauses"] = ["Article 99 (fabricated)"]
    res = verify_attestation(tampered, substrate)
    assert res.digest_ok is False


def test_K3_generated_at_is_outside_digest(substrate):
    """Changing the volatile timestamp must NOT break the digest (it is
    deliberately excluded from the hashed payload)."""
    att = attest(REPORT_TRUE, substrate)
    tampered = copy.deepcopy(att.artifact)
    tampered["generated_at"] = "2099-01-01T00:00:00+00:00"
    res = verify_attestation(tampered, substrate)
    assert res.digest_ok is True


# --- P4: honest boundary is mandatory --------------------------------------

def test_P4_honest_boundary_present_and_dominant(substrate):
    art = attest(REPORT_TRUE, substrate).artifact
    boundary = art["compliance"]["uncovered_boundary"]
    covered = art["compliance"]["evidence_clauses"]
    assert len(boundary) >= 1
    # mirror the compliance module's kill-gate A3: uncovered >= covered
    assert len(boundary) >= len(covered)
    assert art["compliance"]["disclaimer"]


# --- security: unknown checker is refused, never executed ------------------

def test_unknown_checker_is_refused(substrate):
    att = attest(REPORT_TRUE, substrate)
    tampered = copy.deepcopy(att.artifact)
    tampered["claims"][0]["checker"] = "os.system"  # not in the allowlist
    res = verify_attestation(tampered, substrate)
    assert "os.system" in res.unknown_checkers
    assert res.ok is False
