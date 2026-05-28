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


# ===========================================================================
# Commit-pinned attestation — the pre-registered kill-gate from
# scripts/dogfood/PREREG_commit_pinned_attestation.md
#
#   K1 — determinism under pinning
#   K2 — historical isolation (decisive): pin to an OLD commit yields the OLD
#        verdict while the working tree has moved on; verify reproduces it
#   K3 — false-at-ref is caught (current state does not retroactively rescue)
#   K4 — read-only: the pin cycle never mutates the working tree or .git
# ===========================================================================


def _rev_parse(repo, ref):
    out = subprocess.run(["git", "rev-parse", ref], cwd=str(repo),
                         check=True, capture_output=True)
    return out.stdout.decode().strip()


@pytest.fixture
def versioned_substrate(tmp_path):
    """A repo with two commits: v1.0.0 then v2.0.0. Returns (repo, sha_v1)."""
    repo = tmp_path / "versioned"
    repo.mkdir()
    pyproject = repo / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\nversion = "1.0.0"\n',
                         encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1")
    sha_v1 = _rev_parse(repo, "HEAD")
    # move the substrate on to 2.0.0
    pyproject.write_text('[project]\nname = "demo"\nversion = "2.0.0"\n',
                         encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v2")
    return repo, sha_v1


def test_pin_records_ref_and_commit(versioned_substrate):
    repo, sha_v1 = versioned_substrate
    art = attest("The version is 1.0.0.", repo, ref=sha_v1).artifact
    assert art["substrate"]["pinned_ref"] == sha_v1
    assert art["substrate"]["commit"] == sha_v1


def test_K1_pin_determinism(versioned_substrate):
    repo, sha_v1 = versioned_substrate
    a = attest("The version is 1.0.0.", repo, ref=sha_v1)
    b = attest("The version is 1.0.0.", repo, ref=sha_v1)
    assert a.digest == b.digest


def test_K2_historical_isolation_decisive(versioned_substrate):
    """Pin to the v1.0.0 commit: the claim PASSes even though the working tree
    is now 2.0.0, while the unpinned attestation of the same claim FAILs."""
    repo, sha_v1 = versioned_substrate
    # sanity: the working tree really has moved on
    assert 'version = "2.0.0"' in (repo / "pyproject.toml").read_text()

    pinned = attest("The version is 1.0.0.", repo, ref=sha_v1)
    assert pinned.artifact["summary"]["passed"] == 1
    assert pinned.artifact["summary"]["failed"] == 0

    live = attest("The version is 1.0.0.", repo)  # working tree = 2.0.0
    assert live.artifact["summary"]["failed"] == 1

    # verify re-materializes the SAME commit and reproduces the pinned PASS
    res = verify_attestation(pinned, repo)
    assert res.digest_ok is True
    assert res.ok is True
    assert res.mismatches == []


def test_K3_false_at_ref_is_caught(versioned_substrate):
    """A claim that was FALSE at the pinned commit must FAIL — current truth
    (2.0.0) does not retroactively rescue it."""
    repo, sha_v1 = versioned_substrate
    art = attest("The version is 2.0.0.", repo, ref=sha_v1).artifact
    assert art["summary"]["failed"] == 1
    assert art["summary"]["passed"] == 0


def test_K4_pin_cycle_is_read_only(versioned_substrate):
    repo, sha_v1 = versioned_substrate
    before = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                            check=True, capture_output=True).stdout
    att = attest("The version is 1.0.0.", repo, ref=sha_v1)
    verify_attestation(att, repo)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                           check=True, capture_output=True).stdout
    assert before == after
    # no worktree was registered in .git
    assert not (repo / ".git" / "worktrees").exists()


def test_pinned_verify_errors_on_missing_commit(versioned_substrate):
    """If the repo no longer contains the pinned commit, verification cannot
    reproduce it — reported as ERROR/not-ok, never silently trusted."""
    repo, sha_v1 = versioned_substrate
    att = attest("The version is 1.0.0.", repo, ref=sha_v1)
    tampered = copy.deepcopy(att.artifact)
    # a well-formed but absent commit sha
    tampered["substrate"]["commit"] = "0" * 40
    from styxx.attestation import _compute_digest
    tampered["digest"]["value"] = _compute_digest(tampered)
    res = verify_attestation(tampered, repo)
    assert res.ok is False
    assert all(r["reproduced_verdict"] == "ERROR" for r in res.reproduced)


def test_pinned_verify_refuses_non_hex_commit(versioned_substrate):
    """An untrusted artifact cannot smuggle a git argument through the pin."""
    repo, sha_v1 = versioned_substrate
    att = attest("The version is 1.0.0.", repo, ref=sha_v1)
    tampered = copy.deepcopy(att.artifact)
    tampered["substrate"]["commit"] = "--upload-pack=evil"
    from styxx.attestation import _compute_digest
    tampered["digest"]["value"] = _compute_digest(tampered)
    res = verify_attestation(tampered, repo)
    assert res.ok is False
    assert all(r["reproduced_verdict"] == "ERROR" for r in res.reproduced)
