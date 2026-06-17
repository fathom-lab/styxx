# -*- coding: utf-8 -*-
"""Security regression tests for the untrusted-verification path.

verify_attestation() re-runs an attestation's checkers against a repo, with EVERY
checker arg reconstructed from the (attacker-controlled) artifact. These tests pin
the three boundaries that hardening closed:

  1. code execution  — a python_attr_* checker naming a module whose import writes
     a file must NOT execute that code on the default (untrusted) path;
  2. arbitrary file read — file/path checkers must refuse absolute paths and ``..``
     traversal, never leaking out-of-repo content into the receipt evidence;
  3. git argument injection — ref/branch/tag args beginning with ``-`` (e.g.
     ``--output=PATH``) must be refused before reaching ``git``.

Each test also confirms the legitimate in-repo behavior still works, so the guard
is a boundary, not a regression.
"""
from __future__ import annotations

import sys

import pytest

from styxx.agent_audit import checkers
from styxx.attestation import verify_attestation


def _artifact(checker: str, args: dict, *, cid: str = "C1") -> dict:
    """Minimal attestation artifact with one claim (no valid digest needed —
    the attacks under test must be blocked regardless of digest validity)."""
    return {
        "claims": [
            {"id": cid, "text": "t", "checker": checker, "args": args,
             "expected": True, "verdict": "PASS"},
        ],
    }


# --------------------------------------------------------------------------- #
# 1. Code execution via python_attr_* is refused on the untrusted path.
# --------------------------------------------------------------------------- #

def test_code_execution_checker_refused_by_default(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    proof = tmp_path / "PROOF_RCE"
    # A substrate module whose mere import would write PROOF_RCE.
    (repo / "pwned_mod.py").write_text(
        f"import pathlib; pathlib.Path(r{str(proof)!r}).write_text('pwned')\n",
        encoding="utf-8",
    )
    sys.modules.pop("pwned_mod", None)

    art = _artifact("python_attr_equals",
                    {"module": "pwned_mod", "attr": "X", "expected": 1})
    res = verify_attestation(art, repo)

    assert not proof.exists(), "verifier executed attacker substrate code"
    assert "python_attr_equals" in res.unsafe_checkers
    assert res.ok is False
    assert any(r.get("reproduced_verdict") == "REFUSED" for r in res.reproduced)
    sys.modules.pop("pwned_mod", None)


def test_code_execution_checker_runs_only_with_trust_substrate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    proof = tmp_path / "PROOF_RCE_TRUSTED"
    (repo / "trusted_mod.py").write_text(
        f"import pathlib; pathlib.Path(r{str(proof)!r}).write_text('ok')\nX = 1\n",
        encoding="utf-8",
    )
    sys.modules.pop("trusted_mod", None)

    art = _artifact("python_attr_equals",
                    {"module": "trusted_mod", "attr": "X", "expected": 1})
    res = verify_attestation(art, repo, trust_substrate=True)

    # Opt-in lifts the refusal: the checker is no longer recorded as unsafe.
    assert "python_attr_equals" not in res.unsafe_checkers
    assert proof.exists(), "trust_substrate=True should allow the import"
    sys.modules.pop("trusted_mod", None)


def test_python_attr_rejects_non_identifier_module(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError):
        checkers.python_attr_equals(repo, module="../evil", attr="X", expected=1)


# --------------------------------------------------------------------------- #
# 2. Arbitrary file read via path traversal / absolute paths is refused.
# --------------------------------------------------------------------------- #

def test_file_checker_rejects_parent_traversal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "SECRET.txt"
    secret.write_text("TOPSECRET-abc123", encoding="utf-8")

    with pytest.raises(ValueError):
        checkers.file_at_path_contains(repo, path="../SECRET.txt", substring="TOPSECRET")


def test_file_checker_rejects_absolute_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "SECRET.txt"
    secret.write_text("TOPSECRET-abc123", encoding="utf-8")

    with pytest.raises(ValueError):
        checkers.file_at_path_contains(repo, path=str(secret), substring="TOPSECRET")


def test_traversal_via_verify_does_not_leak_outside_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "SECRET.txt").write_text("TOPSECRET-abc123", encoding="utf-8")

    art = _artifact("file_at_path_contains",
                    {"path": "../SECRET.txt", "substring": "TOPSECRET"})
    res = verify_attestation(art, repo)

    blob = repr(res.reproduced)
    assert "TOPSECRET" not in blob, "out-of-repo content leaked into receipt evidence"
    assert any(r.get("reproduced_verdict") == "ERROR" for r in res.reproduced)


def test_legit_in_repo_file_check_still_works(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "marker.txt").write_text("hello styxx", encoding="utf-8")

    present, evidence = checkers.file_at_path_contains(
        repo, path="marker.txt", substring="styxx")
    assert present is True
    assert "styxx" in evidence


def test_legit_package_version_check_still_works(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('version = "7.17.0"\n', encoding="utf-8")

    present, _ = checkers.package_version_equals(
        repo, path="pyproject.toml", version="7.17.0")
    assert present is True


# --------------------------------------------------------------------------- #
# 3. Git argument injection (leading-dash refs) is refused before git runs.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("call", [
    lambda r: checkers.git_branch_contains_commit_chain(
        r, branch="--output=PWNED", commits=[]),
    lambda r: checkers.git_tag_exists(r, tag="--output=PWNED"),
    lambda r: checkers.git_show_diff_contains(
        r, commit="--output=PWNED", file="x", substring="y"),
])
def test_git_checkers_reject_option_injection(tmp_path, call):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError):
        call(repo)
    assert not (repo / "PWNED").exists()
    assert not (tmp_path / "PWNED").exists()
