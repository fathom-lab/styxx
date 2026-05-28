# -*- coding: utf-8 -*-
"""Kill-gate for the portable (cross-language) content address.

These tests ARE the pre-registered kill-gate from
scripts/dogfood/PREREG_portable_attestation.md:

  K1 (decisive) — Python<->JS byte-for-byte digest agreement over a fuzz corpus
       of finite doubles AND the 4 real artifact shapes, incl. the saturating-
       score case that diverges under the legacy scheme. (Node tests skip if
       node is unavailable; the Python-side determinism + legacy-untouched gates
       always run.)
  K2 — adding digest.portable leaves the legacy digest.value byte-identical.
  K3 — the JS verifier catches a tampered portable digest and does not assert
       semantic properties (claim/vitals reported NOT CHECKED).
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from styxx.attestation import (
    _compute_digest,
    _compute_portable_digest,
    _es_number_to_string,
    attest,
    attest_chain,
    verify_attestation,
)

REPO = Path(__file__).resolve().parents[1]
JS_VERIFIER = REPO / "web" / "styxx_verify.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")


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
PROMPT = "Report on the version and the notes file."


# ---- ES-number serializer matches JS String(n) (basis of K1) ----------------

@needs_node
def test_K1_es_number_matches_node_string():
    import random
    random.seed(11)
    vals = [0.0, 1.0, -0.0, 0.5, 0.333327006208, 0.000100597705,
            1e-12, 9.99999999e-7, 1e-6, 0.9999999999, 1e21, 1e22, 123.0, 2 / 3]
    for _ in range(5000):
        vals.append(round(random.random(), 12))
        vals.append(round(random.uniform(0, 1), 4))
    py = [[v, _es_number_to_string(float(v))] for v in vals]
    script = (
        "const a=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "let bad=0;for(const [v,p] of a){if(String(v)!==p)bad++;}"
        "process.stdout.write(String(bad));"
    )
    r = subprocess.run([NODE, "-e", script], input=json.dumps(py),
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "0", f"{r.stdout} value(s) diverged Python<->JS"


# ---- K2: legacy digest is byte-identical with portable added ----------------

def test_K2_legacy_digest_excludes_portable(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo, prompt=PROMPT, vitals=True).artifact
    # The legacy digest is recomputed identically whether or not portable exists.
    assert _compute_digest(art) == art["digest"]["value"]
    without_portable = copy.deepcopy(art)
    del without_portable["digest"]["portable"]
    assert _compute_digest(without_portable) == art["digest"]["value"]


def test_K2_portable_digest_determinism(substrate):
    repo, head = substrate
    a = attest(REPORT, repo, ref=head, prompt=PROMPT, vitals=True).artifact
    b = attest(REPORT, repo, ref=head, prompt=PROMPT, vitals=True).artifact
    assert a["digest"]["portable"]["value"] == b["digest"]["portable"]["value"]
    assert _compute_portable_digest(a) == a["digest"]["portable"]["value"]


def test_portable_present_and_verifies(substrate):
    repo, _ = substrate
    att = attest(REPORT, repo, prompt=PROMPT, vitals=True)
    res = verify_attestation(att, repo)
    assert res.portable_present is True
    assert res.portable_ok is True
    assert res.ok is True


def test_portable_tamper_caught_by_library(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo).artifact
    art["digest"]["portable"]["value"] = "0" * 64
    res = verify_attestation(art, repo)
    assert res.portable_ok is False
    assert res.ok is False


# ---- K1: Python digest.portable == Node recomputation over the 4 shapes ------

def _shapes(repo, head):
    return {
        "plain": attest(REPORT, repo).artifact,
        "pinned": attest(REPORT, repo, ref=head).artifact,
        "vitals": attest(REPORT, repo, prompt=PROMPT, vitals=True).artifact,
        "chain": attest_chain([(REPORT, head, PROMPT), (REPORT, head, PROMPT)],
                              repo, vitals=True).artifact,
    }


def _node_portable_digest(artifact):
    script = (
        f"const v=require({json.dumps(str(JS_VERIFIER))});"
        "const a=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(v.portableDigest(a));"
    )
    r = subprocess.run([NODE, "-e", script], input=json.dumps(artifact, ensure_ascii=False),
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _node_verify(artifact, expected_head=None):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        fh.write(json.dumps(artifact, ensure_ascii=False))
        path = fh.name
    cmd = [NODE, str(JS_VERIFIER), path]
    if expected_head is not None:
        cmd.append(expected_head)
    r = subprocess.run(cmd, capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    return r.returncode, r.stdout


@needs_node
def test_K1_python_portable_equals_node_over_shapes(substrate):
    repo, head = substrate
    shapes = _shapes(repo, head)
    for name in ("plain", "pinned", "vitals"):
        art = shapes[name]
        py = art["digest"]["portable"]["value"]
        node = _node_portable_digest(art)
        assert py == node, f"shape {name}: python {py} != node {node}"
    # chain: compare each link's attestation portable digest byte-for-byte
    chain = shapes["chain"]
    for i, link in enumerate(chain["links"]):
        att = link["attestation"]
        py = att["digest"]["portable"]["value"]
        node = _node_portable_digest(att)
        assert py == node, f"chain link {i}: python {py} != node {node}"
        assert link["attestation_portable_digest"] == py


@needs_node
def test_K1_node_verifier_accepts_real_artifacts(substrate):
    repo, head = substrate
    for name, art in _shapes(repo, head).items():
        code, _out = _node_verify(art)
        assert code == 0, f"shape {name}: node verifier rejected a valid artifact"


@needs_node
def test_K1_saturating_score_now_agrees(substrate):
    """The decisive case: an artifact whose coverage saturates to 1.0 diverged
    under the legacy scheme (python '1.0' vs js '1'); the portable digest agrees."""
    repo, _ = substrate
    art = attest(REPORT, repo, prompt=PROMPT, vitals=True).artifact
    assert art["summary"]["coverage"] == 1.0  # the divergent token
    assert art["digest"]["portable"]["value"] == _node_portable_digest(art)


@needs_node
def test_K3_node_catches_portable_tamper(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo).artifact
    art["digest"]["portable"]["value"] = "0" * 64
    code, out = _node_verify(art)
    assert code == 1
    assert "FAIL" in out


@needs_node
def test_K3_node_catches_broken_chain(substrate):
    repo, head = substrate
    chain = attest_chain([(REPORT, head, None), (REPORT, head, None)], repo).artifact
    chain["links"][1]["attestation_portable_digest"] = "0" * 64
    code, out = _node_verify(chain)
    assert code == 1
    assert "FAIL" in out


@needs_node
def test_K3_node_reports_semantic_not_checked(substrate):
    repo, _ = substrate
    art = attest(REPORT, repo, prompt=PROMPT, vitals=True).artifact
    _code, out = _node_verify(art)
    assert "NOT CHECKED" in out
    assert "vitals scores" in out
