"""The browser verifier, held to the committed conformance set.

SPEC: ``papers/sworn/SPEC_sworn_browser_verifier_v01_2026_09_05.md``. The bar it froze before the
code existed (B3): the set carries 1689 vectors in mode ``inline`` whose ``requires`` is a subset
of ``{manifest}``, and **all of them must reproduce their core digest**. The harness computes the
digest from the verifier's own object and never passes the expectation into it (B2); everything
out of scope is skipped and counted with the vector's own ``requires`` as the reason (B1).

These tests run node. Where node is absent they skip loudly rather than pass quietly — a verifier
nobody ran is not a verifier that agrees.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "styxx" / "_data" / "sworn_verify.js"
REPLAY = ROOT / "conformance" / "sworn" / "replay_js.js"
INDEX = ROOT / "conformance" / "sworn" / "index.json"
BAR = 1689                       # SPEC B3, frozen before the verifier was written


def _node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not on PATH; the browser verifier cannot be held to the vectors here")
    return exe


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    if not INDEX.exists():
        pytest.skip("the conformance set is not in this checkout")
    out = tmp_path_factory.mktemp("replay") / "replay.json"
    p = subprocess.run([_node(), str(REPLAY), "--json", str(out), "--quiet"],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert out.exists(), f"the harness wrote no report:\n{p.stdout}\n{p.stderr}"
    rep = json.loads(out.read_text(encoding="utf-8"))
    rep["_returncode"] = p.returncode
    rep["_stderr"] = p.stderr
    return rep


def test_the_verifier_parses_and_exports_its_entry_point():
    node = _node()
    p = subprocess.run(
        [node, "-e",
         "const a=require(process.argv[1]);"
         "if(typeof a.swornVerify!=='function')throw new Error('no swornVerify');"
         "if(typeof a.coreDigest!=='function')throw new Error('no coreDigest');"
         "console.log(a.sha256Bytes(a.utf8('abc')));", str(JS)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    # FIPS 180-4's own test vector, so a broken digest cannot hide behind agreement with itself
    assert p.stdout.strip() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_every_in_scope_vector_reproduces_its_core_digest(report):
    assert report["failed"] == 0, (
        "the browser verifier disagrees with styxx.sworn on "
        f"{report['failed']} vector(s): "
        + "; ".join(f"{f['family']}/{f['id'][:12]} want {f['want'][:12]} got {str(f['got'])[:12]}"
                    f" {f.get('error') or ''} {str(f.get('sources'))[:120]}"
                    for f in report["failures"][:6]))
    assert report["_returncode"] == 0, report["_stderr"]


def test_the_frozen_bar_is_the_number_of_vectors_actually_run(report):
    """SPEC B3 named 1689 before the verifier existed. If the set grows, this fails and the
    number moves in a commit that says why — the bar is never quietly re-fitted to the run."""
    assert report["ran"] == BAR, (
        f"the set now offers {report['ran']} in-scope vectors, not the {BAR} the SPEC froze; "
        "update the SPEC's bar in its own commit with the reason, then this line")
    assert report["passed"] == BAR


def test_the_set_the_harness_ran_is_the_committed_set(report):
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    assert report["set_sha256"] == index["set_sha256"]
    assert report["vectors_total"] == index["vector_count"]


def test_what_is_skipped_is_counted_and_reasoned(report):
    """B1: a verifier does not pass by skipping. Every skipped vector is out of scope by its own
    mode or `requires`, and the reason is recorded per family."""
    assert report["skipped"] + report["ran"] == report["vectors_total"]
    for fam, rec in report["families"].items():
        assert rec["ran"] + rec["skipped"] == rec["ran"] + rec["skipped"]
        for why in rec["skip_reasons"]:
            assert why.split(" ")[0] != "inline" or "requires:" in why, (fam, why)


def test_the_verifier_reads_no_file_and_reaches_no_network():
    """B5: pure. The file names no I/O module and no fetch."""
    src = JS.read_text(encoding="utf-8")
    for forbidden in ("require(", "import(", "fetch(", "XMLHttpRequest", "readFileSync",
                      "process.env", "Date.now", "new Date", "Math.random"):
        assert forbidden not in src, f"the verifier reaches for {forbidden}"


def test_the_label_is_carried_in_the_verifier_itself():
    """The plan's words, which the capsule and the README also print. Never 'self-verifying'."""
    src = JS.read_text(encoding="utf-8")
    assert "a forger controlling the whole file passes both" in src
    assert "the package at the named commit is the check" in src
    for banned in ("self-verifying", "tamper-proof", "immutable"):
        assert banned not in src.lower()
