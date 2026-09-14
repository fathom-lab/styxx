# -*- coding: utf-8 -*-
"""run_deploy_quant.sealed_refusal: the sealed experiment refuses to start unless the governing PREREG's blob
at HEAD is the digest it was sealed under, git answers, no tracked file differs from HEAD, no untracked file
exists under styxx/, and the checksum module that was imported is the checkout's styxx/checksum.py (red team
of 2026-09-14, runner F1; verification of 2026-09-14, RDQ-1 and RDQ-3). Instrument checks are never refused.
The runner imports torch at module level, so these tests skip where torch is absent (CI's matrix); the same
constants and gates are checked without torch, from the runner's source, in
tests/test_run_deploy_quant_sealed_static.py, which runs on CI."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import pytest

pytest.importorskip("torch")

from styxx import clock  # noqa: E402
from tests.test_clock import BLOCKHASH_BYTES, _chain, _tx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "papers", "checksum")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CK, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("run_deploy_quant_under_test", "run_deploy_quant.py")
score = _load("score_for_runner_test", "score.py")


def _prov(**over):
    p = {"git_head": "a" * 40, "prereg_blob_sha256": runner.SEALED["beacon_draw"], "git_dirty_tracked": False,
         "styxx_untracked": [], "checksum_file": os.path.join(ROOT, "styxx", "checksum.py"), "checksum_py_sha256": "b" * 64}
    p.update(over)
    return p


def test_the_runner_and_the_scorer_freeze_the_same_sealed_digests():
    assert runner.SEALED == {k: v["sealed_blob"] for k, v in score.BANDS.items()}
    assert runner.PREREGS == {k: v["prereg"] for k, v in score.BANDS.items()}


def test_an_instrument_check_is_never_refused():
    assert runner.sealed_refusal({}, "beacon_draw", False) is None
    assert runner.sealed_refusal(_prov(prereg_blob_sha256="0" * 64, git_dirty_tracked=True), "beacon_draw", False) is None


def test_the_experiment_runs_only_under_the_sealed_text_on_a_clean_commit():
    assert runner.sealed_refusal(_prov(), "beacon_draw", True) is None
    assert "not the sealed digest" in runner.sealed_refusal(_prov(prereg_blob_sha256="0" * 64), "beacon_draw", True)
    assert "not the sealed digest" in runner.sealed_refusal(_prov(), "deploy_quant", True)      # the other PREREG's digest
    assert "git did not answer" in runner.sealed_refusal(_prov(git_head=None), "beacon_draw", True)
    assert "git did not answer" in runner.sealed_refusal(_prov(prereg_blob_sha256=None), "beacon_draw", True)
    assert "tracked files differ" in runner.sealed_refusal(_prov(git_dirty_tracked=True), "beacon_draw", True)
    # RDQ-3: a git status that did not answer is not a clean tree
    assert "git did not answer" in runner.sealed_refusal(_prov(git_dirty_tracked=None), "beacon_draw", True)
    # RDQ-1: untracked files under styxx/, and a checksum imported from anywhere but the checkout's checksum.py
    assert "untracked files exist under styxx/" in runner.sealed_refusal(
        _prov(styxx_untracked=["styxx/checksum/__init__.py"]), "beacon_draw", True)
    assert "checksum module that was imported" in runner.sealed_refusal(
        _prov(checksum_file=os.path.join(ROOT, "styxx", "checksum", "__init__.py")), "beacon_draw", True)


def test_provenance_records_the_imported_checksum_file_its_sha256_and_the_untracked_files_under_styxx():
    prov = runner.provenance("cpu", runner.PREREGS["beacon_draw"])
    assert prov["checksum_file"] == os.path.abspath(runner.ck.__file__)
    assert runner._same_file(prov["checksum_file"], os.path.join(ROOT, "styxx", "checksum.py"))
    with open(prov["checksum_file"], "rb") as fh:
        assert prov["checksum_py_sha256"] == hashlib.sha256(fh.read()).hexdigest()
    assert prov["styxx_untracked"] is None or isinstance(prov["styxx_untracked"], list)
    assert prov["git_dirty_tracked"] in (True, False, None)


def test_the_imported_runner_runs_the_beacon_draw_experiment_only_under_the_seals_beacon(tmp_path):
    d = runner.SEALED["beacon_draw"]
    path = tmp_path / "anchors.jsonl"
    path.write_text(json.dumps({"n": 6, "kind": "sealed-prereg", "digest": d, "tx": "sig1"}) + "\n", encoding="utf-8")
    fetch = _chain(_tx(clock.memo("sealed-prereg", d)))
    assert runner.beacon_refusal(BLOCKHASH_BYTES.hex(), str(path), d, fetch) is None
    assert "is not the seal's beacon" in runner.beacon_refusal("0" * 64, str(path), d, fetch)
