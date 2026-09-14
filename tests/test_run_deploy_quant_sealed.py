# -*- coding: utf-8 -*-
"""run_deploy_quant.sealed_refusal: the sealed experiment refuses to start unless the governing PREREG's blob
at HEAD is the digest it was sealed under, git answers, and no tracked file differs from HEAD (red team of
2026-09-14, runner F1: nothing tied the run to the sealed text). Instrument checks are never refused. The
runner imports torch at module level, so these tests skip where torch is absent (CI's matrix)."""
from __future__ import annotations

import importlib.util
import os

import pytest

pytest.importorskip("torch")

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
    return {"git_head": "a" * 40, "prereg_blob_sha256": runner.SEALED["beacon_draw"], "git_dirty_tracked": False, **over}


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
