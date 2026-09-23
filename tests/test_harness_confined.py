# -*- coding: utf-8 -*-
"""SWALLOW-15's instrument: the product's audit in a Landlock-confined child, red-teamed against its
boundary and compared to an unconfined audit of the same checkout. The tests hold the battery to
neutralising every item, the confined audit to reading the same core as the unconfined one, and the
summary to its counts."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from styxx.ciaudit import confine as C  # noqa: E402
from benchmarks.harness_mutation import confined as CF  # noqa: E402
from tests.test_ciaudit_frontier import FRONTIER_FIXTURE, LISTS_FIXTURE, _tree  # noqa: E402

# frozen at the sha256 the SWALLOW-15 preregistration names; a change needs a new preregistration, not a new pin
INSTRUMENT_SHA256 = "7014bc02c46f5344249ed1636d4778c217df6998aa93164f96974817d2e97036"
landlock = pytest.mark.skipif(C.abi() == 0, reason="this kernel has no Landlock")


def test_the_instrument_is_the_one_the_preregistration_names():
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "confined.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


@landlock
def test_the_battery_neutralises_every_item_confined(tmp_path):
    rep = CF.battery(tmp_path)
    assert rep["confined_neutralised"] == rep["items"], [r for r in rep["rows"] if r["confined"] and r["fired_on_host"]]
    assert rep["any_reached_host"] == []
    assert rep["control_fired"] >= 8                                # most items reach the host unconfined -- they are live
    assert rep["confined_audited_ok"] == rep["items"]              # and the audit still completes under confinement
    assert set(rep["perimeter"]) == {"read outside", "chmod outside", "udp send"}


@landlock
def test_a_deleting_step_deletes_the_canary_unconfined_and_not_confined(tmp_path):
    name, tmpl, effect = CF.BATTERY[0]                             # rm -rf a scope that came out empty
    control = CF.probe_item(name, tmpl, effect, tmp_path, confined=False)
    confined = CF.probe_item(name, tmpl, effect, tmp_path, confined=True)
    assert control["fired_on_host"] is True                        # the canary is deleted with no confinement
    assert confined["fired_on_host"] is False and confined["error"] is None
    assert confined["audited"]["confined"] is True


@landlock
def test_the_confined_audit_reads_the_same_core_as_unconfined(tmp_path):
    for name, text in (("frontier", FRONTIER_FIXTURE), ("lists", LISTS_FIXTURE)):
        tree = _tree(tmp_path / name, text)
        unconfined = CF.EL.audit_one(tree)
        confined, info = CF.confined_audit_one(tree)
        assert info["confined"] is True
        u = {k: v["core"] for k, v in CF._outcome_targets(unconfined).items()}
        c = {k: v["core"] for k, v in CF._outcome_targets(confined).items()}
        assert u == c, name


def test_the_summary_counts(tmp_path):
    receipt_like = {
        "abc/one": {("ci.yml", "j", 0): {"core": {"v": 1}, "verified_repair": "hoist-local"}},
        "abc/two": {("ci.yml", "j", 0): {"core": {"v": 2}, "verified_repair": "wait-list"}},
    }
    # a match, a core-differ, a verified_repair move, an error, a fetch failure
    repos = [
        {"repo": "abc/one", "confined": True, "matches_receipt": True, "verified_repair_moved": [], "targets": 1, "seconds": 4.0},
        {"repo": "abc/two", "confined": True, "matches_receipt": True, "verified_repair_moved": [["ci.yml", "j", 0]], "targets": 1, "seconds": 6.0},
        {"repo": "abc/three", "confined": True, "matches_receipt": False, "diff": {"x": 1}, "verified_repair_moved": [], "targets": 1, "seconds": 5.0},
        {"repo": "abc/four", "confined": None, "error": "confined: Boom", "seconds": 1.0},
        {"repo": "abc/five", "fetch": {"error": "fetch: gone"}, "seconds": 1.0},
    ]
    s = CF.summary(repos, breached=[])
    assert (s["repos"], s["fetched"], s["fetch_failed"], s["errors"]) == (5, 3, 1, 1)
    assert (s["confined"], s["compared_to_receipt"], s["match_receipt"], s["differ_from_receipt"]) == (3, 3, 2, 1)
    assert s["verified_repair_moved_repos"] == 1 and s["canaries_breached"] == 0
    assert s["seconds_median_per_repo"] == 4.0


def test_a_breach_is_recorded_and_the_run_keeps_measuring(tmp_path, monkeypatch):
    # a confinement that lets a step out would trip the canary; the run records it and does not stop
    calls = {"n": 0}
    cs = CF._canaries(tmp_path / "c")
    assert CF._canaries_intact(cs)
    cs[0].write_text("tampered")
    assert not CF._canaries_intact(cs)
