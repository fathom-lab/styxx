"""CALIB-1's scorer, pinned on the failures that would let a bad run look like a good one.

The run this scores has not happened and needs a key nobody here holds. That is
precisely why the scorer is tested now: the preregistration is frozen, so every
choice the scorer makes is already fixed, and a reader can check that it makes
them before there is any result to be tempted by.

Four of these matter more than the rest.

`test_dev_split_with_one_class_is_unrunnable_not_redrawn` is the one that caught a
real defect. `v14_gates.bucket` puts 3 POSITIVE and 0 NEGATIVE of the 25 items in
the development split. G-C1-1 cannot be run on that, and the tempting repair — a
different split — is the exact act the gate forbids. The scorer records it as
unrunnable and ships nothing.

`test_dry_run_is_refused` stops a hash-generated file being read as evidence.

`test_bin_with_two_items_is_reported_empty` pins G-C1-3's rule that a bin too
small to mean anything is not averaged into the headline.

`test_spend_is_not_invented` pins that an unknown price is published as unknown.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "calib1_score", ROOT / "papers" / "closed-model-frontier" / "calib1_score.py")
C = importlib.util.module_from_spec(_SPEC)
sys.modules["calib1_score"] = C
_SPEC.loader.exec_module(C)

FROZEN = C.PREREG_SHA256_FROZEN


def _url(want_split, n=0):
    """A github URL whose first five segments hash into the split asked for."""
    while True:
        u = f"https://github.com/o/r{n}/pull/1"
        if C.split_of(u) == want_split:
            return u, n + 1
        n += 1


def _raw(items, *, dry_run=False, prereg=FROZEN, repeats=1, spreads=None,
         tokens=(200, 1), ms=120):
    """Build a raw file. `items` is (cls, split, noul) triples."""
    out_items, calls, n = [], [], 0
    for i, (cls, split, noul) in enumerate(items, start=1):
        url, n = _url(split, n)
        out_items.append({
            "id": i, "url": url, "claim": f"claim {i}", "n_files": 3,
            "paths_shown": 3, "paths_truncated": False,
            "decidable": cls == "POSITIVE",
            "reason_code": None if cls == "POSITIVE" else (
                "ambiguous_scope" if cls == "EXCLUDED" else "runtime_behaviour"),
        })
        for r in range(repeats):
            v = noul
            if r > 0 and spreads:
                v = min(1.0, noul + spreads.get(i, 0.0))
            calls.append({"id": i, "repeat": r, "noul": v, "ms": ms,
                          "model": "jev-1", "input_tokens": tokens[0],
                          "output_tokens": tokens[1], "error": None})
    return {"prereg": "PREREG_calib1_jev_2026_09_18.md", "prereg_sha256": prereg,
            "triage_module_sha256": "0" * 64, "dry_run": dry_run,
            "repeats": repeats, "items": out_items, "calls": calls}


def _balanced(pos_noul=0.95, neg_noul=0.05, dev_negatives=2):
    """A population with both classes on both splits, so G-C1-1 can run."""
    items = [("POSITIVE", "DEVELOPMENT", pos_noul) for _ in range(3)]
    items += [("NEGATIVE", "DEVELOPMENT", neg_noul) for _ in range(dev_negatives)]
    items += [("POSITIVE", "HELD_OUT", pos_noul) for _ in range(10)]
    items += [("NEGATIVE", "HELD_OUT", neg_noul) for _ in range(8)]
    return items


def gate(res, name):
    return next(g for g in res["gates"] if g["gate"] == name)


# --------------------------------------------------------------------------- #

def test_dry_run_is_refused():
    res = C.score(_raw(_balanced(), dry_run=True))
    assert res["verdict_token"] == "INVALID__DRY_RUN"
    assert res["ships_thresholds"] is False
    assert any("dry run" in r for r in res["refusals"])


def test_prereg_hash_mismatch_is_refused():
    res = C.score(_raw(_balanced(), prereg="f" * 64))
    assert res["verdict_token"] == "INVALID__PREREG_MOVED"
    assert res["ships_thresholds"] is False


def test_dev_split_with_one_class_is_unrunnable_not_redrawn():
    # The real population: every NEGATIVE lands in HELD_OUT.
    res = C.score(_raw(_balanced(dev_negatives=0)))
    g1 = gate(res, "G-C1-1")
    assert g1["status"] == "UNRUNNABLE"
    assert g1["thresholds"] is None
    assert g1["dev_negative"] == 0
    assert "not redrawn" in g1["why"] or "not redrawn." in g1["why"]
    assert res["ships_thresholds"] is False
    assert res["verdict_token"].startswith("CALIB1__NO_THRESHOLDS__G_C1_1_UNRUNNABLE")
    # Separation is independent of the split and is still measured.
    assert gate(res, "G-C1-2")["status"] == "PASS"


def test_unrunnable_g1_reports_the_band_over_a_grid_instead_of_a_chosen_one():
    g6 = gate(C.score(_raw(_balanced(dev_negatives=0))), "G-C1-6")
    assert g6["status"] == "REPORTED_OVER_GRID"
    assert g6["thresholds"] is None
    assert [r["skip"] for r in g6["grid"]] == [0.10, 0.20, 0.30, 0.40]


def test_separation_passes_when_the_classes_are_far_apart():
    res = C.score(_raw(_balanced()))
    g2 = gate(res, "G-C1-2")
    assert g2["status"] == "PASS"
    assert g2["separation"] == pytest.approx(0.90)
    assert g2["concordance"] == 1.0
    assert res["verdict_token"] == "CALIB1__THRESHOLDS_SELECTED"
    assert gate(res, "G-C1-1")["thresholds"] is not None


def test_no_separation_abandons_the_run():
    flat = [("POSITIVE", "DEVELOPMENT", 0.5)] * 3 + [("NEGATIVE", "DEVELOPMENT", 0.5)] * 2
    flat += [("POSITIVE", "HELD_OUT", 0.5)] * 10 + [("NEGATIVE", "HELD_OUT", 0.5)] * 8
    res = C.score(_raw(flat))
    g2 = gate(res, "G-C1-2")
    assert g2["status"] == "FAIL"
    assert g2["separation"] == pytest.approx(0.0)
    assert res["verdict_token"] == "CALIB1__ABANDONED__G_C1_2_FAILED"
    assert res["ships_thresholds"] is False


def test_a_wide_interval_says_it_does_not_settle_the_question():
    # Two items a side: the concordance interval cannot be narrow.
    tiny = [("POSITIVE", "DEVELOPMENT", 0.9), ("NEGATIVE", "DEVELOPMENT", 0.1),
            ("POSITIVE", "HELD_OUT", 0.9), ("NEGATIVE", "HELD_OUT", 0.1)]
    g2 = gate(C.score(_raw(tiny)), "G-C1-2")
    assert g2["concordance_ci_width_points"] > 30.0
    assert g2["settles_the_question"] is False


def test_bin_with_two_items_is_reported_empty_not_averaged():
    items = _balanced()
    items += [("POSITIVE", "HELD_OUT", 0.5), ("NEGATIVE", "HELD_OUT", 0.5)]
    g3 = gate(C.score(_raw(items)), "G-C1-3")
    middle = [b for b in g3["bins"] if b["lo"] == 0.4][0]
    assert middle["n"] == 2
    assert middle["reported"] == "empty"
    assert middle["accuracy"] is None
    assert g3["items_in_empty_bins"] == 2


def test_excluded_items_are_published_but_never_scored():
    items = _balanced() + [("EXCLUDED", "HELD_OUT", 0.55) for _ in range(4)]
    res = C.score(_raw(items))
    assert res["population"]["n_excluded"] == 4
    assert res["population"]["n_scored"] == 23
    assert [x["noul"] for x in res["excluded_items"]] == [0.55] * 4
    assert gate(res, "G-C1-2")["n_positive"] == 13


def test_nonzero_spread_states_the_argument_against_the_verdict_path():
    # Item 4 is a NEGATIVE at 0.05, so +0.2 is a real spread and not a clamp.
    res = C.score(_raw(_balanced(), repeats=5, spreads={4: 0.2}))
    g4 = gate(res, "G-C1-4")
    assert g4["max_spread_graded"] == pytest.approx(0.2)
    assert g4["prediction_4_held"] is True
    assert g4["verdict_path_statement"] == (
        "Any spread above zero is a permanent argument against Jev on the verdict path.")


def test_zero_spread_does_not_get_called_determinism():
    g4 = gate(C.score(_raw(_balanced(), repeats=5)), "G-C1-4")
    assert g4["max_spread_graded"] == 0.0
    assert g4["prediction_4_held"] is False
    assert "not a promise of determinism" in g4["verdict_path_statement"]


def test_g4_grades_exactly_twenty_items_however_many_were_asked():
    items = _balanced() + [("EXCLUDED", "HELD_OUT", 0.5) for _ in range(4)]
    g4 = gate(C.score(_raw(items, repeats=5)), "G-C1-4")
    assert len(g4["graded_items"]) == 20
    assert g4["graded_items"] == sorted(g4["graded_items"])
    assert len(g4["per_item"]) == 27


def test_spend_is_not_invented():
    g5 = gate(C.score(_raw(_balanced())), "G-C1-5")
    assert g5["spend_usd"] is None
    assert "will not invent it" in g5["spend_note"]
    assert g5["input_tokens"] == 23 * 200
    assert g5["median_latency_ms"] == 120


def test_spend_is_computed_when_a_price_is_supplied():
    g5 = gate(C.score(_raw(_balanced()), price_in=3.0, price_out=15.0), "G-C1-5")
    assert g5["spend_usd"] == pytest.approx(round(23 * 200 / 1e6 * 3 + 23 / 1e6 * 15, 2))
    assert g5["spend_note"] is None


def test_an_unlisted_reason_code_raises_rather_than_being_guessed():
    raw = _raw(_balanced())
    raw["items"][-1]["reason_code"] = "something_new"
    with pytest.raises(ValueError, match="neither"):
        C.score(raw)


def test_the_split_helper_matches_the_repositorys_own_convention():
    from v14_gates import bucket
    for split in ("DEVELOPMENT", "HELD_OUT"):
        url, _ = _url(split)
        b = bucket("/".join(url.split("/")[:5]))
        assert (b < 3) == (split == "DEVELOPMENT")


def test_a_call_that_failed_does_not_become_a_zero():
    raw = _raw(_balanced())
    raw["calls"][0]["noul"] = None
    raw["calls"][0]["error"] = "jev unreachable: ECONNREFUSED"
    res = C.score(raw)
    assert res["population"]["n_scored"] == 22
    assert gate(res, "G-C1-2")["n_positive"] == 12
