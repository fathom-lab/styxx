# -*- coding: utf-8 -*-
"""styxx.horizon — the fitter's contract on synthetic bits; no torch, no model, no GPU.

Load-bearing: the planted-crossing self-test recovers its horizons, their level, a nominal interval
and the D0 split as wired, and the datasheet refuses when it does not; a censored horizon prints
"> 1024" and never a number; a row that fails the screen prints no horizon, ratio or dissociation;
the compare readout (argmax over three slots, or yes/no) is a fact of the receipt, with its chance and
guard printed; a constant responder under exact balance maps to exactly 0 under either readout and
reads SCREEN_FAIL; a cell out of balance is flagged and fails the screen; the bootstrap is a function
of the seed; the family is a fact of a config and its A values, never of a name; the sheet's family
aggregates satisfy a declared composition in the lab's own scorer, and an empty family set scores to
ABSTAIN through it by a readiness block; the bridge-v3 runner's receipt is read in the shape it is
written; nothing this module prints carries a word the lab never prints.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from styxx import horizon as H

ROOT = Path(__file__).resolve().parent.parent
RECALL_FLAT = [0.9, 0.88, 0.86, 0.84]

# The bridge-v3 amendment-17.2 pilot receipt (seed 1012, 48 items at D0, both arms), committed on the
# research branch of bridge v3. Found at its repository path once that branch is merged, or by its git
# blob in a clone that holds the branch; the test is skipped, never failed, where neither is present.
PILOT2_PATH = "papers/frequency-resonance/realmodel_bridge_v3_pilot2.json"
PILOT2_BLOB = "f80853e22d646b4c42069b145b76ead4e16608cf"
PILOT2_SHA256_LF = "fc8840f5273b205f553bb022773d3d68d8147e3a03e5936f71f036f78a7742a0"


@pytest.fixture(scope="module")
def selftest_result():
    return H.selftest()


def _row(compare, recall=RECALL_FLAT, n=600, seed=1, **kw):
    rec = H.load_receipt(H.synthetic_receipt("m", compare, recall, n, seed, n_resamples=300, **kw))
    return H.fit_model(rec)


# -- the self-test and the refusal it gates ---------------------------------------------------------

def test_the_self_test_recovers_every_planted_horizon(selftest_result):
    st = selftest_result
    assert st["pass"], [c for c in st["checks"] if not c["ok"]]
    names = {c["name"] for c in st["checks"]}
    assert {"planted_32", "planted_128", "planted_203", "planted_600", "planted_order", "planted_level",
            "interval_width", "interval_coverage", "d0_split_wiring", "censored", "constant_responder",
            "constant_responder_yes_no", "screen_fail_prints_no_horizon", "compare_only_flag",
            "deterministic_under_seed"} <= names
    assert st["coverage"]["hits"] >= st["coverage"]["needed"]


def test_the_datasheet_refuses_when_the_self_test_fails(monkeypatch):
    row = _row(H._planted_curve(128.0))
    bad = {"pass": False, "checks": [{"name": "planted_128", "ok": False, "detail": "recovered 900"}]}
    with pytest.raises(H.HorizonRefusal, match="self-test failed; no datasheet"):
        H.datasheet(row, selftest_result=bad)
    monkeypatch.setattr(H, "selftest", lambda *a, **k: bad)
    with pytest.raises(H.HorizonRefusal):
        H.datasheet(row)


def _failed_checks(coverage_seeds=8):
    st = H.selftest(n=1200, n_resamples=200, coverage_seeds=coverage_seeds)
    return {c["name"] for c in st["checks"] if not c["ok"]}


def test_the_self_test_sees_the_curve_level(monkeypatch):
    # a normalisation that forgets to divide by (1 - chance) keeps every crossing where it was
    monkeypatch.setattr(H, "normalize", lambda acc, chance: float(acc) - chance)
    assert "planted_level" in _failed_checks()


def test_the_self_test_sees_the_interval_width(monkeypatch):
    real = H._percentiles

    def fifty(values):
        v = np.asarray(values, dtype=float)
        fin = v[np.isfinite(v)]
        if len(fin) == 0:
            return real(values)
        lo, hi = np.percentile(fin, [25, 75])
        return float(lo), float(hi), float(1.0 - len(fin) / len(v))

    monkeypatch.setattr(H, "_percentiles", fifty)
    failed = _failed_checks(coverage_seeds=32)
    assert "interval_width" in failed and "interval_coverage" in failed


def test_the_self_test_sees_the_d0_split_wiring(monkeypatch):
    # the M4 fix mis-wired: the replicate means ignore which half of D0 is the denominator
    real = H._rep_means
    monkeypatch.setattr(H, "_rep_means", lambda correct, idx, keep_mask=None: real(correct, idx, None))
    assert "d0_split_wiring" in _failed_checks()
    # and mis-wired the other way: the denominator read on the screen's half
    real_split = H._split_masks

    def same_half(cell):
        s, d, note = real_split(cell)
        return s, s, note

    monkeypatch.setattr(H, "_rep_means", real)
    monkeypatch.setattr(H, "_split_masks", same_half)
    assert "d0_split_wiring" in _failed_checks()


def test_the_planted_illustration_of_the_design_crosses_where_the_rule_says():
    # the design's text planted "128" with c = (0.9, 0.7, 0.4, 0.2); under the interpolation in log2 D
    # those values cross at about 203, so the self-test plants from the rule, not from that sentence
    x = H.crossing([0.9, 0.7, 0.4, 0.2])
    assert x["status"] == "FINITE" and x["bracket"] == (64, 256) and 200 < x["point"] < 206
    for h in (32.0, 128.0, 203.0, 600.0):
        assert H.crossing(H._planted_curve(h))["point"] == pytest.approx(h, rel=1e-9)


# -- crossing, censoring, non-monotone curves --------------------------------------------------------

def test_no_crossing_is_censored_and_prints_no_number(selftest_result):
    row = _row([0.9, 0.85, 0.8, 0.75])
    hz = row["horizon"]["compare"]
    assert hz["status"] == "CENSORED" and hz["point"] is None and hz["f_cens"] > 0.975
    assert "NO_HORIZON_INSIDE_RANGE" in row["flags"]
    text = H.datasheet(row, selftest_result=selftest_result)
    line = next(l for l in text.splitlines() if "H_compare:" in l)
    assert "> 1024 (none inside the tested range" in line and "~" not in line.split("H_compare:")[1]


def test_a_crossing_near_the_last_grid_point_censors_the_upper_end_of_its_interval():
    row = _row([0.9, 0.8, 0.6, 0.42])
    hz = row["horizon"]["compare"]
    assert hz["status"] == "FINITE" and hz["bracket"] == (256, 1024)
    assert 0.025 < hz["f_cens"] < 0.5 and hz["ci_hi_censored"] is True and hz["print_censored"] is False
    assert "> 1024]" in H._h_text(hz)


def test_c_at_D0_at_or_below_chance_is_not_estimable():
    assert H.crossing([0.0, 0.0, 0.0, 0.0])["status"] == "NOT_ESTIMABLE"
    assert H.crossing([-0.1, 0.2, 0.2, 0.2])["status"] == "NOT_ESTIMABLE"
    assert H.crossing([0.5, float("nan"), 0.2, 0.2])["status"] == "NOT_ESTIMABLE"


def test_a_non_monotone_curve_is_flagged_and_its_earliest_crossing_used():
    x = H.crossing([0.9, 0.3, 0.8, 0.2])
    assert x["status"] == "FINITE" and x["bracket"] == (16, 64)
    row = _row([0.9, 0.3, 0.8, 0.2])
    assert "NONMONOTONE@compare" in row["flags"] and row["horizon"]["compare"]["bracket"] == (16, 64)


def test_the_grid_is_validated():
    with pytest.raises(H.HorizonRefusal, match="increasing"):
        H.crossing([0.9, 0.5, 0.2], grid=[16, 16, 64])
    with pytest.raises(H.HorizonRefusal, match="points"):
        H.crossing([0.9, 0.5], grid=[16, 64, 256])


@pytest.mark.parametrize("frac", [1.5, 1.0, 0.0, -0.2, "half"])
def test_horizon_frac_must_lie_strictly_between_0_and_1(frac):
    # frac 1.5 used to put the crossing at k = 0 and read c[-1]: bracket (1024, 16)
    with pytest.raises(H.HorizonRefusal, match="horizon_frac"):
        H.crossing([0.9, 0.6, 0.3, 0.1], frac=frac)
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 60, 2, n_resamples=20)
    rec["config"]["horizon_frac"] = frac
    with pytest.raises(H.HorizonRefusal, match="horizon_frac"):
        H.load_receipt(rec)


def test_ratio_min_reads_only_the_distances_past_D0():
    # a curve that rises past D0 has ratio_min above 1; a minimum that included D0 would read 1.0
    row = _row([0.6, 0.7, 0.8, 0.9], recall=[0.6, 0.66, 0.72, 0.78])
    assert row["ratio_min"]["compare"]["point"] > 1.05 and row["ratio_min"]["recall"]["point"] > 1.02


# -- chance normalisation, readouts, the constant responder, the screen -----------------------------

def test_normalize_is_zero_at_chance_and_one_at_ceiling():
    assert H.normalize(0.5, 0.5) == 0.0 and H.normalize(1.0, 0.5) == 1.0
    assert H.normalize(1.0 / 3.0, 1.0 / 3.0) == 0.0 and H.normalize(0.0, 0.5) == -1.0
    with pytest.raises(H.HorizonRefusal):
        H.normalize(0.5, 1.0)


@pytest.mark.parametrize("readout,constant,guard_text", [
    ("argmax_largest", 1, "compare guard failed (largest slot share 1.00)"),
    ("yes_no", True, "compare guard failed (yes-rate 1.00)"),
])
def test_a_constant_responder_under_exact_balance_maps_to_exactly_zero_and_fails_the_screen(
        selftest_result, readout, constant, guard_text):
    row = _row([0, 0, 0, 0], [0, 0, 0, 0], constant_compare=constant, constant_recall=0, readout=readout)
    assert row["curves"]["compare"] == [0.0, 0.0, 0.0, 0.0]
    assert row["curves"]["recall"] == [0.0, 0.0, 0.0, 0.0]
    assert row["flags"][0] == "SCREEN_FAIL" and not row["screen"]["pass"]
    assert any(guard_text in r for r in row["screen"]["reasons"])
    assert any("largest slot share 1.00" in r for r in row["screen"]["reasons"])
    assert row["horizon"]["compare"]["status"] == "NOT_ESTIMABLE"
    assert row["cells"]["compare"][16]["auroc"] == 0.5 and row["cells"]["compare"][16]["c_auc"] == 0.0
    text = H.datasheet(row, selftest_result=selftest_result)
    assert "not estimable" in text and "FAIL@.40" in text


def test_a_screen_fail_row_prints_no_horizon_no_ratio_and_no_dissociation(selftest_result):
    # review E1: c(D0) = 0.30 under the bar with a real crossing used to print "64..256 | ~140"
    rec = H.synthetic_receipt("screenfail", [0.30, 0.22, 0.12, 0.05], RECALL_FLAT, 600, 7, arch_family="ATTN",
                              n_resamples=200, controls={"ablated": {"compare": {16: (0.0, None)}},
                                                         "near": {"compare": {1024: (0.25, None)}}})
    row = H.fit_model(H.load_receipt(rec))
    assert row["flags"][0] == "SCREEN_FAIL"
    for h in ("compare", "recall", "compare_auc"):
        assert row["horizon"][h]["status"] == "NOT_ESTIMABLE" and row["horizon"][h]["point"] is None
        assert row["horizon"][h]["reason"].startswith("SCREEN_FAIL")
    assert row["ratio_min"]["compare"]["point"] is None and row["ratio_min"]["recall"]["point"] is None
    assert row["dissociation"]["point"] is None and row["dissociation"]["ci_lo"] is None
    assert np.all(np.isnan(row["_rep"]["dissoc"]))
    assert row["controls"]["near"]["ratio_to_D0"] is None
    assert not {"COMPARE_ONLY_HORIZON", "FORMAT_DECAY@1024", "FORMAT_DRIFT@1024"} & set(row["flags"])
    assert row["curves"]["compare"][1] == pytest.approx(0.22, abs=0.01)    # the curve itself stays printed
    text = H.datasheet(row, selftest_result=selftest_result)
    h_line = next(l for l in text.splitlines() if "H_compare:" in l)
    assert "not estimable (screen failed)" in h_line and "~" not in h_line.split("H_compare:")[1]
    r_line = next(l for l in text.splitlines() if "ratio_min compare" in l)
    assert r_line.split("ratio_min")[1].count("not estimable (screen failed)") == 3
    assert not re.search(r"[+-]\d?\.\d\d", r_line)
    assert "screen failed: compare normalised" in text
    sheet = H.fit_zoo([H.load_receipt(rec)])
    assert sheet["result"]["per_model"]["dissoc"]["screenfail"] is None
    assert sheet["result"]["family_excluded_reasons"]["screenfail"].startswith("SCREEN_FAIL")


def test_the_compare_readout_its_chance_and_its_guard_are_read_from_the_receipt(selftest_result):
    arg = H.load_receipt(H.synthetic_receipt("a", H._planted_curve(128.0), RECALL_FLAT, 60, 3, n_resamples=20))
    assert arg["config"]["readouts"] == {"compare": "argmax_largest", "recall": "argmax_code"}
    assert arg["config"]["chance"] == {"compare": pytest.approx(1 / 3), "recall": pytest.approx(1 / 3)}
    assert arg["config"]["readout_source"]["compare"] == "declared"
    yn = H.load_receipt(H.synthetic_receipt("y", H._planted_curve(128.0), RECALL_FLAT, 60, 3, n_resamples=20,
                                            readout="yes_no"))
    assert yn["config"]["chance"]["compare"] == 0.5 and yn["cells"]["compare"][16]["yes_rate"] is not None
    row = H.fit_model(arg)
    text = H.datasheet(row, selftest_result=selftest_result)
    assert ("readout: compare argmax_largest (declared; chance 1/3; guard largest slot share <= 0.80) | "
            "recall argmax_code") in text
    # a declared chance that is not 1/k of the readout: the retired yes/no chance on an argmax receipt
    bad = H.synthetic_receipt("b", H._planted_curve(128.0), RECALL_FLAT, 60, 3, n_resamples=20)
    bad["config"]["chance"]["compare"] = 0.5
    with pytest.raises(H.HorizonRefusal, match="chance under exact balance"):
        H.load_receipt(bad)
    bad["config"]["chance"]["compare"] = 1 / 3
    bad["config"]["readout"] = "multiple_choice"
    with pytest.raises(H.HorizonRefusal, match="not a readout this instrument scores"):
        H.load_receipt(bad)
    # undeclared: the keys decide, and the sheet says the readout was inferred
    und = H.synthetic_receipt("u", H._planted_curve(128.0), RECALL_FLAT, 60, 3, n_resamples=20)
    for k in ("readout", "chance"):
        und["config"].pop(k)
    got = H.load_receipt(und)
    assert got["config"]["readouts"]["compare"] == "argmax_largest"
    assert got["config"]["readout_source"]["compare"].startswith("inferred")
    # undeclared and undecidable: 0/1 labels with no scores could be either readout
    amb = {"model": "amb", "per_item": {
        "compare": {str(d): {"label": [0, 1] * 30, "prediction": [0, 1] * 30} for d in H.GRID},
        "recall": {str(d): {"label": [0, 1, 2] * 20, "prediction": [0, 1, 2] * 20} for d in H.GRID}}}
    with pytest.raises(H.HorizonRefusal, match="neither declared"):
        H.load_receipt(amb)
    amb["config"] = {"readout": "yes_no"}
    assert H.load_receipt(amb)["config"]["chance"]["compare"] == 0.5


def test_a_cell_out_of_exact_balance_is_flagged_and_fails_the_screen():
    # review E6: labels 90% true and an item-blind responder at yes-rate 0.75 cleared bar and guard silently
    rng = np.random.default_rng(4)
    rec = H.synthetic_receipt("m", [0.6, 0.5, 0.45, 0.4], RECALL_FLAT, 600, 5, n_resamples=50, readout="yes_no")
    y = np.array([True] * 540 + [False] * 60)
    rng.shuffle(y)
    pred = rng.random(600) < 0.75
    lo = np.where(pred, 1.0, -1.0)
    rec["per_item"]["compare"]["16"] = {"label": y.tolist(), "prediction": pred.tolist(), "logodds": lo.tolist()}
    rec["per_item"]["compare"]["64"] = {"label": y.tolist(), "prediction": pred.tolist(), "logodds": lo.tolist()}
    row = H.fit_model(H.load_receipt(rec))
    assert "BALANCE_BROKEN@compare:64" in row["flags"]
    assert {"BALANCE_BROKEN@compare:16/screen", "BALANCE_BROKEN@compare:16/denominator"} <= set(row["flags"])
    assert not row["screen"]["pass"]
    assert any(r.startswith("compare not in exact balance") for r in row["screen"]["reasons"])
    assert row["cells"]["compare"][64]["exact_balance"] is False and row["cells"]["compare"][64]["label_counts"] == [60, 540]


def test_the_d0_split_is_stratified_so_both_halves_keep_exact_balance():
    rec = H.load_receipt(H.synthetic_receipt("m", [0.6, 0.5, 0.4, 0.3], RECALL_FLAT, 600, 3, n_resamples=100))
    cell = rec["cells"]["compare"][16]
    s_mask, d_mask, note = H._split_masks(cell)
    assert s_mask.sum() == 300 and d_mask.sum() == 300 and not np.any(s_mask & d_mask)
    for half in (s_mask, d_mask):
        assert np.bincount(cell["_labels"][half], minlength=3).tolist() == [100, 100, 100]
    assert "stratified" in note
    row = H.fit_model(rec)
    assert row["screen"]["n_compare"] == 300 and row["cells"]["compare"][16]["n"] == 300
    assert row["cells"]["compare"][64]["n"] == 600 and "stratified" in row["split_note"]
    assert not any(f.startswith("BALANCE_BROKEN") for f in row["flags"])


def test_the_screen_reads_one_half_of_D0_and_the_denominator_the_other():
    rec = H.synthetic_receipt("m", [0.9, 0.85, 0.8, 0.75], RECALL_FLAT, 600, 6, n_resamples=400)
    rec["per_item"]["compare"]["16"] = H._plant_halves("argmax_largest", 600, 0.45, 0.90, np.random.default_rng(6))
    row = H.fit_model(H.load_receipt(rec))
    assert row["screen"]["compare_norm_D0"] == pytest.approx(0.45, abs=1e-9)
    assert row["curves"]["compare"][0] == pytest.approx(0.90, abs=1e-9)
    c0 = row["cells"]["compare"][16]
    assert 0.80 < c0["ci_lo"] <= 0.90 <= c0["ci_hi"]


def test_the_screen_cannot_pass_on_correctness_bits_alone():
    rng = np.random.default_rng(0)
    bits = {str(d): (rng.random(600) < 0.9).tolist() for d in H.GRID}
    rec = H.load_receipt({"model": "bits-only", "config": {"readout": "argmax_largest", "pairing": "items"},
                          "per_item": {"compare": bits, "recall": bits}})
    row = H.fit_model(rec, n_resamples=50)
    assert not row["screen"]["pass"] and any("guard not recorded" in r for r in row["screen"]["reasons"])
    assert "SCORES_ABSENT" in row["flags"] and row["horizon"]["compare_auc"]["status"] == "NOT_ESTIMABLE"
    with pytest.raises(H.HorizonRefusal, match="neither declared"):
        H.load_receipt({"model": "bits-only", "per_item": {"compare": bits, "recall": bits}})


def test_the_screen_bar_is_printed_validated_and_flagged(selftest_result):
    rec = H.synthetic_receipt("m", H._planted_curve(203.0), RECALL_FLAT, 600, 8, n_resamples=100)
    for bad in (0.0, 1.0, -0.4, "high"):
        rec["config"]["screen_bar"] = bad
        with pytest.raises(H.HorizonRefusal, match="screen_bar"):
            H.load_receipt(rec)
    rec["config"]["screen_bar"] = 0.30
    row = H.fit_model(H.load_receipt(rec))
    assert "NONSTANDARD_BAR@0.30" in row["flags"] and row["screen"]["bar"] == 0.30
    text = H.datasheet(row, selftest_result=selftest_result)
    assert "PASS@.30" in text
    rec["config"]["screen_bar"] = 0.40
    assert "PASS@.40" in H.datasheet(H.fit_model(H.load_receipt(rec)), selftest_result=selftest_result)
    rec["config"]["guard"] = {"max_slot_fraction": 0.95}
    assert "NONSTANDARD_GUARD" in H.fit_model(H.load_receipt(rec))["flags"]
    rec["config"]["guard"] = {"max_slot_fraction": 0.2}
    with pytest.raises(H.HorizonRefusal, match="max_slot_fraction"):
        H.load_receipt(rec)


def test_a_declared_slot_bound_is_the_bound_the_guard_applies():
    # review 2: a guard that read the runner's 0.80 whatever config.guard declared passed every test
    def slot_cell():
        # 600 items in exact balance; slot 0 takes every item of label 0 and 170 of each other label, spread
        # so that each stratified D0 half also puts 0.90 of its answers on slot 0
        y = np.repeat([0, 1, 2], 200)
        p = np.where((y == 0) | (np.arange(600) % 20 >= 3), 0, y)
        return {"target_slot": y.tolist(), "pred_slot": p.tolist()}

    cell = slot_cell()
    for bound, ok in ((0.95, True), (0.80, False)):
        got = H.cell_summary("compare", cell["target_slot"], cell["pred_slot"], readout="argmax_largest",
                             guard_spec=H._guard_spec({"max_slot_fraction": bound}))
        assert got["max_slot_fraction"] == pytest.approx(0.90) and got["guard_ok"] is ok
    for bound, ok in ((0.95, True), (0.80, False)):
        rec = H.synthetic_receipt("m", H._planted_curve(203.0), RECALL_FLAT, 600, 8, n_resamples=50)
        rec["config"]["guard"] = {"max_slot_fraction": bound}
        rec["per_item"]["compare"]["16"] = slot_cell()
        rec["per_item"]["compare"]["64"] = slot_cell()
        row = H.fit_model(H.load_receipt(rec))
        assert row["cells"]["compare"][64]["guard_ok"] is ok and row["cells"]["compare"][16]["guard_ok"] is ok
        assert any(r.startswith("compare guard failed (largest slot share 0.90)")
                   for r in row["screen"]["reasons"]) is not ok
        assert ("DEGENERATE@compare:64" in " ".join(row["flags"])) is not ok


def test_an_undeclared_pairing_is_printed_as_assumed(selftest_result):
    rec = H.synthetic_receipt("silent", H._planted_curve(128.0), RECALL_FLAT, 600, 10, n_resamples=50)
    rec["config"].pop("pairing")
    row = H.fit_model(H.load_receipt(rec))
    assert row["pairing"] == "items" and "PAIRING_ASSUMED" in row["flags"]
    assert "pairing: items (assumed, undeclared)" in H.datasheet(row, selftest_result=selftest_result)
    rec["config"]["items_prefix_nested"] = True
    row = H.fit_model(H.load_receipt(rec))
    assert "PAIRING_ASSUMED" not in row["flags"] and row["pairing_source"].startswith("declared")


def test_guard_rules_are_the_bridge_v3_runner_s():
    assert H.guard(yes_rate=0.2) and H.guard(yes_rate=0.8) and not H.guard(yes_rate=0.81)
    assert H.guard(max_slot_frac=0.8) and not H.guard(max_slot_frac=0.801) and H.guard() is None


def test_the_replicate_auroc_is_the_auroc_of_each_resample():
    rng = np.random.default_rng(11)
    s = np.round(rng.normal(size=90), 1)            # rounded: plenty of ties
    y = rng.random(90) < 0.4
    idx = rng.integers(0, 90, size=(40, 90))
    keep = rng.random(90) < 0.5
    for km in (None, keep):
        got = H._rep_auroc(s, y, idx, km)
        for b in range(40):
            sel = idx[b] if km is None else idx[b][km[idx[b]]]
            want = H.auroc(s[sel], y[sel])
            assert (np.isnan(got[b]) and want is None) or got[b] == pytest.approx(want, abs=1e-12)
    S = rng.normal(size=(60, 3))
    t = np.array([0, 1, 2] * 20)
    assert H.ovr_auroc(S, t) == pytest.approx(np.mean([H.auroc(S[:, j], t == j) for j in range(3)]))
    assert H.ovr_auroc(S + np.array([5.0, 0.0, -3.0]), t) == pytest.approx(H.ovr_auroc(S, t))


@pytest.mark.parametrize("readout", ["argmax_largest", "yes_no"])
def test_the_replicate_c_auc_on_the_identity_resample_is_the_cell_s_own(readout):
    # the replicate path (vectorised, every slot) must agree with the point path on the unresampled items
    rec = H.load_receipt(H.synthetic_receipt("m", [0.6, 0.5, 0.4, 0.3], RECALL_FLAT, 120, 12, n_resamples=20,
                                             readout=readout, sample=True))
    for fam in H.FAMILIES:
        cell = rec["cells"][fam][64]
        ident = np.tile(np.arange(cell["n"]), (3, 1))
        assert np.allclose(H._rep_c_auc(cell, ident), cell["c_auc"], rtol=0.0, atol=1e-12)
        s_mask, d_mask, _ = H._split_masks(cell)
        half = H._subcell(cell, d_mask, "denominator half")
        assert np.allclose(H._rep_c_auc(cell, ident, d_mask), half["c_auc"], rtol=0.0, atol=1e-12)


# -- the bootstrap: paired, deterministic, shared across models ---------------------------------------

def test_the_bootstrap_is_a_function_of_the_seed():
    rec = H.load_receipt(H.synthetic_receipt("m", H._planted_curve(203.0), RECALL_FLAT, 600, 5, n_resamples=200))
    a, b = H.fit_model(rec), H.fit_model(rec)
    for key in ("horizon", "ratio_min", "dissociation", "curves"):
        assert json.dumps(H._jsonable(a[key])) == json.dumps(H._jsonable(b[key]))
    c = H.fit_model(rec, seed=H.SEED_BOOT + 1)
    assert c["horizon"]["compare"]["ci_lo"] != a["horizon"]["compare"]["ci_lo"]


def test_two_models_with_identical_bits_share_one_resample_and_give_one_family_interval():
    r1 = H.synthetic_receipt("a", H._planted_curve(400.0), RECALL_FLAT, 600, 9, arch_family="ATTN",
                             controls={"ablated": {"compare": {16: (0.02, None)}}}, n_resamples=200)
    r2 = json.loads(json.dumps(r1))
    r2["model"] = "b"
    sheet = H.fit_zoo([H.load_receipt(r1), H.load_receipt(r2)])
    res = sheet["result"]
    assert res["shared_resample"] is True and res["family_set"] == ["a", "b"]
    ra, rb = sheet["rows"]
    assert ra["dissociation"] == rb["dissociation"]
    assert res["attention_family_min_dissoc_ci"]["lo"] == ra["dissociation"]["ci_lo"]
    assert res["attention_family_min_dissoc_ci"]["hi"] == ra["dissociation"]["ci_hi"]
    assert res["attention_family_min_dissoc"] == H._round6(ra["dissociation"]["point"])


def _v3_receipt(n=96, distances=(16, 64, 256, 1024), seed=31, with_d1_controls=True):
    """A receipt in the shape run_realmodel_bridge_v3.py writes from its amendment 17.2:
    per_item[arm][family][D][variant], config.readout / chance / guard, items.prefix_nested flag."""
    rng = np.random.default_rng(seed)
    per_item = {}
    curves = {"M1": ([0.9, 0.6, 0.2, 0.0], [0.9, 0.88, 0.86, 0.84]), "T": ([0.9, 0.88, 0.86, 0.84], RECALL_FLAT)}
    for arm, (cc, rc) in curves.items():
        per_item[arm] = {"compare": {}, "recall": {}}
        for d, c_, r_ in zip(distances, cc, rc):
            for fam, val in (("compare", c_), ("recall", r_)):
                cell = {"plain": H.synthetic_cell(fam, val, n, rng), "ablated": H.synthetic_cell(fam, 0.0, n, rng)}
                for v in cell.values():
                    S = np.asarray(v["scores"])
                    v["margin"] = H._margins(S, np.asarray(v["target_slot"])).tolist()
                if fam == "compare" and d == distances[-1] and with_d1_controls:
                    cell["near"] = H.synthetic_cell(fam, 0.7, n, rng)
                    ow = H.synthetic_cell(fam, 0.05, n, rng)
                    ow["pred_is_overwrite"] = (rng.random(n) < 0.6).astype(int).tolist()
                    cell["overwrite"] = ow
                per_item[arm][fam][str(d)] = cell
    return {"prereg": "PREREG_realmodel_bridge_v3_2026_10_11.md", "mode": "main",
            "config": {"checkpoints": {"M1": "state-spaces/mamba-1.4b-hf", "T": "EleutherAI/pythia-1.4b"},
                       "readout": "argmax_largest", "answer_format": "argmax_largest",
                       "retired_formats": ["yes_no", "true_false"],
                       "chance": {"compare": 1 / 3, "recall": 1 / 3}, "distances": list(distances), "D0": 16,
                       "D1": 1024, "n_per_cell": n, "screen_bar": 0.4, "guard": {"max_slot_fraction": 0.8},
                       "ablated_bar": 0.16, "seeds": {"items": 20261011, "bootstrap": 20261011}},
            "per_item": per_item,
            "items": {"sha256": {}, "prefix_nested_across_distances": False, "code_paired_across_distances": False}}


def test_the_bridge_v3_runner_s_receipt_is_read_by_arm_in_the_shape_it_is_written(selftest_result):
    raw = _v3_receipt()
    rec = H.load_receipt(raw, arm="M1")
    cfg = rec["config"]
    assert rec["model"] == "state-spaces/mamba-1.4b-hf" and cfg["copied_from"] == "v3 receipt arm M1"
    assert cfg["readouts"]["compare"] == "argmax_largest" and cfg["readout_source"]["compare"] == "declared"
    assert cfg["pairing"] == "distance" and cfg["pairing_source"] == "declared (items.prefix_nested_across_distances)"
    assert set(rec["controls"]) == {"ablated", "near", "overwrite"}
    assert sorted(rec["controls"]["ablated"]["recall"]) == [16, 64, 256, 1024]
    row = H.fit_model(rec, n_resamples=300)
    assert row["horizon"]["compare"]["status"] == "FINITE" and row["horizon"]["compare_auc"]["status"] != "NOT_RECORDED"
    assert "FORMAT_DECAY@1024" not in row["flags"] and row["controls"]["overwrite"]["follow_rate"] is not None
    assert "PAIRING_ASSUMED" not in row["flags"] and "SCORES_ABSENT" not in row["flags"]
    text = H.datasheet(row, selftest_result=selftest_result)
    assert "follows the re-plant" in text and "pairing: distance" in text
    # under 'distance' pairing the two families share one resample per distance, drawn afresh per distance
    a, b = H.fit_model(rec, n_resamples=50), H.fit_model(rec, n_resamples=50)
    assert a["dissociation"] == b["dissociation"]
    # the two arms fitted together on one seed share their resamples
    sheet = H.fit_zoo([rec, H.load_receipt(raw, arm="T")], n_resamples=100)
    assert sheet["result"]["shared_resample"] is True and sheet["result"]["pairing"] == ["distance"]
    # the older shape: per_item[arm][family][D] is the cell itself
    old = {"config": raw["config"], "per_item": {"M1": {f: {d: v["plain"] for d, v in by_d.items()}
                                                         for f, by_d in raw["per_item"]["M1"].items()}}}
    assert H.load_receipt(old, arm="M1")["cells"]["compare"][64]["acc_norm"] == rec["cells"]["compare"][64]["acc_norm"]
    with pytest.raises(H.HorizonRefusal, match="Z4"):
        H.load_receipt({"config": raw["config"]}, arm="M1")
    with pytest.raises(H.HorizonRefusal, match="no per_item bits for arm 'M3'"):
        H.load_receipt(raw, arm="M3")
    with pytest.raises(H.HorizonRefusal, match="read by arm"):
        H.load_receipt(raw)
    # every recorded field must agree with every other
    broken = json.loads(json.dumps(raw))
    cell = broken["per_item"]["M1"]["compare"]["64"]["plain"]
    cell["pred_slot"] = [(p + 1) % 3 for p in cell["pred_slot"]]
    with pytest.raises(H.HorizonRefusal, match="argmax of the scores"):
        H.load_receipt(broken, arm="M1")
    broken = json.loads(json.dumps(raw))
    broken["per_item"]["M1"]["recall"]["16"]["plain"]["margin"][0] += 0.5
    with pytest.raises(H.HorizonRefusal, match="margin disagrees"):
        H.load_receipt(broken, arm="M1")
    broken = json.loads(json.dumps(raw))
    broken["per_item"]["M1"]["recall"]["16"]["plain"]["correct"][0] ^= 1
    with pytest.raises(H.HorizonRefusal, match="correctness bits disagree"):
        H.load_receipt(broken, arm="M1")


def test_a_one_distance_receipt_is_read_as_cells_and_refused_as_a_curve(tmp_path, capsys, monkeypatch,
                                                                         selftest_result):
    raw = _v3_receipt(n=48, distances=(16,), with_d1_controls=False)
    with pytest.raises(H.HorizonRefusal, match=r"the grid \[16\] has 1 distance"):
        H.load_receipt(raw, arm="M1")
    rec = H.load_receipt(raw, arm="M1", min_grid=1)
    assert rec["cells"]["compare"][16]["n"] == 48
    with pytest.raises(H.HorizonRefusal, match="one distance"):
        H.fit_model(rec)
    p = tmp_path / "v3_pilot_like.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(H, "selftest", lambda *a, **k: selftest_result)
    assert H.main(["fit", "--receipt", str(p), "--arm", "M1"]) == 2
    assert "has 1 distance" in capsys.readouterr().err


def _pilot2_raw():
    data = None
    p = ROOT / PILOT2_PATH
    if p.exists():
        data = p.read_bytes()
    else:
        try:
            r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", PILOT2_BLOB], capture_output=True,
                               timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            r = None
        if r is not None and r.returncode == 0:
            data = r.stdout
    if data is None:
        pytest.skip("the bridge-v3 amendment-17.2 pilot receipt is not in this checkout or its object store")
    data = data.replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == PILOT2_SHA256_LF, (
        "the committed pilot2 receipt's bytes changed: a receipt is history and is never rewritten")
    return json.loads(data.decode("utf-8"))


def test_the_committed_v3_pilot2_receipt_s_per_item_bits_reproduce_its_own_numbers():
    raw = _pilot2_raw()
    assert raw["config"]["readout"] == "argmax_largest" and raw["config"]["distances"] == [16]
    expect_share = {"M1": "0.92", "T": "1.00"}
    for arm in ("M1", "T"):
        rec = H.load_receipt(raw, arm=arm, min_grid=1)
        assert rec["model"] == raw["config"]["checkpoints"][arm]
        assert rec["config"]["readouts"] == {"compare": "argmax_largest", "recall": "argmax_code"}
        assert rec["config"]["readout_source"]["compare"] == "declared"
        assert rec["config"]["pairing"] == "distance" and rec["config"]["pairing_declared"]
        for fam in H.FAMILIES:
            cell = rec["cells"][fam][16]
            g = raw["guard"][arm][fam]["16"]
            assert cell["n"] == 48 and cell["exact_balance"] and cell["label_counts"] == [16, 16, 16]
            assert cell["acc_norm"] == pytest.approx(raw["acc_norm"][arm][fam]["16"], abs=1e-12)
            assert cell["slot_fractions"] == pytest.approx(g["slot_fractions"], abs=1e-12)
            assert cell["guard_ok"] is g["guard_ok"]
            assert cell["auroc"] == pytest.approx(g["ovr_auroc"], abs=1e-12)
            assert cell["margin_mean"] == pytest.approx(g["margin_mean"], abs=1e-9)
            assert cell["near_ties"] == g["near_ties"]
            abl = rec["controls"]["ablated"][fam][16]
            ga = raw["controls"]["ablated"][arm][fam]["16"]
            assert abl["acc_norm"] == pytest.approx(ga["acc_norm"], abs=1e-12)
            assert abl["auroc"] == pytest.approx(ga["ovr_auroc"], abs=1e-12)
        scr = H.screen(rec["cells"]["compare"][16], rec["cells"]["recall"][16])
        assert scr["pass"] is raw["screen"][arm]["clears"] is False
        assert f"compare guard failed (largest slot share {expect_share[arm]})" in scr["reasons"]
        assert not any(r.startswith("recall") for r in scr["reasons"])
        with pytest.raises(H.HorizonRefusal, match="has 1 distance"):
            H.load_receipt(raw, arm=arm)
        with pytest.raises(H.HorizonRefusal, match="one distance"):
            H.fit_model(rec)
    assert raw["pilot_outcome"] == "NEITHER"


def test_the_family_aggregates_satisfy_a_declared_composition_in_the_lab_s_scorer():
    from styxx.protocol import Experiment
    recs = [H.load_receipt(H.synthetic_receipt("attn-1", [0.9, 0.85, 0.8, 0.7], RECALL_FLAT, 600, 11, arch_family="ATTN",
                                               controls={"ablated": {"compare": {16: (0.02, None)}}}, n_resamples=100)),
            H.load_receipt(H.synthetic_receipt("attn-2", H._planted_curve(300.0), RECALL_FLAT, 600, 12, arch_family="ATTN",
                                               controls={"ablated": {"compare": {16: (0.30, None)}}}, n_resamples=100)),
            H.load_receipt(H.synthetic_receipt("decay-1", H._planted_curve(100.0), RECALL_FLAT, 600, 13, arch_family="DECAY",
                                               controls={"ablated": {"compare": {16: (0.01, None)}}}, n_resamples=100)),
            H.load_receipt(H.synthetic_receipt("T-copy", [0.9, 0.9, 0.9, 0.9], RECALL_FLAT, 600, 14, arch_family="ATTN",
                                               controls={"ablated": {"compare": {16: (0.01, None)}}}, n_resamples=100))]
    sheet = H.fit_zoo(recs, exclude=["T-copy"])
    res = sheet["result"]
    assert res["family_set"] == ["attn-1"] and res["n_attention_screen_pass"] == 1 and res["n_family_screened"] == 2
    assert set(res["family_excluded"]) == {"attn-2", "decay-1", "T-copy"}
    assert res["family_excluded_reasons"]["attn-2"].startswith("PROBE_READS_PRIORS")
    # M1: the ablated maximum is over the screened set BEFORE the ablation exclusion, so it can fail
    assert res["family_set_ablated_max_norm_acc"] == pytest.approx(0.30, abs=0.02) and res["family_set_ablated_max_norm_acc"] > H.ABLATED_BAR
    exp = Experiment.__new__(Experiment)
    exp.composition = {"G": {"agg": "min", "over": "result.per_model.dissoc", "excluding": "result.family_excluded"}}
    exp._check_composition("G", res["attention_family_min_dissoc"], {"result": H._jsonable(res)})
    with pytest.raises(Exception, match="COMPOSITION VIOLATION"):
        exp._check_composition("G", res["attention_family_min_dissoc"] - 0.5, {"result": H._jsonable(res)})


def _commit_prereg(d: Path, spec: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "PREREG_zoo_case.md"
    p.write_text("# case\n\n```gates\n" + json.dumps(spec) + "\n```\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "case"], cwd=d, check=True)
    return p


_READINESS = {"gates": {"G_A_family_n": {"metric": "result.n_attention_screen_pass", "op": ">=", "value": 3,
                                         "power_basis": "a family read off fewer than three models is a model"}},
              "outcomes": [{"when": {"G_A_family_n": False}, "verdict": "ABSTAIN__fewer_than_three_attention_models"},
                           {"when": {"G_A_family_n": True}, "verdict": "READY__score_the_family_block"}],
              "smoke_verdict": "INVALID__smoke_plumbing_only"}
_FAMILY = {"gates": {"G_A_family_n": _READINESS["gates"]["G_A_family_n"],
                     "G_C_abl": {"metric": "result.family_set_ablated_max_norm_acc", "op": "<=", "value": 0.16,
                                 "power_basis": "twice the chance band"},
                     "G_F": {"metric": "result.attention_family_min_dissoc", "op": ">=", "value": -0.2,
                             "agg": "min", "over": "result.per_model.dissoc", "excluding": "result.family_excluded",
                             "power_basis": "the dissociation the record predicts"}},
           "outcomes": [{"when": {"G_A_family_n": False}, "verdict": "ABSTAIN__fewer_than_three_attention_models"},
                        {"when": {"G_A_family_n": True, "G_C_abl": False}, "verdict": "INVALID__probe_reads_priors"},
                        {"when": {"G_A_family_n": True, "G_C_abl": True, "G_F": True}, "verdict": "FAMILY_HOLDS"},
                        {"when": {"G_A_family_n": True, "G_C_abl": True, "G_F": False}, "verdict": "FAMILY_FAILS"}],
           "smoke_verdict": "INVALID__smoke_plumbing_only"}


def test_an_empty_family_set_scores_to_abstain_through_styxx_protocol(tmp_path):
    # review E7: nine SCREEN_FAIL rows (NEXT_AFTER_V3 3.4) — every family aggregate is None, which the
    # scorer refuses by design; the counts are finite, and a readiness block reading them alone walks
    from styxx.protocol import Experiment, GateSpecError
    fails = [H.load_receipt(H.synthetic_receipt(f"attn-{k}", [0, 0, 0, 0], [0, 0, 0, 0], 60, 40 + k, arch_family="ATTN",
                                                constant_compare=1, constant_recall=0, n_resamples=50,
                                                controls={"ablated": {"compare": {16: (0.0, None)}}}))
             for k in range(3)]
    sheet = H.fit_zoo(fails)
    res = sheet["result"]
    assert res["family_set"] == [] and res["n_attention_screen_pass"] == 0 and res["n_family_screened"] == 0
    assert res["attention_family_min_dissoc"] is None and "readiness" in res["family_aggregates_note"]
    ready = _commit_prereg(tmp_path / "ready", _READINESS)
    v = Experiment(ready).score({"result": H._jsonable(res)})
    assert v.verdict == "ABSTAIN__fewer_than_three_attention_models" and v.gates == {"G_A_family_n": False}
    assert H._score_prereg(str(ready), sheet)["verdict"] == "ABSTAIN__fewer_than_three_attention_models"
    family = _commit_prereg(tmp_path / "family", _FAMILY)
    with pytest.raises(GateSpecError):
        Experiment(family).score({"result": H._jsonable(res)})
    with pytest.raises(H.HorizonRefusal, match="styxx.protocol refused to score the sheet"):
        H._score_prereg(str(family), sheet)
    # a family set of one: the family block itself walks to ABSTAIN, and its composition holds
    one = H.load_receipt(H.synthetic_receipt("attn-ok", [0.9, 0.85, 0.8, 0.7], RECALL_FLAT, 600, 50, arch_family="ATTN",
                                             controls={"ablated": {"compare": {16: (0.02, None)}}}, n_resamples=100))
    sheet1 = H.fit_zoo(fails + [one])
    assert sheet1["result"]["family_set"] == ["attn-ok"]
    assert Experiment(family).score({"result": H._jsonable(sheet1["result"])}).verdict == "ABSTAIN__fewer_than_three_attention_models"


def test_a_sheet_scored_against_a_prereg_reads_only_declared_readouts_and_pairings(tmp_path):
    rec = H.synthetic_receipt("attn", [0.9, 0.85, 0.8, 0.7], RECALL_FLAT, 60, 51, arch_family="ATTN", n_resamples=50)
    rec["config"].pop("pairing")
    sheet = H.fit_zoo([H.load_receipt(rec)])
    ready = _commit_prereg(tmp_path / "ready", _READINESS)
    with pytest.raises(H.HorizonRefusal, match="the pairing is assumed"):
        H._score_prereg(str(ready), sheet)
    rec["config"]["pairing"] = "items"
    rec["config"]["screen_bar"] = 0.3
    with pytest.raises(H.HorizonRefusal, match="the screen bar is 0.3"):
        H._score_prereg(str(ready), H.fit_zoo([H.load_receipt(rec)]))


# -- receipts: both shapes, the flat list, the refusals -----------------------------------------------

def test_the_flat_item_list_and_the_cell_block_give_the_same_row():
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 300, 21, n_resamples=100)
    items = []
    for fam, by_d in rec["per_item"].items():
        for d, cell in by_d.items():
            n = len(next(iter(cell.values())))
            for i in range(n):
                it = {"family": fam, "distance": int(d), "item": i}
                it.update({k: v[i] for k, v in cell.items()})
                items.append(it)
    flat = {"schema": H.RECEIPT_SCHEMA, "model": "m", "config": rec["config"], "items": items[::-1]}
    a, b = H.fit_model(H.load_receipt(rec)), H.fit_model(H.load_receipt(flat))
    assert a["curves"] == b["curves"] and a["horizon"]["compare"] == b["horizon"]["compare"]
    assert a["curve_auc"] == b["curve_auc"]


@pytest.mark.parametrize("edit,match", [
    (lambda r: r["per_item"].pop("recall"), "no recall cells"),
    (lambda r: r["per_item"]["compare"].pop("1024"), "no cell at distance"),
    (lambda r: r.update(schema="styxx.horizon/receipt/9.9"), "schema"),
    (lambda r: r.update(controls={"shuffled": {}}), "unknown control"),
    (lambda r: r["per_item"]["compare"]["16"].update(pred_slot=[(p + 1) % 3 for p in r["per_item"]["compare"]["16"]["pred_slot"]]),
     "argmax of the scores"),
    (lambda r: r["config"].update(pairing="sometimes"), "pairing"),
    (lambda r: r.update(arch_family={"family": "HYBRID"}), "arch_family"),
    (lambda r: r["per_item"]["recall"]["64"].update(target_slot=[7] * 60), "slots must be 0, 1 or 2"),
    (lambda r: r["per_item"]["recall"]["64"].update(target_slot=[True] * 60), "expected a flat list of slot indices"),
    (lambda r: r["config"].update(readout=None, readouts={"compare": "argmax_code"}), "not one the instrument scores"),
])
def test_malformed_receipts_are_refused_by_name(edit, match):
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 60, 23, n_resamples=20)
    edit(rec)
    with pytest.raises(H.HorizonRefusal, match=match):
        H.load_receipt(rec)


def test_a_yes_no_cell_whose_predictions_disagree_with_its_log_odds_is_refused():
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 60, 23, n_resamples=20, readout="yes_no")
    rec["per_item"]["compare"]["16"]["prediction"] = [not p for p in rec["per_item"]["compare"]["16"]["prediction"]]
    with pytest.raises(H.HorizonRefusal, match="disagree with the sign"):
        H.load_receipt(rec)


def test_cells_of_unequal_size_are_refused_under_item_pairing_and_accepted_when_declared_independent():
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 60, 24, n_resamples=20)
    rng = np.random.default_rng(0)
    rec["per_item"]["compare"]["64"] = H.synthetic_cell("compare", 0.6, 42, rng)
    with pytest.raises(H.HorizonRefusal, match="differ in item count"):
        H.fit_model(H.load_receipt(rec))
    rec["config"]["pairing"] = "distance"
    with pytest.raises(H.HorizonRefusal, match="share items at one distance"):
        H.fit_model(H.load_receipt(rec))
    rec["config"]["pairing"] = "independent"
    row = H.fit_model(H.load_receipt(rec))
    assert row["pairing"] == "independent" and row["horizon"]["compare"]["n_resamples"] == 20


def test_rows_are_json_without_nan(tmp_path):
    row = _row([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], constant_compare=1, constant_recall=1)
    out = tmp_path / "row.json"
    H._write_json(str(out), row)
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "_rep" not in text and "_guard_spec" not in text
    back = json.loads(text)
    assert back["ratio_min"]["compare"]["point"] is None and back["model"] == "m"


# -- controls and their flags ------------------------------------------------------------------------

def test_controls_set_their_flags_and_their_absence_is_printed():
    ctrl = {"ablated": {"compare": {16: (0.02, None), 1024: (0.05, None)}, "recall": {16: (0.0, None)}},
            "near": {"compare": {1024: (0.85, None)}}, "overwrite": {"compare": {1024: (0.03, 198)}}}
    row = _row(H._planted_curve(203.0), controls=ctrl)
    assert {"COMPARE_ONLY_HORIZON", "FORMAT_DISTANCE_EXCLUDED", "POSITIVE_CONTROL_FIRED"} <= set(row["flags"])
    assert not row["controls"]["ablated"]["reads_priors"]
    bare = _row(H._planted_curve(203.0))
    assert {"ABLATED_CONTROL_ABSENT", "FORMAT_DISTANCE_NOT_EXCLUDED", "FAR_BINDING_CONTROL_ABSENT"} <= set(bare["flags"])
    decayed = _row(H._planted_curve(203.0), controls={"near": {"compare": {1024: (0.3, None)}},
                                                      "overwrite": {"compare": {1024: (0.6, 198)}},
                                                      "ablated": {"compare": {16: (0.4, None)}}})
    assert "FORMAT_DECAY@1024" in decayed["flags"] and "POSITIVE_CONTROL_SILENT@1024" in decayed["flags"]
    assert "PROBE_READS_PRIORS" in decayed["flags"]


def test_degenerate_cells_past_D0_are_flagged_with_both_numbers():
    rec = H.synthetic_receipt("m", [0.9, 0.8, 0.7, 0.6], RECALL_FLAT, 600, 31, n_resamples=50)
    rng = np.random.default_rng(1)
    rec["per_item"]["compare"]["1024"] = H.synthetic_cell("compare", 0.0, 600, rng, constant=1)
    row = H.fit_model(H.load_receipt(rec))
    flag = next(f for f in row["flags"] if f.startswith("DEGENERATE@compare:1024"))
    assert "c=+0.00" in flag and "c_auc=+0.00" in flag and row["cells"]["compare"][1024]["degenerate"]


# -- the architecture family --------------------------------------------------------------------------

def test_the_family_is_a_fact_of_config_and_weights_never_a_name():
    with pytest.raises(H.HorizonRefusal, match="not a fact"):
        H.assert_family("state-spaces/mamba-1.4b-hf")
    with pytest.raises(H.HorizonRefusal, match="no model_type"):
        H.assert_family({"architectures": ["MambaForCausalLM"]})
    with pytest.raises(H.HorizonRefusal, match="needs its A values"):
        H.assert_family({"model_type": "mamba"})
    ok = H.assert_family({"model_type": "mamba", "num_hidden_layers": 2}, A=[[-1.0, -3.7], [-0.18, -65.9]])
    assert ok["family"] == "DECAY" and ok["fact"]["max_A"] == -0.18 and ok["fact"]["layers"] == 2
    assert H.assert_family({"model_type": "mamba"}, A={"A_log": [[0.0, 1.0]]})["fact"]["min_A"] == pytest.approx(-np.e)
    with pytest.raises(H.HorizonRefusal, match="not strictly negative"):
        H.assert_family({"model_type": "mamba"}, A=[[-1.0, 0.0]])
    with pytest.raises(H.HorizonRefusal, match="complex"):
        H.assert_family({"model_type": "mamba"}, A=[np.array([-1.0 + 0.5j])])
    attn = H.assert_family({"model_type": "gemma2", "max_position_embeddings": 8192, "sliding_window": 4096})
    assert attn["family"] == "ATTN" and attn["fact"]["reach_tokens"] == 4096
    assert H.assert_family({"model_type": "qwen2", "max_position_embeddings": 32768, "sliding_window": 512,
                            "use_sliding_window": False})["fact"]["reach_tokens"] == 32768
    with pytest.raises(H.HorizonRefusal, match="reaches 512 tokens"):
        H.assert_family({"model_type": "llama", "max_position_embeddings": 131072, "sliding_window": 512})
    with pytest.raises(H.HorizonRefusal, match="refuses to guess"):
        H.assert_family({"model_type": "jamba"})


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = ((arr.view(np.uint32) >> 16).astype(np.uint16).tobytes() if dt == "BF16"
             else arr.astype("<f2" if dt == "F16" else "<f4").tobytes())
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def test_A_log_is_read_from_safetensors_without_torch_and_without_a_model_load(tmp_path):
    a0 = np.log(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
    a1 = np.log(np.array([[0.5, 2.0, 8.0]], dtype=np.float32))
    _write_safetensors(tmp_path / "m.safetensors", {
        "backbone.layers.1.mixer.A_log": (a1, "BF16"), "backbone.layers.0.mixer.A_log": (a0, "F32"),
        "backbone.layers.0.mixer.D": (np.ones(3, dtype=np.float32), "F32")})
    got = H.mamba_A_from_safetensors([str(tmp_path / "m.safetensors")])
    assert got["layers"] == 2 and got["names"][0].endswith("layers.0.mixer.A_log")
    assert np.allclose(got["A"][0], -np.arange(1, 7).reshape(2, 3))
    assert np.allclose(got["A"][1], [[-0.5, -2.0, -8.0]], rtol=1e-2)   # the BF16 layer
    fact = H.assert_family({"model_type": "mamba"}, A=got["A"])
    assert fact["family"] == "DECAY" and fact["fact"]["min_A"] == pytest.approx(-8.0, rel=1e-2)
    _write_safetensors(tmp_path / "none.safetensors", {"x.D": (np.ones(2, dtype=np.float32), "F32")})
    with pytest.raises(H.HorizonRefusal, match="no tensor named"):
        H.mamba_A_from_safetensors([str(tmp_path / "none.safetensors")])
    (tmp_path / "junk.safetensors").write_bytes(b"\x00" * 4)
    with pytest.raises(H.HorizonRefusal, match="short header"):
        H.mamba_A_from_safetensors([str(tmp_path / "junk.safetensors")])


# -- the datasheet and its words ---------------------------------------------------------------------

def _lab_banned_words():
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    words = list(H._HOUSE_WORDS)
    if src.exists():
        m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
        if m:
            words = sorted(set(words) | set(re.findall(r'"([^"]+)"', m.group(1))))
    return re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)


def test_the_datasheet_prints_its_limits_in_the_lab_s_words(selftest_result):
    ctrl = {"ablated": {"compare": {16: (0.02, None)}}, "near": {"compare": {1024: (0.85, None)}}}
    rec = H.synthetic_receipt("decay-like", H._planted_curve(203.0), RECALL_FLAT, 600, 41, arch_family="DECAY",
                              controls=ctrl, n_resamples=200)
    rec["config"]["native_D"] = {"16": {"min": 15, "mean": 15.6, "max": 18}, "64": {"min": 61, "mean": 62.4, "max": 66},
                                 "256": {"min": 246, "mean": 249.5, "max": 258}, "1024": {"min": 990, "mean": 998.0, "max": 1010}}
    rec["config"].update({"dtype": "fp16", "params": "1372M", "ctx": 2048, "ctx_is_training_length": True})
    row = H.fit_model(H.load_receipt(rec))
    text = H.datasheet(row, selftest_result=selftest_result)
    assert "64..256 | ~" in text and "native" in text and "H_recall:  > 1024" in text
    assert "licenses no sentence about rotation, honesty, causation or ranking" in text
    assert "inside the tested range" in text and "inside context" not in text
    assert "2048*" in text and "COMPARE_ONLY_HORIZON" in text and "verdict:" not in text
    assert "gates sha256: none (unscored)" in text and "spearman(c, log2 D): compare -1.00" in text
    assert row["horizon"]["compare"]["native_point"] is not None
    banned = _lab_banned_words()
    assert not banned.findall(text), banned.findall(text)
    sheet = H.fit_zoo([H.load_receipt(rec)], exclude=[])
    full = H.datasheet(sheet, selftest_result=selftest_result)
    assert "verdict: unscored: no committed prereg" in full and "DECAY family" not in full
    assert not banned.findall(full), banned.findall(full)


def test_the_datasheet_refuses_to_print_a_house_word(monkeypatch, selftest_result):
    row = _row(H._planted_curve(128.0))
    monkeypatch.setattr(H, "LIMITS", H.LIMITS + ("a " + "ground" + "breaking" + " instrument",))
    with pytest.raises(H.HorizonRefusal, match="never prints"):
        H.datasheet(row, selftest_result=selftest_result)


def test_the_module_its_readme_and_this_file_carry_no_house_word():
    banned = _lab_banned_words()
    for p in (ROOT / "styxx" / "horizon.py", ROOT / "styxx" / "horizon_README.md", Path(__file__)):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p.name} has a BOM"
        hits = banned.findall(raw.decode("utf-8"))
        assert not hits, f"{p.name}: {hits}"


# -- the CLI --------------------------------------------------------------------------------------------

def test_the_cli_fits_a_receipt_and_refuses_a_missing_one(tmp_path, capsys, monkeypatch, selftest_result):
    monkeypatch.setattr(H, "selftest", lambda *a, **k: selftest_result)
    rec = H.synthetic_receipt("cli", H._planted_curve(128.0), RECALL_FLAT, 300, 51, n_resamples=100)
    p = tmp_path / "horizon_cli.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    out = tmp_path / "row.json"
    assert H.main(["fit", "--receipt", str(p), "--json", str(out)]) == 0
    assert "styxx.horizon datasheet" in capsys.readouterr().out and json.loads(out.read_text(encoding="utf-8"))["model"] == "cli"
    assert H.main(["sheet", "--receipts", str(tmp_path)]) == 0
    assert "unscored: no committed prereg" in capsys.readouterr().out
    assert H.main(["fit", "--receipt", str(tmp_path / "absent.json")]) == 2
    assert "REFUSED: cannot read receipt" in capsys.readouterr().err
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 131072}), encoding="utf-8")
    assert H.main(["family", "--config", str(cfg)]) == 0 and '"ATTN"' in capsys.readouterr().out
    assert H.main(["family", "--config", "a-name"]) == 2
