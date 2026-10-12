# -*- coding: utf-8 -*-
"""PREREG_pmnist_robustness_2026_10_11 and run_pmnist_robustness.py, on CPU, with no data and no GPU.

What is pinned here: the gates block parses through the lab's own scanner and every one of the 1,024 boolean
assignments reaches a row; `Experiment` scores a receipt of the PREREG's shape to each of the thirteen verdicts and
refuses nothing it should accept; the frozen hashes refuse a mutated copy of either imported runner (and accept a CRLF
copy, which is the trap SEALS names); the permutation derivation is deterministic, skips 1234 and repeats, and the
fallback list equals the one the PREREG prints; the aggregate arithmetic on synthetic leaves gives the hand-computed
numbers, counts a collapsed FREE run as 0 in the floor, excludes a collapsed decay run from the gap, maps a missing
leaf, a foreign import hash, a read-back mismatch and a non-finite metric to plumbing_ok = 0; the _mnist.npz pin
refuses a different file; the sealed digest in SEALS row 7 is the committed blob's; and none of the three files
prints the words the lab never prints. Nothing here loads a checkpoint, reads the data or touches the card.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FREQ = ROOT / "papers" / "frequency-resonance"
PREREG = FREQ / "PREREG_pmnist_robustness_2026_10_11.md"
RUNNER = FREQ / "run_pmnist_robustness.py"
SEALS = ROOT / "papers" / "checksum" / "SEALS_2026_09_13.md"
TEST_FILE = Path(__file__).resolve()
LIVE_DATA = Path(r"C:/Users/heyzo/clawd/styxx/papers/frequency-resonance/_mnist.npz")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_pmnist_robustness", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _load_runner()
VERDICTS = [
    "INVALID__plumbing", "INVALID__training_unstable", "ABSTAIN__free_arm_below_floor",
    "HEADLINE_NOT_REPRODUCED__anchor_permutation_three_new_seeds", "NOT_ROBUST__gap_confined_to_anchor_permutation",
    "ROBUST__TIMESCALE_DIVERSITY__rotation_not_load_bearing_here", "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity",
    "ROBUST__PARTIAL__diversity_recovers_some", "BUDGET_CONFOUND__gap_does_not_survive_3x_steps",
    "ATTENUATED_VARIABLE__TIMESCALE_DIVERSITY__interval_reaches_below_widens_bar",
    "ATTENUATED_VARIABLE__ROTATION_LOAD_BEARING__interval_reaches_below_widens_bar",
    "ATTENUATED_VARIABLE__PARTIAL__interval_reaches_below_widens_bar",
    "ATTENUATED__gap_below_half_receipt_on_permutation_average",
]


# -- the gates block --------------------------------------------------------------------------------------

def test_gates_block_parses_and_every_assignment_reaches_a_row():
    spec = RB.gates_spec()
    names = list(spec["gates"])
    assert names == ["G_P_plumbing", "G_D_stable", "G_F_floor", "G_C_anchor", "G_G_exists", "G_G_mean", "G_G_ci",
                     "G_B_long", "G_R_diversity", "G_R_rotation"]
    assert spec["smoke_verdict"] == "INVALID__smoke_plumbing_only"
    assert [r["verdict"] for r in spec["outcomes"]] == VERDICTS
    seen = {}
    for bits in itertools.product((False, True), repeat=len(names)):
        v = RB.walk_outcomes(spec, dict(zip(names, bits)))
        assert v is not None, dict(zip(names, bits))
        seen[v] = seen.get(v, 0) + 1
    assert set(seen) == set(VERDICTS)                       # every row reachable, none vacuous
    assert sum(seen.values()) == 1024
    mentioned = {k for row in spec["outcomes"] for k in row["when"]}
    assert mentioned == set(names)
    for g in spec["gates"].values():
        assert isinstance(g["power_basis"], str) and len(g["power_basis"].strip()) > 40
        assert g["metric"].startswith("result.") and g["metric"][len("result."):] in RB.GATED
    assert {g["metric"][len("result."):] for g in spec["gates"].values()} == set(RB.GATED)
    # the realizable paths
    T = dict(zip(names, [True] * 10))
    assert RB.walk_outcomes(spec, T) == "ROBUST__TIMESCALE_DIVERSITY__rotation_not_load_bearing_here"
    assert RB.walk_outcomes(spec, dict(T, G_R_diversity=False)) == "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"
    assert RB.walk_outcomes(spec, dict(T, G_R_diversity=False, G_R_rotation=False)) == "ROBUST__PARTIAL__diversity_recovers_some"
    assert RB.walk_outcomes(spec, dict(T, G_B_long=False)) == "BUDGET_CONFOUND__gap_does_not_survive_3x_steps"
    assert RB.walk_outcomes(spec, dict(T, G_G_ci=False, G_R_diversity=False)) == \
        "ATTENUATED_VARIABLE__ROTATION_LOAD_BEARING__interval_reaches_below_widens_bar"
    assert RB.walk_outcomes(spec, dict(T, G_G_mean=False, G_G_ci=False)) == "ATTENUATED__gap_below_half_receipt_on_permutation_average"
    assert RB.walk_outcomes(spec, dict(T, G_G_exists=False, G_G_mean=False, G_G_ci=False)) == \
        "NOT_ROBUST__gap_confined_to_anchor_permutation"
    assert RB.walk_outcomes(spec, dict(T, G_C_anchor=False)) == "HEADLINE_NOT_REPRODUCED__anchor_permutation_three_new_seeds"
    assert RB.walk_outcomes(spec, dict(T, G_F_floor=False)) == "ABSTAIN__free_arm_below_floor"
    assert RB.walk_outcomes(spec, dict(T, G_D_stable=False)) == "INVALID__training_unstable"
    assert RB.walk_outcomes(spec, dict(T, G_P_plumbing=False)) == "INVALID__plumbing"


def test_gates_block_has_one_fence_and_no_duplicate_keys():
    from styxx.protocol import _select_gates_block
    text = PREREG.read_text(encoding="utf-8")
    block = _select_gates_block(text)

    def no_dup(pairs):
        d = {}
        for k, v in pairs:
            assert k not in d, f"duplicate key {k}"
            d[k] = v
        return d
    json.loads(block, object_pairs_hook=no_dup)
    assert text.count("```gates") == 1


def _receipt(**kw):
    base = {"plumbing_ok": 1.0, "degenerate_fraction": 0.0, "free_perm_mean": 0.92, "anchor_gap_newseed": 0.31,
            "gap_perm_mean": 0.28, "gap_perm_ci_lo": 0.21, "gap_long": 0.30,
            "free_minus_real2_perm_mean": 0.27, "real2_minus_clamped_perm_mean": 0.01}
    base.update(kw)
    return {"result": base}


def _experiment():
    from styxx.protocol import Experiment, PrologueError
    try:
        return Experiment(PREREG, repo_root=ROOT)
    except PrologueError:
        pytest.skip("the PREREG is not committed in this checkout")


def test_experiment_scores_the_receipt_shape_to_every_verdict():
    exp = _experiment()
    assert exp.vacuous_gates == [] and exp.undeclared_power_gates == []
    assert all(v["usable"] for v in exp.check_metrics(RB.synthetic_result()).values())
    assert exp.score(_receipt()).verdict == "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"
    assert exp.score(_receipt(free_minus_real2_perm_mean=0.02, real2_minus_clamped_perm_mean=0.26)).verdict == \
        "ROBUST__TIMESCALE_DIVERSITY__rotation_not_load_bearing_here"
    assert exp.score(_receipt(free_minus_real2_perm_mean=0.14, real2_minus_clamped_perm_mean=0.14)).verdict == \
        "ROBUST__PARTIAL__diversity_recovers_some"
    assert exp.score(_receipt(gap_long=0.12)).verdict == "BUDGET_CONFOUND__gap_does_not_survive_3x_steps"
    assert exp.score(_receipt(gap_perm_ci_lo=0.04)).verdict == \
        "ATTENUATED_VARIABLE__ROTATION_LOAD_BEARING__interval_reaches_below_widens_bar"
    assert exp.score(_receipt(gap_perm_ci_lo=0.04, free_minus_real2_perm_mean=0.02, real2_minus_clamped_perm_mean=0.26)).verdict == \
        "ATTENUATED_VARIABLE__TIMESCALE_DIVERSITY__interval_reaches_below_widens_bar"
    assert exp.score(_receipt(gap_perm_ci_lo=0.04, free_minus_real2_perm_mean=0.14, real2_minus_clamped_perm_mean=0.14)).verdict == \
        "ATTENUATED_VARIABLE__PARTIAL__interval_reaches_below_widens_bar"
    assert exp.score(_receipt(gap_perm_mean=0.10, gap_perm_ci_lo=0.03)).verdict == \
        "ATTENUATED__gap_below_half_receipt_on_permutation_average"
    assert exp.score(_receipt(gap_perm_mean=0.10, gap_perm_ci_lo=-0.02)).verdict == "NOT_ROBUST__gap_confined_to_anchor_permutation"
    assert exp.score(_receipt(anchor_gap_newseed=0.14)).verdict == "HEADLINE_NOT_REPRODUCED__anchor_permutation_three_new_seeds"
    assert exp.score(_receipt(free_perm_mean=0.84)).verdict == "ABSTAIN__free_arm_below_floor"
    assert exp.score(_receipt(degenerate_fraction=0.1167)).verdict == "INVALID__training_unstable"
    assert exp.score(_receipt(degenerate_fraction=0.10)).verdict == "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"
    assert exp.score(_receipt(plumbing_ok=0.0)).verdict == "INVALID__plumbing"
    assert exp.score(_receipt(gap_perm_mean=0.9), smoke=True).verdict == "INVALID__smoke_plumbing_only"
    # the bars themselves, on their edges
    assert exp.score(_receipt(gap_perm_ci_lo=0.05)).verdict.startswith("ROBUST__")
    assert exp.score(_receipt(gap_perm_ci_lo=0.0, gap_perm_mean=0.1)).verdict == "NOT_ROBUST__gap_confined_to_anchor_permutation"
    assert exp.score(_receipt(real2_minus_clamped_perm_mean=0.03)).verdict == "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"
    assert exp.score(_receipt(real2_minus_clamped_perm_mean=0.0301)).verdict == "ROBUST__PARTIAL__diversity_recovers_some"


def test_protocol_walk_agrees_with_the_runner_walk():
    exp = _experiment()
    spec = RB.gates_spec()
    for kw in ({}, {"gap_long": 0.0}, {"gap_perm_ci_lo": 0.01}, {"gap_perm_mean": 0.05, "gap_perm_ci_lo": 0.01},
               {"anchor_gap_newseed": 0.0}, {"plumbing_ok": 0.0}):
        v = exp.score(_receipt(**kw))
        assert RB.walk_outcomes(spec, v.gates) == v.verdict


# -- frozen bytes --------------------------------------------------------------------------------------------

def test_prereg_text_freezes_what_the_runner_asserts():
    text = PREREG.read_text(encoding="utf-8")
    for name, h in RB.FROZEN_RUNNERS.items():
        assert h in text, name
        assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert RB.DATA_SHA256 in text
    assert "`98d685052d7a6ab8b6982491df8ec3eb366a252fd5415aa58f878bbe6510d4e8`" in text
    for p in RB.fallback_perms():
        assert f"`{p}`" in text
    assert RB.FALLBACK_TAG in text
    assert "6 GiB (6,144 MiB)" in text and RB.FREE_GPU_MIN_BYTES == 6 * 2 ** 30
    assert "18.0 minutes" in text and RB.PER_RUN_MIN_REFUSE == 18.0
    assert "8.0 hours" in text and RB.NIGHT_HOURS_REFUSE == 8.0
    assert "default_rng(20261011)" in text and RB.BOOT_SEED == 20261011 and RB.BOOT_N == 2000
    assert "{2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}" in text and RB.T975 == {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def test_the_committed_runners_hash_to_the_frozen_blobs():
    checks = RB.check_frozen_runners(FREQ)
    assert all(v["ok"] for v in checks.values()), checks
    # the LF-normalised hash is the git blob's, whatever the working copy's line endings
    for name in RB.FROZEN_RUNNERS:
        raw = (FREQ / name).read_bytes()
        assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == RB.FROZEN_RUNNERS[name]


def test_frozen_hashes_refuse_a_mutated_runner_copy_and_accept_a_crlf_copy(tmp_path):
    for name in RB.FROZEN_RUNNERS:
        shutil.copy(FREQ / name, tmp_path / name)
    RB.assert_frozen_runners(tmp_path)
    # CRLF copy of both: still the blob (F1)
    for name in RB.FROZEN_RUNNERS:
        lf = (FREQ / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(lf.replace(b"\n", b"\r\n"))
    RB.assert_frozen_runners(tmp_path)
    # one byte changed in the untied runner: refused, by name
    p = tmp_path / "run_pmnist_untied.py"
    raw = p.read_bytes()
    p.write_bytes(raw.replace(b"uniform_(0.9, 0.999)", b"uniform_(0.8, 0.999)", 1))
    assert p.read_bytes() != raw
    with pytest.raises(RB.Refusal, match="run_pmnist_untied.py"):
        RB.assert_frozen_runners(tmp_path)
    # an absent file is refused too
    p.unlink()
    with pytest.raises(RB.Refusal):
        RB.assert_frozen_runners(tmp_path)


def test_data_pin_refuses_a_different_file(tmp_path):
    bad = tmp_path / "_mnist.npz"
    bad.write_bytes(b"\x00" * 11507660)
    with pytest.raises(RB.Refusal, match="not the pinned"):
        RB.assert_data_pin(bad)
    with pytest.raises(RB.Refusal, match="absent"):
        RB.assert_data_pin(tmp_path / "missing.npz")
    assert RB.check_data_pin(bad)["ok"] is False
    if LIVE_DATA.exists():                                  # the 2026-07-23 bytes, when this box has them
        d = RB.assert_data_pin(LIVE_DATA)
        assert d["ok"] and d["bytes"] == 11507660


def test_prereg_bytes_check_refuses_an_edited_copy(tmp_path):
    copy = tmp_path / PREREG.name
    copy.write_bytes(PREREG.read_bytes())
    assert RB.check_prereg_bytes(copy)["ok"] == RB.check_prereg_bytes(PREREG)["ok"]
    copy.write_bytes(PREREG.read_bytes().replace(b'"value": 0.15', b'"value": 0.05', 1))
    with pytest.raises(RB.Refusal, match="sealed blob"):
        RB.assert_prereg_bytes(copy)


def _git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=60)
    if r.returncode != 0:
        pytest.skip(f"git {' '.join(args)} failed here: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout


def test_the_sealed_digest_is_the_committed_blob_and_the_disk_copy_lf_normalised():
    rel = "papers/frequency-resonance/" + PREREG.name
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"HEAD:{rel}"], capture_output=True, timeout=60, env=env)
    if r.returncode != 0:
        pytest.skip("the PREREG is not in HEAD here")
    blob_id = r.stdout.decode().strip()
    blob = _git("cat-file", "blob", blob_id)
    assert hashlib.sha256(blob).hexdigest() == RB.PREREG_BLOB_SHA256, "the committed PREREG is not the sealed bytes"
    assert RB.check_prereg_bytes(PREREG)["ok"], "the PREREG on disk (LF-normalised) is not the sealed bytes"
    seals = SEALS.read_text(encoding="utf-8")
    row = [ln for ln in seals.splitlines() if ln.startswith("| 7 |")]
    assert len(row) == 1 and PREREG.name in row[0] and RB.PREREG_BLOB_SHA256 in row[0]
    assert "`3598e59d`" in row[0]


# -- permutations ----------------------------------------------------------------------------------------

def test_permutation_derivation_is_deterministic_frozen_and_skips_the_anchor():
    b1 = "a" * 64
    p1, p2 = RB.derive_perms(b1), RB.derive_perms(b1)
    assert p1 == p2 and len(p1) == 5 and len(set(p1)) == 5
    assert all(0 <= v < 2 ** 32 and v != 1234 for v in p1)
    assert RB.derive_perms("b" * 64) != p1
    # the rule, by hand: top 32 bits of sha256("<source>:perm:k")
    assert p1[0] == int(hashlib.sha256(f"{b1}:perm:1".encode()).hexdigest()[:8], 16)
    assert RB.fallback_perms() == RB.FALLBACK_PERMS == [3584514041, 27380109, 2994719670, 3317084913, 2856893334]
    assert RB.fallback_perms()[0] == int(hashlib.sha256(b"PREREG_pmnist_robustness_2026_10_11:perm:1").hexdigest()[:8], 16)
    # a value in `exclude` is skipped and k continues, so the list shifts by one
    shifted = RB.derive_perms(RB.FALLBACK_TAG, exclude=(1234, RB.FALLBACK_PERMS[0]))
    assert shifted[:4] == RB.FALLBACK_PERMS[1:] and shifted[4] == 1407215571
    assert RB.is_beacon_hex(b1) and not RB.is_beacon_hex("A" * 64) and not RB.is_beacon_hex("a" * 63)


def test_resolve_perms_labels_the_derivation():
    sel = RB.resolve_perms("c" * 64, None, False)
    assert sel["derivation"] == "beacon" and sel["perms"] == RB.derive_perms("c" * 64)
    sel = RB.resolve_perms(None, None, True)
    assert sel["derivation"] == "sha256-of-tag fallback" and sel["perms"] == RB.FALLBACK_PERMS and sel["beacon"] is None
    assert RB.resolve_perms(None, None, False)["perms"] is None
    with pytest.raises(RB.Refusal):
        RB.resolve_perms("xyz", None, False)
    with pytest.raises(RB.Refusal):
        RB.resolve_perms("c" * 64, None, True)


# -- schedule ---------------------------------------------------------------------------------------------

def test_schedule_is_sixty_five_runs_over_three_nights():
    perms = RB.fallback_perms()
    n1, n2, n3 = RB.schedule(1), RB.schedule(2, perms), RB.schedule(3, perms)
    assert (len(n1), len(n2), len(n3)) == (18, 20, 27)
    allr = n1 + n2 + n3
    names = [RB.leaf_name(r) for r in allr]
    assert len(names) == len(set(names)) == 65
    assert sum(1 for r in allr if r["steps"] == 4000) == 60 and sum(1 for r in allr if r["steps"] == 12000) == 5
    assert {r["perm"] for r in n1} == {1234}
    assert {r["perm"] for r in n2 if r["steps"] == 4000} == set(perms[:2])
    assert {r["perm"] for r in n3} == set(perms[2:])
    long_ = [(r["arm"], r["seed"]) for r in allr if r["steps"] == 12000]
    assert sorted(long_) == [("clamped", 0), ("clamped", 1), ("clamped", 2), ("free", 0), ("free", 1)]
    assert RB.leaf_name(n1[0]) == "p1234_free_s0_4000_h64.json"
    with pytest.raises(RB.Refusal):
        RB.schedule(2, None)
    with pytest.raises(RB.Refusal):
        RB.schedule(4, perms)
    assert len(RB.schedule(1, width=True)) == 12 and {r["h"] for r in RB.schedule(1, width=True)} == {32, 128}
    assert RB.expected_read_back(n1[0]) == {"BATCH": 64, "LR": 3e-3, "WD": 0.01, "T_LEN": 784, "N_BLK": 3, "H": 64,
                                            "D_SSM": 64, "PERM_SEED": 1234, "STEPS": 4000, "EVAL_EVERY": 500}
    assert abs(RB.per_run_minutes({"steps": 12000, "h": 64}, 13.6) - 40.8) < 1e-9


# -- aggregate arithmetic on synthetic leaves -------------------------------------------------------------

def _leaf(r, acc, *, degenerate=False, derivation="sha256-of-tag fallback", imports=None, read_back=None,
          curve=None, mismatch=False):
    n = r["steps"] // RB.EVAL_EVERY
    curve = curve if curve is not None else [round(acc * (i + 1) / n, 4) for i in range(n - 1)] + [acc]
    imports = imports or {k: {"lf_sha256": v, "frozen": v, "ok": True} for k, v in RB.FROZEN_RUNNERS.items()}
    anchor = r["perm"] == RB.ANCHOR_PERM
    return {"schema": RB.LEAF_SCHEMA, "perm": r["perm"], "arm": r["arm"], "seed": r["seed"], "steps": r["steps"],
            "h": r["h"], "perm_derivation": None if anchor else derivation, "beacon": None,
            "read_back": read_back or RB.expected_read_back(r), "final_acc": acc, "curve_every": RB.EVAL_EVERY,
            "curve": curve, "curve_final_mismatch": mismatch, "losses": [0.5], "nonfinite_loss": False,
            "degenerate": degenerate, "within_model_theta0_reliance": 0.8 if r["arm"] == "free" else None,
            "seconds": 800.0, "imports": imports, "data_sha256": RB.DATA_SHA256,
            "prereg_blob_sha256": RB.PREREG_BLOB_SHA256, "preflight_ok": True, "smoke": False}


# per-permutation accuracies: FREE, CLAMPED, REAL2 by seed index; the anchor's seeds 0, 1 are deliberately odd so
# that a test can show they enter no gated metric
ACC = {
    "free": {1234: [0.50, 0.50, 0.92, 0.93, 0.91], "p0": [0.92, 0.93, 0.91], "p1": [0.90, 0.92, 0.91],
             "p2": [0.93, 0.92, 0.94], "p3": [0.91, 0.90, 0.92], "p4": [0.92, 0.92, 0.92]},
    "clamped": {1234: [0.90, 0.90, 0.60, 0.66, 0.57], "p0": [0.60, 0.66, 0.57], "p1": [0.62, 0.58, 0.64],
                "p2": [0.55, 0.61, 0.59], "p3": [0.63, 0.60, 0.66], "p4": [0.61, 0.59, 0.60]},
    "real2": {1234: [0.90, 0.90, 0.61, 0.64, 0.60], "p0": [0.61, 0.64, 0.60], "p1": [0.63, 0.60, 0.62],
              "p2": [0.58, 0.60, 0.61], "p3": [0.62, 0.63, 0.64], "p4": [0.60, 0.62, 0.61]},
}
LONG_ACC = {("free", 0): 0.94, ("free", 1): 0.93, ("clamped", 0): 0.68, ("clamped", 1): 0.70, ("clamped", 2): 0.66}


def _synthetic_leaves(perms, **overrides):
    leaves = []
    for r in RB.full_schedule(perms):
        key = 1234 if r["perm"] == 1234 else f"p{perms.index(r['perm'])}"
        if r["steps"] == 12000:
            acc = LONG_ACC[(r["arm"], r["seed"])]
        else:
            acc = ACC[r["arm"]][key][r["seed"]]
        leaves.append(_leaf(r, acc))
    return leaves


def _chain_for(leaves):
    entries, prev = [], "0" * 64
    for i, l in enumerate(leaves):
        sha = hashlib.sha256(json.dumps(l, indent=1).encode()).hexdigest()
        e = RB.chain_entry(prev, RB.leaf_name(l), sha, i + 1, "t")
        entries.append(e)
        prev = e["entry_id"]
    current = {RB.leaf_name(l): hashlib.sha256(json.dumps(l, indent=1).encode()).hexdigest() for l in leaves}
    return RB.verify_chain(entries, current), entries, current


DISK_OK = {"runners": True, "data": True, "prereg": True}


def _agg(leaves, perms, derivation="sha256-of-tag fallback", beacon=None, chain=None, disk=DISK_OK):
    chain = chain if chain is not None else _chain_for(leaves)[0]
    return RB.aggregate_leaves(leaves, [1234] + perms, derivation, beacon, chain, disk)


def _mean(xs):
    return sum(xs) / len(xs)


def test_aggregate_arithmetic_on_synthetic_leaves_matches_hand_computation():
    perms = RB.fallback_perms()
    leaves = _synthetic_leaves(perms)
    assert len(leaves) == 65
    out = _agg(leaves, perms)
    res = out["result"]
    assert out["plumbing"] == {k: True for k in out["plumbing"]}, out["plumbing_detail"]
    assert res["plumbing_ok"] == 1.0 and res["degenerate_fraction"] == 0.0 and res["nonfinite_metrics"] == []
    gaps = [_mean(ACC["free"][f"p{i}"]) - _mean(ACC["clamped"][f"p{i}"]) for i in range(5)]
    gm = _mean(gaps)
    sd = (sum((g - gm) ** 2 for g in gaps) / 4) ** 0.5
    assert res["gap_perm_mean"] == round(gm, 4)
    assert res["gap_perm_ci_lo"] == round(gm - 2.776 * sd / 5 ** 0.5, 4)
    assert out["reported"]["gap_perm_ci_hi"] == round(gm + 2.776 * sd / 5 ** 0.5, 4)
    # the anchor gate reads seeds 2, 3, 4 only (seeds 0 and 1 carry 0.50 / 0.90 and would wreck it)
    assert res["anchor_gap_newseed"] == round(_mean([0.92, 0.93, 0.91]) - _mean([0.60, 0.66, 0.57]), 4)
    assert out["anchor"]["free_01"] == 0.5 and out["anchor"]["clamped_01"] == 0.9
    assert out["anchor"]["anchor_max_abs_dev"] == round(max(abs(0.5 - 0.9195), abs(0.9 - 0.6073)), 4)
    # the floor: six permutations at three seeds, the anchor's new seeds among them
    floors = [_mean([0.92, 0.93, 0.91])] + [_mean(ACC["free"][f"p{i}"]) for i in range(5)]
    assert res["free_perm_mean"] == round(_mean(floors), 4)
    assert res["gap_long"] == round(_mean([0.94, 0.93]) - _mean([0.68, 0.70, 0.66]), 4)
    fr2 = [_mean(ACC["free"][f"p{i}"]) - _mean(ACC["real2"][f"p{i}"]) for i in range(5)]
    r2c = [_mean(ACC["real2"][f"p{i}"]) - _mean(ACC["clamped"][f"p{i}"]) for i in range(5)]
    assert res["free_minus_real2_perm_mean"] == round(_mean(fr2), 4)
    assert res["real2_minus_clamped_perm_mean"] == round(_mean(r2c), 4)
    assert abs(res["free_minus_real2_perm_mean"] + res["real2_minus_clamped_perm_mean"] - res["gap_perm_mean"]) < 2e-4
    # recovery, its interval and class
    recs = [b / a for a, b in zip(gaps, r2c)]
    assert out["reported"]["recovery_perm_mean"] == round(_mean(recs), 4)
    assert out["reported"]["recovery_class"] == "rotation"       # every recovery here is under 0.1
    assert len(out["reported"]["recovery_defined_on"]) == 5
    # LONG: between-schedule gain and the within-curve mid-gain from the synthetic linear curves
    # between schedules, seed-paired on seeds 0-2 (the odd 0.90s of the re-run seeds are in it; it is reported, not gated)
    assert out["long"]["clamped_long_gain"] == round(_mean([0.68, 0.70, 0.66]) - _mean([0.90, 0.90, 0.60]), 4)
    assert out["long"]["free_long_gain"] == round(_mean([0.94, 0.93]) - _mean([0.50, 0.50]), 4)
    assert out["long"]["clamped_long_midgain"] == \
        round(_mean([round(a * 16 / 24, 4) - round(a * 8 / 24, 4) for a in (0.68, 0.70, 0.66)]), 4)
    # pooled interval and bootstrap are present, deterministic, and bracket the mean
    assert out["reported"]["pooled_n"] == [15, 15]
    lo, hi = out["reported"]["pooled_ci"]
    assert lo < out["reported"]["pooled_gap"] < hi
    b = out["bootstrap"]
    assert b["ci_lo"] <= res["gap_perm_mean"] <= b["ci_hi"] and 0.0 <= b["between_perm_var_share"] <= 1.0
    assert _agg(leaves, perms)["bootstrap"] == b
    # the walk over this receipt lands on the row the numbers say
    spec = RB.gates_spec()
    fired = {}
    for name, g in spec["gates"].items():
        v = res[g["metric"][len("result."):]]
        fired[name] = {">=": v >= g["value"], "<=": v <= g["value"], ">": v > g["value"], "==": v == g["value"]}[g["op"]]
    assert RB.walk_outcomes(spec, fired) == "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"
    assert set(out["unexcluded"]) == set(RB.GATED) - {"plumbing_ok"}
    assert out["config"]["held_out_perms"] == perms and out["config"]["perm_derivation"] == "sha256-of-tag fallback"


def test_degeneracy_is_counted_by_arm():
    perms = RB.fallback_perms()
    leaves = _synthetic_leaves(perms)
    base = _agg(leaves, perms)["result"]

    def find(perm, arm, seed, steps=4000):
        return next(l for l in leaves if (l["perm"], l["arm"], l["seed"], l["steps"]) == (perm, arm, seed, steps))
    # a collapsed CLAMPED run on P1 is excluded: the gap on P1 rises, the floor is untouched
    c = find(perms[0], "clamped", 1)
    c["final_acc"], c["degenerate"] = 0.1135, True
    out = _agg(leaves, perms)
    assert out["result"]["degenerate_fraction"] == round(1 / 60, 4)
    assert out["per_permutation"][str(perms[0])]["n"]["clamped"] == 2
    assert out["per_permutation"][str(perms[0])]["gap"] == round(_mean(ACC["free"]["p0"]) - _mean([0.60, 0.57]), 4)
    assert out["result"]["free_perm_mean"] == base["free_perm_mean"]
    assert out["degeneracy"]["by_arm"]["clamped"] == [RB.leaf_name(c)] and out["degeneracy"]["free_counted_as_zero_in_floor"] == []
    assert out["unexcluded"]["gap_perm_mean"] > out["result"]["gap_perm_mean"]   # excluding decay collapses is conservative
    # a collapsed FREE run on P2 counts as 0 in the floor (one is tolerated) and is excluded from the gap
    f = find(perms[1], "free", 2)
    f["final_acc"], f["degenerate"] = 0.098, True
    out = _agg(leaves, perms)
    floors = [_mean([0.92, 0.93, 0.91])] + [_mean(ACC["free"][f"p{i}"]) for i in range(5)]
    floors[2] = _mean([0.90, 0.92, 0.0])
    assert out["result"]["free_perm_mean"] == round(_mean(floors), 4)
    assert out["result"]["free_perm_mean"] >= 0.85
    assert out["per_permutation"][str(perms[1])]["n"]["free"] == 2
    assert out["degeneracy"]["free_counted_as_zero_in_floor"] == [RB.leaf_name(f)]
    # a second FREE collapse fails the floor
    f2 = find(perms[3], "free", 0)
    f2["final_acc"], f2["degenerate"] = 0.1, True
    out = _agg(leaves, perms)
    assert out["result"]["free_perm_mean"] < 0.85 and out["result"]["degenerate_fraction"] == round(3 / 60, 4)
    # seven collapses cross the 0.10 cap
    for s, perm in ((0, perms[4]), (1, perms[4]), (2, perms[4]), (1, perms[2])):
        x = find(perm, "real2", s)
        x["final_acc"], x["degenerate"] = 0.1, True
    out = _agg(leaves, perms)
    assert out["result"]["degenerate_fraction"] == round(7 / 60, 4) and out["result"]["degenerate_fraction"] > 0.10


def test_plumbing_reads_zero_on_a_missing_leaf_a_foreign_import_a_read_back_mismatch_and_a_broken_chain():
    perms = RB.fallback_perms()
    leaves = _synthetic_leaves(perms)
    chain_ok, entries, current = _chain_for(leaves)
    assert chain_ok["ok"] and chain_ok["rewrites"] == []
    # missing leaf
    out = _agg(leaves[:-1], perms, chain=chain_ok)
    assert out["result"]["plumbing_ok"] == 0.0 and out["plumbing"]["leaves_complete"] is False
    assert out["plumbing_detail"]["missing"] == [RB.leaf_name(leaves[-1])]
    # foreign import hash in one leaf
    bad = json.loads(json.dumps(leaves))
    bad[7]["imports"]["run_pmnist_untied.py"]["lf_sha256"] = "58396a4fe430a4d0ff6192afe5a026eecf61fea89178d6a56dbd92e08ef06f94"
    out = _agg(bad, perms, chain=chain_ok)
    assert out["result"]["plumbing_ok"] == 0.0 and out["plumbing_detail"]["imports_bad"] == [RB.leaf_name(bad[7])]
    # read-back mismatch (a run that trained with a different EVAL_EVERY)
    bad = json.loads(json.dumps(leaves))
    bad[3]["read_back"]["EVAL_EVERY"] = 1000
    out = _agg(bad, perms, chain=chain_ok)
    assert out["plumbing"]["read_back_ok"] is False and out["result"]["plumbing_ok"] == 0.0
    # final accuracy not equal to the last curve point
    bad = json.loads(json.dumps(leaves))
    bad[5]["curve_final_mismatch"] = True
    assert _agg(bad, perms, chain=chain_ok)["plumbing"]["no_final_mismatch"] is False
    # a leaf whose derivation label differs from the others
    bad = json.loads(json.dumps(leaves))
    bad[40]["perm_derivation"] = "beacon"
    out = _agg(bad, perms, chain=chain_ok)
    assert out["plumbing"]["derivation_consistent"] is False and out["result"]["plumbing_ok"] == 0.0
    # permutations that are not the derivation's
    out = RB.aggregate_leaves(leaves, [1234] + perms[:4] + [99], "sha256-of-tag fallback", None, chain_ok, DISK_OK)
    assert out["plumbing"]["perms_match_derivation"] is False
    # the disk re-checks
    out = _agg(leaves, perms, chain=chain_ok, disk={"runners": False, "data": True, "prereg": True})
    assert out["plumbing"]["disk_runners_ok"] is False and out["result"]["plumbing_ok"] == 0.0
    # the chain: a tampered entry, a stale leaf, a re-run leaf
    tampered = json.loads(json.dumps(entries))
    tampered[10]["sha256"] = "0" * 64
    v = RB.verify_chain(tampered, current)
    assert not v["ok"] and v["broken_at"] == 11
    stale = dict(current)
    stale[entries[2]["leaf"]] = "f" * 64
    v = RB.verify_chain(entries, stale)
    assert not v["ok"] and v["stale"] == [entries[2]["leaf"]]
    rerun = entries + [RB.chain_entry(entries[-1]["entry_id"], entries[2]["leaf"], "e" * 64, len(entries) + 1, "t")]
    cur2 = dict(current)
    cur2[entries[2]["leaf"]] = "e" * 64
    v = RB.verify_chain(rerun, cur2)
    assert v["ok"] and v["rewrites"] == [entries[2]["leaf"]]
    out = _agg(leaves, perms, chain=RB.verify_chain(tampered, current))
    assert out["plumbing"]["chain_ok"] is False and out["result"]["plumbing_ok"] == 0.0
    # every check true again when nothing is wrong
    assert _agg(leaves, perms)["result"]["plumbing_ok"] == 1.0


def test_a_non_finite_metric_reads_invalid_plumbing_not_an_exception():
    perms = RB.fallback_perms()
    leaves = _synthetic_leaves(perms)
    for l in leaves:
        if l["perm"] == perms[2] and l["arm"] == "clamped":
            l["final_acc"], l["degenerate"] = 0.1, True      # no surviving CLAMPED run on P3: gap_p is NaN
    out = _agg(leaves, perms)
    res = out["result"]
    assert set(res["nonfinite_metrics"]) == {"gap_perm_mean", "gap_perm_ci_lo", "real2_minus_clamped_perm_mean"}
    assert res["gap_perm_mean"] == 0.0 and res["plumbing_ok"] == 0.0 and out["plumbing"]["metrics_finite"] is False
    for m in RB.GATED:
        assert isinstance(res[m], float) and res[m] == res[m]
    exp = _experiment()
    assert exp.score(out).verdict == "INVALID__plumbing"


def test_t_ci_uses_the_frozen_quantiles():
    ci = RB.t_ci([0.30, 0.25, 0.35, 0.28, 0.32])
    assert ci["n"] == 5 and ci["t"] == 2.776
    assert abs(ci["lo"] - (0.30 - 2.776 * ci["sd"] / 5 ** 0.5)) < 1e-12
    assert RB.t_ci([0.1, 0.2])["t"] == 12.706 and RB.t_ci([0.1])["lo"] != RB.t_ci([0.1])["lo"]   # NaN for n < 2
    assert RB.t_ci([0.1] * 6)["t"] is None


# -- the three arms on CPU (construction only) -------------------------------------------------------------

def test_three_arms_share_rng_state_and_non_ssm_tensors_at_seeds_0_to_4():
    pytest.importorskip("torch")
    R, U = RB._load_runners()
    out = RB.assert_rng_sharing(R, U, seeds=range(5))
    assert set(out) == {"0", "1", "2", "3", "4"} and all(v["shared_tensors"] == 34 for v in out.values())
    assert (R.nparams(True), R.nparams(False), U.nparams(U.RealClassifier())) == (101002, 100810, 101002)
    for k, v in RB.COMMITTED.items():
        assert getattr(R, k) == v


# -- the words the lab never prints ------------------------------------------------------------------------

def test_the_lab_never_prints_these_words():
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    if not src.exists():
        pytest.skip("the lab's BANNED tuple is not in this checkout")
    m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "BANNED tuple not found"
    words = re.findall(r'"([^"]+)"', m.group(1))
    assert len(words) >= 8
    banned = re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)
    for p in (PREREG, RUNNER, TEST_FILE):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p.name} has a BOM"
        hits = [m.group(0) for m in banned.finditer(raw.decode("utf-8"))]
        assert not hits, f"{p.name}: {hits}"


def test_cli_derive_perms_prints_the_fallback_without_touching_torch():
    r = subprocess.run([sys.executable, str(RUNNER), "--derive-perms"], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert [int(ln.split()[1]) for ln in lines[:5]] == RB.FALLBACK_PERMS
    assert "sha256-of-tag fallback" in lines[5]
    r = subprocess.run([sys.executable, str(RUNNER), "--derive-perms", "--beacon", "d" * 64], capture_output=True,
                       text=True, timeout=120, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0 and [int(ln.split()[1]) for ln in r.stdout.strip().splitlines()[:5]] == RB.derive_perms("d" * 64)
    r = subprocess.run([sys.executable, str(RUNNER), "--night", "2"], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 2 and "refused:" in r.stderr
