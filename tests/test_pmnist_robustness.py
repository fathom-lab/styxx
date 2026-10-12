# -*- coding: utf-8 -*-
"""PREREG_pmnist_robustness_2026_10_11 and run_pmnist_robustness.py, on CPU, with no data and no GPU.

What is pinned here: the gates block parses through the lab's own scanner and every one of the 1,024 boolean
assignments reaches a row; `Experiment` scores a receipt of the PREREG's shape to each of the thirteen verdicts and
refuses nothing it should accept; the frozen hashes refuse a mutated copy of either imported runner (and accept a CRLF
copy, which is the trap SEALS names); the permutation derivation is deterministic, skips 1234 and repeats, and the
fallback list equals the one the PREREG prints; the aggregate arithmetic on synthetic leaves gives the hand-computed
numbers, counts a collapsed FREE run as 0 in the floor, excludes a collapsed decay run from the gap, maps a missing
leaf, a foreign import hash, a read-back mismatch and a non-finite metric to plumbing_ok = 0; the receipt's seeds 0
and 1 enter degenerate_fraction by value and G_P_plumbing as leaves, and no other gated metric; the _mnist.npz pin
refuses a different file; the PREREG's section 15 is appended to the row-7 bytes and moves no gate; the sealed digest
in SEALS row 8 is the committed blob's, row 7 stands unedited and only row 8's memo is listed; the leaf log is written
before its leaf, and a torn log is refused by name, by a night before any run trains; a non-finite loss is written
as null and every file the runner writes is strict JSON; the GPU guard runs before the red team touches the device;
every refusal that needs no torch happens with torch unimportable; the leaves, the log and the receipt are
committable, and the leaves and the log byte-pinned; and none of the three files prints the words the lab never
prints. Nothing here loads a checkpoint,
reads the data or touches the card, and nothing here needs a GPU, torch, _mnist.npz or a path outside the checkout:
a test that needs one of them skips when it is absent.
"""
from __future__ import annotations

import argparse
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
REL_PREREG = "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md"
# SEALS row 7: the PREREG before its section 15 (frozen at 3598e59d), superseded by row 8 and never to be anchored
ROW7_SHA256 = "2068e6733c1d5d2c4305cddc89e6e5f2ddd3671406e8bcf56332251d05db648e"
ROW7_BYTES = 56960
# the gates block's sha256 as styxx.protocol computes it, before and after section 15
GATES_SHA256 = "578924f8e776573cc82db8fbbd8bb9cf84106b81171a5ba6d5783e422c96e854"


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
    assert RB.check_data_pin(bad)["path"] == bad.name        # a path outside the checkout is printed by name only
    if RB.DATA_FILE.exists():                               # the 2026-07-23 bytes, when copied beside the runner
        d = RB.assert_data_pin(RB.DATA_FILE)
        assert d["ok"] and d["bytes"] == 11507660
        assert d["path"] == "papers/frequency-resonance/_mnist.npz"


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


def _seal_rows() -> dict:
    rows = {}
    for ln in SEALS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (\d+b?) \| ", ln)
        if m:
            assert m.group(1) not in rows, f"SEALS row {m.group(1)} appears twice"
            rows[m.group(1)] = ln
    return rows


def test_the_sealed_digest_is_the_committed_blob_and_the_disk_copy_lf_normalised():
    assert RB.check_prereg_bytes(PREREG)["ok"], "the PREREG on disk (LF-normalised) is not the sealed bytes"
    row = _seal_rows()["8"]
    assert PREREG.name in row and RB.PREREG_BLOB_SHA256 in row
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"HEAD:{REL_PREREG}"], capture_output=True, timeout=60,
                       env=env)
    if r.returncode != 0:
        pytest.skip("the PREREG is not in HEAD here")
    blob = _git("cat-file", "blob", r.stdout.decode().strip())
    assert hashlib.sha256(blob).hexdigest() == RB.PREREG_BLOB_SHA256, "the committed PREREG is not the sealed bytes"
    # row 8's "frozen at" commit holds exactly these bytes (skipped on a clone too shallow to hold that commit)
    frozen_at = [c.strip().strip("`") for c in row.split("|")][4]
    assert re.fullmatch(r"[0-9a-f]{8}", frozen_at), row
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-p", f"{frozen_at}:{REL_PREREG}"], capture_output=True,
                       timeout=60, env=env)
    if r.returncode != 0:
        pytest.skip(f"commit {frozen_at} is not in this clone")
    assert hashlib.sha256(r.stdout).hexdigest() == RB.PREREG_BLOB_SHA256


def test_section_15_is_appended_to_the_row_7_bytes_and_moves_no_gate():
    lf = PREREG.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(lf[:ROW7_BYTES]).hexdigest() == ROW7_SHA256, "the text above section 15 is not row 7's blob"
    tail = lf[ROW7_BYTES:].decode("utf-8")
    assert tail.startswith("\n---\n\n## 15. Amendment, before any data (2026-10-12)\n")
    assert "```" not in tail and "<!--" not in tail                 # no second fence, nothing hidden
    assert "56,960" in tail and ROW7_SHA256 in tail and GATES_SHA256 in tail
    from styxx.protocol import _select_gates_block
    block = _select_gates_block(PREREG.read_text(encoding="utf-8"))
    assert hashlib.sha256(block.encode("utf-8")).hexdigest() == GATES_SHA256
    assert _select_gates_block(lf[:ROW7_BYTES].decode("utf-8")) == block
    assert ROW7_SHA256 != RB.PREREG_BLOB_SHA256 and hashlib.sha256(lf).hexdigest() == RB.PREREG_BLOB_SHA256


def test_seals_row_8_supersedes_row_7_which_stands_unedited_and_only_row_8_is_a_memo():
    from styxx.clock import memo
    rows = _seal_rows()
    r7, r8 = rows["7"], rows["8"]
    assert list(rows)[-2:] == ["7", "8"]
    assert ROW7_SHA256 in r7 and "`3598e59d`" in r7 and REL_PREREG in r7 and RB.PREREG_BLOB_SHA256 not in r7
    assert RB.PREREG_BLOB_SHA256 in r8 and REL_PREREG in r8 and "`sealed-prereg`" in r8 and ROW7_SHA256 not in r8
    assert "supersedes row 7" in r8 and "not to be anchored" in r8
    memos = [ln.strip() for ln in SEALS.read_text(encoding="utf-8").splitlines() if ln.startswith("    styxx ")]
    assert memo("sealed-prereg", RB.PREREG_BLOB_SHA256) in memos
    assert memo("sealed-prereg", ROW7_SHA256) not in memos
    assert memos[-1] == memo("sealed-prereg", RB.PREREG_BLOB_SHA256)
    raw = SEALS.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")


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
# that a test can show their values enter no gated metric but degenerate_fraction (PREREG section 15.1)
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
    assert "nonfinite_fields" not in _strict_loads(RB.strict_json(out))     # every number in it went through _r
    exp = _experiment()
    assert exp.score(out).verdict == "INVALID__plumbing"


def test_seeds_0_and_1_enter_degenerate_fraction_by_value_and_plumbing_as_leaves():
    # PREREG section 15.1, read as the RESULT must repeat it: the six re-run trainings (FREE, CLAMPED, REAL2 at seeds
    # 0 and 1 on permutation 1234) move no gated metric by value but degenerate_fraction, and like every leaf they
    # count toward G_P_plumbing; degenerate_fraction counts the sixty 4000-step runs, not the 12,000-step ones
    perms = RB.fallback_perms()
    leaves = _synthetic_leaves(perms)
    base = _agg(leaves, perms)["result"]
    rerun = [RB.leaf_name(l) for l in leaves if l["perm"] == 1234 and l["seed"] in RB.RERUN_SEEDS and l["steps"] == 4000]
    assert len(rerun) == 6
    moved = json.loads(json.dumps(leaves))
    for l in moved:
        if RB.leaf_name(l) in rerun:
            l["final_acc"] = 0.3
    assert _agg(moved, perms)["result"] == base
    collapsed = next(l for l in moved if RB.leaf_name(l) == "p1234_real2_s0_4000_h64.json")
    collapsed["final_acc"], collapsed["degenerate"] = 0.1, True
    res = _agg(moved, perms)["result"]
    assert res["degenerate_fraction"] == round(1 / 60, 4)
    assert {k: v for k, v in res.items() if k != "degenerate_fraction"} == \
        {k: v for k, v in base.items() if k != "degenerate_fraction"}
    short = [l for l in leaves if RB.leaf_name(l) != "p1234_clamped_s1_4000_h64.json"]
    out = _agg(short, perms)
    assert out["result"]["plumbing_ok"] == 0.0 and out["plumbing_detail"]["missing"] == ["p1234_clamped_s1_4000_h64.json"]
    long_ = json.loads(json.dumps(leaves))
    for l in long_:
        if RB.leaf_name(l) == "p1234_clamped_s0_12000_h64.json":
            l["final_acc"], l["degenerate"] = 0.1, True
    out = _agg(long_, perms)
    assert out["result"]["degenerate_fraction"] == 0.0 and out["degeneracy"]["long_count"] == 1


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


# the runner as a script with torch made unimportable (sys.modules["torch"] = None): whatever this box has installed,
# a step that reaches for torch fails, so a refusal seen here happened before torch, and no card is ever touched
_NO_TORCH = ("import runpy, sys; sys.modules['torch'] = None; sys.argv = sys.argv[1:]; "
             "runpy.run_path(sys.argv[0], run_name='__main__')")


def _cli_without_torch(*argv):
    return subprocess.run([sys.executable, "-c", _NO_TORCH, str(RUNNER), *argv], capture_output=True, text=True,
                          timeout=120, cwd=str(ROOT),
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""))


def test_cli_derive_perms_prints_the_fallback_without_touching_torch():
    r = _cli_without_torch("--derive-perms")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert [int(ln.split()[1]) for ln in lines[:5]] == RB.FALLBACK_PERMS
    assert "sha256-of-tag fallback" in lines[5]
    r = _cli_without_torch("--derive-perms", "--beacon", "d" * 64)
    assert r.returncode == 0, r.stderr
    assert [int(ln.split()[1]) for ln in r.stdout.strip().splitlines()[:5]] == RB.derive_perms("d" * 64)


def test_cli_refuses_by_name_before_torch_is_imported():
    # a night on held-out permutations with no derivation: refused before anything else
    r = _cli_without_torch("--night", "2")
    assert r.returncode == 2 and "refused: night 2 runs the held-out permutations" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr
    # night 1 and the smoke: refused at a check that needs no torch (the data pin or the smoke receipt on a box
    # without them), or, on a box that has both, at the torch import itself -- by name, never a traceback
    for argv in (("--night", "1"), ("--smoke",)):
        r = _cli_without_torch(*argv)
        assert r.returncode == 2 and "refused:" in r.stderr, (argv, r.stderr)
        assert "Traceback" not in r.stderr
        if not RB.DATA_FILE.exists():
            assert "_mnist.npz" in r.stderr and "absent" in r.stderr, r.stderr


# -- section 15.4: the GPU guard before the red team touches the device --------------------------------------

class _FakeR:
    BATCH, LR, WD, T_LEN, N_BLK, H, D_SSM = 64, 3e-3, 0.01, 784, 3, 64, 64
    SMOKE = False

    def __init__(self, calls):
        self.calls = calls

    def redteam(self):
        self.calls.append("R.redteam")

    def nparams(self, free):
        return 101002 if free else 100810


class _FakeU:
    def __init__(self, calls):
        self.calls = calls

    def redteam(self):
        self.calls.append("U.redteam")

    def RealClassifier(self):
        return "real2"

    def nparams(self, m):
        return 101002


def test_the_gpu_guard_runs_before_the_red_team_touches_the_device(monkeypatch):
    calls = []

    def guard_refuses():
        calls.append("guard")
        raise RB.Refusal("CUDA is not available")
    monkeypatch.setattr(RB, "require_gpu", guard_refuses)
    monkeypatch.setattr(RB, "assert_rng_sharing", lambda R, U: calls.append("rng") or {})
    with pytest.raises(RB.Refusal, match="CUDA"):
        RB.preflight_torch(_FakeR(calls), _FakeU(calls), {})
    assert calls == ["guard"]                                # nothing reached the red team
    calls.clear()
    monkeypatch.setattr(RB, "require_gpu", lambda: calls.append("guard") or {"cuda": True})
    pre = RB.preflight_torch(_FakeR(calls), _FakeU(calls), {})
    assert calls == ["guard", "R.redteam", "U.redteam", "rng"]
    assert pre["preflight_ok"] is True and pre["params"] == {"free": 101002, "clamped": 100810, "real2": 101002}
    # a committed constant that moved is refused before the guard
    calls.clear()
    bad = _FakeR(calls)
    bad.LR = 1e-3
    with pytest.raises(RB.Refusal, match="R.LR"):
        RB.preflight_torch(bad, _FakeU(calls), {})
    assert calls == []
    # a failing committed red team is a refusal by name, not an AssertionError
    calls.clear()
    failing = _FakeR(calls)

    def boom():
        raise AssertionError("scan!=seq (1.0e+00)")
    failing.redteam = boom
    with pytest.raises(RB.Refusal, match="scan!=seq"):
        RB.preflight_torch(failing, _FakeU(calls), {})


# -- section 15.3: the leaf log before the leaf, and a torn log refused -------------------------------------

def _tmp_runs(monkeypatch, tmp_path):
    runs = tmp_path / "pmnist_robustness_runs"
    monkeypatch.setattr(RB, "RUNS_DIR", runs)
    monkeypatch.setattr(RB, "CHAIN_FILE", runs / "leaf_chain.jsonl")
    return runs


def test_the_log_line_is_written_before_the_leaf_and_a_crash_between_them_reads_as_a_rewrite(monkeypatch, tmp_path):
    runs = _tmp_runs(monkeypatch, tmp_path)
    perms = RB.fallback_perms()
    sched = RB.full_schedule(perms)
    r0, r1 = sched[0], sched[1]
    p = RB.write_leaf(r0, _leaf(r0, 0.92))
    assert p == runs / RB.leaf_name(r0) and not list(runs.glob("*.tmp"))
    entries = RB._read_chain()
    assert len(entries) == 1 and entries[0]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert b"\r" not in p.read_bytes() and b"\r" not in RB.CHAIN_FILE.read_bytes()
    # a crash after the log line and before the rename: the log names a leaf that is not there
    class PowerCut(Exception):
        pass
    real_replace = os.replace

    def crash(src, dst):
        raise PowerCut(f"{src} -> {dst}")
    monkeypatch.setattr(RB.os, "replace", crash)
    with pytest.raises(PowerCut):
        RB.write_leaf(r1, _leaf(r1, 0.90))
    monkeypatch.setattr(RB.os, "replace", real_replace)
    assert not RB.leaf_path(r1).exists() and len(RB._read_chain()) == 2
    assert [q.name for q in runs.glob("p*.json")] == [RB.leaf_name(r0)]   # the .tmp is not a leaf to the aggregate
    # the night re-runs it: a second line for the name, visible as a rewrite, and the chain verifies
    RB.write_leaf(r1, _leaf(r1, 0.91))
    current = {q.name: RB.lf_sha256(q) for q in runs.glob("p*.json")}
    v = RB.verify_chain(RB._read_chain(), current)
    assert v["ok"] and v["rewrites"] == [RB.leaf_name(r1)] and v["entries"] == 3
    # a CRLF copy of a leaf (a checkout without the -text pin) still matches the log, LF-normalised
    q = RB.leaf_path(r0)
    q.write_bytes(q.read_bytes().replace(b"\n", b"\r\n"))
    current = {q.name: RB.lf_sha256(q) for q in runs.glob("p*.json")}
    assert RB.verify_chain(RB._read_chain(), current)["ok"]


def test_a_torn_leaf_log_is_refused_by_name(monkeypatch, tmp_path):
    runs = _tmp_runs(monkeypatch, tmp_path)
    r0 = RB.schedule(1)[0]
    RB.write_leaf(r0, _leaf(r0, 0.92))
    good = RB.CHAIN_FILE.read_bytes()
    RB.CHAIN_FILE.write_bytes(good + good[:40])                   # an append cut short
    with pytest.raises(RB.Refusal, match="torn"):
        RB._read_chain()
    with pytest.raises(RB.Refusal, match="torn"):
        RB.write_leaf(RB.schedule(1)[1], _leaf(RB.schedule(1)[1], 0.9))
    RB.CHAIN_FILE.write_bytes(good + b"{not json}\n")
    with pytest.raises(RB.Refusal, match="line 2 does not parse"):
        RB._read_chain()
    assert [q.name for q in runs.glob("p*.json")] == [RB.leaf_name(r0)]


def test_a_night_refuses_a_torn_leaf_log_before_any_run_trains(monkeypatch, tmp_path):
    _tmp_runs(monkeypatch, tmp_path)
    smoke = tmp_path / "pmnist_robustness_smoke.json"
    smoke.write_text(json.dumps({"projected_min_per_4000": 13.6}), encoding="utf-8")
    monkeypatch.setattr(RB, "SMOKE_JSON", smoke)
    monkeypatch.setattr(RB, "preflight_pure", lambda: {})           # items 1-3 are tested on their own above

    class ReachedTheRunners(Exception):
        pass

    def load():
        raise ReachedTheRunners()
    monkeypatch.setattr(RB, "_load_runners", load)
    args = argparse.Namespace(night=1, beacon=None, anchors=None, fallback_perms=False, width=False,
                              accept_schedule=False)
    with pytest.raises(ReachedTheRunners):                       # no log yet: the night goes on to the runners
        RB.run_night(args)
    r0 = RB.schedule(1)[0]
    RB.write_leaf(r0, _leaf(r0, 0.92))
    with pytest.raises(ReachedTheRunners):                       # a sound log: the same
        RB.run_night(args)
    RB.CHAIN_FILE.write_bytes(RB.CHAIN_FILE.read_bytes() + b'{"n": 2, "leaf"')    # an append cut short
    with pytest.raises(RB.Refusal, match="torn"):                # refused before the runners load and a run trains
        RB.run_night(args)


# -- strict JSON: a bare NaN is not JSON, and section 15.3 commits the leaves and the receipt ----------------

def _strict_loads(text):
    def refuse(constant):
        raise ValueError(f"{constant} is not JSON")
    return json.loads(text, parse_constant=refuse)


def test_a_non_finite_loss_is_written_as_null_and_every_file_the_runner_writes_is_strict_json(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _strict_loads(json.dumps({"loss": float("nan")}))       # what the runner wrote before: NaN, refused here
    printed = ("    step  500 loss 2.301 test 0.1135\n    step 1000 loss nan test 0.0980\n"
               "    step 1500 loss inf test 0.0980\n    step 2000 loss 1.2e-1 test 0.5000\n")
    losses, nonfinite = RB.parse_losses(printed)
    assert losses == [2.301, None, None, 0.12] and nonfinite is True
    assert RB.parse_losses("    step  500 loss 2.301 test 0.9000\n") == ([2.301], False)
    _tmp_runs(monkeypatch, tmp_path)
    r0, r1 = RB.schedule(1)[:2]
    leaf = _leaf(r0, 0.098, degenerate=True)
    leaf["losses"], leaf["nonfinite_loss"] = losses, nonfinite
    got = _strict_loads(RB.write_leaf(r0, leaf).read_text(encoding="utf-8"))
    assert got["losses"] == [2.301, None, None, 0.12] and got["nonfinite_loss"] is True and got["degenerate"] is True
    assert "nonfinite_fields" not in got
    # a clean leaf is written as the plain indent=1 dump, the bytes the log names
    assert RB.write_leaf(r1, _leaf(r1, 0.9)).read_bytes() == json.dumps(_leaf(r1, 0.9), indent=1).encode("utf-8")
    # any other non-finite number is written as null and named, never as a bare NaN, and nothing stops over it
    bad = dict(_leaf(r0, 0.9), within_model_theta0_reliance=float("nan"), curve=[0.5, float("-inf")])
    got = _strict_loads(RB.strict_json(bad))
    assert got["within_model_theta0_reliance"] is None and got["curve"] == [0.5, None]
    assert sorted(got["nonfinite_fields"]) == ["/curve/1", "/within_model_theta0_reliance"]
    assert bad["within_model_theta0_reliance"] != bad["within_model_theta0_reliance"]     # the caller's copy is untouched


def test_aggregate_writes_a_strict_json_receipt_whose_leaf_chain_verifies(monkeypatch, tmp_path):
    _tmp_runs(monkeypatch, tmp_path)
    result = tmp_path / "pmnist_robustness_result.json"
    monkeypatch.setattr(RB, "RESULT_JSON", result)
    perms = RB.fallback_perms()
    for leaf in _synthetic_leaves(perms):
        RB.write_leaf(leaf, leaf)
    assert RB.aggregate(argparse.Namespace(beacon=None, anchors=None, fallback_perms=False)) == 0
    out = _strict_loads(result.read_text(encoding="utf-8"))
    assert out["leaf_chain"]["ok"] and out["leaf_chain"]["entries"] == 65 and out["leaf_chain"]["rewrites"] == []
    assert out["schedule"] == {"expected_leaves": 65, "present": 65} and "nonfinite_fields" not in out
    assert out["config"]["perm_derivation"] == "sha256-of-tag fallback" and out["config"]["held_out_perms"] == perms
    assert out["plumbing"]["leaves_complete"] and out["plumbing"]["chain_ok"] and out["plumbing"]["metrics_finite"]


# -- section 15.3: what is committed -------------------------------------------------------------------------

def test_the_leaves_the_log_and_the_receipt_are_committable_the_leaves_and_log_pinned_and_the_rest_is_ignored():
    base = "papers/frequency-resonance/"
    leaf = base + "pmnist_robustness_runs/p1234_free_s0_4000_h64.json"
    expect_ignored = {leaf: False, base + "pmnist_robustness_runs/leaf_chain.jsonl": False,
                      base + "pmnist_robustness_result.json": False, leaf + ".tmp": True,
                      base + "pmnist_robustness_smoke.json": True, base + "_mnist.npz": True}
    for path, ignored in expect_ignored.items():
        try:
            r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index", path], capture_output=True,
                               timeout=60)
        except OSError:
            pytest.skip("git is not available here")
        if r.returncode not in (0, 1):
            pytest.skip(f"git check-ignore failed here: {r.stderr[:200]!r}")
        assert (r.returncode == 0) is ignored, path
    out = _git("check-attr", "text", "--", leaf, base + "pmnist_robustness_runs/leaf_chain.jsonl").decode()
    assert out.count("text: unset") == 2, out
