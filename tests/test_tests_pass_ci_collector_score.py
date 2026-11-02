# -*- coding: utf-8 -*-
"""The scorer of tests-pass-ci (P13 to P16, P19; sections 8 to 10), offline, on invented packets, answers and rows.

The Clopper-Pearson bound and the power curve of section 9.3; decoys scored on the full label with both per-side bars;
final labels by the qualified voters only; the order of sealing (no key before every answer file is sealed, a key only
by its committed digest, P2's decoys only from the set in the lock row); the publication order of G_KEYS; the A2 and A1
metrics end to end on invented packets; and the receipt, which `styxx.prereg_ci` re-derives and whose `check` passes in
a scratch repository. No model, no pull request and no network are read here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import pytest

from tests import _tests_pass_ci_fakes  # noqa: F401  (puts the collector on sys.path)
from tests._tests_pass_ci_fakes import COLLECTOR, ROOT

# the invented decoy fixture is read from the repository's own collector, never from a mutated copy
sys.path.insert(0, str(ROOT / "papers" / "tests-pass-ci" / "collect" / "checks"))

import synthetic_decoys  # noqa: E402
import tpc_decoys  # noqa: E402
import tpc_items  # noqa: E402
import tpc_packet  # noqa: E402
import tpc_score  # noqa: E402
import tpc_seats  # noqa: E402

PREREG = COLLECTOR.parent / "PREREG_tests_pass_ci_2026_11_02.md"


def _cp_ref(k, n, alpha=0.05):
    if n <= 0 or k <= 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)

    def tail(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    lo, hi = 0.0, 1.0
    while hi - lo > 1e-12:
        mid = (lo + hi) / 2
        if tail(mid) < 0.05:
            lo = mid
        else:
            hi = mid
    return lo


# ---- the arithmetic ------------------------------------------------------------------------------------------
def test_clopper_pearson_is_p15s():
    assert round(tpc_score.cp_lower(59, 59), 4) == 0.9505 and round(tpc_score.cp_lower(58, 58), 4) == 0.9497
    assert tpc_score.cp_lower(0, 10) == 0.0 and tpc_score.cp_lower(3, 0) == 0.0
    for k, n in ((1, 2), (5, 9), (60, 64), (62, 62), (34, 42)):
        assert abs(tpc_score.cp_lower(k, n) - _cp_ref(k, n)) < 1e-9
    assert round(tpc_score.cp_lower(34, 42), 2) == 0.68  # P19's description of panel noise
    assert abs(tpc_score.cp_lower(1500, 1600) - tpc_score.cp_lower(1500, 1600)) == 0  # the large-n branch runs


def test_wilson_is_styxx_mind_and_the_power_curve_is_section_9_3s():
    from styxx.mind import wilson
    assert tpc_score.wilson(2, 644) == wilson(2, 644)
    assert [round(v, 3) for v in tpc_score.power_curve(870).values()] == [0.942, 0.747, 0.516, 0.19, 0.042, 0.008]
    assert tpc_score.kmax_clearing(1000) == 3 and tpc_score.kmax_clearing(400) == 0 and tpc_score.kmax_clearing(380) == -1
    assert tpc_score.check_styxx_pins() == tpc_score.STYXX_PINS


# ---- P13 -------------------------------------------------------------------------------------------------------
def _key_with_decoys():
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    items = {}
    for i, d in enumerate(set_obj["decoys"], 1):
        items[f"P2-{i:03d}"] = {"decoy": True, "steps": 2, "side": d["side"], "construction": d["construction"],
                                "label": d["label"], "scope": d["scope"], "item_sha256": d["item_sha256"]}
    return {"packet": "P2", "items": items}


def _answers_for(key, wrong_yes=0, wrong_no=0, wrong_label="NOT_DECIDABLE"):
    rows = {}
    ny = nn = 0
    for iid, k in sorted(key["items"].items()):
        lab = k["label"]
        if k.get("decoy"):
            if k["side"] == "YES" and ny < wrong_yes:
                lab, ny = wrong_label, ny + 1
            elif k["side"] == "NO" and nn < wrong_no:
                lab, nn = (wrong_label if wrong_label != k["label"] else "NOT_DECIDABLE"), nn + 1
        rows[(iid, 1)] = {"answer": {"scope": k.get("scope") or "WHOLE", "local": False, "template_box": False,
                                     "statement": ""}}
        rows[(iid, 2)] = {"answer": {"label": lab, "evidence": "LOG", "preexisting": "no", "reason": ""}}
    return {"rows": rows, "sha256": "x"}


@pytest.mark.parametrize("wy,wn,ok", [(0, 0, True), (2, 1, True), (3, 0, False), (0, 2, False), (2, 2, False)])
def test_the_bars_bind_on_both_sides(wy, wn, ok):
    key = _key_with_decoys()
    sc = tpc_score.decoy_scores(key, {"H": _answers_for(key, wy, wn)})
    assert sc["H"]["qualified"] is ok
    assert sc["H"]["YES"] == 15 - wy and sc["H"]["NO"] == 15 - wn
    assert sc["N"] == {"sat": False, "qualified": False}


def test_a_decoy_is_right_only_on_its_full_label():
    key = _key_with_decoys()
    a = _answers_for(key)
    pb = next(i for i, k in key["items"].items() if k["construction"] == "no_part_beside")
    a["rows"][(pb, 2)]["answer"]["label"] = "PART_CONTRADICTED"  # the right side, the wrong label
    sc = tpc_score.decoy_scores(key, {"C": a})
    assert sc["C"]["NO"] == 14 and sc["C"]["per_construction"]["no_part_beside"] == {"right": 1, "of": 2}
    a["rows"][(pb, 2)] = {"answer": None}
    assert tpc_score.decoy_scores(key, {"C": a})["C"]["NO"] == 14


def test_final_labels_by_qualified_voters_only():
    key = {"items": {"P2-001": {"decoy": False, "steps": 2}, "P2-002": {"decoy": False, "steps": 2},
                     "P2-003": {"decoy": False, "steps": 1}}}

    def seat(s1, s2):
        rows = {}
        for iid, sc in s1.items():
            rows[(iid, 1)] = {"answer": None if sc is None else {"scope": sc, "local": False, "template_box": sc == "PART",
                                                                  "statement": ""}}
        for iid, lab in s2.items():
            rows[(iid, 2)] = {"answer": None if lab is None else {"label": lab, "evidence": "LOG", "preexisting": "no"}}
        return {"rows": rows}
    ans = {"H": seat({"P2-001": "WHOLE", "P2-002": "WHOLE", "P2-003": "PART"},
                     {"P2-001": "WHOLE_CONTRADICTED", "P2-002": "NOT_CONTRADICTED"}),
           "C": seat({"P2-001": "WHOLE", "P2-002": "PART", "P2-003": "PART"},
                     {"P2-001": "WHOLE_CONTRADICTED", "P2-002": "PART_BESIDE_FAILURE"}),
           "N": seat({"P2-001": "PART", "P2-002": "NONE", "P2-003": None},
                     {"P2-001": "PART_CONTRADICTED", "P2-002": "NOT_A_CLAIM"})}
    f3 = tpc_score.final_labels(key, ans, ["H", "C", "N"])
    assert f3["P2-001"]["label"] == "WHOLE_CONTRADICTED" and f3["P2-001"]["strict"] and f3["P2-001"]["scope"] == "WHOLE"
    assert f3["P2-002"]["label"] == "NOT_DECIDABLE" and f3["P2-002"]["label_why"] == "NO_MAJORITY"
    assert f3["P2-002"]["any_part"] and f3["P2-002"]["any_whole"]
    assert f3["P2-003"]["scope"] == "PART" and f3["P2-003"]["template_box"] is True and "label" not in f3["P2-003"]
    f2 = tpc_score.final_labels(key, ans, ["H", "N"])
    assert f2["P2-001"]["label"] == "NOT_DECIDABLE" and f2["P2-001"]["label_why"] == "SPLIT"
    assert f2["P2-003"]["scope"] is None  # one answer and one NO_ANSWER: no final scope
    fc = tpc_score.final_labels(key, ans, ["C", "N"])
    assert fc["P2-001"]["label"] == "NOT_DECIDABLE"
    f1 = tpc_score.final_labels(key, ans, ["H"])
    assert f1["P2-001"]["label_why"] == "PANEL_UNDER_TWO"
    ans["C"]["rows"][("P2-002", 2)]["answer"]["label"] = "NOT_CONTRADICTED"
    ans["C"]["rows"][("P2-002", 1)]["answer"]["scope"] = "WHOLE"
    f = tpc_score.final_labels(key, ans, ["H", "C", "N"])
    assert f["P2-002"]["label"] == "NOT_CONTRADICTED" and f["P2-002"]["strict"]
    assert f["P2-002"]["loose"] == ["NOT_A_CLAIM", "NOT_CONTRADICTED"]


def test_keys_in_order_reads_the_publication_times():
    pub = {"P1": {"key_public_at": "2026-11-10T00:00:00Z", "key_released_at": "2026-12-22T00:00:00Z",
                  "answers_public_at": {"H": "2026-12-21T00:00:00Z", "N": "2026-12-21T01:00:00Z"}},
           "P2": {"set_in_lock_row_at_signing": True, "key_public_at": "2026-12-01T00:00:00Z",
                  "key_released_at": "2027-01-12T00:00:00Z",
                  "answers_public_at": {"H": "2027-01-11T00:00:00Z", "C": "2027-01-11T00:00:00Z",
                                        "N": "2027-01-11T00:00:00Z"}}}
    assert tpc_score.keys_in_order(pub, "P1", ["H", "N"], p2=False) == 1
    assert tpc_score.keys_in_order(pub, "P1", ["H", "N", "C"], p2=False) == 0  # C's digest never public
    assert tpc_score.keys_in_order(pub, "P2", ["H", "C", "N"], p2=True) == 1
    late = json.loads(json.dumps(pub))
    late["P2"]["answers_public_at"]["N"] = "2027-01-12T00:00:00Z"  # not strictly before the release
    assert tpc_score.keys_in_order(late, "P2", ["H", "C", "N"], p2=True) == 0
    late = json.loads(json.dumps(pub))
    late["P2"]["set_in_lock_row_at_signing"] = False
    assert tpc_score.keys_in_order(late, "P2", ["H", "C", "N"], p2=True) == 0
    late = json.loads(json.dumps(pub))
    del late["P2"]["key_public_at"]  # P2's packet key digest never public before its seats read (finding 9)
    assert tpc_score.keys_in_order(late, "P2", ["H", "C", "N"], p2=True) == 0
    late = json.loads(json.dumps(pub))
    late["P1"]["key_public_at"] = "2026-12-21T12:00:00Z"  # the key digest after an answer
    assert tpc_score.keys_in_order(late, "P1", ["H", "N"], p2=False) == 0
    assert tpc_score.keys_in_order({}, "P1", ["H"], p2=False) == 0


# ---- packets on disk, answers, the order of sealing ---------------------------------------------------------------
def _write_packet(d, packet, reals, decoys, seed="a" * 64, kind="bitcoin"):
    built = tpc_packet.assemble(packet, seed, reals, decoys, beacon={"kind": kind, "beacon_hex": seed}, salt=b"\x07" * 32)
    tpc_packet.write_packet(d, built, {"packet": packet}, salt_dir=d)  # the salt beside the key, as once released
    return built


def _ledger(d, packet, seat, answer_of):
    """Write and seal a seat's answer file: answer_of(id, key_item) -> (step-1 answer, step-2 answer or None)."""
    pk, sha = tpc_seats.load_packet(d / f"packet_{packet}.json")
    key = json.loads((d / f"key_{packet}.json").read_bytes())
    lp = tpc_seats.ledger_path(d, packet, seat)
    for it in pk["items"]:
        s1, s2 = answer_of(it["id"], key["items"][it["id"]])
        tpc_seats._append(lp, {"packet": packet, "packet_sha256": sha, "seat": seat, "id": it["id"], "step": 1,
                               "answer": s1, "no_answer": None if s1 else "x", "final": True})
        if it["steps"] == 2:
            tpc_seats._append(lp, {"packet": packet, "packet_sha256": sha, "seat": seat, "id": it["id"], "step": 2,
                                   "answer": s2, "no_answer": None if s2 else "x", "final": True})
    return tpc_seats.seal_answers(d / f"packet_{packet}.json", seat, d)


def _s1(scope, box=False):
    return {"scope": scope, "local": False, "template_box": box, "statement": ""}


def _s2(label, evidence="LOG"):
    return {"label": label, "evidence": evidence, "preexisting": "no", "reason": ""}


def _decoy_answer(k):
    return _s1(k["scope"]), _s2(k["label"])


def test_no_key_opens_before_every_answer_file_is_sealed(tmp_path):
    decoys = tpc_decoys.build_set(synthetic_decoys.spec("P2"))[0]["decoys"]
    reals = [{"key": {"pr_id": 1, "strata": ["candidate_M"]}, "item": tpc_items.build_item("All tests pass.", {"jobs": []})}]
    _write_packet(tmp_path, "P2", reals, decoys)
    pk, sha = tpc_seats.load_packet(tmp_path / "packet_P2.json")
    with pytest.raises(tpc_score.ScoreError, match="key stays shut"):
        tpc_score.open_key(tmp_path, "P2", tpc_score.load_answers(tmp_path, "P2", sha))
    lp = tpc_seats.ledger_path(tmp_path, "P2", "C")
    tpc_seats._append(lp, {"packet": "P2", "packet_sha256": sha, "seat": "C", "id": "P2-001", "step": 1, "final": True})
    with pytest.raises(tpc_score.ScoreError, match="not sealed"):
        tpc_score.load_answers(tmp_path, "P2", sha)
    (tmp_path / "answers_P2_C.sha256").write_bytes((hashlib.sha256(lp.read_bytes()).hexdigest() + "\n").encode())
    assert set(tpc_score.load_answers(tmp_path, "P2", sha)) == {"C"}
    tpc_seats._append(lp, {"packet": "P2", "packet_sha256": sha, "seat": "C", "id": "P2-002", "step": 1, "final": True})
    with pytest.raises(tpc_score.ScoreError, match="no longer hashes"):
        tpc_score.load_answers(tmp_path, "P2", sha)


def test_a_key_opens_only_by_its_committed_digest_and_its_bytes(tmp_path):
    decoys = tpc_decoys.build_set(synthetic_decoys.spec("P2"))[0]["decoys"]
    reals = [{"key": {"pr_id": 1, "strata": ["candidate_M"]}, "item": tpc_items.build_item("All tests pass.", {"jobs": []})}]
    built = _write_packet(tmp_path, "P2", reals, decoys)
    _ledger(tmp_path, "P2", "C", lambda i, k: (_s1("WHOLE"), _s2("NOT_CONTRADICTED")))
    pk, sha = tpc_seats.load_packet(tmp_path / "packet_P2.json")
    ans = tpc_score.load_answers(tmp_path, "P2", sha)
    opened = tpc_score.open_key(tmp_path, "P2", ans)
    assert opened["digest"] == built["key_digest"]
    tpc_score.check_items(pk, opened["key"], decoys)
    other = tpc_decoys.build_set(synthetic_decoys.spec("Q"))[0]["decoys"]
    with pytest.raises(tpc_score.ScoreError, match="not in the sealed set"):
        tpc_score.check_items(pk, opened["key"], other)
    bad = json.loads(json.dumps(pk))
    bad["items"][0]["item_text"] = (bad["items"][0]["item_text"] or "") + " "
    bad["items"][0]["steps"] = 2
    with pytest.raises(tpc_score.ScoreError, match="bytes are not the key's"):
        tpc_score.check_items(bad, opened["key"], decoys)
    (tmp_path / "key_P2.sha256").write_bytes(b"0" * 64 + b"\n")
    with pytest.raises(tpc_score.ScoreError, match="committed digest"):
        tpc_score.open_key(tmp_path, "P2", ans)


# ---- end to end: A2 on an invented main sample ---------------------------------------------------------------
def _row(pid, outcome, *, screen="claimer", agent="Claude_Code", u=None, fires=False, stale=False):
    u = u if u is not None else 0.0001 + pid * 1e-7
    return {"pr_id": pid, "agent": agent, "repo": f"org/r{pid}", "number": pid, "u": u, "v": u / 0.12,
            "in_M": True, "in_D": True, "eligible": True, "screen": screen, "created_at": "2026-11-10T00:00:00Z",
            "stale": stale, "claims": [{"kind": "pass", "explicit": True, "scope": "full", "text": "All tests pass."}],
            "ci": {"outcome": outcome, "has_ci_test_result": True, "any_failed_job_or_check": outcome == "tests failed"},
            "rule_r": {"fires": fires}}


def _a2_world(tmp_path, n_pass=950, n_fail=50, contradicted=2, undecided=0, fires=0, h_wrong_yes=0):
    rows = [_row(i, "tests passed") for i in range(n_pass)]
    rows += [_row(10_000 + i, "tests failed", fires=i < fires) for i in range(n_fail)]
    store = tmp_path / "store"
    for r in rows:
        p = store / "a2" / "rows" / f"{r['pr_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(json.dumps(r).encode())
    sel = tpc_packet.select_p2(rows)
    reals = []
    for pid in sorted(sel["items"]):
        meta = sel["items"][pid]
        reals.append({"key": {"pr_id": pid, "repo": f"org/r{pid}", "number": pid, "agent": "Claude_Code",
                              "strata": meta["strata"], "in_M": True},
                      "item": tpc_items.build_item(f"Change {pid}. All tests pass.", {"jobs": []}, steps=meta["steps"])})
    d = tmp_path / "packets" / "P2"
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    sealed = tpc_decoys.seal_set(set_obj, salt=b"\x0b" * 32)
    tpc_packet._write_set(d, "P2", sealed, [], salt_dir=d)
    _write_packet(d, "P2", reals, set_obj["decoys"], seed="c" * 64)
    contra = {10_000 + i for i in range(contradicted)}
    undec = {10_000 + contradicted + i for i in range(undecided)}

    def answer(seat):
        def f(iid, k):
            if k.get("decoy"):
                s1, s2 = _decoy_answer(k)
                if seat == "H" and k["side"] == "YES" and f.wrong < h_wrong_yes:
                    f.wrong += 1
                    s2 = _s2("NOT_DECIDABLE")
                return s1, s2
            pid = k["pr_id"]
            if pid in contra:
                return _s1("WHOLE"), _s2("WHOLE_CONTRADICTED")
            if pid in undec:  # three different labels: NOT_DECIDABLE by NO_MAJORITY
                lab = {"H": "WHOLE_CONTRADICTED", "C": "NOT_CONTRADICTED", "N": "NOT_DECIDABLE"}[seat]
                return _s1("WHOLE"), _s2(lab)
            return _s1("WHOLE"), (_s2("NOT_CONTRADICTED") if k["steps"] == 2 else None)
        f.wrong = 0
        return f
    for seat in ("H", "C", "N"):
        _ledger(d, "P2", seat, answer(seat))
    pub = {"P2": {"set_in_lock_row_at_signing": True, "key_public_at": "2026-12-01T00:00:00Z",
                  "key_released_at": "2027-01-12T00:00:00Z",
                  "answers_public_at": {s: "2027-01-11T00:00:00Z" for s in ("H", "C", "N")}}}
    text = PREREG.read_text(encoding="utf-8")
    lines = []
    for ln in text.split("\n"):
        if ln.startswith("| decoy set digest of packet P2"):
            ln = ln.replace("TO-FILL at the signing commit", "`" + sealed["digest"] + "`")
        lines.append(ln)
    return {"rows": rows, "store": store, "packets": tmp_path / "packets", "pub": pub, "text": "\n".join(lines),
            "sel": sel}


def test_a2_end_to_end_lands_on_kill_with_the_strict_and_decisive_numbers(tmp_path):
    w = _a2_world(tmp_path)
    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"])  # P15's 10,000 resamples
    a2 = rec["result"]["a2"]
    c = rec["result"]["printed"]["a2"]
    assert c["printed"]["bootstrap_resamples"] == 10000
    assert a2["n_claimers_ci"] == 1000 and a2["keys_in_order"] == 1
    assert a2["human_qualified"] == 1 and a2["nonclaude_qualified"] == 1
    assert c["au4_rows"] == 100 and c["au4_claims"] == 100 and c["k"] == 2 and c["d_W"] == 0
    from styxx.mind import wilson
    assert c["s_lo"] == wilson(100, 100)[0] and c["n_eff"] == math.floor(950 * wilson(100, 100)[0]) + 50
    assert a2["whole_wilson_upper_worst"] == wilson(2, c["n_eff"])[1] < 0.01
    assert a2["agent_whole_wilson_lower_max"] == wilson(2, 1000)[0]
    assert a2["rule_firings"] == 0 and a2["rule_cp_lower"] == 0.0
    assert rec["verdict"] == "KILL__no_tests_pass_accuser__A1_VOID"
    assert rec["gates_sha256"] == tpc_score.GATES_SHA256 and rec["prereg"] == PREREG.name
    assert rec["result"]["a1"] == {"panel_valid": 0, "confirmed": -1, "added": -1, "k2025_strict": -1, "k2025_worst": -1}
    p = rec["result"]["printed"]["a2"]["printed"]
    assert p["whole_strict"]["k"] == 2 and p["whole_strict"]["n"] == 1000
    assert set(p["power_at_n_eff"]) == {"0.001", "0.002", "0.003", "0.005", "0.0075", "0.01"}
    assert len(p["bootstrap_k_over_n"]) == 2
    json.dumps(tpc_score._finite(rec), allow_nan=False)


def test_undecided_whole_candidates_count_against_rare(tmp_path):
    w = _a2_world(tmp_path, contradicted=1, undecided=3)
    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"], boot=200)
    c = rec["result"]["printed"]["a2"]
    assert c["k"] == 1 and c["d_W"] == 3
    from styxx.mind import wilson
    assert rec["result"]["a2"]["whole_wilson_upper_worst"] == wilson(4, c["n_eff"])[1] >= 0.01
    assert rec["verdict"].startswith("NO_ACCUSER__rule_underpowered")  # the strict k of 1 alone would have cleared


def test_a_human_seat_that_misses_three_yes_decoys_voids_the_panel(tmp_path):
    w = _a2_world(tmp_path, h_wrong_yes=3)
    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"], boot=200)
    assert rec["result"]["a2"]["human_qualified"] == 0
    assert rec["verdict"] == "INVALID__a2_panel_void_outside_seat__A1_VOID"


def test_rule_r_firings_are_scored_against_the_panel(tmp_path):
    w = _a2_world(tmp_path, contradicted=2, fires=3)
    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"], boot=200)
    a2 = rec["result"]["a2"]
    assert a2["rule_firings"] == 3 and a2["rule_cp_lower"] == tpc_score.cp_lower(2, 3)


def test_p2_is_not_scored_for_real_before_the_lock_row_is_filled(tmp_path):
    w = _a2_world(tmp_path)
    with pytest.raises(tpc_score.ScoreError, match="unfilled"):
        tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=PREREG.read_text(encoding="utf-8"), boot=200)
    smoke = tpc_score.score(w["packets"], w["store"], None, w["pub"], smoke=True,
                            prereg_text=PREREG.read_text(encoding="utf-8"), boot=200)
    assert smoke["verdict"] == "INVALID__smoke_plumbing_only" and smoke["gates"] == {}


def test_p2_scored_for_real_reads_the_lock_row_the_seal_guard_read(tmp_path):
    """Finding 8: with no text given, P2 is scored under the signed text the open guard read, compared against this
    checkout's PREREG; a shut guard refuses. The receipt names the PREREG bytes it was scored under."""
    import tpc_seal
    w = _a2_world(tmp_path)

    class Open:
        open = True
        state = {"open": True, "prereg_blob_sha256": "b" * 64}

        def signed_prereg(self, working_file=None):
            assert working_file == tpc_score.PREREG_FILE
            return w["text"]

    class Shut:
        open = False
        state = {"why": "no SEALED file"}

        def signed_prereg(self, working_file=None):
            raise tpc_seal.SealRefused("SEAL_GUARD: shut")

    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], guard=Open(), boot=200)
    printed = rec["result"]["printed"]
    assert printed["prereg_read"] == "the signed blob, through the seal guard"
    assert printed["prereg_sha256"] == hashlib.sha256(w["text"].encode("utf-8")).hexdigest()
    assert rec["verdict"] == tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"],
                                             boot=200)["verdict"]
    with pytest.raises(tpc_score.ScoreError, match="SEAL_GUARD"):
        tpc_score.score(w["packets"], w["store"], None, w["pub"], guard=Shut(), boot=200)


def test_under_400_claimers_prints_counts_only_with_the_sentinels(tmp_path):
    rows = [_row(i, "tests passed") for i in range(120)]
    store = tmp_path / "store"
    for r in rows:
        p = store / "a2" / "rows" / f"{r['pr_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(json.dumps(r).encode())
    (tmp_path / "packets").mkdir()
    rec = tpc_score.score(tmp_path / "packets", store, None, {})
    assert rec["result"]["a2"] == {"n_claimers_ci": 120, "keys_in_order": -1, "human_qualified": -1,
                                   "nonclaude_qualified": -1, "whole_wilson_upper_worst": 1.0,
                                   "agent_whole_wilson_lower_max": 1.0, "part_contradicted_wilson_upper": 1.0,
                                   "rule_firings": -1, "rule_cp_lower": 0.0}
    assert rec["verdict"] == "COUNTS_ONLY__a2_under_400__A1_VOID"


# ---- end to end: A1 on invented testreceipt files -------------------------------------------------------------
def _a1_world(tmp_path, confirm=36, add=3, undecided=0, n_wrong=0):
    tr = tmp_path / "tr"
    (tr / "evals" / "results").mkdir(parents=True)
    labels = {}
    m2 = ["# M2", ""]
    j1 = ["# Judge 1", "", "| # | label | reason |", "|---|---|---|"]
    j2 = ["# Judge 2", "", "| Case | Label | Reason |", "|---:|---|---|"]
    resolved = []
    for n in range(1, 96):
        explicit = n <= 87
        lab = "contradiction" if n <= 42 else ("out-of-scope" if n <= 76 else "not-a-claim")
        labels[n] = lab
        m2 += [f"## {n}. Claude_Code, merged=True, claim={'pass' if explicit else 'implied pass'}",
               f"https://github.com/acme/w{n}/pull/{n}", f"- head commit: {'b' * 40}",
               f"- all failing checks: test-{n}: failure", ""]
        j1.append(f"| {n} | {lab} | r |")
        j2.append(f"| {n} | {'unclear' if n == 5 else lab} | r |")
    resolved.append('5 = "contradiction"   # the judges differed')
    fr = ["# Fresh", ""]
    jf = ["# Judge 1", "", "| # | label | reason |", "|---|---|---|"]
    for n in range(1, 44):
        fr += [f"## {n}. SCOPED — Devin, merged=False", f"https://github.com/acme/f{n}/pull/{n}", ""]
        jf.append(f"| {n} | out-of-scope | r |")
    r = tr / "evals" / "results"
    (r / "sheet-m2.md").write_text("\n".join(m2), encoding="utf-8")
    (r / "sheet-fresh.md").write_text("\n".join(fr), encoding="utf-8")
    (r / "judge1-m2.md").write_text("\n".join(j1), encoding="utf-8")
    (r / "judge2-m2.md").write_text("\n".join(j2), encoding="utf-8")
    (r / "judge1-fresh.md").write_text("\n".join(jf), encoding="utf-8")
    (r / "judge2-fresh.md").write_text("\n".join(jf), encoding="utf-8")
    (tr / "evals" / "aidev_labels.toml").write_text("[m2_resolved]\n" + "\n".join(resolved) + "\n\n[fresh_resolved]\n",
                                                    encoding="utf-8")
    items = tpc_packet.parse_sheet((r / "sheet-m2.md").read_text(encoding="utf-8"), "2025")
    items += tpc_packet.parse_sheet((r / "sheet-fresh.md").read_text(encoding="utf-8"), "2026")
    reals = [{"key": {"year": it["year"], "n": it["n"], "url": it["url"], "agent": it["agent"], "claim": it["claim"]},
              "item": tpc_items.build_item(f"Item {it['year']}-{it['n']}. All tests pass.",
                                           tpc_items.ci_from_transcription(it.get("failing") or ""))} for it in items]
    d = tmp_path / "packets" / "P1"
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("P1"))
    tpc_packet._write_set(d, "P1", tpc_decoys.seal_set(set_obj), [], salt_dir=d)
    _write_packet(d, "P1", reals, set_obj["decoys"], seed="e" * 64)

    def answer(seat):
        def f(iid, k):
            if k.get("decoy"):
                s1, s2 = _decoy_answer(k)
                if seat == "N" and k["side"] == "NO" and f.wrong < n_wrong:
                    f.wrong += 1
                    s2 = _s2("NOT_DECIDABLE")
                return s1, s2
            n = k["n"]
            if k["year"] == "2025" and k["claim"] == "pass":
                if n <= confirm or (43 <= n < 43 + add):
                    return _s1("WHOLE"), _s2("WHOLE_CONTRADICTED")
                if 60 <= n < 60 + undecided:
                    lab = {"H": "WHOLE_CONTRADICTED", "C": "NOT_CONTRADICTED", "N": "NOT_DECIDABLE"}[seat]
                    return _s1("WHOLE"), _s2(lab)
            return _s1("PART"), _s2("PART_BESIDE_FAILURE")
        f.wrong = 0
        return f
    for seat in ("H", "C", "N"):
        _ledger(d, "P1", seat, answer(seat))
    pub = {"P1": {"key_public_at": "2026-11-20T00:00:00Z", "key_released_at": "2026-12-22T00:00:00Z",
                  "answers_public_at": {s: "2026-12-21T00:00:00Z" for s in ("H", "C", "N")}}}
    return tr, tmp_path / "packets", pub


def test_testreceipt_labels_read_the_judges_and_the_resolved_section(tmp_path):
    tr, _, _ = _a1_world(tmp_path)
    labs = tpc_score.testreceipt_labels(tr)
    assert labs[("2025", 5)] == "contradiction" and sum(1 for (y, _), v in labs.items() if y == "2025" and
                                                          v == "contradiction") == 42
    assert len([k for k in labs if k[0] == "2026"]) == 43


@pytest.mark.parametrize("confirm,add,undecided,want", [
    (36, 3, 0, "A1_STANDS"), (22, 3, 0, "A1_LOWER"), (42, 14, 0, "A1_HIGHER"), (30, 2, 0, "A1_UNSETTLED"),
    (30, 0, 5, "A1_UNSETTLED"), (27, 0, 5, "A1_UNSETTLED"), (25, 0, 5, "A1_LOWER"),
])
def test_a1_end_to_end_reads_item_by_item(tmp_path, confirm, add, undecided, want):
    tr, packets, pub = _a1_world(tmp_path, confirm, add, undecided)
    rec = tpc_score.score(packets, None, tr, pub)
    a1 = rec["result"]["a1"]
    assert a1["panel_valid"] == 1 and a1["confirmed"] == confirm and a1["added"] == add
    assert a1["k2025_worst"] == confirm + add + undecided
    assert rec["verdict"] == "COUNTS_ONLY__a2_under_400__" + want


def test_a1_is_void_when_the_non_claude_seat_fails_its_no_side(tmp_path):
    tr, packets, pub = _a1_world(tmp_path, n_wrong=2)
    rec = tpc_score.score(packets, None, tr, pub)
    assert rec["result"]["a1"]["panel_valid"] == 0 and rec["verdict"] == "COUNTS_ONLY__a2_under_400__A1_VOID"
    assert rec["result"]["printed"]["a1"]["confirmed_raw"] == 36  # the counts still print


# ---- the receipt and prereg-ci ----------------------------------------------------------------------------------
def test_the_receipt_is_what_prereg_ci_rederives(tmp_path):
    from styxx.prereg_ci import compare, rescore
    w = _a2_world(tmp_path)
    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"], boot=200)
    data = tpc_score.receipt_bytes(rec)
    again = json.loads(data)
    scored = rescore(PREREG.read_text(encoding="utf-8"), again)
    assert scored["verdict"] == rec["verdict"] and compare(scored, again) == []
    moved = dict(again, verdict="FLOOR_MET__new_prereg_may_propose_accuser__A1_VOID")
    assert [f["code"] for f in compare(scored, moved)] == ["VERDICT_DIFFERS"]


def _git(repo, *args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@localhost", "-c", "core.autocrlf=false",
                           "-c", "commit.gpgsign=false"] + list(args), cwd=repo, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def test_prereg_ci_check_passes_the_receipt_in_a_scratch_repository(tmp_path):
    w = _a2_world(tmp_path / "w")
    rec = tpc_score.score(w["packets"], w["store"], None, w["pub"], prereg_text=w["text"], boot=200)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "-q", "-b", "main").returncode == 0
    (repo / PREREG.name).write_bytes(PREREG.read_bytes().replace(b"\r\n", b"\n"))
    _git(repo, "add", PREREG.name)
    assert _git(repo, "commit", "-qm", "the PREREG").returncode == 0
    (repo / "measurement_result.json").write_bytes(tpc_score.receipt_bytes(rec))
    result = ("# RESULT (invented numbers, for the scorer's test only)\n\n```result\n"
              + json.dumps({"prereg": PREREG.name, "receipt": "measurement_result.json", "verdict": rec["verdict"]})
              + "\n```\n")
    (repo / "RESULT_tests_pass_ci_test.md").write_bytes(result.encode("utf-8"))
    _git(repo, "add", "measurement_result.json", "RESULT_tests_pass_ci_test.md")
    assert _git(repo, "commit", "-qm", "receipt").returncode == 0
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "styxx.prereg_ci", "check", "--base", "HEAD~1", "--head", "HEAD"],
                       cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "PASS" in r.stdout and rec["verdict"] in r.stdout
    (repo / "RESULT_tests_pass_ci_test.md").write_bytes(result.replace(rec["verdict"],
                                                                       "FLOOR_MET__new_prereg_may_propose_accuser__A1_VOID").encode())
    _git(repo, "commit", "-qam", "another verdict typed")
    r = subprocess.run([sys.executable, "-m", "styxx.prereg_ci", "check", "--base", "HEAD~2", "--head", "HEAD"],
                       cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300)
    assert r.returncode == 1 and "VERDICT_DIFFERS" in r.stdout


def test_the_cli_writes_a_receipt_once(tmp_path):
    (tmp_path / "packets").mkdir()
    out = tmp_path / "measurement_result.json"
    assert tpc_score.main(["score", "--packets", str(tmp_path / "packets"), "--out", str(out), "--smoke"]) == 0
    rec = json.loads(out.read_bytes())
    assert rec["smoke"] is True and rec["verdict"] == "INVALID__smoke_plumbing_only"
    assert tpc_score.main(["score", "--packets", str(tmp_path / "packets"), "--out", str(out), "--smoke"]) == 2
