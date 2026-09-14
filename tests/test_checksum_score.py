# -*- coding: utf-8 -*-
"""papers/checksum/score.py executes the two frozen PREREGs' reading rules on a certs file.

v2 of these tests, after the red team of 2026-09-14 showed v1 bound bands by formatted needles (seven
band changes and six ≤/< flips passed every v1 test). Now: every number in BANDS is PARSED out of the
frozen text and compared with the code; every comparator is pinned by certs sitting exactly on each
edge; the sealed digests are the frozen files' bytes and the SEALS rows; and the scorer is shown to
re-derive what v1 trusted — K1 from the floor, the sealed digest without --expect-blob, the 48-item
draw, every arm grading the same set.

After the verification of 2026-09-14 (S1-S7): a beacon_draw card without --expect-beacon is not a
result and says so first; when K1 did not fire every arm cert, the held-out cert and the canary hash
must exist; is_the_experiment is checked against tag, smoke and git; a hand set whose K1 fired leaves
H5 PENDING; H6 reads only a portability record bound to these certs; malformed certs yield problems,
not exceptions. Every check has a test that refuses on its own named ground, and
papers/checksum/score_mutations.py publishes the mutations these tests kill, one by one."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "papers", "checksum")


def _load(name="score", file="score.py"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CK, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load()


def _doc(name):
    t = open(os.path.join(CK, name), encoding="utf-8").read()
    t = re.sub(r"<sworn[^>]*>|</sworn>", "", t)
    return re.sub(r"\s+", " ", t)


def _certs(name):
    return copy.deepcopy(json.load(open(os.path.join(CK, name), encoding="utf-8")))


BEACON_CERTS = "beacon_draw_certs_dryrun_qwen0.5b.json"
HAND_CERTS = "deploy_quant_certs_dryrun_qwen0.5b.json"
BEACON = _certs(BEACON_CERTS)["canaries"]["draw"]["beacon"]


def _experiment(prereg):
    """The committed 0.5B instrument check reshaped as the experiment: the only fields changed are the ones
    that say it is not (is_the_experiment, model, tag), plus what the current runner writes and those certs
    predate — provenance.git_dirty_tracked / prereg_blob_is_sealed, and for the hand set the canaries block
    (run_deploy_quant.py writes it in both branches; the committed deploy_quant certs are not edited).
    Everything the scorer re-derives stays real."""
    c = _certs(BEACON_CERTS if prereg == "beacon_draw" else HAND_CERTS)
    c["is_the_experiment"], c["model"], c["tag"] = True, score.MODEL, ""
    c["provenance"].update(git_dirty_tracked=False, prereg_blob_is_sealed=True)
    if prereg == "deploy_quant":
        assert "canaries" not in c                          # the committed file predates the block
        c["canaries"] = {"n": score.N_CANARIES, "canary_sha256": score.HAND_CANARY_SHA256, "draw": None,
                         "source": "styxx.checksum.CANARIES (the hand-written set)"}
    return c


def _score(c, prereg, **kw):
    """score() as a result is read: under beacon_draw with the ANCHORED beacon unless a test says otherwise."""
    if prereg == "beacon_draw":
        kw.setdefault("expect_beacon", BEACON)
    return score.score(c, prereg, **kw)


def _detail(card, prereg):
    d = card["gates"]["K5" if prereg == "beacon_draw" else "provenance"]["detail"]
    return d if isinstance(d, list) else [d]


def _put(path, value):
    def edit(c):
        *head, last = path.split(".")
        node = c
        for k in head:
            node = node[k]
        node[last] = value
    return edit


def _drop(path):
    def edit(c):
        *head, last = path.split(".")
        node = c
        for k in head:
            node = node[k]
        del node[last]
    return edit


# ----------------------------------------------------------------------------- the bands are the frozen text

def test_every_beacon_draw_band_is_parsed_out_of_the_frozen_text_and_equals_the_code():
    b = score.BANDS["beacon_draw"]
    t = _doc(b["prereg"])
    m = re.search(r"\*\*H1 \(floor, held-out form\)\.\*\* The worst-pairwise null floor is ≤ (\S+) nats/token; zero is allowed", t)
    assert m and float(m.group(1)) == b["h1_floor_max"] and b["h1_floor_min_exclusive"] is None
    m = re.search(r"\*\*H2 \(NF4 on the drawn set\)\.\*\* A vs Q4 reads DRIFT; mean \|Δ log-prob\| in \*\*\[([\d.]+), ([\d.]+)\] "
                  r"nats/token\*\*; geometry r ≥ ([\d.]+); argmax lost on ≤ (\d+) of 48", t)
    assert m and (float(m.group(1)), float(m.group(2))) == b["h2_band"]
    assert float(m.group(3)) == b["h2_r_min"] and int(m.group(4)) == b["h2_top1_loss_max"]
    m = re.search(r"\*\*H3 \(int8\)\.\*\* A vs Q8 reads DRIFT with mean \|Δ log-prob\| below Q4's; r ≥ ([\d.]+); argmax lost on ≤ (\d+) of 48", t)
    assert m and float(m.group(1)) == b["h3_r_min"] and int(m.group(2)) == b["h3_top1_loss_max"]
    m = re.search(r"\*\*H4 \(far control\)\.\*\* A vs R: mean \|Δ log-prob\| > ([\d.]+) nats/token; r < ([\d.]+); top-1 hits ≤ (\d+) of 48", t)
    assert m and float(m.group(1)) == b["h4_mean_min"] and float(m.group(2)) == b["h4_r_max"] and int(m.group(3)) == b["h4_hits_max"]
    assert "NF4 mean lies within a factor of two of the hand set's, either way" in t and b["h5_ratio_band"] == (0.5, 2.0)
    m = re.search(r"Q4's and Q8's mean \|Δ log-prob\| move by at most ([\d.]+) nats/token and R's by at most ([\d.]+)", t)
    assert m and b["h6_move_max"] == {"Q4": float(m.group(1)), "Q8": float(m.group(1)), "R": float(m.group(2))}
    m = re.search(r"\*\*K1\.\*\* Worst-pairwise null floor > (\S+) nats/token", t)
    assert m and float(m.group(1)) == b["k1_floor"]
    m = re.search(r"\*\*K3\.\*\* A vs Q4 > ([\d.]+) nats/token, or argmax lost on > (\d+) of 48", t)
    assert m and float(m.group(1)) == b["k3_mean"] and int(m.group(2)) == b["k3_top1_loss"]


def test_every_deploy_quant_band_is_parsed_out_of_the_frozen_text_and_equals_the_code():
    b = score.BANDS["deploy_quant"]
    t = _doc(b["prereg"])
    m = re.search(r"\*\*H1 \(floor\)\.\*\* The null floor on GPU is > (\S+) and ≤ (\S+) nats/token\.", t)
    assert m and float(m.group(1)) == b["h1_floor_min_exclusive"] and float(m.group(2)) == b["h1_floor_max"]
    m = re.search(r"\*\*H2 \(NF4\)\.\*\* A vs Q4 reads DRIFT, mean \|Δ log-prob\| in \*\*\[([\d.]+), ([\d.]+)\] nats/token\*\*, "
                  r"belief geometry r ≥ ([\d.]+), top-1 loss ≤ (\d+) of 48", t)
    assert m and (float(m.group(1)), float(m.group(2))) == b["h2_band"]
    assert float(m.group(3)) == b["h2_r_min"] and int(m.group(4)) == b["h2_top1_loss_max"]
    m = re.search(r"\*\*H4 \(far control\)\.\*\* A vs R: mean \|Δ log-prob\| > ([\d.]+) nats/token, r < ([\d.]+), top-1 ≤ (\d+) of 48", t)
    assert m and float(m.group(1)) == b["h4_mean_min"] and float(m.group(2)) == b["h4_r_max"] and int(m.group(3)) == b["h4_hits_max"]
    m = re.search(r"\*\*K1\.\*\* Null floor > (\S+) nats/token", t)
    assert m and float(m.group(1)) == b["k1_floor"]
    m = re.search(r"\*\*K3\.\*\* A vs Q4 > ([\d.]+) nats/token or Q4 top-1 loss > (\d+)", t)
    assert m and float(m.group(1)) == b["k3_mean"] and int(m.group(2)) == b["k3_top1_loss"]


def test_the_sealed_digests_are_the_frozen_files_and_the_seal_rows_and_the_hand_set_is_the_named_one():
    seals = open(os.path.join(CK, "SEALS_2026_09_13.md"), encoding="utf-8").read()
    for b in score.BANDS.values():
        assert hashlib.sha256(open(os.path.join(CK, b["prereg"]), "rb").read()).hexdigest() == b["sealed_blob"]
        assert b["sealed_blob"] in seals
    from styxx import checksum as ck
    assert ck.canary_sha256(ck.CANARIES) == score.HAND_CANARY_SHA256
    assert score.HAND_CANARY_SHA256[:12] in _doc(score.BANDS["deploy_quant"]["prereg"])


# ----------------------------------------------------------------------------- every comparator, on its edge

def _set(c, **v):
    s = c["sanity"]
    if "floor" in v:
        s["null_floor_nats"] = v["floor"]
        c["k1"]["floor_nats"], c["k1"]["fired"] = v["floor"], v["floor"] > 1e-2
    for key, arm, field in (("m4", "Q4", "mean_abs_nats"), ("r4", "Q4", "rdm_r"), ("m8", "Q8", "mean_abs_nats"),
                            ("r8", "Q8", "rdm_r"), ("mr", "R", "mean_abs_nats"), ("rr", "R", "rdm_r")):
        if key in v:
            c[arm]["distance"][field] = v[key]
    if "l4" in v:
        s["top1_loss_vs_A"]["Q4"] = v["l4"]
    if "l8" in v:
        s["top1_loss_vs_A"]["Q8"] = v["l8"]
    if "hr" in v:
        s["first_token_top1_hits"]["R"] = v["hr"]
    return c


EDGES = [
    # (prereg, fields, where, expected) — where is (hypothesis, clause index) or ("gate", name)
    ("beacon_draw", {"floor": 1e-3}, ("H1", 0), True),
    ("beacon_draw", {"floor": 1.000001e-3}, ("H1", 0), False),
    ("beacon_draw", {"floor": 0.0}, ("H1", 0), True),
    ("beacon_draw", {"m4": 0.60}, ("H2", 1), True),
    ("beacon_draw", {"m4": 0.05, "m8": 0.01}, ("H2", 1), True),
    ("beacon_draw", {"m4": 0.6000001}, ("H2", 1), False),
    ("beacon_draw", {"m4": 0.0499999, "m8": 0.01}, ("H2", 1), False),
    ("beacon_draw", {"r4": 0.90}, ("H2", 2), True),
    ("beacon_draw", {"r4": 0.8999999}, ("H2", 2), False),
    ("beacon_draw", {"l4": 24}, ("H2", 3), True),
    ("beacon_draw", {"l4": 25}, ("H2", 3), False),
    ("beacon_draw", {"m8": 0.34657728324075127}, ("H3", 1), False),        # equal to Q4's mean: "below" is strict
    ("beacon_draw", {"r8": 0.98}, ("H3", 2), True),
    ("beacon_draw", {"r8": 0.9799999}, ("H3", 2), False),
    ("beacon_draw", {"l8": 8}, ("H3", 3), True),
    ("beacon_draw", {"l8": 9}, ("H3", 3), False),
    ("beacon_draw", {"mr": 5.0}, ("H4", 0), False),
    ("beacon_draw", {"mr": 5.0000001}, ("H4", 0), True),
    ("beacon_draw", {"rr": 0.6}, ("H4", 1), False),
    ("beacon_draw", {"rr": 0.5999999}, ("H4", 1), True),
    ("beacon_draw", {"hr": 4}, ("H4", 2), True),
    ("beacon_draw", {"hr": 5}, ("H4", 2), False),
    ("beacon_draw", {"m4": 1.5}, ("gate", "K3"), False),
    ("beacon_draw", {"m4": 1.5000001}, ("gate", "K3"), True),
    ("beacon_draw", {"l4": 36}, ("gate", "K3"), False),
    ("beacon_draw", {"l4": 37}, ("gate", "K3"), True),
    ("beacon_draw", {"mr": 0.60}, ("gate", "K4"), True),
    ("beacon_draw", {"mr": 0.6000001}, ("gate", "K4"), False),
    ("beacon_draw", {"mr": 0.05}, ("gate", "K4"), True),                  # K4's lower edge is inclusive too
    ("beacon_draw", {"mr": 0.0499999}, ("gate", "K4"), False),
    ("beacon_draw", {"floor": 1e-2}, ("gate", "K1"), False),
    ("beacon_draw", {"floor": 1.0000001e-2}, ("gate", "K1"), True),
    ("deploy_quant", {"floor": 0.0}, ("H1", 0), False),
    ("deploy_quant", {"floor": 1e-9}, ("H1", 0), True),
    ("deploy_quant", {"floor": 1e-3}, ("H1", 0), True),
    ("deploy_quant", {"floor": 1.000001e-3}, ("H1", 0), False),
    ("deploy_quant", {"m4": 0.30}, ("H2", 1), True),
    ("deploy_quant", {"m4": 0.01, "m8": 0.001}, ("H2", 1), True),
    ("deploy_quant", {"m4": 0.3000001}, ("H2", 1), False),
    ("deploy_quant", {"m4": 0.0099999, "m8": 0.001}, ("H2", 1), False),
    ("deploy_quant", {"r4": 0.95}, ("H2", 2), True),
    ("deploy_quant", {"r4": 0.9499999}, ("H2", 2), False),
    ("deploy_quant", {"l4": 4}, ("H2", 3), True),
    ("deploy_quant", {"l4": 5}, ("H2", 3), False),
    ("deploy_quant", {"mr": 5.0}, ("H4", 0), False),
    ("deploy_quant", {"mr": 5.0000001}, ("H4", 0), True),
    ("deploy_quant", {"rr": 0.3}, ("H4", 1), False),
    ("deploy_quant", {"rr": 0.2999999}, ("H4", 1), True),
    ("deploy_quant", {"hr": 2}, ("H4", 2), True),
    ("deploy_quant", {"hr": 3}, ("H4", 2), False),
    ("deploy_quant", {"m4": 1.0}, ("gate", "K3"), False),
    ("deploy_quant", {"m4": 1.0000001}, ("gate", "K3"), True),
    ("deploy_quant", {"l4": 12}, ("gate", "K3"), False),
    ("deploy_quant", {"l4": 13}, ("gate", "K3"), True),
    ("deploy_quant", {"mr": 0.30}, ("gate", "K4"), True),
    ("deploy_quant", {"mr": 0.3000001}, ("gate", "K4"), False),
    ("deploy_quant", {"mr": 0.01}, ("gate", "K4"), True),
    ("deploy_quant", {"mr": 0.0099999}, ("gate", "K4"), False),
    ("deploy_quant", {"floor": 1e-2}, ("gate", "K1"), False),
    ("deploy_quant", {"floor": 1.0000001e-2}, ("gate", "K1"), True),
]


@pytest.mark.parametrize("prereg,fields,where,expected", EDGES, ids=[f"{p}-{w[0]}{w[1]}-{f}" for p, f, w, _ in EDGES])
def test_every_comparator_on_its_edge(prereg, fields, where, expected):
    card = score.score(_set(_experiment(prereg), **fields), prereg)
    if where[0] == "gate":
        assert card["gates"][where[1]]["fired"] is expected
    else:
        assert card["hypotheses"][where[0]]["clauses"][where[1]]["holds"] is expected


@pytest.mark.parametrize("prereg", ["beacon_draw", "deploy_quant"])
def test_k2_fires_exactly_when_a_vs_q4_reads_same(prereg):
    c = _experiment(prereg)
    c["Q4"]["distance"]["verdict"] = "DRIFT"
    assert _score(c, prereg)["gates"]["K2"] == {"fired": False, "detail": "not fired"}
    c["Q4"]["distance"]["verdict"] = "SAME"
    k2 = _score(c, prereg)["gates"]["K2"]
    assert k2["fired"] is True and "cannot see a deployed quantization" in k2["detail"]
    assert "K2" in _score(c, prereg)["run_reading"]
    del c["Q4"]
    assert _score(c, prereg)["gates"]["K2"]["fired"] is None


def _seal_portability(rec):
    """The digest styxx.portability.compare writes: sha256 over the canonical body without the fields it keeps outside."""
    body = {k: v for k, v in rec.items() if k not in ("digest", "labels", "created", "inputs")}
    rec["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return rec


def _reseal_arms(certs):
    """Re-derive every arm cert's digest as styxx.checksum writes it (sha256 of the body without digest and created)."""
    for a in score.ARMS:
        body = {k: v for k, v in certs[a].items() if k not in ("digest", "created")}
        certs[a]["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return certs


def _portability(certs, moves=None, verdicts="AGREE"):
    """A record in the shape styxx.portability.compare writes over these certs (machine 0) and a second machine
    whose arm values sit `moves` above them; with verdicts="FLIP" the second machine reads Q4 differently."""
    moves = {"Q4": 0.10, "Q8": 0.10, "R": 0.22} if moves is None else moves
    other = sorted(hashlib.sha256(f"the second machine's {a}".encode()).hexdigest() for a in score.ARMS)
    arms = {}
    for a in score.ARMS:
        d = certs[a]["distance"]
        there = d["verdict"] if not (verdicts == "FLIP" and a == "Q4") else ("SAME" if d["verdict"] != "SAME" else "DRIFT")
        values = [d["mean_abs_nats"], d["mean_abs_nats"] + moves.get(a, 0.0)]
        arms[a] = {"verdicts": [d["verdict"], there],
                   "numbers": {"mean_abs_nats": {"values": values, "max_abs_diff": max(values) - min(values)}}}
    rec = {"schema": "styxx.portability/v1", "n_machines": 2,
           "input_cert_digests": [sorted(certs[a]["digest"] for a in score.ARMS), other],
           "arms": arms, "verdicts": verdicts}
    return _seal_portability(rec)


def _zero_means(certs):
    """These certs with every arm's mean |Δ log-prob| at 0.0, resealed, so a move of m is a spread of exactly m."""
    for a in score.ARMS:
        certs[a]["distance"]["mean_abs_nats"] = 0.0
    return _reseal_arms(certs)


def test_h5_ratio_band_is_inclusive_both_ways_and_h6_moves_are_inclusive():
    hand = _experiment("deploy_quant")
    hm4 = hand["Q4"]["distance"]["mean_abs_nats"]
    for m4, holds in ((2.0 * hm4, True), (0.5 * hm4, True), (2.0000001 * hm4, False), (0.4999999 * hm4, False)):
        c = _set(_experiment("beacon_draw"), m4=m4, m8=0.01, l4=10)
        h5 = score.score(c, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]
        assert h5["clauses"][-1]["holds"] is holds, (m4 / hm4, h5)
    c = _zero_means(_experiment("beacon_draw"))
    port = _portability(c)
    h6 = score.score(c, "beacon_draw", portability=port)["hypotheses"]["H6"]
    assert h6["status"] == "HELD" and h6["portability_digest"] == port["digest"]
    assert [x["observed"] for x in h6["clauses"][1:]] == [0.10, 0.10, 0.22]          # exactly on each edge
    port = _portability(c, moves={"Q4": 0.10, "Q8": 0.1000001, "R": 0.22})
    h6 = score.score(c, "beacon_draw", portability=port)["hypotheses"]["H6"]
    assert h6["status"] == "FAILED" and [x["holds"] for x in h6["clauses"]] == [True, True, False, True]


def test_h6_needs_every_verdict_to_agree():
    c = _zero_means(_experiment("beacon_draw"))
    h6 = score.score(c, "beacon_draw", portability=_portability(c, verdicts="FLIP"))["hypotheses"]["H6"]
    assert h6["status"] == "FAILED" and [x["holds"] for x in h6["clauses"]] == [False, True, True, True]


def test_beacon_h1_applies_the_correction_rules_when_the_floor_is_above_zero():
    text = _doc("CORRECTION_prereg_beacon_draw_2026_09_14.md")
    c = _set(_experiment("beacon_draw"), floor=5e-4)
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "INCONCLUSIVE"
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "HELD on the floor, INCONCLUSIVE on the pair by construction" and "rule 3" in h1["rule"]
    assert h1["status"] in text
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "DRIFT"
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "FAILED" and "rule 4" in h1["rule"]
    c = _set(_experiment("beacon_draw"), floor=0.0)                   # bit-identical loads: evaluated as frozen
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "INCONCLUSIVE"
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "FAILED" and h1["rule"] is None
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "DRIFT"         # rule 4's note is for a floor above zero only
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "FAILED" and h1["rule"] is None
    c = _set(_experiment("beacon_draw"), floor=2e-3)                  # the floor clause fails: no by-construction reading
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "INCONCLUSIVE"
    assert score.score(c, "beacon_draw")["hypotheses"]["H1"]["status"] == "FAILED"


def test_deploy_quant_h1_applies_correction_rule_2_in_its_words():
    text = _doc("CORRECTION_prereg_deploy_quant_H1_2026_09_13.md")
    c = _set(_experiment("deploy_quant"), floor=5e-4)
    c["A2"]["distance"]["verdict"] = "INCONCLUSIVE"
    h1 = score.score(c, "deploy_quant")["hypotheses"]["H1"]
    assert h1["status"] == "HELD on the floor, INCONCLUSIVE on the pair by construction" and h1["status"] in text
    assert h1["rule"] == score.BANDS["deploy_quant"]["h1_rule"]
    c["A2"]["distance"]["verdict"] = "SAME"
    assert score.score(c, "deploy_quant")["hypotheses"]["H1"]["status"] == "HELD"
    c = _set(_experiment("deploy_quant"), floor=0.0)                  # the floor clause fails: rule 2 does not apply
    c["A2"]["distance"]["verdict"] = "INCONCLUSIVE"
    assert score.score(c, "deploy_quant")["hypotheses"]["H1"]["status"] == "FAILED"


# ----------------------------------------------------------------------------- what v1 trusted, v2 re-derives

def test_without_expect_beacon_a_beacon_draw_card_is_not_a_result_and_its_reading_says_so_first():
    card = score.score(_experiment("beacon_draw"), "beacon_draw")
    assert card["counts_as_result"] is False and card["gates"]["K5"]["fired"] is None
    assert card["gates"]["K5"]["detail"] == "not evaluable: " + score.BEACON_UNCHECKED + "; every other K5 clause holds"
    assert card["run_reading"].startswith("NOT A RESULT: K5's beacon clause not evaluated (no --expect-beacon); ")
    assert "INSTRUMENT CHECK" not in card["run_reading"]
    assert any("beacon clause was not checked" in n for n in card["notes"])
    card = score.score(_experiment("beacon_draw"), "beacon_draw", expect_beacon=BEACON)
    assert card["counts_as_result"] is True and card["gates"]["K5"]["fired"] is False
    assert not card["run_reading"].startswith(("NOT A RESULT", "INSTRUMENT CHECK"))
    card = score.score(_certs(BEACON_CERTS), "beacon_draw")         # an instrument check without the beacon says both
    assert card["run_reading"].startswith("NOT A RESULT: K5's beacon clause not evaluated (no --expect-beacon); INSTRUMENT CHECK; ")
    assert card["gates"]["K5"]["fired"] is True and card["gates"]["K5"]["detail"][-1] == score.BEACON_UNCHECKED
    card = score.score(_experiment("deploy_quant"), "deploy_quant")  # the hand set has no beacon to check
    assert card["gates"]["provenance"]["fired"] is False and card["counts_as_result"] is True


def test_the_docstring_and_the_cli_help_say_expect_beacon_is_required_for_a_beacon_draw_result(capsys):
    assert "--expect-beacon is REQUIRED for the card to count as a result" in " ".join(score.__doc__.split())
    with pytest.raises(SystemExit):
        score.main(["--help"])
    assert "required for a beacon_draw result" in " ".join(capsys.readouterr().out.split())


def test_k1_is_re_derived_from_the_floor_and_a_contradicting_k1_record_invalidates_the_certs():
    c = _experiment("beacon_draw")
    c["sanity"]["null_floor_nats"] = 0.05                      # the runner's k1 record still says fired: false
    card = _score(c, "beacon_draw")
    assert card["gates"]["K1"]["fired"] is True and card["counts_as_result"] is False
    assert any("k1 record" in d for d in card["gates"]["K5"]["detail"])
    del c["k1"]
    assert score.score(c, "beacon_draw")["gates"]["K1"]["fired"] is True


def test_a_valid_sealed_run_whose_k1_fired_is_a_result_inconclusive_and_names_every_hypothesis():
    c = _set(_experiment("beacon_draw"), floor=0.05)
    for a in score.ARMS:
        del c[a]
    del c["h1_held_out"]                                       # the runner's K1 branch writes no arm and no held-out cert
    card = _score(c, "beacon_draw")
    assert card["counts_as_result"] is True and card["run_reading"] == "INCONCLUSIVE (K1)"
    assert list(card["hypotheses"]) == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert all(v["status"] == "NOT_EVALUATED" for v in card["hypotheses"].values())
    assert all(card["gates"][g]["fired"] is None for g in ("K2", "K3", "K4"))
    assert score.score(c, "beacon_draw")["run_reading"] == "NOT A RESULT: K5's beacon clause not evaluated (no --expect-beacon); INCONCLUSIVE (K1)"
    c = _set(_experiment("deploy_quant"), floor=0.05)
    for a in score.ARMS:
        del c[a]
    del c["h1_held_out"]
    card = score.score(c, "deploy_quant")
    assert card["counts_as_result"] is True and card["run_reading"] == "INCONCLUSIVE (K1)"
    assert list(card["hypotheses"]) == ["H1", "H2", "H3", "H4"]


def test_the_sealed_digest_is_checked_without_expect_blob_and_a_wrong_expect_blob_is_refused():
    c = _experiment("beacon_draw")
    c["provenance"]["prereg_blob_sha256"] = "0" * 64
    card = _score(c, "beacon_draw")
    assert card["gates"]["K5"]["fired"] is True and card["counts_as_result"] is False
    assert any("not the sealed digest" in d for d in card["gates"]["K5"]["detail"])
    card = _score(_experiment("beacon_draw"), "beacon_draw", expect_blob="1" * 64)
    assert any("--expect-blob" in d for d in card["gates"]["K5"]["detail"])
    assert _score(_experiment("beacon_draw"), "beacon_draw", expect_blob=score.BANDS["beacon_draw"]["sealed_blob"])["counts_as_result"] is True
    c = _experiment("deploy_quant")
    c["provenance"]["prereg_blob_sha256"] = "0" * 64
    assert score.score(c, "deploy_quant")["gates"]["provenance"]["fired"] is True


def test_an_honest_draw_of_the_wrong_size_and_arms_grading_another_set_fire_k5():
    from styxx import beacon, checksum as ck
    c = _experiment("beacon_draw")
    draw = c["canaries"]["draw"]
    eight = ck.canary_sha256(beacon.select(draw["beacon"], 8))
    for rec in [c["canaries"]["draw"]] + [c[a]["draw"] for a in score.ARMS] + [c["h1_held_out"]["cert"]["draw"]]:
        rec["n"], rec["canary_sha256"] = 8, eight
    c["canaries"]["canary_sha256"] = eight
    for a in score.ARMS:
        c[a]["canary_sha256"] = eight
    c["h1_held_out"]["cert"]["canary_sha256"] = eight
    card = _score(c, "beacon_draw")
    assert card["gates"]["K5"]["fired"] is True and any("n=8" in d for d in card["gates"]["K5"]["detail"])
    c = _experiment("beacon_draw")
    c["Q4"]["canary_sha256"], c["Q4"]["draw"] = "f" * 64, dict(c["Q4"]["draw"], beacon="e" * 64)
    detail = _score(c, "beacon_draw")["gates"]["K5"]["detail"]
    assert any("the Q4 cert grades canary set" in d for d in detail) and any("the Q4 cert carries a different draw" in d for d in detail)
    c = _experiment("beacon_draw")
    c["h1_held_out"]["unpreregistered"] = True
    assert any("not marked preregistered" in d for d in _score(c, "beacon_draw")["gates"]["K5"]["detail"])
    c = _experiment("deploy_quant")
    c["R"]["canary_sha256"] = "f" * 64
    assert score.score(c, "deploy_quant")["gates"]["provenance"]["fired"] is True


def test_a_lying_beacon_fires_k5_and_the_expected_beacon_is_checked():
    c = _experiment("beacon_draw")
    lying = copy.deepcopy(c)
    lying["canaries"]["draw"]["beacon"] = "b" * 64
    assert any("does not produce the canaries" in d for d in _score(lying, "beacon_draw")["gates"]["K5"]["detail"])
    card = score.score(c, "beacon_draw", expect_beacon="c" * 64)
    assert card["counts_as_result"] is False and any("ANCHORED" in d for d in card["gates"]["K5"]["detail"])
    card = score.score(c, "beacon_draw", expect_beacon=BEACON.upper())
    assert card["counts_as_result"] is True and not any("beacon clause was not checked" in n for n in card["notes"])


# ----------------------------------------------------------------------------- every check refuses on its own named ground

REFUSALS = [
    # (id, prereg, edit, the words the refusal must carry)
    ("sanity.n_canaries", "beacon_draw", _put("sanity.n_canaries", 47), "sanity.n_canaries is 47"),
    ("sanity.n_canaries", "deploy_quant", _put("sanity.n_canaries", 47), "sanity.n_canaries is 47"),
    ("canaries.n", "beacon_draw", _put("canaries.n", 47), "canaries.n is 47"),
    ("canaries.n", "deploy_quant", _put("canaries.n", 47), "canaries.n is 47"),
    ("pool", "beacon_draw", _put("canaries.draw.pool_sha256", "0" * 64), "the draw's pool hash is not this checkout's pool"),
    ("canary-hash-vs-set", "deploy_quant", _put("canaries.canary_sha256", "0" * 64), "canaries.canary_sha256 is not the set the PREREG names"),
    ("canary-hash-vs-set", "beacon_draw", _put("canaries.canary_sha256", "0" * 64), "canaries.canary_sha256 is not the set the PREREG names"),
    ("draw-hash-vs-canary-hash", "beacon_draw", _put("canaries.canary_sha256", "0" * 64), "the draw record's canary hash is not the certs' canary hash"),
    ("canary-hash-absent", "deploy_quant", _drop("canaries.canary_sha256"), "canaries.canary_sha256 is absent"),
    ("canary-hash-absent", "beacon_draw", _drop("canaries.canary_sha256"), "canaries.canary_sha256 is absent"),
    ("draw-record-under-hand-set", "deploy_quant", _put("canaries.draw", _certs(BEACON_CERTS)["canaries"]["draw"]), "the certs carry a draw record"),
    ("k1-threshold", "beacon_draw", _put("k1.threshold_nats", 0.02), "the certs' k1 record"),
    ("k1-threshold", "deploy_quant", _put("k1.threshold_nats", 0.02), "the certs' k1 record"),
    ("arm-n-items", "beacon_draw", _put("Q4.n_items", 47), "the Q4 cert grades 47 items"),
    ("held-out-set", "deploy_quant", _put("h1_held_out.cert.canary_sha256", "f" * 64), "the h1_held_out cert grades canary set"),
    ("held-out-draw", "beacon_draw", _put("h1_held_out.cert.draw", None), "the h1_held_out cert carries a different draw record"),
    # S3: when K1 did not fire, the runner writes every arm and the held-out cert
    ("A2-absent", "beacon_draw", _drop("A2"), "the A2 cert is absent"),
    ("Q4-absent", "deploy_quant", _drop("Q4"), "the Q4 cert is absent"),
    ("Q8-absent", "beacon_draw", _drop("Q8"), "the Q8 cert is absent"),
    ("R-absent", "deploy_quant", _drop("R"), "the R cert is absent"),
    ("A2-absent", "deploy_quant", _drop("A2"), "the A2 cert is absent"),
    ("held-out-absent", "deploy_quant", _drop("h1_held_out"), "h1_held_out.cert is absent"),
    ("held-out-cert-absent", "beacon_draw", _drop("h1_held_out.cert"), "h1_held_out.cert is absent"),
    # S4: is_the_experiment is never trusted alone
    ("tag", "beacon_draw", _put("tag", "_dryrun"), "tag is '_dryrun'"),
    ("tag", "deploy_quant", _put("tag", "_dryrun"), "tag is '_dryrun'"),
    ("smoke", "beacon_draw", _put("smoke", True), "smoke is true"),
    ("smoke", "deploy_quant", _put("smoke", True), "smoke is true"),
    ("git-head-null", "beacon_draw", _put("provenance.git_head", None), "provenance.git_head is missing"),
    ("git-head-absent", "deploy_quant", _drop("provenance.git_head"), "provenance.git_head is missing"),
    ("git-dirty-tracked", "beacon_draw", _put("provenance.git_dirty_tracked", True), "provenance.git_dirty_tracked is true"),
    ("git-dirty-tracked", "deploy_quant", _put("provenance.git_dirty_tracked", True), "provenance.git_dirty_tracked is true"),
    ("blob-not-sealed", "beacon_draw", _put("provenance.prereg_blob_is_sealed", False), "provenance.prereg_blob_is_sealed is not true"),
    ("blob-not-sealed", "deploy_quant", _put("provenance.prereg_blob_is_sealed", False), "provenance.prereg_blob_is_sealed is not true"),
    # S7: malformed certs are problems, never exceptions; a floor is a mean absolute distance
    ("canaries-list", "deploy_quant", _put("canaries", [1]), "canaries is not an object"),
    ("canaries-list", "beacon_draw", _put("canaries", [1]), "canaries is not an object"),
    ("sanity-list", "deploy_quant", _put("sanity", [1]), "sanity is not an object"),
    ("sanity-list", "beacon_draw", _put("sanity", [1]), "sanity is not an object"),
    ("provenance-list", "deploy_quant", _put("provenance", [1]), "provenance is not an object"),
    ("k1-list", "beacon_draw", _put("k1", [1]), "k1 is not an object"),
    ("held-out-list", "beacon_draw", _put("h1_held_out", [1]), "h1_held_out is not an object"),
    ("arm-list", "deploy_quant", _put("Q4", [1]), "the Q4 cert is not an object"),
    ("arm-list", "beacon_draw", _put("R", "a string"), "the R cert is not an object"),
    ("floor-negative", "beacon_draw", _put("sanity.null_floor_nats", -1.0), "cannot be negative"),
    ("floor-negative", "deploy_quant", _put("sanity.null_floor_nats", -1e-6), "cannot be negative"),
    ("floor-nan", "beacon_draw", _put("sanity.null_floor_nats", float("nan")), "not a finite number"),
    ("floor-inf", "deploy_quant", _put("sanity.null_floor_nats", float("inf")), "not a finite number"),
    ("floor-absent", "beacon_draw", _drop("sanity.null_floor_nats"), "no numeric sanity.null_floor_nats"),
]


@pytest.mark.parametrize("prereg,edit,needle", [r[1:] for r in REFUSALS], ids=[f"{r[1]}-{r[0]}" for r in REFUSALS])
def test_each_check_refuses_on_its_own_named_ground(prereg, edit, needle):
    c = _experiment(prereg)
    assert _score(c, prereg)["counts_as_result"] is True           # the unedited shape is the experiment
    edit(c)
    card = _score(c, prereg)                                        # never raises
    assert card["counts_as_result"] is False
    assert any(needle in d for d in _detail(card, prereg)), _detail(card, prereg)
    assert "INSTRUMENT CHECK; " in card["run_reading"]


def test_the_runner_shape_of_the_experiment_counts_only_with_its_optional_fields_absent_or_clean():
    for prereg in ("beacon_draw", "deploy_quant"):
        c = _experiment(prereg)
        del c["provenance"]["git_dirty_tracked"], c["provenance"]["prereg_blob_is_sealed"]
        assert _score(c, prereg)["counts_as_result"] is True           # each S4 field is checked when present
        c["tag"] = None
        assert _score(c, prereg)["counts_as_result"] is True


def test_the_committed_hand_set_certs_predate_the_canaries_block_and_so_cannot_be_the_experiment():
    c = _certs(HAND_CERTS)
    c["is_the_experiment"], c["model"], c["tag"] = True, score.MODEL, ""
    card = score.score(c, "deploy_quant")
    assert card["counts_as_result"] is False
    assert "canaries.canary_sha256 is absent: nothing in the certs names the set they graded" in card["gates"]["provenance"]["detail"]


def test_a_floor_that_is_not_finite_leaves_k1_not_evaluable():
    c = _experiment("beacon_draw")
    _put("sanity.null_floor_nats", float("nan"))(c)
    card = _score(c, "beacon_draw")
    assert card["gates"]["K1"]["fired"] is None and card["gates"]["K1"]["detail"] == "not evaluable: no floor"


# ----------------------------------------------------------------------------- H5 and H6 read only what they are bound to

def test_h5_needs_a_valid_hand_set_experiment_on_the_same_model_and_device():
    c = _experiment("beacon_draw")
    assert score.score(c, "beacon_draw", hand_set=_certs(HAND_CERTS))["hypotheses"]["H5"]["status"] == "PENDING"
    hand = _experiment("deploy_quant")
    h5 = score.score(c, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]
    assert h5["status"] == "HELD" and h5["clauses"][-1]["observed"] == pytest.approx(0.34658 / 0.42703, rel=1e-3)
    other_model = copy.deepcopy(c)
    other_model["model"] = "Qwen/Qwen2.5-7B"                  # the hand set is valid; the models differ
    assert score.score(other_model, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]["status"] == "PENDING"
    no_device, hand_no_device = copy.deepcopy(c), copy.deepcopy(hand)
    no_device["provenance"]["cuda_device"] = hand_no_device["provenance"]["cuda_device"] = None
    assert score.score(no_device, "beacon_draw", hand_set=hand_no_device)["hypotheses"]["H5"]["status"] == "PENDING"
    hand["provenance"]["cuda_device"] = "another device"
    assert score.score(c, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]["status"] == "PENDING"


def test_h5_is_pending_when_the_hand_set_experiment_is_valid_but_its_k1_fired():
    hand = _set(_experiment("deploy_quant"), floor=0.05)
    for a in score.ARMS:
        del hand[a]
    del hand["h1_held_out"]
    hand["verdict"] = "INCONCLUSIVE"                          # the runner's K1 branch, as run_deploy_quant.py writes it
    assert score.score(hand, "deploy_quant")["counts_as_result"] is True
    h5 = score.score(_experiment("beacon_draw"), "beacon_draw", hand_set=hand)["hypotheses"]["H5"]
    assert h5["status"] == "PENDING" and h5["clauses"] == [] and "K1 fired" in h5["reason"]


def test_an_h5_clause_whose_reading_does_not_exist_is_not_evaluable_not_failed():
    c = _experiment("beacon_draw")
    del c["Q8"]["distance"]["verdict"]
    h5 = score.score(c, "beacon_draw", hand_set=_experiment("deploy_quant"))["hypotheses"]["H5"]
    assert h5["clauses"][2]["clause"] == "Q8 DRIFT on both" and h5["clauses"][2]["holds"] is None
    assert h5["status"] == "NOT_EVALUABLE"


def test_h6_reads_a_record_styxx_portability_wrote_over_these_certs_and_refuses_one_it_did_not():
    from styxx import portability as pt
    c = _experiment("beacon_draw")
    other = copy.deepcopy(c)
    for a in score.ARMS:
        other[a]["digest"] = hashlib.sha256(f"the second machine's {a}".encode()).hexdigest()
    rec = pt.compare([c, other], ["here", "there"])
    h6 = score.score(c, "beacon_draw", portability=rec)["hypotheses"]["H6"]
    assert h6["status"] == "HELD" and h6["portability_digest"] == rec["digest"]

    def reason(certs, record):
        h6 = score.score(certs, "beacon_draw", portability=record)["hypotheses"]["H6"]
        assert h6["status"] == "PENDING" and h6["clauses"] == []
        assert h6["reason"].startswith(score.PORTABILITY_UNBOUND + ": ")
        return h6["reason"][len(score.PORTABILITY_UNBOUND) + 2:]

    elsewhere = copy.deepcopy(c)
    elsewhere["Q4"]["note"] = "another run's Q4 cert"
    assert reason(_reseal_arms(elsewhere), rec) == "none of its input_cert_digests is these certs' arm digests"
    pasted = copy.deepcopy(c)
    pasted["Q4"]["digest"] = "0" * 64
    assert reason(pasted, rec) == "these certs' Q4 cert digest does not re-derive from that cert's body"
    forged = copy.deepcopy(rec)
    forged["reading"] = "every verdict and every compared number survive the move"
    assert reason(c, forged) == "its digest does not re-derive from its body"
    no_digest = copy.deepcopy(c)
    del no_digest["R"]["digest"]
    assert reason(no_digest, rec) == "these certs carry no R cert digest"
    armless = _seal_portability({k: v for k, v in rec.items() if k not in ("arms", "digest")})
    assert reason(c, armless) == "it compares no arms"
    two_lines = {"verdicts": "AGREE", "arms": {a: {"numbers": {"mean_abs_nats": {"max_abs_diff": 0}}} for a in ("Q4", "Q8", "R")}}
    assert reason(c, two_lines) == "its digest does not re-derive from its body"
    committed = json.load(open(os.path.join(CK, "portability_smollm_quant_two_machines_2026_09_13_v1.json"), encoding="utf-8"))
    assert reason(c, committed) == "these certs carry no int8 cert digest"      # it binds the SmolLM certs, not these


def _h6_reason(certs, record):
    h6 = score.score(certs, "beacon_draw", portability=record)["hypotheses"]["H6"]
    if h6["status"] != "PENDING":
        return h6["status"]
    assert h6["clauses"] == [] and h6["reason"].startswith(score.PORTABILITY_UNBOUND + ": ")
    return h6["reason"][len(score.PORTABILITY_UNBOUND) + 2:]


def test_h6_refuses_a_record_whose_binding_rests_on_digest_fields_pasted_onto_other_bytes():
    """The repair round's probe: a record over two unrelated certs, with the first one's arm digests pasted onto
    these certs (bodies unchanged), read HELD. Each cert digest is now re-derived from its body."""
    from styxx import portability as pt
    c = _experiment("beacon_draw")
    x, y = copy.deepcopy(c), copy.deepcopy(c)
    for a in score.ARMS:
        x[a]["digest"] = hashlib.sha256(f"x{a}".encode()).hexdigest()
        y[a]["digest"] = hashlib.sha256(f"y{a}".encode()).hexdigest()
        x[a]["distance"]["mean_abs_nats"] = y[a]["distance"]["mean_abs_nats"] = 9.0
    rec = pt.compare([x, y], ["x", "y"])
    forged = copy.deepcopy(c)
    for a in score.ARMS:
        forged[a]["digest"] = x[a]["digest"]
    assert _h6_reason(forged, rec) == "these certs' A2 cert digest does not re-derive from that cert's body"
    # the converse: these certs' true digests pasted onto other bodies before compare — the digests bind, the columns do not
    x2 = copy.deepcopy(x)
    for a in score.ARMS:
        x2[a]["digest"] = c[a]["digest"]
    assert _h6_reason(c, pt.compare([x2, y], ["x", "y"])) == "its A2 mean_abs_nats values do not record these certs' A2 mean for their machine"
    x3 = copy.deepcopy(c)
    for a in score.ARMS:
        x3[a]["distance"]["verdict"] = "DRIFT" if c[a]["distance"]["verdict"] == "SAME" else "SAME"
    assert _h6_reason(c, pt.compare([x3, y], ["x", "y"])) == "its A2 verdicts do not record these certs' A2 verdict for their machine"


def test_h6_refuses_these_certs_compared_with_themselves_and_reads_an_honest_second_machine():
    from styxx import portability as pt
    c = _experiment("beacon_draw")
    assert _h6_reason(c, pt.compare([c, c], ["here", "here-again"])) == \
        "it lists these certs as more than one machine: nothing in it shows a second machine"
    there = copy.deepcopy(c)
    for a, mv in (("A2", 0.0), ("Q4", 0.03), ("Q8", 0.02), ("R", 0.05)):
        there[a]["distance"]["mean_abs_nats"] += mv
    rec = pt.compare([_reseal_arms(there), c], ["there", "here"])          # these certs as the second column
    h6 = score.score(c, "beacon_draw", portability=rec)["hypotheses"]["H6"]
    assert h6["status"] == "HELD", h6
    assert [x["observed"] for x in h6["clauses"][1:]] == pytest.approx([0.03, 0.02, 0.05])
    bigger = copy.deepcopy(there)
    bigger["R"]["distance"]["mean_abs_nats"] += 0.2
    h6 = score.score(c, "beacon_draw", portability=pt.compare([c, _reseal_arms(bigger)], ["here", "there"]))["hypotheses"]["H6"]
    assert h6["status"] == "FAILED" and [x["holds"] for x in h6["clauses"]] == [True, True, True, False]


def test_h6_refuses_a_resealed_record_whose_summary_is_not_its_own_columns():
    """A record's digest re-derives from any body its writer hashes; the fields H6 reads must follow from its columns."""
    from styxx import portability as pt
    c = _experiment("beacon_draw")
    there = copy.deepcopy(c)
    there["Q4"]["distance"]["mean_abs_nats"] += 0.5
    rec = pt.compare([c, _reseal_arms(there)], ["here", "there"])
    assert _h6_reason(c, rec) == "FAILED"                                                  # 0.5 > 0.10, honestly
    small = copy.deepcopy(rec)
    small["arms"]["Q4"]["numbers"]["mean_abs_nats"]["max_abs_diff"] = 0.01
    assert _h6_reason(c, _seal_portability(small)) == "its Q4 max_abs_diff is not the spread of its own values"
    flip = copy.deepcopy(c)
    flip["Q8"]["distance"]["verdict"] = "DRIFT" if c["Q8"]["distance"]["verdict"] == "SAME" else "SAME"
    rec = pt.compare([c, _reseal_arms(flip)], ["here", "there"])
    assert rec["verdicts"] == "FLIP" and _h6_reason(c, rec) == "FAILED"
    agree = copy.deepcopy(rec)
    agree["verdicts"], agree["verdict_flips"] = "AGREE", []
    assert _h6_reason(c, _seal_portability(agree)) == "its verdicts field is not what its per-arm verdicts say"
    three = copy.deepcopy(pt.compare([c, _reseal_arms(there)], ["here", "there"]))
    three["n_machines"] = 3
    assert _h6_reason(c, _seal_portability(three)) == "it does not list one input per machine for at least two machines"
    one = copy.deepcopy(three)
    one["n_machines"], one["input_cert_digests"] = 1, one["input_cert_digests"][:1]
    for entry in one["arms"].values():
        entry["verdicts"] = entry["verdicts"][:1]
        leaf = entry["numbers"]["mean_abs_nats"]
        leaf["values"], leaf["max_abs_diff"] = leaf["values"][:1], 0.0
    one["verdicts"] = "AGREE"
    assert _h6_reason(c, _seal_portability(one)) == "it does not list one input per machine for at least two machines"
    ragged = copy.deepcopy(three)
    ragged["n_machines"], ragged["input_cert_digests"] = 2, [ragged["input_cert_digests"][0], "not a list"]
    assert _h6_reason(c, _seal_portability(ragged)) == "it does not list one input per machine for at least two machines"


# ----------------------------------------------------------------------------- the committed checks, the CLI, the published mutations

@pytest.mark.parametrize("prereg,certs,card_file,statuses,gate", [
    ("beacon_draw", BEACON_CERTS, "beacon_draw_scorecard_v2_dryrun_qwen0.5b.json",
     {"H1": "HELD", "H2": "HELD", "H3": "HELD", "H4": "HELD", "H5": "PENDING", "H6": "PENDING"}, "K5"),
    ("deploy_quant", HAND_CERTS, "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json",
     {"H1": "FAILED", "H2": "FAILED", "H3": "HELD", "H4": "HELD"}, "provenance"),
])
def test_the_committed_instrument_checks_read_as_committed(prereg, certs, card_file, statuses, gate):
    card = score.score(_certs(certs), prereg)
    assert card["counts_as_result"] is False and card["gates"][gate]["fired"] is True
    assert {h: v["status"] for h, v in card["hypotheses"].items()} == statuses
    committed = json.load(open(os.path.join(CK, card_file), encoding="utf-8"))
    assert committed["schema"] == score.SCHEMA == "styxx.checksum/scorecard/v2"
    for k in ("hypotheses", "gates", "counts_as_result", "run_reading"):
        assert committed[k] == card[k], f"the committed scorecard's {k} is not what the scorer reads today"
    assert committed["certs_sha256"] == hashlib.sha256(open(os.path.join(CK, certs), "rb").read()).hexdigest()


@pytest.mark.parametrize("prereg,certs", [("beacon_draw", BEACON_CERTS), ("deploy_quant", HAND_CERTS)])
def test_the_cli_writes_a_scorecard_even_on_a_cp1252_console(tmp_path, prereg, certs):
    out = tmp_path / "card.json"
    env = dict(os.environ, PYTHONIOENCODING="cp1252")           # v1 died here on the deploy_quant path
    env.pop("PYTHONUTF8", None)
    r = subprocess.run([sys.executable, os.path.join(CK, "score.py"), "--prereg", prereg, os.path.join(CK, certs), "--out", str(out)],
                       capture_output=True, timeout=300, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    card = json.loads(out.read_text(encoding="utf-8"))
    assert card["schema"] == score.SCHEMA and card["certs_file"] == f"papers/checksum/{certs}" and len(card["certs_sha256"]) == 64


@pytest.mark.skipif(bool(os.environ.get("STYXX_SCORE_MUTANT")),
                    reason="inside score_mutations.py's own run the scorer copy is mutated on purpose; this test reads the published result")
def test_the_published_mutation_list_ran_on_this_scorer_and_these_tests_and_every_mutation_was_killed():
    mut = _load("score_mutations", "score_mutations.py")
    res = json.load(open(os.path.join(CK, "score_mutations_result.json"), encoding="utf-8"))
    ids = [m[0] for m in mut.MUTATIONS]
    assert len(set(ids)) == len(ids) and [m["id"] for m in res["mutations"]] == ids
    assert res["schema"] == mut.SCHEMA and res["baseline"]["passed"] is True
    assert res["score_py_sha256_lf"] == mut.lf_sha256(os.path.join(CK, "score.py")), "score.py changed after the published run"
    assert res["tests_sha256_lf"] == mut.lf_sha256(os.path.abspath(__file__)), "these tests changed after the published run"
    text = open(os.path.join(CK, "score.py"), encoding="utf-8").read()
    for (mid, _desc, old, new), row in zip(mut.MUTATIONS, res["mutations"]):
        assert text.count(old) == 1 and old != new, mid
        assert row["old_text_found_once"] is True and row["killed"] is True, row
    groups = {m.split("-")[0] for m in ids}
    assert {"v2", "S1", "S2", "S3", "S4", "S5", "S6", "S7"} <= groups
    assert sum(1 for m in ids if re.fullmatch(r"v2-\d\d", m)) == 28
