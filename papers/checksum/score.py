#!/usr/bin/env python3
"""score.py — the frozen PREREG's reading rules, executed by code, on a certs file the runner wrote.

    python papers/checksum/score.py --prereg beacon_draw papers/checksum/beacon_draw_certs.json \
        [--expect-beacon <64 hex>] [--hand-set-certs papers/checksum/deploy_quant_certs.json] \
        [--portability <styxx.portability v1 record>] [--out <scorecard.json>]
    python papers/checksum/score.py --prereg deploy_quant papers/checksum/deploy_quant_certs.json [--out ...]

What this is for. A PREREG freezes hypotheses with numeric bands and kill gates before the run. The
RESULT is supposed to read the run's numbers against those bands and nothing else — and the person
reading is the person who wanted the hypotheses to hold. This script reads them instead. The bands
below are copied from the two frozen documents (`PREREG_checksum_deploy_quant_2026_09_13.md`,
`PREREG_checksum_beacon_draw_2026_09_14.md`); `tests/test_checksum_score.py` parses every number out
of the frozen text and compares it with the value here, and pins every comparator (≤ vs <, inclusive
bands) with certs sitting exactly on each edge. The scorecard it writes (`styxx.checksum/scorecard/v2`)
carries, per hypothesis and per clause, the predicted band, the observed value and whether it holds;
per gate, whether it fired and on what; and whether the certs are the experiment at all. A RESULT
then swears to the scorecard, and a stranger re-runs this script on the certs to get the same card.

What the scorer re-derives rather than trusts (v2, after the red team of 2026-09-14 found v1 trusting
the certs' own flags): K1 from the recorded null floor, never from the runner's `k1.fired`; the sealed
PREREG digest, frozen here per PREREG and compared unconditionally with `provenance.prereg_blob_sha256`
(`--expect-blob` survives only as an assertion that must agree with it); for the beacon-draw PREREG,
the draw from the beacon and the committed pool, with n = 48; and for both, that every arm's cert and
the held-out cert grade the same 48-item set the PREREG names and carry the same draw record.

What it refuses to be: a judge of meaning. It reads the numbers as the PREREG wrote them, marks a
hypothesis whose inputs do not exist yet PENDING, marks a run that is not the experiment as an
instrument check (every reading still printed, none of it a result), and evaluates nothing when the
first kill gate fired, because the PREREG says so — a valid, sealed run whose K1 fired IS a result
(INCONCLUSIVE, titled with the kill), not an instrument check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCHEMA = "styxx.checksum/scorecard/v2"
MODEL = "Qwen/Qwen2.5-1.5B"
N_CANARIES = 48
ARMS = ("A2", "Q4", "Q8", "R")
# the hand-written set's hash, as the 2026-09-13 PREREG names it ("hash `71f8e5c9973c…`, the full hash is in every cert")
HAND_CANARY_SHA256 = "71f8e5c9973c6c9aa111d25a6c6ee8c8faf2b557ef6e97705aa02b34db868eac"

# The bands, verbatim from the frozen documents. tests/test_checksum_score.py parses each out of its text.
BANDS = {
    "deploy_quant": {
        "prereg": "PREREG_checksum_deploy_quant_2026_09_13.md",
        "sealed_blob": "b3b9871090fdcd4491cc4a3171c27e1468414409da9d3b66b4d3ff2fd4b6dd9e",   # SEALS_2026_09_13.md row 3
        "h1_floor_min_exclusive": 0.0, "h1_floor_max": 1e-3,          # "> 0 and ≤ 1e-3 nats/token"
        "h2_band": (0.01, 0.30), "h2_r_min": 0.95, "h2_top1_loss_max": 4,
        "h4_mean_min": 5.0, "h4_r_max": 0.3, "h4_hits_max": 2,
        "k1_floor": 1e-2, "k3_mean": 1.0, "k3_top1_loss": 12,
        "h1_rule": "CORRECTION_prereg_deploy_quant_H1_2026_09_13.md",
    },
    "beacon_draw": {
        "prereg": "PREREG_checksum_beacon_draw_2026_09_14.md",
        "sealed_blob": "d6a98261f44f31664cdedbc24769532d5a701bf46d2c8b9ce30cf6a227fb5519",   # SEALS_2026_09_13.md row 6
        "h1_floor_min_exclusive": None, "h1_floor_max": 1e-3,          # "≤ 1e-3 nats/token; zero is allowed"
        "h2_band": (0.05, 0.60), "h2_r_min": 0.90, "h2_top1_loss_max": 24,
        "h3_r_min": 0.98, "h3_top1_loss_max": 8,
        "h4_mean_min": 5.0, "h4_r_max": 0.6, "h4_hits_max": 4,
        "h5_ratio_band": (0.5, 2.0),                                    # "within a factor of two ..., either way"
        "h6_move_max": {"Q4": 0.10, "Q8": 0.10, "R": 0.22},
        "k1_floor": 1e-2, "k3_mean": 1.5, "k3_top1_loss": 36,
    },
}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _clause(name, predicted, observed, holds):
    return {"clause": name, "predicted": predicted, "observed": observed, "holds": bool(holds) if holds is not None else None}


def _status(clauses):
    hs = [c["holds"] for c in clauses]
    if any(h is None for h in hs):
        return "NOT_EVALUABLE"
    return "HELD" if all(hs) else "FAILED"


def _dist(certs, arm):
    d = (certs.get(arm) or {}).get("distance") if isinstance(certs.get(arm), dict) else None
    return d if isinstance(d, dict) else None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _set_problems(certs: dict, expected_hash: str | None, draw: dict | None) -> list[str]:
    """Every graded cert must grade the 48-item set the PREREG names, under the certs' own draw record."""
    out = []
    san = certs.get("sanity") or {}
    can = certs.get("canaries") if isinstance(certs.get("canaries"), dict) else {}
    if san.get("n_canaries") != N_CANARIES:
        out.append(f"sanity.n_canaries is {san.get('n_canaries')!r}; the PREREG's set has {N_CANARIES} items")
    if can.get("n") is not None and can.get("n") != N_CANARIES:
        out.append(f"canaries.n is {can.get('n')!r}; the PREREG's set has {N_CANARIES} items")
    if expected_hash is not None and can.get("canary_sha256") is not None and can.get("canary_sha256") != expected_hash:
        out.append("canaries.canary_sha256 is not the set the PREREG names")
    graded = [(arm, certs.get(arm)) for arm in ARMS if isinstance(certs.get(arm), dict)]
    ho = (certs.get("h1_held_out") or {}).get("cert") if isinstance(certs.get("h1_held_out"), dict) else None
    if isinstance(ho, dict):
        graded.append(("h1_held_out", ho))
    for name, c in graded:
        if expected_hash is not None and c.get("canary_sha256") != expected_hash:
            out.append(f"the {name} cert grades canary set {str(c.get('canary_sha256'))[:12]}…, not the set the PREREG names")
        if c.get("n_items") != N_CANARIES:
            out.append(f"the {name} cert grades {c.get('n_items')!r} items, not {N_CANARIES}")
        if (c.get("draw") or None) != (draw or None):
            out.append(f"the {name} cert carries a different draw record than the certs' canaries")
    return out


def score(certs: dict, prereg: str, expect_beacon: str | None = None, expect_blob: str | None = None,
          hand_set: dict | None = None, portability: dict | None = None) -> dict:
    b = BANDS[prereg]
    prov = certs.get("provenance") or {}
    sanity = certs.get("sanity") or {}
    hits = sanity.get("first_token_top1_hits") or {}
    loss = sanity.get("top1_loss_vs_A") or {}
    floor = _num(sanity.get("null_floor_nats"))
    k1_rec = certs.get("k1") if isinstance(certs.get("k1"), dict) else {}
    card = {"schema": SCHEMA, "prereg": b["prereg"], "prereg_named_by_certs": certs.get("prereg"),
            "sealed_blob": b["sealed_blob"], "prereg_blob_sha256": prov.get("prereg_blob_sha256"),
            "model": certs.get("model"), "device": certs.get("device"),
            "cuda_device": prov.get("cuda_device"), "git_head": prov.get("git_head"), "git_dirty": prov.get("git_dirty"),
            "is_the_experiment": certs.get("is_the_experiment") is True, "tag": certs.get("tag"), "smoke": certs.get("smoke"),
            "hypotheses": {}, "gates": {}, "notes": []}
    problems = []
    if certs.get("prereg") != b["prereg"]:
        problems.append(f"the certs name {certs.get('prereg')!r}, not {b['prereg']}")
    if prov.get("prereg_blob_sha256") != b["sealed_blob"]:
        problems.append(f"provenance.prereg_blob_sha256 is {str(prov.get('prereg_blob_sha256'))[:12]}…, not the sealed digest "
                        f"{b['sealed_blob'][:12]}… (the PREREG that governed the run is not the sealed text)")
    if expect_blob is not None and expect_blob.lower() != b["sealed_blob"]:
        problems.append("--expect-blob is not the digest this scorer froze for the PREREG; one of them is wrong")
    if certs.get("model") != MODEL:
        problems.append(f"model is {certs.get('model')!r}, the PREREG's is {MODEL}")
    if certs.get("is_the_experiment") is not True:
        problems.append("is_the_experiment is not true (a tag, a smoke, or another model)")
    if prereg == "deploy_quant" and (certs.get("canaries") or {}).get("draw"):
        problems.append("the certs carry a draw record; the 2026-09-13 PREREG froze the hand-written set (score it under --prereg beacon_draw)")

    # K1 — re-derived from the recorded floor; the runner's own k1 record is compared, never trusted
    k1_fired = floor is not None and floor > b["k1_floor"]
    if floor is None:
        problems.append("the certs carry no numeric sanity.null_floor_nats; K1 cannot be evaluated")
    if k1_rec and (bool(k1_rec.get("fired")) != k1_fired or k1_rec.get("threshold_nats") != b["k1_floor"]):
        problems.append(f"the certs' k1 record (fired={k1_rec.get('fired')!r}, threshold={k1_rec.get('threshold_nats')!r}) "
                        f"disagrees with their own floor under the PREREG's threshold {b['k1_floor']:g}")
    card["gates"]["K1"] = {"fired": k1_fired if floor is not None else None, "floor_nats": floor, "threshold_nats": b["k1_floor"],
                           "detail": ("the serving is not deterministic enough for this probe; nothing else is evaluated" if k1_fired
                                      else ("not evaluable: no floor" if floor is None else "not fired"))}

    if prereg == "beacon_draw":
        draw = (certs.get("canaries") or {}).get("draw") if isinstance(certs.get("canaries"), dict) else None
        k5, expected_hash = [], None
        if not isinstance(draw, dict):
            k5.append("the certs carry no draw record")
        else:
            try:
                from styxx import beacon as _beacon
                from styxx import checksum as ck
                if draw.get("pool_sha256") != _beacon.pool_sha256():
                    k5.append("the draw's pool hash is not this checkout's pool")
                if draw.get("n") != N_CANARIES:
                    k5.append(f"the draw record says n={draw.get('n')!r}; the PREREG draws {N_CANARIES}")
                items = _beacon.select(str(draw["beacon"]), int(draw["n"]))
                rederived = ck.canary_sha256(items)
                if rederived != draw.get("canary_sha256"):
                    k5.append("the beacon does not produce the canaries the draw record names")
                else:
                    expected_hash = rederived
                if draw.get("canary_sha256") != (certs.get("canaries") or {}).get("canary_sha256"):
                    k5.append("the draw record's canary hash is not the certs' canary hash")
            except Exception as e:  # noqa: BLE001
                k5.append(f"the draw record could not be re-derived: {e}")
            if expect_beacon and str(draw.get("beacon", "")).lower() != expect_beacon.lower():
                k5.append("the certs' beacon is not the beacon the ANCHORED line prints")
            if not expect_beacon:
                card["notes"].append("K5's beacon clause was not checked: pass --expect-beacon with the beacon `python -m styxx.clock verify` prints ANCHORED for this PREREG's digest")
        k5 += _set_problems(certs, expected_hash, draw if isinstance(draw, dict) else None)
        if isinstance(certs.get("h1_held_out"), dict) and certs["h1_held_out"].get("unpreregistered") is not False:
            k5.append("h1_held_out is not marked preregistered: these certs were not written under the beacon-draw PREREG's runner path")
        k5 += problems
        card["gates"]["K5"] = {"fired": bool(k5), "detail": k5 or "the draw re-derives; every cert grades the drawn 48; the certs are the experiment"}
        card["gates"]["K6"] = {"fired": None, "detail": "a precondition on the chain, evaluated by `python -m styxx.clock verify`, not offline; if the seal line is not ANCHORED the run did not start, and if it started K5 applies"}
        invalid = bool(k5)
    else:
        problems += _set_problems(certs, HAND_CANARY_SHA256, None)
        card["gates"]["provenance"] = {"fired": bool(problems), "detail": problems or "the certs are the experiment"}
        invalid = bool(problems)

    card["counts_as_result"] = card["is_the_experiment"] and not invalid
    if not card["counts_as_result"]:
        card["notes"].append("INSTRUMENT CHECK or invalid run: every reading below is printed, none is a result")
    prefix = "" if card["counts_as_result"] else "INSTRUMENT CHECK; "

    if k1_fired:
        for h in ("H1", "H2", "H3", "H4") + (("H5", "H6") if prereg == "beacon_draw" else ()):
            card["hypotheses"][h] = {"status": "NOT_EVALUATED", "clauses": [], "reason": "K1 fired"}
        for g in ("K2", "K3", "K4"):
            card["gates"][g] = {"fired": None, "detail": "not evaluated: K1 fired"}
        card["run_reading"] = prefix + "INCONCLUSIVE (K1)"
        return card

    dq4, dq8, dr = _dist(certs, "Q4"), _dist(certs, "Q8"), _dist(certs, "R")
    ho = ((certs.get("h1_held_out") or {}).get("cert") or {}).get("distance") or {} if isinstance(certs.get("h1_held_out"), dict) else {}
    # H1
    if prereg == "beacon_draw":
        c1 = _clause("worst-pairwise null floor ≤ 1e-3 nats/token (zero allowed)", "≤ 0.001", floor, floor <= b["h1_floor_max"] if floor is not None else None)
        c2 = _clause("A vs A′ reads SAME against the held-out floor", "SAME", ho.get("verdict"), ho.get("verdict") == "SAME" if ho else None)
        st, rule = _status([c1, c2]), None
        if c1["holds"] and floor is not None and floor > 0 and ho.get("verdict") == "INCONCLUSIVE":
            # CORRECTION_prereg_beacon_draw_2026_09_14 rule 3, in those words: neither HELD nor FAILED for H1 as written
            st, rule = "HELD on the floor, INCONCLUSIVE on the pair by construction", "CORRECTION_prereg_beacon_draw_2026_09_14.md rule 3"
        elif floor is not None and floor > 0 and ho.get("verdict") == "DRIFT":
            rule = ("CORRECTION_prereg_beacon_draw_2026_09_14.md rule 4: FAILED as written; the probe saw this rule read DRIFT "
                    "on identical weights at two of four nonzero scales")
        card["hypotheses"]["H1"] = {"status": st, "clauses": [c1, c2], "rule": rule,
                                    "held_out_floor_nats": (certs.get("h1_held_out") or {}).get("floor_nats"),
                                    "preregistered": (certs.get("h1_held_out") or {}).get("unpreregistered") is False}
    else:
        d2 = _dist(certs, "A2") or {}
        c1 = _clause("null floor > 0 and ≤ 1e-3 nats/token", "(0, 0.001]", floor,
                     (floor > b["h1_floor_min_exclusive"] and floor <= b["h1_floor_max"]) if floor is not None else None)
        c2 = _clause("A vs A′ reads SAME (against the worst-pairwise floor, as frozen)", "SAME", d2.get("verdict"),
                     d2.get("verdict") == "SAME" if d2 else None)
        st = _status([c1, c2])
        if c1["holds"] and d2.get("verdict") == "INCONCLUSIVE":
            st = "HELD on the floor, INCONCLUSIVE on the pair by construction"   # CORRECTION rule 2, in those words
        card["hypotheses"]["H1"] = {"status": st, "clauses": [c1, c2], "rule": b["h1_rule"],
                                    "held_out_reading_unpreregistered": {"verdict": ho.get("verdict"),
                                                                         "floor_nats": (certs.get("h1_held_out") or {}).get("floor_nats"),
                                                                         "decides": "nothing in this run (CORRECTION rule 3)"}}
    # H2
    lo, hi = b["h2_band"]
    m4 = _num(dq4.get("mean_abs_nats")) if dq4 else None
    r4 = _num(dq4.get("rdm_r")) if dq4 else None
    l4 = _num(loss.get("Q4"))
    cl = [_clause("A vs Q4 reads DRIFT", "DRIFT", dq4.get("verdict") if dq4 else None, dq4.get("verdict") == "DRIFT" if dq4 else None),
          _clause("mean |Δ log-prob| in the band (inclusive)", f"[{lo}, {hi}]", m4, lo <= m4 <= hi if m4 is not None else None),
          _clause("geometry r ≥ minimum", f"≥ {b['h2_r_min']}", r4, r4 >= b["h2_r_min"] if r4 is not None else None),
          _clause("argmax lost on at most N of 48", f"≤ {b['h2_top1_loss_max']}", l4, l4 <= b["h2_top1_loss_max"] if l4 is not None else None)]
    card["hypotheses"]["H2"] = {"status": _status(cl), "clauses": cl}
    # H3
    m8 = _num(dq8.get("mean_abs_nats")) if dq8 else None
    r8 = _num(dq8.get("rdm_r")) if dq8 else None
    l8 = _num(loss.get("Q8"))
    cl = [_clause("A vs Q8 reads DRIFT", "DRIFT", dq8.get("verdict") if dq8 else None, dq8.get("verdict") == "DRIFT" if dq8 else None),
          _clause("Q8 mean |Δ log-prob| below Q4's", f"< {m4}", m8, m8 < m4 if (m8 is not None and m4 is not None) else None)]
    if prereg == "beacon_draw":
        cl += [_clause("geometry r ≥ minimum", f"≥ {b['h3_r_min']}", r8, r8 >= b["h3_r_min"] if r8 is not None else None),
               _clause("argmax lost on at most N of 48", f"≤ {b['h3_top1_loss_max']}", l8, l8 <= b["h3_top1_loss_max"] if l8 is not None else None)]
    card["hypotheses"]["H3"] = {"status": _status(cl), "clauses": cl}
    # H4
    mr = _num(dr.get("mean_abs_nats")) if dr else None
    rr = _num(dr.get("rdm_r")) if dr else None
    hr = _num(hits.get("R"))
    cl = [_clause("A vs R mean |Δ log-prob| above minimum", f"> {b['h4_mean_min']}", mr, mr > b["h4_mean_min"] if mr is not None else None),
          _clause("geometry r below maximum", f"< {b['h4_r_max']}", rr, rr < b["h4_r_max"] if rr is not None else None),
          _clause("top-1 hits at most N of 48", f"≤ {b['h4_hits_max']}", hr, hr <= b["h4_hits_max"] if hr is not None else None)]
    card["hypotheses"]["H4"] = {"status": _status(cl), "clauses": cl}
    # K2-K4
    card["gates"]["K2"] = {"fired": dq4.get("verdict") == "SAME" if dq4 else None,
                           "detail": "A vs Q4 reads SAME: the canary set cannot see a deployed quantization" if dq4 and dq4.get("verdict") == "SAME" else "not fired"}
    k3 = (m4 is not None and m4 > b["k3_mean"]) or (l4 is not None and l4 > b["k3_top1_loss"])
    card["gates"]["K3"] = {"fired": bool(k3) if (m4 is not None or l4 is not None) else None,
                           "thresholds": {"mean_nats": b["k3_mean"], "top1_loss": b["k3_top1_loss"]}, "observed": {"mean_nats": m4, "top1_loss": l4},
                           "detail": "the quantization pipeline, not the model, is suspect" if k3 else "not fired"}
    k4 = mr is not None and lo <= mr <= hi
    card["gates"]["K4"] = {"fired": bool(k4) if mr is not None else None,
                           "reading": "'inside H2's band' is read as A vs R's mean |Δ log-prob| within H2's mean band, inclusive",
                           "detail": "A vs R inside H2's band: the metric is broken" if k4 else "not fired"}

    if prereg == "beacon_draw":
        # H5 — the drawn set against the hand set's sealed run, same model and machine
        if not hand_set:
            card["hypotheses"]["H5"] = {"status": "PENDING", "clauses": [], "reason": "the 2026-09-13 experiment's certs were not given (--hand-set-certs)"}
        else:
            hand_card = score(hand_set, "deploy_quant")
            hp = hand_set.get("provenance") or {}
            same_machine = hp.get("cuda_device") is not None and hp.get("cuda_device") == prov.get("cuda_device")
            if not hand_card["counts_as_result"] or hand_set.get("model") != certs.get("model") or not same_machine:
                card["hypotheses"]["H5"] = {"status": "PENDING", "clauses": [],
                                            "reason": "the hand-set certs are not a valid 2026-09-13 experiment on the same model and device"
                                                      + (f" ({'; '.join(hand_card['gates']['provenance']['detail'])})" if hand_card["gates"]["provenance"]["fired"] else "")}
            else:
                hho = ((hand_set.get("h1_held_out") or {}).get("cert") or {}).get("distance") or {}
                cl = [_clause("A′ SAME under the held-out reading on both", "SAME/SAME", [ho.get("verdict"), hho.get("verdict")],
                              ho.get("verdict") == "SAME" and hho.get("verdict") == "SAME")]
                for arm in ("Q4", "Q8", "R"):
                    a, h = _dist(certs, arm) or {}, _dist(hand_set, arm) or {}
                    cl.append(_clause(f"{arm} DRIFT on both", "DRIFT/DRIFT", [a.get("verdict"), h.get("verdict")], a.get("verdict") == "DRIFT" and h.get("verdict") == "DRIFT"))
                hm4 = _num((_dist(hand_set, "Q4") or {}).get("mean_abs_nats"))
                ratio = (m4 / hm4) if (m4 is not None and hm4) else None
                r0, r1 = b["h5_ratio_band"]
                cl.append(_clause("drawn-set NF4 mean within a factor of two of the hand set's, either way", f"ratio in [{r0}, {r1}]", ratio,
                                  r0 <= ratio <= r1 if ratio is not None else None))
                card["hypotheses"]["H5"] = {"status": _status(cl), "clauses": cl, "hand_set_git_head": hp.get("git_head"),
                                            "same_machine_reading": "the same cuda_device name — a device model, not a machine identity"}
        # H6 — portability under this beacon
        if not portability:
            card["hypotheses"]["H6"] = {"status": "PENDING", "clauses": [], "reason": "no second machine's run was given (--portability)"}
        else:
            cl = [_clause("verdicts AGREE on every arm", "AGREE", portability.get("verdicts"), portability.get("verdicts") == "AGREE")]
            arms = portability.get("arms") or {}
            for arm, mx in b["h6_move_max"].items():
                mv = _num(((arms.get(arm) or {}).get("numbers") or {}).get("mean_abs_nats", {}).get("max_abs_diff"))
                cl.append(_clause(f"{arm} mean |Δ log-prob| moves by at most {mx}", f"≤ {mx}", mv, mv <= mx if mv is not None else None))
            card["hypotheses"]["H6"] = {"status": _status(cl), "clauses": cl, "portability_digest": portability.get("digest")}

    fired = [g for g, v in card["gates"].items() if v.get("fired")]
    card["run_reading"] = prefix + (f"gates fired: {', '.join(fired)}" if fired else "no gate fired") + "; " + \
        ", ".join(f"{h} {v['status']}" for h, v in card["hypotheses"].items())
    return card


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):        # the clause names carry ≤, Δ, ′: never let a cp1252 console kill the run
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("certs")
    ap.add_argument("--prereg", choices=sorted(BANDS), required=True)
    ap.add_argument("--expect-beacon", default=None)
    ap.add_argument("--expect-blob", default=None, help="optional: asserted equal to the sealed digest this scorer froze")
    ap.add_argument("--hand-set-certs", default=None)
    ap.add_argument("--portability", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    certs = json.load(open(a.certs, encoding="utf-8"))
    hand = json.load(open(a.hand_set_certs, encoding="utf-8")) if a.hand_set_certs else None
    port = json.load(open(a.portability, encoding="utf-8")) if a.portability else None
    card = score(certs, a.prereg, a.expect_beacon, a.expect_blob, hand, port)
    card["certs_file"] = os.path.relpath(os.path.abspath(a.certs), ROOT).replace("\\", "/")
    card["certs_sha256"] = _sha(a.certs)
    print(f"scorecard for {card['certs_file']} under {card['prereg']}")
    print(f"  experiment: {card['is_the_experiment']}   counts as a result: {card['counts_as_result']}")
    for h, v in card["hypotheses"].items():
        print(f"  {h}: {v['status']}")
        for c in v.get("clauses", []):
            print(f"      {'ok ' if c['holds'] else ('-- ' if c['holds'] is None else 'NO ')} {c['clause']}: predicted {c['predicted']}, observed {c['observed']}")
        if v.get("reason"):
            print(f"      ({v['reason']})")
    for g, v in card["gates"].items():
        print(f"  {g}: {'FIRED' if v.get('fired') else ('not evaluated' if v.get('fired') is None else 'not fired')} — {v.get('detail')}")
    for n in card["notes"]:
        print(f"  note: {n}")
    print(f"  reading: {card['run_reading']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(card, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
