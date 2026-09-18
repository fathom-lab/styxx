"""CALIB-1 scorer — applies the frozen gates to whatever `calib1_ask.ts` recorded.

    python papers/closed-model-frontier/calib1_score.py               # reads calib1_raw.json
    python papers/closed-model-frontier/calib1_score.py --raw X.json --out Y.json

This program never calls Jev and never chooses anything a reader cannot re-derive
from `calib1_raw.json`. That is the point of splitting it from the asker: the
answers are recorded once and scored in the open, as many times as anyone likes.

Prereg: PREREG_calib1_jev_2026_09_18.md, frozen at sha256 7d550cd5... and carrying
Amendment A, appended before any call was made; the file now hashes to 8398db36... A raw file whose prereg hash does not
match the frozen one is refused, not scored.

The gates are G-C1-1 through G-C1-6 and they are implemented here in the order
the preregistration states them. Two of them can fail in a way that stops the
run from shipping anything, and both say so in their own words rather than
through an exit code.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v14_gates import bucket                                   # noqa: E402

PREREG_SHA256_AT_FREEZE = "7d550cd50473c6642770662149553eaf3db1e1a52308e8d80e1d518b3ad94aaa"
#: Amendment A was appended before the first call; see A6. Both are published.
PREREG_SHA256_FROZEN = "8398db36a7633fd980dd62dce8ed9d9a663eb99d472a5a445657b89f10e6296f"

#: Prediction 1 / G-C1-2.
SEPARATION_BAR = 0.35
#: Prediction 3. The run predicts ECE will be WORSE than this.
ECE_PREDICTED_FLOOR = 0.10
#: Prediction 4. The run predicts at least one spread WIDER than this.
DETERMINISM_PREDICTED_SPREAD = 0.05
#: G-C1-3: a bin with fewer than this many items is reported empty, not averaged.
MIN_BIN = 3
#: The interval-width test in the prereg's population section, in points.
INTERVAL_WIDTH_BAR = 30.0
#: G-C1-4 is computed over exactly this many items, lowest `id` first (Amendment A3).
DETERMINISM_N = 20

NEGATIVE_CODES = {
    "runtime_behaviour",
    "prose_or_documentation",
    "other_refers_to_different_change",
}
EXCLUDED_CODES = {"ambiguous_scope"}


# --------------------------------------------------------------------------- #
# small statistics, written out rather than imported, so the arithmetic is
# readable next to the claim it supports.
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes in n trials."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def concordance(pos: list[float], neg: list[float]) -> tuple[int, int]:
    """Pairs where a POSITIVE outranks a NEGATIVE, ties counted as half.

    Returned doubled (wins*2, pairs*2) so a tie is an integer and the result can
    go straight into `wilson` without rounding a half away.
    """
    wins2 = 0
    for a in pos:
        for b in neg:
            wins2 += 2 if a > b else (1 if a == b else 0)
    return wins2, 2 * len(pos) * len(neg)


def ece(pairs: list[tuple[float, int]], bins: int = 5, min_bin: int = MIN_BIN) -> dict:
    """Expected calibration error over equal-width bins.

    A bin holding fewer than `min_bin` items is reported as empty and excluded
    from the average rather than contributing a one- or two-item "accuracy",
    which is what G-C1-3 asks for. Excluding bins changes the number, so the
    weight that was dropped is published beside it.
    """
    edges = [i / bins for i in range(bins + 1)]
    rows, used_n, total_n, acc = [], 0, len(pairs), 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        members = [
            (p, y) for (p, y) in pairs
            if (p >= lo and p < hi) or (i == bins - 1 and p == hi)
        ]
        row = {"lo": lo, "hi": hi, "n": len(members)}
        if len(members) < min_bin:
            row.update({"reported": "empty", "confidence": None, "accuracy": None})
        else:
            conf = sum(p for p, _ in members) / len(members)
            accy = sum(y for _, y in members) / len(members)
            row.update({"reported": "scored", "confidence": conf, "accuracy": accy})
            acc += len(members) * abs(conf - accy)
            used_n += len(members)
        rows.append(row)
    return {
        "bins": rows,
        "ece": (acc / used_n) if used_n else None,
        "items_scored": used_n,
        "items_total": total_n,
        "items_in_empty_bins": total_n - used_n,
    }


# --------------------------------------------------------------------------- #
# population
# --------------------------------------------------------------------------- #

def classify(item: dict) -> str:
    if item["decidable"]:
        return "POSITIVE"
    if item["reason_code"] in EXCLUDED_CODES:
        return "EXCLUDED"
    if item["reason_code"] in NEGATIVE_CODES:
        return "NEGATIVE"
    raise ValueError(
        f"item {item['id']}: reason_code {item['reason_code']!r} is in neither the "
        f"NEGATIVE nor the EXCLUDED list frozen in the preregistration"
    )


def split_of(url: str) -> str:
    """DEVELOPMENT or HELD_OUT, by the repository's existing convention.

    `v14_gates.bucket` on the first five URL segments, `< 3` being the split that
    is designed on and not scored (`v14_packet.py`). Imported, not reimplemented,
    because the preregistration says so.
    """
    return "DEVELOPMENT" if bucket("/".join((url or "").split("/")[:5])) < 3 else "HELD_OUT"


def first_repeat(calls: list[dict]) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for c in calls:
        if c["repeat"] == 0:
            out[c["id"]] = c["noul"]
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_1_thresholds(by_split: dict) -> dict:
    """G-C1-1 (blocking): thresholds chosen on DEVELOPMENT, scored on HELD_OUT.

    A development split that holds only one class cannot discriminate between
    threshold pairs: every pair that reads everything scores identically. When
    that happens the gate is unrunnable, and it is recorded as unrunnable rather
    than repaired by re-drawing the split. Re-drawing a split after looking at
    its class balance is the act this gate exists to forbid, and it stays
    forbidden when the person re-drawing it has good intentions.
    """
    dev = by_split["DEVELOPMENT"]
    pos = sum(1 for x in dev if x["cls"] == "POSITIVE")
    neg = sum(1 for x in dev if x["cls"] == "NEGATIVE")
    if pos == 0 or neg == 0:
        return {
            "gate": "G-C1-1",
            "blocking": True,
            "status": "UNRUNNABLE",
            "dev_positive": pos,
            "dev_negative": neg,
            "why": (
                f"the DEVELOPMENT split holds {pos} POSITIVE and {neg} NEGATIVE items. "
                "A threshold pair cannot be chosen on a split that holds one class, so "
                "no threshold pair is chosen, and none ships. The split is not redrawn."
            ),
            "thresholds": None,
        }
    grid = [round(x / 100, 2) for x in range(0, 101, 5)]
    best, best_key = None, None
    for skip in grid:
        for read in grid:
            if not skip < read:
                continue
            tp = sum(1 for x in dev if x["cls"] == "POSITIVE" and x["noul"] >= read)
            tn = sum(1 for x in dev if x["cls"] == "NEGATIVE" and x["noul"] <= skip)
            bal = (tp / pos + tn / neg) / 2
            # Ties break toward the widest refusal band: of two rules that score
            # the same on development, prefer the one that refuses more.
            key = (bal, read - skip)
            if best_key is None or key > best_key:
                best_key, best = key, {"skip": skip, "read": read}
    return {
        "gate": "G-C1-1",
        "blocking": True,
        "status": "RUN",
        "dev_positive": pos,
        "dev_negative": neg,
        "selection_rule": "max balanced accuracy on DEVELOPMENT, ties to the widest band",
        "thresholds": best,
        "dev_balanced_accuracy": best_key[0] if best_key else None,
    }


def gate_2_separation(scored: list[dict]) -> dict:
    """G-C1-2 (blocking): Prediction 1, median POSITIVE minus median NEGATIVE."""
    pos = [x["noul"] for x in scored if x["cls"] == "POSITIVE"]
    neg = [x["noul"] for x in scored if x["cls"] == "NEGATIVE"]
    if not pos or not neg:
        return {"gate": "G-C1-2", "blocking": True, "status": "UNRUNNABLE",
                "why": "one of the two classes has no usable noul"}
    sep = statistics.median(pos) - statistics.median(neg)
    wins2, pairs2 = concordance(pos, neg)
    ci = wilson(wins2, pairs2)
    width = (ci[1] - ci[0]) * 100 if ci else None
    return {
        "gate": "G-C1-2",
        "blocking": True,
        "status": "PASS" if sep >= SEPARATION_BAR else "FAIL",
        "median_positive": statistics.median(pos),
        "median_negative": statistics.median(neg),
        "separation": sep,
        "bar": SEPARATION_BAR,
        "n_positive": len(pos),
        "n_negative": len(neg),
        "concordance": wins2 / pairs2 if pairs2 else None,
        "concordance_ci95": ci,
        "concordance_ci_width_points": width,
        "settles_the_question": (width is not None and width <= INTERVAL_WIDTH_BAR),
        "note": (
            "The concordance proportion is the headline separation that carries an "
            "interval; a difference of medians has no Wilson interval. Its pairs are "
            "not independent — 13 x 8 pairs come from 21 items — so the interval is "
            "indicative, and it is published to be read as such."
        ),
    }


def gate_3_calibration(scored: list[dict]) -> dict:
    pairs = [(x["noul"], 1 if x["cls"] == "POSITIVE" else 0) for x in scored]
    r = ece(pairs)
    return {
        "gate": "G-C1-3",
        "blocking": False,
        "status": "REPORTED",
        "prediction_3_ece_exceeds": ECE_PREDICTED_FLOOR,
        "prediction_3_held": (r["ece"] is not None and r["ece"] > ECE_PREDICTED_FLOOR),
        **r,
    }


def gate_4_determinism(calls: list[dict], ids: list[int], repeats: int) -> dict:
    per = {}
    for i in ids:
        vals = [c["noul"] for c in calls if c["id"] == i and c["noul"] is not None]
        per[i] = {"n": len(vals), "spread": (max(vals) - min(vals)) if vals else None}
    graded = sorted(ids)[:DETERMINISM_N]
    spreads = [per[i]["spread"] for i in graded if per[i]["spread"] is not None]
    worst = max(spreads) if spreads else None
    any_nonzero = any(s > 0 for s in spreads)
    return {
        "gate": "G-C1-4",
        "blocking": False,
        "status": "REPORTED",
        "repeats": repeats,
        "graded_items": graded,
        "graded_note": (
            f"every item was asked {repeats} times; the gate is computed over the "
            f"{DETERMINISM_N} lowest ids exactly as frozen, and the rest are published "
            "beside it without being able to move it (Amendment A3)."
        ),
        "per_item": per,
        "max_spread_graded": worst,
        "prediction_4_spread_exceeds": DETERMINISM_PREDICTED_SPREAD,
        "prediction_4_held": (worst is not None and worst > DETERMINISM_PREDICTED_SPREAD),
        "verdict_path_statement": (
            "Any spread above zero is a permanent argument against Jev on the verdict path."
            if any_nonzero else
            "Every graded spread is zero on this run. That is not a promise of determinism: "
            "nothing in the vendor's documentation offers one, and a later run may differ."
        ),
    }


def gate_5_cost(calls: list[dict], price_in: float | None, price_out: float | None) -> dict:
    lat = sorted(c["ms"] for c in calls if c["ms"] is not None)
    tin = sum(c["input_tokens"] or 0 for c in calls)
    tout = sum(c["output_tokens"] or 0 for c in calls)
    spend = None
    if price_in is not None and price_out is not None:
        spend = round(tin / 1e6 * price_in + tout / 1e6 * price_out, 2)
    return {
        "gate": "G-C1-5",
        "blocking": False,
        "status": "REPORTED",
        "calls": len(calls),
        "input_tokens": tin,
        "output_tokens": tout,
        "median_latency_ms": statistics.median(lat) if lat else None,
        "spend_usd": spend,
        "spend_note": (
            None if spend is not None else
            "No per-token price is recorded anywhere in this repository, so the spend is "
            "not published. Tokens and latency are. Supply --price-per-mtok-in and "
            "--price-per-mtok-out to have it computed; an estimated spend is not a "
            "measured one and this program will not invent it (Amendment A4)."
        ),
    }


def gate_6_band(all_items: list[dict], thresholds: dict | None) -> dict:
    """G-C1-6: how many of the 25 land inside the refusal band.

    When G-C1-1 could not choose a pair, occupancy is reported over a grid of
    candidate bands instead of a chosen one, so the reader can still see whether
    a band that separates anything would swallow the stratum.
    """
    usable = [x for x in all_items if x["noul"] is not None]
    if thresholds:
        inside = [x["id"] for x in usable
                  if thresholds["skip"] < x["noul"] < thresholds["read"]]
        return {"gate": "G-C1-6", "blocking": False, "status": "REPORTED",
                "thresholds": thresholds, "n_total": len(all_items),
                "n_inside_band": len(inside), "inside_ids": sorted(inside),
                "changes_nothing_for": len(inside)}
    grid = []
    for skip, read in ((0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.60)):
        inside = [x["id"] for x in usable if skip < x["noul"] < read]
        excluded_inside = [x["id"] for x in usable
                           if x["cls"] == "EXCLUDED" and skip < x["noul"] < read]
        grid.append({"skip": skip, "read": read, "n_inside_band": len(inside),
                     "inside_ids": sorted(inside),
                     "excluded_items_inside": sorted(excluded_inside)})
    return {
        "gate": "G-C1-6", "blocking": False, "status": "REPORTED_OVER_GRID",
        "thresholds": None, "n_total": len(all_items), "grid": grid,
        "why_grid": "G-C1-1 chose no thresholds, so there is no single band to report.",
    }


# --------------------------------------------------------------------------- #

def score(raw: dict, price_in=None, price_out=None) -> dict:
    refusals = []
    if raw.get("dry_run"):
        refusals.append("the raw file is a dry run: its answers came from a hash, not from Jev")
    if raw.get("prereg_sha256") != PREREG_SHA256_FROZEN:
        refusals.append(
            f"prereg sha256 is {raw.get('prereg_sha256')}, frozen at {PREREG_SHA256_FROZEN}"
        )

    noul0 = first_repeat(raw["calls"])
    items = []
    for it in raw["items"]:
        items.append({
            "id": it["id"],
            "url": it["url"],
            "cls": classify(it),
            "split": split_of(it["url"]),
            "noul": noul0.get(it["id"]),
            "n_files": it["n_files"],
            "paths_truncated": it["paths_truncated"],
        })
    scored = [x for x in items if x["cls"] != "EXCLUDED" and x["noul"] is not None]
    by_split = {"DEVELOPMENT": [x for x in items if x["split"] == "DEVELOPMENT"
                                and x["cls"] != "EXCLUDED" and x["noul"] is not None],
                "HELD_OUT": [x for x in items if x["split"] == "HELD_OUT"
                             and x["cls"] != "EXCLUDED" and x["noul"] is not None]}

    g1 = gate_1_thresholds(by_split)
    g2 = gate_2_separation(scored)
    g3 = gate_3_calibration(scored)
    g4 = gate_4_determinism(raw["calls"], [x["id"] for x in items], raw.get("repeats", 1))
    g5 = gate_5_cost(raw["calls"], price_in, price_out)
    g6 = gate_6_band(items, g1.get("thresholds"))

    excluded = [x for x in items if x["cls"] == "EXCLUDED"]
    if refusals:
        token = "INVALID__DRY_RUN" if raw.get("dry_run") else "INVALID__PREREG_MOVED"
    elif g1["status"] == "UNRUNNABLE":
        token = ("CALIB1__NO_THRESHOLDS__G_C1_1_UNRUNNABLE__SEPARATION_"
                 + ("PASS" if g2["status"] == "PASS" else g2["status"]))
    elif g2["status"] != "PASS":
        token = "CALIB1__ABANDONED__G_C1_2_FAILED"
    else:
        token = "CALIB1__THRESHOLDS_SELECTED"

    return {
        "prereg": raw.get("prereg"),
        "prereg_sha256": raw.get("prereg_sha256"),
        "prereg_sha256_at_freeze": PREREG_SHA256_AT_FREEZE,
        "triage_module_sha256": raw.get("triage_module_sha256"),
        "dry_run": bool(raw.get("dry_run")),
        "refusals": refusals,
        "verdict_token": token,
        "population": {
            "n_total": len(items),
            "n_positive": sum(1 for x in items if x["cls"] == "POSITIVE"),
            "n_negative": sum(1 for x in items if x["cls"] == "NEGATIVE"),
            "n_excluded": len(excluded),
            "n_scored": len(scored),
            "by_split": {k: len(v) for k, v in by_split.items()},
        },
        "excluded_items": [
            {"id": x["id"], "noul": x["noul"]} for x in sorted(excluded, key=lambda y: y["id"])
        ],
        "gates": [g1, g2, g3, g4, g5, g6],
        "ships_thresholds": token == "CALIB1__THRESHOLDS_SELECTED",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(HERE / "calib1_raw.json"))
    ap.add_argument("--out", default=str(HERE / "calib1_result.json"))
    ap.add_argument("--price-per-mtok-in", type=float, default=None)
    ap.add_argument("--price-per-mtok-out", type=float, default=None)
    a = ap.parse_args()

    raw = json.loads(Path(a.raw).read_text(encoding="utf-8"))
    res = score(raw, a.price_per_mtok_in, a.price_per_mtok_out)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print(f"verdict_token: {res['verdict_token']}")
    for g in res["gates"]:
        print(f"  {g['gate']:<8} {g['status']}")
    for r in res["refusals"]:
        print(f"  REFUSED: {r}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
