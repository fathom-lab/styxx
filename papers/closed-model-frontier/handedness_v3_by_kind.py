# -*- coding: utf-8 -*-
"""EXPLORATORY — the token-kind stratification of the committed handedness v3 rows.

Reads ONE committed receipt, ``handedness_v3_result.json`` (frozen by
``PREREG_handedness_v3_join_rule_2026_09_02.md``, verdict ``HEADER_HANDED_ACCUSES_TRUER``), and
re-cuts its 349 panel-judged rows by token kind: a token is DECIMAL when it prints a ``.`` and
INTEGER otherwise. Nothing is re-judged, nothing is fetched, no gate moves.

WHY IT EXISTS. The v3 RESULT reports header-handed accusations genuine at 0.9515 against
line-handed at 0.6391 and reads the gap as structure. A same-day objection said the gap is mostly
token KIND — the header cell is mostly decimals, and a decimal is a claim whoever hands it. That
objection circulated with a number attached ("kind-adjusted gap 0.117") and NO receipt: no
stratified file exists anywhere in this repository at main ``320b303``. This script produces the
receipt the objection lacked, so the next preregistration in this lane can declare it as a
contaminated prior by digest instead of quoting a number from memory.

WHAT IT IS NOT. Not a result. Not preregistered. Computed after the v3 verdict was read, on the
rows that produced it, by the lab that produced them. It may never be quoted as a finding; the
only sentence it licenses is "the kind split exists as a committed exploratory table". The odds
ratios below are printed with Wilson intervals because a stratified cell of 23 rows has a wide
one, and printing the point alone would be the M2 defect (a number measured where the target was
handed) in a document about M2.

    python papers/closed-model-frontier/handedness_v3_by_kind.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "handedness_v3_result.json"
OUT = HERE / "handedness_v3_by_kind_result.json"

GENUINE = {"CLAIM", "GENUINE", "CHECKABLE_CLAIM"}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def is_genuine(row: dict) -> bool:
    # The v3 receipt records the panel's answer per row; "genuine" in the RESULT means the panel
    # called the token a checkable claim. Any label outside the NOT_A_CLAIM family counts as
    # genuine, and the label vocabulary observed is printed in the receipt so this rule is auditable.
    return str(row.get("panel", "")).upper() not in ("NOT_A_CLAIM", "NOT-A-CLAIM", "NOTACLAIM")


def kind_of(token: str) -> str:
    return "decimal" if "." in str(token) else "integer"


def main() -> int:
    r = json.loads(SRC.read_text(encoding="utf-8"))
    rows = r["rows"]
    labels = sorted({str(x.get("panel")) for x in rows})
    cells: dict = {}
    repos: dict = {}
    for x in rows:
        cell = x.get("cell")
        if cell not in ("header", "line"):
            continue
        k = kind_of(x.get("token"))
        c = cells.setdefault((cell, k), {"n": 0, "genuine": 0})
        c["n"] += 1
        c["genuine"] += 1 if is_genuine(x) else 0
        repos[x.get("repo")] = repos.get(x.get("repo"), 0) + 1

    def share(c):
        return round(c["genuine"] / c["n"], 4) if c["n"] else None

    def odds_ratio(a, b):
        # OR of genuine for a over b, Haldane-Anscombe corrected when a cell is zero
        a1, a0 = a["genuine"], a["n"] - a["genuine"]
        b1, b0 = b["genuine"], b["n"] - b["genuine"]
        if 0 in (a1, a0, b1, b0):
            a1, a0, b1, b0 = a1 + .5, a0 + .5, b1 + .5, b0 + .5
        return round((a1 / a0) / (b1 / b0), 4)

    table = {}
    for (cell, k), c in sorted(cells.items()):
        table["%s/%s" % (cell, k)] = {"n": c["n"], "genuine": c["genuine"],
                                     "genuine_share": share(c), "wilson95": wilson(c["genuine"], c["n"])}
    hd, hi = cells.get(("header", "decimal"), {"n": 0, "genuine": 0}), cells.get(("header", "integer"), {"n": 0, "genuine": 0})
    ld, li = cells.get(("line", "decimal"), {"n": 0, "genuine": 0}), cells.get(("line", "integer"), {"n": 0, "genuine": 0})
    header_all = {"n": hd["n"] + hi["n"], "genuine": hd["genuine"] + hi["genuine"]}
    line_all = {"n": ld["n"] + li["n"], "genuine": ld["genuine"] + li["genuine"]}
    # kind-adjusted gap: reweight the header cell to the LINE cell's kind mix
    line_mix_dec = ld["n"] / line_all["n"] if line_all["n"] else None
    adj_header = None
    if line_mix_dec is not None and hd["n"] and hi["n"]:
        adj_header = round(line_mix_dec * (hd["genuine"] / hd["n"]) + (1 - line_mix_dec) * (hi["genuine"] / hi["n"]), 4)
    top_repo = max(repos.items(), key=lambda kv: kv[1]) if repos else (None, 0)
    out = {
        "what": "EXPLORATORY token-kind stratification of handedness_v3_result.json rows; not a result, not preregistered; may not be quoted as a finding",
        "source": "papers/closed-model-frontier/handedness_v3_result.json",
        "source_verdict": r.get("verdict"),
        "panel_labels_observed": labels,
        "genuine_rule": "panel label not in the NOT_A_CLAIM family",
        "kind_rule": "decimal iff the printed token contains '.'; integer otherwise",
        "rows_considered": header_all["n"] + line_all["n"],
        "cells": table,
        "raw_cells_recomputed": {
            "header": {"n": header_all["n"], "genuine_share": share(header_all)},
            "line": {"n": line_all["n"], "genuine_share": share(line_all)},
        },
        "raw_gap_header_minus_line": (round(share(header_all) - share(line_all), 4)
                                      if header_all["n"] and line_all["n"] else None),
        "header_decimal_mix": round(hd["n"] / header_all["n"], 4) if header_all["n"] else None,
        "line_decimal_mix": round(line_mix_dec, 4) if line_mix_dec is not None else None,
        "kind_adjusted_header_share": adj_header,
        "kind_adjusted_gap": (round(adj_header - share(line_all), 4) if adj_header is not None else None),
        "odds_ratio_header_over_line": {
            "integer_stratum": odds_ratio(hi, li) if hi["n"] and li["n"] else None,
            "decimal_stratum": odds_ratio(hd, ld) if hd["n"] and ld["n"] else None,
        },
        "top_repository": {"repo": top_repo[0], "rows": top_repo[1],
                           "share_of_rows": round(top_repo[1] / max(1, header_all["n"] + line_all["n"]), 4)},
        "what_this_is_not": [
            "not a result: computed after the v3 verdict was read, on the rows that produced it",
            "not a measurement of structure: a header cell is also a table cell in benchmark idiom, and the panel saw table rows as rows",
            "one panel of one model family judged every row (the v3 RESULT's own limit)",
            "the kind-adjusted gap reweights one cell to the other's mix; it is a description, not an estimate of a causal effect",
        ],
    }
    # newline="\n": on Windows, text mode would translate every LF to CRLF and the receipt would
    # hash differently on each platform — the styxx/centroids lesson, applied before it bites.
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print("header/line by kind:")
    for k, v in table.items():
        print("  %-16s n=%-4d genuine=%-4d share=%s wilson=%s" % (k, v["n"], v["genuine"], v["genuine_share"], v["wilson95"]))
    print("raw gap %s | kind-adjusted gap %s | OR integer %s decimal %s | top repo %s (%s rows)"
          % (out["raw_gap_header_minus_line"], out["kind_adjusted_gap"],
             out["odds_ratio_header_over_line"]["integer_stratum"],
             out["odds_ratio_header_over_line"]["decimal_stratum"], top_repo[0], top_repo[1]))
    print("-> %s" % OUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
