#!/usr/bin/env python3
"""build_sand_survey_pass4_margins.py — the margin check beside pass 4's record: blind re-codings of every source the
record puts at distance one on the fingerprint clause, checked and counted.

    python papers/plates/build_sand_survey_pass4_margins.py <margins_run.json> <record.json> <date> [<texts_dir>]

Outside the frozen protocol, and it changes no status in the record. The protocol decides C4a from the first
reading and, only when that reading codes E1-E5 all true, from two blind confirmers. Pass 3's ERRATUM showed that
the clause's margin can rest on how one reader codes one element. So every source at distance one was re-read by
further readers, blind to the first coding and told nothing of which element was in question. This script checks
each re-coding as the record's builder checks a reading: the last-line and midpoint proofs, and every quote for an
element coded true, verbatim. It then counts, per source and element, how many counted re-codings carry it.

A re-coding that fails either proof is recorded and not counted. The output says, per source, whether any counted
re-coding carries all five elements and whether a majority does. It also says, for the element the first reading
found missing, how many re-codings found it present.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sand_survey_pass4 as b  # noqa: E402

E = ["E1", "E2", "E3", "E4", "E5"]


def main():
    run = json.load(open(sys.argv[1], encoding="utf-8"))
    record_path = sys.argv[2]
    rec = json.load(open(record_path, encoding="utf-8"))
    date = sys.argv[3]
    texts = sys.argv[4] if len(sys.argv) > 4 else None

    sources = {}
    for sid in sorted({m["id"] for m in run["margins"]}):
        src = rec["sources"][sid]
        first = src["verdicts"]["C4a"]
        rows = []
        for m in sorted((m for m in run["margins"] if m["id"] == sid), key=lambda m: m["reader"]):
            chk = b.text_checks(texts, sid, m)
            shown = None if chk is None else bool(chk.get("text_present") and chk.get("last_line_ok") and chk.get("midpoint_ok"))
            els = {}
            for e in E + ["E6"]:
                ev = m["elements"].get(e) or {}
                els[e] = {"value": bool(ev.get("value")), "quote": ev.get("quote", ""),
                          "quote_found": b.quote_found(chk, ev.get("quote", "")) if ev.get("value") else None}
            missing = [e for e in E if not els[e]["value"]]
            rows.append({"reader": m["reader"], "shown_read_to_end": shown,
                         "proof": None if chk is None else {"last_line_ok": chk.get("last_line_ok"), "midpoint_ok": chk.get("midpoint_ok")},
                         "elements": els, "missing": missing, "distance": len(missing), "object": m.get("object"),
                         "reason": m.get("reason"), "hardest_element": m.get("hardest_element"),
                         "hardest_element_why": m.get("hardest_element_why")})
        counted = [r for r in rows if r["shown_read_to_end"] is not False]
        tally = {e: sum(1 for r in counted if r["elements"][e]["value"]) for e in E + ["E6"]}
        all_five = [r["reader"] for r in counted if r["distance"] == 0]
        sources[sid] = {
            "title": src.get("title"),
            "first_reading": {"reader": src.get("read_by"), "missing": first.get("missing"), "distance": first.get("distance")},
            "margin_readings": rows,
            "n_margin_readings": len(rows), "n_counted": len(counted),
            "element_true_counts": tally,
            "first_readings_missing_element_found_present_in": {e: tally[e] for e in first.get("missing") or []},
            "counted_readings_carrying_all_five": all_five,
            "a_majority_of_counted_readings_carry_all_five": len(counted) > 0 and len(all_five) * 2 > len(counted),
            "distance_counts": {str(d): sum(1 for r in counted if r["distance"] == d) for d in range(6)},
            "hardest_elements": {e: sum(1 for r in counted if r["hardest_element"] == e) for e in E + ["E6"]},
            "quotes_true": sum(1 for r in counted for e in E + ["E6"] if r["elements"][e]["value"]),
            "quotes_true_found": sum(1 for r in counted for e in E + ["E6"] if r["elements"][e]["value"] and r["elements"][e]["quote_found"]),
        }

    out = {
        "check": "the sand, pass 4: blind margin re-codings of every source at distance one on C4a",
        "record": record_path.replace(os.sep, "/"), "run": run.get("run"), "script": run.get("script"),
        "run_date": date.replace("_", "-"),
        "sources": sources,
        "sources_with_a_counted_reading_carrying_all_five": sorted(s for s, v in sources.items() if v["counted_readings_carrying_all_five"]),
        "sources_where_a_majority_carry_all_five": sorted(s for s, v in sources.items() if v["a_majority_of_counted_readings_carry_all_five"]),
        "what_it_does_not_do": "It changes no status in the record. The frozen protocol prices C4a from the first reading and its confirmers; this check measures how much that pricing rests on one reader's coding.",
    }
    path = f"papers/plates/sand_prior_art_survey_pass4_margins_{date}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote", path)
    for sid, v in sources.items():
        print(sid, "counted", v["n_counted"], "of", v["n_margin_readings"], "true counts", v["element_true_counts"],
              "all five:", v["counted_readings_carrying_all_five"], "first missing found present:", v["first_readings_missing_element_found_present_in"],
              "quotes", v["quotes_true_found"], "/", v["quotes_true"])
    print("any all five:", out["sources_with_a_counted_reading_carrying_all_five"], "majority:", out["sources_where_a_majority_carry_all_five"])


if __name__ == "__main__":
    main()
