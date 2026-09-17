#!/usr/bin/env python3
"""build_sand_survey_pass3_correction.py — what the red team of 2026-09-14 found in the sand survey's pass 3, as a
record beside it. Pass 3's record and SURVEY are sworn and are not edited.

    python papers/plates/build_sand_survey_pass3_correction.py <pass-3 protocol commit>
    # writes papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json

Three things, each mechanical:

1. The inputs. Pass 3's SURVEY told a stranger that build_sand_survey_pass3.py rebuilds the record "from the
   readers' returns and the fetch record" and that fetch_pass3.py re-fetches — and neither the returns nor the
   script were in the tree (red team survey F5 / ferry F1, confirmed). They are committed now under
   papers/plates/sand_survey_pass3_inputs/ (the four returns, unescaped exactly as the builder read them, and
   the fetch script as it ran). This script rebuilds the pass-3 record from those committed inputs into a
   scratch file and compares it with the committed record field by field. The only fields it cannot rebuild
   are the verbatim-quote check's, because that check read the extracted full texts, which are not committed:
   they are seventeen publications' full text, other people's work; their sha256 are in the fetch record.

2. Self-reports. `chars_read` is what each reader reported, and each reader was told its files' sizes; the
   record's total is a sum of reports, not a measurement. Recorded as such.

3. The two terms the protocol left undefined. The pass-3 protocol stated the fingerprint clause's OBJECT
   before reading — self-comparison, a fixed item set, a floor measured on the same weights under the serving
   in use, an interval on the comparison — and its pricing rule retires a clause only for "the same thing for
   the same object", where the THING is the clause's operational row: teacher-forced log-probabilities on a
   fixed, hashed item set. It never defined "interval on the comparison" or "measured on the same weights",
   and the readers coded them differently (a threshold t-test was an interval for reader-4, a z-test at a
   critical value was not for reader-2; red team survey F1 and F4, plausible). This script recodes the
   elements under four stated readings — each override naming the recorded text it rests on, which the script
   asserts is in the record — and counts, per reading, the sources carrying all four object elements and the
   sources carrying the object AND the thing. A source in the second list would retire the clause.
"""
import hashlib
import json
import os
import subprocess
import sys

REC = "papers/plates/sand_prior_art_survey_pass3_2026_09_14.json"
FETCH = "papers/plates/sand_survey_fetch_record_pass3_2026_09_14.json"
INPUTS = "papers/plates/sand_survey_pass3_inputs"
BUILDER = "papers/plates/build_sand_survey_pass3.py"
OUT = "papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json"
OBJECT = ["self_comparison", "fixed_item_set", "same_weights_floor_in_situ", "interval_on_comparison"]
THING = "log_prob_object"

L03_TEST = ["L03", "interval_on_comparison", True, "larger than the critical value 2.58 of z-score at the 99% confidence level"]
L06_TEST = ["L06", "interval_on_comparison", True, "If the p-value <= 0.05, we reject the null hypothesis"]
READINGS = {
    "as_recorded": {
        "rule": "the element flags exactly as the four readers returned them",
        "overrides": []},
    "a_stated_level_test_on_the_difference_is_an_interval": {
        "rule": "a significance test on the difference at a stated level counts as an interval on the comparison — the reading "
                "reader-4 applied to L11 and L12 — applied also to the two sources whose reader coded it otherwise",
        "overrides": [L03_TEST, L06_TEST]},
    "only_a_confidence_interval_on_the_compared_quantity_is_an_interval": {
        "rule": "only a confidence interval on the compared quantity counts; a threshold test, or a spread of repeated runs, does not",
        "overrides": [["L05", "interval_on_comparison", False, "not a confidence interval on the teacher-student difference"],
                      ["L12", "interval_on_comparison", False, "one-tailed T-test to calculate the lower bound"]]},
    "the_lenient_interval_and_a_floor_only_on_weights_known_identical": {
        "rule": "the stated-level-test reading of the interval, and a floor counts as measured on the same weights only when the "
                "weights are known identical, not inferred from a period in which the scores were stable",
        "overrides": [L03_TEST, L06_TEST,
                      ["L03", "same_weights_floor_in_situ", False, "'same version' within a period is an inference from stability, not a measurement"]]},
}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _strip(rec):
    r = json.loads(json.dumps(rec))
    for s in r.get("sources", {}).values():
        for v in s.get("verdicts", {}).values():
            v.pop("quote_found_in_text", None)
    for k in ("verdict_quotes_checked", "verdict_quotes_found_in_text"):
        r.get("counts", {}).pop(k, None)
    r.pop("run_date", None)
    return r


def main():
    protocol_commit = sys.argv[1]
    rec = json.load(open(REC, encoding="utf-8"))

    # 1. rebuild from the committed inputs
    scratch = "papers/plates/sand_prior_art_survey_pass3_rebuild_check.json"
    r = subprocess.run([sys.executable, BUILDER, FETCH, INPUTS, "rebuild_check", protocol_commit],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("the builder refused the committed inputs:\n" + r.stdout + r.stderr)
    rebuilt = json.load(open(scratch, encoding="utf-8"))
    os.remove(scratch)
    a, b = _strip(rec), _strip(rebuilt)
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    source_fields_differing = sorted({f"{sid}.{k}" for sid in a.get("sources", {})
                                      for k in set(a["sources"][sid]) | set(b["sources"].get(sid, {}))
                                      if a["sources"][sid].get(k) != b["sources"].get(sid, {}).get(k)})
    inputs = {f: sha(os.path.join(INPUTS, f)) for f in sorted(os.listdir(INPUTS))}

    # 2. self-reports
    srcs = rec["sources"]
    chars = {sid: {"reported": s.get("chars_read"), "file": (s.get("fulltext") or {}).get("chars")} for sid, s in srcs.items()}
    empty_quotes = sum(1 for s in srcs.values() for v in s["verdicts"].values() if not (v.get("quote") or "").strip())

    # 3. the readings
    c4a = {sid: dict(s["verdicts"]["C4a"]["elements"]) for sid, s in srcs.items()
           if "C4a" in s["verdicts"] and isinstance(s["verdicts"]["C4a"].get("elements"), dict)}
    readings = {}
    for name, rd in READINGS.items():
        els = {sid: dict(e) for sid, e in c4a.items()}
        applied = []
        for sid, el, val, text in rd["overrides"]:
            blob = json.dumps(srcs[sid], ensure_ascii=False)
            assert text in blob, f"{name}: the text an override rests on is not in the record for {sid}: {text!r}"
            applied.append({"source": sid, "element": el, "recorded": c4a[sid][el], "read_as": val, "rests_on": text})
            els[sid][el] = val
        four = sorted(sid for sid, e in els.items() if all(e.get(k) for k in OBJECT))
        both = sorted(sid for sid in four if els[sid].get(THING))
        readings[name] = {
            "rule": rd["rule"], "overrides": applied,
            "element_true_counts": {k: sum(1 for e in els.values() if e.get(k)) for k in OBJECT + [THING]},
            "sources_with_the_four_object_elements": four, "n_sources_with_the_four_object_elements": len(four),
            "sources_with_the_object_and_the_thing": both, "n_sources_with_the_object_and_the_thing": len(both),
            "would_retire_C4a": bool(both),
        }
    lists = {tuple(v["sources_with_the_four_object_elements"]) for v in readings.values()}
    out = {
        "schema": "styxx.plates/sand-survey-correction/v0",
        "corrects": REC, "pass3_survey": "papers/plates/SURVEY_sand_neighbours_pass3_2026_09_14.md",
        "protocol": "papers/plates/PROTOCOL_sand_prior_art_pass3_2026_09_14.md", "protocol_commit": protocol_commit,
        "red_team": "2026-09-14: survey F5 and ferry F1 confirmed (inputs not in the tree); survey F1 and F4 plausible "
                    "(interval coded inconsistently; L05's flag against its own note); survey F2 and F3 refuted",
        "inputs": {"dir": INPUTS, "sha256": inputs, "n_reader_returns": sum(1 for f in inputs if f.startswith("reader-")),
                   "fetch_script": "fetch_pass3.py" in inputs},
        "rebuild": {"rebuilt_from_committed_inputs": True, "top_level_fields_differing": differing,
                    "n_top_level_fields_differing": len(differing),
                    "source_fields_differing": source_fields_differing,
                    "n_source_fields_differing": len(source_fields_differing),
                    "n_sources_rebuilt": len(b.get("sources", {})),
                    "matches_except_the_quote_check": not differing and not source_fields_differing,
                    "not_rebuildable_here": ["sources.*.verdicts.*.quote_found_in_text", "counts.verdict_quotes_checked",
                                             "counts.verdict_quotes_found_in_text"],
                    "why": "the quote check read the extracted full texts, which are not committed (other people's work); "
                           "their sha256 are in the fetch record (fulltext.sha256); re-deriving the check needs a re-fetch "
                           "and the same extractor, and a source revised since 2026-09-14 will not reproduce"},
        "self_reports": {"chars_read_equals_the_file_size_each_reader_was_told": all(v["reported"] == v["file"] for v in chars.values()),
                         "n_sources": len(chars), "chars_reported_total": sum(v["reported"] or 0 for v in chars.values()),
                         "note": "READ and chars_read are the readers' reports; nothing measured how much of a text a reader read",
                         "empty_verdict_quotes": empty_quotes},
        "undefined_terms": {"object_elements": OBJECT, "thing_element": THING,
                            "thing_element_note": "the operational row also says 'hashed' item set; no reader coded it and no source was flagged as hashing its set",
                            "retires_only_if": "a source carries all four object elements AND the thing (pricing rule: the same thing for the same object)",
                            "readings": readings,
                            "n_readings": len(readings),
                            "n_readings_under_which_C4a_would_retire": sum(1 for v in readings.values() if v["would_retire_C4a"]),
                            "the_four_object_elements_under_some_reading": sorted({s for t in lists for s in t})},
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote", OUT)
    print("rebuild matches except the quote check:", out["rebuild"]["matches_except_the_quote_check"], differing, source_fields_differing[:5])
    for name, v in readings.items():
        print(f"  {name}: interval {v['element_true_counts']['interval_on_comparison']}, four object elements {v['sources_with_the_four_object_elements']}, "
              f"object+thing {v['sources_with_the_object_and_the_thing']}")


if __name__ == "__main__":
    main()
