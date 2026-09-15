#!/usr/bin/env python3
"""build_sand_survey_pass3_margins.py — for every source and every reading, which of the fingerprint clause's five
elements the source misses.

    python papers/plates/build_sand_survey_pass3_margins.py
    # writes papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json

Why this exists. `CORRECTION_sand_neighbours_pass3_2026_09_14.md` (sworn) said the fingerprint clause "survives on
what it measures more than on its floor or its interval", and the REPORT and CHANGELOG repeated it more strongly.
The lab's verification of 2026-09-14 (confirmed by both skeptics) showed the record says otherwise for at least
one source: Xu et al. carries the self-comparison, the fixed item set, the interval and the log-probability object,
and misses only the floor measured on the same weights. The margin is per source. The sworn correction is not
edited; `ERRATUM_sand_neighbours_pass3_correction_2026_09_14.md` swears to this record instead.

The five elements are the four of the clause's OBJECT, as the pass-3 protocol stated them before reading, and the
THING of its operational row (teacher-forced log-probabilities). The readings are the four of the correction's
builder, exactly as written there, plus a fifth that the verification noted was missing: the lenient interval
together with a lenient floor — a measured null that grades the comparison counts as a floor whether or not it was
measured on the same weights. Every override names recorded text, and the script refuses to run if that text is
not in the pass-3 record. For each reading the record lists each source's missing elements, the fewest any source
misses, and which sources miss that few.
"""
import importlib.util
import json
import os

REC = "papers/plates/sand_prior_art_survey_pass3_2026_09_14.json"
OUT = "papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json"
HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("pass3_correction", os.path.join(HERE, "build_sand_survey_pass3_correction.py"))
_corr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_corr)

ELEMENTS = _corr.OBJECT + [_corr.THING]
SHORT = {"self_comparison": "self-comparison", "fixed_item_set": "fixed item set", "same_weights_floor_in_situ": "floor",
         "interval_on_comparison": "interval", "log_prob_object": "log-probabilities"}

READINGS = dict(_corr.READINGS)
READINGS["the_lenient_interval_and_any_measured_null_that_grades_the_comparison_as_a_floor"] = {
    "rule": "the stated-level-test reading of the interval, and a measured null that grades the comparison counts as a floor "
            "whether or not it was measured on the same weights (a seed population, retrainings, independently trained models)",
    "overrides": [_corr.L03_TEST, _corr.L06_TEST,
                  ["L06", "same_weights_floor_in_situ", True, "The floor is the between-seed variance of 30 independent trainings"],
                  ["L08", "same_weights_floor_in_situ", True, "the measured null floor is seed variance across different weights"],
                  ["L12", "same_weights_floor_in_situ", True, "the floor that grades it is the distribution of independently trained negative models"],
                  ["L14", "same_weights_floor_in_situ", True, "Section 5 is the one place in these five sources where a measured nondeterminism floor is used to grade a comparison"]],
}


def main():
    rec = json.load(open(REC, encoding="utf-8"))
    srcs = rec["sources"]
    base = {sid: dict(s["verdicts"]["C4a"]["elements"]) for sid, s in srcs.items()
            if "C4a" in s["verdicts"] and isinstance(s["verdicts"]["C4a"].get("elements"), dict)}
    names = {sid: srcs[sid]["who"] for sid in base}
    out_readings = {}
    for name, rd in READINGS.items():
        els = {sid: dict(e) for sid, e in base.items()}
        applied = []
        for sid, el, val, text in rd["overrides"]:
            assert text in json.dumps(srcs[sid], ensure_ascii=False), f"{name}: override text not in the record for {sid}: {text!r}"
            applied.append({"source": sid, "element": el, "recorded": base[sid][el], "read_as": val, "rests_on": text})
            els[sid][el] = val
        missing = {sid: [k for k in ELEMENTS if not e.get(k)] for sid, e in els.items()}
        fewest = min(len(m) for m in missing.values())
        nearest = sorted(sid for sid, m in missing.items() if len(m) == fewest)
        out_readings[name] = {
            "rule": rd["rule"], "overrides": applied,
            "missing_by_source": {sid: [SHORT[k] for k in m] for sid, m in sorted(missing.items())},
            "fewest_missing": fewest,
            "sources_missing_the_fewest": nearest,
            "n_sources_missing_the_fewest": len(nearest),
            "what_the_nearest_miss": {sid: ", ".join(SHORT[k] for k in missing[sid]) for sid in nearest},
            "n_sources_missing_nothing": sum(1 for m in missing.values() if not m),
        }
    distinct = sorted({v for r in out_readings.values() for v in r["what_the_nearest_miss"].values()})
    out = {
        "schema": "styxx.plates/sand-survey-margins/v0",
        "corrects": "papers/plates/CORRECTION_sand_neighbours_pass3_2026_09_14.md",
        "record": REC,
        "elements": [SHORT[k] for k in ELEMENTS],
        "sources": {sid: names[sid] for sid in sorted(names)},
        "n_sources": len(base),
        "n_readings": len(out_readings),
        "readings": out_readings,
        "n_readings_with_a_source_missing_nothing": sum(1 for r in out_readings.values() if r["n_sources_missing_nothing"]),
        "fewest_missing_under_every_reading": min(r["fewest_missing"] for r in out_readings.values()),
        "what_the_nearest_miss_across_readings": distinct,
        "n_distinct_elements_the_nearest_miss": len({e for v in distinct for e in v.split(", ")}),
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote", OUT)
    for name, r in out_readings.items():
        print(f"  {name}: fewest missing {r['fewest_missing']}; nearest {r['what_the_nearest_miss']}")
    print("  across readings the nearest miss:", distinct)


if __name__ == "__main__":
    main()
