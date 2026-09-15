"""Summarise the non-browser re-fetch for the erratum's sworn tags, from committed files only.

    python papers/plates/sand_survey_user_agent_inputs/build_refetch_summary.py
    # writes refetch_nonbrowser_summary.json beside this file

Reads refetch_nonbrowser_record.json (the run under the rules frozen at 8273cdda), the three fetchers' user-agent
lines, the correction's lookup record, and the pass-3 and correction survey records for the bounty clause.
"""
import collections
import json
import os
import re
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES = os.path.dirname(HERE)
OUTCOMES = ["SAME_BYTES", "SAME_KIND", "KIND_CHANGED", "NOW_ANSWERS", "STILL_REFUSED", "NOW_REFUSED", "RATE_LIMITED"]


def load(rel):
    with open(os.path.join(PLATES, rel), encoding="utf-8") as fh:
        return json.load(fh)


def ua_line(rel):
    with open(os.path.join(PLATES, rel), encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            m = re.match(r'UA = "([^"]+)"', line)
            if m:
                return {"file": rel, "line": i, "user_agent": m.group(1), "browser_form": m.group(1).startswith("Mozilla/5.0")}
    raise SystemExit(f"no UA line in {rel}")


def count_requests(x):
    if isinstance(x, dict):
        return (1 if "url" in x and "http" in x else 0) + sum(count_requests(v) for v in x.values())
    if isinstance(x, list):
        return sum(count_requests(v) for v in x)
    return 0


def main():
    rec = load("sand_survey_user_agent_inputs/refetch_nonbrowser_record.json")
    rows = rec["rows"]
    by_record = collections.Counter(a["record"] for r in rows for a in r["recorded"])
    outcomes = {k: sum(1 for r in rows if r["outcome"] == k) for k in OUTCOMES}
    assert sum(outcomes.values()) == len(rows), "an outcome outside the frozen list"
    same_kind_diff = [abs((r["now"]["bytes"] or 0) - min((a["bytes"] or 0) for a in r["recorded"])) for r in rows if r["outcome"] == "SAME_KIND"]
    still = collections.Counter(urlparse(r["url"]).hostname for r in rows if r["outcome"] == "STILL_REFUSED")
    now_refused = [{"row": i, "id": r["recorded"][0]["id"], "record": r["recorded"][0]["record"], "url": r["url"],
                    "recorded_http": r["recorded"][0]["http"], "recorded_bytes": r["recorded"][0]["bytes"],
                    "recorded_at": r["recorded"][0]["fetched_at"], "now_http": r["now"]["http"], "now_at": r["now"]["fetched_at"]}
                   for i, r in enumerate(rows) if r["outcome"] == "NOW_REFUSED"]
    times = sorted(r["now"]["fetched_at"] for r in rows)

    p3 = load("sand_prior_art_survey_pass3_2026_09_14.json")["clauses"]["C5"]
    cor = load("sand_prior_art_survey_pass4_correction_2026_09_15.json")["clauses"]["C5"]
    refused_ids = {x["id"] for x in now_refused}
    c5 = {
        "pass3_status": p3["status"], "pass3_occupied_by": p3["occupied_by"], "pass3_retired_by": p3["retired_by"],
        "pass3_occupied_by_without_now_refused": [s for s in p3["occupied_by"] if s not in refused_ids],
        "pass3_status_without_now_refused": "OCCUPIED" if [s for s in p3["occupied_by"] if s not in refused_ids] and not p3["retired_by"] else "RECOMPUTE",
        "pass1_status": p3.get("pass1_status"), "pass2_status": p3.get("pass2_status"),
        "correction_status": cor["status"],
        "correction_fields": {k: v for k, v in cor.items() if isinstance(v, (str, list)) and k != "clause"},
    }
    summary = {
        "_note": "derived by build_refetch_summary.py from committed files; the erratum's sworn tags point here and at the record",
        "frozen_commit": "8273cdda",
        "run_first_request_at": times[0], "run_last_request_at": times[-1],
        "refetch_user_agent": rec["user_agent"],
        "fetchers": [ua_line("sand_survey_pass3_inputs/fetch_pass3.py"), ua_line("sand_survey_pass4_inputs/fetch_pass4.py")],
        "attempts_by_record": dict(by_record), "attempts_total": sum(by_record.values()), "distinct_urls": len(rows),
        "correction_lookup_requests_recorded": count_requests(load("sand_survey_pass4_correction_inputs/located_lookups_correction.json")),
        "outcomes": outcomes,
        "same_kind_max_byte_difference": max(same_kind_diff) if same_kind_diff else 0,
        "still_refused_by_host": dict(sorted(still.items())),
        "now_refused": now_refused,
        "bounty_clause": c5,
    }
    out = os.path.join(HERE, "refetch_nonbrowser_summary.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
