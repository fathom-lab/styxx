# -*- coding: utf-8 -*-
"""The shapes P9 now refuses in decoys, counted over real items of the development store (pull requests opened
before 2026-11-02). Prints counts only: no pull request, repository or text is printed.

    py -3.12 checks/decoy_shape_check.py <development store>

What it counts, each over the store's two-step-like items (eligible claimers with a CI test result, and eligible
rows read as no claim beside "tests failed"), rendered by the builder's own code:
- the combination of the screen's reading and testreceipt's CI reading, against `tpc_decoys.P2_COMBINATIONS`;
- failing Actions check runs with no annotation (`tpc_decoys.annotation_problems`);
- date traces left after P8.1's mask (`tpc_items.date_traces`), and the log excerpts by how they were found;
- repository names left in the CI block: the owner/name, a runner's working path, the short name as a token;
and, over every failing Actions job with a failing step, whether a step flip at that step needs a reference run
(a later step that is not a post step) and whether the flip, given the job's own later conclusions, reproduces its
listing.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import tpc_decoys  # noqa: E402
import tpc_items  # noqa: E402
import tpc_packet  # noqa: E402
from tpc_const import WINDOW_START, parse_utc  # noqa: E402
from tpc_screen import _map, screen  # noqa: E402


def _listing(job: dict) -> list:
    return [(s.get("name"), s.get("conclusion")) for s in sorted(job.get("steps") or [],
                                                                  key=lambda s: (s.get("number") or 0))]


def main() -> int:
    store = Path(sys.argv[1])
    rows = tpc_packet._read_rows(store)
    late = [r for r in rows if parse_utc(r["created_at"]) >= WINDOW_START]
    if late:
        print(f"refused: {len(late)} rows opened on or after 2026-11-02")
        return 2
    out: dict = {"rows": len(rows)}
    combos, hows, kinds = Counter(), Counter(), Counter()
    items = in_combo = with_trace = no_ann_items = no_ann_checks = 0
    left = Counter()
    flips = Counter()
    for r in rows:
        if not r.get("eligible") or not (r.get("ci") or {}).get("has_ci_test_result"):
            continue
        ok_screen = r.get("screen") == "claimer" or (r.get("screen") == "none"
                                                     and r["ci"]["outcome"] == "tests failed")
        try:
            ci = tpc_items.ci_from_row(store, r)
        except (OSError, ValueError, KeyError):
            continue
        for j in ci.get("jobs") or []:
            failing = [s for s in sorted(j.get("steps") or [], key=lambda s: (s.get("number") or 0))
                       if _map(s.get("conclusion")) == "failure"]
            if _map(j.get("conclusion")) != "failure" or not failing:
                continue
            flips["failing_jobs"] += 1
            step = failing[0]["name"] or ""
            later = tpc_decoys.later_steps(j, step)
            if not later:
                flips["no_later_step_but_post"] += 1
                continue
            flips["later_steps"] += 1
            concl = Counter(s.get("conclusion") for s in later)
            flips["later_all_skipped" if set(concl) <= {"skipped"} else "later_some_ran"] += 1
            src = copy.deepcopy(dict(ci, logs={}))
            sj = next(x for x in src["jobs"] if x["id"] == j["id"])
            k = [s.get("name") for s in sorted(sj["steps"], key=lambda s: (s.get("number") or 0))].index(step)
            for s in sorted(sj["steps"], key=lambda s: (s.get("number") or 0))[k:]:
                if not (s.get("name") or "").startswith("Post ") and s.get("name") not in tpc_decoys.ALWAYS_RUN:
                    s["conclusion"] = "success"
            try:
                got = tpc_decoys._fail_from(src, j["id"], step, None,
                                            later={s.get("name"): s.get("conclusion") for s in later})
                same = _listing(next(x for x in got["jobs"] if x["id"] == j["id"])) == _listing(j)
                flips["reproduced_with_reference" if same else "differs_with_reference"] += 1
            except tpc_decoys.DecoyError:
                flips["refused_with_reference"] += 1
        if not ok_screen:
            continue
        body = tpc_items.body_from_row(store, r)
        it = tpc_items.build_item(body, ci, r.get("repo"))
        items += 1
        readings = {screen(body)["screen"], screen(it["step1_text"])["screen"]}
        outcome = tpc_decoys.ci_outcome(ci)
        failed_any = tpc_decoys.any_failed_job_or_check(ci)
        for kind in readings:
            combos[f"{kind} | {outcome} | failed_any={failed_any}"] += 1
        if all(any(kind == k and outcome == o and (f is None or failed_any == f)
                   for k, o, f in tpc_decoys.P2_COMBINATIONS) for kind in readings):
            in_combo += 1
        probs = tpc_decoys.annotation_problems(ci)
        no_ann_checks += len(probs)
        no_ann_items += int(bool(probs))
        t = tpc_items.date_traces(it["step1_text"]) + tpc_items.date_traces(it["item_text"])
        kinds.update(t)
        with_trace += int(bool(t))
        for x in it["build"]["logs"]:
            hows[x["how"]] += 1
        block = it["item_text"].split("\n" + tpc_items.CI_HEAD + "\n", 1)[1]
        repo = (r.get("repo") or "").lower()
        if repo and repo in block.lower():
            left["owner_name"] += 1
        if re.search(r"(?i)(/home/runner/work|/Users/runner/work|/__w|/_work|[a-z]:\\a)[/\\](?!<repo>)[^_/\\\s]", block):
            left["runner_path"] += 1
        toks = tpc_items._name_tokens([r.get("repo") or ""])
        if any(re.search(r"(?i)(?<![A-Za-z0-9<])" + re.escape(x) + r"(?![A-Za-z0-9>])", block) for x in toks):
            left["short_name_or_owner_token"] += 1
    out.update({"two_step_like_items": items, "in_p2_combinations": in_combo, "combinations": dict(combos),
                "items_with_failing_actions_check_without_annotation": no_ann_items,
                "failing_actions_checks_without_annotation": no_ann_checks,
                "items_with_date_trace": with_trace, "date_traces_by_kind": dict(kinds),
                "log_excerpts_by_how": dict(hows), "repository_names_left_in_ci_block": dict(left),
                "step_flips": dict(flips)})
    sys.stdout.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
