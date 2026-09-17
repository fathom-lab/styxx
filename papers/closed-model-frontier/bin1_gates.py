"""BIN-1 gates (PREREG_bin1_binary_files_2026_09_16): the repaired parsers, scored.

    python bin1_gates.py            # writes bin1_gates.json (counts only; no PR named)

  G-BIN-1  the 91 EXTERNAL-5 live diffs, re-fetched: the repaired parse's file count equals the
           count of `diff --git` headers on every PR served. Blocking.
  G-BIN-2  read from the differential's own outputs (py_out.json / js_out.json next to the port,
           produced by py_side.py + js_side.js + differential.py): 0 disagreements, and every record
           of the 3,199 pre-BIN-1 pairs identical to the pre-BIN-1 run (a copy of py_out.json made
           before the repair, path below). Blocking.
  G-BIN-3  a 2,000-PR sample of `external4_ledger.jsonl` (seed 20260916) re-derived through the
           EXTERNAL-1 reconstruction with this checkout's instrument: claims (kind, text, verdict,
           why) identical. Blocking on the sample.
  G-BIN-4  `external5_items.jsonl` re-derived under this checkout (external5_source.py fetch, run
           beforehand into external5_items_bin1.jsonl): files_by_parse == live_files on every
           served PR; the three truthful counts read OVERTURNED under the instrument's own parse;
           the outcome table identical to the RESULT's completed-list reading. Blocking.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import styxx.diffgate as dg  # noqa: E402
sys.path.insert(0, str(HERE))
from external1_harness import reconstruct  # noqa: E402
import external5_source as e5  # noqa: E402

DB = HERE / "external2_shelf.sqlite"
E4 = HERE / "external4_ledger.jsonl"
DIFF = ROOT / "web" / "gate" / "differential"
BEFORE = Path("/tmp/py_out_before_bin1.json")          # web/gate/differential/py_out.json as produced BEFORE the repair
                                                        # (check out the pre-repair diffgate.py, run py_side.py, copy the output here)
E5_BEFORE = HERE / "external5_items.jsonl"                # the EXTERNAL-5 run (completed-list reading)
E5_AFTER = HERE / "external5_items_bin1.jsonl"            # the same run under this checkout
OUT = HERE / "bin1_gates.json"
HEADER = dg._DIFF_GIT
INSTRUMENT = hashlib.sha256(Path(dg.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def g_bin1() -> dict:
    urls = sorted({it["url"] for it in e5.survivors()})
    served = 0; agree = 0; rows = []
    for url in urls:
        status, body = e5.fetch(e5.diff_url(url))
        if status != 200 or not body:
            continue
        text = body.decode("utf-8", errors="replace")
        n_headers = sum(1 for line in text.splitlines() if line.startswith("diff --git "))
        n_parse = len(dg.parse_unified_diff(text)[0])
        served += 1; agree += (n_headers == n_parse)
        rows.append({"sha256": hashlib.sha256(body).hexdigest()[:16], "headers": n_headers, "parse": n_parse})
    return {"pull_requests": len(urls), "served": served, "parse_equals_headers": agree,
            "pass": served > 0 and agree == served, "per_diff": rows}


def g_bin2() -> dict:
    py = {d["id"]: d for d in json.loads((DIFF / "py_out.json").read_text(encoding="utf-8"))}
    js = {d["id"]: d for d in json.loads((DIFF / "js_out.json").read_text(encoding="utf-8"))}
    before = {d["id"]: d for d in json.loads(BEFORE.read_text(encoding="utf-8"))}
    dis = sum(1 for k in py if json.dumps(py[k], sort_keys=True) != json.dumps(js.get(k), sort_keys=True))
    same = sum(1 for k in before if json.dumps(before[k], sort_keys=True) == json.dumps(py.get(k), sort_keys=True))
    return {"pairs": len(py), "disagreements": dis, "pre_bin1_pairs": len(before), "identical_after": same,
            "pass": dis == 0 and same == len(before) and len(py) > len(before)}


def g_bin3(n: int = 2000) -> dict:
    recs = [json.loads(l) for l in E4.open(encoding="utf-8")]
    rng = random.Random(20260916)
    sample = rng.sample(recs, n)
    con = sqlite3.connect(DB)
    same = diff = 0; kinds = Counter()
    for r in sample:
        files = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?", (r["pr_id"],)).fetchall()
        pr = con.execute("SELECT title, body FROM pr WHERE id=?", (r["pr_id"],)).fetchone()
        diff_text, _implied = reconstruct(files)
        g = dg.gate_diff_text(f"{pr[0] or ''}\n\n{pr[1]}", diff_text, run=None, strict=False)
        got = [(c.kind, c.text, c.verdict, c.why) for c in g.claims]
        want = [(c["kind"], c["text"], c["verdict"], c["why"]) for c in r["claims"]]
        if got == want:
            same += 1
        else:
            diff += 1
        for c in g.claims:
            kinds[c.kind] += 1
    con.close()
    return {"sample": n, "identical": same, "different": diff, "claims_in_sample": sum(kinds.values()),
            "pass": diff == 0}


def g_bin4() -> dict:
    key = lambda it: (it["pr_id"], it["claim_index"])  # noqa: E731
    before = {key(it): it for it in (json.loads(l) for l in E5_BEFORE.open(encoding="utf-8"))}
    after = {key(it): it for it in (json.loads(l) for l in E5_AFTER.open(encoding="utf-8"))}
    served = [it for it in after.values() if it["source"]["http"] == 200 and not it["source"].get("orphaned")]
    eq = sum(1 for it in served if it["source"]["files_by_parse"] == it["source"]["live_files"])
    same_outcome = sum(1 for k in before if k in after and before[k]["reading"]["outcome"] == after[k]["reading"]["outcome"])
    was_blind = [k for k, it in before.items() if it["source"].get("binary_unregistered")]
    now_seen = sum(1 for k in was_blind if after[k]["source"]["files_by_parse"] == after[k]["source"]["live_files"])
    tb = Counter(it["reading"]["outcome"] for it in before.values())
    ta = Counter(it["reading"]["outcome"] for it in after.values())
    return {"items": len(after), "served": len(served), "files_by_parse_equals_live": eq,
            "items_on_diffs_the_parse_was_blind_to": len(was_blind), "of_those_now_counted_by_the_parse": now_seen,
            "outcomes_before": dict(tb), "outcomes_after": dict(ta), "same_outcome_per_item": same_outcome,
            "pass": eq == len(served) and same_outcome == len(before) and now_seen == len(was_blind)}


def main() -> int:
    out = {"prereg": "PREREG_bin1_binary_files_2026_09_16.md", "instrument_sha256": INSTRUMENT,
           "G-BIN-1_live_diffs": g_bin1(), "G-BIN-2_differential": g_bin2(),
           "G-BIN-3_corpus_sample": g_bin3(), "G-BIN-4_external5_reread": g_bin4()}
    out["all_blocking_gates_pass"] = all(out[k]["pass"] for k in ("G-BIN-1_live_diffs", "G-BIN-2_differential",
                                                                   "G-BIN-3_corpus_sample", "G-BIN-4_external5_reread"))
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_diff"})
                      for k, v in out.items()}, indent=1))
    return 0 if out["all_blocking_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
