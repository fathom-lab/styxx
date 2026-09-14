#!/usr/bin/env python3
"""record_search.py — turn the pass-4 search workflow's return into the committed search and screen record and the
closed list, mechanically.

    python papers/plates/sand_survey_pass4_inputs/record_search.py <workflow_return.json> <raw_dir> <out_dir> <run_ids>

<workflow_return.json> is the object the search workflow returned (queries, searches, raw_engine_returns, dedupe_rule,
n_results, n_distinct, unscreened, list_b, below_cap, part_a_found_by_search, excluded). <raw_dir> holds the raw engine
responses the engine agents saved; each is hashed into the record (the responses themselves are search-engine output
and are committed beside the record). Writes, into <out_dir>:
  search_record.json  every query result, merged into works, with both screeners' decisions and where each work went
  list.json           Part A (the nine leads, as the protocol lists them) then B01..B25, with might_occupy
"""
import hashlib
import json
import os
import sys

PART_A = [
    ("A01", "Cao, Jia & Gong", 2021), ("A02", "Lukas, Zhang & Kerschbaum", 2021), ("A03", "Finlayson, Ren & Swayamdipta", 2024),
    ("A04", "Russinovich & Salem", 2024), ("A05", "Carlini, Paleka, Dvijotham et al.", 2024), ("A06", "Aiyappa, An, Kwak & Ahn", 2023),
    ("A07", "Hooker, Moorosi, Clark, Bengio & Denton", 2020), ("A08", "Ouyang, Zhang, Harman & Wang", 2025), ("A09", "Song, Wang, Li & Lin", 2024),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ret = json.load(open(sys.argv[1], encoding="utf-8"))
    raw_dir, out_dir, run_ids = sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    here = os.path.dirname(os.path.abspath(__file__))
    located = {x["id"]: x for x in json.load(open(os.path.join(here, "part_a_located.json"), encoding="utf-8"))}

    raw = []
    for dp, _, fs in os.walk(raw_dir):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            raw.append({"file": os.path.relpath(p, raw_dir).replace(os.sep, "/"), "bytes": os.path.getsize(p), "sha256": sha(p)})
    raw.sort(key=lambda r: r["file"])

    def screened(it, where):
        return {"key": it["key"], "title": it["title"], "authors": it.get("authors", ""), "year": it.get("year"), "urls": it.get("urls", []),
                "abstract": it.get("abstract", ""), "hits": it["hits"], "n_engine_query_pairs": it["n_engine_query_pairs"],
                "screener1": it.get("screener1"), "screener2": it.get("screener2"), "went": where}

    works = [screened(it, it["id"]) for it in ret["list_b"]]
    works += [screened(it, f"below cap, rank {it['rank_below_cap']}") for it in ret["below_cap"]]
    works += [screened(it, it["part_a"]) for it in ret["part_a_found_by_search"]]
    works += [screened(it, "excluded") for it in ret["excluded"]]

    notes = {}
    for k, v in (ret.get("raw_engine_returns") or {}).items():
        if v:
            notes[k] = {"engine_as_reported": v.get("engine"), "fetched_at": v.get("fetched_at"), "notes": v.get("notes"),
                        "results": len(v.get("results", [])), "failures": v.get("failures", [])}
    record = {
        "record": "sand survey pass 4 — the frozen search (Part B) and its screen",
        "protocol": "papers/plates/PROTOCOL_sand_prior_art_pass4_2026_09_15.md", "protocol_commit": "5d7f39ef",
        "workflow_runs": run_ids, "queries": [{"id": q[0], "clause": q[1], "text": q[2]} for q in ret["queries"]],
        "engine_returns": notes, "searches": ret["searches"], "raw_files": raw,
        "dedupe_rule": ret.get("dedupe_rule"), "n_results": ret.get("n_results"), "n_distinct": ret["n_distinct"],
        "unscreened": ret["unscreened"], "cap": 25, "rank_rule": "(engine, query) pairs desc, then later year, then title",
        "list_b": [w for w in works if w["went"].startswith("B")], "below_cap": [w for w in works if w["went"].startswith("below cap")],
        "part_a_found_by_search": [w for w in works if w["went"].lower().startswith("part a")],
        "excluded": [w for w in works if w["went"] == "excluded"],
    }
    with open(os.path.join(out_dir, "search_record.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    lst = []
    for sid, who, year in PART_A:
        lst.append({"id": sid, "part": "A", "title": located[sid]["title"], "who": who, "year": year, "urls": located[sid]["urls"], "might_occupy": ["C4a"]})
    for it in ret["list_b"]:
        clauses = sorted({d["clause"] for d in (it.get("screener1"), it.get("screener2")) if d and d.get("include") and d.get("clause")})
        if not clauses:
            clauses = sorted({next(q[1] for q in ret["queries"] if q[0] == h["query_id"]) for h in it["hits"]})
        # a source is a candidate only for the clause its screeners included it under: a bounty page that cannot be
        # fetched unprices C5, not the fingerprint clause (readers still return a C4a verdict for every source)
        lst.append({"id": it["id"], "part": "B", "title": it["title"], "who": it.get("authors") or "", "year": it.get("year"),
                    "urls": it.get("urls", []), "might_occupy": sorted(set(clauses))})
    with open(os.path.join(out_dir, "list.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(lst, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("works", len(works), "list_b", len(record["list_b"]), "below_cap", len(record["below_cap"]),
          "part_a_found", len(record["part_a_found_by_search"]), "excluded", len(record["excluded"]), "list", len(lst), "raw files", len(raw))


if __name__ == "__main__":
    main()
