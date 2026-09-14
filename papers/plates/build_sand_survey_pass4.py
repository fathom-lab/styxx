#!/usr/bin/env python3
"""build_sand_survey_pass4.py — the record of pass 4 of the sand survey, built mechanically from committed inputs.

    python papers/plates/build_sand_survey_pass4.py <inputs_dir> <date YYYY_MM_DD> <protocol_commit> [<texts_dir>]
    # reads, from <inputs_dir> (papers/plates/sand_survey_pass4_inputs/):
    #   list.json            the closed list: Part A (the leads) and Part B (the search, capped), with might_occupy
    #   search_record.json   every query result, both screeners' decisions, the rank and the cap
    #   fetch_record*.json   per source: URL, bytes, sha256, the title check, the extracted text's sha256 (one file per
    #                        fetch run, merged; a source in two records is refused)
    #   readings*.json       the readers' and blind confirmers' returns, one file per reading run, merged
    # and the pass-3 record (sworn, never edited); writes papers/plates/sand_prior_art_survey_pass4_<date>.json

What it does, per PROTOCOL_sand_prior_art_pass4_2026_09_15.md, and nothing the protocol does not say:

- A source is READ when its reader says it read end to end AND, when the texts are given, the reader's quoted last
  line lies in the text's last three non-empty lines (extracted PDFs end in page numbers and footers) AND the
  reader's midpoint quote is found in the text and lies in its middle third by line number (the unit a reader sees;
  character position only for a quote that spans a wrapped line). The midpoint is required because an
  extracted PDF's last line is often a bare page number, which proves nothing about reading. Confirmers are held to
  the same two checks. Readings from several runs (readings*.json) are merged, each reader named by its run.
  Neighbour phrases are HTML-unescaped when the sentence is composed. A reader who cannot quote the end did not
  show that they reached it: the source is SKIMMED, and a RETIRES from a SKIMMED source is UNCHECKABLE — it neither
  retires the clause nor clears it, and the clause is UNPRICED.
  Every element quote is checked verbatim after whitespace, ligature, quote-mark and line-end-hyphen normalisation.
- C4a: a source whose reader codes E1-E5 all true is a candidate retirement. It RETIRES only when two blind
  confirmers each code E1-E5 true and each shows they reached the end of the text; otherwise it is DISPUTED. Any
  confirmed retirement makes C4a RETIRED; any DISPUTED source, or an unfetchable C4a candidate, makes it UNPRICED.
- C2 and C5: the inherited rule — a READ source that RETIRES makes the clause RETIRED; an unfetchable candidate makes
  it UNPRICED; otherwise the clause stays OCCUPIED, as passes 1 to 3 left it.
- The sentence: RETIRED if C4a or C5 is RETIRED (sentence rule 1); UNLICENSED if C4a is UNPRICED (the protocol: a
  sentence without its fingerprint clause is not the sand's sentence); otherwise pass 3's wording for C1, C2 and C5,
  with a deleted clause removed, and the fingerprint clause's parenthesis rebuilt by count: for each of E1-E5, the
  source at distance one whose only missing element is that one — pass 4's coding first, pass 3's as recorded
  (flagged: coded before the terms were defined) second — and, if no source anywhere is at distance one, the sources
  at the smallest distance (at most three), each with what it lacks. Readers' "nearer" flags decide nothing.
"""
import html
import json
import os
import re
import sys

P3 = "papers/plates/sand_prior_art_survey_pass3_2026_09_14.json"
PROTOCOL = "papers/plates/PROTOCOL_sand_prior_art_pass4_2026_09_15.md"
E = ["E1", "E2", "E3", "E4", "E5"]
ENAME = {"E1": "a comparison of the model with its own self", "E2": "a fixed item set",
         "E3": "a floor measured on the same weights", "E4": "an interval on the comparison", "E5": "log-probabilities"}
P3MAP = {"self_comparison": "E1", "fixed_item_set": "E2", "same_weights_floor_in_situ": "E3",
         "interval_on_comparison": "E4", "log_prob_object": "E5"}
P3_PARTS = {
    "C1": "binds every published number to bytes at a commit (where Deterministic Integrity Gates and Cited-but-Not-Verified bind sentences and citations of a manuscript, as the 2026-09-05 survey names them)",
    "C2": "re-derives every verdict from those bytes into a chained log (where Certificate Transparency and Rekor chain certificates and signed metadata, not verdicts)",
    "C4a_head": "fingerprints a model's behavior on hashed canaries against a measured null floor",
    "C5": "pays a standing bounty against its own verifier (where Immunefi pays against deployed code and the Preregistration Challenge paid for preregistering)",
}


def norm(t):
    t = t or ""
    for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("’", "'"), ("‘", "'"),
                 ("“", '"'), ("”", '"'), ("−", "-"), ("–", "-"), ("—", "-")):
        t = t.replace(a, b)
    t = re.sub(r"-\s*\n\s*", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load(path):
    return json.load(open(path, encoding="utf-8"))


def text_checks(texts_dir, sid, reading):
    if not texts_dir:
        return None
    p = os.path.join(texts_dir, sid + ".txt")
    if not os.path.exists(p):
        return {"text_present": False}
    raw = open(p, encoding="utf-8").read()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    n = len(raw)
    whole = norm(raw)
    last_ok = bool(lines) and norm(reading.get("last_line_quote", "")) != "" and \
        norm(reading.get("last_line_quote", "")) in norm("\n".join(lines[-3:]))
    mq = norm(reading.get("midpoint_quote", ""))
    mid_ok = False
    if mq:
        # the middle third by line number: readers see the file through the Read tool's numbered lines, and a
        # reference list of short lines moves the character midpoint far from the line midpoint
        all_lines = raw.split("\n")
        hits = [i for i, ln in enumerate(all_lines) if norm(ln) and (mq in norm(ln) or (len(norm(ln)) > 20 and norm(ln) in mq))]
        if hits:
            mid_ok = any(1 / 3 <= i / len(all_lines) <= 2 / 3 for i in hits)
        else:
            pos = whole.find(mq)  # a quote spanning a wrapped line: fall back to character position
            mid_ok = pos >= 0 and 1 / 3 <= pos / max(len(whole), 1) <= 2 / 3
    return {"text_present": True, "chars": n, "last_line_ok": last_ok, "midpoint_ok": mid_ok, "_whole": whole}


def quote_found(checks, q):
    if not checks or not checks.get("text_present"):
        return None
    q = norm(q)
    return bool(q) and q in checks["_whole"]


def main():
    inputs, date, protocol_commit = sys.argv[1], sys.argv[2], sys.argv[3]
    texts_dir = sys.argv[4] if len(sys.argv) > 4 else None
    lst = load(os.path.join(inputs, "list.json"))
    search = load(os.path.join(inputs, "search_record.json"))
    fetch = {}
    for fn in sorted(f for f in os.listdir(inputs) if f.startswith("fetch_record") and f.endswith(".json")):
        for sid, entry in load(os.path.join(inputs, fn)).items():
            assert sid not in fetch, f"{sid} fetched in two records"
            fetch[sid] = dict(entry, fetch_record=fn)
    rd = {"readings": [], "confirmations": []}
    for fn in sorted(f for f in os.listdir(inputs) if f.startswith("readings") and f.endswith(".json")):
        run = fn[: -len(".json")]
        part = load(os.path.join(inputs, fn))
        rd["readings"] += [dict(r, reader=f"{run}:{r['reader']}") for r in part.get("readings", [])]
        rd["confirmations"] += [dict(c, confirmer=f"{run}:{c.get('confirmer')}") for c in part.get("confirmations", [])]
    p3 = load(P3)

    readings = {}
    for r in rd.get("readings", []):
        for s in r.get("sources", []):
            assert s["id"] not in readings, f"{s['id']} read twice"
            readings[s["id"]] = {"reader": r["reader"], **s}
    confirms = {}
    for c in rd.get("confirmations", []):
        confirms.setdefault((c["id"], c.get("clause") or "C4a"), []).append(c)

    sources = {}
    for item in lst:
        sid = item["id"]
        f = fetch.get(sid, {})
        src = {k: item.get(k) for k in ("title", "who", "year", "part", "might_occupy", "urls")}
        src.update({k: f.get(k) for k in ("url", "url_effective", "http", "bytes", "sha256", "fetched_at", "kind", "pages", "fulltext", "title_words_found")})
        if f.get("status") != "FETCHED":
            src["status"] = "UNFETCHABLE"
            src["unfetchable_reason"] = f.get("reason")
            src["verdicts"] = {c: {"verdict": "UNCHECKABLE"} for c in item["might_occupy"]}
            sources[sid] = src
            continue
        r = readings.get(sid)
        if r is None:
            src["status"] = "UNREAD"
            src["verdicts"] = {c: {"verdict": "UNSCORED"} for c in item["might_occupy"]}
            sources[sid] = src
            continue
        chk = text_checks(texts_dir, sid, r)
        shown_end = chk is None or bool(chk.get("text_present") and chk.get("last_line_ok") and chk.get("midpoint_ok"))
        src["status"] = "READ" if (r.get("read_end_to_end") and shown_end) else "SKIMMED"
        src["read_by"] = r["reader"]
        src["proof_of_reading"] = None if chk is None else {k: chk.get(k) for k in ("last_line_ok", "midpoint_ok")}
        src["bearing_quotes"], src["leads"], src["notes"] = r.get("bearing_quotes", []), r.get("leads", []), r.get("notes", "")
        src["verdicts"] = {}
        for v in r.get("verdicts", []):
            c = v["clause"]
            entry = {"verdict": v["verdict"], "object": v.get("object"), "reason": v.get("reason"), "neighbour_phrase": v.get("neighbour_phrase")}
            if c == "C4a" and isinstance(v.get("elements"), dict):
                els = {}
                for e in E + ["E6"]:
                    ev = v["elements"].get(e) or {}
                    els[e] = {"value": bool(ev.get("value")), "quote": ev.get("quote", ""),
                              "quote_found": quote_found(chk, ev.get("quote", "")) if ev.get("value") else None}
                entry["elements"] = els
                entry["distance"] = sum(1 for e in E if not els[e]["value"])
                entry["missing"] = [e for e in E if not els[e]["value"]]
                candidate = entry["distance"] == 0
                if v["verdict"] == "RETIRES" and not candidate:
                    entry["verdict_note"] = "the reader wrote RETIRES without coding E1-E5 all true; recorded as OCCUPIES"
                    entry["verdict"] = "OCCUPIES"
                if candidate:
                    cs = confirms.get((sid, "C4a"), [])
                    rows = []
                    for cf in cs:
                        cchk = text_checks(texts_dir, sid, cf)
                        all5 = all(bool((cf.get("elements") or {}).get(e, {}).get("value")) for e in E)
                        rows.append({"confirmer": cf.get("confirmer"), "all_five": all5, "object": cf.get("object"), "reason": cf.get("reason"),
                                     "shown_end": None if cchk is None else bool(cchk.get("last_line_ok") and cchk.get("midpoint_ok")),
                                     "missing": [e for e in E if not (cf.get("elements") or {}).get(e, {}).get("value")]})
                    confirmed = len(rows) >= 2 and all(x["all_five"] and x["shown_end"] is not False for x in rows)
                    entry["confirmations"] = rows
                    if src["status"] != "READ":
                        entry["verdict"], entry["retirement"] = "UNCHECKABLE", "SKIMMED: a source not shown read to the end neither retires nor clears the clause"
                    elif confirmed:
                        entry["verdict"], entry["retirement"] = "RETIRES", "CONFIRMED by two blind readers"
                    else:
                        entry["verdict"], entry["retirement"] = "DISPUTED", "the two blind confirmations did not both code E1-E5 true"
            if c in ("C2", "C5") and entry["verdict"] == "RETIRES":
                rows = []
                for cf in confirms.get((sid, c), []):
                    cchk = text_checks(texts_dir, sid, cf)
                    rows.append({"confirmer": cf.get("confirmer"), "verdict": cf.get("verdict"), "object": cf.get("object"), "reason": cf.get("reason"),
                                 "shown_end": None if cchk is None else bool(cchk.get("last_line_ok") and cchk.get("midpoint_ok"))})
                entry["confirmations"] = rows
                if src["status"] != "READ":
                    entry["verdict"], entry["retirement"] = "UNCHECKABLE", "SKIMMED: a source not shown read to the end neither retires nor clears the clause"
                elif len(rows) >= 2 and all(x["verdict"] == "RETIRES" and x["shown_end"] is not False for x in rows):
                    entry["retirement"] = "CONFIRMED by two blind readers"
                else:
                    entry["verdict"], entry["retirement"] = "DISPUTED", "the two blind confirmations did not both find RETIRES"
            src["verdicts"][c] = entry
        for c in item["might_occupy"]:
            src["verdicts"].setdefault(c, {"verdict": "UNSCORED", "reason": "the reader returned no verdict for this clause"})
        sources[sid] = src

    # clause statuses
    clauses = {k: dict(v) for k, v in p3["clauses"].items()}
    for cid in ("C2", "C4a", "C5"):
        cl = clauses[cid]
        cl["pass3_status"] = cl["status"]
        cands = [sid for sid, s in sources.items() if cid in (s.get("might_occupy") or [])]
        v = {sid: sources[sid]["verdicts"].get(cid, {}).get("verdict") for sid in cands}
        cl["pass4"] = {"retired_by": sorted(s for s, x in v.items() if x == "RETIRES"),
                       "disputed_by": sorted(s for s, x in v.items() if x == "DISPUTED"),
                       "occupied_by": sorted(s for s, x in v.items() if x == "OCCUPIES"),
                       "silent": sorted(s for s, x in v.items() if x == "SILENT"),
                       "uncheckable": sorted(s for s, x in v.items() if x == "UNCHECKABLE"),
                       "unscored": sorted(s for s, x in v.items() if x == "UNSCORED")}
        p4 = cl["pass4"]
        if cl["status"] == "RETIRED" or p4["retired_by"]:
            cl["status"] = "RETIRED"
        elif p4["disputed_by"] or p4["uncheckable"] or p4["unscored"]:
            cl["status"] = "UNPRICED"
        else:
            cl["status"] = "OCCUPIED"

    # nearness for C4a, counted
    pool = []
    for sid, s in sources.items():
        e = s["verdicts"].get("C4a", {})
        if s["status"] == "READ" and "distance" in e:
            pool.append({"id": sid, "pass": 4, "who": s.get("who") or s.get("title"), "missing": e["missing"], "distance": e["distance"],
                         "phrase": html.unescape(e.get("neighbour_phrase") or "")})
    for sid, s in p3["sources"].items():
        v = (s.get("verdicts") or {}).get("C4a") or {}
        els = v.get("elements")
        if isinstance(els, dict):
            missing = [P3MAP[k] for k in P3MAP if not els.get(k)]
            pool.append({"id": f"pass3:{sid}", "pass": 3, "who": s.get("who"), "missing": missing, "distance": len(missing),
                         "phrase": html.unescape(v.get("neighbour_phrase") or ""), "coded_under": "pass 3, before the terms were defined"})
    by_element = {}
    for e in E:
        at1 = [p for p in pool if p["missing"] == [e]]
        at1.sort(key=lambda p: (-p["pass"], p["id"]))
        by_element[e] = at1[0] if at1 else None
    fewest = min((p["distance"] for p in pool), default=None)
    nearest_overall = sorted([p for p in pool if p["distance"] == fewest], key=lambda p: (-p["pass"], p["id"]))[:3] if pool else []
    named = [by_element[e] for e in E if by_element[e]] or nearest_overall
    clauses["C4a"]["nearness"] = {"by_element": by_element, "fewest_missing": fewest, "nearest_overall": nearest_overall,
                                  "named_in_sentence": [p["id"] for p in named],
                                  "rule": "per element, the source at distance one missing only that element (pass 4 first); if none anywhere, the sources at the smallest distance, at most three"}

    # the sentence
    c = {k: clauses[k]["status"] for k in clauses}
    if c.get("C4a") == "RETIRED" or c.get("C5") == "RETIRED":
        sentence = {"status": "RETIRED", "text": "RETIRED", "rule": "sentence rule 1: C4a or C5 retired"}
    elif c.get("C4a") == "UNPRICED":
        sentence = {"status": "UNLICENSED", "text": "UNLICENSED",
                    "rule": "the fingerprint clause is UNPRICED (disputed or unchecked); a sentence without it is not the sand's sentence"}
    else:
        def lacks(p):
            return " and ".join(ENAME[x] for x in p["missing"])
        c4a = P3_PARTS["C4a_head"] + " (" + "; ".join(f"where {p['phrase'].rstrip('.')}, without {lacks(p)}" for p in named) + ")"
        parts = [P3_PARTS["C1"]]
        if c.get("C2") == "OCCUPIED":
            parts.append(P3_PARTS["C2"])
        parts.append(c4a)
        if c.get("C5") == "OCCUPIED":
            parts.append(P3_PARTS["C5"])
        text = "We know of no lab that " + ", ".join(parts[:-1]) + ", and " + parts[-1] + " — at once."
        deleted = [k for k in ("C2",) if c.get(k) != "OCCUPIED"]
        sentence = {"status": "SURVIVES" if not deleted else "SURVIVES_WITHOUT_" + "_".join(deleted), "text": text,
                    "licensed_form": "we know of no ... (never 'first', 'novel' or 'revolutionary')",
                    "pass3_text": p3["sentence"]["text"], "changed_from_pass3": text != p3["sentence"]["text"]}

    pass4_c4a = [s["verdicts"]["C4a"] for s in sources.values() if s["status"] == "READ" and "elements" in s["verdicts"].get("C4a", {})]
    quotes = [el for v in pass4_c4a for el in v["elements"].values() if el["value"]]
    counts = {
        "sources_in_list": len(lst),
        "part_a": sum(1 for x in lst if x.get("part") == "A"), "part_b": sum(1 for x in lst if x.get("part") == "B"),
        "read": sum(1 for s in sources.values() if s["status"] == "READ"),
        "skimmed": sum(1 for s in sources.values() if s["status"] == "SKIMMED"),
        "unfetchable": sum(1 for s in sources.values() if s["status"] == "UNFETCHABLE"),
        "unread": sum(1 for s in sources.values() if s["status"] == "UNREAD"),
        "shown_last_line": sum(1 for s in sources.values() if (s.get("proof_of_reading") or {}).get("last_line_ok")),
        "shown_midpoint": sum(1 for s in sources.values() if (s.get("proof_of_reading") or {}).get("midpoint_ok")),
        "c4a_candidate_retirements": sum(1 for v in pass4_c4a if v.get("distance") == 0),
        "c4a_confirmed_retirements": sum(1 for v in pass4_c4a if v.get("retirement", "").startswith("CONFIRMED")),
        "c4a_disputed": sum(1 for v in pass4_c4a if v.get("verdict") == "DISPUTED"),
        "c4a_element_true": {e: sum(1 for v in pass4_c4a if v["elements"][e]["value"]) for e in E + ["E6"]},
        "c4a_fewest_missing_pass4": min((v["distance"] for v in pass4_c4a), default=None),
        "c4a_fewest_missing_all_passes": fewest,
        "element_quotes": len(quotes), "element_quotes_found": sum(1 for q in quotes if q["quote_found"]),
        "search_results": sum(len(s.get("results", [])) for s in search.get("searches", [])),
        "search_distinct": search.get("n_distinct"),
        "search_failures": sum(len(s.get("failures", [])) for s in search.get("searches", [])),
        "search_included": len(search.get("list_b", [])) + len(search.get("below_cap", [])),
        "search_below_cap": len(search.get("below_cap", [])),
    }
    leads = sorted({ld for s in sources.values() for ld in s.get("leads", [])} |
                   {x["title"] for x in search.get("below_cap", [])})
    out = {"survey": "the sand: pass 4 — the terms defined, nearness counted, a frozen search beside the leads",
           "protocol": PROTOCOL, "protocol_commit": protocol_commit, "pass3": P3, "run_date": date.replace("_", "-"),
           "counts": counts, "clauses": clauses, "sentence": sentence, "sources": sources,
           "leads_not_scored": leads, "n_leads_not_scored": len(leads)}
    path = f"papers/plates/sand_prior_art_survey_pass4_{date}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote", path)
    print(json.dumps(counts, indent=0))
    print("clauses:", {k: clauses[k]["status"] for k in clauses})
    print("sentence:", sentence["status"])
    print(sentence["text"])


if __name__ == "__main__":
    main()
