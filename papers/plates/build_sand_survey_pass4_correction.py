#!/usr/bin/env python3
"""build_sand_survey_pass4_correction.py — the correction record of pass 4 of the sand survey, built mechanically
under PROTOCOL_sand_pass4_correction_2026_09_15.md.

    python papers/plates/build_sand_survey_pass4_correction.py <pass4_inputs> <correction_inputs> <date> \
        <protocol_commit> <pass4_texts|-> <correction_texts|->
    # writes papers/plates/sand_prior_art_survey_pass4_correction_<date>.json

Inputs:
- From <pass4_inputs> (papers/plates/sand_survey_pass4_inputs): list.json, search_record.json, fetch_record*.json,
  fetched_text_scope*.json and readings*.json.
- From <correction_inputs> (papers/plates/sand_survey_pass4_correction_inputs): list_correction.json (the earlier
  passes' fingerprint sources re-read), fetch_record*.json (their bytes, and located article text for pass-4 sources,
  marked located_for_scope), fetched_text_scope*.json and readings*.json.
- The pass-2 and pass-3 records supply the earlier clause statuses and the titles of every listed source.

A text directory given as "-" skips the proofs of reading and the quote checks for that origin.

Rules, numbered as in the protocol:
1. READ, SKIMMED and UNFETCHABLE are decided as pass 4 applied them (text_checks is imported unchanged).
   - A source in a scope file is SKIMMED unless its reading is of a located copy fetched under the correction.
   - A SKIMMED source's SILENT or OCCUPIES is kept and marked from_abstract.
   - A SKIMMED source's RETIRES, or its C4a carrying E1-E5, is recorded as OCCUPIES from abstract, with the
     reader's verdict kept beside it.
2. An element is carried only when it is coded true and its quote is found in the text that was read.
3. A retirement needs two confirmers. They must be distinct from each other and from the first reader, each must pass
   both proofs, and each must carry E1-E5 (for C4a) or write RETIRES (for C2 or C5). Otherwise the source is
   DISPUTED. The same confirmation appearing in two files is refused. A C4a RETIRES that does not carry E1-E5 is
   recorded as OCCUPIES.
4. Clause pricing:
   - A clause is priced from every source that returned a verdict on it.
   - A confirmed retirement retires the clause.
   - An UNFETCHABLE or unscored listed candidate, or a DISPUTED source, makes the clause UNPRICED.
   - Any DISPUTED source makes the sentence UNLICENSED.
   - A verdict word outside RETIRES / OCCUPIES / SILENT is refused.
5. Nearness:
   - Only READ sources coded under the pass-4 definitions enter; pass-3 flags never do.
   - For each element, every source at distance one without only that element is named.
   - If no source is at distance one, every source at the smallest distance is named.
   - More than five names is refused, and the sentence is UNLICENSED.
6. The conjunction's status is computed per the 2026-09-13 protocol.
7. Leads are normalised. Any lead naming a source listed in passes 1-4 is dropped. Exact duplicates and leads that
   share an arXiv id are merged. Every count is recorded.
8. The sentence follows the inherited rules and rule 4's UNLICENSED.
   - Unchecked candidates that could retire the sentence are listed.
   - Only a trailing ", without ..." clause is removed from a phrase.
   - Only five HTML entities are unescaped.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sand_survey_pass4 import ENAME, P3_PARTS, norm, text_checks  # noqa: E402

P2 = "papers/plates/sand_prior_art_survey_pass2_2026_09_13.json"
P3 = "papers/plates/sand_prior_art_survey_pass3_2026_09_14.json"
PROTOCOL = "papers/plates/PROTOCOL_sand_pass4_correction_2026_09_15.md"
E = ["E1", "E2", "E3", "E4", "E5"]
VERDICTS = ("RETIRES", "OCCUPIES", "SILENT")
WEAKEST_FIRST = ["RETIRED", "UNPRICED", "OCCUPIED", "FREE"]
ENTITIES = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&amp;", "&")]
ARXIV = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
CLAUSE_ORDER = ["C1", "C2", "C3", "C4a", "C4b", "C5"]


class Refused(Exception):
    pass


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def unescape(s):
    for k, v in ENTITIES:
        s = s.replace(k, v)
    return s


def bare(phrase):
    return re.sub(r",\s+without\b[^,]*$", "", unescape(phrase or "").strip().rstrip("."))


def title_key(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def content_key(c):
    fields = {k: c.get(k) for k in ("id", "clause", "verdict", "elements", "last_line_quote", "midpoint_quote", "object", "reason")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def proofs_pass(chk):
    return chk is None or bool(chk.get("text_present") and chk.get("last_line_ok") and chk.get("midpoint_ok"))


def elements_of(v, chk):
    els = {}
    for e in E + ["E6"]:
        ev = (v.get("elements") or {}).get(e) or {}
        value, quote = bool(ev.get("value")), ev.get("quote", "") or ""
        if not value or chk is None:
            found = None
        else:
            found = bool(chk.get("text_present")) and bool(norm(quote)) and norm(quote) in chk["_whole"]
        els[e] = {"value": value, "quote": quote, "quote_found": found, "carried": value and found is not False}
    return els, [e for e in E if not els[e]["carried"]]


def merge_fetch(dirs):
    fetch = {}
    for origin, d in dirs:
        for fn in sorted(f for f in os.listdir(d) if f.startswith("fetch_record") and f.endswith(".json")):
            for sid, entry in load(os.path.join(d, fn)).items():
                entry = dict(entry, fetch_record=f"{origin}/{fn}", fetch_origin=origin)
                prev = fetch.get(sid)
                if prev is None:
                    fetch[sid] = entry
                    continue
                located = origin == "correction" and entry.get("located_for_scope")
                if prev.get("status") == "FETCHED" and not located:
                    raise Refused(f"{sid} fetched in two records")
                earlier = prev.get("earlier_attempts", []) + [{k: prev.get(k) for k in ("fetch_record", "status", "url", "sha256", "attempts", "reason")}]
                if prev.get("status") == "FETCHED" and entry.get("status") != "FETCHED":
                    # a located copy that could not be fetched leaves the page already fetched in force
                    fetch[sid] = dict(prev, located_attempt={k: entry.get(k) for k in ("fetch_record", "status", "reason", "attempts", "located_by")})
                    continue
                fetch[sid] = dict(entry, earlier_attempts=earlier)
    return fetch


def load_runs(origin, d):
    readings, confirmations = [], []
    for fn in sorted(f for f in os.listdir(d) if f.startswith("readings") and f.endswith(".json")):
        run = f"{origin}/{fn[: -len('.json')]}"
        part = load(os.path.join(d, fn))
        for r in part.get("readings", []):
            for s in r.get("sources", []):
                readings.append(dict(s, reader=f"{run}:{r['reader']}", origin=origin))
        for c in part.get("confirmations", []):
            confirmations.append(dict(c, confirmer=f"{run}:{c.get('confirmer')}", origin=origin))
    return readings, confirmations


def build(argv):
    p4_inputs, c_inputs, date, protocol_commit, p4_texts, c_texts = argv[1:7]
    texts = {"pass4": None if p4_texts == "-" else p4_texts, "correction": None if c_texts == "-" else c_texts}
    lst4 = load(os.path.join(p4_inputs, "list.json"))
    lstc = load(os.path.join(c_inputs, "list_correction.json"))
    search = load(os.path.join(p4_inputs, "search_record.json"))
    p2, p3 = load(P2), load(P3)

    meta = {}
    for x in lst4:
        meta[x["id"]] = dict(x, listed_in="pass 4")
    for x in lstc:
        if x["id"] in meta:
            raise Refused(f"{x['id']} listed twice")
        meta[x["id"]] = dict(x, listed_in=f"pass {x.get('pass')}")

    fetch = merge_fetch([("pass4", p4_inputs), ("correction", c_inputs)])
    scopes = {}
    for origin, d in (("pass4", p4_inputs), ("correction", c_inputs)):
        for fn in sorted(f for f in os.listdir(d) if f.startswith("fetched_text_scope") and f.endswith(".json")):
            for k, v in load(os.path.join(d, fn)).items():
                if not k.startswith("_"):
                    scopes[k] = dict(v, scope_file=f"{origin}/{fn}")

    r4, c4 = load_runs("pass4", p4_inputs)
    rc, cc = load_runs("correction", c_inputs)
    seen = {}
    for c in c4 + cc:
        key, run = content_key(c), c["confirmer"].rsplit(":", 1)[0]
        if key in seen and seen[key] != run:
            raise Refused(f"the same confirmation of {c.get('id')} appears twice, in {seen[key]} and {run}")
        seen.setdefault(key, run)
    first = {}
    for r in r4 + rc:
        slot = first.setdefault(r["id"], {})
        if r["origin"] in slot:
            raise Refused(f"{r['id']} read twice in {r['origin']}")
        slot[r["origin"]] = r
    confirms = {}
    for c in c4 + cc:
        confirms.setdefault((c["origin"], c["id"], c.get("clause") or "C4a"), []).append(c)

    sources = {}
    for sid, m in meta.items():
        f = fetch.get(sid, {})
        might = list(m.get("might_occupy") or [])
        src = {k: m.get(k) for k in ("title", "who", "year", "part", "pass", "listed_in", "urls")}
        src["might_occupy"] = might
        src.update({k: f.get(k) for k in ("url", "url_effective", "http", "bytes", "sha256", "kind", "pages", "fulltext", "fetch_record",
                                         "earlier_attempts", "located_attempt", "located_by", "source_bytes")})
        if f.get("status") != "FETCHED":
            src["status"] = "UNFETCHABLE"
            src["unfetchable_reason"] = f.get("reason") or "no fetch record"
            src["verdicts"] = {c: {"verdict": "UNCHECKABLE"} for c in might}
            sources[sid] = src
            continue
        slot = first.get(sid, {})
        r = slot.get("correction") or slot.get("pass4")
        if slot.get("correction") and slot.get("pass4"):
            src["superseded_readings"] = [slot["pass4"]]
        if r is None:
            src["status"] = "UNREAD"
            src["verdicts"] = {c: {"verdict": "UNSCORED"} for c in might}
            sources[sid] = src
            continue
        origin = r["origin"]
        tdir = texts[origin]
        chk = text_checks(tdir, sid, r)
        src["status"] = "READ" if (r.get("read_end_to_end") and proofs_pass(chk)) else "SKIMMED"
        src["read_by"], src["reading_origin"] = r["reader"], origin
        src["proof_of_reading"] = None if chk is None else {
            "last_line_ok": chk.get("last_line_ok"), "midpoint_ok": chk.get("midpoint_ok"),
            "last_line_is_a_page_number": bool(re.fullmatch(r"\s*\d+\s*", r.get("last_line_quote", "") or ""))}
        located_read = origin == "correction" and f.get("fetch_origin") == "correction" and bool(f.get("located_for_scope"))
        if sid in scopes and not located_read:
            if src["status"] == "READ":
                src["status"] = "SKIMMED"
            src["skimmed_reason"] = f"the fetched bytes are {scopes[sid]['scope']}: {scopes[sid]['reason']}"
        src["bearing_quotes"], src["leads"], src["notes"] = r.get("bearing_quotes", []), r.get("leads", []), r.get("notes", "")
        src["verdicts"] = {}
        for v in r.get("verdicts", []):
            c = v.get("clause")
            if v.get("verdict") not in VERDICTS:
                raise Refused(f"{sid}: verdict word {v.get('verdict')!r} on {c}")
            entry = {"verdict": v["verdict"], "reader_verdict": v["verdict"], "object": v.get("object"), "reason": v.get("reason"),
                     "neighbour_phrase": v.get("neighbour_phrase")}
            if c == "C4a" and isinstance(v.get("elements"), dict):
                els, missing = elements_of(v, chk)
                entry.update(elements=els, missing=missing, distance=len(missing))
            if c == "C4a":
                candidate = entry.get("distance") == 0
                if v["verdict"] == "RETIRES" and not candidate:
                    entry["verdict"] = "OCCUPIES"
                    entry["verdict_note"] = "a C4a RETIRES that does not carry E1-E5 is recorded as OCCUPIES"
            else:
                candidate = v["verdict"] == "RETIRES"
            if src["status"] != "READ":
                if candidate or entry["verdict"] == "RETIRES":
                    entry["verdict"] = "OCCUPIES"
                    entry["skimmed_rule"] = "a SKIMMED source may only be recorded as SILENT or as OCCUPIES from abstract; it may not RETIRE"
                entry["from_abstract"] = True
            elif candidate:
                rows = []
                for cf in confirms.get((origin, sid, c), []):
                    cchk = text_checks(tdir, sid, cf)
                    if c == "C4a":
                        _, cmissing = elements_of(cf, cchk)
                        carries = not cmissing
                    else:
                        cmissing, carries = None, cf.get("verdict") == "RETIRES"
                    rows.append({"confirmer": cf["confirmer"], "verdict": cf.get("verdict"), "carries": carries, "missing": cmissing,
                                 "shown_read_to_end": proofs_pass(cchk), "object": cf.get("object"), "reason": cf.get("reason"),
                                 "content_sha256": content_key(cf)})
                good_rows = [x for x in rows if x["carries"] and x["shown_read_to_end"] and x["confirmer"] != r["reader"]]
                good = {x["confirmer"] for x in good_rows}
                entry["confirmations"] = rows
                entry["distinct_confirmers_carrying"] = sorted(good)
                # distinct confirmers: two names AND two different returns (one return under two names is one confirmation)
                if len(good) >= 2 and len({x["content_sha256"] for x in good_rows}) >= 2:
                    entry["verdict"], entry["retirement"] = "RETIRES", "CONFIRMED by two distinct blind readers"
                else:
                    entry["verdict"], entry["retirement"] = "DISPUTED", "fewer than two distinct confirmers carried it and passed both proofs"
            src["verdicts"][c] = entry
        for c in might:
            src["verdicts"].setdefault(c, {"verdict": "UNSCORED", "reason": "the reader returned no verdict for this clause"})
        sources[sid] = src

    clauses = {k: {"clause": v.get("clause"), "status": v["status"], "pass3_status": v["status"]} for k, v in p3["clauses"].items()}
    for cid in ("C2", "C4a", "C5"):
        b = {"retired_by": [], "disputed_by": [], "occupied_by": [], "occupied_from_abstract_by": [], "silent": [], "uncheckable": [], "unscored": []}
        for sid, s in sources.items():
            v = s["verdicts"].get(cid)
            if not v:
                continue
            w = v["verdict"]
            if w == "RETIRES":
                b["retired_by"].append(sid)
            elif w == "DISPUTED":
                b["disputed_by"].append(sid)
            elif w == "OCCUPIES":
                b["occupied_from_abstract_by" if v.get("from_abstract") else "occupied_by"].append(sid)
            elif w == "SILENT":
                b["silent"].append(sid)
            elif w == "UNCHECKABLE":
                b["uncheckable"].append(sid)
            elif w == "UNSCORED":
                b["unscored"].append(sid)
        for key in b:
            b[key].sort()
        cl = clauses[cid]
        cl["correction"] = b
        if cl["pass3_status"] == "RETIRED" or b["retired_by"]:
            cl["status"] = "RETIRED"
        elif b["disputed_by"] or b["uncheckable"] or b["unscored"]:
            cl["status"] = "UNPRICED"
        else:
            cl["status"] = "OCCUPIED"

    pool = []
    for sid in sorted(sources):
        s = sources[sid]
        v = s["verdicts"].get("C4a") or {}
        if s["status"] == "READ" and "distance" in v:
            pool.append({"id": sid, "listed_in": s["listed_in"], "who": s.get("who") or s.get("title"), "missing": v["missing"],
                         "distance": v["distance"], "phrase": bare(v.get("neighbour_phrase") or "")})
    at_one = {e: [p["id"] for p in pool if p["missing"] == [e]] for e in E}
    named = [p for e in E for p in pool if p["missing"] == [e]]
    fewest = min((p["distance"] for p in pool), default=None)
    rule_used = "per element, every READ source at distance one whose only missing element is that one"
    if not named and pool:
        named = [p for p in pool if p["distance"] == fewest]
        rule_used = "no source at distance one: every READ source at the smallest distance"
    nearness_refused = len(named) > 5
    clauses["C4a"]["nearness"] = {"pool": [p["id"] for p in pool], "at_distance_one_by_element": at_one, "fewest_missing": fewest,
                                  "named_in_sentence": [p["id"] for p in named], "named": named, "rule_used": rule_used,
                                  "refused_more_than_five": nearness_refused,
                                  "pass3_flags": "not used: pass 3 coded its elements before the terms were defined and quoted no element"}

    statuses = {k: clauses[k]["status"] for k in CLAUSE_ORDER if k in clauses}
    does_all = sorted(sid for sid, s in sources.items() if s["status"] == "READ" and all(
        (s["verdicts"].get(k) or {}).get("verdict") in ("RETIRES", "OCCUPIES") for k in ("C2", "C3", "C4a", "C4b", "C5")))
    conjunction = {"status": "RETIRED" if does_all else min(statuses.values(), key=WEAKEST_FIRST.index),
                   "one_source_does_all": does_all,
                   "surviving_clauses": [k for k in CLAUSE_ORDER if statuses.get(k) == "OCCUPIED"],
                   "deleted_clauses": [k for k in CLAUSE_ORDER if statuses.get(k) in ("RETIRED", "UNPRICED")],
                   "rule": "RETIRED if one READ source retires or occupies every one of C2-C5; otherwise the weakest status among C1-C5 (RETIRED > UNPRICED > OCCUPIED > FREE)"}

    disputed = sorted({sid for k in ("C2", "C4a", "C5") for sid in clauses[k]["correction"]["disputed_by"]})
    unchecked = sorted({sid for k in ("C4a", "C5") for sid in clauses[k]["correction"]["uncheckable"] + clauses[k]["correction"]["unscored"]})
    c = statuses
    if c["C4a"] == "RETIRED" or c["C5"] == "RETIRED":
        sentence = {"status": "RETIRED", "text": "RETIRED", "rule": "sentence rule 1: C4a or C5 retired"}
    elif disputed:
        sentence = {"status": "UNLICENSED", "text": "UNLICENSED", "rule": "a disputed retirement: " + ", ".join(disputed)}
    elif c["C4a"] == "UNPRICED":
        sentence = {"status": "UNLICENSED", "text": "UNLICENSED", "rule": "the fingerprint clause is UNPRICED; a sentence without it is not the sand's sentence"}
    elif nearness_refused:
        sentence = {"status": "UNLICENSED", "text": "UNLICENSED", "rule": "the nearness rule would name more than five sources"}
    else:
        def lacks(p):
            return " and ".join(ENAME[x] for x in p["missing"])
        c4a = P3_PARTS["C4a_head"] + " (" + "; ".join(f"where {p['phrase']}, without {lacks(p)}" for p in named) + ")"
        parts = [P3_PARTS["C1"]]
        if c["C2"] == "OCCUPIED":
            parts.append(P3_PARTS["C2"])
        parts.append(c4a)
        if c["C5"] == "OCCUPIED":
            parts.append(P3_PARTS["C5"])
        text = "We know of no lab that " + ", ".join(parts[:-1]) + ", and " + parts[-1] + " — at once."
        deleted = [k for k in ("C2", "C5") if c[k] != "OCCUPIED"]
        sentence = {"status": "SURVIVES" if not deleted else "SURVIVES_WITHOUT_" + "_".join(deleted), "text": text,
                    "licensed_form": "we know of no ... (never 'first', 'novel' or 'revolutionary')"}
    sentence["unchecked_candidates_that_could_retire_the_sentence"] = unchecked
    sentence["c2_neighbour"] = "inherited (Certificate Transparency and Rekor); no pass-4 reader was asked for a 'nearer' flag"

    raw = set()
    for s in sources.values():
        raw.update(s.get("leads") or [])
        for sr in s.get("superseded_readings") or []:
            raw.update(sr.get("leads") or [])
    raw.update(x["title"] for x in search.get("below_cap", []))
    listed = [title_key(s.get("title")) for s in p2["sources"].values()] + [title_key(s.get("title")) for s in p3["sources"].values()] + \
        [title_key(x.get("title")) for x in lst4 + lstc]
    listed = sorted({t for t in listed if len(t.split()) >= 3})
    kept, dropped = [], []
    for lead in sorted(raw):
        k = f" {title_key(lead)} "
        hit = next((t for t in listed if f" {t} " in k), None)
        (dropped if hit else kept).append({"lead": lead, "names_listed_source": hit} if hit else {"lead": lead})
    distinct = {}
    for x in kept:
        mm = ARXIV.search(x["lead"])
        key = f"arxiv:{mm.group(1)}" if mm else f"text:{title_key(x['lead'])}"
        distinct.setdefault(key, []).append(x["lead"])
    leads = {"raw": sorted(raw), "naming_a_listed_source": dropped, "kept": kept, "distinct": distinct,
             "rule": "normalised to lowercase alphanumerics; dropped when a listed source's title (3+ words) appears in it; merged when identical or sharing an arXiv id"}

    listed4 = [sid for sid in meta if meta[sid]["listed_in"] == "pass 4"]
    recoded = [sid for sid in meta if meta[sid]["listed_in"] != "pass 4"]

    def tally(ids):
        return {st: sum(1 for i in ids if sources[i]["status"] == st) for st in ("READ", "SKIMMED", "UNFETCHABLE", "UNREAD")}
    coded = [(sid, e, v["elements"][e]) for sid, s in sources.items() for v in [s["verdicts"].get("C4a") or {}] if "elements" in v for e in E + ["E6"]]
    counts = {
        "pass4_list": len(listed4), "pass4_list_status": tally(listed4),
        "recoded_earlier_sources": len(recoded), "recoded_status": tally(recoded),
        "read_from_a_located_copy": sorted(sid for sid, s in sources.items() if s.get("reading_origin") == "correction" and sid in scopes and s["status"] != "SKIMMED"),
        "superseded_readings": sum(1 for s in sources.values() if s.get("superseded_readings")),
        "c4a_pool": len(pool),
        "c4a_carried": {e: sum(1 for p in pool if e not in p["missing"]) for e in E},
        "c4a_fewest_missing": fewest,
        "element_quotes_coded_true": sum(1 for _, _, x in coded if x["value"]),
        "element_quotes_found": sum(1 for _, _, x in coded if x["value"] and x["quote_found"]),
        "element_quotes_not_found": sorted(f"{sid}:{e}" for sid, e, x in coded if x["value"] and x["quote_found"] is False),
        "end_proofs_that_are_page_numbers": sorted(sid for sid, s in sources.items() if (s.get("proof_of_reading") or {}).get("last_line_is_a_page_number")),
        "confirmations": len(c4) + len(cc),
        "disputed": disputed,
        "leads_raw": len(raw), "leads_naming_a_listed_source": len(dropped), "leads_kept": len(kept), "leads_distinct": len(distinct),
    }
    out = {"survey": "the sand, pass 4: the correction record", "protocol": PROTOCOL, "protocol_commit": protocol_commit,
           "pass4_protocol": "papers/plates/PROTOCOL_sand_prior_art_pass4_2026_09_15.md", "run_date": date.replace("_", "-"),
           "counts": counts, "clauses": clauses, "conjunction": conjunction, "sentence": sentence, "sources": sources, "leads": leads}
    path = f"papers/plates/sand_prior_art_survey_pass4_correction_{date}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote", path)
    print(json.dumps(counts, ensure_ascii=False, indent=0))
    print("clauses:", statuses, "| conjunction:", conjunction["status"])
    print("sentence:", sentence["status"])
    print(sentence["text"])


def main():
    try:
        build(sys.argv)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
