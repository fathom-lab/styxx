#!/usr/bin/env python3
"""build_sand_survey_pass2.py — the second pass of the sand survey: the skimmed papers read end to end,
and the first pass's own audit findings applied.

    python papers/plates/build_sand_survey_pass2.py <journal.jsonl> <date YYYY_MM_DD>
    # reads papers/plates/sand_prior_art_survey.json (pass 1, sworn, never edited) and the workflow journal
    # of the four full readers; writes papers/plates/sand_prior_art_survey_pass2_<date>.json

Pass 1 (2026-09-13) SKIMMED nine papers, so under the protocol none of them could RETIRE a clause. Four
readers then read those papers' full text (extracted from the PDFs whose sha256 pass 1 recorded) and
returned, per source and clause, a verdict, the object, a verbatim quote, a one-sentence reason, and
whether they agree with pass 1. This script merges those readings mechanically: a source read end to
end becomes READ; its verdicts for the clauses the reader covered replace pass 1's, with pass 1's kept
beside them; clause statuses, the conjunction and the sentence are recomputed under the same rule.

It also applies what pass 1's protocol audit found in pass 1 itself, each as a field, not a silent
edit: C1's neighbours are named (from the 2026-09-05 sworn survey's receipt, whose clause is about
binding whole sentences, a narrower wording than C1 here — recorded as such); the sentence is one
sentence, built from the surviving clauses with a neighbour for each, and the remark about the deleted
seal clause moves to a note; the fetch of S10 is recorded as the substitution it was; S06's snapshot is
recorded as fetched-but-empty; the reading rule "OCCUPIES from the pages read" is stated. The surveyor
typed no verdict into this file; the readers' verdicts are taken from the journal as returned.
"""
import json
import sys

pass1_path = "papers/plates/sand_prior_art_survey.json"
journal_path, date = sys.argv[1], sys.argv[2]
p1 = json.load(open(pass1_path, encoding="utf-8"))

readings, readers = {}, []
for line in open(journal_path, encoding="utf-8"):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("type") != "result":
        continue
    r = e.get("result") or {}
    if isinstance(r, dict) and "sources" in r and "reader" in r:
        readers.append(r["reader"])
        for s in r["sources"]:
            readings[s["id"]] = {"reader": r["reader"], **s}

sources = json.loads(json.dumps(p1["sources"]))
changed = []
for sid, rd in readings.items():
    src = sources[sid]
    src["pass1_status"] = src["status"]
    if rd.get("read_end_to_end"):
        src["status"] = "READ"
        src["pages_read"] = "end to end, full text extracted from the PDF whose sha256 pass 1 recorded"
    src["read_by"] = rd["reader"]
    src["chars_read"] = rd.get("chars_read")
    src["bearing_quotes"] = rd.get("bearing_quotes", [])
    src["leads"] = rd.get("leads", [])
    for v in rd["verdicts"]:
        cid = v["clause"]
        old = src["verdicts"].get(cid)
        src["verdicts"][cid] = {"verdict": v["verdict"], "object": v["object"], "quote": v["quote"], "reason": v["reason"],
                                "agrees_with_pass1": v["agrees_with_surveyor"], "pass1": old}
        if old is None or old.get("verdict") != v["verdict"]:
            changed.append(f"{sid}.{cid}: {old.get('verdict') if old else 'unscored'} -> {v['verdict']}")

# audit findings applied as fields
sources["S10"]["substitution"] = ("the closed list named the Preregistration Challenge programme page (2015-2018); the page located "
                                  "was the Center for Open Science's general preregistration page, which describes the campaign in "
                                  "passing; recorded as a substitution, not as the listed page")
sources["S06"]["fetch_note"] = ("located and fetched (HTTP 200, 3203 bytes): the 2021 archived page is a script shell whose only text "
                                "is its title; no earlier snapshot was tried; the protocol has no fetched-but-empty status, so "
                                "UNFETCHABLE stands and C4b keeps S06 as an unpriced candidate")
sources["S17"]["pass1_read_rule"] = "counted READ in pass 1 on the strength of the summariser having the whole PDF as input; read end to end in pass 2"

CLAUSE_IDS = ("C2", "C3", "C4a", "C4b", "C5")
C1_NEAREST = [
    "Deterministic Integrity Gates for LLM-Assisted Clinical Manuscript Preparation (Nam, Jeong & Kim, arXiv 2606.09500) — per the 2026-09-05 sworn survey, whose clause is 'binds whole sentences'",
    "Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents (Onweller et al., arXiv 2605.06635) — same receipt",
]
clauses = {"C1": {**p1["clauses"]["C1"], "nearest": C1_NEAREST,
                  "note": ("C1 as worded here ('binds every published number to bytes at a commit') is broader than the clause the "
                           "2026-09-05 survey priced ('binds whole sentences'); C1 is carried as OCCUPIED on that survey's authority and "
                           "was not re-priced by this one")}}
for cid in CLAUSE_IDS:
    retired, occupied, unfetchable = [], [], []
    for sid, rec in sources.items():
        v = rec["verdicts"].get(cid)
        if not v:
            continue
        if v["verdict"] == "RETIRES" and rec["status"] == "READ":
            retired.append(sid)
        elif v["verdict"] in ("OCCUPIES", "RETIRES"):
            occupied.append(sid)
        elif v["verdict"] == "UNCHECKABLE":
            unfetchable.append(sid)
    status = "RETIRED" if retired else ("UNPRICED" if unfetchable else ("OCCUPIED" if occupied else "FREE"))
    clauses[cid] = {"clause": p1["clauses"][cid]["clause"], "status": status, "retired_by": retired, "occupied_by": occupied,
                    "unfetchable_candidates": unfetchable, "nearest": p1["clauses"][cid]["nearest"], "pass1_status": p1["clauses"][cid]["status"]}

order = ["RETIRED", "UNPRICED", "OCCUPIED", "FREE"]
one_source_all = [sid for sid, rec in sources.items() if rec["status"] == "READ" and all(
    rec["verdicts"].get(c, {}).get("verdict") in ("OCCUPIES", "RETIRES") for c in CLAUSE_IDS)]
weakest = min([clauses[c]["status"] for c in ("C1",) + CLAUSE_IDS], key=order.index)
conj = "RETIRED" if one_source_all else weakest
surviving = [c for c in ("C1", "C2", "C3", "C4a", "C5") if clauses[c]["status"] == "OCCUPIED"]
deleted = [c for c in ("C1",) + CLAUSE_IDS if clauses[c]["status"] in ("RETIRED", "UNPRICED")]
fatal = any(clauses[c]["status"] == "RETIRED" for c in ("C4a", "C5"))

# the one licensed sentence: every surviving clause with its neighbour named beside it
PHRASES = {
    "C1": "binds every published number to bytes at a commit (where Deterministic Integrity Gates and Cited-but-Not-Verified bind sentences and citations of a manuscript, as the 2026-09-05 survey names them)",
    "C2": "re-derives every verdict from those bytes into a chained log (where Certificate Transparency and Rekor chain certificates and signed metadata, not verdicts)",
    "C3": "gives every receipt a face a stranger reads without json (where random art gives a host key one and an identicon gives an address one)",
    "C4a": "fingerprints a model's behavior on hashed canaries against a measured null floor (where Dutta et al. measure flips and KL divergence under compression, Chen, Zaharia and Zou grade a service's drift against its own repeat-run disagreement, and Thinking Machines measure the same-weights spread of served completions)",
    "C5": "pays a standing bounty against its own verifier (where Immunefi pays against deployed code and the Preregistration Challenge paid for preregistering)",
}
if fatal:
    sentence_text, sentence_status = None, "RETIRED"
else:
    parts = [PHRASES[c] for c in surviving]
    sentence_text = "We know of no lab that " + ", ".join(parts[:-1]) + (", and " if len(parts) > 1 else "") + parts[-1] + " — at once."
    sentence_status = "SURVIVES_WITHOUT_" + "_".join(deleted) if deleted else "SURVIVES"
notes = []
if clauses["C4b"]["status"] == "RETIRED":
    notes.append("the seal on a public chain is not in the sentence: general-purpose document time-stamping (Haber and Stornetta; OpenTimestamps) does that for any data, and a preregistration is data")
if clauses["C3"]["status"] == "RETIRED":
    notes.append("the face is not in the sentence: Perrig and Song defined hash visualization over inputs of arbitrary finite length and named a visual checksum of downloaded software as an application, which contains a receipt digest; the plate is that idea applied to receipts, and the sentence may not claim it")

leads = list(p1["leads_not_scored"])
for sid, rd in readings.items():
    for l in rd.get("leads", []):
        if l not in leads:
            leads.append(f"[{sid} full read] {l}")

counts = {"sources_in_list": len(sources), "sources_scored": len(sources),
          "read": sum(1 for r in sources.values() if r["status"] == "READ"),
          "skimmed": sum(1 for r in sources.values() if r["status"] == "SKIMMED"),
          "unfetchable": sum(1 for r in sources.values() if r["status"] == "UNFETCHABLE"),
          "read_in_pass2": len([s for s in readings.values() if s.get("read_end_to_end")]),
          "verdicts_changed_by_full_read": len(changed),
          "leads_not_scored": len(leads),
          "clauses_occupied": sum(1 for c in clauses.values() if c["status"] == "OCCUPIED"),
          "clauses_retired": sum(1 for c in clauses.values() if c["status"] == "RETIRED"),
          "clauses_unpriced": sum(1 for c in clauses.values() if c["status"] == "UNPRICED"),
          "clauses_free": sum(1 for c in clauses.values() if c["status"] == "FREE")}

out = {"survey": "the sand: the neighbours of one sentence — pass 2, the skimmed papers read end to end",
       "pass1": pass1_path, "protocol": p1["protocol"], "protocol_commit": p1["protocol_commit"],
       "run_by": "four reading agents, one per group of papers, each reading the extracted full text end to end; verdicts merged by this script; no fetch; pass 1's audit findings applied as fields",
       "reading_rule": "a SKIMMED source may OCCUPY from the pages read (pass 1 wrote 'from abstract'; the builder's rule was always the pages read); only a source read end to end may RETIRE",
       "run_date": date.replace("_", "-"), "status": "COMPLETE",
       "counts": counts, "verdicts_changed_by_full_read": changed, "clauses": clauses,
       "conjunction": {"status": conj, "one_source_does_all": one_source_all, "surviving_clauses": surviving, "deleted_clauses": deleted},
       "sentence": {"status": sentence_status, "licensed_form": p1["sentence"]["licensed_form"], "text": sentence_text, "notes": notes},
       "sources": sources, "leads_not_scored": leads, "readers": sorted(set(readers))}
path = f"papers/plates/sand_prior_art_survey_pass2_{date}.json"
with open(path, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(json.dumps(counts, indent=1))
print("changed:", changed)
print({c: v["status"] for c, v in clauses.items()}, conj, sentence_status)
print(sentence_text)
print("wrote", path)
