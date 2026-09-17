#!/usr/bin/env python3
"""build_sand_survey_pass3.py — the third pass of the sand survey: seventeen of pass 2's forty leads,
closed as a list by PROTOCOL_sand_prior_art_pass3_2026_09_14.md before any fetch, read end to end.

    python papers/plates/build_sand_survey_pass3.py <fetch_record.json> <readings_dir> <date YYYY_MM_DD> <protocol_commit>
    # reads papers/plates/sand_prior_art_survey_pass2_2026_09_13.json (sworn, never edited), the fetch
    # record (URL, bytes, sha256, extracted text per source) and one JSON return per reader; writes
    # papers/plates/sand_prior_art_survey_pass3_<date>.json

The surveyor types no verdict into this file. Each reader returned, per source and clause, a verdict
(RETIRES / OCCUPIES / SILENT), the object, a verbatim quote, a one-sentence reason, and — for OCCUPIES —
whether the source is a nearer neighbour than pass 2's and a phrase to name it by. This script merges
those returns mechanically under the inherited rule: a clause is RETIRED if any source read end to end
RETIRES it; else UNPRICED if any candidate that might occupy it was UNFETCHABLE; else OCCUPIED. C3 and
C4b stay RETIRED (pass 2); C1 stays OCCUPIED (the 2026-09-05 survey). The sentence rule is pass 1's:
if C4a or C5 is RETIRED the sentence is RETIRED entirely; a retired or unpriced clause is deleted; an
occupied clause survives with its neighbours named — pass 2's, plus every pass-3 source its reader
marked nearer.
"""
import json
import os
import re
import sys

P2 = "papers/plates/sand_prior_art_survey_pass2_2026_09_13.json"
PROTOCOL = "papers/plates/PROTOCOL_sand_prior_art_pass3_2026_09_14.md"
LIST = {  # the closed list, verbatim from the protocol: id -> (title, who, year, might_occupy)
    "L01": ("Did the Model Change? Efficiently Assessing Machine Learning API Shifts", "Chen, Cai, Zaharia & Zou", 2021, ["C4a"]),
    "L02": ("HAPI: A Large-scale Longitudinal Dataset of Commercial ML API Predictions", "Chen, Jin, Eyuboglu, Ré, Zaharia & Zou", 2022, ["C4a"]),
    "L03": ("ChatLog: Recording and Analyzing ChatGPT Across Time", "Tu et al.", 2023, ["C4a"]),
    "L04": ("A Fingerprint for Large Language Models", "Yang & Wu", 2024, ["C4a"]),
    "L05": ("Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression", "Xu et al.", 2021, ["C4a"]),
    "L06": ("What Do Compressed Deep Neural Networks Forget?", "Hooker, Courville, Clark, Dauphin & Frome", 2019, ["C4a"]),
    "L07": ("Non-Determinism of \"Deterministic\" LLM Settings", "Atil et al.", 2024, ["C4a"]),
    "L08": ("Quantifying Variance in Evaluation Benchmarks", "Madaan et al.", 2024, ["C4a"]),
    "L09": ("LLMmap: Fingerprinting For Large Language Models", "Pasquini, Kornaropoulos & Ateniese", 2024, ["C4a"]),
    "L10": ("Hide and Seek: Fingerprinting Large Language Models with Evolutionary Learning", "Iourovitski, Sharma & Talwar", 2024, ["C4a"]),
    "L11": ("Dataset Inference: Ownership Resolution in Machine Learning", "Maini, Yaghini & Papernot", 2021, ["C4a"]),
    "L12": ("Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models", "Chen et al.", 2022, ["C4a"]),
    "L13": ("SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud", "Ghodsi, Gu & Garg", 2017, ["C4a"]),
    "L14": ("High Accuracy and High Fidelity Extraction of Neural Networks", "Jagielski, Carlini, Berthelot, Kurakin & Papernot", 2020, ["C4a"]),
    "L15": ("Is GPT-4 getting worse over time?", "Narayanan & Kapoor", 2023, ["C4a"]),
    "L16": ("Artifact Review and Badging — Current", "ACM", 2020, ["C5"]),
    "L17": ("NeurIPS 2019 Reproducibility Challenge", "Sinha, Pineau, Forde, Ke & Larochelle", 2020, ["C5"]),
}
PRICED = ["C2", "C4a", "C5"]
VERDICTS = {"RETIRES", "OCCUPIES", "SILENT"}

fetch_path, readings_dir, date, protocol_commit = sys.argv[1:5]
texts_dir = sys.argv[5] if len(sys.argv) > 5 else None   # the extracted full texts, for the verbatim-quote check


def _norm(t):
    for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("’", "'"), ("‘", "'"),
                 ("“", '"'), ("”", '"'), ("−", "-"), ("–", "-"), ("—", "-")):
        t = t.replace(a, b)
    t = re.sub(r"-\s*\n\s*", "", t)          # the PDF extraction breaks words at line ends with a hyphen
    return re.sub(r"\s+", " ", t).strip()


def quote_found(sid, quote):
    """True if the quote is in the extracted text after ligature, quote-mark and line-hyphen normalisation."""
    if not texts_dir:
        return None
    try:
        raw = open(os.path.join(texts_dir, sid + ".txt"), encoding="utf-8").read()
    except OSError:
        return None
    return _norm(quote) in _norm(raw)


p2 = json.load(open(P2, encoding="utf-8"))
fetch = json.load(open(fetch_path, encoding="utf-8"))

readings, readers = {}, []
for fn in sorted(os.listdir(readings_dir)):
    if not fn.endswith(".json"):
        continue
    r = json.load(open(os.path.join(readings_dir, fn), encoding="utf-8"))
    readers.append(r["reader"])
    for s in r["sources"]:
        assert s["id"] in LIST, f"{s['id']} is not on the closed list"
        assert s["id"] not in readings, f"{s['id']} read twice"
        readings[s["id"]] = {"reader": r["reader"], **s}

sources = {}
for sid, (title, who, year, might) in LIST.items():
    f = fetch.get(sid, {})
    src = {"title": title, "who": who, "year": year, "might_occupy": might,
           "url": f.get("url"), "url_effective": f.get("url_effective"), "http": f.get("http"),
           "content_type": f.get("content_type"), "bytes": f.get("bytes"), "sha256": f.get("sha256"),
           "fetched_at": f.get("fetched_at"), "kind": f.get("kind"), "pages": f.get("pages"),
           "fulltext": f.get("fulltext"), "verdicts": {}}
    if f.get("status") != "FETCHED":
        src["status"] = "UNFETCHABLE"
        for c in might:
            src["verdicts"][c] = {"verdict": "UNCHECKABLE"}
        sources[sid] = src
        continue
    rd = readings.get(sid)
    if rd is None:
        src["status"] = "UNREAD"          # fetched, assigned to no reader: recorded, never scored
        for c in might:
            src["verdicts"][c] = {"verdict": "UNSCORED"}
        sources[sid] = src
        continue
    src["status"] = "READ" if rd.get("read_end_to_end") else "SKIMMED"
    src["read_by"] = rd["reader"]
    src["chars_read"] = rd.get("chars_read")
    src["chars_available"] = (f.get("fulltext") or {}).get("chars")
    src["bearing_quotes"] = rd.get("bearing_quotes", [])
    src["leads"] = rd.get("leads", [])
    src["notes"] = rd.get("notes", "")
    for v in rd["verdicts"]:
        c = v["clause"]
        assert c in might, f"{sid}: reader scored {c}, which the list does not name for it"
        assert v["verdict"] in VERDICTS, f"{sid}/{c}: {v['verdict']!r}"
        verdict = v["verdict"]
        if verdict == "RETIRES" and src["status"] != "READ":
            verdict = "OCCUPIES"          # a SKIMMED source may not RETIRE (protocol)
            v = {**v, "downgraded_from": "RETIRES", "downgrade_reason": "SKIMMED sources may not RETIRE"}
        src["verdicts"][c] = {"verdict": verdict, "object": v.get("object"), "quote": v.get("quote"),
                              "quote_found_in_text": quote_found(sid, v.get("quote", "")),
                              "reason": v.get("reason"), "elements": v.get("elements"),
                              "nearer_than_pass2_neighbour": bool(v.get("nearer_than_pass2_neighbour")),
                              "nearer_reason": v.get("nearer_reason", ""),
                              "neighbour_phrase": v.get("neighbour_phrase", "")}
        if "downgraded_from" in v:
            src["verdicts"][c]["downgraded_from"] = v["downgraded_from"]
            src["verdicts"][c]["downgrade_reason"] = v["downgrade_reason"]
    for c in might:
        assert c in src["verdicts"], f"{sid}: no verdict for {c}"
    sources[sid] = src

clauses = json.loads(json.dumps(p2["clauses"]))
for cid, cl in clauses.items():
    cl["pass2_status"] = cl["status"]
    cl.setdefault("retired_by", []); cl.setdefault("occupied_by", []); cl.setdefault("nearest", [])
    cl["retired_by_pass3"], cl["occupied_by_pass3"], cl["nearer_pass3"], cl["unfetchable_pass3"] = [], [], [], []
for sid, src in sources.items():
    for c, v in src["verdicts"].items():
        cl = clauses[c]
        if v["verdict"] == "UNCHECKABLE":
            cl["unfetchable_pass3"].append(sid)
        elif v["verdict"] == "RETIRES":
            cl["retired_by_pass3"].append(sid)
        elif v["verdict"] == "OCCUPIES":
            cl["occupied_by_pass3"].append(sid)
            if v["nearer_than_pass2_neighbour"]:
                cl["nearer_pass3"].append(sid)
for cid, cl in clauses.items():
    if cid not in PRICED:
        cl["pass3"] = "not re-priced"
        continue
    if cl["pass2_status"] == "RETIRED" or cl["retired_by_pass3"]:
        cl["status"] = "RETIRED"
    elif cl["unfetchable_pass3"]:
        cl["status"] = "UNPRICED"
    else:
        cl["status"] = "OCCUPIED"         # pass 2 left no priced clause FREE; pass 3 cannot free one
    cl["retired_by"] = cl["retired_by"] + cl["retired_by_pass3"]
    cl["occupied_by"] = cl["occupied_by"] + cl["occupied_by_pass3"]
    cl["nearest"] = cl["nearest"] + cl["nearer_pass3"]
    cl["nearer_pass3_joined"] = ", ".join(cl["nearer_pass3"])
    if cid == "C4a":
        els = ["self_comparison", "fixed_item_set", "same_weights_floor_in_situ", "interval_on_comparison", "log_prob_object"]
        rows = {sid: (sources[sid]["verdicts"]["C4a"].get("elements") or {}) for sid in sources if "C4a" in sources[sid]["verdicts"]}
        cl["elements_true_counts"] = {e: sum(1 for r in rows.values() if r.get(e)) for e in els}
        cl["sources_with_all_five_elements"] = [sid for sid, r in rows.items() if all(r.get(e) for e in els)]
        cl["n_sources_with_all_five_elements"] = len(cl["sources_with_all_five_elements"])
        cl["elements_by_source"] = {sid: {e: bool(r.get(e)) for e in els} for sid, r in rows.items()}

statuses = [clauses[c]["status"] for c in ("C1", "C2", "C3", "C4a", "C4b", "C5")]
order = ["RETIRED", "UNPRICED", "OCCUPIED", "FREE"]
conj = {"status": order[min(order.index(s) for s in statuses)],
        "one_source_does_all": [],
        "surviving_clauses": [c for c in ("C1", "C2", "C4a", "C5") if clauses[c]["status"] == "OCCUPIED"],
        "deleted_clauses": [c for c in ("C1", "C2", "C3", "C4a", "C4b", "C5") if clauses[c]["status"] in ("RETIRED", "UNPRICED")]}

# the sentence: pass 2's text is kept verbatim for every clause whose status did not change and whose
# reader marked no nearer neighbour; a nearer neighbour is appended inside that clause's parenthesis
# by the phrase the reader returned; a clause that fell is deleted; C4a or C5 falling retires the sentence.
if clauses["C4a"]["status"] == "RETIRED" or clauses["C5"]["status"] == "RETIRED":
    sentence = {"status": "RETIRED", "licensed_form": "none: rule 1 of the sentence rule (C4a or C5 retired)", "text": "RETIRED",
                "retired_by": {c: clauses[c]["retired_by"] for c in ("C4a", "C5") if clauses[c]["status"] == "RETIRED"}}
else:
    p2_text = p2["sentence"]["text"]
    parts = {
        "C1": "binds every published number to bytes at a commit (where Deterministic Integrity Gates and Cited-but-Not-Verified bind sentences and citations of a manuscript, as the 2026-09-05 survey names them)",
        "C2": "re-derives every verdict from those bytes into a chained log (where Certificate Transparency and Rekor chain certificates and signed metadata, not verdicts)",
        "C4a": "fingerprints a model's behavior on hashed canaries against a measured null floor (where Dutta et al. measure flips and KL divergence under compression, Chen, Zaharia and Zou grade a service's drift against its own repeat-run disagreement, and Thinking Machines measure the same-weights spread of served completions)",
        "C5": "pays a standing bounty against its own verifier (where Immunefi pays against deployed code and the Preregistration Challenge paid for preregistering)",
    }
    for c, part in parts.items():
        assert part in p2_text, f"pass 2's sentence no longer carries the {c} clause verbatim"
    for c in PRICED:
        phrases = [sources[s]["verdicts"][c]["neighbour_phrase"].strip().rstrip(".") for s in clauses[c]["nearer_pass3"]
                   if sources[s]["verdicts"][c].get("neighbour_phrase", "").strip()]
        if phrases and clauses[c]["status"] == "OCCUPIED":
            parts[c] = parts[c][:-1] + ", and " + ", ".join(phrases) + ")"
    surviving = [parts[c] for c in ("C1", "C2", "C4a", "C5") if clauses[c]["status"] == "OCCUPIED"]
    text = "We know of no lab that " + ", ".join(surviving[:-1]) + (", and " if len(surviving) > 1 else "") + surviving[-1] + " — at once."
    sentence = {"status": "SURVIVES_WITHOUT_" + "_".join(conj["deleted_clauses"]) if conj["deleted_clauses"] else "SURVIVES",
                "licensed_form": p2["sentence"]["licensed_form"], "text": text, "pass2_text": p2_text,
                "changed_from_pass2": text != p2_text, "notes": p2["sentence"].get("notes", [])}

leads = sorted({ld for s in sources.values() for ld in s.get("leads", [])})
counts = {"sources_in_list": len(LIST), "sources_scored": sum(1 for s in sources.values() if s["status"] in ("READ", "SKIMMED")),
          "read": sum(1 for s in sources.values() if s["status"] == "READ"),
          "skimmed": sum(1 for s in sources.values() if s["status"] == "SKIMMED"),
          "unfetchable": sum(1 for s in sources.values() if s["status"] == "UNFETCHABLE"),
          "unread": sum(1 for s in sources.values() if s["status"] == "UNREAD"),
          "verdicts_retires": sum(1 for s in sources.values() for v in s["verdicts"].values() if v["verdict"] == "RETIRES"),
          "verdicts_occupies": sum(1 for s in sources.values() for v in s["verdicts"].values() if v["verdict"] == "OCCUPIES"),
          "verdicts_silent": sum(1 for s in sources.values() for v in s["verdicts"].values() if v["verdict"] == "SILENT"),
          "nearer_neighbours_named": sum(len(clauses[c]["nearer_pass3"]) for c in PRICED),
          "leads_not_scored": len(leads),
          "clauses_retired": sum(1 for c in clauses.values() if c["status"] == "RETIRED"),
          "clauses_occupied": sum(1 for c in clauses.values() if c["status"] == "OCCUPIED"),
          "clauses_unpriced": sum(1 for c in clauses.values() if c["status"] == "UNPRICED"),
          "clauses_free": sum(1 for c in clauses.values() if c["status"] == "FREE"),
          "chars_read_total": sum(s.get("chars_read") or 0 for s in sources.values()),
          "verdict_quotes_checked": sum(1 for s in sources.values() for v in s["verdicts"].values() if v.get("quote_found_in_text") is not None),
          "verdict_quotes_found_in_text": sum(1 for s in sources.values() for v in s["verdicts"].values() if v.get("quote_found_in_text"))}

out = {"survey": "the sand: the neighbours of one sentence — pass 3, seventeen of pass 2's leads read end to end",
       "pass1": p2["pass1"], "pass2": P2, "protocol": PROTOCOL, "protocol_commit": protocol_commit,
       "run_by": "four reading agents, one per group of sources, each reading the extracted full text end to end; verdicts merged by this script; the surveyor fetched and extracted, and typed no verdict",
       "reading_rule": "only a source read end to end may RETIRE; a SKIMMED source may OCCUPY from the pages read; C3 and C4b are not re-priced (RETIRED by pass 2); C1 is not re-priced (OCCUPIED by the 2026-09-05 survey)",
       "c4a_object_stated_before_reading": "a served model's behaviour, compared with its own earlier or differently-served self, on a fixed item set, graded against a floor measured on the same weights under the serving in use, with an interval on the comparison",
       "run_date": date.replace("_", "-"), "status": "COMPLETE" if counts["unread"] == 0 else "INCOMPLETE",
       "counts": counts, "clauses": clauses, "conjunction": conj, "sentence": sentence,
       "sources": sources, "leads_not_scored": leads, "readers": readers}
path = f"papers/plates/sand_prior_art_survey_pass3_{date}.json"
with open(path, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
    fh.write("\n")
print("wrote", path)
print(json.dumps(counts, indent=0))
print("clauses:", {c: clauses[c]["status"] for c in clauses})
print("sentence:", sentence["status"])
print(sentence["text"])
