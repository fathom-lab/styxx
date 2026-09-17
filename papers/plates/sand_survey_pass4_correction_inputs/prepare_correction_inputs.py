#!/usr/bin/env python3
"""prepare_correction_inputs.py — the correction's texts and fetch record, per PROTOCOL_sand_pass4_correction_2026_09_15.md.

    python papers/plates/sand_survey_pass4_correction_inputs/prepare_correction_inputs.py <scratch_dir> <texts_out>

1. The earlier passes' fingerprint sources, from the bytes those passes hashed:
   - pass 2's S11-S14, S17 and S19 from the PDFs whose sha256 equals fulltext_pdf_sha256;
   - S15 and S16 from the HTML bytes whose sha256 equals sha256;
   - pass 3's L01-L15 from the texts whose sha256 equals fulltext.sha256.
   Texts are extracted with fetch_pass4.py's own extractor (PDF and HTML); pass-3 texts are copied byte for byte. A
   source whose bytes do not match its record gets status NOT_RECODED and is not read.
2. The pass-4 sources whose fetched text is not the article: located once by the routes the protocol fixes, using
   fetch_pass4.py's fetch, extraction and title check, with located_for_scope set. B12 has no open route and is
   recorded without a fetch. B23's DOIs, Europe PMC and archive captures are tried and recorded.
Writes list_correction.json, fetch_record_correction.json and fetched_text_scope_correction.json beside this file,
and the texts to <texts_out> (not committed).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PLATES, "sand_survey_pass4_inputs"))
import fetch_pass4 as fp  # noqa: E402

UA = fp.UA


def sha(b):
    return hashlib.sha256(b).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return {"file": os.path.basename(path), "chars": len(text), "sha256": sha(text.encode("utf-8"))}


def curl_json(url):
    r = subprocess.run(["curl", "-sL", "-A", UA, "--max-time", "60", url], capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return {"_unparsed": r.stdout[:300]}


def main():
    scratch, texts_out = sys.argv[1], sys.argv[2]
    os.makedirs(texts_out, exist_ok=True)
    p2 = json.load(open(os.path.join(PLATES, "sand_prior_art_survey_pass2_2026_09_13.json"), encoding="utf-8"))
    p3 = json.load(open(os.path.join(PLATES, "sand_prior_art_survey_pass3_2026_09_14.json"), encoding="utf-8"))
    lst4 = {x["id"]: x for x in json.load(open(os.path.join(PLATES, "sand_survey_pass4_inputs", "list.json"), encoding="utf-8"))}
    p4dir = os.path.join(PLATES, "sand_survey_pass4_inputs")
    p4fetch = {}
    for fn in sorted(f for f in os.listdir(p4dir) if f.startswith("fetch_record") and f.endswith(".json")):
        for k, v in json.load(open(os.path.join(p4dir, fn), encoding="utf-8")).items():
            if v.get("status") == "FETCHED":
                p4fetch[k] = v
    lst, record = [], {}

    for sid in ["S11", "S12", "S13", "S14", "S15", "S16", "S17", "S19"]:
        s = p2["sources"][sid]
        lst.append({"id": sid, "pass": 2, "title": s["title"], "who": s.get("who"), "year": s.get("year"), "url": s.get("url"), "might_occupy": ["C4a"]})
        pdf, binp = os.path.join(scratch, "survey", sid + ".pdf"), os.path.join(scratch, "survey", sid + ".bin")
        if os.path.exists(pdf):
            raw, want, path, which = open(pdf, "rb").read(), s.get("fulltext_pdf_sha256"), pdf, "pass-2 fulltext PDF (fulltext_pdf_sha256)"
        else:
            raw, want, path, which = open(binp, "rb").read(), s.get("sha256"), binp, "pass-2 fetched bytes (sha256)"
        if sha(raw) != want:
            record[sid] = {"status": "NOT_RECODED", "reason": f"{which} does not match its record", "source_bytes": {"which": which, "sha256": sha(raw)}}
            continue
        text, kind, pages = fp.extract(raw, path)
        frac, missing = fp.title_match(s["title"], text)
        ft = write_text(os.path.join(texts_out, sid + ".txt"), text)
        record[sid] = {"status": "FETCHED" if (len(text) > 1500 and frac >= 0.6) else "NOT_RECODED", "url": s.get("url"), "sha256": s.get("sha256"),
                       "source_bytes": {"which": which, "sha256": sha(raw), "bytes": len(raw), "matches_record": True},
                       "kind": kind, "pages": pages, "title_words_found": round(frac, 3), "title_words_missing": missing, "fulltext": ft}

    for i in range(1, 16):
        sid = f"L{i:02d}"
        s = p3["sources"][sid]
        lst.append({"id": sid, "pass": 3, "title": s["title"], "who": s.get("who"), "year": s.get("year"), "url": s.get("url"), "might_occupy": ["C4a"]})
        src = os.path.join(scratch, "survey3", sid + ".txt")
        raw = open(src, "rb").read()
        if sha(raw) != (s.get("fulltext") or {}).get("sha256"):
            record[sid] = {"status": "NOT_RECODED", "reason": "pass-3 text does not match fulltext.sha256", "source_bytes": {"sha256": sha(raw)}}
            continue
        shutil.copyfile(src, os.path.join(texts_out, sid + ".txt"))
        record[sid] = {"status": "FETCHED", "url": s.get("url"), "sha256": s.get("sha256"), "kind": s.get("kind"), "pages": s.get("pages"),
                       "source_bytes": {"which": "pass-3 extracted text (fulltext.sha256)", "sha256": sha(raw), "matches_record": True},
                       "fulltext": dict(s["fulltext"])}

    # located copies for the pass-4 sources whose fetched text is not the article
    b04_pmc = curl_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:42323341%20AND%20SRC:MED&format=json&resultType=core")
    b04_pmcid = ((b04_pmc.get("resultList") or {}).get("result") or [{}])[0].get("pmcid") if isinstance(b04_pmc, dict) else None
    b23_epmc = curl_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1126/science.aeh8588%22&format=json&resultType=core")
    b23_hit = ((b23_epmc.get("resultList") or {}).get("result") or [{}])[0] if isinstance(b23_epmc, dict) else {}
    b23_arch = [curl_json(f"https://archive.org/wayback/available?url={u}") for u in (
        "https://www.science.org/content/article/offering-scientists-cash-spot-errors-published-papers-doesn-t-work",
        "https://www.science.org/doi/10.1126/science.zh9l2q0", "https://www.science.org/doi/10.1126/science.aeh8588",
        "https://www.science.org/doi/full/10.1126/science.aeh8588")]
    b23_caps = [((a.get("archived_snapshots") or {}).get("closest") or {}).get("url") for a in b23_arch if isinstance(a, dict)]
    b23_caps = [u for u in b23_caps if u]
    located = [
        {"id": "B04", "urls": ["https://www.nature.com/articles/s41598-026-56100-9.pdf"] +
         ([f"https://www.ebi.ac.uk/europepmc/webservices/rest/{b04_pmcid}/fullTextXML"] if b04_pmcid else []),
         "located_by": f"the landing page's citation_pdf_url; Europe PMC by PMID 42323341 (PMCID found: {b04_pmcid})"},
        {"id": "B05", "urls": ["https://www.ijsat.org/papers/2026/1/10626.pdf"], "located_by": "the landing page's citation_pdf_url"},
        {"id": "B07", "urls": ["https://cspub-ijcisim.org/index.php/ijcisim/article/download/4475/3594"], "located_by": "the landing page's citation_pdf_url"},
        {"id": "B11", "urls": ["https://arxiv.org/pdf/2603.19022"], "located_by": "arXiv 2603.19022, the paper alphaXiv's page summarises"},
        {"id": "B23", "urls": ["https://doi.org/10.1126/science.zh9l2q0", "https://doi.org/10.1126/science.aeh8588"] +
         ([f"https://www.ebi.ac.uk/europepmc/webservices/rest/{b23_hit.get('pmcid')}/fullTextXML"] if b23_hit.get("pmcid") else []) + b23_caps,
         "located_by": ("Crossref: online DOI 10.1126/science.zh9l2q0 and print version 10.1126/science.aeh8588 (Chawla, Science 392:133, "
                        f"2026-04-09, retitled); Europe PMC {b23_hit.get('pmid')} pmcid {b23_hit.get('pmcid')} full text "
                        f"{b23_hit.get('inEPMC')}/{b23_hit.get('hasPDF')}; archive captures found: {len(b23_caps)}")},
    ]
    for loc in located:
        sid = loc["id"]
        title = lst4[sid]["title"]
        entry = {"title": title, "attempts": [], "status": "UNFETCHABLE", "located_for_scope": True, "located_by": loc["located_by"]}
        for i, url in enumerate(fp.candidates(loc["urls"])):
            path = os.path.join(texts_out, f"{sid}.located.{i}.bin")
            code, ctype, eff, err = fp.fetch(url, path)
            raw = open(path, "rb").read() if os.path.exists(path) else b""
            att = {"url": url, "url_effective": eff, "http": code, "content_type": ctype, "bytes": len(raw), "sha256": sha(raw) if raw else None,
                   "fetched_at": now(), "error": err or None}
            text = ""
            if code == "200" and len(raw) > 2000:
                try:
                    text, kind, pages = fp.extract(raw, path)
                    att["kind"], att["pages"] = kind, pages
                except Exception as e:  # noqa: BLE001
                    att["extract_error"] = repr(e)
            frac, missing = fp.title_match(title, text) if text else (0.0, fp.words(title))
            att["title_words_found"], att["title_words_missing"] = round(frac, 3), missing
            # a route that leads back to the page pass 4 already fetched (a PDF link redirecting to the landing page) is
            # not a located copy of the article's text
            same_page = bool(text) and sha(text.encode("utf-8")) == ((p4fetch.get(sid) or {}).get("fulltext") or {}).get("sha256")
            att["same_text_as_the_pass4_page"] = same_page
            att["accepted"] = len(text) > 1500 and frac >= 0.6 and not same_page
            entry["attempts"].append(att)
            if att["accepted"]:
                entry.update({k: v for k, v in att.items() if k != "error"})
                entry["fulltext"] = write_text(os.path.join(texts_out, sid + ".txt"), text)
                entry["status"] = "FETCHED"
                break
        if entry["status"] != "FETCHED":
            entry["reason"] = "no located route returned the listed work's text"
        record[sid] = entry
        print(sid, entry["status"], (entry.get("fulltext") or {}).get("chars"), entry.get("title_words_found"), (entry.get("url") or "")[:80], flush=True)
    record["B12"] = {"title": lst4["B12"]["title"], "status": "UNFETCHABLE", "located_for_scope": True, "attempts": [],
                     "located_by": "none open: the article is behind Nature's paywall (the protocol's table)", "reason": "no open route"}

    scope = {"_rule": "PROTOCOL_sand_pass4_correction_2026_09_15.md 'Scope: text that is not the article'; judged from the pass-4 readers' own notes and the red team's evidence before any correction reading",
             "B11": {"scope": "alphaXiv's page for arXiv 2603.19022: the abstract, a generated 'AI Overview' and citation cards", "reason": "not the authors' text; its reader said so (readings_run_B1.json)"},
             "B12": {"scope": "Nature's paywall page for the article", "reason": "headline, standfirst and one sentence, then access options and prices; its reader said so (readings_run_B1.json)"}}
    for name, obj in (("list_correction.json", lst), ("fetch_record_correction.json", record), ("fetched_text_scope_correction.json", scope)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
    print("recoded sources:", {sid: record[sid]["status"] for sid in [x["id"] for x in lst]})
    print("located:", {x["id"]: record[x["id"]]["status"] for x in located}, "B12", record["B12"]["status"])


if __name__ == "__main__":
    main()
