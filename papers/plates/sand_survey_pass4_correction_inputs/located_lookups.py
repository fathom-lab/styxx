#!/usr/bin/env python3
"""located_lookups.py — the catalogue lookups behind the correction's located routes, recorded with their raw answers.

    python papers/plates/sand_survey_pass4_correction_inputs/located_lookups.py
    # writes located_lookups_correction.json beside this file

A review of the correction found that prepare_correction_inputs.py swallowed Europe PMC failures. A query that
failed read in the fetch record like a query that found nothing: B23's located_by says "Europe PMC None", although
Europe PMC indexes the article. This script re-runs only the catalogue lookups, never the fetches. For each query it
records the URL, the HTTP code, the parse error if any, and the fields that decide a route (hitCount, pmid, pmcid,
inPMC, inEPMC, hasPDF, doi, title).

The bytes the readers read are untouched. B04's PMID 42323341 is the one Semantic Scholar's search response gave in
externalIds (search/s2/Q09.json). B23's DOIs come from a Crossref title query, also recorded here.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sand_survey_pass4_inputs"))
import fetch_pass4 as fp  # noqa: E402


def get(url):
    r = subprocess.run(["curl", "-sL", "-A", fp.UA, "--max-time", "60", "-w", "\n%{http_code}", url], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    body, _, code = r.stdout.rpartition("\n")
    out = {"url": url, "http": code.strip(), "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        out["json"] = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        out["parse_error"] = repr(exc)
        out["body_head"] = body[:300]
    return out


def epmc(query):
    res = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({"query": query, "format": "json", "resultType": "core"}))
    hits = []
    if "json" in res:
        res["hitCount"] = res["json"].get("hitCount")
        for h in (res["json"].get("resultList") or {}).get("result") or []:
            hits.append({k: h.get(k) for k in ("pmid", "pmcid", "doi", "title", "inPMC", "inEPMC", "hasPDF", "isOpenAccess")})
        res.pop("json")
    res["query"], res["hits"] = query, hits
    return res


def crossref(title):
    res = get("https://api.crossref.org/works?" + urllib.parse.urlencode({"query.title": title, "rows": 5, "select": "DOI,title,issued,author,container-title"}))
    items = []
    if "json" in res:
        for it in ((res["json"].get("message") or {}).get("items") or []):
            items.append({"DOI": it.get("DOI"), "title": (it.get("title") or [""])[0], "issued": it.get("issued"), "container": (it.get("container-title") or [""])[0]})
        res.pop("json")
    res["query_title"], res["items"] = title, items
    return res


def main():
    out = {
        "_note": "catalogue lookups only; no bytes fetched; see this script's docstring",
        "B04": {"pmid_source": "search/s2/Q09.json externalIds.PubMed", "europe_pmc_by_pmid": epmc("EXT_ID:42323341 AND SRC:MED")},
        "B23": {"crossref_by_title": crossref("Offering scientists cash to spot errors in published papers doesn't work"),
                "europe_pmc_by_doi_print": epmc('DOI:"10.1126/science.aeh8588"'),
                "europe_pmc_by_doi_online": epmc('DOI:"10.1126/science.zh9l2q0"'),
                "europe_pmc_by_pmid": epmc("EXT_ID:41955357 AND SRC:MED")},
    }
    path = os.path.join(HERE, "located_lookups_correction.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    for sid, block in out.items():
        if sid.startswith("_"):
            continue
        for name, res in block.items():
            if isinstance(res, dict):
                print(sid, name, res.get("http"), res.get("parse_error"), res.get("hitCount"), (res.get("hits") or res.get("items") or [])[:2])


if __name__ == "__main__":
    main()
