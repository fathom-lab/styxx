# -*- coding: utf-8 -*-
"""Zenodo upload script for styxx v7.7.9 (= v25 on the concept chain).

Creates a NEW VERSION of the existing concept deposit (DOI 10.5281/zenodo.19326174,
parent record 20418532 / v7.7.7 / v24), uploads the bundle zip + key files,
sets the metadata for v7.7.9, and publishes.

Authentication: ZENODO_TOKEN env var (extracted from secrets/arxiv-creds.txt
by the caller; never echoed to stdout).

Run from styxx repo root:
    python zenodo/_upload_v7.7.9.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict

import requests

ZENODO_API = "https://zenodo.org/api"
CONCEPT_RECORD_ID = "19326174"
PARENT_RECORD_ID = "20418532"  # v7.7.7 / v24 — the last published version
BUNDLE_DIR = pathlib.Path(__file__).resolve().parent / "v7.7.9"
BUNDLE_ZIP = pathlib.Path(__file__).resolve().parent / "styxx-v7.7.9-zenodo-bundle.zip"

TOKEN = os.environ.get("ZENODO_TOKEN")
if not TOKEN:
    print("ERROR: ZENODO_TOKEN env var not set.")
    sys.exit(1)

S = requests.Session()
S.headers.update({"Authorization": f"Bearer {TOKEN}"})


DESCRIPTION_HTML = """<p><strong>Headline: the gauntlet catches itself. Two consecutive bar-weakness discoveries in a single session — D3 length-control caught by accident (Baseline-007 unexpected PASS), D4 capitalization-control caught by deliberate systematic confound audit — with regression-tested fixes, three pre-registered detection submissions tested under the strengthened v3 bars, and a 3584-word preprint documenting the recursion-of-discipline pattern.</strong></p>

<p><strong>What's new in v25 (styxx 7.7.9) vs v24 (styxx 7.7.7):</strong></p>
<ul>
<li><strong>D3 length-control bar</strong> (7.7.8) &mdash; detector AUC must beat the length-only oracle by &ge; 0.10 on both partitions. Shipped six minutes after Baseline-007's accidental v1 PASS exposed the length artifact in <code>expected_consensus</code> (truth ~3.9w, folklore ~7.5w).</li>
<li><strong>D4 capitalization-control bar</strong> (7.7.9) &mdash; detector AUC must beat the cap-ratio oracle's <em>absolute</em> AUC by &ge; 0.10 on both partitions. Discovered by <code>audit_confounds()</code>: cap-ratio is INVERTED on this corpus (truth has higher proper-noun density because canonical answers like "Paris", "Newton" are mostly proper nouns).</li>
<li><strong><code>audit_confounds()</code> primitive</strong> &mdash; scans 8 surface features as oracle-detectors, reports direction-agnostic AUC + Spearman &rho; to length. The structural counterpart to D3, available via <code>styxx gauntlet-audit-confounds</code>.</li>
<li><strong>Three pre-registered detection submissions under v3 bars:</strong>
<ul>
<li>Baseline-008 (embedding similarity): 3/4 &mdash; pre-stated 60% modal "pass D1+D2 fail D3" validated.</li>
<li>Baseline-009 (length-residualized embedding): 1/4 &mdash; pre-stated 30% "D2 only" modal validated; n=1 demonstration that residualization removes signal rather than escaping the artifact.</li>
<li>Baseline-010 (NLI cross-encoder entailment): 0/4 &mdash; pre-stated 20% "total fail" band validated; direction-of-effect FALSIFIED (eighth in-session falsification).</li>
</ul></li>
<li><strong>Recursive-discipline preprint</strong> (<code>PAPER_recursive_discipline_2026_05_27.md</code>, 3584 words) &mdash; arXiv-submittable preprint arguing the contribution is the recursion (bars revising under empirical pressure), not the bars or the methods.</li>
<li><strong>Eight in-session falsifications</strong> documented in four paper-grade FINDINGs at <code>papers/agent-self-audit/</code>.</li>
</ul>

<p><strong>Calibration record (across 5 pre-stated artifacts):</strong></p>
<ul>
<li>Outcome-band predictions: well-calibrated (every pre-stated submission landed in its predicted modal region)</li>
<li>AUC-range predictions: roughly two-thirds reliable</li>
<li>Direction-of-effect predictions: SYSTEMATICALLY worst &mdash; two of the eight in-session falsifications were direction misses on this same domain (cap-ratio confound, NLI entailment). The discipline lesson is now durable: always include <code>|AUC &minus; 0.5|</code> as the prediction range; treat direction as a separate sub-prediction.</li>
</ul>

<p><strong>The detection frontier under v3 bars (what's empirically been ruled out):</strong></p>
<ul>
<li>Surface-form lexical (token overlap) &mdash; fails D3 (tracks length too closely)</li>
<li>Classical NLP (TF-IDF) &mdash; fails K1/K3 on classification (folklore lacks lexical signature)</li>
<li>Raw semantic embedding similarity &mdash; fails D3 (structurally equivalent to length on D1)</li>
<li>Length-corrected embedding &mdash; fails D1+D3+D4 (residualization subtracts signal)</li>
<li>Pre-trained NLI cross-encoder &mdash; fails all four (MNLI training doesn't transfer to factual-restatement detection)</li>
</ul>

<p>Methods still untested under v3 bars: cross-vendor consensus disagreement, perplexity against a calibrated prior LM, knowledge-graph lookup, fine-tuned classifier on (q, r) pairs. All operator-territory at present.</p>

<p><strong>Methodological contribution:</strong></p>
<p>We argue that the moat of disciplined AI evaluation is not the bars or the benchmark but the recursive pattern: bars revise under empirical pressure from real submissions; the discipline of pre-registration BEFORE each run makes wrongness <em>visible</em> rather than hidden; the infrastructure improves itself on a session timescale. The asymmetry is intentional: submitters may run their methods privately, only publishing their best results; we publish every gauntlet run we make against ourselves, every prediction we lock, every falsification that follows. Over time, this compounds &mdash; the floor accumulates rigor faster than any single submission can erode.</p>

<p><strong>Reproduce:</strong></p>
<pre>pip install styxx==7.7.9
styxx gauntlet --method styxx.gauntlet:_majority_baseline_predict --task classification
styxx gauntlet-audit-confounds
styxx leaderboard --rows-only</pre>

<p><strong>Source:</strong> <a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a> &middot; <strong>Tag:</strong> <a href="https://github.com/fathom-lab/styxx/releases/tag/v7.7.9">v7.7.9</a> &middot; <strong>PyPI:</strong> <a href="https://pypi.org/project/styxx/7.7.9/">pypi.org/project/styxx/7.7.9</a> &middot; <strong>Recursive-discipline preprint:</strong> <a href="https://github.com/fathom-lab/styxx/blob/main/papers/PAPER_recursive_discipline_2026_05_27.md">PAPER_recursive_discipline_2026_05_27.md</a></p>

<p><strong>Predecessor:</strong> v24 (<a href="https://doi.org/10.5281/zenodo.20418532">10.5281/zenodo.20418532</a>) &mdash; Fathom v24 / styxx v7.7.7, the seven-method empirical floor with the closed-loop self-audit demonstration. v25 advances by shipping the bars-catching-themselves recursion (D3 by accident, D4 by deliberate scan), three pre-registered baselines under v3 bars (all modal-validated, none cleared), and the recursive-discipline preprint.</p>

<p><strong>License:</strong> CC-BY-4.0 (this deposit, paper, data, findings, leaderboard). MIT (styxx Python code).</p>

<p><strong>Notes:</strong> Drafted with assistance from Claude Opus 4.7 (1M context) during a continuous 2026-05-27 session. The recursive-discipline preprint &sect;5 specifically describes how this paper itself caught a prediction error in its own draft text (the cap-ratio direction was originally described as "predicted in the positive direction" before correction). The methodological recursion the paper proposes &mdash; pre-stated gauntlets that catch their own bar weaknesses in production &mdash; is the contribution we expect the field to either replicate or refute within a year.</p>
"""

METADATA: Dict[str, Any] = {
    "metadata": {
        "title": "Fathom v25 / styxx v7.7.9: The Gauntlet that Catches Itself — pre-registered AI evaluation infrastructure with in-production bar-weakness detection",
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "publication_date": "2026-05-27",
        "description": DESCRIPTION_HTML,
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"},
        ],
        "keywords": [
            "pre-registered evaluation", "gauntlet infrastructure", "bar-weakness detection",
            "decorrelation ceiling", "consensus hallucination", "reference-free detection",
            "cross-vendor council", "AI integrity", "cognometric instruments",
            "dark core", "justification divergence", "perturbation fragility",
            "agreement fracture", "injected competitor test",
            "length confound", "capitalization confound", "confound audit",
            "regression-tested bars", "falsifiability", "in-session falsification",
            "pre-stated prediction", "direction-of-effect calibration",
            "Pareto frontier", "cultural priors", "folklore",
            "AI alignment", "LLM safety", "styxx", "Fathom",
        ],
        "version": "v25",
        "language": "eng",
        "license": "cc-by-4.0",
        "access_right": "open",
        "related_identifiers": [
            {"identifier": "10.5281/zenodo.20418532", "relation": "isNewVersionOf",
             "resource_type": "publication-workingpaper", "scheme": "doi"},
            {"identifier": "https://github.com/fathom-lab/styxx",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "https://github.com/fathom-lab/styxx/releases/tag/v7.7.9",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "https://pypi.org/project/styxx/7.7.9/",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
        ],
        "notes": (
            "Drafted with assistance from Claude Opus 4.7 (1M context) during a continuous "
            "2026-05-27 session. The recursive-discipline preprint §5 describes how the "
            "paper itself caught a prediction error in its own draft text. The methodological "
            "recursion proposed — pre-stated gauntlets that catch their own bar weaknesses "
            "in production — is the contribution we expect the field to either replicate or "
            "refute within a year."
        ),
    }
}


def die(msg: str, resp: requests.Response = None) -> None:
    print(f"ERROR: {msg}")
    if resp is not None:
        print(f"  status: {resp.status_code}")
        try:
            print(f"  body: {json.dumps(resp.json(), indent=2)[:500]}")
        except Exception:
            print(f"  body: {resp.text[:500]}")
    sys.exit(2)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def main() -> int:
    step("1. Verify parent record exists")
    r = S.get(f"{ZENODO_API}/records/{PARENT_RECORD_ID}")
    if r.status_code != 200:
        die(f"could not fetch parent record {PARENT_RECORD_ID}", r)
    parent = r.json()
    print(f"  parent: {parent['metadata']['title'][:80]}")
    print(f"  parent doi: {parent.get('doi')}")
    print(f"  concept doi: {parent.get('conceptdoi')}")

    step("2. Create new version")
    r = S.post(f"{ZENODO_API}/deposit/depositions/{PARENT_RECORD_ID}/actions/newversion")
    if r.status_code not in (201, 200):
        die("newversion failed", r)
    new_dep = r.json()
    latest_draft_url = new_dep.get("links", {}).get("latest_draft")
    if not latest_draft_url:
        die("no latest_draft link in newversion response", r)
    print(f"  latest_draft: {latest_draft_url}")
    r = S.get(latest_draft_url)
    if r.status_code != 200:
        die("could not fetch latest_draft deposit", r)
    draft = r.json()
    draft_id = draft["id"]
    print(f"  new draft id: {draft_id}")

    step("3. Clear old files from draft")
    files_url = draft["links"]["files"]
    bucket_url = draft["links"].get("bucket")
    if bucket_url:
        print(f"  bucket: {bucket_url}")
    for f in draft.get("files", []):
        f_id = f.get("id")
        fname = f.get("filename") or f.get("key") or f_id
        del_url = f"{ZENODO_API}/deposit/depositions/{draft_id}/files/{f_id}"
        dr = S.delete(del_url)
        if dr.status_code in (204, 200):
            print(f"  removed: {fname}")
        else:
            print(f"  WARNING: could not remove {fname}: {dr.status_code}")

    step("4. Upload bundle zip + key files")
    if not BUNDLE_ZIP.exists():
        die(f"bundle zip not found: {BUNDLE_ZIP}")
    if not BUNDLE_DIR.exists():
        die(f"bundle dir not found: {BUNDLE_DIR}")

    def upload(local: pathlib.Path, key: str) -> None:
        with open(local, "rb") as fh:
            if bucket_url:
                r = S.put(f"{bucket_url}/{key}", data=fh)
            else:
                r = S.post(files_url, data={"name": key}, files={"file": fh})
        if r.status_code not in (200, 201):
            die(f"upload {key} failed", r)
        print(f"  uploaded: {key} ({local.stat().st_size:,} bytes)")

    upload(BUNDLE_ZIP, BUNDLE_ZIP.name)
    for f in [
        "styxx-7.7.9-py3-none-any.whl",
        "styxx-7.7.9.tar.gz",
        "PAPER_recursive_discipline_2026_05_27.md",
        "PAPER_decorrelation_ceiling_2026_05_27.md",
        "darkcore_benchmark_2026_05_27.json",
        "LEADERBOARD.md",
        "README.md",
        "CHANGELOG.md",
    ]:
        local = BUNDLE_DIR / f
        if local.exists():
            upload(local, f)
        else:
            print(f"  SKIP (missing): {f}")

    step("5. Update metadata")
    r = S.put(f"{ZENODO_API}/deposit/depositions/{draft_id}",
              json=METADATA,
              headers={"Content-Type": "application/json"})
    if r.status_code not in (200, 201):
        die("metadata update failed", r)
    print(f"  metadata set; title: {r.json()['metadata']['title'][:80]}")

    step("6. Publish")
    r = S.post(f"{ZENODO_API}/deposit/depositions/{draft_id}/actions/publish")
    if r.status_code not in (200, 202):
        die("publish failed", r)
    published = r.json()
    new_doi = published.get("doi") or published.get("metadata", {}).get("doi")
    new_record_url = published.get("links", {}).get("record_html") or published.get("links", {}).get("record")
    print(f"\n  PUBLISHED")
    print(f"  DOI: {new_doi}")
    print(f"  URL: {new_record_url}")
    print(f"  concept DOI: 10.5281/zenodo.19326174 (latest now points here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
