# -*- coding: utf-8 -*-
"""Zenodo upload script for styxx v7.7.7.

Creates a NEW VERSION of the existing concept deposit (DOI 10.5281/zenodo.19326174,
parent record 20130041 / v7.2.0), uploads the bundle zip + key files, sets the
metadata for v7.7.7, and publishes.

Authentication: ZENODO_TOKEN env var (extracted from secrets/arxiv-creds.txt
by the caller; never echoed to stdout).

Run from styxx repo root:
    python zenodo/_upload_v7.7.7.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict

import requests

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
ZENODO_API = "https://zenodo.org/api"
CONCEPT_RECORD_ID = "19326174"                  # the concept DOI's record id
PARENT_RECORD_ID = "20130041"                   # v7.2.0 / v23 — the last published version
BUNDLE_DIR = pathlib.Path(__file__).resolve().parent / "v7.7.7"
BUNDLE_ZIP = pathlib.Path(__file__).resolve().parent / "styxx-v7.7.7-zenodo-bundle.zip"

TOKEN = os.environ.get("ZENODO_TOKEN")
if not TOKEN:
    print("ERROR: ZENODO_TOKEN env var not set. Extract from secrets/arxiv-creds.txt first.")
    sys.exit(1)

S = requests.Session()
S.headers.update({"Authorization": f"Bearer {TOKEN}"})

# ──────────────────────────────────────────────────────────────────────
# Metadata for the new version
# ──────────────────────────────────────────────────────────────────────
DESCRIPTION_HTML = """<p><strong>Headline: seven independent pre-registered methods on the consensus-hallucination dark core, all closed-negative within their corpus's scope. The synthesis stating the bimodal prediction was committed two days before half of them ran. The data resolved on the load-bearing-floor branch the synthesis named.</strong></p>

<p><strong>Methods (all bars locked + pushed to public origin before each probe fired; verifiable from git history):</strong></p>
<ul>
<li><strong>Detection #1</strong> — perturbation-fragility (Dark Matter swing #1): partial, fragile shell only.</li>
<li><strong>Detection #2</strong> — agreement-fracture (CVPD): clean negative, lift &minus;0.32.</li>
<li><strong>Detection #3</strong> — justification-divergence (JD): clean negative, <strong>INVERTED</strong> &mdash; stubborn dark-core items have the <em>most</em> convergent across-vendor justifications (mean JD 0.022 vs truth 0.067). Three vendors share the wrong fact AND the supporting story.</li>
<li><strong>Constructive #1</strong> &mdash; neutral injection (ICT, n_folk=4): immovability floor, 0/4 folklore yield.</li>
<li><strong>Constructive #2</strong> &mdash; neutral injection on hand-curated 30-folklore corpus: SHORTFALL n_folk=2, but a genuine finding fell out &mdash; 28/30 well-known cultural myths are already corrected in 2026 frontier-model council baseline; the practical dark core is narrower than the curation assumed.</li>
<li><strong>Constructive #3</strong> &mdash; authoritative injection on same corpus: SHORTFALL + descriptive: same 2 folk items lifted in both framings (no differential framing effect), +0.05 truth-yield to authoritative falsehood (auth-sycophancy direction signal, n=1).</li>
<li><strong>Classification #1</strong> &mdash; sentence-transformer + balanced LR: FAIL K2 + K3. The dark core is also dark to text-only classification with this stack at n &asymp; 80 training items.</li>
</ul>

<p><strong>Methodological contributions:</strong></p>
<ul>
<li><strong>The closed-loop self-audit demonstration.</strong> The producer's own product (<code>styxx audit</code>, <code>styxx critique</code>) caught the producer drifting from the producer's own derived discipline within the same session; the producer revised in the corrected register; the gate cleared. Composite 0.358 &rarr; 0.174 with the Pareto trade-off (sycophancy&harr;overconfidence on text-only register) observed live. We are unaware of prior published demonstrations of this recursion in the AI-integrity-instrument literature.</li>
<li><strong>Five in-session falsifications</strong> of the producer's own claims, all recorded in place with strikethrough rather than rewritten. Proposed as a methodological pattern for AI-research integrity.</li>
<li><strong>The paper itself self-audits.</strong> The abstract scores sycophancy 0.66 and &sect;5 scores 0.62 &mdash; the documented closed-negative restrained-FP firing on the paper's own status-reporting register. The paper acknowledges this in real time rather than gaming the gate.</li>
</ul>

<p><strong>Deployable artifacts:</strong></p>
<ul>
<li><strong>The dark-core benchmark dataset</strong> (108 labeled records across 4 classes; folklore 34 / pseudoscience 6 / factual-error 13 / truth 55) with the seven-method empirical floor baked in as the bar future routing approaches need to beat.</li>
<li><strong>The public-challenge leaderboard</strong> (<code>LEADERBOARD.md</code>) with the seven-method floor as Baseline-001 + three concrete reference baselines (002 classifier, 003 length heuristic, 004 random class). External submissions go through CI auto-verification.</li>
<li><strong>The styxx CLI family</strong>: <code>styxx audit</code> (per-turn auditing), <code>styxx critique</code> (audit + register-fix suggestions with mandatory scope-bound), <code>styxx gauntlet</code> (run a candidate method against the floor), <code>styxx leaderboard</code> (terminal view of the board), <code>styxx data-dir</code> (discover the active chart.jsonl path).</li>
</ul>

<p><strong>What this deposit does NOT claim:</strong></p>
<ul>
<li>A deployable positive routing primitive. Three deployable-positive paths tested closed-negative.</li>
<li>External replication or adoption. Out-of-session by construction.</li>
<li>The &ldquo;extraordinary&rdquo; framing the prompt sought. What this work bought is the <em>foundation</em> for credibility &mdash; receipts, discipline pattern, falsifications in place, deployable artifacts, public challenge &mdash; not the audience reception that compounds over time.</li>
</ul>

<p><strong>Reproduce:</strong></p>
<pre>pip install styxx==7.7.7
styxx leaderboard --rows-only      # see the floor
styxx gauntlet --method styxx.gauntlet:_majority_baseline_predict --task classification</pre>

<p><strong>Source:</strong> <a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a> &middot; <strong>Tag:</strong> <a href="https://github.com/fathom-lab/styxx/releases/tag/v7.7.7">v7.7.7</a> &middot; <strong>PyPI:</strong> <a href="https://pypi.org/project/styxx/7.7.7/">pypi.org/project/styxx/7.7.7</a> &middot; <strong>Paper:</strong> <a href="https://github.com/fathom-lab/styxx/blob/main/papers/PAPER_decorrelation_ceiling_2026_05_27.md">PAPER_decorrelation_ceiling_2026_05_27.md</a></p>

<p><strong>Predecessor:</strong> v23 (<a href="https://doi.org/10.5281/zenodo.20130041">10.5281/zenodo.20130041</a>) &mdash; Fathom v23 / styxx v7.2.0, F10 Self-Healing Reflex + Cognometric Inversion. v24 advances by shipping the seven-method empirical floor, the closed-loop self-audit demonstration, the public-challenge leaderboard infrastructure, and the styxx CLI family for audit/critique/gauntlet/leaderboard.</p>

<p><strong>License:</strong> CC-BY-4.0 (this deposit, paper, data, findings, leaderboard). MIT (styxx Python code, see <code>LICENSE</code> file in the bundle).</p>

<p><strong>Notes:</strong> Drafted with assistance from Claude Opus 4.7 (1M context) during a 2026-05-27 session. Claude-authored commits in the styxx repo carry the Claude noreply author signature. The closed-loop self-audit demonstration in &sect;5 of the paper means this paper is itself eligible for audit under the same <code>styxx audit</code> CLI it describes &mdash; a methodological recursion we expect the field to take less than a year to either replicate or refute.</p>
"""

METADATA: Dict[str, Any] = {
    "metadata": {
        "title": "Fathom v24 / styxx v7.7.7: The Decorrelation Ceiling — seven-method empirical floor on reference-free detection of cross-vendor consensus hallucination",
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "publication_date": "2026-05-27",
        "description": DESCRIPTION_HTML,
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"},
        ],
        "keywords": [
            "decorrelation ceiling", "consensus hallucination", "reference-free detection",
            "cross-vendor council", "AI integrity", "cognometric instruments",
            "dark core", "justification divergence", "perturbation fragility",
            "agreement fracture", "injected competitor test",
            "pre-registration", "falsifiability", "dogfood", "self-audit",
            "Pareto frontier", "cultural priors", "folklore",
            "AI alignment", "LLM safety", "styxx", "Fathom",
        ],
        "version": "v24",
        "language": "eng",
        "license": "cc-by-4.0",
        "access_right": "open",
        "related_identifiers": [
            {"identifier": "10.5281/zenodo.20130041", "relation": "isNewVersionOf",
             "resource_type": "publication-workingpaper", "scheme": "doi"},
            {"identifier": "https://github.com/fathom-lab/styxx",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "https://github.com/fathom-lab/styxx/releases/tag/v7.7.7",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "https://pypi.org/project/styxx/7.7.7/",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "https://styxx.org",
             "relation": "isDocumentedBy", "resource_type": "publication-other", "scheme": "url"},
        ],
        "notes": (
            "Drafted with assistance from Claude Opus 4.7 (1M context) during a 2026-05-27 "
            "session. Claude-authored commits in the styxx repo carry the Claude noreply "
            "author signature. The closed-loop self-audit demonstration in §5 of the paper "
            "means this paper is itself eligible for audit under the same `styxx audit` CLI "
            "it describes — a methodological recursion we expect the field to take less than "
            "a year to either replicate or refute."
        ),
    }
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# Main flow
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    step("1. Verify parent record exists")
    r = S.get(f"{ZENODO_API}/records/{PARENT_RECORD_ID}")
    if r.status_code != 200:
        die(f"could not fetch parent record {PARENT_RECORD_ID}", r)
    parent = r.json()
    print(f"  parent: {parent['metadata']['title'][:80]}")
    print(f"  parent doi: {parent.get('doi')}")
    print(f"  concept doi: {parent.get('conceptdoi')}")
    if parent.get("conceptdoi") and "19326174" not in parent.get("conceptdoi", ""):
        die(f"parent record {PARENT_RECORD_ID} concept DOI {parent.get('conceptdoi')} != expected 10.5281/zenodo.19326174")

    step("2. Create new version (POST /deposit/depositions/<parent>/actions/newversion)")
    # newversion is on the DEPOSIT endpoint (which requires the deposit id, same as record id)
    r = S.post(f"{ZENODO_API}/deposit/depositions/{PARENT_RECORD_ID}/actions/newversion")
    if r.status_code not in (201, 200):
        die("newversion failed", r)
    new_dep = r.json()
    # The response gives us the parent deposit; the actual new version's id is in `links.latest_draft`
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

    step(f"3. Clear old files from draft (the newversion copies parent files; we want fresh)")
    files_url = draft["links"]["files"]
    # Use the bucket API for newer deposits
    bucket_url = draft["links"].get("bucket")
    if bucket_url:
        print(f"  bucket: {bucket_url}")
    # Delete existing files in the new draft (carried over from parent)
    for f in draft.get("files", []):
        f_id = f.get("id")
        fname = f.get("filename") or f.get("key") or f_id
        del_url = f"{ZENODO_API}/deposit/depositions/{draft_id}/files/{f_id}"
        dr = S.delete(del_url)
        if dr.status_code in (204, 200):
            print(f"  removed: {fname}")
        else:
            print(f"  WARNING: could not remove {fname}: {dr.status_code}")

    step("4. Upload the bundle zip + individual key files")
    if not BUNDLE_ZIP.exists():
        die(f"bundle zip not found: {BUNDLE_ZIP}")
    if not BUNDLE_DIR.exists():
        die(f"bundle dir not found: {BUNDLE_DIR}")

    def upload(local: pathlib.Path, key: str) -> None:
        with open(local, "rb") as fh:
            if bucket_url:
                r = S.put(f"{bucket_url}/{key}", data=fh)
            else:
                # Fallback to old deposit-files API
                r = S.post(files_url, data={"name": key}, files={"file": fh})
        if r.status_code not in (200, 201):
            die(f"upload {key} failed", r)
        print(f"  uploaded: {key} ({local.stat().st_size:,} bytes)")

    upload(BUNDLE_ZIP, BUNDLE_ZIP.name)
    # Plus a curated subset of individual files for direct browsability:
    for f in [
        "styxx-7.7.7-py3-none-any.whl",
        "styxx-7.7.7.tar.gz",
        "PAPER_decorrelation_ceiling_2026_05_27.md",
        "REPORT_decorrelation_ceiling_v2_2026_05_27.md",
        "darkcore_benchmark_2026_05_27.json",
        "LEADERBOARD.md",
        "README.md",
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

    step("6. Publish (POST /deposit/depositions/<id>/actions/publish)")
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
