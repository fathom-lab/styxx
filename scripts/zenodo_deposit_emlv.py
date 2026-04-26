"""Deposit every-mind-leaves-vitals.md to Zenodo via API.

Usage:
    ZENODO_TOKEN=... python scripts/zenodo_deposit_emlv.py

Steps:
  1. POST /api/deposit/depositions   → create empty deposition
  2. PUT  <bucket>/<filename>        → upload files
  3. PUT  /api/deposit/depositions/{id}  → metadata
  4. POST /api/deposit/depositions/{id}/actions/publish → get DOI
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

ZENODO_API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN")
if not TOKEN:
    print("ERROR: set ZENODO_TOKEN env var", file=sys.stderr)
    sys.exit(2)

AUTH_PARAMS = {"access_token": TOKEN}

FILES_TO_UPLOAD = [
    ROOT / "papers" / "every-mind-leaves-vitals.md",
    ROOT / "release" / "phase-transition-chart.png",
]

DESCRIPTION = """\
<p><strong>Every mind leaves vitals.</strong></p>

<p>This paper extends the Cognometry Manifesto (Rodabaugh, 2026,
<a href="https://doi.org/10.5281/zenodo.19703527">10.5281/zenodo.19703527</a>)
beyond LLMs. We show that three independent calibrated cognometric
instruments &mdash; hallucination (AUC 0.998), refusal (AUC 0.976),
tool-call drift (AUC 0.916) &mdash; all exhibit phase-transition
structure under feature-count ablation: detection of cognitive states
does not scale smoothly with classifier capacity but jumps discretely
at a critical feature, replicated across three independent feature
bases.</p>

<p>We argue this result is a property of cognition rather than of
language models, and bridge to thirty years of independent evidence
from forensic linguistics, computational psychiatry, and crisis-text
classification that the same observability layer applies to biological
cognition. The substrates differ; the observability does not.</p>

<p>The paper records six constitutional commitments for cognometric
instruments shipped under the Fathom name &mdash; MIT license, weights
and reproducers in-tree, failure modes declared in-weights, calibration
fingerprint required, CPU and browser-runnable, no private detectors
under the Fathom name. The constitutional commitments are written so
that if they are ever broken, the paper makes the breaking visible.</p>

<p>The paper is falsifiable on its sharpest claim: any calibrated
text-based cognitive-state detector whose feature-count ablation shows
smooth AUC scaling without a critical-K jump would falsify the central
result, and we will retract or amend at the same DOI.</p>

<p><strong>Reproduce.</strong>
<a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a> &mdash;
every number reruns from <code>random_state=0</code> in under five
minutes on CPU.<br>
<strong>Run live.</strong>
<a href="https://fathom.darkflobi.com/cognometry">fathom.darkflobi.com/cognometry</a> &mdash;
three instruments in your browser, no install.<br>
<strong>Cite the manifesto.</strong>
<a href="https://doi.org/10.5281/zenodo.19703527">10.5281/zenodo.19703527</a></p>"""


METADATA = {
    "metadata": {
        "title": (
            "Every Mind Leaves Vitals: On the Cognometric Layer, "
            "Substrate-Independence, and the One-Time Choice We Have"
        ),
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "description": DESCRIPTION,
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"},
        ],
        "keywords": [
            "cognometry",
            "cognitive observability",
            "calibration fingerprint",
            "phase transitions",
            "hallucination detection",
            "refusal detection",
            "tool-call drift",
            "AI safety",
            "interpretability",
            "measurement standard",
            "forensic linguistics",
            "computational psychiatry",
            "open science",
            "position paper",
            "styxx",
        ],
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-4.0",
        "notes": (
            "Position paper extending the Cognometry Manifesto. "
            "Software companion: pip install styxx. "
            "All reproducers at github.com/fathom-lab/styxx. "
            "Constitutional commitments at fathom.darkflobi.com/cognometric-disclosure."
        ),
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.19703527",
                "relation": "continues",
                "resource_type": "publication-workingpaper",
                "scheme": "doi",
            },
            {
                "identifier": "https://github.com/fathom-lab/styxx",
                "relation": "isSupplementedBy",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://pypi.org/project/styxx/",
                "relation": "isSupplementedBy",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://fathom.darkflobi.com/cognometry",
                "relation": "isDocumentedBy",
                "resource_type": "publication",
                "scheme": "url",
            },
        ],
    },
}


def step(msg):
    print(f"  > {msg}")


def main():
    # 1. Create deposition
    print("1/4 creating deposition...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions",
        params=AUTH_PARAMS,
        json={},
    )
    if r.status_code >= 400:
        print(f"FAIL: {r.status_code} {r.text}")
        sys.exit(1)
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    step(f"deposition id: {dep_id}")
    step(f"bucket: {bucket[:70]}...")

    # 2. Upload files
    print("\n2/4 uploading files...")
    for p in FILES_TO_UPLOAD:
        if not p.exists():
            print(f"  skip (not found): {p}")
            continue
        with open(p, "rb") as f:
            r = requests.put(
                f"{bucket}/{p.name}",
                data=f,
                params=AUTH_PARAMS,
            )
        if r.status_code >= 400:
            print(f"FAIL upload {p.name}: {r.status_code} {r.text}")
            sys.exit(1)
        step(f"uploaded {p.name} ({p.stat().st_size} bytes)")

    # 3. Attach metadata
    print("\n3/4 setting metadata...")
    r = requests.put(
        f"{ZENODO_API}/deposit/depositions/{dep_id}",
        params=AUTH_PARAMS,
        json=METADATA,
    )
    if r.status_code >= 400:
        print(f"FAIL metadata: {r.status_code}")
        print(r.text[:2000])
        sys.exit(1)
    step("metadata attached")

    # 4. Publish
    print("\n4/4 publishing...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions/{dep_id}/actions/publish",
        params=AUTH_PARAMS,
    )
    if r.status_code >= 400:
        print(f"FAIL publish: {r.status_code}")
        print(r.text[:2000])
        print("\nThe deposition exists at "
              f"https://zenodo.org/deposit/{dep_id} — complete manually.")
        sys.exit(1)

    published = r.json()
    doi = published.get("doi") or published["metadata"].get("doi")
    doi_url = f"https://doi.org/{doi}" if doi else None
    record_url = published.get("links", {}).get("record_html")

    print("\n" + "=" * 60)
    print(f"  DOI: {doi}")
    print(f"  URL: {doi_url}")
    print(f"  Record page: {record_url}")
    print("=" * 60)

    # Emit machine-readable output
    out_path = ROOT / "release" / "zenodo-deposit-receipt-emlv.json"
    out_path.write_text(
        json.dumps({
            "deposition_id": dep_id,
            "doi": doi,
            "doi_url": doi_url,
            "record_url": record_url,
            "title": METADATA["metadata"]["title"],
            "files": [p.name for p in FILES_TO_UPLOAD if p.exists()],
            "timestamp": published.get("created"),
        }, indent=2)
    )
    step(f"receipt written: {out_path}")


if __name__ == "__main__":
    main()
