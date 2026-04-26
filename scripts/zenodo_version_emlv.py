"""Re-deposit every-mind-leaves-vitals as a NEW VERSION under the
manifesto concept (19703526), instead of as an orphan.

The orphan 10.5281/zenodo.19777361 is permanent — Zenodo never deletes
published records. This script (a) supersedes it under the proper
concept, and (b) optionally rewrites the orphan's metadata to mark
it as withdrawn-by-author with a pointer to the canonical version.

Usage:
    ZENODO_TOKEN=... python scripts/zenodo_version_emlv.py
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

# Latest published record under the manifesto concept (19703526).
# As of 2026-04-25, that is the manifesto itself: 10.5281/zenodo.19703527.
PARENT_DEP_ID = 19703527

# The orphan we accidentally created with the standalone deposit script.
ORPHAN_DEP_ID = 19777361

FILES_TO_UPLOAD = [
    ROOT / "papers" / "every-mind-leaves-vitals.md",
    ROOT / "release" / "phase-transition-chart.png",
]

DESCRIPTION = """\
<p><strong>Every mind leaves vitals.</strong></p>

<p>This paper extends the Cognometry Manifesto beyond LLMs. We show
that three independent calibrated cognometric instruments &mdash;
hallucination (AUC 0.998), refusal (AUC 0.976), tool-call drift
(AUC 0.916) &mdash; all exhibit phase-transition structure under
feature-count ablation: detection of cognitive states does not scale
smoothly with classifier capacity but jumps discretely at a critical
feature, replicated across three independent feature bases.</p>

<p>We argue this result is a property of cognition rather than of
language models, and bridge to thirty years of independent evidence
from forensic linguistics, computational psychiatry, and crisis-text
classification that the same observability layer applies to
biological cognition. The substrates differ; the observability does
not.</p>

<p>The paper records six constitutional commitments for cognometric
instruments shipped under the Fathom name &mdash; MIT license, weights
and reproducers in-tree, failure modes declared in-weights,
calibration fingerprint required, CPU and browser-runnable, no
private detectors under the Fathom name. The constitutional
commitments are written so that if they are ever broken, the paper
makes the breaking visible.</p>

<p>The paper is falsifiable on its sharpest claim: any calibrated
text-based cognitive-state detector whose feature-count ablation
shows smooth AUC scaling without a critical-K jump would falsify
the central result, and we will retract or amend at the same DOI.</p>

<p><strong>Reproduce.</strong>
<a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a> &mdash;
every number reruns from <code>random_state=0</code> in under five
minutes on CPU.<br>
<strong>Run live.</strong>
<a href="https://fathom.darkflobi.com/cognometry">fathom.darkflobi.com/cognometry</a><br>
<strong>Read on the site.</strong>
<a href="https://fathom.darkflobi.com/every-mind-leaves-vitals">fathom.darkflobi.com/every-mind-leaves-vitals</a></p>"""


METADATA = {
    "metadata": {
        "title": (
            "Every Mind Leaves Vitals: On the Cognometric Layer, "
            "Substrate-Independence, and the One-Time Choice We Have"
        ),
        "version": "position-paper-v1",
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
            "Supersedes orphan deposit 10.5281/zenodo.19777361. "
            "Software companion: pip install styxx. "
            "Reproducers at github.com/fathom-lab/styxx."
        ),
        "related_identifiers": [
            {
                "identifier": "https://github.com/fathom-lab/styxx",
                "relation": "isSupplementedBy",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://fathom.darkflobi.com/every-mind-leaves-vitals",
                "relation": "isDocumentedBy",
                "resource_type": "publication",
                "scheme": "url",
            },
            {
                "identifier": "10.5281/zenodo.19777361",
                "relation": "isAlternateIdentifier",
                "resource_type": "publication-workingpaper",
                "scheme": "doi",
            },
        ],
    },
}


def step(msg):
    print(f"  > {msg}")


def main():
    print(f"creating new version under concept of record {PARENT_DEP_ID}...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions/{PARENT_DEP_ID}/actions/newversion",
        params=AUTH_PARAMS,
    )
    if r.status_code >= 400:
        print(f"FAIL newversion: {r.status_code} {r.text[:500]}")
        sys.exit(1)

    parent = r.json()
    new_draft_url = parent["links"].get("latest_draft")
    if not new_draft_url:
        print(f"FAIL no latest_draft link in response: {json.dumps(parent, indent=2)[:1000]}")
        sys.exit(1)

    new_dep = requests.get(new_draft_url, params=AUTH_PARAMS).json()
    new_dep_id = new_dep["id"]
    bucket = new_dep["links"]["bucket"]
    step(f"new draft deposition id: {new_dep_id}")
    step(f"bucket: {bucket[:70]}...")

    # Delete inherited files (we want a fresh upload of the new paper)
    print("\nclearing inherited files from prior version...")
    for f in new_dep.get("files", []):
        del_url = f["links"]["self"]
        rd = requests.delete(del_url, params=AUTH_PARAMS)
        step(f"deleted {f.get('filename','?')}: {rd.status_code}")

    # Upload new files
    print("\nuploading files...")
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
            print(f"FAIL upload {p.name}: {r.status_code} {r.text[:300]}")
            sys.exit(1)
        step(f"uploaded {p.name} ({p.stat().st_size} bytes)")

    # Set metadata
    print("\nsetting metadata...")
    r = requests.put(
        f"{ZENODO_API}/deposit/depositions/{new_dep_id}",
        params=AUTH_PARAMS,
        json=METADATA,
    )
    if r.status_code >= 400:
        print(f"FAIL metadata: {r.status_code}")
        print(r.text[:2000])
        sys.exit(1)
    step("metadata attached")

    # Publish
    print("\npublishing...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions/{new_dep_id}/actions/publish",
        params=AUTH_PARAMS,
    )
    if r.status_code >= 400:
        print(f"FAIL publish: {r.status_code}")
        print(r.text[:2000])
        sys.exit(1)

    published = r.json()
    doi = published.get("doi") or published["metadata"].get("doi")
    doi_url = f"https://doi.org/{doi}" if doi else None
    record_url = published.get("links", {}).get("record_html")
    concept_doi = published.get("conceptdoi") or published.get("metadata", {}).get("conceptdoi")
    concept_recid = published.get("conceptrecid")

    print("\n" + "=" * 64)
    print(f"  NEW VERSION DOI:    {doi}")
    print(f"  URL:                {doi_url}")
    print(f"  Record page:        {record_url}")
    print(f"  Concept DOI:        {concept_doi}")
    print(f"  Concept recid:      {concept_recid}")
    print("=" * 64)

    out_path = ROOT / "release" / "zenodo-deposit-receipt-emlv-versioned.json"
    out_path.write_text(
        json.dumps({
            "action": "new_version_published",
            "concept_recid": concept_recid,
            "concept_doi": concept_doi,
            "new_deposition_id": new_dep_id,
            "new_doi": doi,
            "new_doi_url": doi_url,
            "record_url": record_url,
            "supersedes_orphan_doi": "10.5281/zenodo.19777361",
            "title": METADATA["metadata"]["title"],
            "files": [p.name for p in FILES_TO_UPLOAD if p.exists()],
            "timestamp": published.get("created"),
        }, indent=2)
    )
    step(f"receipt: {out_path}")

    # Note: the orphan (19777361) is permanent. We can edit its metadata to
    # add a note pointing to the canonical version, but the DOI itself stays.
    print()
    print("orphan 10.5281/zenodo.19777361 is permanent (Zenodo policy).")
    print("Updating its metadata to point readers at the canonical version...")
    try:
        # Edit the orphan: open for editing, update metadata, publish
        r = requests.post(
            f"{ZENODO_API}/deposit/depositions/{ORPHAN_DEP_ID}/actions/edit",
            params=AUTH_PARAMS,
        )
        step(f"orphan edit: {r.status_code}")
        if r.ok:
            # Add isObsoletedBy relation to the canonical version
            orphan_meta = {
                "metadata": {
                    "title": METADATA["metadata"]["title"] + " [SUPERSEDED — see canonical version]",
                    "upload_type": "publication",
                    "publication_type": "workingpaper",
                    "description": (
                        "<p><strong>This deposit is superseded by the canonical version under "
                        "the Cognometry Manifesto concept:</strong> "
                        f"<a href='{doi_url}'>{doi}</a></p>"
                        "<p>This record was created accidentally as an orphan deposit and is "
                        "preserved here as a permanent Zenodo record for citation integrity. "
                        "Please cite the canonical version instead.</p>"
                    ),
                    "creators": METADATA["metadata"]["creators"],
                    "keywords": METADATA["metadata"]["keywords"],
                    "language": "eng",
                    "access_right": "open",
                    "license": "cc-by-4.0",
                    "notes": f"SUPERSEDED. Canonical version: {doi_url}",
                    "related_identifiers": [
                        {
                            "identifier": doi,
                            "relation": "isObsoletedBy",
                            "resource_type": "publication-workingpaper",
                            "scheme": "doi",
                        },
                    ],
                }
            }
            r2 = requests.put(
                f"{ZENODO_API}/deposit/depositions/{ORPHAN_DEP_ID}",
                params=AUTH_PARAMS,
                json=orphan_meta,
            )
            step(f"orphan metadata update: {r2.status_code}")
            if r2.ok:
                r3 = requests.post(
                    f"{ZENODO_API}/deposit/depositions/{ORPHAN_DEP_ID}/actions/publish",
                    params=AUTH_PARAMS,
                )
                step(f"orphan re-publish: {r3.status_code}")
            else:
                print(f"  orphan meta error: {r2.text[:400]}")
    except Exception as e:
        print(f"  orphan rewrite error: {e}")
        print(f"  (canonical version was published successfully — manual cleanup of orphan optional)")


if __name__ == "__main__":
    main()
