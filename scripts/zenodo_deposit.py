"""Deposit cognometry-v0.md + PDF to Zenodo via API.

Usage:
    ZENODO_TOKEN=... python scripts/zenodo_deposit.py

Steps:
  1. POST /api/deposit/depositions   → create empty deposition
  2. PUT  <bucket>/<filename>        → upload files
  3. PUT  /api/deposit/depositions/{id}  → metadata
  4. POST /api/deposit/depositions/{id}/actions/publish → get DOI
"""
from __future__ import annotations

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
    ROOT / "papers" / "cognometry-v0.pdf",
    ROOT / "papers" / "cognometry-v0.md",
    ROOT / "papers" / "cognometry-research-agenda-2026.md",
]

DESCRIPTION = """\
<p>We define <strong>cognometry</strong> as the empirical quantification
of cognitive states in machine systems&mdash;refusal, confabulation,
retrieval, reasoning, and adversarial drift&mdash;from signals already
carried on the token stream and residual activations of a language
model during inference. We publish three falsifiable laws of
cognometry (vitals exist, vitals transfer across substrates, vitals
are causally actionable) with cross-validated numerical support for
each, and ship the first open-source instrument
(<a href="https://pypi.org/project/styxx/">styxx on PyPI</a>) that
realizes the measurement.</p>

<p>The central empirical claim of this paper is narrower: a 9-signal
logistic regression fused over text, entity, novelty, grounding,
and NLI contradiction signals achieves cross-validated hallucination
discrimination across <strong>8 public benchmarks</strong>&mdash;
HaluEval-QA, Dialog, Summarization, TruthfulQA, and four HaluBench
subsets (DROP, PubMedQA, FinanceBench, RAGTruth)&mdash;with honest
per-dataset performance ranging from near-perfect (AUC 0.998 on
HaluEval-QA) to below chance (AUC 0.424 on DROP).</p>

<p>We openly report and taxonomize the failure modes: reading-
comprehension extractive-span errors and financial arithmetic errors
are not detected by the present signal stack because both classes
of error pass the entailment (NLI) and novelty bars by construction.
Failure modes are declared in the weights module itself.</p>

<p>This is the first 8-benchmark cross-validated hallucination detector
in the open literature. Above-chance performance on 5/8 benchmarks
with 3/8 near-perfect is the reproducible empirical floor we lay
down. Two below-chance results are the reproducible research agenda
we lay down.</p>

<p><strong>Manifesto:</strong>
<a href="https://fathom.darkflobi.com/cognometry">https://fathom.darkflobi.com/cognometry</a><br>
<strong>Software:</strong>
<a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a>
| <code>pip install styxx==4.0.1[nli]</code><br>
<strong>Leaderboard:</strong>
<a href="https://fathom.darkflobi.com/cognometry/leaderboard">fathom.darkflobi.com/cognometry/leaderboard</a></p>"""


METADATA = {
    "metadata": {
        "title": (
            "Cognometry v0: 8-Benchmark Cross-Validated Hallucination "
            "Detection in Production LLMs"
        ),
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "description": DESCRIPTION,
        "creators": [
            {"name": "Flobi", "affiliation": "Fathom Lab"},
        ],
        "keywords": [
            "hallucination detection",
            "large language models",
            "cognometry",
            "interpretability",
            "cognitive states",
            "NLI",
            "entailment",
            "benchmarking",
            "cross-validation",
            "reproducibility",
            "AI safety",
            "open science",
            "residual stream",
            "activation probing",
            "HaluEval",
            "TruthfulQA",
            "HaluBench",
            "styxx",
        ],
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-4.0",
        "notes": (
            "Software companion: pip install styxx==4.0.1[nli]. "
            "All reproducers in the GitHub repository. "
            "Manifesto at fathom.darkflobi.com/cognometry."
        ),
        "related_identifiers": [
            {
                "identifier": "https://github.com/fathom-lab/styxx",
                "relation": "isSupplementedBy",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://pypi.org/project/styxx/4.0.1/",
                "relation": "isSupplementedBy",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://huggingface.co/datasets/PatronusAI/HaluBench",
                "relation": "references",
                "resource_type": "dataset",
                "scheme": "url",
            },
            {
                "identifier": "https://huggingface.co/datasets/pminervini/HaluEval",
                "relation": "references",
                "resource_type": "dataset",
                "scheme": "url",
            },
            {
                "identifier": "https://huggingface.co/datasets/truthful_qa",
                "relation": "references",
                "resource_type": "dataset",
                "scheme": "url",
            },
            {
                "identifier": "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                "relation": "references",
                "resource_type": "software",
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
    import json
    out_path = ROOT / "release" / "zenodo-deposit-receipt.json"
    out_path.write_text(
        json.dumps({
            "doi": doi,
            "doi_url": doi_url,
            "record_url": record_url,
            "deposition_id": dep_id,
            "uploaded_files": [p.name for p in FILES_TO_UPLOAD
                                if p.exists()],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nReceipt saved: {out_path}")


if __name__ == "__main__":
    main()
