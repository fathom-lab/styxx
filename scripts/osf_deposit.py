"""Create a standalone OSF project for Cognometry v0 and upload the
paper. Does NOT touch the existing wtkzg parent project — creates an
independent public project + uploads the PDF + sets description.

OSF API docs: https://developer.osf.io/
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CREDS = Path(r"C:\Users\heyzo\clawd\secrets\arxiv-creds.txt")
PAPER = ROOT / "papers" / "cognometry-v0.pdf"


def token():
    txt = CREDS.read_text(encoding="utf-8")
    m = re.search(r"\[OSF\](.*?)(?:\n\[|\Z)", txt, re.DOTALL)
    section = m.group(1) if m else ""
    tm = re.search(r"api_token:\s*(\S+)", section)
    return tm.group(1) if tm else None


TOKEN = token()
if not TOKEN:
    sys.exit("OSF token not found in arxiv-creds.txt")

HDRS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/vnd.api+json",
}

DESCRIPTION = """\
We define cognometry as the empirical quantification of cognitive states in \
machine systems — refusal, confabulation, retrieval, reasoning, and adversarial \
drift — from signals already carried on the token stream and residual activations \
of a language model during inference. We publish three falsifiable laws of \
cognometry with cross-validated numerical support for each, and ship the first \
open-source instrument (styxx on PyPI) that realizes the measurement. \

The central empirical claim of this paper is narrower: a 9-signal logistic \
regression fused over text, entity, novelty, grounding, and NLI contradiction \
signals achieves cross-validated hallucination discrimination across 8 public \
benchmarks — HaluEval-QA/Dialog/Summ, TruthfulQA, and four HaluBench subsets \
(DROP, PubMedQA, FinanceBench, RAGTruth) — with honest per-dataset performance \
ranging from near-perfect (AUC 0.998 on HaluEval-QA) to below chance (AUC 0.424 \
on DROP). Failure modes are declared openly in the weights module itself.

Software: pip install styxx==4.0.2[nli]. \
Zenodo DOI: https://doi.org/10.5281/zenodo.19703527. \
Code: https://github.com/fathom-lab/styxx (MIT on code, CC-BY-4.0 on weights)."""


def main():
    s = requests.Session()
    s.headers.update(HDRS)

    # Create node. OSF v2 API: POST /v2/nodes/
    payload = {
        "data": {
            "type": "nodes",
            "attributes": {
                "title": "Cognometry v0 — 8-Benchmark Cross-Validated Hallucination Detection in Production LLMs",
                "description": DESCRIPTION,
                "category": "project",
                "public": True,
                "tags": [
                    "hallucination-detection",
                    "cognometry",
                    "large-language-models",
                    "NLI",
                    "HaluEval",
                    "TruthfulQA",
                    "HaluBench",
                    "benchmark",
                    "cross-validation",
                    "open-science",
                ],
            },
        }
    }
    r = s.post("https://api.osf.io/v2/nodes/", json=payload, timeout=30)
    if r.status_code not in (201, 202):
        print(f"node creation FAILED: {r.status_code}")
        print(r.text[:800])
        sys.exit(1)
    node = r.json()["data"]
    node_id = node["id"]
    node_url = node["links"]["html"]
    print(f"created: {node_url}")
    print(f"  id: {node_id}")

    # Link to parent wtkzg as a related work (node_links)
    try:
        link_payload = {
            "data": {"type": "node_links",
                     "relationships": {
                         "nodes": {"data": {"type": "nodes", "id": "wtkzg"}}
                     }}
        }
        r2 = s.post(
            f"https://api.osf.io/v2/nodes/{node_id}/node_links/",
            json=link_payload, timeout=30,
        )
        print(f"  linked to wtkzg: HTTP {r2.status_code}")
    except Exception as e:
        print(f"  link skipped: {e}")

    # Upload paper PDF to OSF Storage on the new node
    # Waterbutler endpoint: PUT
    # https://files.us.osf.io/v1/resources/{node_id}/providers/osfstorage/
    if PAPER.exists():
        upload_url = (
            f"https://files.us.osf.io/v1/resources/{node_id}/"
            f"providers/osfstorage/?kind=file&name={PAPER.name}"
        )
        with open(PAPER, "rb") as f:
            r3 = requests.put(
                upload_url,
                headers={"Authorization": f"Bearer {TOKEN}"},
                data=f,
                timeout=60,
            )
        print(f"  paper upload: HTTP {r3.status_code}")
        if r3.status_code < 300:
            d = r3.json()
            print(f"    file id: {d.get('data', {}).get('id', '?')}")
        else:
            print(f"    {r3.text[:300]}")
    else:
        print(f"  no PDF at {PAPER}")

    # Upload the markdown paper too (some readers prefer)
    md = ROOT / "papers" / "cognometry-v0.md"
    if md.exists():
        upload_url = (
            f"https://files.us.osf.io/v1/resources/{node_id}/"
            f"providers/osfstorage/?kind=file&name={md.name}"
        )
        with open(md, "rb") as f:
            r4 = requests.put(
                upload_url,
                headers={"Authorization": f"Bearer {TOKEN}"},
                data=f,
                timeout=60,
            )
        print(f"  markdown upload: HTTP {r4.status_code}")

    print(f"\nOSF project LIVE: {node_url}")


if __name__ == "__main__":
    main()
