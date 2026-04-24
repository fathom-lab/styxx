"""Deposit the styxx v6.0.0 research bundle to OSF as a new public
project linked to the Fathom wtkzg parent node.

What gets uploaded:
  - papers/cognometry-v0.5.pdf (current submittable paper)
  - papers/drift_phase_transitions.md (phase-transition addendum shipped today)
  - papers/figures/drift_phase_transitions.png (headline figure)
  - benchmarks/drift_feature_scaling.json (raw numbers)
  - benchmarks/drift_calibrated_v0.json (baseline)
  - scripts/drift_feature_scaling.py (reproducer)
  - scripts/drift_calibrated_v0.py (baseline reproducer)

OSF API v2: https://developer.osf.io/
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CREDS = Path(r"C:\Users\heyzo\clawd\secrets\arxiv-creds.txt")

UPLOAD_PATHS = [
    ROOT / "papers" / "cognometry-v0.5.pdf",
    ROOT / "papers" / "drift_phase_transitions.md",
    ROOT / "papers" / "figures" / "drift_phase_transitions.png",
    ROOT / "benchmarks" / "drift_feature_scaling.json",
    ROOT / "benchmarks" / "drift_calibrated_v0.json",
    ROOT / "scripts" / "drift_feature_scaling.py",
    ROOT / "scripts" / "drift_calibrated_v0.py",
]


def get_osf_token():
    txt = CREDS.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[OSF\](.*?)(?:\n\[|\Z)", txt, re.DOTALL)
    section = m.group(1) if m else ""
    tm = re.search(r"api_token:\s*(\S+)", section)
    return tm.group(1) if tm else None


TOKEN = get_osf_token()
if not TOKEN:
    sys.exit("OSF token not found in arxiv-creds.txt")

HDRS_JSON = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/vnd.api+json",
}

DESCRIPTION = """\
Styxx v6.0.0 reproducer bundle — three calibrated cognometric instruments and \
a phase-transition ablation shipped alongside.

Cognometry is the empirical measurement of cognitive states in machine systems \
(refusal, confabulation, tool-call drift, reasoning, retrieval) from signals \
already carried on the token stream and residual activations during inference. \

This OSF deposit contains the v6.0.0 release bundle:

- cognometry-v0.5.pdf — full arxiv-submittable paper (§1-8, 259KB). \
Introduces the three published laws of cognometry, documents the 3-instrument \
suite, compares against field (HHEM, Granite Guardian, Healy et al.), and \
declares failure modes openly.

- drift_phase_transitions.md + figure + raw numbers — a feature-count \
scaling ablation on the tool-call drift detector shipped the same day as \
v6.0.0 (2026-04-23). Finds that per-failure-class detection AUC does NOT \
improve smoothly with feature count — it phase-transitions in discrete \
jumps as specific features enter the classifier.

- drift_calibrated_v0.* — baseline pooled AUC 0.916 ± 0.004 on BFCL v3 \
(beats the only published hidden-state baseline at 0.72 text-only).

- Reproducers — run in <5 minutes on CPU, no API required, no LLM inference.

Software: pip install styxx==6.0.0. \
Zenodo DOI (v4.0): https://doi.org/10.5281/zenodo.19703527. \
Code: https://github.com/fathom-lab/styxx (MIT on code, CC-BY-4.0 on weights)."""


def main():
    s = requests.Session()
    s.headers.update(HDRS_JSON)

    # Create new OSF node
    payload = {
        "data": {
            "type": "nodes",
            "attributes": {
                "title": "Styxx v6.0.0 — Three Calibrated Cognometric Instruments + Phase-Transition Ablation",
                "description": DESCRIPTION,
                "category": "project",
                "public": True,
                "tags": [
                    "cognometry",
                    "hallucination-detection",
                    "tool-call-drift",
                    "refusal-detection",
                    "phase-transitions",
                    "large-language-models",
                    "LLM-safety",
                    "calibrated-lr",
                    "benchmark",
                    "open-science",
                    "cross-validation",
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
    print(f"created OSF project: {node_url}")
    print(f"  id: {node_id}")

    # Link to the parent wtkzg project as a related work
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
    print(f"  linked to wtkzg parent: HTTP {r2.status_code}")
    if r2.status_code >= 300:
        print(f"    (non-fatal) {r2.text[:200]}")

    # Upload each artifact
    print()
    for path in UPLOAD_PATHS:
        if not path.exists():
            print(f"  [skip] {path.name}: file not found")
            continue
        size_kb = path.stat().st_size / 1024
        upload_url = (
            f"https://files.us.osf.io/v1/resources/{node_id}/"
            f"providers/osfstorage/?kind=file&name={path.name}"
        )
        with open(path, "rb") as f:
            rr = requests.put(
                upload_url,
                headers={"Authorization": f"Bearer {TOKEN}"},
                data=f,
                timeout=120,
            )
        if rr.status_code < 300:
            print(f"  [ok]   {path.name} ({size_kb:.1f} KB)")
        else:
            print(f"  [FAIL] {path.name}: HTTP {rr.status_code}")
            print(f"         {rr.text[:200]}")

    print()
    print(f"OSF project LIVE: {node_url}")
    print("next: visit and confirm files render correctly")


if __name__ == "__main__":
    main()
