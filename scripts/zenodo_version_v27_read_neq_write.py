"""Deposit read-neq-write paper v0.2 as Fathom v27 -- a new version of the series lineage.

v0.2 folds in three same-day escalations: E2 INCONCLUSIVE + E2' STANDS_vs_strong (strong-attacker
axis strengthened) and E3' NO_BITE (the adaptive claim demoted). Deposited under concept DOI
10.5281/zenodo.19326174 as the successor to v26 (21241185).

Usage: ZENODO_TOKEN=... python scripts/zenodo_version_v27_read_neq_write.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN")
if not TOKEN:
    print("ERROR: set ZENODO_TOKEN env var", file=sys.stderr)
    sys.exit(2)
P = {"access_token": TOKEN}

LATEST_IN_LINEAGE = 21241185        # Fathom v26 (concept 10.5281/zenodo.19326174)
CONCEPT_DOI = "10.5281/zenodo.19326174"

PKG = ROOT / "arxiv" / "read_neq_write"
FILES = [PKG / "main.pdf", PKG / "source.md", PKG / "source.certificate.json"]

DESCRIPTION = """\
<p><strong>Fathom v27 (v0.2 of the read!=write attack-defense study; supersedes v26,
DOI 10.5281/zenodo.21241185).</strong> Linear probes on a language model's residual stream can read
whether the model is being honest; converging 2026 work established the read channel as real and
dissociated from the write channel, on the unexamined assumption that the weights being read are not
adversarial. We attack that assumption.</p>

<p>A knowledge-preserving LoRA attacker blinds a difference-of-means honesty probe while a replay
term preserves the model's true/false judgment. Under the registered re-lock protocol the attack
appears decisive (probe AUROC 0.461-0.507, knowledge kept) -- but that is <strong>calibration
poisoning, not erasure</strong>: a probe fit on a private clean split the attacker never saw reads
straight through the attack (AUROC 0.711-0.838). The defense also holds against a
<strong>4x-capacity, whole-stack attacker forced to preserve knowledge</strong> (E2': clean-calibrated
read 0.754-0.759 while the frozen read is driven to 0.53-0.55), the strongest attacker this bench
supports.</p>

<p><strong>Self-correction in this version:</strong> a pre-committed verification (E3') showed the
"adaptive" attacker of v26 did NOT out-attack the naive one -- on the read it targets it left more
signal than naive -- so we DEMOTE v26's "survives an adaptive attacker" claim to what is earned
(robustness to the naive and to the knowledge-preserving strong attacker; genuine adaptive escalation
open). The demotion was produced by our own frozen verification, not external review.</p>

<p>The transferable audit rule: <strong>calibrate the probe on data the audited party did not
see.</strong> The defense ships as <code>styxx.mount.ConscienceMount.relock</code>. All
preregistrations, code, result artifacts, and machine-checked certificates (OATH-HELD, 146 verified /
0 contradicted against seven result JSONs) are public at commit-level granularity;
<code>papers/read-neq-write/</code> at github.com/fathom-lab/styxx.</p>"""

METADATA = {
    "metadata": {
        "title": (
            "Fathom v27 / styxx v7.24.3: Calibration Poisoning, Not Erasure "
            "(v0.2) -- Substrate Honesty Probes Survive Knowledge-Preserving "
            "Weight Attacks When the Auditor's Calibration Is Private"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": DESCRIPTION,
        "creators": [{"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"}],
        "keywords": [
            "interpretability", "honesty probes", "adversarial robustness",
            "calibration poisoning", "AI audit", "pre-registration", "self-falsification",
            "linear probes", "activation probing", "residual stream", "AI safety",
            "read-write dissociation", "LoRA", "styxx", "fathom",
        ],
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "v27",
        "notes": (
            "Fathom series v27 (v0.2). Supersedes v26 (10.5281/zenodo.21241185): "
            "E2/E2' strengthen the strong-attacker result; E3' demotes v26's adaptive "
            "claim (the tested adaptive scheme did not out-attack naive). Pre-registered "
            "attack-defense study; every kill-gate frozen on a public commit before its "
            "run. source.md machine-certified against seven result JSONs "
            "(source.certificate.json, OATH-HELD 146/0). Reruns on one 8 GB consumer GPU."
        ),
        "related_identifiers": [
            {"identifier": "https://github.com/fathom-lab/styxx",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "arXiv:2606.24952", "relation": "references", "scheme": "arxiv"},
            {"identifier": "arXiv:2502.03407", "relation": "references", "scheme": "arxiv"},
        ],
    },
}


def die(msg, r):
    print(f"FAIL {msg}: {r.status_code}\n{r.text[:1500]}")
    sys.exit(1)


def main():
    print("1/4 creating new version in the Fathom lineage (off v26)...")
    r = requests.post(f"{BASE}/deposit/depositions/{LATEST_IN_LINEAGE}/actions/newversion",
                      params=P, timeout=60)
    if r.status_code not in (200, 201):
        die("newversion", r)
    draft_url = r.json()["links"]["latest_draft"]
    nv_id = int(draft_url.rstrip("/").split("/")[-1])
    r = requests.get(f"{BASE}/deposit/depositions/{nv_id}", params=P, timeout=30)
    if r.status_code != 200:
        die("draft fetch", r)
    nv = r.json()
    bucket = nv["links"]["bucket"]
    print(f"  > draft deposition: {nv_id}")

    print("2/4 replacing files...")
    for f in nv.get("files", []):
        requests.delete(f"{BASE}/deposit/depositions/{nv_id}/files/{f['id']}", params=P, timeout=30)
    for p in FILES:
        if not p.exists():
            print(f"FAIL: missing {p}")
            sys.exit(1)
        with open(p, "rb") as fh:
            r = requests.put(f"{bucket}/{p.name}", data=fh, params=P, timeout=120)
        if r.status_code >= 400:
            die(f"upload {p.name}", r)
        print(f"  > uploaded {p.name}")

    print("3/4 setting v27 metadata...")
    r = requests.put(f"{BASE}/deposit/depositions/{nv_id}", params=P, json=METADATA, timeout=30)
    if r.status_code >= 400:
        die("metadata", r)

    print("4/4 publishing v27...")
    r = requests.post(f"{BASE}/deposit/depositions/{nv_id}/actions/publish", params=P, timeout=60)
    if r.status_code >= 400:
        die("publish", r)
    pub = r.json()
    doi = pub.get("doi") or pub["metadata"].get("doi")

    receipt = {
        "series_doi": doi,
        "series_doi_url": f"https://doi.org/{doi}",
        "concept_doi": CONCEPT_DOI,
        "concept_doi_url": f"https://doi.org/{CONCEPT_DOI}",
        "deposition_id": nv_id,
        "supersedes": "10.5281/zenodo.21241185",
        "version": "v27 / v0.2",
        "uploaded_files": [p.name for p in FILES],
    }
    out = PKG / "zenodo-deposit-receipt-v27.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"  Fathom v27 DOI: {doi}")
    print(f"  Concept DOI (all versions): {CONCEPT_DOI}")
    print(f"  Receipt: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
