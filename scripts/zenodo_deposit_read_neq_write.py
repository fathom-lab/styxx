"""Deposit the read-neq-write paper (Calibration Poisoning, Not Erasure v0.1) to Zenodo.

Usage:
    ZENODO_TOKEN=... python scripts/zenodo_deposit_read_neq_write.py

Mirrors scripts/zenodo_deposit.py (house pattern):
  1. POST /api/deposit/depositions   -> create empty deposition
  2. PUT  <bucket>/<filename>        -> upload files
  3. PUT  /api/deposit/depositions/{id}  -> metadata
  4. POST /api/deposit/depositions/{id}/actions/publish -> DOI
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

PKG = ROOT / "arxiv" / "read_neq_write"
FILES_TO_UPLOAD = [
    PKG / "main.pdf",
    PKG / "source.md",
    PKG / "source.certificate.json",
]

DESCRIPTION = """\
<p>Linear probes on a language model's residual stream can read whether the model
is being honest, and converging 2026 work (Jacobian-lens workspace readouts; the
detection-vs-steering orthogonality result) establishes the read channel as real
and dissociated from the write channel. That literature shares an unexamined
assumption: <strong>the weights being read are not adversarial.</strong> We attack
that assumption directly.</p>

<p>In a fully pre-registered attack&ndash;defense study, a knowledge-preserving
LoRA attacker fine-tunes the model to blind a difference-of-means honesty probe
while a replay term preserves the model's own true/false judgment. Under the
registered re-lock protocol the attack <em>appears</em> decisive: the re-fit probe
reads chance (AUROC 0.461&ndash;0.507) while held-out knowledge stays at
0.817&ndash;0.890, on both seeds. A pre-committed interpretation map, frozen before
that number existed, forbade the headline and mandated the resolving experiment.
The resolution: giving the auditor a private calibration split the attacker never
saw recovers the read through the same attacked weights (AUROC 0.711&ndash;0.838)
&mdash; the apparent evasion was <strong>calibration poisoning, not erasure</strong>
of the honesty signal. The defense survives an <em>adaptive</em> attacker that
re-fits a moving probe on its own clean split and scrubs along it (clean-calibrated
read 0.733&ndash;0.782, two model families).</p>

<p>The transferable audit rule: <strong>calibrate the probe on data the audited
party did not see.</strong> We report the load-bearing caveats (adaptive bite
unverified, LoRA-scale attacker, 1&ndash;1.5B models); the escalation to a stronger
multi-layer attacker is frozen as a public pre-registration before its run. All
preregistrations, code, result artifacts, and machine-checked certificates
(OATH-HELD, 114 verified / 0 contradicted) are public at commit-level granularity;
the study reruns end-to-end on a single 8 GB consumer GPU.</p>

<p><strong>Receipts:</strong>
<a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a>,
<code>papers/read-neq-write/</code> &mdash; preregs frozen on public commits before
each run.<br>
<strong>Software:</strong> <code>pip install styxx</code> (PyPI, 130 releases since
2026-04-11).</p>"""

METADATA = {
    "metadata": {
        "title": (
            "Calibration Poisoning, Not Erasure: Substrate Honesty Probes "
            "Survive Knowledge-Preserving Weight Attacks When the Auditor's "
            "Calibration Is Private"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": DESCRIPTION,
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"},
        ],
        "keywords": [
            "interpretability",
            "honesty probes",
            "adversarial robustness",
            "calibration poisoning",
            "AI audit",
            "pre-registration",
            "linear probes",
            "activation probing",
            "residual stream",
            "AI safety",
            "read-write dissociation",
            "LoRA",
            "styxx",
        ],
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "0.1",
        "notes": (
            "Pre-registered attack-defense study; every kill-gate frozen on a "
            "public commit before the corresponding run. source.md is the "
            "canonical text, machine-certified against the four result JSONs "
            "(source.certificate.json, verdict OATH-HELD). Reruns on one 8 GB "
            "consumer GPU. E2 escalation pre-registered at public commit "
            "af5e184 before its run."
        ),
        "related_identifiers": [
            {
                "identifier": "https://github.com/fathom-lab/styxx",
                "relation": "isSupplementedBy",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "10.5281/zenodo.19326174",
                "relation": "continues",
                "scheme": "doi",
            },
            {
                "identifier": "arXiv:2606.24952",
                "relation": "references",
                "scheme": "arxiv",
            },
            {
                "identifier": "arXiv:2502.03407",
                "relation": "references",
                "scheme": "arxiv",
            },
        ],
    },
}


def step(msg):
    print(f"  > {msg}")


def main():
    print("1/4 creating deposition...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions", params=AUTH_PARAMS, json={},
        timeout=30,
    )
    if r.status_code >= 400:
        print(f"FAIL: {r.status_code} {r.text[:1000]}")
        sys.exit(1)
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    step(f"deposition id: {dep_id}")

    print("\n2/4 uploading files...")
    for p in FILES_TO_UPLOAD:
        if not p.exists():
            print(f"FAIL: missing file {p}")
            sys.exit(1)
        with open(p, "rb") as f:
            r = requests.put(
                f"{bucket}/{p.name}", data=f, params=AUTH_PARAMS, timeout=120,
            )
        if r.status_code >= 400:
            print(f"FAIL upload {p.name}: {r.status_code} {r.text[:500]}")
            sys.exit(1)
        step(f"uploaded {p.name} ({p.stat().st_size} bytes)")

    print("\n3/4 setting metadata...")
    r = requests.put(
        f"{ZENODO_API}/deposit/depositions/{dep_id}",
        params=AUTH_PARAMS, json=METADATA, timeout=30,
    )
    if r.status_code >= 400:
        print(f"FAIL metadata: {r.status_code}")
        print(r.text[:2000])
        print(f"\nDeposition draft exists: https://zenodo.org/deposit/{dep_id}")
        sys.exit(1)
    step("metadata attached")

    print("\n4/4 publishing...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions/{dep_id}/actions/publish",
        params=AUTH_PARAMS, timeout=60,
    )
    if r.status_code >= 400:
        print(f"FAIL publish: {r.status_code}")
        print(r.text[:2000])
        print(f"\nDeposition draft exists: https://zenodo.org/deposit/{dep_id}"
              " -- complete manually.")
        sys.exit(1)

    published = r.json()
    doi = published.get("doi") or published["metadata"].get("doi")
    record_url = published.get("links", {}).get("record_html")

    print("\n" + "=" * 60)
    print(f"  DOI: {doi}")
    print(f"  URL: https://doi.org/{doi}")
    print(f"  Record page: {record_url}")
    print("=" * 60)

    out_path = PKG / "zenodo-deposit-receipt.json"
    out_path.write_text(
        json.dumps({
            "doi": doi,
            "doi_url": f"https://doi.org/{doi}" if doi else None,
            "record_url": record_url,
            "deposition_id": dep_id,
            "uploaded_files": [p.name for p in FILES_TO_UPLOAD],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nReceipt saved: {out_path}")


if __name__ == "__main__":
    main()
