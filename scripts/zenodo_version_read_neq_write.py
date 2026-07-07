"""Fold the read-neq-write paper into the Fathom series lineage (concept DOI 10.5281/zenodo.19326174)
as Fathom v26, then mark the standalone record 21240661 as superseded.

Usage:
    ZENODO_TOKEN=... python scripts/zenodo_version_read_neq_write.py

House flow (mirrors clawd/scripts/zenodo-new-version.py + zenodo-fix-orphan.py, token from env):
  1. POST /deposit/depositions/{latest}/actions/newversion  -> draft in the lineage
  2. DELETE inherited files; PUT our files to the bucket
  3. PUT metadata (Fathom v26 title, this paper's abstract/keywords)
  4. POST publish -> new DOI under concept 10.5281/zenodo.19326174
  5. edit standalone 21240661: [SUPERSEDED] title + forward pointer, re-publish
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

LATEST_IN_LINEAGE = 20419662        # Fathom v25 / styxx v7.7.9 (concept 10.5281/zenodo.19326174)
STANDALONE = 21240661               # this morning's orphan deposit
CONCEPT_DOI = "10.5281/zenodo.19326174"

PKG = ROOT / "arxiv" / "read_neq_write"
FILES = [PKG / "main.pdf", PKG / "source.md", PKG / "source.certificate.json"]

DESCRIPTION = """\
<p><strong>Fathom v26.</strong> Linear probes on a language model's residual stream can read whether
the model is being honest, and converging 2026 work (Jacobian-lens workspace readouts; the
detection-vs-steering orthogonality result) establishes the read channel as real and dissociated
from the write channel. That literature shares an unexamined assumption: <strong>the weights being
read are not adversarial.</strong> We attack that assumption directly.</p>

<p>In a fully pre-registered attack&ndash;defense study, a knowledge-preserving LoRA attacker
fine-tunes the model to blind a difference-of-means honesty probe while a replay term preserves the
model's own true/false judgment. Under the registered re-lock protocol the attack <em>appears</em>
decisive: the re-fit probe reads chance (AUROC 0.461&ndash;0.507) while held-out knowledge stays at
0.817&ndash;0.890, on both seeds. A pre-committed interpretation map, frozen before that number
existed, forbade the headline and mandated the resolving experiment. The resolution: giving the
auditor a private calibration split the attacker never saw recovers the read through the same
attacked weights (AUROC 0.711&ndash;0.838) &mdash; the apparent evasion was <strong>calibration
poisoning, not erasure</strong> of the honesty signal. The defense survives an <em>adaptive</em>
attacker that re-fits a moving probe on its own clean split and scrubs along it (clean-calibrated
read 0.733&ndash;0.782, two model families).</p>

<p>The transferable audit rule: <strong>calibrate the probe on data the audited party did not
see.</strong> Load-bearing caveats reported (adaptive bite unverified, LoRA-scale attacker,
1&ndash;1.5B models); the escalation to a stronger multi-layer attacker is frozen as a public
pre-registration. All preregistrations, code, result artifacts, and machine-checked certificates
(OATH-HELD, 114 verified / 0 contradicted) are public at commit-level granularity; the study reruns
end-to-end on a single 8 GB consumer GPU. The defense ships in the software as
<code>styxx.mount.ConscienceMount.relock</code>.</p>

<p><strong>Receipts:</strong>
<a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a>,
<code>papers/read-neq-write/</code>.<br>
<strong>Software:</strong> <code>pip install styxx</code>.</p>"""

METADATA = {
    "metadata": {
        "title": (
            "Fathom v26 / styxx v7.24.3: Calibration Poisoning, Not Erasure "
            "— Substrate Honesty Probes Survive Knowledge-Preserving Weight "
            "Attacks When the Auditor's Calibration Is Private"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": DESCRIPTION,
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"},
        ],
        "keywords": [
            "interpretability", "honesty probes", "adversarial robustness",
            "calibration poisoning", "AI audit", "pre-registration",
            "linear probes", "activation probing", "residual stream",
            "AI safety", "read-write dissociation", "LoRA", "styxx", "fathom",
        ],
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "v26",
        "notes": (
            "Fathom series entry v26. Pre-registered attack-defense study; every "
            "kill-gate frozen on a public commit before the corresponding run. "
            "source.md is the canonical text, machine-certified against the four "
            "result JSONs (source.certificate.json, verdict OATH-HELD). E2 "
            "escalation pre-registered at public commit af5e184 before its run. "
            "Supersedes standalone record 10.5281/zenodo.21240661 (deposited in "
            "error outside the series lineage the same morning)."
        ),
        "related_identifiers": [
            {"identifier": "https://github.com/fathom-lab/styxx",
             "relation": "isSupplementedBy", "resource_type": "software", "scheme": "url"},
            {"identifier": "https://pypi.org/project/styxx/",
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
    # 1. new version in the lineage
    print("1/5 creating new version in the Fathom lineage...")
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

    # 2. clear inherited files, upload ours
    print("2/5 replacing files...")
    for f in nv.get("files", []):
        requests.delete(f"{BASE}/deposit/depositions/{nv_id}/files/{f['id']}", params=P, timeout=30)
    print(f"  > cleared {len(nv.get('files', []))} inherited file(s)")
    for p in FILES:
        if not p.exists():
            print(f"FAIL: missing {p}")
            sys.exit(1)
        with open(p, "rb") as fh:
            r = requests.put(f"{bucket}/{p.name}", data=fh, params=P, timeout=120)
        if r.status_code >= 400:
            die(f"upload {p.name}", r)
        print(f"  > uploaded {p.name}")

    # 3. metadata
    print("3/5 setting v26 metadata...")
    r = requests.put(f"{BASE}/deposit/depositions/{nv_id}", params=P, json=METADATA, timeout=30)
    if r.status_code >= 400:
        die("metadata", r)

    # 4. publish
    print("4/5 publishing v26...")
    r = requests.post(f"{BASE}/deposit/depositions/{nv_id}/actions/publish", params=P, timeout=60)
    if r.status_code >= 400:
        die("publish", r)
    pub = r.json()
    doi = pub.get("doi") or pub["metadata"].get("doi")
    print(f"  > v26 DOI: {doi}")

    # 5. mark the standalone superseded (house fix-orphan pattern, but preserving metadata)
    print("5/5 marking standalone superseded...")
    r = requests.post(f"{BASE}/deposit/depositions/{STANDALONE}/actions/edit", params=P, timeout=30)
    if r.status_code not in (200, 201):
        die("edit-unlock standalone", r)
    r = requests.get(f"{BASE}/deposit/depositions/{STANDALONE}", params=P, timeout=30)
    if r.status_code != 200:
        die("fetch standalone", r)
    meta = r.json()["metadata"]
    for k in ("doi", "prereserve_doi"):  # server-managed keys the PUT must not echo back
        meta.pop(k, None)
    meta["title"] = "[SUPERSEDED] " + meta["title"]
    meta["description"] = (
        "<p><strong>SUPERSEDED</strong> — this record was deposited as a standalone the same "
        f"morning it was folded into the Fathom series. The canonical record is Fathom v26: "
        f"<a href='https://doi.org/{doi}'>https://doi.org/{doi}</a>; all versions of the series: "
        f"<a href='https://doi.org/{CONCEPT_DOI}'>https://doi.org/{CONCEPT_DOI}</a>.</p>"
        + meta.get("description", "")
    )
    rel = [x for x in meta.get("related_identifiers", []) if x.get("relation") != "continues"]
    rel.append({"identifier": doi, "relation": "isSupplementedBy", "scheme": "doi"})
    meta["related_identifiers"] = rel
    r = requests.put(f"{BASE}/deposit/depositions/{STANDALONE}", params=P,
                     json={"metadata": meta}, timeout=30)
    if r.status_code >= 400:
        die("standalone metadata", r)
    r = requests.post(f"{BASE}/deposit/depositions/{STANDALONE}/actions/publish", params=P, timeout=60)
    if r.status_code >= 400:
        die("standalone re-publish", r)
    print("  > standalone marked [SUPERSEDED] with forward pointer")

    receipt = {
        "series_doi": doi,
        "series_doi_url": f"https://doi.org/{doi}",
        "concept_doi": CONCEPT_DOI,
        "concept_doi_url": f"https://doi.org/{CONCEPT_DOI}",
        "deposition_id": nv_id,
        "superseded_standalone": f"10.5281/zenodo.{STANDALONE}",
        "uploaded_files": [p.name for p in FILES],
    }
    out = PKG / "zenodo-deposit-receipt.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"  Fathom v26 DOI: {doi}")
    print(f"  Concept DOI (all versions): {CONCEPT_DOI}")
    print(f"  Receipt: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
