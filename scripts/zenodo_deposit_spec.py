"""Deposit the Cognometric Fingerprint Specification v1.0 as a new
version of the existing Fathom Zenodo concept record.

Concept record (stable DOI): 10.5281/zenodo.19703526
Latest version as of 2026-04-24: ID 19719347 on conceptrecid 19703526.

This script:
 1. Finds the latest version on the concept record.
 2. Creates a new draft version via the `newversion` action link.
 3. Clears the draft's file list (Zenodo copies the previous files —
    we want to upload fresh).
 4. Uploads the Spec v1.0 markdown and the reference fingerprint JSON.
 5. Updates metadata (title, description, version, keywords, related
    identifiers pointing back to prior versions).
 6. Publishes.
 7. Prints the new DOI + record URL.

Token is read from clawd/secrets/arxiv-creds.txt [ZENODO] section,
never echoed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

SECRETS = pathlib.Path(r"C:\Users\heyzo\clawd\secrets\arxiv-creds.txt")
REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC_MD = REPO / "papers" / "cognometric-fingerprint-spec-v1.0.md"
FOUNDATIONS_MD = REPO / "papers" / "foundations-of-cognometric-engineering-v0.1.md"
FINGERPRINT_JSON = REPO / "scratch" / "fingerprint_seed-bench_reference.json"

CONCEPT_RECID = "19326174"  # Fathom main concept chain (v19 was previous latest)


def load_token() -> str:
    in_zenodo = False
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "[ZENODO]":
            in_zenodo = True
            continue
        if s.startswith("[") and s.endswith("]"):
            in_zenodo = False
            continue
        if in_zenodo and s.startswith("zenodo_token"):
            return s.split(":", 1)[1].strip()
    raise SystemExit("zenodo_token not found in secrets")


def api(method: str, url: str, token: str, data=None, headers=None, files=None):
    """Minimal JSON API client using urllib (no external deps)."""
    body_bytes = None
    final_headers = {"Authorization": f"Bearer {token}"}
    if headers:
        final_headers.update(headers)
    if data is not None and files is None:
        body_bytes = json.dumps(data).encode("utf-8")
        final_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, method=method, headers=final_headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            ct = r.headers.get("content-type", "")
            if "json" in ct and raw:
                return r.status, json.loads(raw)
            return r.status, raw
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} on {method} {url}")
        print(body[:1000])
        raise


def upload_file(url: str, token: str, path: pathlib.Path) -> None:
    """Upload a file to a Zenodo deposit's bucket URL."""
    data = path.read_bytes()
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        status = r.status
        body = r.read()
    if status >= 400:
        raise RuntimeError(f"upload failed {status}: {body[:500]}")
    print(f"    uploaded {path.name}  ({len(data):,} bytes  status={status})")


def main() -> None:
    token = load_token()
    print(f"token: {len(token)} chars loaded\n")

    # ── Step 1: find the latest version on the concept record ───────
    print(f"step 1: looking up latest version on concept record {CONCEPT_RECID}")
    status, rec = api("GET", f"https://zenodo.org/api/records/{CONCEPT_RECID}/versions/latest", token)
    latest_id = rec.get("id")
    latest_doi = rec.get("doi")
    latest_version = rec.get("metadata", {}).get("version", "?")
    latest_title = rec.get("metadata", {}).get("title", "?")
    print(f"    latest version:  id={latest_id}  doi={latest_doi}")
    print(f"    latest version:  version={latest_version}")
    print(f"    latest title:    {latest_title}")

    # ── Step 2: get the deposition (editable form) ───────────────────
    print(f"\nstep 2: fetching deposition {latest_id}")
    status, dep = api("GET", f"https://zenodo.org/api/deposit/depositions/{latest_id}", token)
    newversion_url = dep.get("links", {}).get("newversion")
    if not newversion_url:
        raise SystemExit(f"no newversion link on deposition {latest_id} — state may be a draft")
    print(f"    newversion link: {newversion_url}")

    # ── Step 3: trigger newversion action ───────────────────────────
    print(f"\nstep 3: creating new draft version")
    status, drafted = api("POST", newversion_url, token)
    # The response is the previous record — the draft lives under .links.latest_draft
    draft_url = drafted.get("links", {}).get("latest_draft")
    if not draft_url:
        # Fallback: re-fetch and look for draft
        status, dep2 = api("GET", f"https://zenodo.org/api/deposit/depositions/{latest_id}", token)
        draft_url = dep2.get("links", {}).get("latest_draft")
    print(f"    draft url: {draft_url}")

    # Fetch the new draft deposition
    status, draft = api("GET", draft_url, token)
    draft_id = draft.get("id")
    print(f"    new draft id:    {draft_id}")
    print(f"    new draft state: {draft.get('state')}  submitted={draft.get('submitted')}")

    # Bucket url for file uploads
    bucket_url = draft.get("links", {}).get("bucket")
    print(f"    bucket url:      {bucket_url}")

    # ── Step 4: remove copied files from the new draft ───────────────
    existing_files = draft.get("files", [])
    print(f"\nstep 4: clearing {len(existing_files)} copied file(s) from draft")
    for f in existing_files:
        file_id = f.get("id")
        file_name = f.get("filename", f.get("key", "?"))
        print(f"    deleting {file_name}")
        del_url = f"https://zenodo.org/api/deposit/depositions/{draft_id}/files/{file_id}"
        api("DELETE", del_url, token)

    # ── Step 5: upload our new files ─────────────────────────────────
    print(f"\nstep 5: uploading new files")
    for path in [SPEC_MD, FOUNDATIONS_MD, FINGERPRINT_JSON]:
        if not path.exists():
            print(f"    SKIP missing: {path}")
            continue
        upload_file(f"{bucket_url}/{path.name}", token, path)

    # ── Step 6: update metadata ──────────────────────────────────────
    print(f"\nstep 6: updating metadata")
    metadata = {
        "metadata": {
            "title": "Fathom v20 / styxx v6.2.0: Cognometric Fingerprint Specification v1.0 — Open Reference for Measuring AI Cognition",
            "upload_type": "publication",
            "publication_type": "workingpaper",
            "description": (
                "<p><b>Fathom v20 · styxx v6.2.0</b> — the first open reference specification for measuring, classifying, and "
                "comparing the cognitive state of a language model during generation. "
                "Defines three orthogonal measurement axes — K (reasoning depth), C (coherence/commitment), "
                "and D (dissociation/drift) — a canonical taxonomy of seven fault kinds, and the "
                "cognometric fingerprint: a calibrated, reproducible multi-dimensional readout "
                "comparable across model families, substrates, and time.</p>"
                "<p>Orthogonality of the three axes empirically verified at 86.7–91.9° on Llama-3.2-1B "
                "layer 10. Cross-validated cognometric instruments achieve AUC 0.998 on HaluEval-QA, "
                "0.976 on XSTest GPT-4 out-of-family, and 0.943 on BFCL v3 — with published failure modes.</p>"
                "<p>This deposit includes:</p>"
                "<ul>"
                "<li><b>cognometric-fingerprint-spec-v1.0.md</b> — the specification itself (MUST/SHOULD/MAY "
                "conformance), CC-BY-4.0</li>"
                "<li><b>foundations-of-cognometric-engineering-v0.1.md</b> — research-programme outline "
                "for a 400-page monograph proposing cognitive engineering as a new engineering discipline, "
                "CC-BY-SA-4.0</li>"
                "<li><b>fingerprint_seed-bench_reference.json</b> — first spec-v1.0-conformant "
                "cognometric fingerprint on record, SHA-256-attested, produced by styxx v6.2.0 "
                "against the 10-prompt Seed-Bench v0</li>"
                "</ul>"
                "<p>Patent priority anchored via US Provisionals 64/020,489 (K axis), "
                "64/021,113 (C axis), 64/026,964 (three-axis spectrometry + cognitive governor).</p>"
                "<p>Reference implementation: <b>styxx v6.2.0</b> at "
                "<a href='https://pypi.org/project/styxx/6.2.0/'>pypi.org/project/styxx</a>.</p>"
                "<p>Landing: <a href='https://fathom.darkflobi.com/spec'>fathom.darkflobi.com/spec</a></p>"
            ),
            "creators": [
                {"name": "Fathom Lab", "affiliation": "Fathom Lab"}
            ],
            "keywords": [
                "cognometrics",
                "cognitive fingerprint",
                "llm observability",
                "hallucination detection",
                "refusal detection",
                "tool-call drift",
                "ai measurement standard",
                "specification",
                "reference implementation",
                "styxx",
                "fathom lab"
            ],
            "license": "CC-BY-4.0",
            "version": "v20",
            "related_identifiers": [
                {
                    "relation": "isNewVersionOf",
                    "identifier": latest_doi,
                    "resource_type": "publication-workingpaper"
                },
                {
                    "relation": "isSupplementTo",
                    "identifier": "10.5281/zenodo.19703526",
                    "resource_type": "publication-workingpaper"
                },
                {
                    "relation": "isDocumentedBy",
                    "identifier": "https://pypi.org/project/styxx/6.2.0/",
                    "resource_type": "software"
                },
                {
                    "relation": "isDocumentedBy",
                    "identifier": "https://fathom.darkflobi.com/spec",
                    "resource_type": "publication-workingpaper"
                }
            ],
            "notes": (
                "This version ships the first open reference specification for "
                "cognometric measurement. It intentionally renames the deposit "
                "series from the prior Cognometry benchmark deposits to reflect "
                "the shift from empirical paper series to foundational spec."
            )
        }
    }
    status, updated = api("PUT", draft_url, token, data=metadata)
    print(f"    metadata update status: {status}")

    # ── Step 7: publish ───────────────────────────────────────────────
    publish_url = draft.get("links", {}).get("publish")
    if not publish_url:
        # re-fetch to get the link after metadata update
        status, draft2 = api("GET", draft_url, token)
        publish_url = draft2.get("links", {}).get("publish")
    print(f"\nstep 7: publishing via {publish_url}")
    status, published = api("POST", publish_url, token)
    print(f"    publish status: {status}")

    # ── Report ───────────────────────────────────────────────────────
    new_doi = published.get("doi")
    new_doi_url = published.get("doi_url")
    record_url = published.get("links", {}).get("record_html") or \
                 published.get("links", {}).get("html")
    concept_doi = published.get("conceptdoi")
    print()
    print("═" * 64)
    print("PUBLISHED")
    print("═" * 64)
    print(f"new version DOI:   {new_doi}")
    print(f"new version URL:   {new_doi_url}")
    print(f"record page:       {record_url}")
    print(f"stable concept DOI (always latest): {concept_doi}")
    print()

    # Save receipt
    receipt = {
        "action": "new_version_published",
        "concept_recid": CONCEPT_RECID,
        "concept_doi": concept_doi,
        "new_deposition_id": published.get("id"),
        "new_doi": new_doi,
        "new_doi_url": new_doi_url,
        "record_url": record_url,
        "previous_latest": {
            "id": latest_id,
            "doi": latest_doi,
            "title": latest_title,
            "version": latest_version,
        },
        "this_version": {
            "version": "spec-v1.0",
            "title": metadata["metadata"]["title"],
            "uploaded_files": [
                SPEC_MD.name, FOUNDATIONS_MD.name, FINGERPRINT_JSON.name
            ],
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_path = REPO / "release" / f"zenodo-deposit-receipt-spec-v1.0.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"receipt written: {receipt_path}")


if __name__ == "__main__":
    main()
