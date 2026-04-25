"""
Deposit the styxx v6.2.0 source bundle to Zenodo as a software record.

Creates a NEW concept on Zenodo (separate from the Fathom paper chain
at concept 19326174 / DOI 10.5281/zenodo.19326174). The software
record is cross-linked back to the Spec deposit via "isSupplementTo"
and forward-linked to PyPI via "isAlternateIdentifier".

Run:
    python scripts/zenodo_deposit_software.py

Token from clawd/secrets/arxiv-creds.txt [ZENODO] section, never echoed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

SECRETS = pathlib.Path(r"C:\Users\heyzo\clawd\secrets\arxiv-creds.txt")
REPO = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = REPO / "release" / "styxx-v6.2.0-source-bundle.zip"


def load_token() -> str:
    in_z = False
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "[ZENODO]": in_z = True; continue
        if s.startswith("[") and s.endswith("]"): in_z = False; continue
        if in_z and s.startswith("zenodo_token"):
            return s.split(":", 1)[1].strip()
    raise SystemExit("zenodo_token not found")


def api(method, url, token, data=None):
    body = None
    headers = {"Authorization": f"Bearer {token}"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            if raw and "json" in r.headers.get("content-type", ""):
                return r.status, json.loads(raw)
            return r.status, raw
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} on {method} {url}")
        print(body[:1500])
        raise


def upload(url, token, path):
    data = path.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    })
    with urllib.request.urlopen(req, timeout=600) as r:
        if r.status >= 400:
            raise RuntimeError(f"upload failed {r.status}")
    print(f"  uploaded {path.name}  ({len(data):,} bytes)")


def main():
    token = load_token()

    if not BUNDLE.exists():
        raise SystemExit(f"bundle missing: {BUNDLE}")

    print(f"depositing {BUNDLE.name}  ({BUNDLE.stat().st_size:,} bytes)\n")

    # Step 1: create NEW empty deposition
    print("step 1: create new empty deposition (software type)")
    status, dep = api("POST", "https://zenodo.org/api/deposit/depositions", token, data={})
    deposit_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"  new deposition id: {deposit_id}")
    print(f"  bucket: {bucket}\n")

    # Step 2: upload bundle
    print("step 2: upload source bundle")
    upload(f"{bucket}/{BUNDLE.name}", token, BUNDLE)
    print()

    # Step 3: set metadata
    print("step 3: set metadata")
    metadata = {
        "metadata": {
            "title": "styxx v6.2.0 — Reference Python Implementation of the Cognometric Fingerprint Specification",
            "upload_type": "software",
            "description": (
                "<p><b>styxx v6.2.0</b> is the reference Python implementation of the "
                "<a href='https://doi.org/10.5281/zenodo.19746215'>Cognometric Fingerprint "
                "Specification v1.0</a> (Fathom v20).</p>"
                "<p>Open-source cognitive observability for LLM agents — measure hallucination, refusal, "
                "tool-call drift, sycophancy, phase transitions, low-trust events, and incoherence per "
                "step in any agent run. Self-contained HTML flamegraph output, LangSmith trace export, "
                "Datadog span export, and the new <code>@styxx.profile</code> decorator API.</p>"
                "<p><b>This deposit includes:</b></p>"
                "<ul>"
                "<li><code>styxx/</code> &mdash; the Python package (90+ modules, 11 adapter integrations: openai, anthropic, langchain, llamaindex, crewai, autogen, langsmith, langfuse, raw, etc.)</li>"
                "<li><code>papers/</code> &mdash; the Cognometric Fingerprint Specification v1.0 and the Foundations of Cognometric Engineering v0.1 outline</li>"
                "<li><code>scripts/</code> &mdash; <code>generate_launch_profile.py</code>, <code>produce_fingerprint.py</code>, <code>telescope_run.py</code>, <code>zenodo_deposit_*.py</code></li>"
                "<li><code>packages/styxx-scope/</code> &mdash; browser extension v0.1.0 (Chrome/Firefox) for end-user cognitive transparency on chat.openai.com, claude.ai, gemini.google.com</li>"
                "<li><code>.github/workflows/telescope.yml</code> &mdash; Cognitive Telescope GitHub Actions workflow for daily frontier-model profiling</li>"
                "<li><code>tests/</code>, <code>examples/</code>, <code>benchmarks/</code>, <code>docs/</code>, <code>mcp/</code>, <code>openapi/</code></li>"
                "</ul>"
                "<p><b>Empirical validation:</b> AUC 0.998 on HaluEval-QA hallucination, 0.976 on XSTest GPT-4 refusal, 0.943 on BFCL v3 tool-call drift. Cross-validated across 8 public benchmarks with published failure modes. Three USPTO provisional patents protect the underlying measurement architecture (64/020,489 &middot; 64/021,113 &middot; 64/026,964).</p>"
                "<p><b>Live distribution:</b></p>"
                "<ul>"
                "<li>PyPI: <a href='https://pypi.org/project/styxx/6.2.0/'>pypi.org/project/styxx/6.2.0/</a></li>"
                "<li>Source: <a href='https://github.com/fathom-lab/styxx'>github.com/fathom-lab/styxx</a></li>"
                "<li>Documentation: <a href='https://fathom.darkflobi.com/profile'>fathom.darkflobi.com/profile</a></li>"
                "<li>Specification: <a href='https://doi.org/10.5281/zenodo.19746215'>doi:10.5281/zenodo.19746215</a></li>"
                "<li>Browser extension: <a href='https://fathom.darkflobi.com/scope'>fathom.darkflobi.com/scope</a></li>"
                "<li>Live scoreboard: <a href='https://fathom.darkflobi.com/scoreboard'>fathom.darkflobi.com/scoreboard</a></li>"
                "</ul>"
                "<p>Code: MIT. Atlas data: CC-BY-4.0. Spec methodology: CC-BY-4.0. Foundations monograph: CC-BY-SA-4.0.</p>"
            ),
            "creators": [
                {"name": "Fathom Lab", "affiliation": "Fathom Lab"}
            ],
            "keywords": [
                "cognometrics", "cognitive observability", "llm agents",
                "hallucination detection", "refusal detection", "tool-call drift",
                "sycophancy detection", "phase transitions", "ai safety",
                "interpretability", "python", "specification implementation",
                "styxx", "fathom lab", "open source"
            ],
            "license": "MIT",
            "version": "6.2.0",
            "language": "eng",
            "related_identifiers": [
                {
                    "relation": "isSupplementTo",
                    "identifier": "10.5281/zenodo.19746215",
                    "resource_type": "publication-workingpaper",
                },
                {
                    "relation": "isPartOf",
                    "identifier": "10.5281/zenodo.19326174",
                    "resource_type": "publication-workingpaper",
                },
                {
                    "relation": "isAlternateIdentifier",
                    "identifier": "https://pypi.org/project/styxx/6.2.0/",
                    "resource_type": "software",
                },
                {
                    "relation": "isAlternateIdentifier",
                    "identifier": "https://github.com/fathom-lab/styxx",
                    "resource_type": "software",
                },
                {
                    "relation": "isDocumentedBy",
                    "identifier": "https://fathom.darkflobi.com/profile",
                    "resource_type": "publication-workingpaper",
                },
            ],
            "notes": (
                "First Zenodo software deposit for the styxx project. "
                "Future minor versions (6.2.x patches) will be uploaded as new versions "
                "of this concept record. Major versions (7.x) may either continue here "
                "or branch as separate concept records depending on architectural changes."
            ),
        }
    }
    status, _ = api("PUT", f"https://zenodo.org/api/deposit/depositions/{deposit_id}", token, data=metadata)
    print(f"  metadata status: {status}\n")

    # Step 4: publish
    print("step 4: publish")
    status, published = api("POST", f"https://zenodo.org/api/deposit/depositions/{deposit_id}/actions/publish", token)
    print(f"  publish status: {status}\n")

    # Step 5: fetch public record view for the final URLs
    req = urllib.request.Request(f"https://zenodo.org/api/records/{deposit_id}",
                                  headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rec = json.load(r)

    new_doi = rec.get("doi")
    concept_doi = rec.get("conceptdoi")

    print("=" * 64)
    print("PUBLISHED — software record")
    print("=" * 64)
    print(f"  software DOI:   {new_doi}")
    print(f"  software DOI url: https://doi.org/{new_doi}")
    print(f"  concept DOI:    {concept_doi}")
    print(f"  record page:    https://zenodo.org/records/{rec.get('id')}")
    print(f"  concept page:   https://zenodo.org/records/{rec.get('conceptrecid')}")

    receipt = {
        "action": "software_deposit_published",
        "deposit_id": rec.get("id"),
        "software_doi": new_doi,
        "software_doi_url": f"https://doi.org/{new_doi}",
        "concept_doi": concept_doi,
        "concept_doi_url": f"https://doi.org/{concept_doi}",
        "record_url": f"https://zenodo.org/records/{rec.get('id')}",
        "concept_url": f"https://zenodo.org/records/{rec.get('conceptrecid')}",
        "title": rec.get("metadata", {}).get("title"),
        "version": rec.get("metadata", {}).get("version"),
        "files": [f.get("key") for f in rec.get("files", [])],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPO / "release" / "zenodo-deposit-receipt-software-v6.2.0.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
