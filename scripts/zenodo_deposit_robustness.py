"""Deposit the Spec v1.0 Robustness Supplement as Fathom v22.

New version on the canonical Fathom concept chain (19326174 / DOI
10.5281/zenodo.19326174). Successor to Fathom v20 (the spec) and
v21 (which would be the placeholder space for any sibling work).

Files:
    papers/spec-v1.0-robustness-supplement.md
    packages/styxx-scope/_test_adversarial.js
    packages/styxx-scope/_adversarial_report.json
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request

SECRETS = pathlib.Path(r"C:\Users\heyzo\clawd\secrets\arxiv-creds.txt")
REPO = pathlib.Path(__file__).resolve().parent.parent

CONCEPT_RECID = "19326174"
FILES = [
    REPO / "papers" / "spec-v1.0-robustness-supplement.md",
    REPO / "packages" / "styxx-scope" / "_test_adversarial.js",
    REPO / "packages" / "styxx-scope" / "_adversarial_report.json",
]


def load_token():
    in_z = False
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "[ZENODO]": in_z = True; continue
        if s.startswith("[") and s.endswith("]"): in_z = False; continue
        if in_z and s.startswith("zenodo_token"):
            return s.split(":", 1)[1].strip()
    raise SystemExit("zenodo_token missing")


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
        print(f"HTTP {e.code} on {method} {url}\n{body[:1500]}")
        raise


def upload(url, token, path):
    data = path.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    })
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status >= 400:
            raise RuntimeError(f"upload failed {r.status}")
    print(f"  uploaded {path.name} ({len(data):,} bytes)")


def main():
    token = load_token()

    print("step 1: get latest version on canonical Fathom chain")
    status, rec = api("GET", f"https://zenodo.org/api/records/{CONCEPT_RECID}/versions/latest", token)
    latest_id = rec["id"]
    latest_doi = rec["doi"]
    latest_version = rec.get("metadata", {}).get("version", "?")
    print(f"  latest: id={latest_id}  doi={latest_doi}  v={latest_version}\n")

    print("step 2: get newversion link")
    status, dep = api("GET", f"https://zenodo.org/api/deposit/depositions/{latest_id}", token)
    newversion_url = dep["links"]["newversion"]

    print("step 3: create new draft")
    api("POST", newversion_url, token)
    status, dep2 = api("GET", f"https://zenodo.org/api/deposit/depositions/{latest_id}", token)
    draft_url = dep2["links"]["latest_draft"]
    status, draft = api("GET", draft_url, token)
    draft_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"  draft id: {draft_id}\n")

    print("step 4: clear copied files")
    for f in draft.get("files", []):
        del_url = f"https://zenodo.org/api/deposit/depositions/{draft_id}/files/{f['id']}"
        api("DELETE", del_url, token)
        print(f"  deleted {f.get('filename')}")

    print("\nstep 5: upload new files")
    for path in FILES:
        if path.exists():
            upload(f"{bucket}/{path.name}", token, path)

    print("\nstep 6: set metadata")
    metadata = {
        "metadata": {
            "title": "Fathom v22 / styxx v6.2.0: Cognometric Fingerprint Specification v1.0 — Robustness Supplement (v0.1)",
            "upload_type": "publication",
            "publication_type": "workingpaper",
            "description": (
                "<p><b>Fathom v22 · styxx v6.2.0 — Robustness Supplement to Spec v1.0</b></p>"
                "<p>First systematic adversarial robustness audit of the Tier-3 (text-only proxy-signal) classifier "
                "specified in §5.1.2 of the <a href='https://doi.org/10.5281/zenodo.19746215'>Cognometric Fingerprint "
                "Specification v1.0</a> (Fathom v20). Constructs 24 canonical attack prompts spanning paraphrase, "
                "obfuscation, unicode-substitution, case-folding, density-thresholding, meta-discussion, inversion, "
                "and interleaving strategies, organized across the seven fault kinds defined in Spec §4.</p>"
                "<p><b>Headline result:</b></p>"
                "<ul>"
                "<li>Baseline classifier: <b>66.7% false-negative</b> attack success · <b>66.7% false-positive</b> attack success</li>"
                "<li>After three iterations of targeted hardening (v0.2.0 → v0.2.3): <b>16.7% false-negative</b> (4× reduction) · <b>50% false-positive</b> (1.3× reduction)</li>"
                "<li>Five fault kinds (drift, refusal, low_trust, incoherence, partial adversarial) went from 100% to 0% evasion</li>"
                "<li>All 26 canonical validation tests still pass post-hardening — zero regressions</li>"
                "</ul>"
                "<p><b>Defensive techniques formalized in this supplement:</b> Unicode NFKD normalization + Cyrillic homoglyph folding · "
                "punctuation-strip pass for adversarial matching · soft-refusal pattern bank · helpful-disclaimer suppressor · "
                "hedge-not-refusal subtraction (\"I cannot guarantee\" ≠ refusal) · tool-flip drift detection · heavy-hedge low-trust override · "
                "topic-jump incoherence via lexical-overlap · meta-discussion suppressor · single-token agreement / wordCount sycophant tightening · "
                "unverified_claims warning rule.</p>"
                "<p><b>Honest residual limits documented:</b> confabulation vs retrieval is fundamentally text-ambiguous (no text-only classifier "
                "can distinguish a real fact about a real compound from a fabricated fact about a fictional one without a knowledge base); "
                "sycophant evasion via substantive padding requires semantic-similarity comparison beyond Tier-3 capability; "
                "refusal-preamble masking confabulation is a single-category-per-response limitation requiring chunk-level classification.</p>"
                "<p>The audit suite is fully reproducible (<code>node _test_adversarial.js</code>) and intended to grow. We welcome "
                "adversarial submissions via PR.</p>"
                "<p><b>This is the first public robustness benchmark in the cognometric observability space.</b></p>"
                "<p>Reference implementation: styxx v6.2.0 ([doi:10.5281/zenodo.19758619](https://doi.org/10.5281/zenodo.19758619)). "
                "Spec v1.0: doi:[10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215). "
                "Concept (always-latest): doi:[10.5281/zenodo.19326174](https://doi.org/10.5281/zenodo.19326174).</p>"
            ),
            "creators": [{"name": "Fathom Lab", "affiliation": "Fathom Lab"}],
            "keywords": [
                "cognometric fingerprint", "adversarial robustness", "text classifier audit",
                "ai safety evaluation", "robustness benchmark", "spec v1.0",
                "false-negative attack", "false-positive attack", "tier-3 pipeline",
                "styxx", "fathom lab", "robustness supplement"
            ],
            "license": "CC-BY-4.0",
            "version": "v22",
            "language": "eng",
            "related_identifiers": [
                {"relation": "isNewVersionOf", "identifier": latest_doi, "resource_type": "publication-workingpaper"},
                {"relation": "isSupplementTo", "identifier": "10.5281/zenodo.19746215", "resource_type": "publication-workingpaper"},
                {"relation": "isDocumentedBy", "identifier": "10.5281/zenodo.19758619", "resource_type": "software"},
                {"relation": "isPartOf", "identifier": "10.5281/zenodo.19326174", "resource_type": "publication-workingpaper"},
            ],
            "notes": "v22 of the canonical Fathom chain. Robustness Supplement formalizes the first systematic adversarial audit of cognometric measurement.",
        }
    }
    status, _ = api("PUT", draft_url, token, data=metadata)
    print(f"  metadata: {status}")

    print("\nstep 7: publish")
    status, dep3 = api("GET", draft_url, token)
    publish_url = dep3["links"]["publish"]
    api("POST", publish_url, token)

    # Fetch the public record view
    req = urllib.request.Request(f"https://zenodo.org/api/records/{draft_id}",
                                  headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rec = json.load(r)
    new_doi = rec["doi"]
    print(f"\n{'=' * 64}")
    print(f"PUBLISHED — Fathom v22 / Robustness Supplement")
    print(f"{'=' * 64}")
    print(f"  DOI:           {new_doi}")
    print(f"  DOI URL:       https://doi.org/{new_doi}")
    print(f"  Record:        https://zenodo.org/records/{rec['id']}")
    print(f"  Concept:       https://doi.org/{rec.get('conceptdoi')}")
    print()

    receipt = {
        "action": "v22_robustness_supplement_published",
        "deposit_id": rec["id"],
        "doi": new_doi,
        "concept_doi": rec.get("conceptdoi"),
        "record_url": f"https://zenodo.org/records/{rec['id']}",
        "title": rec["metadata"]["title"],
        "version": rec["metadata"]["version"],
        "files": [f["key"] for f in rec.get("files", [])],
        "predecessor": {"id": latest_id, "doi": latest_doi, "version": latest_version},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPO / "release" / "zenodo-deposit-receipt-v22-robustness.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"  receipt: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
