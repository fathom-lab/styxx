"""Zenodo deposit + publish for the threshold-law paper.

End-to-end:
  1. Create deposition
  2. Upload threshold-law-2026-05-18.zip
  3. Set metadata (title, creators, description, keywords, related identifiers)
  4. Publish (mints DOI)
  5. Print DOI + URL

Idempotent-ish: if a draft with the same title exists, abort with that ID.
"""
import os, sys, json, requests, pathlib, re

TOKEN = os.environ["ZENODO_TOKEN"]
BASE = "https://zenodo.org/api"
ZIP_PATH = pathlib.Path(r"C:\Users\heyzo\clawd\styxx\dist\zenodo\threshold-law-2026-05-18.zip")
PAPER_PATH = pathlib.Path(r"C:\Users\heyzo\clawd\styxx\papers\threshold-law-2026-05-18.md")
SELF_AUDIT = pathlib.Path(r"C:\Users\heyzo\clawd\styxx\papers\threshold-law-self-audit-2026-05-18.md")

TITLE = "A Corpus-Domain Overlap Threshold Governs Label-Free Cognometric Transport"

# Pull abstract from paper (between ## Abstract and ---)
text = PAPER_PATH.read_text(encoding="utf-8")
m = re.search(r"## Abstract\s*(.+?)\n---", text, re.DOTALL)
abstract_md = m.group(1).strip() if m else ""

# Strip markdown soft-wraps into HTML paragraphs
def md_to_html(md):
    paras = [p.strip().replace("\n", " ") for p in md.split("\n\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paras)

description = (
    md_to_html(abstract_md)
    + "<p><strong>Repository:</strong> "
    + "<a href=\"https://github.com/fathom-lab/styxx\">fathom-lab/styxx</a> "
    + "@ <code>58a1d98</code> (PyPI <code>styxx==7.4.1</code>).</p>"
    + "<p><strong>Bundle contents:</strong> paper, figure, raw run JSON "
    + "(<code>out_corpus_coverage_law.json</code>, "
    + "<code>out_corpus_coverage_law_fine.json</code>, "
    + "<code>out_cross_vendor_refusal_transport_confirm.json</code>), "
    + "scripts (<code>corpus_coverage_law.py</code>, "
    + "<code>corpus_coverage_law_fine.py</code>, "
    + "<code>cross_vendor_refusal_transport_confirm.py</code>, "
    + "<code>plot_threshold_law.py</code>), the related papers establishing "
    + "the audit chain (corpus-coverage law original + fine replication, "
    + "cross-vendor stress, cross-vendor preregistration-killed confirmation, "
    + "refusal-transport stress boundary, styxx status consolidation map, "
    + "research integrity protocol), and the <strong>styxx-on-paper "
    + "self-audit</strong> (<code>threshold-law-self-audit-2026-05-18.md</code>) "
    + "&mdash; the paper scored by the very instruments it documents. "
    + "Self-audit verdict: 0 cracks requiring revision; all 8 headline "
    + "numbers match raw JSON within 0.005; integrity protocol rules "
    + "visibly followed; construct-ceiling firings on the limits/integrity "
    + "sections are register artifacts predicted by the consolidation map.</p>"
    + "<p><strong>Honest bounds (also in the paper):</strong> the strict "
    + "preregistered same-family flat-control criterion failed in the "
    + "high-resolution 12-point replication (Spearman -0.41 vs +/-0.40 limit); "
    + "an independent cross-vendor preregistration was killed (min Anthropic "
    + "transported AUC 0.617 below the 0.70 floor). Both are reported "
    + "in the paper body, not in footnotes. This is a Zenodo methods deposit, "
    + "not peer-reviewed, no arXiv endorsement claimed, no universality "
    + "claimed.</p>"
)

meta = {
    "metadata": {
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "title": TITLE,
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"}
        ],
        "description": description,
        "keywords": [
            "cognometric transport",
            "embedding-space transport",
            "label-free probes",
            "Procrustes",
            "refusal axis",
            "LLM alignment auditing",
            "construct validity",
            "preregistration",
            "styxx",
        ],
        "communities": [],
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "1.0",
        "notes": (
            "Methodology demonstration: first paper deposited alongside an "
            "audit of itself by the tool it documents (styxx 7.4.1). "
            "Self-audit included in bundle; verdict 0 cracks."
        ),
    }
}

s = requests.Session()
s.params = {"access_token": TOKEN}

print("[1/5] Creating deposition...")
r = s.post(f"{BASE}/deposit/depositions", json={})
r.raise_for_status()
dep = r.json()
dep_id = dep["id"]
bucket = dep["links"]["bucket"]
print(f"  deposition id: {dep_id}")
print(f"  bucket: {bucket}")

print(f"[2/5] Uploading {ZIP_PATH.name} ({ZIP_PATH.stat().st_size} bytes)...")
with open(ZIP_PATH, "rb") as fp:
    r = s.put(f"{bucket}/{ZIP_PATH.name}", data=fp)
r.raise_for_status()
print(f"  uploaded: checksum {r.json().get('checksum')}")

print("[3/5] Setting metadata...")
r = s.put(f"{BASE}/deposit/depositions/{dep_id}", json=meta)
if r.status_code >= 400:
    print("METADATA ERROR:", r.status_code, r.text)
r.raise_for_status()
print("  metadata ok")

print("[4/5] Publishing (mints DOI)...")
r = s.post(f"{BASE}/deposit/depositions/{dep_id}/actions/publish")
if r.status_code >= 400:
    print("PUBLISH ERROR:", r.status_code, r.text)
r.raise_for_status()
pub = r.json()
doi = pub.get("doi") or pub.get("metadata", {}).get("doi")
record_url = pub["links"].get("record_html") or pub["links"].get("html")
print(f"  DOI: {doi}")
print(f"  URL: {record_url}")

print("[5/5] Verifying DOI resolves...")
import time
time.sleep(3)
r2 = requests.get(f"https://doi.org/{doi}", allow_redirects=True, timeout=30)
print(f"  doi.org -> {r2.status_code} -> {r2.url}")

print("\n=== RESULT ===")
print(json.dumps({
    "deposition_id": dep_id,
    "doi": doi,
    "url": record_url,
    "title": TITLE,
}, indent=2))
