"""Edit the published threshold-law deposit to add related identifiers
linking it to the Fathom chain + styxx repo so it's not an orphan record.

Workflow on a published Zenodo record:
  POST .../actions/edit       -> open editable draft
  PUT  .../              json -> update metadata
  POST .../actions/publish    -> re-publish (DOI stays the same)
"""
import os, json, requests, copy

TOKEN = os.environ["ZENODO_TOKEN"]
DEP_ID = 20278945
BASE = "https://zenodo.org/api"

s = requests.Session()
s.params = {"access_token": TOKEN}

# Fetch current metadata
print("[1/5] Fetching current metadata...")
r = s.get(f"{BASE}/deposit/depositions/{DEP_ID}")
r.raise_for_status()
dep = r.json()
meta = copy.deepcopy(dep["metadata"])

# Open for editing
print("[2/5] Opening edit session...")
r = s.post(f"{BASE}/deposit/depositions/{DEP_ID}/actions/edit")
if r.status_code not in (200, 201):
    print("  edit-open status:", r.status_code, r.text[:300])
r.raise_for_status()

# Related identifiers: link to the Fathom DOI chain + styxx repo + GitHub commit
related = [
    # styxx main Zenodo — direct predecessor (this deposit supplements that tool line)
    {"identifier": "10.5281/zenodo.20130041", "relation": "isSupplementTo",
     "resource_type": "publication-workingpaper",
     "scheme": "doi"},
    # Fathom methodology references (lineage, not continuation)
    {"identifier": "10.5281/zenodo.19777921", "relation": "references",
     "resource_type": "publication-workingpaper",
     "scheme": "doi"},
    {"identifier": "10.5281/zenodo.19758619", "relation": "references",
     "resource_type": "publication-workingpaper",
     "scheme": "doi"},
    {"identifier": "10.5281/zenodo.19502716", "relation": "references",
     "resource_type": "publication-workingpaper",
     "scheme": "doi"},
    # styxx tool + commit pin
    {"identifier": "https://github.com/fathom-lab/styxx",
     "relation": "isSupplementTo",
     "resource_type": "software",
     "scheme": "url"},
    {"identifier": "https://github.com/fathom-lab/styxx/tree/58a1d98",
     "relation": "isDerivedFrom",
     "resource_type": "software",
     "scheme": "url"},
    {"identifier": "https://pypi.org/project/styxx/7.4.1/",
     "relation": "isSupplementTo",
     "resource_type": "software",
     "scheme": "url"},
]

meta["related_identifiers"] = related

# Also strengthen description footer with the lineage
extra_lineage = (
    "<hr><p><strong>Lineage:</strong> this deposit supplements "
    "<a href=\"https://doi.org/10.5281/zenodo.20130041\">Fathom v23 / styxx v7.2.0</a> "
    "and is methodologically downstream of the Fathom Cognometric series "
    "(<a href=\"https://doi.org/10.5281/zenodo.19777921\">Every Mind Leaves Vitals</a>, "
    "<a href=\"https://doi.org/10.5281/zenodo.19758619\">styxx v6.2.0 ref impl</a>, "
    "<a href=\"https://doi.org/10.5281/zenodo.19502716\">Fathom Cognitive Atlas v0.3</a>). "
    "It is a supplement to the <code>styxx</code> tool "
    "(<a href=\"https://github.com/fathom-lab/styxx\">repo</a>, "
    "<a href=\"https://pypi.org/project/styxx/7.4.1/\">PyPI 7.4.1</a>, "
    "commit <code>58a1d98</code>). It is not a continuation of the depth/"
    "geometry line; it is a narrower empirical finding about label-free "
    "cognometric transport, audited by the same research-integrity protocol.</p>"
)
# strip any prior (wrong) lineage block before reapplying
import re as _re
meta["description"] = _re.sub(
    r"<hr><p><strong>Lineage:</strong>.*?</p>", "",
    meta.get("description", ""), flags=_re.DOTALL
)
meta["description"] = meta["description"] + extra_lineage

print("[3/5] PUTting updated metadata...")
r = s.put(f"{BASE}/deposit/depositions/{DEP_ID}",
          json={"metadata": meta})
if r.status_code >= 400:
    print("PUT ERROR:", r.status_code, r.text[:1500])
r.raise_for_status()
print("  metadata accepted")

print("[4/5] Re-publishing...")
r = s.post(f"{BASE}/deposit/depositions/{DEP_ID}/actions/publish")
if r.status_code >= 400:
    print("PUBLISH ERROR:", r.status_code, r.text[:500])
r.raise_for_status()
pub = r.json()
print(f"  DOI: {pub.get('doi')}")
print(f"  URL: {pub['links'].get('record_html') or pub['links'].get('html')}")

print("[5/5] Verifying related_identifiers are live...")
r = requests.get(f"{BASE}/records/{DEP_ID}", timeout=30)
rec = r.json()
print(json.dumps(rec.get("metadata", {}).get("related_identifiers", []),
                 indent=2))
