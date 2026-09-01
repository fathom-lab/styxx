"""Deposit the capsule arc and the day two instruments failed, as a new version
in the styxx concept chain (10.5281/zenodo.19326174).

Prepared 2026-08-31. Needs exactly one thing the preparer does not have:

    ZENODO_TOKEN=... python scripts/zenodo_capsule_arc.py            # dry run
    ZENODO_TOKEN=... python scripts/zenodo_capsule_arc.py --publish  # for real

Everything else — file selection, metadata, the related-DOI chain, and the
integrity check that every named file exists and every certified paper carries
its certificate — runs without credentials and is verified before anything is
uploaded. The dry run prints exactly what would be deposited and exits.

Why a new version rather than the draft prepared earlier: that draft froze a
corpus of 200 certificates and predates the capsule arc, both preregistered
failures, the three corrections, and the collateral census. Publishing it would
make the weaker record the permanent one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "papers" / "closed-model-frontier"
ZENODO_API = "https://zenodo.org/api"
CONCEPT_RECID = "19326174"          # always-latest styxx concept DOI

# ── what goes in ────────────────────────────────────────────────────────────
# The arc, in the order it happened: what was built, what was measured, what
# failed, what was corrected, and what the failures licensed.
FILES = [
    # the invention
    P / "SPEC_oath_capsule_v01_2026_08_31.md",
    P / "SPEC_oath_capsule_v02_2026_08_31.md",
    P / "SPEC_worklog_v01_2026_08_31.md",
    # the capsules themselves — each verifies itself offline
    P / "RESULT_obligate2_split_verdict_2026_08_31.capsule.html",
    P / "RESULT_obligate1_does_not_ship_2026_08_31.capsule.html",
    P / "CORPUS_STATE_2026_08_31.capsule.html",
    P / "HANDOFF_capsule_v02_2026_08_31.capsule.html",
    P / "RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.capsule.html",
    P / "RESULT_collateral_census_2026_08_31.capsule.html",
    # the failures, preregistered before the data was touched
    P / "PREREG_external1_aidev_2026_08_31.md",
    P / "RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md",
    P / "PREREG_v13_repair_2026_08_31.md",
    # the corrections, published the same day
    P / "CORRECTION_external1_cause_2026_08_31.md",
    # the ground the next design stands on
    P / "PREREG_collateral_census_2026_08_31.md",
    P / "RESULT_collateral_census_2026_08_31.md",
    P / "collateral_census.json",
    P / "collateral_census_fidelity.json",
    # the split, frozen mid-stream before any number was known
    P / "SPLIT_external_corpus_2026_08_31.md",
    # the blind panel, so a stranger can re-adjudicate it
    P / "external1_packet.json",
    P / "external1_key_digest.txt",
    P / "external1_answers.json",
    P / "external1_adjudication.json",
]

DESCRIPTION = """<p><strong>The proof-carrying document, and the day two instruments failed in public.</strong></p>

<p>This deposit records a single day's work on styxx, an instrument stack for checking claims against committed evidence. It contains an invention and two failures, and the failures are the reason the invention is worth depositing.</p>

<p><strong>OATH Capsules.</strong> A capsule is one self-contained HTML file carrying a document's exact bytes, every receipt's exact bytes, and the certificate that binds them &mdash; with two layers of verification the reader runs. Layer one re-hashes every embedded byte in any browser, offline, with zero network requests, and paints each number with the epistemic band the certificate assigned it. Layer two re-runs the real verifier over the embedded bytes and compares the verdict, the counts and the full per-token ledger. Creation refuses to mint a capsule around a certificate that does not reproduce. Version 0.2 extends the format to agent work, sealing an agent's summary, the diff it describes, and the gate record &mdash; which is a pure function of those two byte streams. Several capsules are included and each one verifies itself.</p>

<p><strong>The corpus capsule</strong> seals the hash of every certificate this lab has stored, so that altering any historical result &mdash; softening a failure, inflating a count &mdash; is detectable from a single file. It was tested by forging a historical failure; the census named the forged certificate.</p>

<p><strong>EXTERNAL-1.</strong> The shipped claim gate was run over 71,016 agent-authored pull requests from a corpus this lab did not collect (AIDev, the MSR 2026 mining-challenge dataset). A precision floor of 0.95 was preregistered and the adjudication key sealed before any answer existed. A blind three-seat panel, which called 30 of 30 hidden decoys correctly, put the observed accusation precision at 0.23. The preregistered consequence was paid the same day: the accusing verdict for that claim class was disabled in shipped code, and four tests pinning real catches were marked as expected failures so the repair cannot land silently. A first repair recovered roughly a third of the false accusations against a two-thirds bar and also failed.</p>

<p><strong>Two corrections to our own diagnosis</strong> are included rather than quietly folded in: the leading defect named in the first write-up did not exist, and a material share of the failure was traced to the harness feeding the instrument a false account of the diff rather than to the instrument itself.</p>

<p><strong>The collateral census</strong> measures the ground any successor stands on: across 1,386,104 changed files, the share that is lock files, generated output, snapshots, migrations or whitespace-equivalent churn &mdash; the things nobody would ever describe &mdash; is under 11 percent. No claim is read and nothing is accused anywhere in that measurement, by construction.</p>

<p>Every artifact here is re-runnable from public data, and the preregistrations were frozen and pushed before the corresponding measurements ran. Failures are deposited under the same seal as successes.</p>
"""

METADATA = {
    "metadata": {
        "title": ("OATH Capsules: proof-carrying documents, and two preregistered "
                  "failures of an agent-claim verifier on 71,016 pull requests"),
        "upload_type": "publication",
        "publication_type": "report",
        "description": DESCRIPTION,
        "creators": [{"name": "Fathom Lab"}],
        "keywords": [
            "verification", "AI agents", "reproducibility", "preregistration",
            "proof-carrying documents", "provenance", "software engineering",
            "negative results", "mining software repositories",
        ],
        "license": "mit",
        "access_right": "open",
        "related_identifiers": [
            {"identifier": "10.5281/zenodo.19326174", "relation": "isNewVersionOf",
             "scheme": "doi"},
            {"identifier": "https://github.com/fathom-lab/styxx",
             "relation": "isSupplementTo", "scheme": "url"},
            {"identifier": "10.5281/zenodo.16919272", "relation": "references",
             "scheme": "doi"},
        ],
    }
}


def preflight() -> list[Path]:
    """Everything checkable without a credential, checked before anything else."""
    missing = [p for p in FILES if not p.exists()]
    if missing:
        print("REFUSED — named files do not exist:")
        for m in missing:
            print(f"  {m.relative_to(ROOT)}")
        raise SystemExit(1)

    # every capsule named here must still verify at the installed instrument
    sys.path.insert(0, str(ROOT))
    from styxx.capsule import verify_capsule
    bad = []
    for p in FILES:
        if p.name.endswith(".capsule.html"):
            rep = verify_capsule(p)
            if not rep.get("ok"):
                bad.append((p.name, rep.get("problems")))
    if bad:
        print("REFUSED — a capsule in this deposit does not verify:")
        for n, why in bad:
            print(f"  {n}: {why}")
        raise SystemExit(1)

    total = sum(p.stat().st_size for p in FILES)
    print(f"preflight OK — {len(FILES)} files, {total/1024:.0f} KB, "
          f"every capsule verifies at the installed instrument")
    for p in FILES:
        d = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        print(f"  {d}  {p.name}")
    return FILES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true",
                    help="actually create, upload and publish (default: dry run)")
    a = ap.parse_args()

    files = preflight()
    if not a.publish:
        print("\nDRY RUN — nothing was sent. Re-run with --publish and ZENODO_TOKEN set.")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: set ZENODO_TOKEN", file=sys.stderr)
        return 1
    auth = {"access_token": token}

    print(f"\n1/4 new version of concept {CONCEPT_RECID} ...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions/{CONCEPT_RECID}/actions/newversion",
        params=auth, timeout=60)
    if r.status_code >= 400:
        print(f"FAIL: {r.status_code} {r.text[:400]}")
        return 1
    latest = r.json()["links"]["latest_draft"]
    dep = requests.get(latest, params=auth, timeout=60).json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"  draft {dep_id}")

    # a new version inherits the previous version's files — clear them
    for f in dep.get("files", []):
        requests.delete(
            f"{ZENODO_API}/deposit/depositions/{dep_id}/files/{f['id']}",
            params=auth, timeout=60)
    print(f"  cleared {len(dep.get('files', []))} inherited file(s)")

    print("\n2/4 uploading ...")
    for p in files:
        with open(p, "rb") as fh:
            r = requests.put(f"{bucket}/{p.name}", data=fh, params=auth, timeout=300)
        if r.status_code >= 400:
            print(f"FAIL upload {p.name}: {r.status_code} {r.text[:300]}")
            return 1
        print(f"  {p.name}")

    print("\n3/4 metadata ...")
    r = requests.put(f"{ZENODO_API}/deposit/depositions/{dep_id}",
                     params=auth, json=METADATA, timeout=60)
    if r.status_code >= 400:
        print(f"FAIL metadata: {r.status_code} {r.text[:1000]}")
        return 1
    print("  attached")

    print("\n4/4 publishing ...")
    r = requests.post(f"{ZENODO_API}/deposit/depositions/{dep_id}/actions/publish",
                      params=auth, timeout=120)
    if r.status_code >= 400:
        print(f"FAIL publish: {r.status_code} {r.text[:600]}")
        return 1
    out = r.json()
    print(f"\nPUBLISHED  doi: {out.get('doi')}")
    print(f"           url: {out.get('links', {}).get('record_html')}")
    print(json.dumps({"doi": out.get("doi"), "id": out.get("id")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
