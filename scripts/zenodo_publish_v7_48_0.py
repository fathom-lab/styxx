"""Publish the styxx 7.48.0 Zenodo draft, only after re-verifying it from Zenodo's side.

Run: py -3.12 zenodo_publish_v7_48_0.py --token-file PATH [--draft-id ID]
The draft id defaults to the one in zenodo-draft-receipt-v7.48.0.json.
Publishing mints a permanent DOI; the operator authorized it on 2026-09-29.
The token is sent only in an Authorization header and never printed or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zenodo_deposit_software_v7_48_0 import load_token  # noqa: E402  (same loader, [ZENODO] zenodo_token only)

API = "https://zenodo.org/api"
CONCEPT_RECID = "19758618"
EXPECT_VERSION = "7.48.0"
EXPECT_LINKS = {("isSupplementTo", "10.5281/zenodo.19746215"), ("isPartOf", "10.5281/zenodo.19326174")}
BANNED = ("first", "novel", "revolutionary", "groundbreaking", "breakthrough", "tamper-proof", "self-verifying",
          "hallucination detector")
UA = "fathom-lab-styxx-research/1 (+https://github.com/fathom-lab/styxx)"
TOKEN = ""


def redact(s: str) -> str:
    return s.replace(TOKEN, "<token>") if TOKEN else s


def call(method: str, path: str) -> dict:
    url = f"{API}{path}"
    r = requests.request(method, url, headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": UA}, timeout=120)
    if r.status_code >= 400:
        raise SystemExit(f"HTTP {r.status_code} on {method} {path}: {redact(r.text)[:800]}")
    return r.json() if r.text else {}


def local_md5(name: str) -> str:
    h = hashlib.md5()
    with open(HERE / "bundle" / name, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--draft-id")
    a = ap.parse_args()
    receipt_path = HERE / "zenodo-draft-receipt-v7.48.0.json"
    draft_id = a.draft_id
    if not draft_id:
        rec = json.loads(receipt_path.read_text(encoding="utf-8"))
        draft_id = str(rec.get("draft_id") or rec.get("deposition_id") or "")
        if rec.get("status") != "draft_ready_unpublished":
            print(f"REFUSED: draft receipt status is {rec.get('status')!r}, not draft_ready_unpublished")
            return 2
    if not draft_id:
        print("REFUSED: no draft id")
        return 2
    TOKEN, source = load_token(a.token_file)
    print(f"token: from {source} (value not shown); draft {draft_id}")

    d = call("GET", f"/deposit/depositions/{draft_id}")
    m = d.get("metadata") or {}
    checks = []
    checks.append(("unpublished", d.get("submitted") is False and d.get("state") in ("unsubmitted", "inprogress")))
    checks.append(("in the software concept chain 19758618", str(d.get("conceptrecid")) == CONCEPT_RECID))
    checks.append(("version 7.48.0", m.get("version") == EXPECT_VERSION))
    links = {(x.get("relation"), x.get("identifier")) for x in m.get("related_identifiers") or []}
    checks.append(("links to the spec and the research series", EXPECT_LINKS <= links))
    files = {f.get("filename"): f.get("checksum", "").replace("md5:", "") for f in d.get("files") or []}
    want = {n: local_md5(n) for n in ("styxx-v7.48.0-source-bundle.zip", "styxx-7.48.0-py3-none-any.whl",
                                      "styxx-7.48.0.tar.gz")}
    checks.append(("exactly the three files, md5 equal to the local bytes", files == want))
    text = ((m.get("description") or "") + " " + (m.get("notes") or "") + " " + (m.get("title") or "")).lower()
    checks.append(("no charter word in title, description or notes", not any(f" {b}" in f" {text}" for b in BANNED)))
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        print("REFUSED: a check failed; nothing was published")
        return 2

    pub = call("POST", f"/deposit/depositions/{draft_id}/actions/publish")
    doi = pub.get("doi") or (pub.get("metadata") or {}).get("doi")
    out = {
        "action": "software_deposit_published",
        "version": EXPECT_VERSION,
        "deposit_id": pub.get("id"),
        "software_doi": doi,
        "software_doi_url": f"https://doi.org/{doi}" if doi else None,
        "concept_recid": str(pub.get("conceptrecid")),
        "concept_doi": pub.get("conceptdoi"),
        "record_url": (pub.get("links") or {}).get("record_html") or (pub.get("links") or {}).get("html"),
        "files": {n: {"md5": want[n]} for n in want},
        "related_identifiers": sorted(links),
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (HERE / "zenodo-deposit-receipt-software-v7.48.0.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
