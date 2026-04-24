"""Continue osf_deposit_v6 from a pre-created node — use when the
initial POST timed out on the client side but the node was actually
created (OSF API is occasionally slow to return).

Idempotent where possible: file uploads will 409 if the file already
exists (we treat that as success).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CREDS = Path(r"C:\Users\heyzo\clawd\secrets\arxiv-creds.txt")

NODE_ID = "6syq4"
PARENT_ID = "wtkzg"

UPLOAD_PATHS = [
    ROOT / "papers" / "cognometry-v0.5.pdf",
    ROOT / "papers" / "drift_phase_transitions.md",
    ROOT / "papers" / "figures" / "drift_phase_transitions.png",
    ROOT / "benchmarks" / "drift_feature_scaling.json",
    ROOT / "benchmarks" / "drift_calibrated_v0.json",
    ROOT / "scripts" / "drift_feature_scaling.py",
    ROOT / "scripts" / "drift_calibrated_v0.py",
]


def get_osf_token():
    txt = CREDS.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[OSF\](.*?)(?:\n\[|\Z)", txt, re.DOTALL)
    section = m.group(1) if m else ""
    tm = re.search(r"api_token:\s*(\S+)", section)
    return tm.group(1) if tm else None


def main():
    token = get_osf_token()
    if not token:
        sys.exit("OSF token missing")

    auth = {"Authorization": f"Bearer {token}"}
    json_hdrs = {**auth, "Content-Type": "application/vnd.api+json"}

    print(f"[osf] continuing deposit on node {NODE_ID}")
    print(f"[osf] https://osf.io/{NODE_ID}/")
    print()

    # Link to parent wtkzg
    print(f"[link] {NODE_ID} -> {PARENT_ID} ...")
    link_payload = {
        "data": {
            "type": "node_links",
            "relationships": {
                "nodes": {"data": {"type": "nodes", "id": PARENT_ID}}
            },
        }
    }
    try:
        r = requests.post(
            f"https://api.osf.io/v2/nodes/{NODE_ID}/node_links/",
            headers=json_hdrs,
            json=link_payload,
            timeout=90,
        )
        print(f"  status: {r.status_code}")
        if r.status_code >= 400:
            print(f"  body:   {r.text[:400]}")
    except requests.exceptions.ReadTimeout:
        print("  TIMEOUT (may have succeeded server-side)")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
    print()

    # Upload each file
    print(f"[upload] {len(UPLOAD_PATHS)} files to {NODE_ID}/osfstorage/")
    for path in UPLOAD_PATHS:
        if not path.exists():
            print(f"  [skip] {path.name}: file not found")
            continue
        size_kb = path.stat().st_size / 1024
        url = (
            f"https://files.us.osf.io/v1/resources/{NODE_ID}/"
            f"providers/osfstorage/?kind=file&name={path.name}"
        )
        try:
            with open(path, "rb") as f:
                rr = requests.put(url, headers=auth, data=f, timeout=180)
            if rr.status_code < 300:
                print(f"  [ok]   {path.name} ({size_kb:.1f} KB)")
            elif rr.status_code == 409:
                print(f"  [dup]  {path.name} ({size_kb:.1f} KB) — already exists")
            else:
                print(f"  [FAIL] {path.name}: HTTP {rr.status_code}")
                print(f"         {rr.text[:300]}")
        except requests.exceptions.ReadTimeout:
            print(f"  [timeout] {path.name} ({size_kb:.1f} KB) — may have succeeded")
        except Exception as e:
            print(f"  [EXC]  {path.name}: {e}")

    print()
    print(f"[done] https://osf.io/{NODE_ID}/")


if __name__ == "__main__":
    main()
