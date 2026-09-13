# -*- coding: utf-8 -*-
"""styxx.challenge — the bounty's object.

    python -m styxx.challenge <sworn document> <the lab's committed receipt> --repo . [--out challenge.json]

Re-derives the receipt of a sworn document at the commit the lab's receipt names and compares:
same digest, same verdict, same verifier build. The result is a challenge record — a small json
that says, without prose, whether a stranger reproduced the lab's receipt. A record with
`agree: false` is a bounty claim; attach it to the issue. A record with `agree: true` is a
replication; it belongs in REPLICATIONS.md. The record hashes both receipts so it cannot be
edited to say something else after the fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(doc: str, lab_receipt: str, repo: str = ".", out: str | None = None) -> dict:
    lab = json.load(open(lab_receipt, encoding="utf-8"))
    commit = lab.get("commit")
    if not commit:
        raise SystemExit("the lab's receipt names no commit; nothing to re-derive against")
    mine_path = out or os.path.join(tempfile.mkdtemp(), "mine.sworn-receipt.json")
    cmd = [sys.executable, "-m", "styxx.sworn", "verify", doc, "--repo", repo, "--commit", commit, "--out", mine_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not os.path.exists(mine_path):
        raise SystemExit(f"verify did not write a receipt:\n{proc.stdout}\n{proc.stderr}")
    mine = json.load(open(mine_path, encoding="utf-8"))
    lab_build = (lab.get("verifier") or {}).get("sworn_sha256") or (lab.get("verifier") or {}).get("digest")
    my_build = (mine.get("verifier") or {}).get("sworn_sha256") or (mine.get("verifier") or {}).get("digest")
    record = {
        "schema": "styxx.challenge/v0",
        "document": os.path.basename(doc), "document_sha256": _sha(doc), "commit": commit,
        "lab_receipt_sha256": _sha(lab_receipt), "lab_digest": lab.get("digest"), "lab_verdict": lab.get("document_verdict"),
        "my_digest": mine.get("digest"), "my_verdict": mine.get("document_verdict"), "my_counts": mine.get("counts"),
        "same_build": lab_build == my_build and lab_build is not None,
        "agree": lab.get("digest") == mine.get("digest") and lab.get("document_verdict") == mine.get("document_verdict"),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    blob = json.dumps({k: v for k, v in record.items() if k != "when"}, sort_keys=True, separators=(",", ":")).encode()
    record["record_sha256"] = hashlib.sha256(blob).hexdigest()
    return record


def main(argv=None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="styxx.challenge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc"); ap.add_argument("lab_receipt"); ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="challenge.json")
    a = ap.parse_args(argv)
    rec = run(a.doc, a.lab_receipt, a.repo)
    json.dump(rec, open(a.out, "w"), indent=1)
    kind = "REPLICATION" if rec["agree"] else "CHALLENGE"
    print(f"{kind}  agree={rec['agree']} same_build={rec['same_build']} lab={str(rec['lab_digest'])[:12]} "
          f"mine={str(rec['my_digest'])[:12]} verdict lab={rec['lab_verdict']} mine={rec['my_verdict']} -> {a.out}")
    return 0 if rec["agree"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
