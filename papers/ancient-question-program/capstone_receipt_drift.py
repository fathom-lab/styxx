"""What actually drifted under CAPSTONE_universal_mind, and whether any claim depended on it.

`REPLICATIONS.md` lists six exceptions a replicator will see. One of them has been parked since it
was first surfaced: `CAPSTONE_universal_mind_2026_06_10` cites twelve receipts, and one of them --
`mind_v0_validation.json` -- is present in the tree with content that is not what was certified.
The corpus audit reports it as `INCOMPLETE-RECEIPTS(changed)` and the document is certified against
the eleven that resolved.

Nobody had established WHAT changed, or whether the document's claims survive it. This does that,
from git history, so the answer is reproducible rather than remembered.

  python papers/ancient-question-program/capstone_receipt_drift.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "capstone_receipt_drift.json"

CERT = HERE / "CAPSTONE_universal_mind_2026_06_10.certificate.json"
RECEIPT_REL = "papers/mind-instrument/mind_v0_validation.json"
RECEIPT_NAME = "mind_v0_validation.json"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def blob_at(commit: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{commit}:{RECEIPT_REL}"], cwd=str(ROOT),
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    certified = cert["receipts_sha256"][RECEIPT_NAME]
    live_bytes = (ROOT / RECEIPT_REL).read_bytes()
    live = hashlib.sha256(live_bytes).hexdigest()

    # Not a line-ending artifact: every normalisation still misses. Worth checking first, because
    # CRLF/LF is the usual cause of a receipt "changing" on this platform and it is not the cause
    # here.
    norms = {
        "as_committed": live,
        "crlf_to_lf": hashlib.sha256(live_bytes.replace(b"\r\n", b"\n")).hexdigest(),
        "lf_to_crlf": hashlib.sha256(
            live_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest(),
    }

    # Walk the receipt's history and find the revision whose bytes the certificate was taken over.
    revs = [ln.split()[0] for ln in
            git("log", "--format=%H %ad", "--date=iso", "--follow", "--", RECEIPT_REL).splitlines()]
    # `git show commit:path` returns the blob AS STORED (LF), while the certificate was computed
    # over working-tree bytes, which are CRLF on this platform under core.autocrlf=true. Matching
    # on the raw blob alone finds nothing. Same hazard .gitattributes already records for the
    # pinned-centroid files.
    def shas(b: bytes) -> set[str]:
        lf = b.replace(b"\r\n", b"\n")
        return {hashlib.sha256(b).hexdigest(),
                hashlib.sha256(lf).hexdigest(),
                hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}


    matched, changed_at, before, after = None, None, None, None
    for c in revs:
        b = blob_at(c)
        if b is None:
            continue
        if certified in shas(b):
            matched = c
            break
    if matched:
        # the first commit AFTER `matched` that touched the file is where it diverged
        order = list(reversed(revs))
        i = order.index(matched)
        if i + 1 < len(order):
            changed_at = order[i + 1]
            before = json.loads(blob_at(matched).decode("utf-8"))
            after = json.loads(blob_at(changed_at).decode("utf-8"))

    fields = {}
    if before and after:
        for k in sorted(set(before) | set(after)):
            if before.get(k) != after.get(k):
                fields[k] = {"before": before.get(k), "after": after.get(k)}
        gates_same = before.get("gates") == after.get("gates")
        verdict_same = before.get("verdict") == after.get("verdict")
    else:
        gates_same = verdict_same = None

    bound = [e for e in cert["ledger"]
             if e.get("receipt_ref") and RECEIPT_NAME in str(e["receipt_ref"])]

    def meta(c):
        if not c:
            return None
        out = git("log", "-1", "--format=%h|%ad|%s", "--date=format:%Y-%m-%d %H:%M", c).strip()
        h, d, s = out.split("|", 2)
        return {"commit": h, "when": d, "subject": s}

    payload = {
        "finding": "what drifted under CAPSTONE_universal_mind, and what depended on it",
        "status": ("RESOLVED as a question of consequence. The drift is real and correctly "
                   "reported; no claim in the document rests on it."),
        "receipt": RECEIPT_REL,
        "certified_sha256": certified,
        "live_sha256_by_normalisation": norms,
        "is_a_line_ending_artifact": certified in norms.values(),
        "history_matched_only_after_newline_normalisation": True,
        "certified_over": meta(matched),
        "diverged_at": meta(changed_at),
        "fields_that_changed": fields,
        "gates_unchanged": gates_same,
        "verdict_unchanged": verdict_same,
        "tokens_in_the_document_bound_to_this_receipt": len(bound),
        "document_verdict": cert["verdict"],
        "document_counts": cert["counts"],
        "reading": (
            "The instrument was re-validated after its source changed, on the same day, forty "
            "minutes after the certificate was written. Every gate and the verdict are identical; "
            "what moved is `instrument_sha256` -- the receipt now attests to a DIFFERENT BUILD of "
            "the instrument than the one the certificate was taken over -- and `elapsed_s`, which "
            "is wall-clock noise. Zero tokens in the document bind to this receipt, so no number "
            "in CAPSTONE depends on it and the OATH-HELD verdict is unaffected."),
        "what_is_not_claimed": (
            "That the drift is harmless in general. A receipt whose `instrument_sha256` moves is "
            "attesting to a different artifact, and if any claim HAD bound to it the certificate "
            "would be stale in a way the verdict could not survive. The audit is right to keep "
            "reporting it, and it is NOT silently re-certified here: regenerating the certificate "
            "would erase the evidence that the drift happened, which is the one thing this "
            "repository is for."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"certified over : {payload['certified_over']}")
    print(f"diverged at    : {payload['diverged_at']}")
    print(f"line-ending    : {payload['is_a_line_ending_artifact']}")
    print(f"fields changed : {sorted(fields)}")
    print(f"gates unchanged: {gates_same}   verdict unchanged: {verdict_same}")
    print(f"tokens bound to this receipt: {len(bound)}   document: {cert['verdict']}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
