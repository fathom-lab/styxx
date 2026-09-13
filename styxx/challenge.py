# -*- coding: utf-8 -*-
"""styxx.challenge — the bounty's object.

    python -m styxx.challenge <sworn document> <the lab's committed receipt> --repo . [--out challenge.json]

Re-derives the receipt of a sworn document at the commit the lab's receipt names and compares:
same digest, same verdict, same verifier build. The result is a challenge record — a small json
that says, without prose, whether a stranger reproduced the lab's receipt.

What a record is, stated plainly: a self-report. `record_sha256` names the record's own bytes and
nothing else; whoever edits the record recomputes it in one line. Nothing signs it. A record is
therefore a claim the lab settles by re-running the stranger's stated steps, never evidence on its
own — which is why the record carries what a re-run needs: the commit, the document name the
receipt names, the hash of the stranger's own receipt (written beside the record by the CLI), the
counts, and both verifier builds.

Three refusals, each because the record it would otherwise produce is exactly the shape the bounty
pays, and would be wrong:

- the commit the lab's receipt names is not in `--repo` (a shallow clone, a zip): every span would
  read UNRESOLVED commit_absent while the verdict still printed SWORN-HELD — reproduced 2026-09-13;
- the document's basename is not the one the receipt names: sworn digests the basename, so a
  renamed copy disagrees on the digest with every span HELD;
- the stranger's own receipt has UNRESOLVED spans or no HELD span: nothing was checked, so nothing
  was reproduced or challenged.

Verdicts: REPLICATION (`agree`) — a line for REPLICATIONS.md. CHALLENGE — the verdicts or the
digests disagree, and `why` says which; attach the record and the receipt beside it to an issue
titled `challenge: <document>`. When the verifier build differs, the digest differs by
construction (the verifier's own hash is inside the digested core), so a CHALLENGE with
`same_build: false` is first of all an instruction: check out the commit the receipt names and
run again.
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


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True)


def _has_commit(repo: str, commit: str) -> bool:
    try:
        return _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0
    except Exception:  # pragma: no cover - git missing
        return False


def _blob_sha256(repo: str, commit: str, doc: str) -> str | None:
    """sha256 of the document's bytes AT THE COMMIT (LF as stored), beside the working-copy hash,
    because a Windows checkout may hold CRLF for a file git stores as LF."""
    try:
        rel = os.path.relpath(os.path.abspath(doc), os.path.abspath(repo)).replace(os.sep, "/")
        p = _git(repo, "show", f"{commit}:{rel}")
        return hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None
    except Exception:  # pragma: no cover
        return None


def run(doc: str, lab_receipt: str, repo: str = ".", out: str | None = None) -> dict:
    """Re-derive and compare. `out` is where the stranger's own receipt is written (a temp file if
    omitted). Refuses, with the reason, in the three cases the docstring names."""
    lab = json.load(open(lab_receipt, encoding="utf-8"))
    commit = lab.get("commit")
    if not commit:
        raise SystemExit("REFUSED: the lab's receipt names no commit; nothing to re-derive against")
    lab_document = (lab.get("document") or {}).get("name")
    if lab_document and os.path.basename(doc) != lab_document:
        raise SystemExit(f"REFUSED: the receipt names document {lab_document!r}; you gave {os.path.basename(doc)!r}. "
                         "sworn digests the basename, so a renamed copy disagrees for no reason. Keep the name.")
    if not _has_commit(repo, commit):
        raise SystemExit(f"REFUSED: commit {commit} is not in {os.path.abspath(repo)}. A shallow clone or a zip cannot "
                         "re-derive a receipt: every span would read UNRESOLVED (commit_absent) and the record would mean "
                         "nothing. Clone with full history and check out that commit.")
    mine_path = out or os.path.join(tempfile.mkdtemp(), "mine.sworn-receipt.json")
    cmd = [sys.executable, "-m", "styxx.sworn", "verify", doc, "--repo", repo, "--commit", commit, "--out", mine_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not os.path.exists(mine_path):
        raise SystemExit(f"verify did not write a receipt:\n{proc.stdout}\n{proc.stderr}")
    mine = json.load(open(mine_path, encoding="utf-8"))
    counts = mine.get("counts") or {}
    held = int(counts.get("HELD", 0) or 0)
    unresolved = int(counts.get("UNRESOLVED", 0) or 0)
    if unresolved or not held:
        raise SystemExit(f"REFUSED: the re-derived receipt has HELD={held} UNRESOLVED={unresolved}; nothing was checked, "
                         f"so there is nothing to replicate or challenge. The receipt at {mine_path} says why, span by span.")
    lab_build = (lab.get("verifier") or {}).get("sworn_sha256") or (lab.get("verifier") or {}).get("digest")
    my_build = (mine.get("verifier") or {}).get("sworn_sha256") or (mine.get("verifier") or {}).get("digest")
    same_build = lab_build == my_build and lab_build is not None
    agree_digest = lab.get("digest") == mine.get("digest")
    agree_verdict = lab.get("document_verdict") == mine.get("document_verdict")
    if agree_digest and agree_verdict:
        why = ""
    elif not same_build:
        why = ("the verifier build differs, so the digest differs by construction; check out the commit the "
               "receipt names and run again before calling this a challenge")
    elif not agree_verdict:
        why = f"the verdict differs: lab {lab.get('document_verdict')}, mine {mine.get('document_verdict')}"
    else:
        why = "the verdict agrees but the digest differs: a span-level difference; compare the two receipts"
    record = {
        "schema": "styxx.challenge/v1",
        "document": os.path.basename(doc),
        "document_sha256_working_copy": _sha(doc),
        "document_sha256_at_commit": _blob_sha256(repo, commit, doc),
        "lab_document": lab_document,
        "commit": commit,
        "lab_receipt_sha256": _sha(lab_receipt), "lab_digest": lab.get("digest"), "lab_verdict": lab.get("document_verdict"),
        "mine_receipt_sha256": _sha(mine_path), "my_digest": mine.get("digest"), "my_verdict": mine.get("document_verdict"),
        "my_counts": counts,
        "lab_build": lab_build, "my_build": my_build, "same_build": same_build,
        "agree_digest": agree_digest, "agree_verdict": agree_verdict,
        "agree": agree_digest and agree_verdict,
        "why": why,
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record["record_sha256"] = record_sha256(record)
    return record


def record_sha256(record: dict) -> str:
    """The record's own hash over its fields except `when` and itself. It names these bytes and
    proves nothing about them: anyone who edits the record recomputes it."""
    blob = json.dumps({k: v for k, v in record.items() if k not in ("when", "record_sha256")},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.challenge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc"); ap.add_argument("lab_receipt"); ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="challenge.json")
    a = ap.parse_args(argv)
    stem = a.out[:-5] if a.out.endswith(".json") else a.out
    mine_path = stem + ".mine.sworn-receipt.json"
    rec = run(a.doc, a.lab_receipt, a.repo, out=mine_path)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    kind = "REPLICATION" if rec["agree"] else "CHALLENGE"
    print(f"{kind}  agree={rec['agree']} (digest {rec['agree_digest']}, verdict {rec['agree_verdict']}) "
          f"same_build={rec['same_build']} lab={str(rec['lab_digest'])[:12]} mine={str(rec['my_digest'])[:12]} "
          f"verdict lab={rec['lab_verdict']} mine={rec['my_verdict']} -> {a.out} (+ {os.path.basename(mine_path)})")
    if rec["why"]:
        print("  why: " + rec["why"])
    return 0 if rec["agree"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
