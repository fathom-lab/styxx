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
titled `challenge: <document>`.

The verifier build is two fields of the receipt, and both sit inside the digested core: the hash
of sworn.py (`verifier.sworn_sha256`) and the version string the issuing build carried
(`verifier.styxx_version`).
`same_build` is true only when BOTH match: the same sworn.py AND the same styxx version. A receipt
issued by 7.47.0 and one re-derived by 7.48.0 are not from the same verifier build even when
sworn.py did not move, because their digests differ by construction. A lab receipt that carries
no `verifier.styxx_version`, or a null one, matches no build. papers/plates/SAND_CHECK.md pays for
a record with `agree: false` and `same_build: true`; that shape therefore requires the same
sworn.py AND the same styxx version, and a difference in the version alone can never produce it.
Both fields are read from the lab's receipt only when that receipt re-issues to the digest it
states; a receipt edited after issue vouches for nothing, so it matches no build either.
A CHALLENGE with `same_build: false` is, before anything else, an instruction: check out the
commit the receipt names and run again with the styxx that commit carries.

Version skew. When the digests differ, both receipts are re-issued through
`styxx.sworn.issue_receipt`, the receipt module's own canonicalisation, with the version set
aside. If they then agree, the lab's receipt re-issues to its own digest, and both receipts name a
styxx version and the two differ, the record says `version_skew: true` and `same_build: false`,
names both versions, and `why` says that the verdict and every span agree. It stays a CHALLENGE
(`agree: false`, exit 3): `agree` is digest and verdict, the lab's receipt was not reproduced byte
for byte, and SAND_CHECK calls a receipt reproduced when the two digests are equal. A lab receipt
with no version is never called version skew: `why` says it carries no `verifier.styxx_version`.

What `why` names. When the digests still differ with the version set aside, `why` compares the
digested parts separately and names each part that differs: the spans and the counts taken from
them ("a span-level difference", said only then), the document bytes, the commit, the manifest
digest, or any other digested field by its key. A lab receipt whose own fields do not re-issue to
its digest is named as that (`lab_digest_reissues: false`), never as version skew or a span-level
difference. When the digest reproduces all the same (the lab's receipt body, or its digest, was
edited after issue), `agree` and the exit code stand, because the digest is what SAND_CHECK
compares, and `why` and the CLI line say that the body does not re-issue to it.
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


def _reissued_digest(receipt: dict, *, without_version: bool = False) -> str | None:
    """The digest `styxx.sworn.issue_receipt` gives this receipt's own fields. That is the receipt
    module's canonicalisation (JCS of the core without digest, timestamp and coverage), not a
    second one written here. With `without_version`, `verifier.styxx_version` is set aside before
    the re-issue. None for a receipt that is not a /v1 verdict receipt (a /v0 receipt digested
    other fields, so nothing here may call it version skew), or whose fields cannot be
    canonicalised."""
    from styxx import sworn
    if not isinstance(receipt, dict) or receipt.get("schema") != sworn.RECEIPT_SCHEMA:
        return None
    # coverage sits outside the digest (issue_receipt drops it); leaving it out here only spares
    # issue_receipt hashing it beside, which this comparison never reads
    core = {k: v for k, v in receipt.items() if k != "coverage"}
    if without_version:
        ver = core.get("verifier")
        if not isinstance(ver, dict):
            return None
        core["verifier"] = {k: v for k, v in ver.items() if k != "styxx_version"}
    try:
        return sworn.issue_receipt(core, timestamp="-")["digest"]
    except (TypeError, ValueError):
        return None


def _short(value) -> str:
    """A value short enough for a `why` line, in ASCII so any console can print it."""
    s = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return s if len(s) <= 16 else s[:12] + "..."


def _differing_parts(lab: dict, mine: dict) -> list:
    """The digested parts of two receipts that differ, each as a phrase for `why`, with
    `verifier.styxx_version` set aside (the caller reports the version on its own). The fields
    compared are the ones `styxx.sworn.issue_receipt` digests. The spans and the fields counted
    from them are one part, "a span-level difference"; the document, the commit and the manifest
    digest are each named as themselves; any other digested field is named by its key."""
    from styxx import sworn

    def core(r: dict) -> dict:
        c = {k: v for k, v in r.items() if k not in sworn._RECEIPT_OUTSIDE_DIGEST}
        if isinstance(c.get("verifier"), dict):
            c["verifier"] = {k: v for k, v in c["verifier"].items() if k != "styxx_version"}
        return c

    def keys_that_differ(x, y) -> list:
        if isinstance(x, dict) and isinstance(y, dict):
            return sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
        return []

    a, b = core(lab), core(mine)
    parts = []
    span_bits = []
    xs, ys = a.get("spans"), b.get("spans")
    if xs != ys:
        if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys):
            moved = [i for i, (x, y) in enumerate(zip(xs, ys)) if x != y]
            keys = keys_that_differ(xs[moved[0]], ys[moved[0]])
            span_bits.append(f"{len(moved)} of {len(xs)} spans differ, the lowest at index {moved[0]}"
                             + (f" in {', '.join(keys)}" if keys else ""))
        else:
            span_bits.append(f"the lab's receipt has {len(xs) if isinstance(xs, list) else _short(xs)} spans "
                             f"and yours {len(ys) if isinstance(ys, list) else _short(ys)}")
    counted = [k for k in ("counts", "sworn_total", "unresolved", "rungs") if a.get(k) != b.get(k)]
    if counted:
        span_bits.append("the fields counted from the spans differ: " + ", ".join(counted))
    if span_bits:
        parts.append("a span-level difference (" + "; ".join(span_bits) + ")")
    if a.get("document") != b.get("document"):
        x, y = a.get("document"), b.get("document")
        keys = keys_that_differ(x, y)
        detail = (", ".join(f"{k} lab {_short(x.get(k))} yours {_short(y.get(k))}" for k in keys) if keys
                  else f"lab {_short(x)} yours {_short(y)}")
        parts.append(("the document bytes differ" if "inline_sha256" in keys else "the document entry differs")
                     + f" ({detail})")
    if a.get("commit") != b.get("commit"):
        parts.append(f"the commit differs (lab {_short(a.get('commit'))} yours {_short(b.get('commit'))})")
    if a.get("manifest_digest") != b.get("manifest_digest"):
        parts.append(f"the manifest digest differs (lab {_short(a.get('manifest_digest'))} "
                     f"yours {_short(b.get('manifest_digest'))})")
    if a.get("verifier") != b.get("verifier"):
        keys = keys_that_differ(a.get("verifier"), b.get("verifier"))
        parts.append("the verifier block differs with the version set aside"
                     + (f" ({', '.join(keys)})" if keys else ""))
    named = {"spans", "counts", "sworn_total", "unresolved", "rungs", "document", "commit", "manifest_digest",
             "verifier"}
    other = sorted(k for k in set(a) | set(b) if k not in named and a.get(k) != b.get(k))
    if other:
        parts.append("the digested field" + ("s " if len(other) > 1 else " ") + ", ".join(other)
                     + (" differ" if len(other) > 1 else " differs"))
    return parts


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
    lab_version = (lab.get("verifier") or {}).get("styxx_version")
    my_version = (mine.get("verifier") or {}).get("styxx_version")
    # the build is sworn.py AND the styxx version: both are inside the digested core, so a receipt
    # from another release differs in its digest by construction, and a version difference alone
    # must never read as the shape SAND_CHECK pays for (agree false with same_build true). A lab
    # receipt with no version string matches no build.
    same_sworn = lab_build is not None and lab_build == my_build
    lab_names_version = isinstance(lab_version, str)
    agree_digest = lab.get("digest") == mine.get("digest")
    agree_verdict = lab.get("document_verdict") == mine.get("document_verdict")
    # the version is inside the digested core: re-issue both with it set aside, through sworn's own
    # issue_receipt, and call the difference version skew only if nothing else differs, the lab's
    # receipt re-issues to the digest it states (a receipt edited after issue is not version skew),
    # and both receipts name a version (a receipt that names none is not skew from anything)
    lab_reissued = _reissued_digest(lab)
    lab_digest_reissues = None if lab_reissued is None else lab_reissued == lab.get("digest")
    # the build fields count only when the lab's receipt re-issues to the digest it states: a receipt
    # edited after issue could carry any sworn_sha256 and any version, so it matches no build
    same_build = (same_sworn and lab_names_version and lab_version == my_version
                  and lab_digest_reissues is True)
    lab_bare = _reissued_digest(lab, without_version=True)
    agree_without_version = lab_bare is not None and lab_bare == _reissued_digest(mine, without_version=True)
    version_skew = (not agree_digest and lab_digest_reissues is True and agree_without_version
                    and lab_names_version and isinstance(my_version, str) and lab_version != my_version)
    # absent and null read the same; anything else that is not a string is named, not called absent
    lab_carries = ("carries no verifier.styxx_version" if lab_version is None else
                   f"carries no verifier.styxx_version string (it carries {_short(lab_version)})")
    if not lab_names_version:
        version_note = f" (the lab's receipt {lab_carries}; yours carries styxx {my_version})"
    elif lab_version != my_version:
        version_note = f" (the styxx versions differ as well, lab {lab_version} and yours {my_version})"
    else:
        version_note = ""
    if agree_digest and agree_verdict:
        why = ""
        if lab_digest_reissues is False:
            why = ("the digest reproduced, but the lab's receipt body does not re-issue to it: the body or the digest "
                   "was edited after issue, so what replicated is the digest, not the receipt as it stands; compare "
                   "the two receipts")
    elif not same_sworn:
        why = ("the verifier build differs, so the digest differs by construction; check out the commit the "
               "receipt names and run again before calling this a challenge")
    elif not agree_verdict:
        why = (f"the verdict differs: lab {lab.get('document_verdict')}, mine {mine.get('document_verdict')}"
               + version_note)
    elif lab_digest_reissues is False:
        why = ("the lab's receipt does not re-issue to its own digest: its fields or its digest were edited after "
               "issue, so comparing the two digests says nothing about the spans; compare the two receipts")
    elif version_skew:
        why = (f"version skew, not a disagreement: the lab's receipt was issued by styxx {lab_version} and "
               f"yours by styxx {my_version}, with the same sworn.py; the verdict and every span agree, and the two "
               "digests agree once verifier.styxx_version is set aside. The version is inside the digested core, so "
               "the digest differs by construction and this is not the same verifier build (same_build is false); "
               "check out the commit the receipt names, run again with the styxx that commit carries, and only then "
               "call it a challenge")
    elif not lab_names_version and agree_without_version:
        why = (f"the lab's receipt {lab_carries}, and yours carries styxx {my_version}. The verdict "
               "and every span agree once the version is set aside, but this is not version skew: styxx.sworn stamps "
               "the version on every receipt it issues, inside the digested core, so a receipt with none names no "
               "release to re-run with and was not issued as it stands; compare the two receipts and ask the lab "
               "which styxx issued it")
    else:
        parts = _differing_parts(lab, mine)
        why = ("the verdict agrees but the digest differs: " + "; ".join(parts) if parts else
               "the verdict agrees but the digest differs, and no digested field differs as compared here")
        why += "; compare the two receipts"
        if version_note:
            why += version_note[:-1] + (", and the digests still differ with the version set aside)"
                                        if lab_bare is not None and lab_names_version else ")")
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
        "lab_styxx_version": lab_version, "my_styxx_version": my_version,
        "lab_digest_reissues": lab_digest_reissues, "agree_without_version": agree_without_version,
        "version_skew": version_skew,
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
          f"same_build={rec['same_build']} version_skew={rec['version_skew']} "
          f"lab_digest_reissues={rec['lab_digest_reissues']} lab={str(rec['lab_digest'])[:12]} mine={str(rec['my_digest'])[:12]} "
          f"verdict lab={rec['lab_verdict']} mine={rec['my_verdict']} -> {a.out} (+ {os.path.basename(mine_path)})")
    if rec["why"]:
        print("  why: " + rec["why"])
    return 0 if rec["agree"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
