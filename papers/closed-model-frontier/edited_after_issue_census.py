"""edited_after_issue_census.py — the ten documents that moved under their certificates, made legible;
and the receipt-binding census reconciled with Charon's ferry log, line by line.

Reads (never writes) ``receipt_binding_census_result.json`` (the committed census that
``RESULT_oath_receipt_binding_2026_09_05.md`` swears to) and ``papers/charon/charon.log.jsonl``
(the committed log that ``RESULT_charon_v01_ships`` swears to). For every certificate whose
document cell is ``at_issue`` — the working document no longer hashes to ``document_sha256``, and
the sworn document was recovered from the issuing commit — it diffs the sworn document against the
working one and asks the only two questions a reader needs answered: did the edit remove a line
the certificate had a ledger row on (a VERIFIED, ABSTAIN or UNGROUNDED token), and did it add
lines carrying numbers the certificate never examined. Then it lays the census beside Charon: for
every OATH line in the log, what Charon said at ingest (reproduced true / false / null) against
what the census says now (stands over sworn bytes true / false / null), and names every
disagreement with its reason.

Its one output is ``edited_after_issue_census_result.json``; it refuses to overwrite a tracked
one. Nothing here is a verdict. A line removed from a document is a fact about bytes, not about
truth; the live corpus audit already re-certifies the working document, and this census says what
the PUBLISHED certificate beside it no longer describes.

    python papers/closed-model-frontier/edited_after_issue_census.py [--out PATH]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx import receipt_binding as rb              # noqa: E402
from styxx.certify import certify_doc, extract_numbers   # noqa: E402
from styxx.corpus_audit import _resolve_receipts, verdict_class   # noqa: E402

CENSUS = HERE / "receipt_binding_census_result.json"
CHARON = ROOT / "papers" / "charon" / "charon.log.jsonl"
OUT = HERE / "edited_after_issue_census_result.json"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _tracked(repo: rb.Repo, path: Path) -> bool:
    rel = repo.rel_or_none(path)
    if rel is None:
        return False
    return bool(repo._run(["ls-files", "-z", "--", rel], ok_codes=(0, 1)).strip(b"\0"))


def _hunks(old_lines, new_lines):
    """Old line numbers removed and new line numbers added, from a zero-context unified diff."""
    removed, added = set(), set()
    n_removed = n_added = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0):
        m = _HUNK.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2) or "1")
            c, d = int(m.group(3)), int(m.group(4) or "1")
            removed |= set(range(a, a + b)) if b else set()
            added |= set(range(c, c + d)) if d else set()
        elif line.startswith("-") and not line.startswith("---"):
            n_removed += 1
        elif line.startswith("+") and not line.startswith("+++"):
            n_added += 1
    return removed, added, n_removed, n_added


def _working_document(repo: rb.Repo, cert_path: Path, cert: dict, head: str):
    """The document beside the certificate, or — for a staging copy whose certificate names a
    document living elsewhere — the file at HEAD with the certificate's own ``document`` name."""
    beside = cert_path.with_name(cert_path.name.replace(".certificate.json", ".md"))
    if beside.exists():
        return beside
    name = cert.get("document")
    if name:
        for p in repo.paths_named(head, name):
            wp = repo.top / p
            if wp.is_file():
                return wp
    return None


def edited_documents(repo: rb.Repo, census: dict, head: str) -> list:
    out = []
    for r in census["records"]:
        doc = r.get("document") or {}
        if doc.get("cell") != "at_issue":
            continue
        cp = ROOT / r["certificate"]
        cert = json.loads(cp.read_text(encoding="utf-8"))
        sworn = repo.cat(doc["blob"]).decode("utf-8", "replace").replace("\r\n", "\n")
        wp = _working_document(repo, cp, cert, head)
        rec = {"certificate": r["certificate"], "sworn_document": doc["path"],
               "issuing_commit": r["issuing_commit"], "working_document": None,
               "recorded_class": r["recorded_class"],
               "stands_over_sworn_bytes": r["stands_over_sworn_bytes"]}
        if wp is None:
            rec.update(status="NO_WORKING_DOCUMENT")
            out.append(rec)
            continue
        rec["working_document"] = repo.rel_or_none(wp)
        rec["edited_at"] = repo._run(["log", "-1", "--format=%H %cI", "--",
                                      rec["working_document"]]).decode().strip() or None
        current = wp.read_bytes().decode("utf-8", "replace").replace("\r\n", "\n")
        old_lines, new_lines = sworn.split("\n"), current.split("\n")
        removed, added, n_removed, n_added = _hunks(old_lines, new_lines)
        rows = cert.get("ledger") or []
        touched = [row for row in rows if row.get("line") in removed]
        by_status = Counter(row.get("status") for row in touched)
        added_numbers = [n for n in extract_numbers(current) if n.get("line") in added]
        live = None
        try:
            receipts, _missing, _drift = _resolve_receipts(cp, cert, ROOT)
            if receipts:
                live = verdict_class(certify_doc(wp, receipts)["verdict"])
        except Exception as e:   # noqa: BLE001 — a live re-certification failure is reported, not raised
            live = f"error: {str(e)[:80]}"
        rec.update(status="EDITED",
                   lines_removed=n_removed, lines_added=n_added,
                   ledger_rows_on_removed_lines={"total": len(touched),
                                                 "VERIFIED": by_status.get("VERIFIED", 0),
                                                 "ABSTAIN": by_status.get("ABSTAIN", 0),
                                                 "UNGROUNDED": by_status.get("UNGROUNDED", 0),
                                                 "tokens": [{"line": row["line"], "token": row["token"],
                                                             "status": row["status"]}
                                                            for row in touched[:20]]},
                   numbers_on_added_lines={"total": len(added_numbers),
                                           "tokens": [{"line": n["line"], "token": n["token"]}
                                                      for n in added_numbers[:20]]},
                   live_class_over_working_document=live,
                   live_class_equals_recorded=(live == r["recorded_class"]) if live else None)
        out.append(rec)
    return out


def charon_reconciliation(census: dict) -> dict:
    by_doc = {}
    for r in census["records"]:
        doc = r["certificate"].replace(".certificate.json", ".md")
        by_doc[doc] = r
    lines = [json.loads(l) for l in CHARON.read_text(encoding="utf-8").splitlines() if l.strip()]
    oath = [l for l in lines if l.get("kind") == "oath-certificate"]
    cats = Counter()
    rows = []
    unmatched = []
    for l in oath:
        path = (l.get("subject") or {}).get("path")
        r = by_doc.get(path)
        if r is None:
            unmatched.append(path)
            continue
        ch = l.get("reproduced")
        st = r["stands_over_sworn_bytes"]
        moved = []
        if (r.get("cells") or {}).get("at_issue"):
            moved.append("receipt at issue")
        if (r.get("document") or {}).get("cell") == "at_issue":
            moved.append("document at issue")
        elif (r.get("document") or {}).get("note"):
            moved.append("document found by the certificate's own document field, elsewhere in the tree")
        if ch is True and st is True:
            cat = "both_reproduce"
        elif ch is False and st is True:
            cat = "charon_not_reproduced_census_stands"
        elif ch is None and st is True:
            cat = "charon_unresolved_census_stands"
        elif ch is False and st is False:
            cat = "both_not_verifier_moved"
        else:
            cat = f"other:{ch}:{st}"
        cats[cat] += 1
        if cat != "both_reproduce":
            rows.append({"document": path, "charon_reproduced": ch, "charon_verdict": l.get("verdict"),
                         "census_stands": st, "census_reason": r.get("stands_reason"),
                         "what_moved": moved or ["nothing — every byte in place"],
                         "category": cat})
    return {"charon_oath_lines": len(oath), "census_certificates": len(census["records"]),
            "unmatched_charon_lines": unmatched, "categories": dict(cats), "disagreements": rows,
            "reading": ("Charon reproduces a certificate over the WORKING tree at ingest and records "
                        "true/false/null; the census re-derives over the bytes at the ISSUING commit. "
                        "A line Charon could not reproduce or resolve that the census says stands is "
                        "a certificate whose receipt or document moved after issue; a line both call "
                        "false is the verifier having moved with every byte in place.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--census", default=str(CENSUS),
                    help="the committed receipt-binding census to read (never written)")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    census_path = Path(a.census).resolve()
    try:
        repo = rb.Repo(ROOT)
    except rb.RepoUnavailable as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    try:
        if out.exists() and _tracked(repo, out):
            print(f"REFUSED: {out} is tracked; a committed census is history — use --out", file=sys.stderr)
            return 2
        head = repo.head()
        census = json.loads(census_path.read_text(encoding="utf-8"))
        edited = edited_documents(repo, census, head)
        recon = charon_reconciliation(census)
    finally:
        repo.close()
    e = [x for x in edited if x.get("status") == "EDITED"]
    totals = {
        "documents_at_issue": len(edited),
        "with_working_document": len(e),
        "without_working_document": len(edited) - len(e),
        "lines_removed_total": sum(x["lines_removed"] for x in e),
        "lines_added_total": sum(x["lines_added"] for x in e),
        "documents_with_ledger_rows_removed": sum(1 for x in e if x["ledger_rows_on_removed_lines"]["total"]),
        "documents_with_verified_rows_removed": sum(1 for x in e if x["ledger_rows_on_removed_lines"]["VERIFIED"]),
        "verified_rows_removed_total": sum(x["ledger_rows_on_removed_lines"]["VERIFIED"] for x in e),
        "ledger_rows_removed_total": sum(x["ledger_rows_on_removed_lines"]["total"] for x in e),
        "documents_with_numbers_added": sum(1 for x in e if x["numbers_on_added_lines"]["total"]),
        "numbers_added_total": sum(x["numbers_on_added_lines"]["total"] for x in e),
        "live_class_equals_recorded": sum(1 for x in e if x.get("live_class_equals_recorded") is True),
        "live_class_differs": sum(1 for x in e if x.get("live_class_equals_recorded") is False),
    }
    result = {
        "schema": "styxx-oath/edited-after-issue-census/v1",
        "inputs": {"census": repo.rel_or_none(census_path) or str(census_path),
                   "census_head": census.get("head"),
                   "charon_log": "papers/charon/charon.log.jsonl"},
        "head": head,
        "totals": totals,
        "documents": edited,
        "charon": recon,
        "reading": ("a removed line with a ledger row is a line the published certificate examined "
                    "and the working document no longer has; an added line with numbers is one the "
                    "published certificate never examined (the live audit does). Neither is a verdict."),
        "limits": ["line-level diff; a number moved within a line is not seen",
                   "numbers on added lines are counted by the verifier's own extractor, so its "
                   "exclusions (years, ids, versions) apply",
                   "one clone, one day"],
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    t = totals
    print(f"edited after issue: {t['documents_at_issue']} documents ({t['with_working_document']} with a working "
          f"document); lines removed {t['lines_removed_total']}, added {t['lines_added_total']}; "
          f"ledger rows on removed lines {t['ledger_rows_removed_total']} in "
          f"{t['documents_with_ledger_rows_removed']} documents (VERIFIED {t['verified_rows_removed_total']} in "
          f"{t['documents_with_verified_rows_removed']}); numbers on added lines {t['numbers_added_total']} in "
          f"{t['documents_with_numbers_added']}; live class equals recorded {t['live_class_equals_recorded']}, "
          f"differs {t['live_class_differs']}")
    print(f"charon: {recon['charon_oath_lines']} OATH lines vs {recon['census_certificates']} census rows; "
          f"categories {recon['categories']}; unmatched {len(recon['unmatched_charon_lines'])}")
    for d in recon["disagreements"]:
        print(f"  {d['category']:38} {d['document']}  ({', '.join(d['what_moved'])})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
