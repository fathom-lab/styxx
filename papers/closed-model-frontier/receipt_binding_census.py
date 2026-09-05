"""receipt_binding_census.py — where the bytes every tracked certificate swore to are today.

SPEC_oath_receipt_binding_2026_09_04 R6, as amended by its ERRATA. Population: every
``*.certificate.json`` that ``git ls-files`` returns at the commit this runs on — the arXiv
staging copies included as their own rows, because they are tracked. For every citation, the
cell (same / at_issue / elsewhere / unbacked / unrecoverable) and its evidence; for the document,
its own cell; for every certificate, the issuing commit and whether the current verifier
reproduces the recorded verdict class over the document and receipt bytes at the issuing commit.
*Issuing commit unrecoverable* is its own cell in both tables.

It rebuilds nothing under ``papers/`` except its own result, and it REFUSES to overwrite a result
that is already tracked: a census that has been committed is history, whether or not a RESULT has
sworn to it yet. A new census is a new file with a new date (``--out``).

    python papers/closed-model-frontier/receipt_binding_census.py [--out PATH] [--root REPO]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent
sys.path.insert(0, str(DEFAULT_ROOT))

from styxx import corpus_audit, receipt_binding as rb   # noqa: E402

OUT = HERE / "receipt_binding_census_result.json"
SPEC = "papers/closed-model-frontier/SPEC_oath_receipt_binding_2026_09_04.md"
CODE = ("papers/closed-model-frontier/receipt_binding_census.py", "styxx/receipt_binding.py",
        "styxx/corpus_audit.py", "styxx/certify.py")


def population(repo: rb.Repo) -> list:
    out = repo._run(["ls-files", "-z", "--", "*.certificate.json"])
    return sorted(p.decode("utf-8", "replace") for p in out.split(b"\0") if p)


def _tracked(repo: rb.Repo, path: Path) -> bool:
    rel = repo.rel_or_none(path)
    if rel is None:
        return False
    out = repo._run(["ls-files", "-z", "--", rel], ok_codes=(0, 1))
    return bool(out.strip(b"\0"))


def _provenance(repo: rb.Repo, root: Path, head: str) -> dict:
    """The blob ids of the code that ran, beside the blob ids at ``head`` — so a reader can see
    whether the census was produced by committed code (battery A8: the first census named the
    SPEC commit as its head while the code that computed it was still uncommitted)."""
    rows = {}
    all_committed = True
    for rel in CODE:
        p = root / rel
        if not p.exists():
            rows[rel] = {"working_blob": None, "blob_at_head": None, "committed_at_head": False}
            all_committed = False
            continue
        raw = p.read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        working = {rb.git_blob_id(raw), rb.git_blob_id(lf), rb.git_blob_id(lf.replace(b"\n", b"\r\n"))}
        at_head = repo.blob_at(head, rel)
        ok = bool(at_head) and at_head in working
        rows[rel] = {"working_blob": rb.git_blob_id(lf), "blob_at_head": at_head,
                     "committed_at_head": ok}
        all_committed = all_committed and ok
    return {"code": rows, "code_committed_at_head": all_committed,
            "note": ("head is the commit the working tree was at; when code_committed_at_head is "
                     "false the census ran from a working tree ahead of head, and the commit that "
                     "carries this file also carries that code")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.out).resolve()
    repo, why = corpus_audit.open_history(root, "on")
    if repo is None:
        print(f"REFUSED: history unavailable ({why}); the census needs a full clone", file=sys.stderr)
        return 2
    try:
        if out.exists() and _tracked(repo, out):
            print(f"REFUSED: {out} is tracked; a committed census is history and is never "
                  f"regenerated in place — write a new dated file with --out", file=sys.stderr)
            return 2
        head = repo.head()
        certs = population(repo)
        records = []
        cit = Counter()
        norm = Counter()
        same_split = Counter()
        doc_cells = Counter()
        per = Counter()
        stands = Counter()
        reasons = Counter()
        examples = {"at_issue": [], "elsewhere": [], "unbacked": [], "unrecoverable": [],
                    "document_at_issue": [], "document_moved": [], "not_standing": [],
                    "not_standing_same_only": [], "stands_null": []}
        for i, rel in enumerate(certs, 1):
            cp = root / rel
            cert = json.loads(cp.read_text(encoding="utf-8"))
            doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
            resolved_paths, _missing, _drift = corpus_audit._resolve_receipts(cp, cert, root)
            resolved = {p.name: p for p in resolved_paths}
            b = corpus_audit._binding_for(repo, cp, cert, resolved, doc)
            rec = {"certificate": rel, "recorded_verdict": cert.get("verdict"),
                   "recorded_class": corpus_audit.verdict_class(cert.get("verdict")),
                   "n_citations": len(cert.get("receipts_sha256", {})),
                   "issuing_commit": b.get("issuing_commit"),
                   "cells": b.get("cells"), "document": b.get("document"),
                   "stands_over_sworn_bytes": b.get("stands_over_sworn_bytes"),
                   "verdict_over_sworn_bytes": b.get("verdict_over_sworn_bytes"),
                   "stands_reason": b.get("stands_reason"),
                   "citations": b.get("citations", [])}
            if b.get("unrecoverable_reason"):
                rec["unrecoverable_reason"] = b["unrecoverable_reason"]
            records.append(rec)
            cells = b.get("cells") or {}
            for c, n in cells.items():
                cit[c] += n
            for c in b.get("citations", []):
                if c.get("normalisation"):
                    norm[c["normalisation"]] += 1
                if c.get("blob_normalisation"):
                    norm["blob_" + c["blob_normalisation"]] += 1
                if c["cell"] == "same":
                    same_split["at_issue_too" if c.get("at_issue_too") else "not_at_issue"] += 1
                    if c.get("note"):
                        same_split["outside_audit_root"] += 1
            dcell = (b.get("document") or {}).get("cell")
            if dcell:
                doc_cells[dcell] += 1
                if dcell == "at_issue":
                    examples["document_at_issue"].append(rel)
                if dcell == "moved":
                    examples["document_moved"].append(rel)
            per["n"] += 1
            if b.get("issuing_commit") is None:
                per["issuing_commit_unrecoverable"] += 1
                examples["unrecoverable"].append(rel)
            if cells and sum(cells.values()) == cells.get("same", 0) and cells.get("same", 0) > 0:
                per["all_same"] += 1
            for cell in ("at_issue", "elsewhere", "unbacked"):
                if cells.get(cell):
                    per[f"any_{cell}"] += 1
                    examples[cell].append(rel)
            s = b.get("stands_over_sworn_bytes")
            key = "true" if s is True else "false" if s is False else "null"
            stands[key] += 1
            if s is None:
                reasons[b.get("stands_reason") or "?"] += 1
                if b.get("issuing_commit") is not None:
                    examples["stands_null"].append(rel)
            moved = (cells.get("at_issue", 0) + cells.get("elsewhere", 0)) if cells else 0
            if moved and s is True:
                per["regenerated_and_standing"] += 1
            if s is False:
                per["not_standing"] += 1
                examples["not_standing"].append(rel)
                if cells and sum(cells.values()) == cells.get("same", 0) and dcell == "same":
                    per["not_standing_same_only"] += 1
                    examples["not_standing_same_only"].append(rel)
            if i % 25 == 0:
                print(f"  {i}/{len(certs)}", file=sys.stderr)
        provenance = _provenance(repo, root, head)
    finally:
        repo.close()
    result = {
        "schema": "styxx-oath/receipt-binding-census/v2",
        "spec": SPEC,
        "head": head,
        "provenance": provenance,
        "population": {"rule": "git ls-files -- '*.certificate.json' at head", "n": len(certs)},
        "citations": {"n": sum(cit.values()), **{c: cit.get(c, 0) for c in rb.CELLS},
                      "same_at_issue_too": same_split.get("at_issue_too", 0),
                      "same_not_at_issue": same_split.get("not_at_issue", 0),
                      "same_outside_audit_root": same_split.get("outside_audit_root", 0),
                      "normalisation": {k: norm.get(k, 0) for k in ("content", "raw", "lf", "crlf")},
                      "blob_normalisation": {k: norm.get("blob_" + k, 0) for k in ("content", "raw", "lf", "crlf")}},
        "documents": {c: doc_cells.get(c, 0) for c in rb.DOCUMENT_CELLS},
        "certificates": {"n": per["n"],
                         "issuing_commit_unrecoverable": per["issuing_commit_unrecoverable"],
                         "all_same": per["all_same"],
                         "any_at_issue": per["any_at_issue"],
                         "any_elsewhere": per["any_elsewhere"],
                         "any_unbacked": per["any_unbacked"],
                         "stands_over_sworn_bytes": {k: stands.get(k, 0) for k in ("true", "false", "null")},
                         "stands_null_reasons": dict(reasons),
                         "regenerated_and_standing": per["regenerated_and_standing"],
                         "not_standing": per["not_standing"],
                         "not_standing_same_only": per["not_standing_same_only"]},
        "examples": {k: sorted(v) for k, v in examples.items()},
        "reading": ("a citation's cell says where the bytes the certificate swore to are, at the "
                    "repository root (so same means the working tree, anywhere in it); the "
                    "document cell says the same of the document; stands_over_sworn_bytes says "
                    "whether the CURRENT verifier reproduces the recorded verdict class over the "
                    "document and receipt bytes at the issuing commit. `regenerated_and_standing` "
                    "is the plan's 'receipt regenerated under a certificate'; `any_unbacked` and "
                    "`not_standing` are its 'certificate wrong'; `not_standing_same_only` is the "
                    "verifier having moved with every byte in place (SKEW, not binding). Nothing "
                    "here says a receipt's content was true."),
        "limits": ["basename search: a receipt renamed before its first commit reads unbacked; "
                   "one renamed after a commit reads elsewhere or at_issue",
                   "content identity is modulo newlines; a legacy receipt with MIXED newlines can "
                   "match no reading, and only a certificate carrying its own receipt_binding block "
                   "is matched on content_sha256 (reading `content`)",
                   "stands_over_sworn_bytes is null with its reason whenever any citation is unbacked "
                   "or unrecoverable, when there are no citations, or when the document at the "
                   "issuing commit is not the sworn document",
                   "cells are computed at the repository root; the same audit run under a "
                   "subdirectory reads a receipt outside that root as same-with-note",
                   "one clone, one day; not a measurement of anything but this corpus at this head"],
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    c, p, d = result["citations"], result["certificates"], result["documents"]
    print(f"census at {head[:8]} (code committed at head: {provenance['code_committed_at_head']}): "
          f"{p['n']} certificates, {c['n']} citations")
    print(f"  citations: same {c['same']} (at issue too {c['same_at_issue_too']}, not {c['same_not_at_issue']}, "
          f"outside root {c['same_outside_audit_root']})  at-issue {c['at_issue']}  elsewhere {c['elsewhere']}  "
          f"unbacked {c['unbacked']}  unrecoverable {c['unrecoverable']} | working reading {c['normalisation']} | "
          f"blob reading {c['blob_normalisation']}")
    print(f"  documents: {d}")
    print(f"  certificates: issuing-commit-unrecoverable {p['issuing_commit_unrecoverable']}  "
          f"all-same {p['all_same']}  any-at-issue {p['any_at_issue']}  any-elsewhere {p['any_elsewhere']}  "
          f"any-unbacked {p['any_unbacked']} | stands {p['stands_over_sworn_bytes']} null-reasons {p['stands_null_reasons']} | "
          f"regenerated-and-standing {p['regenerated_and_standing']}  not-standing {p['not_standing']} "
          f"(same-only {p['not_standing_same_only']})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
