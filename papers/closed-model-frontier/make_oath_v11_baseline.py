"""Snapshot the SHIPPED verifier before `styxx/certify.py` is touched — the v0.11 OFF-arm pin.

Run ONCE, on the pre-change tree. Two things are recorded, and both are unprovable after the
fact without this file:

1. **The certified frame's ledger**, per token, as `(status, receipt_ref)`. The v0.11 battery's
   OFF arm must reproduce it exactly. Recording the REASON alongside the status is stricter
   than the v0.10 baseline, which recorded status alone: this cycle ships a clause whose whole
   visible signature is a new reason code, so a baseline blind to reasons could not tell a
   severable clause from one that silently re-labels existing abstentions.

2. **The extraction stream, repo-wide**, as a per-document digest over the full
   `extract_numbers` output. Gate G1 demands extractor replication mismatches = 0, and this
   cycle refactors the table-header machinery that `extract_numbers` uses (the v0.11 clause
   reads that machinery rather than copying it, per the prereg's conjunct 1). A refactor that
   moved a single token would invalidate every number downstream, so the check is run over
   every markdown document under `papers/`, not just the 140 in frame.

   Honest scope, stated because the phrase is inherited: the v0.10 census's
   `extractor_mismatches_vs_live_extract_numbers` compared a SHADOW re-implementation against
   live extraction. This file compares live extraction BEFORE the change against live
   extraction AFTER it. Those are different controls answering different questions — the
   census asked "did the census read extraction correctly", this asks "did the change move
   extraction". Only the second is available to a battery that must not copy the machinery it
   gates.

  python papers/closed-model-frontier/make_oath_v11_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc, extract_numbers                     # noqa: E402
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v11_baseline_ledger.json"


def resolvable_docs() -> list[tuple[Path, list[Path]]]:
    """The certified frame: `papers/**` documents carrying a certificate whose receipts ALL
    resolve. Identical definition to the v0.10 census's `frame.certified`, and to the
    `_resolvable()` helper in `tests/test_certificate_reproduces.py`."""
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def extraction_digests() -> dict[str, dict]:
    """Per-document digest of the FULL extract_numbers output, repo-wide under papers/."""
    digests: dict[str, dict] = {}
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        try:
            toks = extract_numbers(md.read_text(encoding="utf-8"))
        except Exception:
            continue
        blob = json.dumps(toks, sort_keys=True, ensure_ascii=False)
        digests[md.relative_to(ROOT).as_posix()] = {
            "n": len(toks),
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        }
    return digests


def main() -> int:
    docs = resolvable_docs()
    ledger, verdicts, counts = {}, {}, {}
    frame_counts = {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception as exc:                                  # pragma: no cover - defensive
            print(f"SKIP {doc.name}: {exc}")
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        counts[rel] = cert["counts"]
        for s in frame_counts:
            frame_counts[s] += cert["counts"][s]
        for i, e in enumerate(cert["ledger"]):
            # v0.10's keying, kept verbatim: the ledger ORDINAL is what stops a line carrying
            # the same token string twice from collapsing to a single row.
            ledger[f"{rel}|L{e['line']}|{e['token']}|#{i}"] = [e["status"], e["receipt_ref"]]

    digests = extraction_digests()
    payload = {
        "purpose": "v0.11 OFF-arm pin (PREREG_oath_v11_row_ordinal_retraction_2026_08_25)",
        "key_format": "<rel>|L<line>|<token>|#<ledger ordinal> -> [status, receipt_ref]",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "frame": {
            "documents": len(verdicts),
            "tokens": len(ledger),
            "status_counts": frame_counts,
            "held": sum(1 for v in verdicts.values() if v == "OATH-HELD"),
            "failed": sum(1 for v in verdicts.values() if v != "OATH-HELD"),
        },
        "extraction_repo_wide": {
            "documents": len(digests),
            "tokens": sum(d["n"] for d in digests.values()),
        },
        "verdicts": verdicts,
        "counts": counts,
        "ledger": ledger,
        "extraction": digests,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    f = payload["frame"]
    print(f"frame: {f['documents']} docs  {f['tokens']} tokens  "
          f"V {f['status_counts']['VERIFIED']} / A {f['status_counts']['ABSTAIN']} / "
          f"U {f['status_counts']['UNGROUNDED']}  HELD {f['held']} FAILED {f['failed']}")
    print(f"extraction: {payload['extraction_repo_wide']['documents']} docs  "
          f"{payload['extraction_repo_wide']['tokens']} tokens -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
