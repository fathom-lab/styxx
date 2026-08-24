"""Do the corpus's committed certificates still reproduce at HEAD?

A certificate is an artifact of a document, a receipt set, AND a verifier. It records all three
(`document_sha256`, `receipts_sha256`, `verifier_sha256`) precisely so drift is detectable — but
nothing in this repository checks the third. The verifier moves every cycle, and a certificate
committed at an older one keeps asserting its old verdict.

That is not hypothetical. `PROSPECTUS_knowsay_2026_07_27.certificate.json` says OATH-HELD with zero
UNGROUNDED tokens. Re-run at HEAD, the same document against the same receipts is OATH-FAILED with
four. The failure is real and has been invisible since the verifier moved past the certificate.

This census measures the whole surface: for every document whose cited receipts all resolve,
re-certify at the CURRENT verifier and compare against what the committed certificate asserts.

Four kinds of drift, in ascending order of seriousness:

  verifier   the certificate was built at a different `certify.py` than the tree holds. Expected and
             benign on its own — it is the precondition for the rest, not a defect.
  counts     the VERIFIED/ABSTAIN/UNGROUNDED tallies changed. Usually a recall improvement moving
             tokens between ABSTAIN and VERIFIED; worth knowing, not an integrity failure.
  document   the document's own SHA no longer matches the certificate. The certificate describes a
             file that no longer exists in that form.
  VERDICT    the certificate says OATH-HELD and the current verifier says OATH-FAILED. **This is the
             one that matters**: a document carrying a passing certificate that no longer passes.

Non-destructive: nothing is regenerated and no certificate is written. The only file written is this
census's own result JSON.

  python papers/closed-model-frontier/oath_certificate_drift_census.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                # noqa: E402
from styxx.corpus_audit import _resolve_receipts     # noqa: E402

OUT = HERE / "oath_certificate_drift_census.json"


def main() -> int:
    t0 = time.time()
    tree_sha = hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest()
    rows, unresolvable = [], 0

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
        if not receipts or missing:
            unresolvable += 1
            continue
        try:
            live = certify_doc(doc, receipts)
        except Exception:
            continue
        rows.append({
            "doc": doc.relative_to(ROOT).as_posix(),
            "certificate_verdict": rec.get("verdict"),
            "live_verdict": live["verdict"],
            "certificate_counts": rec.get("counts"),
            "live_counts": live["counts"],
            "certificate_verifier_sha256": rec.get("verifier_sha256"),
            "verifier_drift": rec.get("verifier_sha256") != tree_sha,
            "document_drift": rec.get("document_sha256") != live["document_sha256"],
            "counts_drift": rec.get("counts") != live["counts"],
            "verdict_drift": rec.get("verdict") != live["verdict"],
            "held_to_failed": (rec.get("verdict") == "OATH-HELD"
                               and live["verdict"] != "OATH-HELD"),
        })

    held_to_failed = [r for r in rows if r["held_to_failed"]]
    report = {
        "note": "certificate reproducibility census — does a committed certificate still hold at "
                "the verifier now in the tree?",
        "tree_verifier_sha256": tree_sha,
        "certificates_examined": len(rows),
        "skipped_unresolvable_receipts": unresolvable,
        "verifier_drift": sum(1 for r in rows if r["verifier_drift"]),
        "document_drift": sum(1 for r in rows if r["document_drift"]),
        "counts_drift": sum(1 for r in rows if r["counts_drift"]),
        "verdict_drift": sum(1 for r in rows if r["verdict_drift"]),
        "held_to_failed": [r["doc"] for r in held_to_failed],
        "distinct_certificate_verifiers": len({r["certificate_verifier_sha256"] for r in rows}),
        "certificates_per_verifier": dict(
            Counter(r["certificate_verifier_sha256"][:12] for r in rows).most_common()),
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"certificates examined                        : {report['certificates_examined']}")
    print(f"  built at a verifier other than the tree's  : {report['verifier_drift']}")
    print(f"  document SHA drift                         : {report['document_drift']}")
    print(f"  counts drift                               : {report['counts_drift']}")
    print(f"  VERDICT drift                              : {report['verdict_drift']}")
    print(f"  *** committed HELD, live FAILED            : {len(held_to_failed)} ***")
    for r in held_to_failed:
        print(f"      {r['doc']}")
        print(f"        certificate {r['certificate_counts']} -> live {r['live_counts']}")
    print(f"\ndistinct verifiers across committed certificates: "
          f"{report['distinct_certificate_verifiers']}")
    for sha, c in report["certificates_per_verifier"].items():
        mark = "   <-- the tree" if tree_sha.startswith(sha) else ""
        print(f"   {sha}  x{c}{mark}")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
