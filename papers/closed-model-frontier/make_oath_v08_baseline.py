"""Snapshot the current verifier's ledger — the G5 severability reference for OATH v0.8.

G5 asks whether a severable clause is truly inert when its flag is OFF. That question is only
answerable against a ledger captured at the verifier BEFORE the clause exists, so this must be run
and committed prior to any edit of `styxx/certify.py`.

Records, for every document under `papers/**` whose cited receipts all resolve: the status of every
ledger entry keyed by `<relpath>|L<line>|<token>`, the document verdict, and the verifier SHA the
snapshot was taken at. `run_oath_v08_battery.py` compares the flag-OFF ledger against this file and
fails G5 on any difference.

  python papers/closed-model-frontier/make_oath_v08_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                # noqa: E402
from styxx.corpus_audit import _resolve_receipts     # noqa: E402

OUT = HERE / "oath_v08_baseline_ledger.json"


def resolvable_docs():
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


def main() -> int:
    t0 = time.time()
    docs = resolvable_docs()
    ledger, verdicts, counts = {}, {}, {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        for e in cert["ledger"]:
            ledger[f"{rel}|L{e['line']}|{e['token']}"] = e["status"]
            counts[e["status"]] = counts.get(e["status"], 0) + 1

    report = {
        "note": "G5 severability reference for PREREG_oath_v08_float_field_binding_2026_08_23; "
                "captured BEFORE any edit to styxx/certify.py",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "docs": len(docs),
        "counts": counts,
        "held": sum(1 for v in verdicts.values() if v == "OATH-HELD"),
        "failed": sum(1 for v in verdicts.values() if v != "OATH-HELD"),
        "verdicts": verdicts,
        "ledger": ledger,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"docs {len(docs)}  entries {len(ledger)}  counts {counts}")
    print(f"HELD {report['held']}  FAILED {report['failed']}")
    print(f"verifier {report['verifier_sha256'][:16]}  elapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
