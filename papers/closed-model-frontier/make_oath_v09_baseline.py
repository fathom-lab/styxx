"""Snapshot the SHIPPED ledger across every resolvable document — the v0.9 severability baseline.

Run ONCE, before `styxx/certify.py` is touched. The battery's G5 leg re-certifies the same
documents with both v0.9 flags OFF and demands a status-identical ledger: that is what makes the
clause severable, and it is unprovable after the fact without this file.

  python papers/closed-model-frontier/make_oath_v09_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                      # noqa: E402
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v09_baseline_ledger.json"


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
    docs = resolvable_docs()
    ledger, verdicts, counts = {}, {}, {}
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception as exc:                                  # pragma: no cover - defensive
            print(f"SKIP {doc.name}: {exc}")
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        counts[rel] = cert["counts"]
        for e in cert["ledger"]:
            ledger[f"{rel}|L{e['line']}|{e['token']}"] = e["status"]
    payload = {
        "purpose": "v0.9 severability baseline (PREREG_oath_v09_is_spec_json_idiom_2026_08_23)",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "documents": len(verdicts),
        "tokens": len(ledger),
        "held": sum(1 for v in verdicts.values() if v == "OATH-HELD"),
        "failed": sum(1 for v in verdicts.values() if v != "OATH-HELD"),
        "verdicts": verdicts,
        "counts": counts,
        "ledger": ledger,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"documents {payload['documents']}  tokens {payload['tokens']}  "
          f"HELD {payload['held']}  FAILED {payload['failed']} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
