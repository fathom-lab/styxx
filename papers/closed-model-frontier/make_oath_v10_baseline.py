"""Snapshot the SHIPPED ledger across every resolvable document — the v0.10 severability baseline.

Run ONCE, before `styxx/certify.py` is touched. The battery's severability leg re-certifies the
same documents with both v0.10 flags OFF and demands a status-identical ledger: that is what makes
the clauses severable, and it is unprovable after the fact without this file.

  python papers/closed-model-frontier/make_oath_v10_baseline.py

**Keying differs from `make_oath_v09_baseline.py` on purpose, and the difference is load-bearing.**
v0.9 keyed a ledger entry `"<doc>|L<line>|<token>"`. When one line carries the SAME token string
twice ("10 neutral + 10 in-frame", "0.0854 = 0.0854") that key collides and the dict keeps only
the last write, so both tokens are represented by one status. Repo-wide, 1,932 lines carry a
duplicated token, and a duplicated token is EXACTLY the population this cycle addresses — under
v0.9's keying a severability leg would be structurally blind to the tokens under test. This
snapshot appends the ledger ORDINAL, so every extracted token gets its own row.
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

OUT = HERE / "oath_v10_baseline_ledger.json"


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
        for i, e in enumerate(cert["ledger"]):
            ledger[f"{rel}|L{e['line']}|{e['token']}|#{i}"] = e["status"]
    payload = {
        "purpose": "v0.10 severability baseline (PREREG_oath_v10_token_column_2026_08_23)",
        "key_format": "<rel>|L<line>|<token>|#<ledger ordinal> — the ordinal is what keeps "
                      "same-token-twice lines (1,932 repo-wide) from collapsing to one row",
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
