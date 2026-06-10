"""Deterministic pre-triage of the corpus attestation's UNGROUNDED claims ($0, no API).

Buckets every UNGROUNDED claim in oath_corpus_attestation.json by mechanical evidence, so the
adversarial triage fleet starts from classified ammunition instead of raw noise:

  NEAR_MISS          a receipt leaf sits within 3x the claim's rounding tolerance -> likely a stale
                     printed value or precision drift; the leaf is named (doc-correction candidate).
  INT_HOME_UNBOUND   an integer with an exactly-equal leaf that count-binding rejected -> vocabulary
                     gap between line and path (doc-vocab or receipt-field-name repair).
  INT_NO_HOME        an integer with no equal leaf anywhere -> missing summary field (addendum
                     candidate) or a genuinely wrong count.
  TABLE_CELL         lives in a markdown table row (binding context = header; often cross-receipt).
  CROSS_REF          line references another experiment (B\\d+/R\\d/v\\d/'known'/'prior'/'original')
                     -> receipt likely lives in another paper dir (binding repair).
  PCT                percentage claims (scaling rules apply).
  FLOAT_NO_HOME      everything else -> fleet judgment required.

Writes triage_prep.json (per-bucket counts + full classified ledger).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from styxx.certify import receipt_values  # noqa: E402

CROSS = re.compile(r"\bB\d+\b|\bR\d\b|\bv\d\b|\bknown\b|\bprior\b|\boriginal\b|\bearlier\b|\bcf\b", re.I)


def main() -> int:
    att = json.loads((HERE / "oath_corpus_attestation.json").read_text(encoding="utf-8"))
    ledger, buckets = [], {}
    for e in att["oath_failed_docs"]:
        leaves = []
        for r in e.get("receipts", []):
            p = REPO / r
            if p.is_file():
                try:
                    leaves += [v for _, v in receipt_values(json.loads(p.read_text(encoding="utf-8")))]
                except Exception:
                    pass
        for u in e.get("ungrounded", []):
            tok = u["token"].replace(",", "").lstrip("+")
            try:
                val = float(tok)
            except ValueError:
                continue
            dec = len(tok.split(".")[1]) if "." in tok else 0
            ctx = u["context"]
            tol = 3 * (0.5 * 10 ** (-dec)) if dec else 0
            near = sorted((abs(lv - val), lv) for lv in leaves if abs(lv - val) <= max(tol, 0.0))[:1]
            exact = any(lv == val for lv in leaves)
            if dec and near and near[0][0] > 0:
                b = "NEAR_MISS"
            elif dec == 0 and exact:
                b = "INT_HOME_UNBOUND"
            elif dec == 0:
                b = "INT_NO_HOME"
            elif ctx.lstrip().startswith("|"):
                b = "TABLE_CELL"
            elif CROSS.search(ctx):
                b = "CROSS_REF"
            elif "%" in ctx:
                b = "PCT"
            else:
                b = "FLOAT_NO_HOME"
            buckets[b] = buckets.get(b, 0) + 1
            row = {"doc": e["doc"], "line": u["line"], "token": u["token"], "bucket": b,
                   "context": ctx[:120]}
            if near:
                row["near_leaf_value"] = near[0][1]
            ledger.append(row)

    out = {"prepared_from": "oath_corpus_attestation.json (certify v0.3)",
           "n_ungrounded": len(ledger), "buckets": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
           "fleet_guidance": {
               "NEAR_MISS": "verify leaf vs doc; if leaf is the seeded recomputation, correct doc loudly",
               "INT_HOME_UNBOUND": "align line vocabulary with the receipt field (doc edit) or add alias field",
               "INT_NO_HOME": "derive + persist summary field via an addendum script, or flag wrong count",
               "TABLE_CELL/CROSS_REF": "bind the missing cross-receipt (doc citation or battery set)",
               "FLOAT_NO_HOME/PCT": "adversarial judgment: contradiction vs unpersisted computation",
           },
           "ledger": ledger}
    (HERE / "triage_prep.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ledger)} claims classified:")
    for k, v in out["buckets"].items():
        print(f"  {v:4d}  {k}")
    print("-> triage_prep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
