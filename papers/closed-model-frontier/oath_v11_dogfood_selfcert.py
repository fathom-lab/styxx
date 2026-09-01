"""Dogfood: run the shipped verifier on THIS cycle's own RESULT and grade its own bindings.

The standing rule is that styxx runs its own audit on its own outward claims every cycle. The
RESULT of the row-ordinal retraction certifies OATH-HELD — and that is not the interesting part.
The interesting part is WHAT its VERIFIED tokens are sworn to, because the defect this cycle
retracted is a token grounding at a leaf that merely holds its value: an index equal to its own
subscript, a coordinate, a seed number.

A document that publishes a 0.4274 false-attestation rate on someone else's table and does not
measure the same channel in its own certificate is grading on a curve.

THE DEFINITION, frozen here before the count is read. A VERIFIED binding is STRUCTURALLY
COINCIDENT iff the receipt path's terminal segment is:
  * a bare array subscript — the leaf IS a position, e.g. `coordinates[0][2]`; or
  * one of the index-like names {line, col, seed, token, case, i, index, idx} — the leaf is a
    coordinate or an identifier, not a measurement.
Everything else is NOMINAL: the terminal segment names a quantity, and the token is sworn to
something that could in principle contradict it.

The definition is deliberately STRUCTURAL and mechanical. It is not a hand adjudication and does
not claim to be one — it cannot tell a nominal binding that is nonetheless the wrong quantity from
a right one, so the coincident count is a FLOOR on the false-attestation surface, never a ceiling.
Ties resolve toward NOMINAL, which is the direction that flatters this document.

  python papers/closed-model-frontier/oath_v11_dogfood_selfcert.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                      # noqa: E402

DOC = HERE / "RESULT_oath_v11_row_ordinal_retraction_2026_08_26.md"
RECEIPTS = [
    HERE / "oath_v11_battery_result.json",
    HERE / "oath_v11_baseline_ledger.json",
    HERE / "oath_v11_panel_recheck.json",
    HERE / "oath_v11_adversarial_audit.json",
    HERE / "oath_v10_ordinal_census.json",
    HERE / "oath_v10_panel_isclaim.json",
]
OUT = HERE / "oath_v11_dogfood_selfcert.json"

INDEX_NAMES = frozenset({"line", "col", "seed", "token", "case", "i", "index", "idx"})
_SUBSCRIPT = re.compile(r"\[\d+\]$")


def terminal(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def is_coincident(path: str) -> bool:
    if _SUBSCRIPT.search(path):
        return True
    return _SUBSCRIPT.sub("", terminal(path)).lower() in INDEX_NAMES


def main() -> int:
    cert = certify_doc(DOC, RECEIPTS)
    rows = []
    for e in cert["ledger"]:
        if e["status"] != "VERIFIED" or not e["receipt_ref"]:
            continue
        receipt, _, path = e["receipt_ref"].partition(":")
        rows.append({"line": e["line"], "token": e["token"], "receipt": receipt, "path": path,
                     "terminal": terminal(path), "coincident": is_coincident(path),
                     "context": e["context"][:120]})
    coincident = [r for r in rows if r["coincident"]]
    payload = {
        "purpose": "dogfood — the shipped verifier run on this cycle's own RESULT, with its "
                   "VERIFIED bindings graded by a frozen structural definition",
        "definition_frozen_before_the_count": {
            "structurally_coincident": "the receipt path's terminal segment is a bare array "
                                       "subscript, or one of "
                                       + ", ".join(sorted(INDEX_NAMES)),
            "ties_resolve": "toward NOMINAL — the direction that flatters this document",
            "status": "MECHANICAL, not a hand adjudication. The coincident count is a FLOOR on "
                      "the false-attestation surface, never a ceiling: this definition cannot "
                      "tell a nominal binding that names the WRONG quantity from a right one.",
        },
        "document": DOC.name,
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "verdict": cert["verdict"],
        "counts": cert["counts"],
        "verified_bindings": len(rows),
        "structurally_coincident": len(coincident),
        "nominal": len(rows) - len(coincident),
        "coincident_share": round(len(coincident) / len(rows), 4) if rows else None,
        "coincident_roster": coincident,
        "all_bindings": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{cert['verdict']}  {cert['counts']}")
    print(f"verified bindings {len(rows)}  structurally coincident {len(coincident)}  "
          f"share {payload['coincident_share']} -> {OUT.name}")
    for r in coincident:
        print(f"  L{r['line']} {r['token']:>8}  ->  {r['receipt']}:{r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
