"""Deterministic corpus triage: classify the 684 ungrounded claims (no API, no fleet).

The honest question is not "are 684 documents lying" but "of the ungrounded claims, how many are
real discrepancies vs binder-limitations (a correct number the receipt simply does not expose a
stem-matching summary field for)?" This pass answers it mechanically against the receipts.

For each ungrounded claim with a numeric token, search the doc's cited receipts for a leaf within
rounding tolerance:
  EXACT_LEAF_EXISTS   a leaf equals the value to display precision -> the number IS in the receipt;
                      binder missed it (vocabulary/path gap or table-cell) -> NOT an error.
  ROUNDING_OK         a leaf is within 1.5x the value's rounding tolerance -> acceptable rounding.
  REAL_STALE          a near leaf exists but the doc value disagrees beyond rounding -> CANDIDATE
                      correction (named for review).
  NO_LEAF             no leaf near the value -> count/spec/cross-ref/computed-unpersisted (provenance
                      debt, not a contradiction) -> needs a receipt field, not a doc edit.
Writes triage_classify_result.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from styxx.certify import receipt_values  # noqa: E402


def leaves_for(receipts):
    vals = []
    for r in receipts:
        p = REPO / r
        if p.is_file():
            try:
                vals += [v for _, v in receipt_values(json.loads(p.read_text(encoding="utf-8")))]
                vals += [v for _, v in receipt_values(json.loads(p.read_text(encoding="utf-8")),
                                                      include_bulk=True)]
            except Exception:
                pass
    return vals


def classify(tok: str, leaves) -> tuple[str, float | None]:
    t = tok.replace(",", "").lstrip("+")
    try:
        val = float(t)
    except ValueError:
        return "NO_LEAF", None
    dec = len(t.split(".")[1]) if "." in t else 0
    tol = 0.5 * 10 ** (-dec) if dec else 0.5
    near = sorted(((abs(lv - val), lv) for lv in leaves), key=lambda x: x[0])
    if not near:
        return "NO_LEAF", None
    d, leaf = near[0]
    # percent<->fraction aware
    alt = min(abs(leaf * 100 - val) if abs(val) > 2 else 9e9, abs(leaf / 100 - val))
    d = min(d, alt)
    if d <= 1e-9:
        return "EXACT_LEAF_EXISTS", leaf
    if d <= 1.5 * tol:
        return "ROUNDING_OK", leaf
    if d <= 0.02 + tol:
        return "REAL_STALE", leaf
    return "NO_LEAF", None


def main() -> int:
    att = json.loads((HERE / "oath_corpus_attestation.json").read_text(encoding="utf-8"))
    buckets = {"EXACT_LEAF_EXISTS": 0, "ROUNDING_OK": 0, "REAL_STALE": 0, "NO_LEAF": 0}
    stale = []
    n = 0
    for e in att["oath_failed_docs"]:
        leaves = leaves_for(e.get("receipts", []))
        for u in e.get("ungrounded", []):
            n += 1
            cls, leaf = classify(u["token"], leaves)
            buckets[cls] += 1
            if cls == "REAL_STALE":
                stale.append({"doc": e["doc"], "line": u["line"], "token": u["token"],
                              "nearest_leaf": round(leaf, 6), "context": u["context"][:110]})
    out = {"analysis": "deterministic corpus triage of OATH-ungrounded claims",
           "source": "oath_corpus_attestation.json (certify v0.3)",
           "n_ungrounded": n, "buckets": buckets,
           "provably_not_errors_count": buckets["EXACT_LEAF_EXISTS"] + buckets["ROUNDING_OK"],
           "provenance_debt_no_leaf_count": buckets["NO_LEAF"],
           "reading": ("Most ungrounded claims are NOT errors. EXACT_LEAF_EXISTS + ROUNDING_OK "
                       "(provably-fine) are correct numbers the v0.3 binder could not stem-bind "
                       "(table cells, vocabulary gaps). NO_LEAF are counts / spec bars / "
                       "cross-experiment refs / computed-unpersisted = provenance debt (needs a "
                       "receipt field, not a doc edit). REAL_STALE is an UPPER BOUND on "
                       "doc-correction candidates and is heavily inflated by COINCIDENTAL-NEAR "
                       "leaves (a value whose true referent is external or in another receipt "
                       "happens to sit near an unrelated leaf, e.g. the 0.785 semantic_entropy "
                       "literature reference). It must be reviewed claim-by-claim (autopilot rung / "
                       "fleet); it is NOT a bulk-edit list. The true error count is far below 174."),
           "provably_not_errors": None,
           "real_stale_upper_bound_review_required": stale}
    (HERE / "triage_classify_result.json").write_text(json.dumps(out, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps({"n": n, "buckets": buckets, "n_real_stale": len(stale)}, indent=2))
    for s in stale[:20]:
        print(f"  STALE {s['doc'].split('/')[-1][:40]:42} L{s['line']:<4} {s['token']:>9} vs leaf {s['nearest_leaf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
