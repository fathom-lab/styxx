"""OATH v0.7 pre-fix census — the size and shape of the trigger-recall debt.

Two measurements, both taken at the SHIPPED verifier before any v0.7 edit, both deterministic
and local:

  1. THE UNBOUND POOL. Full-precision tokens (>=7 fractional digits) across every document whose
     cited receipts all resolve, split by the verifier's own obligation predicate. Reproduces the
     published G2b share on an independently-built pool, and adds the fact the share alone hides:
     nearly all unbound tokens are ALREADY VERIFIED on the clean corpus, because VERIFIED is
     awarded on a value-match regardless of obligation. The debt is a tamper hole, not a coverage
     hole.

  2. THE DECIMAL SWEEP. For every decimal width, the count of unbound-line numbers that currently
     value-match a receipt leaf (the tamper surface a "decimals >= T obligates" rule would cover)
     against those that do not (the clean-corpus accusation surface such a rule would create).

  python papers/closed-model-frontier/oath_v07_census.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (certify_doc, extract_numbers, receipt_values,   # noqa: E402
                           _match, _TRIGGERS, _TRIGGERS_CORR)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v07_precfix_census.json"
FULLPREC_MIN = 7


def is_bound_shipped(bctx: str, value: float, decimals: int) -> bool:
    """The v0.6.2 obligation predicate, minus the n= self-scope (which needs `pre`)."""
    if _TRIGGERS.search(bctx):
        return True
    return bool(decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx))


def resolvable_docs():
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
            yield doc, receipts


def main() -> int:
    pool, by_dec = [], defaultdict(lambda: {"unbound_match": 0, "unbound_nomatch": 0,
                                            "bound_match": 0, "bound_nomatch": 0})
    accusation_rows = []
    n_docs = 0
    for doc, receipts in resolvable_docs():
        n_docs += 1
        text = doc.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rvals = []
        for rp in receipts:
            try:
                j = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                continue
            for path, v in receipt_values(j):
                rvals.append((rp.name, path, v))
        try:
            live = certify_doc(doc, receipts)
        except Exception:
            continue
        status_by = {(e["line"], e["token"]): e["status"] for e in live["ledger"]}
        rel = doc.relative_to(ROOT).as_posix()
        for e in extract_numbers(text):
            d = e["decimals"]
            if d == 0:
                continue
            bctx = e.get("binding_context", e["context"])
            ctx_line = lines[e["line"] - 1] if e["line"] <= len(lines) else ""
            allow = "%" in ctx_line or re.search(r"\bpercent", ctx_line, re.I) is not None
            matched = any(_match(e["value"], d, rv, allow) for _, _, rv in rvals)
            bound = is_bound_shipped(bctx, e["value"], d)
            by_dec[d][("bound" if bound else "unbound") + ("_match" if matched else "_nomatch")] += 1
            if not bound and not matched and d >= 5:
                accusation_rows.append({"doc": rel, "line": e["line"], "token": e["token"],
                                        "decimals": d, "status": status_by.get((e["line"], e["token"]), "?"),
                                        "context": bctx[:200]})
            if d >= FULLPREC_MIN:
                pool.append({"doc": rel, "line": e["line"], "token": e["token"], "decimals": d,
                             "bound": bound, "value_matches": matched,
                             "status": status_by.get((e["line"], e["token"]), "?")})

    unbound = [p for p in pool if not p["bound"]]
    dsorted = sorted(by_dec)
    cumulative = {}
    for T in dsorted:
        cumulative[str(T)] = {
            "tamper_surface_unbound_match": sum(by_dec[d]["unbound_match"] for d in dsorted if d >= T),
            "clean_accusations_unbound_nomatch": sum(by_dec[d]["unbound_nomatch"] for d in dsorted if d >= T),
        }

    report = {
        "note": "pre-fix census for PREREG_oath_v07_precision_obligation_2026_08_22",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "docs_with_resolvable_receipts": n_docs,
        "full_precision_pool": len(pool),
        "full_precision_unbound": len(unbound),
        "full_precision_unbound_share": round(len(unbound) / max(len(pool), 1), 4),
        "unbound_status_breakdown": dict(Counter(p["status"] for p in unbound)),
        "unbound_value_matching": sum(1 for p in unbound if p["value_matches"]),
        "per_decimal": {str(d): dict(by_dec[d]) for d in dsorted},
        "cumulative_rule_decimals_ge": cumulative,
        "clean_accusation_candidates_ge5": accusation_rows,
        "pool": pool,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"docs with resolvable receipts: {n_docs}")
    print(f"full-precision pool: {len(pool)}   unbound: {len(unbound)} "
          f"({report['full_precision_unbound_share']})")
    print(f"unbound status: {report['unbound_status_breakdown']}")
    print(f"unbound already value-matching a receipt: {report['unbound_value_matching']}")
    print(f"\n{'T':>3} {'tamper surface':>15} {'clean accusations':>19}")
    for T in dsorted:
        c = cumulative[str(T)]
        print(f"{T:>3} {c['tamper_surface_unbound_match']:>15} "
              f"{c['clean_accusations_unbound_nomatch']:>19}")
    print(f"\nclean accusation candidates at >=5 decimals: {len(accusation_rows)}")
    for r in accusation_rows:
        print(f"  {r['doc']} L{r['line']} {r['token']} ({r['decimals']}dp, {r['status']})")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
