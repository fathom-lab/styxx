"""OATH v0.8 pre-fix census — can claim->field binding for FLOATS be made to pay?

The standing v0.4 debt: `certify_doc` filters an INTEGER claim's value-matches to leaves whose
receipt PATH shares a 4-char word stem with the claim's binding context (the v0.3 count-binding
rule), and deliberately exempts floats — "Floats keep value-only matching (v0.4 owes them full
claim->field binding)". v0.6.2 applies the identical stem test to floats but ATTRIBUTION-ONLY:
it may reorder `hits`, never change a status.

This measures, at the SHIPPED verifier and before any edit, the two quantities that decide whether
promoting that test to status level is worth doing:

  BENEFIT  of the false-VERIFIED mutants in `oath_v07_silentpass_census.json` (a mutated claim that
           coincidentally matches an UNRELATED receipt leaf -- affirmative false attestation), how
           many are FLOATS, and of those how many lose every hit under the stem filter.

  COST     of the float claims the clean corpus currently certifies VERIFIED, how many would lose
           every hit under the same filter -- split by whether the claim is OBLIGATED, because an
           obligated claim whose hits empty becomes an ACCUSATION (UNGROUNDED) while an unobligated
           one merely falls to ABSTAIN.

That split is the design question. A filter that accuses cannot ship if it accuses honest
documents; a filter that only ever DEMOTES to ABSTAIN cannot produce a false accusation by
construction, but converts false attestation into silence rather than into a catch.

Nothing here imports a flag that does not exist yet and nothing writes to the corpus: the hit
computation is replicated from `certify_doc` using its own `_match` / `receipt_values`, and the
only file written is this script's result JSON.

  python papers/closed-model-frontier/oath_v08_fieldbind_precfix_census.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (certify_doc, extract_numbers, receipt_values,   # noqa: E402
                           _match, _TRIGGERS, _TRIGGERS_CORR)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v08_fieldbind_precfix_census.json"
V07_CENSUS = HERE / "oath_v07_silentpass_census.json"


def stems_of(bctx: str) -> set[str]:
    """The v0.3/v0.6.2 stem set of a claim's binding context, verbatim."""
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
    return {w[:4] for w in words} | {s[:4] for w in words
                                     for s in re.split(r"[-_]", w) if len(s) >= 3}


def path_stem_ok(path: str, stems: set[str]) -> bool:
    """The v0.6.2 `_stem_ok` predicate, verbatim."""
    segs = {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}
    return bool({s[:4] for s in segs if len(s) >= 3} & stems)


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


def rvals_for(receipts) -> list[tuple[str, str, float]]:
    rv = []
    for rp in receipts:
        j = json.loads(rp.read_text(encoding="utf-8"))
        for path, v in receipt_values(j):
            rv.append((rp.name, path, v))
    return rv


def is_obligated(bctx: str, pre: str, value: float, decimals: int) -> bool:
    """The SHIPPED obligation predicate (trigger / n= / fractional-correlation / precision)."""
    if _TRIGGERS.search(bctx) or re.search(r"\bn\s*=\s*$", pre, re.I):
        return True
    if decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx):
        return True
    return decimals >= 7


def claim_rows(doc: Path, receipts, rvals):
    """Every float claim in *doc*, with its hits and the stem verdict on those hits."""
    text = doc.read_text(encoding="utf-8", errors="replace")
    doc_lines = text.splitlines()
    rows = []
    for num in extract_numbers(text):
        if num["decimals"] == 0:
            continue
        ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
        bctx = num.get("binding_context", ctx)
        tok_at = ctx.find(num["token"])
        pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
        allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
        hits = [(rn, pth) for rn, pth, rv in rvals
                if _match(num["value"], num["decimals"], rv, allow_scaling)]
        if not hits:
            continue
        stems = stems_of(bctx)
        kept = [(rn, pth) for rn, pth in hits if path_stem_ok(pth, stems)]
        rows.append({
            "doc": doc.name, "line": num["line"], "token": num["token"],
            "decimals": num["decimals"], "n_hits": len(hits), "n_kept": len(kept),
            "survives": bool(kept), "obligated": is_obligated(bctx, pre, num["value"],
                                                              num["decimals"]),
            "ref": f"{hits[0][0]}:{hits[0][1]}",
            "context": ctx[:150],
        })
    return rows


def main() -> int:
    t0 = time.time()
    docs = resolvable_docs()
    print(f"docs with fully-resolvable receipts: {len(docs)}", flush=True)

    # ---------------- COST: clean-corpus float claims that currently VERIFY
    cost_rows = []
    for i, (doc, receipts) in enumerate(docs, 1):
        try:
            base = certify_doc(doc, receipts)
        except Exception:
            continue
        verified = {(e["line"], e["token"]) for e in base["ledger"] if e["status"] == "VERIFIED"}
        rvals = rvals_for(receipts)
        for r in claim_rows(doc, receipts, rvals):
            if (r["line"], r["token"]) in verified:
                cost_rows.append(r)
        if i % 25 == 0:
            print(f"  [{i}/{len(docs)}] clean float claims: {len(cost_rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    demoted = [r for r in cost_rows if not r["survives"]]
    demoted_accusing = [r for r in demoted if r["obligated"]]
    demoted_silent = [r for r in demoted if not r["obligated"]]

    # ---------------- BENEFIT: replay the v0.7 false-VERIFIED mutants through the filter
    v07 = json.loads(V07_CENSUS.read_text(encoding="utf-8"))
    fv = [r for r in v07["rows"] if r["status"] == "VERIFIED"]
    fv_float = [r for r in fv if r["decimals"] > 0]
    by_doc = {}
    for r in fv_float:
        by_doc.setdefault(r["doc"], []).append(r)

    doc_index = {d.name: (d, rc) for d, rc in docs}
    benefit_rows, unreplayed = [], 0
    for dname, rows in sorted(by_doc.items()):
        if dname not in doc_index:
            unreplayed += len(rows)
            continue
        doc, receipts = doc_index[dname]
        rvals = rvals_for(receipts)
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for r in rows:
            ln = r["line"] - 1
            if ln >= len(lines):
                unreplayed += 1
                continue
            # rebuild the mutated line exactly as the v0.7 census did
            if r["token"] not in lines[ln]:
                unreplayed += 1
                continue
            ml = list(lines)
            ml[ln] = ml[ln].replace(r["token"], r["mutant"], 1)
            ctx = ml[ln].strip().replace("−", "-")
            hdr = next((e.get("binding_context") for e in extract_numbers("\n".join(ml))
                        if e["line"] == r["line"] and e["token"] == r["mutant"]), None)
            bctx = hdr or ctx
            allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
            try:
                mval = float(r["mutant"].replace(",", ""))
            except ValueError:
                unreplayed += 1
                continue
            hits = [(rn, pth) for rn, pth, rv in rvals
                    if _match(mval, r["decimals"], rv, allow_scaling)]
            stems = stems_of(bctx)
            kept = [(rn, pth) for rn, pth in hits if path_stem_ok(pth, stems)]
            benefit_rows.append({
                "doc": dname, "line": r["line"], "token": r["token"], "mutant": r["mutant"],
                "decimals": r["decimals"], "n_hits": len(hits), "n_kept": len(kept),
                "still_verifies": bool(kept),
                "ref": f"{hits[0][0]}:{hits[0][1]}" if hits else None,
                "context": ctx[:150],
            })

    killed = [r for r in benefit_rows if not r["still_verifies"]]

    report = {
        "note": "pre-fix census for PREREG_oath_v08_float_field_binding_2026_08_23 "
                "(status-level claim->field binding for FLOAT claims, the standing v0.4 debt)",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "docs": len(docs),
        "v07_census_source": V07_CENSUS.name,
        "v07_false_verified_total": len(fv),
        "v07_false_verified_integer": len(fv) - len(fv_float),
        "v07_false_verified_float": len(fv_float),
        "benefit": {
            "replayed": len(benefit_rows),
            "unreplayed": unreplayed,
            "killed_by_stem_filter": len(killed),
            "kill_share": round(len(killed) / max(len(benefit_rows), 1), 4),
            "survivors": len(benefit_rows) - len(killed),
        },
        "cost": {
            "clean_float_claims_verified": len(cost_rows),
            "demoted_by_stem_filter": len(demoted),
            "demote_share": round(len(demoted) / max(len(cost_rows), 1), 4),
            "demoted_obligated_would_ACCUSE": len(demoted_accusing),
            "demoted_unobligated_would_ABSTAIN": len(demoted_silent),
            "docs_touched": len({r["doc"] for r in demoted}),
            "demoted_by_decimals": {str(k): v for k, v in
                                    sorted(Counter(r["decimals"] for r in demoted).items())},
        },
        "demoted_rows": demoted,
        "benefit_rows": benefit_rows,
        "cost_rows": cost_rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    b, c = report["benefit"], report["cost"]
    print(f"\nv0.7 false-VERIFIED: {len(fv)}  "
          f"(integer {len(fv)-len(fv_float)} | float {len(fv_float)})")
    print(f"BENEFIT  replayed {b['replayed']}  killed by stem filter {b['killed_by_stem_filter']} "
          f"({b['kill_share']})  survivors {b['survivors']}  unreplayed {b['unreplayed']}")
    print(f"COST     clean float VERIFIED {c['clean_float_claims_verified']}  "
          f"demoted {c['demoted_by_stem_filter']} ({c['demote_share']})")
    print(f"           would ACCUSE (obligated): {c['demoted_obligated_would_ACCUSE']}")
    print(f"           would ABSTAIN (unobligated): {c['demoted_unobligated_would_ABSTAIN']}")
    print(f"           across {c['docs_touched']} documents")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
