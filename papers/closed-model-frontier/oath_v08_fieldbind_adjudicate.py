"""OATH v0.8 — what IS a clean-corpus demotion? Evidence dossier for hand-adjudication.

Every design family swept so far stalls at roughly one honest float verification demoted per
false attestation removed. Whether that trade is worth making depends entirely on a question the
ratio cannot answer: when the clause demotes a claim on the CLEAN corpus, is it destroying a
correct verification, or is it withdrawing a verification that was never earned?

For a demoted claim the situation is always:
  * the claim's value matches some receipt leaf, but no matching leaf's path relates to its context;
  * the receipts DO carry a leaf whose path the claim's context names (the NAMEABLE gate).

So the decisive evidence is the value of THAT named leaf. Three outcomes:

  COINCIDENCE   the named leaf exists and holds a DIFFERENT value -> the claim was grounded in an
                unrelated leaf by accident. Demotion is a CORRECTION, not a loss; the doc may also
                carry a real provenance gap worth repairing.
  GENUINE       the named leaf holds the claim's own value, and the stem test simply failed to
                connect them (acronym field, cross-paragraph reference) -> demotion is a true
                coverage loss.
  SPEC          the claim is a bar/floor/threshold whose receipt is the prereg, not a measurement
                -> it should never have been VERIFIED at all; demotion is a correction.

This emits the dossier: claim, context window, every matching leaf, and every receipt leaf the
context NAMES together with its value. Adjudication is done by hand against the frozen definition
in the prereg; this script judges nothing.

  python papers/closed-model-frontier/oath_v08_fieldbind_adjudicate.py [N]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (certify_doc, extract_numbers, receipt_values,   # noqa: E402
                           _match)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v08_fieldbind_dossier.json"
WINDOW = "prev1"
DEC_MAX = 3


def stems_of(bctx):
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
    return {w[:4] for w in words} | {s[:4] for w in words
                                     for s in re.split(r"[-_]", w) if len(s) >= 3}


def path_stems(path):
    segs = {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}
    return {s[:4] for s in segs if len(s) >= 3}


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


def rvals_for(receipts):
    rv = []
    for rp in receipts:
        j = json.loads(rp.read_text(encoding="utf-8"))
        for path, v in receipt_values(j):
            rv.append((rp.name, path, v))
    return rv


def main() -> int:
    t0 = time.time()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    docs = resolvable_docs()
    dossier = []
    for doc, receipts in docs:
        try:
            base = certify_doc(doc, receipts)
        except Exception:
            continue
        verified = {(e["line"], e["token"]) for e in base["ledger"] if e["status"] == "VERIFIED"}
        text = doc.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rvs = rvals_for(receipts)
        for num in extract_numbers(text):
            if num["decimals"] == 0 or num["decimals"] > DEC_MAX:
                continue
            if (num["line"], num["token"]) not in verified:
                continue
            ctx = lines[num["line"] - 1].strip().replace("−", "-")
            base_b = num.get("binding_context", ctx)
            i = num["line"] - 1
            bctx = " ".join(lines[max(0, i - 1):i] + [base_b])[:800]
            allow = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
            hits = [(rn, pth, v) for rn, pth, v in rvs
                    if _match(num["value"], num["decimals"], v, allow)]
            if not hits:
                continue
            stems = stems_of(bctx)
            if any(path_stems(p) & stems for _, p, _ in hits):
                continue
            named = [(rn, p, v) for rn, p, v in rvs if path_stems(p) & stems]
            if not named:
                continue                       # UNNAMEABLE — the clause keeps these
            dossier.append({
                "doc": doc.name, "line": num["line"], "token": num["token"],
                "value": num["value"], "decimals": num["decimals"],
                "context": ctx[:180],
                "prev_line": lines[i - 1].strip()[:180] if i >= 1 else "",
                "matching_leaves": [{"receipt": rn, "path": p, "value": v}
                                    for rn, p, v in hits[:6]],
                "named_leaves": [{"receipt": rn, "path": p, "value": v}
                                 for rn, p, v in named[:8]],
                "named_leaf_holds_claim": any(
                    _match(num["value"], num["decimals"], v, allow) for _, _, v in named),
            })

    OUT.write_text(json.dumps({
        "note": "hand-adjudication dossier for PREREG_oath_v08_float_field_binding_2026_08_23",
        "window": WINDOW, "dec_max": DEC_MAX,
        "total_demotions": len(dossier), "rows": dossier,
    }, indent=2) + "\n", encoding="utf-8")

    holds = sum(1 for r in dossier if r["named_leaf_holds_claim"])
    print(f"clean demotions under {WINDOW}/nameable/dec<={DEC_MAX}: {len(dossier)}")
    print(f"  of these, a NAMED leaf already holds the claim's value: {holds} "
          f"(stem test simply failed to connect them)")
    print(f"  named leaf holds something ELSE: {len(dossier)-holds} "
          f"(candidate COINCIDENCE / real provenance gap)\n")
    for r in dossier[:limit]:
        print(f"{r['doc'][:40]:40s} L{r['line']:<5d} {r['token']:>9s}  "
              f"holds={int(r['named_leaf_holds_claim'])}")
        if r["prev_line"]:
            print(f"    prev: {r['prev_line'][:104]}")
        print(f"    ctx : {r['context'][:104]}")
        print(f"    hit : {r['matching_leaves'][0]['path'][:70]} = "
              f"{r['matching_leaves'][0]['value']}")
        for nl in r["named_leaves"][:3]:
            print(f"    NAME: {nl['path'][:70]} = {nl['value']}")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
