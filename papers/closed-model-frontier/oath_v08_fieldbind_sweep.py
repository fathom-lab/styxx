"""OATH v0.8 predicate sweep — which KEEP rule makes float field binding pay?

The naked v0.3/v0.6.2 stem test, promoted to status level, kills 165 of the 330 FLOAT
false-attestations in `oath_v07_silentpass_census.json` but demotes 575 of 3037 honest float
verifications and would ACCUSE 203 of them. It cannot ship in that form.

Inspection of the demoted clean claims names the reason: they are dominated by legitimate
bar/floor/threshold claims whose receipt field is an ACRONYM (`frozen_gates.CG1_SEP` grounding a
line that reads "floor 0.10"), which shares no word stem with the prose. That is a naming
accident, not a binding failure.

This sweeps candidate KEEP-widenings over BOTH populations at once and prints the cost/benefit
frontier, so the shipped predicate is chosen against measured structure rather than taste. It
writes a result JSON and touches nothing else; `styxx/certify.py` is not imported for any flag
that does not exist yet.

  python papers/closed-model-frontier/oath_v08_fieldbind_sweep.py
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
                           _match, _TRIGGERS, _TRIGGERS_CORR)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v08_fieldbind_sweep.json"
V07_CENSUS = HERE / "oath_v07_silentpass_census.json"

# a receipt container whose leaves ARE specifications: the doc's bar/floor/threshold prose will
# rarely share a stem with the acronym field name underneath it.
_SPEC_CONTAINER = re.compile(r"\b(frozen_gates?|gates?|bars?|floors?|thresholds?|prereg|"
                             r"kill_gates?|criteria|spec)\b", re.I)
# a leaf whose final segment carries no field identity — binding to it is impossible in principle.
_GENERIC_LEAF = re.compile(r"^(value|val|mean|median|score|rate|result|n|x|y|total|sum|avg|"
                           r"\$|\d+)$", re.I)


def stems_of(bctx: str) -> set[str]:
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
    return {w[:4] for w in words} | {s[:4] for w in words
                                     for s in re.split(r"[-_]", w) if len(s) >= 3}


def segs_of(path: str) -> set[str]:
    return {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}


def leaf_of(path: str) -> str:
    tail = re.split(r"[.\[\]]", path.rstrip("]"))
    return next((t for t in reversed(tail) if t), "$")


def keep_predicates():
    """(name, fn(path, receipt_name, stems, bctx) -> bool). Each is a KEEP widening of W1."""
    def w1(path, rn, stems, bctx):
        return bool({s[:4] for s in segs_of(path) if len(s) >= 3} & stems)

    def spec_container(path, rn, stems, bctx):
        return bool(_SPEC_CONTAINER.search(path))

    def spec_prose(path, rn, stems, bctx):
        # the CLAIM announces itself as a bar/floor and the LEAF sits in a spec container
        return bool(_SPEC_CONTAINER.search(path)) and bool(
            re.search(r"\b(bar|bars|floor|floors|gate|gates|threshold|clears?|cleared|"
                      r"against|below|above|exceeds?|misses?|missed)\b", bctx, re.I))

    def generic_leaf(path, rn, stems, bctx):
        return bool(_GENERIC_LEAF.match(leaf_of(path)))

    def receipt_name(path, rn, stems, bctx):
        rstems = {s[:4] for s in re.split(r"[^A-Za-z]+", rn) if len(s) >= 3}
        return bool(rstems & stems)

    return {
        "W1_stem_only": [w1],
        "W2_+spec_container": [w1, spec_container],
        "W3_+spec_container_and_prose": [w1, spec_prose],
        "W4_+generic_leaf": [w1, generic_leaf],
        "W5_+spec_prose+generic_leaf": [w1, spec_prose, generic_leaf],
        "W6_+spec_container+generic_leaf": [w1, spec_container, generic_leaf],
        "W7_+spec_container+generic+receipt": [w1, spec_container, generic_leaf, receipt_name],
    }


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


def is_obligated(bctx, pre, value, decimals):
    if _TRIGGERS.search(bctx) or re.search(r"\bn\s*=\s*$", pre, re.I):
        return True
    if decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx):
        return True
    return decimals >= 7


def collect():
    """Both populations, each row carrying its FULL hit list so predicates can be swept offline."""
    docs = resolvable_docs()
    doc_index = {d.name: (d, rc) for d, rc in docs}
    rv_cache = {}

    def rv(dname):
        if dname not in rv_cache:
            rv_cache[dname] = rvals_for(doc_index[dname][1])
        return rv_cache[dname]

    clean = []
    for i, (doc, receipts) in enumerate(docs, 1):
        try:
            base = certify_doc(doc, receipts)
        except Exception:
            continue
        verified = {(e["line"], e["token"]) for e in base["ledger"] if e["status"] == "VERIFIED"}
        text = doc.read_text(encoding="utf-8", errors="replace")
        doc_lines = text.splitlines()
        for num in extract_numbers(text):
            if num["decimals"] == 0 or (num["line"], num["token"]) not in verified:
                continue
            ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
            bctx = num.get("binding_context", ctx)
            tok_at = ctx.find(num["token"])
            pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
            allow = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
            hits = [(rn, pth) for rn, pth, v in rv(doc.name)
                    if _match(num["value"], num["decimals"], v, allow)]
            if hits:
                clean.append({"doc": doc.name, "line": num["line"], "token": num["token"],
                              "decimals": num["decimals"], "bctx": bctx[:320],
                              "obligated": is_obligated(bctx, pre, num["value"], num["decimals"]),
                              "hits": hits})
        if i % 40 == 0:
            print(f"  clean [{i}/{len(docs)}] {len(clean)}", flush=True)

    v07 = json.loads(V07_CENSUS.read_text(encoding="utf-8"))
    mut = []
    for r in v07["rows"]:
        if r["status"] != "VERIFIED" or r["decimals"] == 0 or r["doc"] not in doc_index:
            continue
        doc, receipts = doc_index[r["doc"]]
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        ln = r["line"] - 1
        if ln >= len(lines) or r["token"] not in lines[ln]:
            continue
        ml = list(lines)
        ml[ln] = ml[ln].replace(r["token"], r["mutant"], 1)
        ctx = ml[ln].strip().replace("−", "-")
        bctx = next((e.get("binding_context") for e in extract_numbers("\n".join(ml))
                     if e["line"] == r["line"] and e["token"] == r["mutant"]), None) or ctx
        allow = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
        try:
            mval = float(r["mutant"].replace(",", ""))
        except ValueError:
            continue
        hits = [(rn, pth) for rn, pth, v in rv(r["doc"])
                if _match(mval, r["decimals"], v, allow)]
        if hits:
            mut.append({"doc": r["doc"], "line": r["line"], "token": r["token"],
                        "mutant": r["mutant"], "decimals": r["decimals"],
                        "bctx": bctx[:320], "hits": hits})
    return clean, mut


def score(rows, preds, dec_max=None):
    """Rows whose hit set is EMPTIED by the composed keep-rule."""
    out = []
    for r in rows:
        if dec_max is not None and r["decimals"] > dec_max:
            continue
        stems = stems_of(r["bctx"])
        kept = [h for h in r["hits"]
                if any(p(h[1], h[0], stems, r["bctx"]) for p in preds)]
        if not kept:
            out.append(r)
    return out


def main() -> int:
    t0 = time.time()
    clean, mut = collect()
    print(f"\nclean float VERIFIED with hits: {len(clean)}   "
          f"float false-VERIFIED mutants: {len(mut)}\n")

    table = []
    print(f"{'predicate':38s} {'dec<=':>5s} {'kill':>5s} {'cost':>5s} {'accuse':>6s} {'ratio':>6s}")
    print("-" * 72)
    for name, preds in keep_predicates().items():
        for dec_max in (None, 4, 3, 2):
            killed = score(mut, preds, dec_max)
            demoted = score(clean, preds, dec_max)
            accuse = [r for r in demoted if r["obligated"]]
            ratio = len(demoted) / max(len(killed), 1)
            row = {"predicate": name, "dec_max": dec_max, "killed": len(killed),
                   "demoted": len(demoted), "would_accuse": len(accuse),
                   "cost_per_kill": round(ratio, 2)}
            table.append(row)
            dm = "all" if dec_max is None else f"{dec_max}"
            print(f"{name:38s} {dm:>5s} {len(killed):5d} {len(demoted):5d} "
                  f"{len(accuse):6d} {ratio:6.2f}")

    OUT.write_text(json.dumps({
        "note": "predicate sweep for PREREG_oath_v08_float_field_binding_2026_08_23",
        "clean_float_verified_with_hits": len(clean),
        "float_false_verified_mutants": len(mut),
        "sweep": table,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
