"""OATH v0.8 — the NAMEABLE gate: demote only where binding was POSSIBLE and failed.

Both prior sweeps stall at roughly one honest verification demoted per false attestation removed
(`oath_v08_fieldbind_sweep.py`: no KEEP-widening beats 1.2; `oath_v08_fieldbind_ctxsweep.py`: the
best window reaches 1.04). Neither is worth shipping on its own, and every point of both sweeps
would ACCUSE dozens of honest claims, which kills the accusing design outright.

Both sweeps share one blind spot. They demote a claim whenever no matching leaf's PATH shares a
stem with its context — without ever asking whether the receipts contain such a path AT ALL. Two
very different situations collapse together:

  UNNAMEABLE  the line's vocabulary names nothing in the receipt set (an acronym field like
              `frozen_gates.CG1_SEP` under a line reading "floor 0.10"). Binding is impossible in
              principle, so a demotion here is a pure coverage loss and buys nothing.

  NAMEABLE    the receipts DO carry a leaf whose path matches the line's vocabulary -- the claim
              simply is not that leaf's value. This is the real thing claim->field binding was
              always supposed to catch: the doc says "AUC 0.83", the receipt's `auroc` field says
              0.82, and the 0.83 grounds only in some unrelated leaf that happens to hold it.

The NAMEABLE test reads receipt PATHS and doc CONTEXT. It never reads the claim's value, so unlike
the "path overlap AND value-match" candidate that died in v0.7 it does NOT evaporate under the
mutation it exists to catch.

  python papers/closed-model-frontier/oath_v08_fieldbind_nameable.py
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

OUT = HERE / "oath_v08_fieldbind_nameable.json"
V07_CENSUS = HERE / "oath_v07_silentpass_census.json"
MODES = ["line", "prev1", "prev2", "para"]


def stems_of(bctx: str) -> set[str]:
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
    return {w[:4] for w in words} | {s[:4] for w in words
                                     for s in re.split(r"[-_]", w) if len(s) >= 3}


def path_stems(path: str) -> set[str]:
    segs = {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}
    return {s[:4] for s in segs if len(s) >= 3}


def window(lines, ln, mode, base):
    i = ln - 1
    if mode == "line":
        return base
    if mode.startswith("prev"):
        k = int(mode[4:])
        return " ".join(lines[max(0, i - k):i] + [base])[:800]
    if mode == "para":
        j = i
        while j > 0 and lines[j - 1].strip():
            j -= 1
        return " ".join(lines[j:i] + [base])[:800]
    raise ValueError(mode)


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
    docs = resolvable_docs()
    doc_index = {d.name: (d, rc) for d, rc in docs}
    rv_cache, ps_cache = {}, {}

    def rv(dname):
        if dname not in rv_cache:
            rv_cache[dname] = rvals_for(doc_index[dname][1])
            # every path stem present anywhere in this doc's receipt set
            ps_cache[dname] = set().union(*(path_stems(p) for _, p, _ in rv_cache[dname])) \
                if rv_cache[dname] else set()
        return rv_cache[dname]

    clean, mut = [], []
    for i, (doc, receipts) in enumerate(docs, 1):
        try:
            base = certify_doc(doc, receipts)
        except Exception:
            continue
        verified = {(e["line"], e["token"]) for e in base["ledger"] if e["status"] == "VERIFIED"}
        text = doc.read_text(encoding="utf-8", errors="replace")
        doc_lines = text.splitlines()
        rvs = rv(doc.name)
        for num in extract_numbers(text):
            if num["decimals"] == 0 or (num["line"], num["token"]) not in verified:
                continue
            ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
            base_b = num.get("binding_context", ctx)
            tok_at = ctx.find(num["token"])
            pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
            allow = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
            hits = [(rn, pth) for rn, pth, v in rvs
                    if _match(num["value"], num["decimals"], v, allow)]
            if not hits:
                continue
            clean.append({"doc": doc.name, "line": num["line"], "token": num["token"],
                          "decimals": num["decimals"], "hits": hits,
                          "obligated": is_obligated(base_b, pre, num["value"], num["decimals"]),
                          "ctx": {m: window(doc_lines, num["line"], m, base_b) for m in MODES}})
        if i % 40 == 0:
            print(f"  clean [{i}/{len(docs)}] {len(clean)}", flush=True)

    v07 = json.loads(V07_CENSUS.read_text(encoding="utf-8"))
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
        base_b = next((e.get("binding_context") for e in extract_numbers("\n".join(ml))
                       if e["line"] == r["line"] and e["token"] == r["mutant"]), None) or ctx
        allow = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
        try:
            mval = float(r["mutant"].replace(",", ""))
        except ValueError:
            continue
        hits = [(rn, pth) for rn, pth, v in rv(r["doc"])
                if _match(mval, r["decimals"], v, allow)]
        if not hits:
            continue
        mut.append({"doc": r["doc"], "line": r["line"], "token": r["token"],
                    "mutant": r["mutant"], "decimals": r["decimals"], "hits": hits,
                    "ctx": {m: window(ml, r["line"], m, base_b) for m in MODES}})
    return clean, mut, ps_cache


def demoted(rows, mode, dec_max, ps_cache, nameable_required: bool):
    out = []
    for r in rows:
        if dec_max is not None and r["decimals"] > dec_max:
            continue
        stems = stems_of(r["ctx"][mode])
        if any(path_stems(h[1]) & stems for h in r["hits"]):
            continue                                    # a matching leaf IS stem-bound: keep
        if nameable_required and not (ps_cache.get(r["doc"], set()) & stems):
            continue                                    # UNNAMEABLE: binding impossible, keep
        out.append(r)
    return out


def main() -> int:
    t0 = time.time()
    clean, mut, ps_cache = collect()
    print(f"\nclean float VERIFIED with hits: {len(clean)}   "
          f"float false-VERIFIED mutants: {len(mut)}\n")

    table = []
    print(f"{'window':7s} {'nameable':>9s} {'dec<=':>5s} {'kill':>5s} {'cost':>5s} "
          f"{'accuse':>6s} {'ratio':>6s}")
    print("-" * 52)
    for mode in MODES:
        for need in (False, True):
            for dec_max in (None, 4, 3, 2):
                k = demoted(mut, mode, dec_max, ps_cache, need)
                c = demoted(clean, mode, dec_max, ps_cache, need)
                a = [r for r in c if r["obligated"]]
                ratio = len(c) / max(len(k), 1)
                table.append({"window": mode, "nameable_required": need, "dec_max": dec_max,
                              "killed": len(k), "demoted": len(c), "would_accuse": len(a),
                              "cost_per_kill": round(ratio, 2)})
                dm = "all" if dec_max is None else str(dec_max)
                print(f"{mode:7s} {str(need):>9s} {dm:>5s} {len(k):5d} {len(c):5d} "
                      f"{len(a):6d} {ratio:6.2f}")

    OUT.write_text(json.dumps({
        "note": "NAMEABLE-gate sweep for PREREG_oath_v08_float_field_binding_2026_08_23",
        "clean_float_verified_with_hits": len(clean),
        "float_false_verified_mutants": len(mut),
        "sweep": table,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
