"""OATH v0.8 context sweep — is the float-binding cost a NAMING problem or a WINDOW problem?

The predicate sweep (`oath_v08_fieldbind_sweep.py`) found no KEEP-widening that breaks the
cost/benefit frontier: every widening buys back honest verifications and gives up kills at roughly
the same rate. Hand-inspection of the demoted clean claims points at a different variable — the
binding context is LINE-LOCAL, while prose names the field a sentence earlier:

    "...the agent's cave rate. It fell from 0.9132 to 0.62 -- nowhere near gone"
      leaf: competent_agent_result.json:cave_rate_3b_agent      (a correct binding)
      line: carries neither "cave" nor "rate"                   (the stem test cannot see it)

So this sweeps the WINDOW rather than the predicate: how far back the binding context reaches
(previous N lines, the enclosing paragraph, the nearest markdown heading), scored on both
populations at once. Widening is not free in one direction only — a coincidental leaf is usually
drawn from the same topically-related receipt, so extra stems can rescue mutants too. That is
exactly what this measures instead of assuming.

  python papers/closed-model-frontier/oath_v08_fieldbind_ctxsweep.py
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

OUT = HERE / "oath_v08_fieldbind_ctxsweep.json"
V07_CENSUS = HERE / "oath_v07_silentpass_census.json"
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def stems_of(bctx: str) -> set[str]:
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
    return {w[:4] for w in words} | {s[:4] for w in words
                                     for s in re.split(r"[-_]", w) if len(s) >= 3}


def path_ok(path: str, stems: set[str]) -> bool:
    segs = {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}
    return bool({s[:4] for s in segs if len(s) >= 3} & stems)


def window(lines: list[str], ln: int, mode: str, base: str) -> str:
    """*ln* is 1-based. Returns the binding context under the named window mode."""
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
    if mode == "head":
        h = ""
        for j in range(i - 1, -1, -1):
            m = _HEADING.match(lines[j])
            if m:
                h = m.group(1)
                break
        return (h + " " + base)[:800]
    if mode == "para+head":
        j = i
        while j > 0 and lines[j - 1].strip():
            j -= 1
        h = ""
        for k in range(j - 1, -1, -1):
            m = _HEADING.match(lines[k])
            if m:
                h = m.group(1)
                break
        return (h + " " + " ".join(lines[j:i] + [base]))[:800]
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


MODES = ["line", "prev1", "prev2", "prev3", "para", "head", "para+head"]


def collect():
    docs = resolvable_docs()
    doc_index = {d.name: (d, rc) for d, rc in docs}
    rv_cache = {}

    def rv(dname):
        if dname not in rv_cache:
            rv_cache[dname] = rvals_for(doc_index[dname][1])
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
        for num in extract_numbers(text):
            if num["decimals"] == 0 or (num["line"], num["token"]) not in verified:
                continue
            ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
            base_b = num.get("binding_context", ctx)
            tok_at = ctx.find(num["token"])
            pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
            allow = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
            hits = [(rn, pth) for rn, pth, v in rv(doc.name)
                    if _match(num["value"], num["decimals"], v, allow)]
            if not hits:
                continue
            clean.append({
                "doc": doc.name, "line": num["line"], "token": num["token"],
                "decimals": num["decimals"], "hits": hits,
                "obligated": is_obligated(base_b, pre, num["value"], num["decimals"]),
                "ctx": {m: window(doc_lines, num["line"], m, base_b) for m in MODES},
            })
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
    return clean, mut


def emptied(rows, mode, dec_max):
    out = []
    for r in rows:
        if dec_max is not None and r["decimals"] > dec_max:
            continue
        stems = stems_of(r["ctx"][mode])
        if not any(path_ok(h[1], stems) for h in r["hits"]):
            out.append(r)
    return out


def main() -> int:
    t0 = time.time()
    clean, mut = collect()
    print(f"\nclean float VERIFIED with hits: {len(clean)}   "
          f"float false-VERIFIED mutants: {len(mut)}\n")

    table = []
    print(f"{'window':12s} {'dec<=':>5s} {'kill':>5s} {'cost':>5s} {'accuse':>6s} {'ratio':>6s}")
    print("-" * 48)
    for mode in MODES:
        for dec_max in (None, 4, 3, 2):
            k = emptied(mut, mode, dec_max)
            c = emptied(clean, mode, dec_max)
            a = [r for r in c if r["obligated"]]
            ratio = len(c) / max(len(k), 1)
            table.append({"window": mode, "dec_max": dec_max, "killed": len(k),
                          "demoted": len(c), "would_accuse": len(a),
                          "cost_per_kill": round(ratio, 2)})
            dm = "all" if dec_max is None else str(dec_max)
            print(f"{mode:12s} {dm:>5s} {len(k):5d} {len(c):5d} {len(a):6d} {ratio:6.2f}")

    OUT.write_text(json.dumps({
        "note": "context-window sweep for PREREG_oath_v08_float_field_binding_2026_08_23",
        "clean_float_verified_with_hits": len(clean),
        "float_false_verified_mutants": len(mut),
        "sweep": table,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
