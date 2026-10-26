# -*- coding: utf-8 -*-
"""What the numeral reader v2 moves, measured over every committed certify output.

Population A: every committed `*.certificate.json` and `CERT_*.json` (`git ls-files`; paths through an
`anc` segment skipped, as corpus_audit skips them) whose document exists and whose receipts resolve
(`corpus_audit._resolve_receipts`, search root `papers/`). Population B: every committed `*.capsule.html`,
through `capsule.verify_capsule`. Population C: `papers/charon/charon.log.jsonl`, through
`charon.verify_log`.

Configurations of `styxx.certify`'s V15_* flags: `v1` (master off, the reader at 879f6be9), `default`
(the flags as shipped, RULE_final 0.1), `explicit_list` (default + V15_MAGNITUDE_BARE + V15_FORM_NOISE:
the lead's explicit list, the first build session's default), `sentence_final` (default +
V15_SENTENCE_FINAL), `all_rules` (every rule but the pair: the design's `v2_all`), `ranges` (default +
V15_MAGNITUDE_RANGES), `explicit_list_ranges`, and each rule alone with what it needs to act
(`only_<FLAG>`: V15_MAGNITUDE_BARE with V15_MAGNITUDE; V15_MAGNITUDE_RANGES with both). For population
A, every configuration is compared with `v1` field by field (verdict, counts, uncovered, uncovered_items,
uncovered_excluded_by_rule, ledger, epistemics_summary, ungrounded, abstained), and each configuration's
verdict class with the committed certificate's. Populations B and C run under `v1`, `default`,
`explicit_list`, `sentence_final`, `only_V15_MAGNITUDE_BARE` and `only_V15_FORM_NOISE`.

The receipt-binding block is stubbed for population A (it reads git and is not part of the reader);
populations B and C run the real block. The REPLICATIONS.md audit line is re-derived under `v1` and
`default` (`corpus_audit.audit_corpus`, history off).

NOTHING IS WRITTEN OUTSIDE THIS FILE'S OWN RESULT. No committed certificate, capsule, receipt or charon
line is touched; a move is reported, never repaired.

  python papers/closed-model-frontier/oath_v15_numeral_reader_census.py            # all populations
  python papers/closed-model-frontier/oath_v15_numeral_reader_census.py --quick    # population A only
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.capsule as CAP            # noqa: E402
import styxx.certify as C              # noqa: E402
import styxx.charon as CH              # noqa: E402
import styxx.corpus_audit as CA        # noqa: E402

OUT = HERE / "oath_v15_numeral_reader_census_result.json"
SCHEMA = "styxx-oath/numeral-reader-census/v1"
COMPARED = ("verdict", "counts", "uncovered", "uncovered_items", "uncovered_excluded_by_rule", "ledger",
            "epistemics_summary", "ungrounded", "abstained")
ALL = tuple(C._V15_RULE_FLAGS)
SHIPPED = {k: bool(getattr(C, k)) for k in ALL}
EXPLICIT = {**SHIPPED, "V15_MAGNITUDE_BARE": True, "V15_FORM_NOISE": True}
CONFIGS: dict = {
    "v1": (False, SHIPPED),
    "default": (True, SHIPPED),
    "explicit_list": (True, EXPLICIT),
    "sentence_final": (True, {**SHIPPED, "V15_SENTENCE_FINAL": True}),
    "all_rules": (True, {k: (k != "V15_MAGNITUDE_RANGES") for k in ALL}),
    "ranges": (True, {**SHIPPED, "V15_MAGNITUDE_RANGES": True}),
    "explicit_list_ranges": (True, {**EXPLICIT, "V15_MAGNITUDE_RANGES": True}),
}
# a rule acts only with the rules it extends: the bare arm extends the magnitude rule, the pair both
NEEDS = {"V15_MAGNITUDE_BARE": ("V15_MAGNITUDE",), "V15_MAGNITUDE_RANGES": ("V15_MAGNITUDE", "V15_MAGNITUDE_BARE")}
for _k in ALL:
    CONFIGS[f"only_{_k}"] = (True, {k: (k == _k or k in NEEDS.get(_k, ())) for k in ALL})
BC_CONFIGS = ("v1", "default", "explicit_list", "sentence_final", "only_V15_MAGNITUDE_BARE", "only_V15_FORM_NOISE")
STUB = {"schema": "stub", "head": None, "receipts": []}


def configure(name: str) -> None:
    master, flags = CONFIGS[name]
    C.V15_READER_V2 = master
    for k in ALL:
        setattr(C, k, bool(flags.get(k, False)))
    C._v15_reset()


def restore_shipped() -> None:
    C.V15_READER_V2 = True
    for k, v in SHIPPED.items():
        setattr(C, k, v)
    C._v15_reset()


def git_files(*patterns: str) -> list:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", *patterns], capture_output=True, text=True,
                         check=True).stdout.split()
    return sorted(p for p in out if "anc" not in Path(p).parts)


def vclass(v) -> str:
    return str(v or "").split(",")[0].strip()


def js(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def doc_for(cp: Path, cert: dict):
    if cp.name.endswith(".certificate.json"):
        d = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        return d if d.exists() else None
    name = cert.get("document")
    if isinstance(name, str) and cp.with_name(name).exists():
        return cp.with_name(name)
    return None


def suffix_of(row: dict) -> str:
    return (row.get("magnitude") or {}).get("suffix", "")


def population_a() -> list:
    """(rel, cert_path, committed cert, document, receipts) for every certificate whose inputs resolve."""
    rows = []
    for rel in git_files("*.certificate.json", "*CERT_*.json"):
        cp = ROOT / rel
        if not cp.name.endswith(".json"):
            continue
        try:
            cert = json.loads(cp.read_text(encoding="utf-8"))
        except Exception as e:                       # noqa: BLE001
            rows.append({"cert": rel, "status": "UNREADABLE", "why": type(e).__name__})
            continue
        if not isinstance(cert, dict) or "ledger" not in cert:
            rows.append({"cert": rel, "status": "NOT_A_CERTIFICATE"})
            continue
        doc = doc_for(cp, cert)
        if doc is None:
            rows.append({"cert": rel, "status": "MISSING_DOC"})
            continue
        receipts, missing, drift = CA._resolve_receipts(cp, cert, ROOT / "papers")
        row = {"cert": rel, "doc": doc.relative_to(ROOT).as_posix(), "committed_verdict": cert.get("verdict"),
               "n_receipts": len(receipts), "missing_receipts": missing, "receipt_drift": drift}
        if not receipts:
            row["status"] = "NO_RECEIPTS"
        else:
            row["status"] = "OK" if not missing else "INCOMPLETE_RECEIPTS"
            row["_doc"], row["_receipts"] = doc, receipts
        rows.append(row)
    return rows


def compare(base: dict, live: dict) -> dict:
    moved = [k for k in COMPARED if js(base.get(k)) != js(live.get(k))]
    rec = {"moved": moved}
    if not moved:
        return rec
    bk = {(e["line"], e.get("col"), e["token"]): e for e in base["ledger"]}
    lk = {(e["line"], e.get("col"), e["token"]): e for e in live["ledger"]}
    rec.update(
        verdict=live["verdict"], counts=live["counts"], uncovered=live["uncovered"],
        class_flip=vclass(base["verdict"]) != vclass(live["verdict"]),
        rows_added=[{"line": k[0], "col": k[1], "token": k[2], "status": lk[k]["status"],
                     "ref": lk[k].get("receipt_ref"), "suffix": suffix_of(lk[k]), "context": lk[k]["context"][:110]}
                    for k in lk if k not in bk],
        rows_removed=[{"line": k[0], "col": k[1], "token": k[2], "status": bk[k]["status"],
                       "context": bk[k]["context"][:110]} for k in bk if k not in lk],
        rows_status_changed=[{"line": k[0], "token": k[2], "from": bk[k]["status"], "to": lk[k]["status"]}
                             for k in bk if k in lk and bk[k]["status"] != lk[k]["status"]],
        excluded_added=[(u["line"], u["token"], u["reason"]) for u in live["uncovered_excluded_by_rule"]
                        if js(u) not in {js(x) for x in base["uncovered_excluded_by_rule"]}],
        excluded_removed=[(u["line"], u["token"], u["reason"]) for u in base["uncovered_excluded_by_rule"]
                          if js(u) not in {js(x) for x in live["uncovered_excluded_by_rule"]}],
        counted_added=[(u["line"], u["token"], u["reason"]) for u in live["uncovered_items"]
                       if js(u) not in {js(x) for x in base["uncovered_items"]}],
        counted_removed=[(u["line"], u["token"], u["reason"]) for u in base["uncovered_items"]
                         if js(u) not in {js(x) for x in live["uncovered_items"]}],
    )
    return rec


def audit_line() -> str:
    rep = CA.audit_corpus(ROOT / "papers", history="off")
    s = rep["summary"]
    return (f"corpus papers: {s['n_certificates']} certificates | HELD {s['held']}  FAILED {s['failed']}  "
            f"unresolved {s['unresolved']}  verdict-drift {s['verdict_changed']}  receipt-drift {s['receipt_drift']}  "
            f"incomplete {s['incomplete_receipts']}  receipt-changed {s['receipt_changed']}\n"
            f"  uncovered: {s.get('uncovered_spans', 0)} numeric spans across {s.get('uncovered_documents', 0)} "
            f"documents that the extractor never examined (reporting only; no verdict moved)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="population A only")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # the measurement names the commit it ran at; a dirty styxx/ or papers/ tree is said, never hidden
    porcelain = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "styxx", "papers"],
                               capture_output=True, text=True).stdout.splitlines()
    tree_clean = not [ln for ln in porcelain if not ln.endswith(OUT.name)]
    certify_lf = hashlib.sha256((ROOT / "styxx/certify.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    # ---- population A
    rows = population_a()
    ok = [r for r in rows if "_doc" in r]
    real_binding = C._receipt_binding_block
    C._receipt_binding_block = lambda rp, near: STUB
    base_by_cert: dict = {}
    configure("v1")
    for r in ok:
        try:
            base_by_cert[r["cert"]] = C.certify_doc(r["_doc"], r["_receipts"])
        except Exception as e:                   # noqa: BLE001
            r["status"] = "V1_ERROR"
            r["why"] = f"{type(e).__name__}: {str(e)[:120]}"
    ok = [r for r in ok if r["cert"] in base_by_cert]
    per_config: dict = {}
    for name in CONFIGS:
        configure(name)
        t = time.time()
        agg = collections.Counter()
        flips, moved_list = [], []
        by_status_added = collections.Counter()
        mag_rows = collections.Counter()
        mag_by_status = collections.Counter()
        excl_added, excl_removed = collections.Counter(), collections.Counter()
        counted_added, counted_removed = collections.Counter(), collections.Counter()
        uncovered_total, uncovered_docs, committed_differs = 0, 0, 0
        for r in ok:
            base = base_by_cert[r["cert"]]
            try:
                live = C.certify_doc(r["_doc"], r["_receipts"]) if name != "v1" else base
            except Exception as e:               # noqa: BLE001
                agg["errors"] += 1
                moved_list.append({"cert": r["cert"], "error": f"{type(e).__name__}: {str(e)[:160]}"})
                continue
            uncovered_total += live["uncovered"]
            uncovered_docs += 1 if live["uncovered"] else 0
            if vclass(live["verdict"]) != vclass(r["committed_verdict"]):
                committed_differs += 1
            for e in live["ledger"]:
                if "magnitude" in e:
                    mag_rows[suffix_of(e)] += 1
                    mag_by_status[e["status"]] += 1
            rec = compare(base, live)
            if not rec["moved"]:
                continue
            agg["certificates_moved"] += 1
            for k in rec["moved"]:
                agg[f"field_{k}"] += 1
            if rec["class_flip"]:
                agg["class_flips"] += 1
                flips.append({"cert": r["cert"], "doc": r["doc"], "v1": base["verdict"], "live": live["verdict"],
                              "committed": r["committed_verdict"],
                              "accused": [x for x in rec["rows_added"] if x["status"] == "UNGROUNDED"]
                              + [{"line": x["line"], "token": x["token"], "from": x["from"], "to": x["to"]}
                                 for x in rec["rows_status_changed"] if x["to"] == "UNGROUNDED"]})
            agg["rows_added"] += len(rec["rows_added"])
            agg["rows_removed"] += len(rec["rows_removed"])
            agg["rows_status_changed"] += len(rec["rows_status_changed"])
            for x in rec["rows_added"]:
                by_status_added[x["status"]] += 1
            for x in rec["excluded_added"]:
                excl_added[x[2]] += 1
            for x in rec["excluded_removed"]:
                excl_removed[x[2]] += 1
            for x in rec["counted_added"]:
                counted_added[x[2]] += 1
            for x in rec["counted_removed"]:
                counted_removed[x[2]] += 1
            moved_list.append({"cert": r["cert"], "moved": rec["moved"], "class_flip": rec["class_flip"],
                               "v1_verdict": base["verdict"], "verdict": rec["verdict"], "v1_counts": base["counts"],
                               "counts": rec["counts"], "rows_added": rec["rows_added"],
                               "rows_removed": rec["rows_removed"], "rows_status_changed": rec["rows_status_changed"],
                               "excluded_added": rec["excluded_added"], "excluded_removed": rec["excluded_removed"],
                               "counted_added": rec["counted_added"], "counted_removed": rec["counted_removed"]})
        master, flags = CONFIGS[name]
        per_config[name] = {
            "flags": {"V15_READER_V2": master, **{k: bool(flags.get(k, False)) for k in ALL}},
            "certificates": len(ok), **dict(agg),
            "class_flips_list": flips,
            "rows_added_by_status": dict(by_status_added),
            "magnitude_rows": {"total": sum(mag_rows.values()), "by_suffix": dict(mag_rows), "by_status": dict(mag_by_status)},
            "excluded_added_by_reason": dict(excl_added), "excluded_removed_by_reason": dict(excl_removed),
            "counted_added_by_reason": dict(counted_added), "counted_removed_by_reason": dict(counted_removed),
            "uncovered_total": uncovered_total, "uncovered_documents": uncovered_docs,
            "committed_class_differs_from_live": committed_differs,
            "moved_certificates": moved_list,
            "seconds": round(time.time() - t, 1),
        }
        print(name, {k: per_config[name].get(k, 0) for k in ("certificates_moved", "class_flips", "rows_added", "rows_removed")},
              round(time.time() - t, 1), "s", flush=True)
    C._receipt_binding_block = real_binding

    # ---- the pinned audit line, under v1 and default
    for name in ("v1", "default"):
        configure(name)
        t = time.time()
        per_config[name]["audit_line"] = audit_line()
        print(name, "audit line", round(time.time() - t, 1), "s", flush=True)

    # ---- populations B and C
    if not a.quick:
        caps = [ROOT / p for p in git_files("*.capsule.html")]
        for name in BC_CONFIGS:
            configure(name)
            t = time.time()
            cap_rows = []
            for cp in caps:
                try:
                    rep = CAP.verify_capsule(cp)
                    cap_rows.append({"capsule": cp.relative_to(ROOT).as_posix(), "ok": bool(rep.get("ok")),
                                     "problems": [p[:240] for p in rep.get("problems", [])][:6],
                                     "numeral_reader_advisory": any("numeral_reader" in x for x in rep.get("advisory", []))})
                except Exception as e:           # noqa: BLE001
                    cap_rows.append({"capsule": cp.relative_to(ROOT).as_posix(), "ok": False,
                                     "error": f"{type(e).__name__}: {str(e)[:200]}"})
            rep = CH.verify_log(ROOT / "papers/charon/charon.log.jsonl", ROOT)
            per_config[name]["capsules"] = {"total": len(cap_rows), "ok": sum(1 for c in cap_rows if c["ok"]),
                                            "rows": cap_rows}
            per_config[name]["charon"] = {"lines": len(rep.get("rows", [])), "by_status": rep.get("by_status"),
                                          "skew_lines": [{"seq": r.get("seq"), "name": r.get("name"), "kind": r.get("kind")}
                                                         for r in rep.get("rows", []) if r.get("status") == "SKEW"]}
            print(name, "capsules ok", per_config[name]["capsules"]["ok"], "/", len(cap_rows), "charon",
                  rep.get("by_status"), round(time.time() - t, 1), "s", flush=True)
    restore_shipped()

    result = {
        "schema": SCHEMA, "head": head, "tree_clean": tree_clean, "certify_sha256_lf": certify_lf,
        "shipped_rules": SHIPPED,
        "population": {"certificate_files": len(rows), "resolved": len(ok),
                       "by_status": dict(collections.Counter(r["status"] for r in rows)),
                       "unresolved": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows if "_doc" not in r
                                      or r["cert"] not in base_by_cert]},
        "compared_fields": list(COMPARED),
        "configurations": per_config,
        "seconds": round(time.time() - t0, 1),
    }
    a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("->", a.out, round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
