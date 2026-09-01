"""The unobligated-oath census: how many VERIFIED tokens was nothing obligated to examine?

`RECON_v13_not_frozen_the_ladder_2026_08_28.md` established that obligation gates accusation, not
verification: a value match produces `VERIFIED` whether or not any clause obligated the verifier to
look. The epistemics annotation (frozen invariant:
`INVARIANT_epistemics_annotation_2026_08_28.md`, verified moved-nothing over all 192 certificates)
makes that countable for the first time. This counts it.

Definitions, exact:

* **unobligated oath** — a `VERIFIED` ledger entry with `epistemics.obligated == false`. The
  verifier swore to a value that nothing required it to examine. It may still be a true, well-bound
  claim; what it is NOT is the product of the obligation predicate.
* **path-checked** — the v0.3 integer count-binding filter ran (`decimals == 0`). For decimals it
  never runs (v0.8 CLOSED_NEGATIVE), so an unobligated decimal oath is the weakest attestation the
  instrument produces: value match alone, path never compared, obligation never consulted.

Live re-certification of every document under `papers/` at the pinned verifier. Stored historical
certificates are untouched; they predate the annotation and remain evidence.

  python papers/closed-model-frontier/oath_unobligated_oath_census.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                    # noqa: E402
from styxx.corpus_audit import _doc_for, _resolve_receipts, discover_certificates  # noqa: E402

OUT = HERE / "oath_unobligated_oath_census.json"


def main() -> int:
    status = collections.Counter()
    branches = collections.Counter()
    ob_src = collections.Counter()
    ver = collections.Counter()          # verified split
    per_doc = []
    docs = errs = 0

    for cp in discover_certificates(ROOT / "papers"):
        try:
            stored = json.loads(cp.read_text(encoding="utf-8"))
            doc = _doc_for(cp)
            paths, _m, _d = _resolve_receipts(cp, stored, ROOT / "papers")
            if not doc.exists() or not paths:
                errs += 1
                continue
            cert = certify_doc(doc, paths)
        except Exception:
            errs += 1
            continue
        docs += 1
        d_ver = d_unob = 0
        for e in cert["ledger"]:
            ep = e["epistemics"]
            status[e["status"]] += 1
            branches[ep["branch"]] += 1
            if ep["obligated"]:
                ob_src[ep["obligation_source"]] += 1
            if e["status"] == "VERIFIED":
                d_ver += 1
                if ep["obligated"]:
                    key = "obligated"
                else:
                    key = "UNOBLIGATED"
                    d_unob += 1
                if not ep.get("path_checked", False) and ep["branch"] == "value-match":
                    key += "+path-unchecked"
                ver[key] += 1
        if d_ver:
            per_doc.append({"document": doc.name, "verified": d_ver,
                            "unobligated": d_unob,
                            "unobligated_share": round(d_unob / d_ver, 4)})

    per_doc.sort(key=lambda r: (-r["unobligated_share"], -r["unobligated"]))
    total_ver = status["VERIFIED"]
    unob = sum(v for k, v in ver.items() if k.startswith("UNOBLIGATED"))
    weakest = ver.get("UNOBLIGATED+path-unchecked", 0)

    payload = {
        "census": "the unobligated oath, counted for the first time",
        "status": "MEASUREMENT over live re-certification; stored certificates untouched",
        "documents": docs, "unresolvable_skipped": errs,
        "status_counts": dict(status),
        "ladder_branches": dict(branches.most_common()),
        "obligation_sources_when_obligated": dict(ob_src.most_common()),
        "verified_split": dict(ver.most_common()),
        "headline": {
            "verified_total": total_ver,
            "unobligated_oaths": unob,
            "unobligated_oath_rate": round(unob / total_ver, 4) if total_ver else None,
            "weakest_attestations": weakest,
            "weakest_share_of_verified": round(weakest / total_ver, 4) if total_ver else None,
            "weakest_means": ("VERIFIED with obligated=false AND path never compared: value "
                              "match alone, on a token nothing required the verifier to read"),
        },
        "documents_most_exposed": per_doc[:12],
        "what_this_does_not_say": (
            "That unobligated oaths are wrong. Many will be true claims whose lines happen to "
            "carry no trigger vocabulary. What it says is narrower and worse for the certificate's "
            "semantics: the affirmative attestation is not scoped by the obligation predicate, so "
            "OATH-HELD's verified count mixes oaths the instrument was required to take with "
            "oaths it volunteered. The blind-panel measurement of how often volunteered oaths land "
            "on non-claims (~1 in 5 externally) is the companion number, measured 2026-08-27."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"documents {docs}   skipped {errs}")
    print(f"status: {dict(status)}")
    print("\nVERIFIED split:")
    for k, v in ver.most_common():
        print(f"   {v:>5}  {k}")
    h = payload["headline"]
    print(f"\nUNOBLIGATED OATH RATE : {h['unobligated_oath_rate']}  "
          f"({h['unobligated_oaths']} of {h['verified_total']} verifications)")
    print(f"weakest attestations  : {h['weakest_attestations']}  "
          f"({h['weakest_share_of_verified']} of all VERIFIED)")
    print("\nmost exposed documents:")
    for r in per_doc[:6]:
        print(f"   {r['unobligated_share']:>7}  {r['unobligated']:>3}/{r['verified']:<4} {r['document'][:52]}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
