"""Blast radius of the v0.13 UNCOVERED band, measured over the certified corpus.

The band counts numeric spans present in a document that `extract_numbers` never examined —
neither VERIFIED nor ABSTAIN nor UNGROUNDED — because every alternative of `_NUM` ends with
`(?![\\w.])` and therefore refuses a numeric span followed by a period. `precision of 0.55.`
extracts zero tokens and certifies OATH-HELD with nothing examined.

This file measures the cost of SAYING SO. `_NUM` is not changed and no token's status moves, so
the only thing that can move is the headline. Two frames:

  A  every `papers/**/*.certificate.json` whose document exists. The band is a pure function of
     the document bytes, so receipts are irrelevant here and the frame is the whole corpus.
  B  frame A restricted to documents whose receipts ALL resolve — re-certified in full, so the
     verdict string, the legacy HELD/FAILED dichotomy, and committed-certificate reproduction can
     be compared against each other.

NOTHING IS WRITTEN OUTSIDE THIS FILE'S OWN RECEIPT. No committed certificate is touched.

  python papers/closed-model-frontier/oath_v13_uncovered_band_census.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc, uncovered_spans          # noqa: E402
from styxx.corpus_audit import _resolve_receipts                # noqa: E402

OUT = HERE / "oath_v13_uncovered_band_census_result.json"


def certified_docs():
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if doc.exists():
            yield cp, doc


def main() -> int:
    frame_a, reasons = {}, collections.Counter()
    for _cp, doc in certified_docs():
        spans = uncovered_spans(doc.read_text(encoding="utf-8"))
        for s in spans:
            reasons[s["reason"]] += 1
        counted = [s for s in spans if s["counted"]]
        frame_a[doc.relative_to(ROOT).as_posix()] = {
            "uncovered": len(counted),
            "excluded_by_rule": len(spans) - len(counted),
            "items": [{"line": s["line"], "token": s["token"], "reason": s["reason"],
                       "context": s["context"]} for s in counted],
        }

    frame_b = {}
    for cp, doc in certified_docs():
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec, ROOT / "papers")
        if not receipts or missing:
            continue
        try:
            live = certify_doc(doc, receipts)
        except Exception as exc:                                # pragma: no cover - defensive
            print(f"SKIP {doc.name}: {exc}")
            continue
        frame_b[doc.relative_to(ROOT).as_posix()] = {
            "committed_verdict": rec.get("verdict"),
            "legacy_verdict": ("OATH-HELD" if live["counts"]["UNGROUNDED"] == 0
                               else "OATH-FAILED"),
            "new_verdict": live["verdict"],
            "uncovered": live["uncovered"],
            "counts": live["counts"],
        }

    n_a = len(frame_a)
    hit = [k for k, v in frame_a.items() if v["uncovered"]]
    tokens = sum(v["uncovered"] for v in frame_a.values())
    items = [i for v in frame_a.values() for i in v["items"]]
    dec = [i for i in items if "." in i["token"]]
    dist = collections.Counter(v["uncovered"] for v in frame_a.values())

    headline_changed = [k for k, v in frame_b.items()
                        if v["new_verdict"] != v["legacy_verdict"]]
    drift_new = [k for k, v in frame_b.items() if v["committed_verdict"] != v["new_verdict"]]
    drift_legacy = [k for k, v in frame_b.items()
                    if v["committed_verdict"] != v["legacy_verdict"]]

    payload = {
        "purpose": "v0.13 UNCOVERED band blast radius over the certified corpus",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "frame_a": {
            "certified_documents": n_a,
            "documents_with_uncovered": len(hit),
            "share_of_documents": round(len(hit) / n_a, 4) if n_a else 0.0,
            "uncovered_tokens": tokens,
            "uncovered_tokens_decimal": len(dec),
            "uncovered_tokens_integer": tokens - len(dec),
            "documents_with_a_decimal_uncovered": len({
                k for k, v in frame_a.items() if any("." in i["token"] for i in v["items"])}),
        },
        "reason_histogram_all_wide_only_spans": dict(reasons.most_common()),
        "uncovered_count_distribution": {str(k): v for k, v in sorted(dist.items())},
        "frame_b": {
            "certifiable_documents": len(frame_b),
            "headline_verdict_string_changed": len(headline_changed),
            "held_under_legacy_dichotomy": sum(1 for v in frame_b.values()
                                               if v["counts"]["UNGROUNDED"] == 0),
            "committed_certificate_mismatch_vs_new_verdict_string": len(drift_new),
            "committed_certificate_mismatch_vs_legacy_dichotomy": len(drift_legacy),
        },
        "note": ("The band is REPORTING. `_NUM` is unchanged, no token's status moves, and "
                 "`counts` is byte-identical to the pre-clause verifier on every document. The "
                 "only thing that moves is the headline: a HELD verdict now carries the count of "
                 "what it did not cover. `held_under_legacy_dichotomy` is what the old "
                 "`verdict == 'OATH-HELD'` test would have reported, and it is unchanged."),
        "per_document_frame_a": frame_a,
        "per_document_frame_b": frame_b,
        "headline_changed_documents": sorted(headline_changed),
        "committed_certificate_mismatch_documents": sorted(drift_legacy),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    a, b = payload["frame_a"], payload["frame_b"]
    print(f"FRAME A  certified={a['certified_documents']}  "
          f"with uncovered={a['documents_with_uncovered']} "
          f"({a['share_of_documents']:.1%})  tokens={a['uncovered_tokens']} "
          f"(decimal {a['uncovered_tokens_decimal']} / integer {a['uncovered_tokens_integer']})")
    for r, c in reasons.most_common():
        print(f"    {r:24s} {c}")
    print("  distribution (documents by uncovered count):")
    for k in sorted(dist):
        print(f"    {k:>4}: {dist[k]}")
    print(f"FRAME B  certifiable={b['certifiable_documents']}  "
          f"headline string changed={b['headline_verdict_string_changed']}  "
          f"HELD under legacy dichotomy={b['held_under_legacy_dichotomy']}  "
          f"committed-cert mismatch (legacy)={b['committed_certificate_mismatch_vs_legacy_dichotomy']}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
