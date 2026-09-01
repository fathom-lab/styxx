"""h — the handedness census, in the two populations that must never be pooled.

The 2026-09-01 audit named M2, the handed-target mechanism: an instrument measured where its
target was supplied collapses when it must find the target itself. For `styxx.certify` the
question *who handed the verifier this token?* has a recorded answer on every obligated ledger
entry — `epistemics.obligation_source` — and `h_mapping.json` declares, for every source the
verifier can emit, whether the target came from the OBJECT'S OWN TEXT, from the OBJECT'S FORM,
from the RECEIPT side, or from a party EXTERNAL to both. This script folds that declaration over
the corpus and writes one receipt. It changes nothing.

TWO POPULATIONS, REPORTED SEPARATELY BY CONSTRUCTION.

  PRINTED  — what the committed certificates say. Only certificates issued under
             `styxx-oath/epistemics-summary/v1` carry an `obligation_sources` block, and only
             ledgers annotated since 2026-08-28 carry per-token `epistemics`. This population is
             stratified by verifier version: it is a minority of the corpus and it is the ONLY
             population a reader of the committed certificates can see.
  LIVE     — every certifiable document re-certified at the CURRENT verifier, right now. This is
             the whole corpus under one verifier build. It is what a census sees and what no
             committed certificate prints.

A share computed in one population and quoted against the other is a handed-target error one
level up: the number would be measured on a population that was never the one being described.
The receipt therefore carries both under separate keys and never a pooled figure.

TWO DENOMINATORS, NAMED. "Bound" has been used in this lane for two different things:
`bound` in `certify.py` means OBLIGATED (an obligation clause fired), while "the number binds"
in the RESULT prose means VERIFIED (a value matched). The share of vocabulary-obligated tokens
is 0.83 of obligated tokens and 0.35 of verified tokens on the same run; both are reported with
their denominator in the key name so neither can be quoted as the other.

  python papers/closed-model-frontier/h_mapping_census.py
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

from styxx.certify import _EPISTEMICS_SOURCES, certify_doc   # noqa: E402
from styxx.corpus_audit import _resolve_receipts             # noqa: E402

MAPPING = HERE / "h_mapping.json"
OUT = HERE / "h_mapping_census_result.json"


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def certified_docs():
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if doc.exists():
            yield cp, doc


def fold_printed(mapping: dict) -> dict:
    """Population PRINTED: fold the committed certificates only. Nothing is re-certified."""
    summary_sources = collections.Counter()
    summary_obligated = 0
    n_summary = 0
    token_sources = collections.Counter()
    n_token_ledgers = 0
    ledger_tokens = 0
    bound = 0
    builds = collections.Counter()
    n_total = 0
    for cp, _doc in certified_docs():
        try:
            cert = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n_total += 1
        builds[cert.get("verifier_sha256")] += 1
        es = cert.get("epistemics_summary")
        if es and es.get("schema") == "styxx-oath/epistemics-summary/v1":
            n_summary += 1
            summary_obligated += es["obligated_total"]
            for k, v in es["obligation_sources"].items():
                summary_sources[k] += v
        ledger = cert.get("ledger") or []
        if ledger and all("epistemics" in e for e in ledger):
            n_token_ledgers += 1
            for e in ledger:
                ledger_tokens += 1
                ep = e["epistemics"]
                if ep.get("obligated"):
                    bound += 1
                    token_sources[ep.get("obligation_source")] += 1
    return {
        "what": "the committed certificates, as they stand; nothing re-certified",
        "certificates_on_disk": n_total,
        "distinct_verifier_builds_on_disk": len(builds),
        "certificates_with_epistemics_summary_v1": n_summary,
        "summary_obligated_total": summary_obligated,
        "summary_obligation_sources": dict(summary_sources),
        "summary_source_share_of_obligated": {
            k: _share(v, summary_obligated) for k, v in summary_sources.items()},
        "certificates_with_per_token_epistemics": n_token_ledgers,
        "ledger_tokens": ledger_tokens,
        "obligated_tokens": bound,
        "token_obligation_sources": dict(token_sources),
        "token_source_share_of_obligated": {k: _share(v, bound) for k, v in token_sources.items()},
        "handedness_share_of_obligated": _classes(token_sources, bound, mapping),
        "coverage_note": (f"{n_summary} of {n_total} committed certificates carry the v1 summary "
                          f"and {n_token_ledgers} carry per-token epistemics; every other "
                          "certificate is silent about who handed it its targets. Absence is "
                          "pre-annotation, never zero."),
    }


def fold_live(mapping: dict) -> dict:
    """Population LIVE: every certifiable document re-certified at the current verifier."""
    sources = collections.Counter()
    by_source_status = collections.Counter()
    docs = skipped = tokens = bound = verified = 0
    verified_by_source = collections.Counter()
    for cp, doc in certified_docs():
        try:
            cert = json.loads(cp.read_text(encoding="utf-8"))
            receipts, missing, _ = _resolve_receipts(cp, cert, ROOT / "papers")
            if not receipts or missing:
                skipped += 1
                continue
            live = certify_doc(doc, receipts)
        except Exception:                                   # pragma: no cover - defensive
            skipped += 1
            continue
        docs += 1
        for e in live["ledger"]:
            tokens += 1
            ep = e["epistemics"]
            if e["status"] == "VERIFIED":
                verified += 1
            if ep["obligated"]:
                bound += 1
                sources[ep["obligation_source"]] += 1
                by_source_status[f"{ep['obligation_source']}|{e['status']}"] += 1
                if e["status"] == "VERIFIED":
                    verified_by_source[ep["obligation_source"]] += 1
    return {
        "what": "every certifiable document re-certified at the current verifier",
        "documents_recertified": docs,
        "documents_skipped_unresolvable": skipped,
        "ledger_tokens": tokens,
        "verified_tokens": verified,
        "obligated_tokens": bound,
        "obligation_sources": dict(sources),
        "source_share_of_obligated": {k: _share(v, bound) for k, v in sources.items()},
        "verified_obligated_by_source": dict(verified_by_source),
        "source_share_of_verified": {
            k: _share(v, verified) for k, v in verified_by_source.items()},
        "by_source_and_status": dict(sorted(by_source_status.items())),
        "handedness_share_of_obligated": _classes(sources, bound, mapping),
    }


def _classes(sources: collections.Counter, denom: int, mapping: dict) -> dict:
    out = collections.Counter()
    for src, n in sources.items():
        out[mapping["declared_sources"][src]["handed_by"]] += n
    return {k: {"tokens": v, "share": _share(v, denom)} for k, v in sorted(out.items())}


def main() -> int:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    declared = set(mapping["declared_sources"])
    emittable = set(_EPISTEMICS_SOURCES)
    undeclared = sorted(emittable - declared)
    if undeclared:
        print(f"REFUSED: the verifier can emit obligation sources the mapping does not declare: "
              f"{undeclared}. Declare them in {MAPPING.name} before measuring h.")
        return 1

    printed = fold_printed(mapping)
    live = fold_live(mapping)
    observed = {k for k, v in live["obligation_sources"].items() if v} | \
               {k for k, v in printed["token_obligation_sources"].items() if v}

    payload = {
        "purpose": "handedness (h) of styxx.certify's obligation predicate, per declared mapping",
        "mapping_schema": mapping["schema"],
        "mapping_sha256": hashlib.sha256(MAPPING.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "sources": {
            "emittable_by_this_verifier": sorted(emittable),
            "declared_in_mapping": sorted(declared),
            "observed_nonzero_in_either_population": sorted(observed),
            "emittable_but_zero_in_both_populations": sorted(emittable - observed),
            "declared_but_not_emittable": sorted(declared - emittable),
        },
        "population_PRINTED": printed,
        "population_LIVE": live,
        "instrument_level": mapping["instrument_level"],
        "never_pool": ("PRINTED and LIVE are different populations under different verifier "
                       "builds. No key in this receipt combines them, and a rate quoted from one "
                       "must name it."),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    p, lv = printed, live
    print(f"sources: emittable={sorted(emittable)}")
    print(f"         observed={sorted(observed)}  zero-in-both={sorted(emittable - observed)}  "
          f"declared-unshipped={sorted(declared - emittable)}")
    print(f"PRINTED  certificates={p['certificates_on_disk']} builds={p['distinct_verifier_builds_on_disk']}  "
          f"with-summary={p['certificates_with_epistemics_summary_v1']}  "
          f"with-token-epistemics={p['certificates_with_per_token_epistemics']}  "
          f"obligated={p['obligated_tokens']}")
    for k, v in p["token_source_share_of_obligated"].items():
        print(f"    {k:20s} {p['token_obligation_sources'][k]:5d}  {v:.4f} of obligated")
    print(f"LIVE     docs={lv['documents_recertified']} skipped={lv['documents_skipped_unresolvable']}  "
          f"tokens={lv['ledger_tokens']} verified={lv['verified_tokens']} obligated={lv['obligated_tokens']}")
    for k, v in lv["source_share_of_obligated"].items():
        print(f"    {k:20s} {lv['obligation_sources'][k]:5d}  {v:.4f} of obligated  "
              f"{lv['source_share_of_verified'].get(k, 0.0):.4f} of verified")
    print("handedness (share of obligated):")
    for pop, d in (("PRINTED", p), ("LIVE", lv)):
        print(f"    {pop:8s} " + "  ".join(f"{k}={v['share']:.4f}" for k, v in
                                          d["handedness_share_of_obligated"].items()))
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
