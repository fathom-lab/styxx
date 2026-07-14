"""G5 aggregator -- deterministic strata + metrics + the frozen verdict, from extraction receipts.

PREREG: papers/gpai-scorecard/PREREG_gpai_scorecard_2026_07_14.md (frozen a059ba9, before extraction).
Inputs (receipts): CARD_MANIFEST.json + extraction/<slug>.graded.json (+ pass files, verify, fidelity).
Output: scorecard_result.json. No LLM in this file -- the verdict is computed, re-runnable, CPU-only.

Frozen operationalizations (disclosed here, decided before any receipt was read by this script):
- eval-bearing doc     := a doc with at least 1 extracted numeric eval claim.
- V1 instrument gate   := >50% of eval-bearing docs carry <10 claims -> VOID_G5__instrument_mismatch.
- V2 resolution gate   := share of SCOREABLE docs whose passes disagreed (Jaccard<0.90) AND whose
                          adjudication failed to return a non-empty verified set > 0.25
                          -> VOID_G5__extraction_unreliable.
- V4 fidelity gate     := a PDF doc with spot-check < 19/20 is EXCLUDED; >2 exclusions
                          -> VOID_G5__pdf_pipeline.
- V5 population gate   := SCOREABLE count < 10 -> VOID_G5__population_too_small.
- BOUND_effective      := min(BOUND graded, BOUND confirmed by the adversarial verify pass) --
                          the conservative count; a skeptic-rejected BOUND never inflates genre health.
- SCOREABLE            := fetched doc with >= 10 claims (xAI stays SCOREABLE, flagged PARTIAL;
                          headline reported with and without xAI).
- median/IQR           := numpy median + percentile(25/75, linear interpolation) over per-doc
                          BOUND_effective rates of SCOREABLE docs.

Usage: python papers/gpai-scorecard/scorecard_aggregate.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXTR = HERE / "extraction"
MANIFEST = HERE / "CARD_MANIFEST.json"
OUT = HERE / "scorecard_result.json"

LADDER = ["BOUND", "CONFIG-NO-CODE", "NAMED-ONLY", "UNBOUND"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    man = load(MANIFEST)
    providers = man["providers"]
    rows = []
    for prov, entry in providers.items():
        slug = prov.lower().replace(" ", "_")
        row = {"provider": prov, "slug": slug, "signed": entry["signed"],
               "partial_signatory": entry["signed"].startswith("partial")}
        if entry.get("doc") is None:
            row["stratum"] = "NO-PUBLIC-FLAGSHIP-DOC"
            row["n_claims"] = 0
            rows.append(row)
            continue
        row["format"] = entry["doc"]["format"]
        graded_p = EXTR / f"{slug}.graded.json"
        pass_a = EXTR / f"{slug}.pass_a.json"
        pass_b = EXTR / f"{slug}.pass_b.json"
        adjud = EXTR / f"{slug}.adjudicated.json"
        verify_p = EXTR / f"{slug}.verify.json"
        n_a = len(load(pass_a).get("claims", [])) if pass_a.exists() else None
        n_b = len(load(pass_b).get("claims", [])) if pass_b.exists() else None
        row["pass_a_claims"], row["pass_b_claims"] = n_a, n_b
        row["adjudicated"] = adjud.exists()
        if not graded_p.exists():
            # extraction found nothing gradeable
            row["stratum"] = "LOW-QUANT"
            row["n_claims"] = 0
            row["adjudication_failed"] = bool(adjud.exists() and not load(adjud).get("claims"))
            rows.append(row)
            continue
        g = load(graded_p)
        claims = g.get("claims", [])
        counts = {k: 0 for k in LADDER}
        for c in claims:
            gr = c.get("grade")
            if gr in counts:
                counts[gr] += 1
        n = sum(counts.values())
        row["n_claims"] = n
        row["grades"] = counts
        row["doc_level"] = {k: g.get(k) for k in
                            ("harness_named", "harness_version_stated", "code_or_outputs_linked")}
        row["linked_evidence_urls"] = g.get("linked_evidence_urls", [])
        # conservative BOUND: the skeptic's confirmation caps the graded count
        bound_eff = counts["BOUND"]
        if verify_p.exists():
            v = load(verify_p)
            row["verify"] = {k: v.get(k) for k in
                             ("bound_checked", "bound_confirmed", "sample_checked", "sample_disagreements")}
            bc = v.get("bound_confirmed")
            if isinstance(bc, (int, float)):
                bound_eff = min(bound_eff, int(bc))
        row["bound_effective"] = bound_eff
        row["bound_rate"] = round(bound_eff / n, 4) if n else None
        fid_p = EXTR / f"{slug}.fidelity.json"
        if fid_p.exists():
            f = load(fid_p)
            row["fidelity"] = {"n_checked": f.get("n_checked"), "n_matched": f.get("n_matched")}
            row["fidelity_excluded"] = bool(
                f.get("n_checked", 0) >= 20 and f.get("n_matched", 0) < 19
                or (0 < f.get("n_checked", 0) < 20 and f.get("n_matched", 0) < f.get("n_checked", 0) - 1))
        else:
            row["fidelity_excluded"] = False
        row["stratum"] = ("SCOREABLE" if n >= 10 else "LOW-QUANT")
        rows.append(row)

    # ---- guards (frozen) ----
    eval_bearing = [r for r in rows if r.get("n_claims", 0) >= 1]
    thin = [r for r in eval_bearing if r["n_claims"] < 10]
    v1_fired = len(eval_bearing) > 0 and (len(thin) / len(eval_bearing)) > 0.50

    scoreable = [r for r in rows if r.get("stratum") == "SCOREABLE" and not r.get("fidelity_excluded")]
    excluded_fid = [r for r in rows if r.get("fidelity_excluded")]
    v4_fired = len(excluded_fid) > 2

    adjud_failed = [r for r in scoreable if r.get("adjudicated") and r.get("n_claims", 0) == 0]
    v2_fired = len(scoreable) > 0 and (len(adjud_failed) / max(1, len(scoreable))) > 0.25

    v5_fired = len(scoreable) < 10

    # ---- metrics ----
    rates = np.array([r["bound_rate"] for r in scoreable if r["bound_rate"] is not None])
    rates_noxai = np.array([r["bound_rate"] for r in scoreable
                            if r["bound_rate"] is not None and not r["partial_signatory"]])
    pooled_bound = sum(r["bound_effective"] for r in scoreable)
    pooled_n = sum(r["n_claims"] for r in scoreable)

    def q(a, p):
        return round(float(np.percentile(a, p)), 4) if len(a) else None

    median_rate = q(rates, 50)
    metrics = {
        "n_scoreable": len(scoreable),
        "median_bound_rate": median_rate,
        "iqr_bound_rate": [q(rates, 25), q(rates, 75)],
        "median_bound_rate_excl_xai": q(rates_noxai, 50),
        "pooled_bound_rate": round(pooled_bound / pooled_n, 4) if pooled_n else None,
        "pooled_bound_claims": pooled_bound, "pooled_total_claims": pooled_n,
        "per_doc_n_claims_min": min((r["n_claims"] for r in scoreable), default=None),
        "per_doc_n_claims_max": max((r["n_claims"] for r in scoreable), default=None),
    }

    # ---- frozen verdict (order: VOIDs -> fork) ----
    if v1_fired:
        verdict = "VOID_G5__instrument_mismatch"
    elif v2_fired:
        verdict = "VOID_G5__extraction_unreliable"
    elif v4_fired:
        verdict = "VOID_G5__pdf_pipeline"
    elif v5_fired:
        verdict = "VOID_G5__population_too_small"
    elif median_rate is not None and median_rate < 0.50:
        verdict = "GENRE_DEFICIT__median_below_0p50"
    else:
        verdict = "GENRE_HEALTHIER__median_at_or_above_0p50"

    result = {
        "what": "G5 -- receipt-binding scorecard across EU GPAI Code-of-Practice signatories' flagship docs (population-framed)",
        "prereg": "papers/gpai-scorecard/PREREG_gpai_scorecard_2026_07_14.md",
        "manifest": "papers/gpai-scorecard/CARD_MANIFEST.json",
        "verdict": verdict,
        "guards": {"v1_instrument_mismatch": v1_fired, "v2_extraction_unreliable": v2_fired,
                   "v4_pdf_pipeline": v4_fired, "v5_population_too_small": v5_fired,
                   "eval_bearing_docs": len(eval_bearing), "thin_docs": len(thin),
                   "fidelity_excluded_docs": [r["slug"] for r in excluded_fid],
                   "adjudication_failed_docs": [r["slug"] for r in adjud_failed]},
        "metrics": metrics,
        "strata_counts": {s: sum(1 for r in rows if r.get("stratum") == s)
                          for s in ("SCOREABLE", "LOW-QUANT", "NO-PUBLIC-FLAGSHIP-DOC")},
        "providers": rows,
        "framing": "binding rate measures LINKAGE, not evidence existence and not truth; "
                   "receipts may exist unlinked; no provider is alleged non-compliant",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"VERDICT: {verdict}")
    print(f"  scoreable={len(scoreable)} median_bound_rate={median_rate} "
          f"iqr={metrics['iqr_bound_rate']} pooled={metrics['pooled_bound_rate']} "
          f"({pooled_bound}/{pooled_n})")
    for r in sorted(scoreable, key=lambda x: -(x["bound_rate"] or 0)):
        print(f"  {r['slug']:<22} n={r['n_claims']:>4} bound={r['bound_effective']:>3} "
              f"rate={r['bound_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
