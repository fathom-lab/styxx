"""Apparatus-MATCHED cross-vendor cliff: re-judge gpt-4o-mini's stored samples with the SAME
NLI judge used for the local open families, so the open<->closed comparison has NO judge confound.

Motivation: the open<->open hallucination cliff is strongly invariant (Spearman 0.77, all NLI-judged);
the naive open<->closed comparison is weak (0.23) AND inverts the refusal/hallucination ordering ->
the signature of a JUDGE confound (gpt-4o-mini was LLM-judged, the open families NLI-judged), not a
real divergence. Fix: gpt-4o-mini's per-item resamples are already on disk in
truthfulqa_benchmark_result.json (no API key needed). Re-judge them with the identical NLI judge,
recompute gpt-4o-mini's cliff under the matched apparatus, then compare to the open families.

No API key. Needs only the NLI model on GPU (run after the read sweep frees the card).

Usage:  python run_xvendor_matched.py
"""
from __future__ import annotations

import json
from pathlib import Path

import run_truthfulqa_benchmark as B
import run_pregeneration_gate as G
from run_local_cliff import NLIJudge, _judge_item, NLI_MODEL
from styxx.audit import (
    _derive_verdict, _DEFAULT_HONEST, _DEFAULT_LOW_STABILITY, _DEFAULT_CONTRADICTION,
)

HERE = Path(__file__).resolve().parent
SRC = HERE / "truthfulqa_benchmark_result.json"          # gpt-4o-mini, LLM-judged (committed a75f1e7)
OUT_BENCH = HERE / "xvendor_gpt4omini_nli_benchmark.json"
OUT_GATE = HERE / "xvendor_gpt4omini_nli_gate.json"


def main() -> int:
    src = json.load(open(SRC, encoding="utf-8"))
    items_in = src["items"]
    print(f"re-judging {len(items_in)} gpt-4o-mini items with NLI (matched apparatus)", flush=True)
    nli = NLIJudge(NLI_MODEL, device="cuda")

    results = []
    for k, it in enumerate(items_in):
        samples = it.get("samples", [])
        n = len(samples)
        if n == 0:
            continue
        jt, jf = _judge_item(nli, samples, it["best"], it["worst"])
        g_t, st_t, c_t = B.grounded_from_batch(jt, n)
        g_f, st_f, c_f = B.grounded_from_batch(jf, n)
        v_t = _derive_verdict(grounded=g_t, stability=st_t, concordance_stateless=c_t,
                              injection_suspected=False, honest=_DEFAULT_HONEST,
                              low_stability=_DEFAULT_LOW_STABILITY, contradiction=_DEFAULT_CONTRADICTION)
        v_f = _derive_verdict(grounded=g_f, stability=st_f, concordance_stateless=c_f,
                              injection_suspected=False, honest=_DEFAULT_HONEST,
                              low_stability=_DEFAULT_LOW_STABILITY, contradiction=_DEFAULT_CONTRADICTION)
        results.append({
            "idx": it["idx"], "question": it["question"], "best": it["best"],
            "worst": it["worst"], "category": it["category"], "samples": samples,
            "g_true": g_t, "stability_true": st_t, "concordance_true": c_t,
            "n_clusters_true": jt["n_clusters"], "matches_true": jt["matches"], "verdict_true": v_t,
            "g_false": g_f, "stability_false": st_f, "concordance_false": c_f,
            "n_clusters_false": jf["n_clusters"], "matches_false": jf["matches"], "verdict_false": v_f,
        })
        if k % 100 == 0:
            print(f"  judged {k}/{len(items_in)}", flush=True)

    receipt = {
        "model": "gpt-4o-mini", "judge": f"NLI-bidir-entail({NLI_MODEL}) (re-judge of stored LLM-judged samples)",
        "source_receipt": "truthfulqa_benchmark_result.json", "n_items": len(results),
        "answer_key_sha256": src.get("answer_key_sha256"),
        "answer_key_sha256_expected": src.get("answer_key_sha256_expected"),
        "items": results,
    }
    OUT_BENCH.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    G.BENCHMARK_RECEIPT = OUT_BENCH
    G.RECEIPT = OUT_GATE
    G.main()
    print(f"\nMatched-apparatus gpt-4o-mini cliff written: {OUT_GATE}")
    print("Now compare its category_competence_cliff_map (NLI-judged) to the open families "
          "(also NLI-judged) for a CLEAN open<->closed hallucination-cliff invariance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
