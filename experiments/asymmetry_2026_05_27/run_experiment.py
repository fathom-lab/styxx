# -*- coding: utf-8 -*-
"""Generation-vs-Critique Asymmetry Experiment.

Pre-stated prediction (PRE_STATED_PREDICTION.md): for each folklore item,
gpt-4o-mini in generation mode should produce something similar to the
council's expected_consensus AND gpt-4o-mini in critique mode should flag
that same consensus as incorrect. The asymmetry is operationalized as
HIGH generation_match AND HIGH critique_no_prob.

This script runs the experiment and writes results to results.json.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import openai
from sentence_transformers import SentenceTransformer

# Resolve paths from the experiment dir up to the repo root
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BENCHMARK = REPO / "papers" / "consensus-hallucination" / "darkcore_benchmark_2026_05_27.json"
OUTPUT = HERE / "results.json"


def main():
    client = openai.OpenAI()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    folklore = [r for r in bench["records"] if r.get("class") == "folklore"]
    print(f"folklore items: {len(folklore)}")

    rows = []
    for i, r in enumerate(folklore):
        q = r["question"]
        consensus = r["expected_consensus"]
        rid = r.get("id", f"folk-{i:04d}")

        # 1. Generation mode: ask gpt-4o-mini to answer the question fresh
        gen_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Question: {q}\n\nAnswer:"}],
            max_tokens=150,
            temperature=0,
        )
        gen_response = gen_completion.choices[0].message.content.strip()

        # 2. Critique mode: ask gpt-4o-mini if the consensus is correct
        crit_prompt = (
            f"Question: {q}\n\n"
            f"Proposed answer: {consensus}\n\n"
            f"Is the proposed answer factually correct? Respond with only the single word YES or NO."
        )
        crit_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": crit_prompt}],
            max_tokens=2,
            temperature=0,
            logprobs=True,
            top_logprobs=10,
        )
        first_token_logprobs = crit_completion.choices[0].logprobs.content[0].top_logprobs
        yes_lp, no_lp = -20.0, -20.0
        for ent in first_token_logprobs:
            tok = ent.token.strip().upper()
            if tok == "YES" and yes_lp == -20.0:
                yes_lp = ent.logprob
            elif tok == "NO" and no_lp == -20.0:
                no_lp = ent.logprob
        m = max(yes_lp, no_lp)
        e_y = math.exp(yes_lp - m)
        e_n = math.exp(no_lp - m)
        p_no = e_n / (e_y + e_n)

        # 3. Embedding similarity between generated answer and consensus
        embs = embedder.encode([gen_response, consensus], normalize_embeddings=True, show_progress_bar=False)
        gen_match = float(embs[0] @ embs[1])

        row = {
            "id": rid,
            "question": q,
            "expected_consensus": consensus,
            "generated_response": gen_response,
            "generation_match": gen_match,
            "critique_no_prob": p_no,
            "high_gen_match": gen_match >= 0.60,
            "high_critique_no": p_no >= 0.50,
            "asymmetry_present": gen_match >= 0.60 and p_no >= 0.50,
        }
        rows.append(row)
        print(f"  [{i+1:2d}/{len(folklore)}] gen_match={gen_match:.3f}  p_no={p_no:.3f}  "
              f"{'ASYMMETRY' if row['asymmetry_present'] else 'no-asym '}  q={q[:50]!r}")

    # Aggregate stats
    n = len(rows)
    mean_gm = sum(r["generation_match"] for r in rows) / n
    mean_pn = sum(r["critique_no_prob"] for r in rows) / n
    asym_rate = sum(r["asymmetry_present"] for r in rows) / n

    quadrants = {
        "HH_asymmetry": sum(1 for r in rows if r["high_gen_match"] and r["high_critique_no"]),
        "HL_sycophantic_consistent": sum(1 for r in rows if r["high_gen_match"] and not r["high_critique_no"]),
        "LH_already_corrected_in_gen": sum(1 for r in rows if not r["high_gen_match"] and r["high_critique_no"]),
        "LL_no_effect": sum(1 for r in rows if not r["high_gen_match"] and not r["high_critique_no"]),
    }

    out = {
        "n_folklore": n,
        "mean_generation_match": round(mean_gm, 4),
        "mean_critique_no_prob": round(mean_pn, 4),
        "asymmetry_rate": round(asym_rate, 4),
        "quadrants": quadrants,
        "quadrant_percentages": {k: round(v / n, 4) for k, v in quadrants.items()},
        "thresholds": {"high_gen_match": 0.60, "high_critique_no": 0.50},
        "per_item": rows,
    }

    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== SUMMARY ===")
    print(f"  mean generation_match  = {mean_gm:.4f}")
    print(f"  mean critique_no_prob  = {mean_pn:.4f}")
    print(f"  asymmetry rate (both high) = {asym_rate:.4f}")
    print(f"  quadrants:")
    for k, v in quadrants.items():
        print(f"    {k:32s} = {v}/{n} = {v/n:.2%}")
    print(f"\n  wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
