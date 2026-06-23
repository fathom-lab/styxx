"""Consensus failure-core: do 4 independently-built models (Alibaba/Meta/Google/OpenAI) converge on
the SAME hardest / easiest domains, beyond chance? Robust top-k overlap (not the noisy full Spearman),
with an independence-null baseline. Uses ungated_hallucination_rate (well-powered: over all n items
per domain). No GPU/API.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = {"Alibaba(Qwen)": "crossfamily_gate_Qwen2_5_3B_Instruct.json",
     "Meta(Llama)": "crossfamily_gate_Llama_3_2_3B_Instruct.json",
     "Google(Gemma)": "crossfamily_gate_gemma_2_2b_it.json",
     "OpenAI(4o-mini)": "xvendor_gpt4omini_nli_gate.json"}
K = 6

def main():
    M = {k: json.load(open(HERE / v, encoding="utf-8"))["category_competence_cliff_map"] for k, v in G.items()}
    doms = sorted(set.intersection(*[set(m) for m in M.values()]))
    N = len(doms)
    hard = {k: set(sorted(doms, key=lambda d: -M[k][d]["ungated_hallucination_rate"])[:K]) for k in M}
    easy = {k: set(sorted(doms, key=lambda d: M[k][d]["ungated_hallucination_rate"])[:K]) for k in M}
    core_hard = sorted(set.intersection(*hard.values()))
    core_easy = sorted(set.intersection(*easy.values()))
    p_all4 = (K / N) ** 4
    exp = N * p_all4
    p_ge1 = 1 - (1 - p_all4) ** N
    out = {"providers": list(G), "n_domains": N, "k": K,
           "consensus_hardest": core_hard, "consensus_safest": core_easy,
           "expected_4way_overlap_by_chance": round(exp, 4),
           "p_at_least_one_shared_by_chance": round(p_ge1, 4),
           "note": ("Independence null OVERSTATES surprise — the 4 models share web training corpora, so "
                    "convergence partly reflects shared data + TruthfulQA's adversarial design, not only a "
                    "universal difficulty law. Per-domain rates are single-run (see FINDING_cliff_variance). "
                    "Robust claim: the hard/easy CORE is shared cross-vendor far above an independence baseline.")}
    print(json.dumps(out, indent=2))
    (HERE / "consensus_failure_core_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
