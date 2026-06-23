"""MMLU consensus failure-core replication test (PREREG_mmlu_consensus_core_2026_06_23, frozen pre-data).

Does the cross-vendor failure-core convergence (TruthfulQA: 4 vendors share the same hardest/easiest
domains far above chance) REPLICATE on non-adversarial MMLU? 3 open families, K=9 (frozen), 10k-shuffle
independence-null permutation p. Runs once run_mmlu_cliff.py has produced the 3 _mmlu gates. No GPU/API.
"""
from __future__ import annotations
import json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATES = {"Alibaba(Qwen)": "crossfamily_gate_Qwen2_5_3B_Instruct_mmlu.json",
         "Meta(Llama)": "crossfamily_gate_Llama_3_2_3B_Instruct_mmlu.json",
         "Google(Gemma)": "crossfamily_gate_gemma_2_2b_it_mmlu.json"}
K = 9            # frozen in the prereg (= round(57 * 6/37), matched to TruthfulQA K/N ~ 0.16)
N_PERM = 10000
SEED = 0


def consensus(rank_sets):
    return sorted(set.intersection(*rank_sets))


def perm_p(observed_overlap, n_domains, k, n_fam, n_perm, seed):
    rng = random.Random(seed)
    universe = list(range(n_domains))
    ge = 0
    for _ in range(n_perm):
        sets = [set(rng.sample(universe, k)) for _ in range(n_fam)]
        if len(set.intersection(*sets)) >= observed_overlap:
            ge += 1
    return ge / n_perm


def main():
    missing = [v for v in GATES.values() if not (HERE / v).exists()]
    if missing:
        print("WAITING — MMLU gates not yet present:", missing)
        return 1
    M = {k: json.load(open(HERE / v, encoding="utf-8"))["category_competence_cliff_map"] for k, v in GATES.items()}
    doms = sorted(set.intersection(*[set(m) for m in M.values()]))
    N = len(doms)
    hard = [set(sorted(doms, key=lambda d: -M[k][d]["ungated_hallucination_rate"])[:K]) for k in M]
    easy = [set(sorted(doms, key=lambda d: M[k][d]["ungated_hallucination_rate"])[:K]) for k in M]
    core_hard = consensus(hard)
    core_easy = consensus(easy)
    p_hard = perm_p(len(core_hard), N, K, len(M), N_PERM, SEED)
    p_easy = perm_p(len(core_easy), N, K, len(M), N_PERM, SEED + 1)

    primary = "REPLICATES (convergence above chance)" if p_hard < 0.05 else \
              "DOES NOT REPLICATE (convergence not above chance — TruthfulQA-specific)"
    out = {"benchmark": "MMLU", "n_subjects": N, "K": K, "n_families": len(M), "n_perm": N_PERM,
           "consensus_hardest": core_hard, "consensus_hardest_overlap": len(core_hard),
           "consensus_hardest_perm_p": round(p_hard, 5),
           "consensus_safest": core_easy, "consensus_safest_overlap": len(core_easy),
           "consensus_safest_perm_p": round(p_easy, 5),
           "PRIMARY_verdict": primary,
           "note": ("Independence-null permutation overstates surprise (shared corpora). 3 OPEN vendors "
                    "only. Specific subjects need not match TruthfulQA (structural, not topical).")}
    print(json.dumps(out, indent=2))
    (HERE / "consensus_core_mmlu_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nPRIMARY: consensus-hardest overlap {len(core_hard)} (subjects {core_hard}), perm p={p_hard:.4f} -> {primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
