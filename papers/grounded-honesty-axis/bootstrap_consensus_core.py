"""Is the cross-vendor consensus failure-core STABLE, or a single-draw fluke?

The permutation test (consensus_failure_core.py) shows the 4-way overlap is above chance; the variance
finding (bootstrap_cliff_variance.py) shows per-domain rates are noisy. This bootstraps the CORE itself:
resample items within each domain (per provider), recompute each provider's bottom-K / top-K, recompute
the 4-way intersection, and track how often each domain stays in the consensus core. A stable core
(domains that keep appearing) is a far stronger claim than one draw. Per-item ungated-correctness via
run_pregeneration_gate._modal_cluster_info. No GPU/API.
"""
from __future__ import annotations
import json, random, statistics as st
from pathlib import Path
import run_pregeneration_gate as G

HERE = Path(__file__).resolve().parent
# matched-judge 24-token open gates + gpt-4o-mini matched (same set as consensus_failure_core.py)
BENCH = {"Alibaba(Qwen)": "crossfamily_benchmark_Qwen2_5_3B_Instruct.json",
         "Meta(Llama)": "crossfamily_benchmark_Llama_3_2_3B_Instruct.json",
         "Google(Gemma)": "crossfamily_benchmark_gemma_2_2b_it.json",
         "OpenAI(4o-mini)": "xvendor_gpt4omini_nli_benchmark.json"}
K = 6
B = 2000
SEED = 0


def per_domain_correct(path):
    d = json.load(open(HERE / path, encoding="utf-8"))
    out = {}
    for it in d["items"]:
        cat = it.get("category") or "Unknown"
        out.setdefault(cat, []).append(1 if G._modal_cluster_info(it)[3] else 0)
    return out


def halluc_rates(correct_by_dom, domains, rng=None):
    r = {}
    for dom in domains:
        items = correct_by_dom.get(dom, [])
        if not items:
            continue
        if rng is not None:
            items = [items[rng.randrange(len(items))] for _ in items]
        r[dom] = 1.0 - sum(items) / len(items)
    return r


def core(rate_maps, domains, k, hardest=True):
    sets = []
    for m in rate_maps:
        order = sorted(domains, key=lambda d: (-m[d] if hardest else m[d]))
        sets.append(set(order[:k]))
    return set.intersection(*sets)


def main():
    cb = {p: per_domain_correct(f) for p, f in BENCH.items()}
    domains = sorted(set.intersection(*[set(v) for v in cb.values()]))
    maps_full = [halluc_rates(cb[p], domains) for p in cb]
    observed_hard = sorted(core(maps_full, domains, K, True))
    observed_easy = sorted(core(maps_full, domains, K, False))

    rng = random.Random(SEED)
    from collections import Counter
    freq_hard, freq_easy, sizes_hard = Counter(), Counter(), []
    for _ in range(B):
        maps = [halluc_rates(cb[p], domains, rng) for p in cb]
        ch = core(maps, domains, K, True)
        ce = core(maps, domains, K, False)
        sizes_hard.append(len(ch))
        for d in ch: freq_hard[d] += 1
        for d in ce: freq_easy[d] += 1

    def stab(observed, freq):
        return {d: round(freq[d] / B, 3) for d in observed}

    out = {"providers": list(BENCH), "K": K, "B": B,
           "observed_consensus_hardest": observed_hard,
           "observed_consensus_safest": observed_easy,
           "hardest_core_stability": stab(observed_hard, freq_hard),
           "safest_core_stability": stab(observed_easy, freq_easy),
           "any_domain_in_hardest_core_freq": {d: round(c / B, 3) for d, c in freq_hard.most_common(8)},
           "hardest_overlap_size_mean": round(st.mean(sizes_hard), 3),
           "hardest_overlap_size_p2.5_97.5": [sorted(sizes_hard)[int(0.025 * B)], sorted(sizes_hard)[int(0.975 * B)]],
           "note": "Stability = fraction of item-bootstraps in which the observed core domain stays in the 4-way intersection. High = robust, not a single-draw fluke."}
    print(json.dumps(out, indent=2))
    (HERE / "bootstrap_consensus_core_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
