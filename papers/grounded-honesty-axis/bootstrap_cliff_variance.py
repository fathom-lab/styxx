"""Bootstrap confidence intervals on the cross-family / cross-vendor competence-cliff Spearman numbers.

The genmatch run exposed that these agreements have NO variance bars (within-open moved 0.77->0.61 on
re-sampling). This puts CIs on them from per-item data already on disk — no GPU, no new runs.

Hierarchical bootstrap (resample the 37 domains with replacement; within each, resample its items with
replacement), recompute per-domain ungated-hallucination rate per model, recompute the pairwise Spearman.
Reports point estimate + 95% CI for within-open and open-closed, at 24-token and 32-token apparatus, and
whether the 24-vs-32 within-open difference is within overlapping CIs (i.e. attributable to noise).
"""
from __future__ import annotations
import json, random, statistics as st
from pathlib import Path
import run_pregeneration_gate as G

HERE = Path(__file__).resolve().parent
B = 2000
SEED = 0

BENCH = {
    "open24": {"Qwen": "crossfamily_benchmark_Qwen2_5_3B_Instruct.json",
               "Llama": "crossfamily_benchmark_Llama_3_2_3B_Instruct.json",
               "gemma": "crossfamily_benchmark_gemma_2_2b_it.json"},
    "open32": {"Qwen": "crossfamily_benchmark_Qwen2_5_3B_Instruct_gm32.json",
               "Llama": "crossfamily_benchmark_Llama_3_2_3B_Instruct_gm32.json",
               "gemma": "crossfamily_benchmark_gemma_2_2b_it_gm32.json"},
    "gpt":    {"OpenAI": "xvendor_gpt4omini_nli_benchmark.json"},
}

def rank(a):
    idx=sorted(range(len(a)),key=lambda i:a[i]); r=[0.0]*len(a); i=0
    while i<len(a):
        j=i
        while j+1<len(a) and a[idx[j+1]]==a[idx[i]]: j+=1
        av=(i+j)/2.0+1
        for k in range(i,j+1): r[idx[k]]=av
        i=j+1
    return r
def pear(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    cov=sum((a-mx)*(b-my) for a,b in zip(x,y)); vx=sum((a-mx)**2 for a in x); vy=sum((b-my)**2 for b in y)
    return float("nan") if vx==0 or vy==0 else cov/((vx*vy)**0.5)
def sp(x,y): return pear(rank(x),rank(y))

def per_domain_correct(path):
    """domain -> list[int 0/1] ungated-correct per item (modal cluster matches the true answer)."""
    d = json.load(open(HERE / path, encoding="utf-8"))
    out = {}
    for it in d["items"]:
        cat = it.get("category") or "Unknown"
        modal_is_true = G._modal_cluster_info(it)[3]
        out.setdefault(cat, []).append(1 if modal_is_true else 0)
    return out

def rates_from(correct_by_dom, domains, rng=None):
    """per-domain hallucination rate (1 - mean correct); optional item bootstrap within domain."""
    r = {}
    for dom in domains:
        items = correct_by_dom.get(dom, [])
        if not items: continue
        if rng is not None:
            items = [items[rng.randrange(len(items))] for _ in items]
        r[dom] = 1.0 - (sum(items)/len(items))
    return r

def mean_pair_spearman(models_rates, group_a, group_b, domains):
    vals = []
    for a in group_a:
        for b in group_b:
            if a == b: continue
            shared = [d for d in domains if d in models_rates[a] and d in models_rates[b]]
            if len(shared) < 4: continue
            vals.append(sp([models_rates[a][d] for d in shared], [models_rates[b][d] for d in shared]))
    return st.mean(vals) if vals else float("nan")

def ci(xs):
    xs = sorted(v for v in xs if v == v)
    n = len(xs)
    return xs[int(0.025*n)], xs[int(0.975*n)], st.mean(xs)

def run_set(open_key):
    cb = {m: per_domain_correct(p) for m, p in BENCH[open_key].items()}
    cb.update({m: per_domain_correct(p) for m, p in BENCH["gpt"].items()})
    opens = list(BENCH[open_key]); gpt = list(BENCH["gpt"])
    all_domains = sorted(set().union(*[set(v) for v in cb.values()]))
    rng = random.Random(SEED)
    res = {}
    for mode in ("item", "hier"):   # item: domains fixed, resample items within (precision of THIS run);
        oo, oc = [], []             # hier: also resample domains (generalization to other domains)
        for _ in range(B):
            dboot = ([all_domains[rng.randrange(len(all_domains))] for _ in all_domains]
                     if mode == "hier" else all_domains)
            rates = {m: rates_from(cb[m], dboot, rng) for m in cb}
            oo.append(mean_pair_spearman(rates, opens, opens, dboot))
            oc.append(mean_pair_spearman(rates, opens, gpt, dboot))
        res[f"within_open_ci_{mode}"] = [round(x,3) for x in ci(oo)[:2]]
        res[f"open_closed_ci_{mode}"] = [round(x,3) for x in ci(oc)[:2]]
    pr = {m: rates_from(cb[m], all_domains) for m in cb}
    res["within_open_point"] = round(mean_pair_spearman(pr, opens, opens, all_domains),3)
    res["open_closed_point"] = round(mean_pair_spearman(pr, opens, gpt, all_domains),3)
    return res

def main():
    res = {"B": B, "24token": run_set("open24"), "32token": run_set("open32")}
    print("="*70)
    print(f"BOOTSTRAP 95pct CI on the hallucination-cliff Spearman  (hierarchical, B={B})")
    print("="*70)
    for tok in ("24token","32token"):
        r = res[tok]
        print(f"\n[{tok}]")
        print(f"  within-open {r['within_open_point']:.3f}  CI item {r['within_open_ci_item']}  hier {r['within_open_ci_hier']}")
        print(f"  open-closed {r['open_closed_point']:.3f}  CI item {r['open_closed_ci_item']}  hier {r['open_closed_ci_hier']}")
    a, b = res["24token"]["within_open_ci_item"], res["32token"]["within_open_ci_item"]
    overlap = not (a[1] < b[0] or b[1] < a[0])
    res["within_open_24_vs_32_ci_overlap"] = overlap
    print("\n" + "-"*70)
    print(f"within-open 24-tok CI {a}  vs  32-tok CI {b}  ->  {'OVERLAP' if overlap else 'DISJOINT'}")
    print("OVERLAP => the 0.77->0.61 drop is within sampling noise (cannot attribute to the token change).")
    print("DISJOINT => the drop exceeds item/domain sampling noise (token effect or sample-draw variance).")
    (HERE / "bootstrap_cliff_variance_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwritten: bootstrap_cliff_variance_result.json")

if __name__ == "__main__":
    main()
