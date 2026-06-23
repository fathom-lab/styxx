"""4-provider cross-vendor competence-cliff agreement matrix — from data already on disk, no API key.

Providers: Alibaba (Qwen-3B), Meta (Llama-3B), Google (Gemma-2B) open-weights + OpenAI (gpt-4o-mini)
closed. All NLI-judged (matched judge). For the open families, prefers the generation-matched (_gm32)
gates if present (so this auto-finalizes once run_genmatch_cliff.py lands), else the committed 24-token
gates. Writes provider_cliff_matrix_result.json.
"""
from __future__ import annotations
import json, itertools, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN = {"Alibaba (Qwen-3B)": "Qwen2_5_3B_Instruct",
        "Meta (Llama-3B)": "Llama_3_2_3B_Instruct",
        "Google (Gemma-2B)": "gemma_2_2b_it"}
GPT = "OpenAI (gpt-4o-mini)"

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
def cmap(p): return json.load(open(p,encoding="utf-8"))["category_competence_cliff_map"]

def open_gate(slug):
    gm = HERE / f"crossfamily_gate_{slug}_gm32.json"
    return (gm, "gen-matched(32)") if gm.exists() else (HERE / f"crossfamily_gate_{slug}.json", "matched-judge(24)")

def main():
    maps, prov_mode = {}, {}
    for name, slug in OPEN.items():
        p, mode = open_gate(slug); maps[name] = cmap(p); prov_mode[name] = mode
    maps[GPT] = cmap(HERE / "xvendor_gpt4omini_nli_gate.json"); prov_mode[GPT] = "matched-judge"
    names = list(maps)

    result = {"providers": names, "open_gate_mode": prov_mode, "matrices": {}}
    for sig in ("ungated_hallucination_rate", "refusal_rate"):
        mat = {}
        for a, b in itertools.combinations(names, 2):
            sh = [d for d in maps[a] if d in maps[b]]
            mat[f"{a} | {b}"] = round(sp([maps[a][d][sig] for d in sh], [maps[b][d][sig] for d in sh]), 4)
        openp = [v for k, v in mat.items() if GPT not in k]
        ocp = [v for k, v in mat.items() if GPT in k]
        result["matrices"][sig] = {"pairs": mat,
                                   "open_open_mean": round(st.mean(openp), 4),
                                   "open_openai_mean": round(st.mean(ocp), 4),
                                   "all_pairs_mean": round(st.mean(list(mat.values())), 4)}
        print(f"\n[{sig}]  open-open {result['matrices'][sig]['open_open_mean']:.3f} | "
              f"open-OpenAI {result['matrices'][sig]['open_openai_mean']:.3f} | "
              f"all-6 {result['matrices'][sig]['all_pairs_mean']:.3f}")
        for k, v in mat.items(): print(f"    {v:+.2f}  {k}")
    print("\nopen-gate mode:", prov_mode)
    (HERE / "provider_cliff_matrix_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("written: provider_cliff_matrix_result.json")

if __name__ == "__main__":
    main()
