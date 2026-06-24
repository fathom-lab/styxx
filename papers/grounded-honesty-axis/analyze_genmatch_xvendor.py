"""Generation-matched cross-vendor cliff analysis (PREREG_genmatch_xvendor_2026_06_23).

Compares per-domain hallucination/refusal-cliff Spearman across the 24-token (committed) and the
32-token (generation-matched, _gm32) open-family gates, vs gpt-4o-mini's matched-NLI gate. Prints the
PRIMARY bar Δ = (gm32 open<->closed hallucination Spearman) - 0.473. No API key.
Run AFTER run_genmatch_cliff.py produces the three crossfamily_gate_*_gm32.json.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMS = ["Qwen2_5_3B_Instruct", "Llama_3_2_3B_Instruct", "gemma_2_2b_it"]
GPT_MATCHED = HERE / "xvendor_gpt4omini_nli_gate.json"
SIGNALS = ["ungated_hallucination_rate", "refusal_rate"]
BASELINE_24 = {"open_open_hall": 0.770, "open_closed_hall": 0.473}  # committed eb115eb

def rankdata(a):
    idx=sorted(range(len(a)),key=lambda i:a[i]); r=[0.0]*len(a); i=0
    while i<len(a):
        j=i
        while j+1<len(a) and a[idx[j+1]]==a[idx[i]]: j+=1
        avg=(i+j)/2.0+1
        for k in range(i,j+1): r[idx[k]]=avg
        i=j+1
    return r
def pearson(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    cov=sum((a-mx)*(b-my) for a,b in zip(x,y)); vx=sum((a-mx)**2 for a in x); vy=sum((b-my)**2 for b in y)
    return float("nan") if vx==0 or vy==0 else cov/((vx*vy)**0.5)
def sp(x,y): return pearson(rankdata(x),rankdata(y))
def load_map(p): return json.load(open(p,encoding="utf-8"))["category_competence_cliff_map"]
def pair(mA,mB,sig):
    sh=[d for d in mA if d in mB]; return sp([mA[d][sig] for d in sh],[mB[d][sig] for d in sh]), len(sh)

def open_closed(open_maps, gpt, sig):
    vals=[pair(m,gpt,sig)[0] for m in open_maps]; return sum(vals)/len(vals), vals
def open_open(open_maps, sig):
    vals=[pair(open_maps[i],open_maps[j],sig)[0] for i,j in itertools.combinations(range(len(open_maps)),2)]
    return sum(vals)/len(vals), vals

def main():
    gpt = load_map(GPT_MATCHED)
    gm32 = [load_map(HERE / f"crossfamily_gate_{f}_gm32.json") for f in FAMS]
    tok24 = [load_map(HERE / f"crossfamily_gate_{f}.json") for f in FAMS]

    print("="*72)
    print("GENERATION-MATCHED CROSS-VENDOR CLIFF  (open families re-sampled at max_new=32)")
    print("="*72)
    rows={}
    for tag, omaps in (("24-token (committed)", tok24), ("32-token (gen-matched)", gm32)):
        oo_h,_=open_open(omaps,"ungated_hallucination_rate"); oo_r,_=open_open(omaps,"refusal_rate")
        oc_h,ocv=open_closed(omaps,gpt,"ungated_hallucination_rate"); oc_r,_=open_closed(omaps,gpt,"refusal_rate")
        rows[tag]=dict(oo_h=oo_h,oo_r=oo_r,oc_h=oc_h,oc_r=oc_r,oc_h_perfam=ocv)
        print(f"\n{tag}")
        print(f"  open<->open   hallucination {oo_h:.3f}  refusal {oo_r:.3f}")
        print(f"  open<->closed hallucination {oc_h:.3f}  refusal {oc_r:.3f}   per-family hall {[round(v,3) for v in ocv]}")

    delta = rows["32-token (gen-matched)"]["oc_h"] - BASELINE_24["open_closed_hall"]
    oo32 = rows["32-token (gen-matched)"]["oo_h"]
    sanity_drift = abs(oo32 - BASELINE_24["open_open_hall"])
    sanity_ok = sanity_drift <= 0.05   # within-open must be STABLE for any primary verdict to be trustworthy
    print("\n" + "-"*72)
    print(f"PRIMARY bar  delta = (gm32 open<->closed hallucination) - 0.473 = {delta:+.3f}")
    print(f"SECONDARY (sanity): open<->open at 32 = {oo32:.3f}  (24-token was {BASELINE_24['open_open_hall']:.3f}; expect ±0.05; drift {sanity_drift:+.3f})")
    # 2026-06-24 self-audit fix: the primary verdict is only trustworthy if the SANITY bar passes. A failed
    # within-open stability check means the metric is noise-dominated, so "RESIDUAL ROBUST" cannot be claimed.
    if not sanity_ok:
        verdict = (f"INCONCLUSIVE — within noise. Secondary within-open sanity bar FAILED (drift {sanity_drift:+.3f} "
                   f"> 0.05): the metric is noise-dominated single-run, so the primary delta {delta:+.3f} (no CI) "
                   f"cannot support a 'real vendor divergence' claim. cf bootstrap_cliff_variance (wide overlapping CIs).")
    elif abs(delta) < 0.05:
        verdict = "RESIDUAL (sanity passed) — open<->closed gap holds under gen-matching; report WITH a bootstrap CI before any 'real divergence' claim."
    elif delta >= 0.10:
        verdict = "APPARATUS DRIVER — max_new_tokens moved the gap toward 0.77; residual was partly apparatus. Correct the finding."
    else:
        verdict = "PARTIAL apparatus contribution (0.05<=|Δ|<0.10)."
    print("VERDICT:", verdict)

    out = HERE / "genmatch_xvendor_result.json"
    out.write_text(json.dumps({"rows":{k:{kk:vv for kk,vv in v.items()} for k,v in rows.items()},
                               "primary_delta":round(delta,4),"verdict":verdict,
                               "sanity_drift":round(sanity_drift,4),"sanity_ok":bool(sanity_ok),
                               "corrigendum":"2026-06-24 self-audit: verdict is now gated on the within-open sanity bar; the original 'RESIDUAL ROBUST / REAL vendor divergence' was an overclaim (no CIs; sanity bar failed; contradicted the finding's 'largely apparatus' title). Raw numbers unchanged.",
                               "baseline_24":BASELINE_24,"signals":SIGNALS,"families":FAMS}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")

if __name__ == "__main__":
    main()
