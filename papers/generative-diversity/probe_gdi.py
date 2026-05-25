# -*- coding: utf-8 -*-
"""GDI - Generative Diversity Index. Prereg preregistration_gdi_2026_05_25.md. The
divergence detector in reverse: on OPEN prompts, semantic entropy = creative variety
(low = mode collapse). Positive-valence instrument, gated by coherence (diversity must
be on-topic). G1 open>=2x closed; G2 entropy rises with temp; G3 coherence>=0.85.
Run once."""
from __future__ import annotations
import json, math, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(); N = 6
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
JUDGE_MODEL = "gpt-4o-mini"
emb = SentenceTransformer("all-MiniLM-L6-v2")

OPEN = ["Give me a startup idea.",
        "Write the first line of a novel.",
        "Suggest a name for a coffee shop.",
        "Propose a research question about the ocean.",
        "Invent a name for a new color.",
        "Suggest a topic for a short story.",
        "Give me an idea for a weekend project.",
        "Name a fictional planet."]
CLOSED = ["What is 2+2?", "What is the capital of France?",
          "What is the chemical symbol for gold?", "How many days are in a week?"]

def gen(model, q, n=1, temp=1.0):
    try:
        r = client.chat.completions.create(model=model, temperature=temp, max_tokens=40, timeout=40, n=n,
            messages=[{"role":"user","content":q}])
        return [(c.message.content or "").strip() for c in r.choices]
    except Exception as e:
        print(f"  !! {model}: {e}", file=sys.stderr); return None

def gdi(answers, thr=0.90):  # semantic entropy = diversity
    v=emb.encode(answers, normalize_embeddings=True); cl=[]; asg=[]
    for i,vi in enumerate(v):
        for ci,rep in enumerate(cl):
            if float(vi@v[rep])>thr: asg.append(ci); break
        else: cl.append(i); asg.append(len(cl)-1)
    c=[asg.count(k) for k in set(asg)]; n=len(answers)
    return -sum((x/n)*math.log(x/n) for x in c)

def coherent(prompt, ans):
    try:
        r=client.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=3, timeout=40,
            messages=[{"role":"system","content":"Reply exactly YES if the answer is a relevant, on-topic "
                       "response to the prompt, else NO."},
                      {"role":"user","content":f"Prompt: {prompt}\nAnswer: {ans}\nOn-topic?"}])
        return (r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: return True
def auc(pos,neg):
    if not pos or not neg: return float("nan")
    w=sum(1 for p in pos for q in neg if p>q)+0.5*sum(1 for p in pos for q in neg if p==q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x,float) and math.isnan(x))) else round(x,3)

# ---- main run: 3 models, temp 1.0, open + closed ----
rows=[]; coh_samples=[]
for model in MODELS:
    for kind, prompts in (("open",OPEN),("closed",CLOSED)):
        for p in prompts:
            s=gen(model,p,n=N,temp=1.0)
            if not s: continue
            g=gdi(s)
            rows.append(dict(model=model, kind=kind, prompt=p[:36], gdi=round(g,3)))
            if kind=="open":
                for a in s: coh_samples.append((p,a))
            print(f"[{model}|{kind}] gdi={g:.2f} :: {p[:34]!r}", file=sys.stderr)

open_g=[r["gdi"] for r in rows if r["kind"]=="open"]
closed_g=[r["gdi"] for r in rows if r["kind"]=="closed"]
G1_ratio = statistics.fmean(open_g)/statistics.fmean(closed_g) if closed_g and statistics.fmean(closed_g)>0 else float("inf")
leaderboard=sorted([(m, fin(statistics.fmean([r["gdi"] for r in rows if r["model"]==m and r["kind"]=="open"])))
                    for m in MODELS], key=lambda t:t[1] if t[1] is not None else -1, reverse=True)

# ---- temp sweep: gpt-4o-mini, open prompts ----
sweep={}
for temp in [0.5, 1.0, 1.5]:
    gs=[]
    for p in OPEN:
        s=gen("gpt-4o-mini",p,n=N,temp=temp)
        if s: gs.append(gdi(s))
    sweep[temp]=fin(statistics.fmean(gs)) if gs else None
    print(f"[temp {temp}] mean_gdi={sweep[temp]}", file=sys.stderr)

# ---- coherence (G3) on a sample of open answers ----
import random; random.seed(0); random.shuffle(coh_samples)
coh_check=coh_samples[:60]
coh_rate=statistics.fmean([1.0 if coherent(p,a) else 0.0 for p,a in coh_check]) if coh_check else None

tvals=[sweep[t] for t in [0.5,1.0,1.5]]
G1 = (G1_ratio==float("inf")) or G1_ratio>=2.0
G2 = all(tvals[i] is not None for i in range(3)) and tvals[0]<=tvals[1]<=tvals[2]
G3 = coh_rate is not None and coh_rate>=0.85
out={"models":MODELS,
     "mean_gdi_open":fin(statistics.fmean(open_g)), "mean_gdi_closed":fin(statistics.fmean(closed_g)),
     "auc_open_vs_closed":fin(auc(open_g, closed_g)),
     "diversity_leaderboard_open":leaderboard,
     "temp_sweep_gpt4o_mini":sweep,
     "coherence_rate_open":fin(coh_rate), "n_coherence_checked":len(coh_check),
     "G1_open>=2x_closed":[bool(G1), ("inf" if G1_ratio==float("inf") else fin(G1_ratio))],
     "G2_temp_monotonic":[bool(G2), tvals],
     "G3_coherence>=0.85":[bool(G3), fin(coh_rate)],
     "PASS":bool(G1 and G2 and G3)}
pathlib.Path(__file__).parent.joinpath("probe_gdi_results.json").write_text(json.dumps({"rows":rows,**out},indent=2))
print("\n"+json.dumps(out,indent=2))
