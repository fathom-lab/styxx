# -*- coding: utf-8 -*-
"""Harden the correction's MECHANISM (no new API; reuses verify_clustering raw samples).
The correction said cosine masks confabulation divergence vs NLI 0.95. But is that
a STRUCTURAL cosine failure, or just a too-low THRESHOLD (0.70)? Correct answers here
are near-verbatim (cosine~1.0); lies share a template (cosine~0.97). If the divergence
lives in the 0.97-1.0 band, a higher cosine threshold might recover it — which would
make the honest lesson 'tune the threshold', not 'cosine can't do it'. Sweep cosine
thresholds, compare to NLI, and characterize the similarity bands. Run once."""
from __future__ import annotations
import json, math, statistics, pathlib
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("all-MiniLM-L6-v2")
D = json.loads(pathlib.Path(__file__).parent.joinpath("verify_clustering_results.json").read_text())
rows = D["rows"]

def cos_entropy(answers, thr):
    v = emb.encode(answers, normalize_embeddings=True); clusters=[]; assign=[]
    for i, vi in enumerate(v):
        for ci, rep in enumerate(clusters):
            if float(vi @ v[rep]) > thr: assign.append(ci); break
        else: clusters.append(i); assign.append(len(clusters)-1)
    counts=[assign.count(c) for c in set(assign)]; n=len(answers)
    return -sum((c/n)*math.log(c/n) for c in counts)

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w=sum(1 for p in pos for q in neg if p>q)+0.5*sum(1 for p in pos for q in neg if p==q)
    return w/(len(pos)*len(neg))

inc_idx=[i for i,r in enumerate(rows) if not r["correct"]]
cor_idx=[i for i,r in enumerate(rows) if r["correct"]]

print(f"n_incorrect(confab)={len(inc_idx)}  n_correct={len(cor_idx)}\n")
print("cosine threshold sweep — AUC(entropy -> confabulation):")
sweep={}
for thr in [0.50,0.60,0.70,0.80,0.90,0.95,0.97,0.98,0.99]:
    ents=[cos_entropy(r["answers"], thr) for r in rows]
    a=auc([ents[i] for i in inc_idx],[ents[i] for i in cor_idx])
    sweep[thr]=round(a,3)
    mc=statistics.fmean([ents[i] for i in inc_idx]); mco=statistics.fmean([ents[i] for i in cor_idx])
    print(f"  thr={thr:.2f}  AUC={a:.3f}   mean_ent confab={mc:.2f} correct={mco:.2f}")

auc_nli=auc([rows[i]["ent_nli"] for i in inc_idx],[rows[i]["ent_nli"] for i in cor_idx])
print(f"\nNLI entailment clustering — AUC={auc_nli:.3f}  (threshold-free)")

# similarity-band characterization: mean within-item pairwise cosine per group
def mean_pairwise_cos(answers):
    v=emb.encode(answers, normalize_embeddings=True); n=len(v); s=[]
    for i in range(n):
        for j in range(i+1,n):
            s.append(float(v[i]@v[j]))
    return statistics.fmean(s) if s else float("nan")

groups={"real (verbatim fact)":[r for r in rows if r["diff"]=="real"],
        "fictional CONFAB (incorrect)":[r for r in rows if r["diff"]=="fictional" and not r["correct"]],
        "fictional ABSTAIN (correct)":[r for r in rows if r["diff"]=="fictional" and r["correct"]]}
print("\nmean within-item pairwise cosine (where the divergence lives):")
for name, g in groups.items():
    mcs=[mean_pairwise_cos(r["answers"]) for r in g]
    print(f"  {name:32s} n={len(g)}  mean_pairwise_cos={statistics.fmean(mcs):.3f}  range=[{min(mcs):.3f},{max(mcs):.3f}]")

best_thr=max(sweep, key=lambda k: (sweep[k] if not math.isnan(sweep[k]) else -1))
out={"cosine_sweep_auc":sweep, "auc_nli":round(auc_nli,3),
     "best_cosine_auc":sweep[best_thr], "best_cosine_threshold":best_thr,
     "nli_beats_best_cosine":round(auc_nli - sweep[best_thr],3)}
pathlib.Path(__file__).parent.joinpath("analyze_clustering_threshold_results.json").write_text(json.dumps(out,indent=2))
print("\n"+json.dumps(out,indent=2))
