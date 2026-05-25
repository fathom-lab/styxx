# -*- coding: utf-8 -*-
"""The Council. Prereg preregistration_council_2026_05_25.md. Reference-free truth via
inter-model agreement: poll K models, cluster their committed votes by meaning; high
agreement on a substantive answer => likely real, divergence/abstention => likely fake.
Tests C1 (real vs fake AUC>=0.75) and C2 (obscure-real vs fake AUC>=0.70, the holy
grail). Watch for correlated confabulation. Run once."""
from __future__ import annotations
import json, math, re, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(); N = 3
CANDIDATES = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4.1-mini"]
JUDGE_MODEL = "gpt-4o-mini"
emb = SentenceTransformer("all-MiniLM-L6-v2")
PROMPT = "Answer in one short sentence with a specific answer."

QA = [
    ("What is the capital of France?", "real_common"),
    ("What is the chemical symbol for gold?", "real_common"),
    ("Who wrote Romeo and Juliet?", "real_common"),
    ("What planet is known as the Red Planet?", "real_common"),
    ("What is the atomic number of tungsten?", "real_common"),
    ("In what year did World War II end?", "real_common"),
    ("Who won the Nobel Prize in Literature in 1938?", "real_obscure"),
    ("What is the name of the deepest point in the Mediterranean Sea?", "real_obscure"),
    ("Who directed the 1971 film The French Connection?", "real_obscure"),
    ("Who composed the opera La Wally?", "real_obscure"),
    ("Who was the 13th President of the United States?", "real_obscure"),
    ("In what year did the mathematician Evariste Galois die?", "real_obscure"),
    ("What is the SI unit of magnetic flux?", "real_obscure"),
    ("Who painted the Arnolfini Portrait?", "real_obscure"),
    ("What is the capital city of the Republic of Vorland?", "fake"),
    ("In what year did Captain Aldous Renwick first reach the Sundering Isles?", "fake"),
    ("Who won the 1947 Pemberton Literary Prize?", "fake"),
    ("Who composed the 1823 symphony 'The Azure Cascade'?", "fake"),
    ("Which philosopher coined the term 'epistemic verdancy'?", "fake"),
    ("What is the capital of the province of Westmark?", "fake"),
    ("Who was awarded the 1962 Hartwell Medal for physics?", "fake"),
    ("Who wrote the 1932 novel 'The Glass Sentinel'?", "fake"),
]

_ABSTAIN = re.compile(r"\b(no such|not aware|does(?:n'?t| not) (?:exist|appear|have|seem)|"
    r"is not a (?:real|recognized|known|widely)|fictional|fictitious|made[- ]?up|couldn'?t find|"
    r"could not find|cannot find|there (?:is|appears to be) no|i'?m not (?:familiar|sure|certain|aware)|"
    r"no (?:record|information|widely|known|recognized|evidence|data)|not (?:a )?(?:widely )?recognized|"
    r"unable to (?:find|locate|verify)|do not have|don'?t have (?:any |specific )?(?:information|data|record)|"
    r"not certain|uncertain|cannot confirm|no verifiable|may not (?:exist|be (?:a )?real)|doesn'?t seem)\b", re.I)
def is_abstain(t): return bool(_ABSTAIN.search(t or ""))

def gen(model, q, n=1, temp=1.0):
    try:
        r = client.chat.completions.create(model=model, temperature=temp, max_tokens=50, timeout=40, n=n,
            messages=[{"role":"system","content":PROMPT},{"role":"user","content":q}])
        return [(c.message.content or "").strip() for c in r.choices]
    except Exception as e:
        print(f"  !! {model}: {e}", file=sys.stderr); return None

# preflight: keep working models
COUNCIL=[]
for m in CANDIDATES:
    if gen(m, "What is 2+2?", n=1) is not None: COUNCIL.append(m)
print(f"council = {COUNCIL}", file=sys.stderr)
assert len(COUNCIL)>=3, "need >=3 models"

_jc={}
def judge_same(a,b):
    if a.strip()==b.strip(): return True
    k=tuple(sorted((a,b)))
    if k in _jc: return _jc[k]
    try:
        r=client.chat.completions.create(model=JUDGE_MODEL,temperature=0,max_tokens=3,timeout=40,
            messages=[{"role":"system","content":"Reply exactly YES if the two answers give the same core "
                       "factual answer (ignore wording), else NO."},
                      {"role":"user","content":f"A: {a}\nB: {b}\nSame core answer?"}])
        v=(r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v=False
    _jc[k]=v; return v

def modal(xs): return max(set(xs), key=xs.count)
def largest_cluster_frac(votes, same):
    """votes = non-abstention answers; fraction of council in the largest equivalence cluster."""
    if not votes: return 0.0
    reps=[]; counts=[]
    for a in votes:
        for i,rep in enumerate(reps):
            if same(a,rep): counts[i]+=1; break
        else: reps.append(a); counts.append(1)
    return max(counts)/len(COUNCIL)
def cos_same(a,b,thr=0.90):
    va,vb=emb.encode([a,b],normalize_embeddings=True); return float(va@vb)>thr
def auc(pos,neg):
    if not pos or not neg: return float("nan")
    w=sum(1 for p in pos for q in neg if p>q)+0.5*sum(1 for p in pos for q in neg if p==q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x,float) and math.isnan(x))) else round(x,3)

rows=[]
for q, tier in QA:
    votes={}; abst={}
    for m in COUNCIL:
        s=gen(m,q,n=N,temp=1.0)
        if s is None or len(s)==0: continue
        v=modal(s); votes[m]=v; abst[m]=is_abstain(v)
    nonabs=[v for m,v in votes.items() if not abst[m]]
    agree_judge=largest_cluster_frac(nonabs, judge_same)
    agree_cos=largest_cluster_frac(nonabs, cos_same)
    rows.append(dict(tier=tier, q=q[:46], n_abstain=sum(abst.values()), n_votes=len(votes),
                     agree_judge=round(agree_judge,3), agree_cos=round(agree_cos,3),
                     votes=[v[:40] for v in votes.values()]))
    print(f"[{tier}] agree_judge={agree_judge:.2f} agree_cos={agree_cos:.2f} abst={sum(abst.values())}/{len(votes)} :: {q[:40]!r}", file=sys.stderr)

real=[r for r in rows if r["tier"].startswith("real")]
obsc=[r for r in rows if r["tier"]=="real_obscure"]
fake=[r for r in rows if r["tier"]=="fake"]
def report(key):
    c1=auc([r[key] for r in real],[r[key] for r in fake])
    c2=auc([r[key] for r in obsc],[r[key] for r in fake])
    return dict(C1_real_vs_fake=fin(c1), C2_obscure_vs_fake=fin(c2),
                mean_real_common=fin(statistics.fmean([r[key] for r in rows if r["tier"]=="real_common"])),
                mean_real_obscure=fin(statistics.fmean([r[key] for r in obsc])),
                mean_fake=fin(statistics.fmean([r[key] for r in fake])))
J=report("agree_judge"); Cz=report("agree_cos")
C1 = J["C1_real_vs_fake"] is not None and J["C1_real_vs_fake"]>=0.75
C2 = J["C2_obscure_vs_fake"] is not None and J["C2_obscure_vs_fake"]>=0.70
out={"council":COUNCIL, "n_models":len(COUNCIL),
     "by_judge_clustering":J, "by_cosine_clustering":Cz,
     "C1_truth_signal(>=0.75)":[bool(C1), J["C1_real_vs_fake"]],
     "C2_obscure_real_vs_fake(>=0.70)":[bool(C2), J["C2_obscure_vs_fake"]],
     "PASS":bool(C1 and C2)}
pathlib.Path(__file__).parent.joinpath("probe_council_results.json").write_text(json.dumps({"rows":rows,**out},indent=2))
print("\n"+json.dumps(out,indent=2))
