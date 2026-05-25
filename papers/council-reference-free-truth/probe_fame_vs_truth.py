# -*- coding: utf-8 -*-
"""Fame vs Truth. Prereg preregistration_fame_vs_truth_2026_05_25.md. Sweep a RARITY
gradient of real facts (common/obscure/ultra-rare) vs fake; does inter-model agreement
stay high (truth: knowers converge regardless of fame) or collapse toward fake (fame)?
Plus correct-cluster mechanism (do >=2 models converge on the TRUE answer amid scatter?).
Run once."""
from __future__ import annotations
import json, math, re, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI

client = OpenAI(); N = 3
CANDIDATES = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4.1-mini"]
JUDGE_MODEL = "gpt-4o-mini"
PROMPT = "Answer in one short sentence with a specific answer."

QA = [
    ("What is the capital of France?", "Paris", "R0_common"),
    ("What is the chemical symbol for gold?", "Au", "R0_common"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "R0_common"),
    ("What planet is known as the Red Planet?", "Mars", "R0_common"),
    ("What is the atomic number of tungsten?", "74", "R0_common"),
    ("In what year did World War II end?", "1945", "R0_common"),

    ("Who won the Nobel Prize in Literature in 1938?", "Pearl S. Buck", "R1_obscure"),
    ("What is the deepest point in the Mediterranean Sea?", "the Calypso Deep", "R1_obscure"),
    ("Who directed the 1971 film The French Connection?", "William Friedkin", "R1_obscure"),
    ("Who composed the opera La Wally?", "Alfredo Catalani", "R1_obscure"),
    ("Who was the 13th President of the United States?", "Millard Fillmore", "R1_obscure"),
    ("In what year did the mathematician Evariste Galois die?", "1832", "R1_obscure"),

    ("What is the capital of Burkina Faso?", "Ouagadougou", "R2_ultrarare"),
    ("What is the capital of Bhutan?", "Thimphu", "R2_ultrarare"),
    ("What is the capital of Kyrgyzstan?", "Bishkek", "R2_ultrarare"),
    ("What is the capital of Eritrea?", "Asmara", "R2_ultrarare"),
    ("What is the capital of Brunei?", "Bandar Seri Begawan", "R2_ultrarare"),
    ("What is the atomic number of einsteinium?", "99", "R2_ultrarare"),
    ("What is the atomic number of technetium?", "43", "R2_ultrarare"),
    ("What is the currency of Bhutan?", "the ngultrum", "R2_ultrarare"),

    ("What is the capital city of the Republic of Vorland?", None, "fake"),
    ("In what year did Captain Aldous Renwick first reach the Sundering Isles?", None, "fake"),
    ("Who won the 1947 Pemberton Literary Prize?", None, "fake"),
    ("Who composed the 1823 symphony 'The Azure Cascade'?", None, "fake"),
    ("Which philosopher coined the term 'epistemic verdancy'?", None, "fake"),
    ("What is the capital of the province of Westmark?", None, "fake"),
    ("Who was awarded the 1962 Hartwell Medal for physics?", None, "fake"),
    ("Who wrote the 1932 novel 'The Glass Sentinel'?", None, "fake"),
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

COUNCIL=[m for m in CANDIDATES if gen(m,"What is 2+2?",n=1) is not None]
print(f"council = {COUNCIL}", file=sys.stderr); assert len(COUNCIL)>=3

_jc={}
def judge_same(a,b):
    if not a or not b: return False
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
def largest_cluster_frac(votes):
    if not votes: return 0.0
    reps=[]; counts=[]
    for a in votes:
        for i,rep in enumerate(reps):
            if judge_same(a,rep): counts[i]+=1; break
        else: reps.append(a); counts.append(1)
    return max(counts)/len(COUNCIL)
def auc(pos,neg):
    if not pos or not neg: return float("nan")
    w=sum(1 for p in pos for q in neg if p>q)+0.5*sum(1 for p in pos for q in neg if p==q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x,float) and math.isnan(x))) else round(x,3)

rows=[]
for q, ref, tier in QA:
    votes={}
    for m in COUNCIL:
        s=gen(m,q,n=N); votes[m]=modal(s) if s else None
    nonabs=[v for v in votes.values() if v and not is_abstain(v)]
    agree=largest_cluster_frac(nonabs)
    rec=dict(tier=tier, q=q[:44], agree=round(agree,3),
             n_abstain=sum(1 for v in votes.values() if v and is_abstain(v)))
    if ref is not None:
        n_correct=sum(1 for v in votes.values() if v and not is_abstain(v) and judge_same(v, ref))
        rec["n_correct"]=n_correct; rec["correct_cluster"]= n_correct>=2
    rows.append(rec)
    extra=f"corr_cluster={rec.get('correct_cluster')}" if ref is not None else ""
    print(f"[{tier}] agree={agree:.2f} abst={rec['n_abstain']} {extra} :: {q[:38]!r}", file=sys.stderr)

def ag(tier): return [r["agree"] for r in rows if r["tier"]==tier]
curve={t: fin(statistics.fmean(ag(t))) for t in ["R0_common","R1_obscure","R2_ultrarare","fake"]}
r2=ag("R2_ultrarare"); fk=ag("fake")
agree_R2=statistics.fmean(r2); agree_fake=statistics.fmean(fk)
r2_rows=[r for r in rows if r["tier"]=="R2_ultrarare"]
frac_correct_cluster=statistics.fmean([1.0 if r["correct_cluster"] else 0.0 for r in r2_rows])
T1 = (agree_R2>=0.70) and (agree_R2-agree_fake>=0.30)
T2 = frac_correct_cluster>=0.75
out={"council":COUNCIL,"agreement_vs_rarity_curve":curve,
     "auc_R2_vs_fake":fin(auc(r2,fk)),
     "agree_R2_ultrarare":fin(agree_R2),"agree_fake":fin(agree_fake),
     "frac_R2_with_correct_cluster":fin(frac_correct_cluster),
     "T1_truth_not_fame(R2>=0.70 & R2-fake>=0.30)":[bool(T1), fin(agree_R2), fin(agree_R2-agree_fake)],
     "T2_correct_cluster_persists(>=0.75)":[bool(T2), fin(frac_correct_cluster)],
     "PASS":bool(T1 and T2)}
pathlib.Path(__file__).parent.joinpath("probe_fame_vs_truth_results.json").write_text(json.dumps({"rows":rows,**out},indent=2))
print("\n"+json.dumps(out,indent=2))
