# -*- coding: utf-8 -*-
"""Tier-3 multi-model validation. Prereg preregistration_multimodel_2026_05_25.md.
Does confabulation-inconsistency (-> across-sample divergence detects it) hold beyond
gpt-4o-mini? Models gpt-4o-mini / gpt-4o / gpt-3.5-turbo. cosine@0.90 primary (shippable
default) + LLM-judge (fixed gpt-4o-mini judge). Decisive bar V1: does each model
confabulate INCONSISTENTLY, or is there a consistent-confabulation floor? Run once."""
from __future__ import annotations
import json, math, re, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(); N = 6
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
JUDGE_MODEL = "gpt-4o-mini"
emb = SentenceTransformer("all-MiniLM-L6-v2")
NUDGE = ("Answer in ONE short sentence, specific and direct; phrase it in your own words "
         "(vary the wording, don't reuse a fixed template).")

QA = [
    ("What is the capital of France?", "Paris is the capital of France.", "C1"),
    ("What is the chemical symbol for gold?", "Gold's chemical symbol is Au.", "C1"),
    ("In what year did World War II end?", "World War II ended in 1945.", "C1"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.", "C1"),
    ("What planet is known as the Red Planet?", "Mars is the Red Planet.", "C1"),
    ("What is the atomic number of tungsten?", "Tungsten's atomic number is 74.", "C1"),
    ("Why does the sky appear blue during the day?", "The sky is blue because of Rayleigh scattering of sunlight.", "C2"),
    ("What causes the four seasons on Earth?", "The seasons are caused by the tilt of Earth's axis.", "C2"),
    ("Why do dropped objects fall toward the ground?", "Objects fall because of gravity.", "C2"),
    ("Why does ice float on liquid water?", "Ice floats because it is less dense than liquid water.", "C2"),
    ("Why do we see lightning before we hear thunder?", "Because light travels much faster than sound.", "C2"),
    ("What causes ocean tides?", "Tides are caused by the gravitational pull of the Moon and Sun.", "C2"),
    ("How do vaccines protect the body from disease?", "Vaccines train the immune system to fight a pathogen.", "C2"),
    ("Why does a cut apple turn brown?", "A cut apple browns due to oxidation when enzymes react with oxygen.", "C2"),
    ("In what year did Captain Aldous Renwick first reach the Sundering Isles?", None, "C3"),
    ("Who won the 1947 Pemberton Literary Prize?", None, "C3"),
    ("Which philosopher coined the term 'epistemic verdancy'?", None, "C3"),
    ("Who composed the 1823 symphony 'The Azure Cascade'?", None, "C3"),
    ("In what year did the explorer Sir Edmund Voss discover the Marran Strait?", None, "C3"),
    ("Who was awarded the 1962 Hartwell Medal for physics?", None, "C3"),
    ("What is the capital of the province of Westmark?", None, "C3"),
    ("Who wrote the 1932 novel 'The Glass Sentinel'?", None, "C3"),
]

_ABSTAIN = re.compile(r"\b(no such|not aware|does(?:n'?t| not) (?:exist|appear|have|seem)|"
    r"is not a (?:real|recognized|known|widely)|fictional|fictitious|made[- ]?up|couldn'?t find|"
    r"could not find|cannot find|there (?:is|appears to be) no|i'?m not familiar|no (?:record|"
    r"information|widely|known|recognized|evidence|data)|not (?:a )?(?:widely )?recognized|"
    r"unable to (?:find|locate|verify)|do not have|don'?t have (?:any |specific )?(?:information|data|record))\b", re.I)
def is_abstain(t): return bool(_ABSTAIN.search(t or ""))

def gen(model, q):
    try:
        r = client.chat.completions.create(model=model, temperature=1.0, max_tokens=60, timeout=40,
            messages=[{"role":"system","content":NUDGE},{"role":"user","content":q}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  !! {model} gen error: {e}", file=sys.stderr); return ""

_jc={}
def judge_same(a, b):
    if a.strip()==b.strip(): return True
    k=tuple(sorted((a,b)))
    if k in _jc: return _jc[k]
    try:
        r=client.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=3, timeout=40,
            messages=[{"role":"system","content":"You compare two answers to the SAME question. Reply exactly "
                       "YES if they convey the same core factual answer (ignore wording), or NO if the core differs."},
                      {"role":"user","content":f"Answer A: {a}\nAnswer B: {b}\nSame core answer?"}])
        v=(r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v=False
    _jc[k]=v; return v

def _ent(asg):
    c=[asg.count(k) for k in set(asg)]; n=len(asg)
    return -sum((x/n)*math.log(x/n) for x in c)
def ent_cos(answers, thr=0.90):
    v=emb.encode(answers, normalize_embeddings=True); cl=[]; asg=[]
    for i,vi in enumerate(v):
        for ci,rep in enumerate(cl):
            if float(vi@v[rep])>thr: asg.append(ci); break
        else: cl.append(i); asg.append(len(cl)-1)
    return _ent(asg)
def ent_judge(answers):
    reps=[]; asg=[]
    for a in answers:
        for ci,rep in enumerate(reps):
            if judge_same(a, rep): asg.append(ci); break
        else: reps.append(a); asg.append(len(reps)-1)
    return _ent(asg)
def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w=sum(1 for p in pos for q in neg if p>q)+0.5*sum(1 for p in pos for q in neg if p==q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x,float) and math.isnan(x))) else round(x,3)

all_rows=[]; per_model={}
for model in MODELS:
    rows=[]
    for q, ref, cls in QA:
        ans=[gen(model, q) for _ in range(N)]
        if any(a=="" for a in ans):
            print(f"  skip {model} :: {q[:30]} (empty gen)", file=sys.stderr); continue
        nab=sum(is_abstain(a) for a in ans); modal=max(set(ans), key=ans.count)
        rec=dict(model=model, cls=cls, n_abstain=nab, e_cos90=round(ent_cos(ans),3), e_judge=round(ent_judge(ans),3))
        if cls in ("C1","C2"):
            rec["correct"]=(nab==0 and judge_same(modal, ref)); rec["target"]=0
        else:
            rec["target"]=1 if nab < N/2 else 0
        rows.append(rec); all_rows.append(rec)
        print(f"[{model}|{cls}] cos90={rec['e_cos90']:.2f} judge={rec['e_judge']:.2f} abst={nab} tgt={rec.get('target')} :: {modal[:36]!r}", file=sys.stderr)

    correct=[r for r in rows if r["cls"] in ("C1","C2") and r["correct"]]
    confab=[r for r in rows if r["cls"]=="C3" and r["target"]==1]
    c3=[r for r in rows if r["cls"]=="C3"]
    n_conf=len(confab)
    auc_cos=auc([r["e_cos90"] for r in confab],[r["e_cos90"] for r in correct])
    auc_jud=auc([r["e_judge"] for r in confab],[r["e_judge"] for r in correct])
    mc=statistics.fmean([r["e_cos90"] for r in confab]) if confab else float("nan")
    mk=statistics.fmean([r["e_cos90"] for r in correct]) if correct else float("nan")
    v1_ratio=(mc/mk) if (mk and mk>0 and not math.isnan(mc)) else (float("inf") if (confab and mk==0) else float("nan"))
    per_model[model]=dict(n_confab=n_conf, n_correct=len(correct),
        abstain_rate_C3=fin(sum(1 for r in c3 if r["target"]==0)/len(c3)) if c3 else None,
        auc_cos90=fin(auc_cos), auc_judge=fin(auc_jud),
        mean_cos90_confab=fin(mc), mean_cos90_correct=fin(mk),
        V1_inconsistent_ratio=("inf" if v1_ratio==float("inf") else fin(v1_ratio)),
        V1_pass=(n_conf>=4 and (v1_ratio==float("inf") or (not math.isnan(v1_ratio) and v1_ratio>=2.0))),
        V2_pass=(n_conf>=4 and not math.isnan(auc_cos) and auc_cos>=0.80))

# pooled
pc=[r for r in all_rows if r["cls"]=="C3" and r["target"]==1]
pk=[r for r in all_rows if r["cls"] in ("C1","C2") and r["correct"]]
pooled_auc=auc([r["e_cos90"] for r in pc],[r["e_cos90"] for r in pk])
evald=[m for m in MODELS if per_model.get(m,{}).get("n_confab",0)>=4]
V1=all(per_model[m]["V1_pass"] for m in evald) if evald else False
V2=all(per_model[m]["V2_pass"] for m in evald) if evald else False
V2=V2 and (not math.isnan(pooled_auc) and pooled_auc>=0.80)
out={"models_evaluated(>=4 confab)":evald, "per_model":per_model,
     "pooled_auc_cos90":fin(pooled_auc), "pooled_n_confab":len(pc), "pooled_n_correct":len(pk),
     "V1_inconsistency_universal":bool(V1), "V2_detector_generalizes":bool(V2),
     "PASS":bool(V1 and V2)}
pathlib.Path(__file__).parent.joinpath("probe_multimodel_results.json").write_text(
    json.dumps({"rows":all_rows, **out}, indent=2))
print("\n"+json.dumps(out, indent=2))
