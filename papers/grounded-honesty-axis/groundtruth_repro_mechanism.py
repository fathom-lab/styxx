# MECHANISM PROBE: is the synthetic "length bias" in the generated WORDS, not the classifiers?
# Replace every ML model with VADER — a rule-based sentiment LEXICON (no training, just a word list).
# If VADER's score ALSO rides length on the synthetic corpus but NOT on real human-labeled data, the
# bias was lexical content the generator injected, present even to a dumb word-counter — not any model.
import json, math, os, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import styxx
from styxx import audit_confound

REPO = r"C:/Users/heyzo/clawd/styxx"; DATA = os.path.join(REPO, "benchmarks/data/external")
random.seed(0)
vader = SentimentIntensityAnalyzer()
def vscore(t): return (vader.polarity_scores(t)["compound"] + 1) / 2  # -> [0,1]

def bow_refit(rows):
    texts=[r["text"] for r in rows]; y=[int(r["label"]) for r in rows]
    oof=cross_val_predict(make_pipeline(TfidfVectorizer(min_df=2,ngram_range=(1,2)),
        LogisticRegression(max_iter=2000)),texts,y,cv=5,method="predict_proba")[:,1]
    return float(roc_auc_score(y,oof))

def audit(rows, name):
    scores=[vscore(r["text"]) for r in rows]
    rep=audit_confound(rows, scores=scores, instrument="VADER-lexicon", confound="log_words",
                       construct_recoverable_auc=bow_refit(rows))
    # also: within-label, does longer text read more positive lexically?
    lab=np.array([r["label"] for r in rows]); cf=np.array([r["confound"] for r in rows]); sc=np.array(scores)
    within={}
    for L in (0,1):
        m=lab==L
        if m.sum()>5:
            within[L]=float(np.corrcoef(cf[m], sc[m])[0,1])  # corr(length, VADER) within this label
    print(f"  {name:10s} VADER length->score: {rep.verdict.split()[0]:18s} coef {rep.confound_score_coef:+.3f} {rep.confound_score_coef_ci95} | corr(len,VADER) within neg={within.get(0):+.2f} pos={within.get(1):+.2f}", flush=True)
    return {"corpus":name,"verdict":rep.verdict.split()[0],"coef":rep.confound_score_coef,
            "ci":list(rep.confound_score_coef_ci95),"within_label_corr":within}

# ---- synthetic ----
syn=[json.loads(l) for l in open(os.path.join(DATA,"sentiment_boundary_lengrid_gemini.jsonl"),encoding="utf-8") if l.strip()]
print("MECHANISM — does a dumb lexicon (VADER) show the same length bias?\n", flush=True)
results=[audit(syn, "synthetic")]

# ---- real: rebuild Yelp + Amazon matched (same CEM as the ground-truth runs) ----
from datasets import load_dataset
def cem(pool0, pool1):
    BIN=8
    def b(items):
        d={}
        for t,wc in items: d.setdefault(wc//BIN,[]).append((t,wc))
        for k in d: d[k].sort(key=lambda x:x[1])
        return d
    b0,b1=b(pool0),b(pool1); rows=[]; n0=n1=0
    for k in sorted(set(b0)&set(b1)):
        take=min(len(b0[k]),len(b1[k]),250-n0,250-n1)
        if take<=0: continue
        for t,wc in b0[k][:take]: rows.append({"text":t,"label":0,"confound":math.log(wc)}); n0+=1
        for t,wc in b1[k][:take]: rows.append({"text":t,"label":1,"confound":math.log(wc)}); n1+=1
    random.shuffle(rows); return rows

def collect_stream(name, neg, pos, cfg=None):
    a=(name,)+((cfg,) if cfg else ())
    ds=load_dataset(*a, split="train", streaming=True)
    pool={0:[],1:[]}
    for ex in ds:
        lab=ex["label"]
        if lab not in (neg,pos): continue
        t=(ex["text"] or "").replace("\\n"," ").replace("\n"," ").strip(); wc=len(t.split())
        if not (10<=wc<=120): continue
        y=0 if lab==neg else 1
        if len(pool[y])<5000: pool[y].append((t,wc))
        if len(pool[0])>=5000 and len(pool[1])>=5000: break
    return cem(pool[0],pool[1])

def collect_full(name, neg, pos):
    ds=load_dataset(name, split="train")
    pool={0:[],1:[]}
    for ex in ds:
        lab=int(ex["label"])
        if lab not in (neg,pos): continue
        t=(ex["text"] or "").replace("\\n"," ").replace("\n"," ").strip(); wc=len(t.split())
        if not (10<=wc<=120): continue
        y=0 if lab==neg else 1
        if len(pool[y])<5000: pool[y].append((t,wc))
        if len(pool[0])>=5000 and len(pool[1])>=5000: break
    return cem(pool[0],pool[1])

results.append(audit(collect_stream("Yelp/yelp_review_full",1,3), "yelp"))
results.append(audit(collect_full("SetFit/amazon_reviews_multi_en",1,3), "amazon"))
json.dump(results, open(r"C:/Users/heyzo/.styxx/confound_reportcard/mechanism_results.json","w"), indent=2)
print("\nDONE", flush=True)
