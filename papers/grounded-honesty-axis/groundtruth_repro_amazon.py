# GROUND-TRUTH confirmation #2 — different domain (Amazon product reviews, human star labels).
# Same design as the Yelp test: 2-star vs 4-star (boundary), length-matched by CEM, audit the fleet.
import json, math, os, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from transformers import pipeline
import styxx
from styxx import audit_confound
from datasets import load_dataset

OUT = r"C:/Users/heyzo/.styxx/confound_reportcard"
random.seed(0)
print("styxx", styxx.__version__, flush=True)
print("loading SetFit/amazon_reviews_multi_en (full train)...", flush=True)
ds = load_dataset("SetFit/amazon_reviews_multi_en", split="train")  # label 0..4 == 1..5 stars
labs = set(int(x) for x in ds["label"][:5000])
print("label values seen:", sorted(labs), flush=True)
NEG_STAR, POS_STAR = 1, 3   # 2-star (label 1, neg) vs 4-star (label 3, pos)
WMIN, WMAX, CAP = 10, 120, 6000
pool = {0: [], 1: []}
for ex in ds:
    lab = int(ex["label"])
    if lab not in (NEG_STAR, POS_STAR):
        continue
    txt = (ex["text"] or "").replace("\\n", " ").replace("\n", " ").strip()
    wc = len(txt.split())
    if not (WMIN <= wc <= WMAX):
        continue
    y = 0 if lab == NEG_STAR else 1
    if len(pool[y]) < CAP:
        pool[y].append((txt, wc))
    if len(pool[0]) >= CAP and len(pool[1]) >= CAP:
        break
print(f"pool: neg(2*)={len(pool[0])} pos(4*)={len(pool[1])}", flush=True)

BIN = 8
def binned(items):
    b = {}
    for t, wc in items:
        b.setdefault(wc // BIN, []).append((t, wc))
    for k in b: b[k].sort(key=lambda x: x[1])
    return b
bneg, bpos = binned(pool[0]), binned(pool[1])
rows, PER = [], 250
n0 = n1 = 0
for k in sorted(set(bneg) & set(bpos)):
    take = min(len(bneg[k]), len(bpos[k]), PER - n0, PER - n1)
    if take <= 0: continue
    for t, wc in bneg[k][:take]: rows.append({"text": t, "label": 0, "confound": math.log(wc)}); n0 += 1
    for t, wc in bpos[k][:take]: rows.append({"text": t, "label": 1, "confound": math.log(wc)}); n1 += 1
random.shuffle(rows)
lab = np.array([r["label"] for r in rows]); cf = np.array([r["confound"] for r in rows])
corr = float(np.corrcoef(lab, cf)[0, 1])
print(f"matched n={len(rows)} (neg={n0} pos={n1}) | corr(label,log_len)={corr:+.3f}", flush=True)

texts = [r["text"] for r in rows]; y = [r["label"] for r in rows]
oof = cross_val_predict(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
      LogisticRegression(max_iter=2000)), texts, y, cv=5, method="predict_proba")[:, 1]
refit = float(roc_auc_score(y, oof))
print(f"BoW recoverable AUC = {refit:.3f}", flush=True)

MODELS = [
    ("distilbert-base-uncased-finetuned-sst-2-english", False, "THRESHOLD-BIASED"),
    ("nlptown/bert-base-multilingual-uncased-sentiment", True, "THRESHOLD-BIASED"),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", False, "THRESHOLD-BIASED"),
    ("siebert/sentiment-roberta-large-english", False, "THRESHOLD-BIASED"),
    ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", False, "ROBUST"),
]
def score_of(d, stars):
    if stars:
        sl = [k for k in d if "star" in k and k.split()[0].isdigit()]
        if sl and len(sl) == len(d):
            w = sum(d[k] for k in sl); return (sum(int(k.split()[0]) * d[k] for k in sl) / max(w, 1e-9) - 1) / 4
    for k in ("positive", "pos"):
        if k in d: return d[k]
    return None

results = []
for mid, stars, synth in MODELS:
    print("==", mid, flush=True)
    try:
        clf = pipeline("text-classification", model=mid, top_k=None, truncation=True, max_length=256)
        scores = [score_of({p["label"].lower(): float(p["score"]) for p in clf(r["text"])[0]}, stars) for r in rows]
        rep = audit_confound(rows, scores=scores, instrument=mid, confound="log_words", construct_recoverable_auc=refit)
        rec = {"model": mid, "synthetic_verdict": synth, "real_verdict": rep.verdict.split()[0],
               "coef": rep.confound_score_coef, "ci": list(rep.confound_score_coef_ci95),
               "within": rep.within_stratum_auc, "gate_ok": rep.gate_ok}
        print(f"   AMAZON -> {rec['real_verdict']:18s} coef {rec['coef']:+.3f} {rec['ci']} (synth {synth})", flush=True)
        results.append(rec)
    except Exception as e:
        print("   ERR", str(e).splitlines()[0], flush=True)
        results.append({"model": mid, "synthetic_verdict": synth, "error": str(e).splitlines()[0]})

json.dump({"styxx": styxx.__version__, "dataset": "SetFit/amazon_reviews_multi_en (2* vs 4*, human stars=gold)",
           "n": len(rows), "corr_label_length": corr, "bow_recoverable_auc": refit, "results": results},
          open(os.path.join(OUT, "groundtruth_amazon_results.json"), "w"), indent=2)
print("DONE", flush=True)
