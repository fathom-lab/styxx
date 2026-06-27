# GROUND-TRUTH validation: do the confound-audit verdicts replicate on REAL, HUMAN-labeled data?
# No model-generated text, no model-assigned labels. Real Yelp reviews; the human star rating IS the
# ground-truth sentiment (2-star = negative, 4-star = positive — genuinely mild/boundary cases).
# Orthogonality (label _|_ length) is achieved by coarsened-exact-matching SUBSAMPLING of real data,
# not by generation. Then audit the deployed sentiment fleet and compare to the synthetic verdicts.
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

OUT = r"C:/Users/heyzo/.styxx/confound_reportcard"
random.seed(0)
print("styxx", styxx.__version__, flush=True)

# ---- 1. pull REAL human-labeled reviews (Yelp stars = gold sentiment), boundary stars 2 vs 4 ----
from datasets import load_dataset
print("loading yelp_review_full (streaming)...", flush=True)
ds = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
NEG_STAR, POS_STAR = 1, 3   # label index: 0..4 == 1..5 stars -> 2-star=1 (neg), 4-star=3 (pos)
WMIN, WMAX, TARGET_POOL = 10, 120, 5000
pool = {0: [], 1: []}       # 0 = negative (2*), 1 = positive (4*)
for ex in ds:
    lab = ex["label"]
    if lab not in (NEG_STAR, POS_STAR):
        continue
    txt = ex["text"].replace("\\n", " ").replace("\n", " ").strip()
    wc = len(txt.split())
    if not (WMIN <= wc <= WMAX):
        continue
    y = 0 if lab == NEG_STAR else 1
    if len(pool[y]) < TARGET_POOL:
        pool[y].append((txt, wc))
    if len(pool[0]) >= TARGET_POOL and len(pool[1]) >= TARGET_POOL:
        break
print(f"pool: neg={len(pool[0])} pos={len(pool[1])}", flush=True)

# ---- 2. coarsened-exact length-match (label _|_ length by SELECTION on real data) ----
BIN = 8
def binned(items):
    b = {}
    for t, wc in items:
        b.setdefault(wc // BIN, []).append((t, wc))
    for k in b:
        b[k].sort(key=lambda x: x[1])  # deterministic
    return b
bneg, bpos = binned(pool[0]), binned(pool[1])
rows, PER_CLASS_CAP = [], 250
n0 = n1 = 0
for k in sorted(set(bneg) & set(bpos)):
    take = min(len(bneg[k]), len(bpos[k]))
    take = min(take, PER_CLASS_CAP - n0, PER_CLASS_CAP - n1)
    if take <= 0:
        continue
    for t, wc in bneg[k][:take]:
        rows.append({"text": t, "label": 0, "confound": math.log(wc)}); n0 += 1
    for t, wc in bpos[k][:take]:
        rows.append({"text": t, "label": 1, "confound": math.log(wc)}); n1 += 1
random.shuffle(rows)
lab = np.array([r["label"] for r in rows]); cf = np.array([r["confound"] for r in rows])
corr = float(np.corrcoef(lab, cf)[0, 1])
print(f"matched corpus n={len(rows)} (neg={n0} pos={n1}) | corr(label,log_len)={corr:+.3f}", flush=True)

# ---- 3. construct-recoverability ceiling on the REAL matched corpus ----
texts = [r["text"] for r in rows]; y = [r["label"] for r in rows]
oof = cross_val_predict(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
      LogisticRegression(max_iter=2000)), texts, y, cv=5, method="predict_proba")[:, 1]
refit = float(roc_auc_score(y, oof))
print(f"BoW construct-recoverable AUC (real, human labels) = {refit:.3f}", flush=True)

# ---- 4. audit the deployed sentiment fleet on the REAL human-labeled corpus ----
MODELS = [
    ("distilbert-base-uncased-finetuned-sst-2-english", False, "synthetic: THRESHOLD-BIASED (control)"),
    ("nlptown/bert-base-multilingual-uncased-sentiment", True, "synthetic: THRESHOLD-BIASED"),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", False, "synthetic: THRESHOLD-BIASED"),
    ("siebert/sentiment-roberta-large-english", False, "synthetic: THRESHOLD-BIASED"),
    ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", False, "synthetic: ROBUST"),
]
def score_of(d, stars):
    if stars:
        sl = [k for k in d if "star" in k and k.split()[0].isdigit()]
        if sl and len(sl) == len(d):
            w = sum(d[k] for k in sl)
            return (sum(int(k.split()[0]) * d[k] for k in sl) / max(w, 1e-9) - 1) / 4
    for k in ("positive", "pos"):
        if k in d:
            return d[k]
    return None

results = []
for mid, stars, synth in MODELS:
    print("==", mid, flush=True)
    try:
        clf = pipeline("text-classification", model=mid, top_k=None, truncation=True, max_length=256)
        scores = [score_of({p["label"].lower(): float(p["score"]) for p in clf(r["text"])[0]}, stars) for r in rows]
        if any(s is None for s in scores):
            raise RuntimeError("map None")
        rep = audit_confound(rows, scores=scores, instrument=mid, confound="log_words", construct_recoverable_auc=refit)
        rec = {"model": mid, "synthetic_verdict": synth, "real_verdict": rep.verdict.split()[0],
               "coef": rep.confound_score_coef, "ci": list(rep.confound_score_coef_ci95),
               "within": rep.within_stratum_auc, "gate_ok": rep.gate_ok, "ortho": rep.orthogonality_corr}
        print(f"   REAL -> {rec['real_verdict']:18s} coef {rec['coef']:+.3f} {rec['ci']} | {synth}", flush=True)
        results.append(rec)
    except Exception as e:
        print("   ERR", str(e).splitlines()[0], flush=True)
        results.append({"model": mid, "synthetic_verdict": synth, "error": str(e).splitlines()[0]})

json.dump({"styxx": styxx.__version__, "dataset": "yelp_review_full (2* vs 4*, human stars=gold)",
           "n": len(rows), "corr_label_length": corr, "bow_recoverable_auc": refit, "results": results},
          open(os.path.join(OUT, "groundtruth_results.json"), "w"), indent=2)
print("DONE", flush=True)
