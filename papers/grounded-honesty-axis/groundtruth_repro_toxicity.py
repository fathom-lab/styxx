# GROUND-TRUTH validation, TOXICITY: do the report card's toxicity verdicts (2 models CONFOUND-DEPENDENT
# / "broken") replicate on REAL human-labeled toxicity? Civil Comments: 'toxicity' = fraction of human
# annotators who marked the comment toxic (gold human labels). toxic >= 0.5 vs non-toxic <= 0.2, real text,
# length-matched by CEM. If the synthetic CONFOUND-DEPENDENT collapse does NOT reappear, that verdict was a
# synthetic-substrate artifact too.
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
print("streaming google/civil_comments ...", flush=True)
ds = load_dataset("google/civil_comments", split="train", streaming=True)
WMIN, WMAX, CAP = 10, 120, 3000
pool = {0: [], 1: []}   # 0 = non-toxic (<=0.2), 1 = toxic (>=0.5)  [human annotator fraction]
seen = 0
for ex in ds:
    seen += 1
    tox = ex["toxicity"]
    if tox is None:
        continue
    if tox >= 0.5:
        y = 1
    elif tox <= 0.2:
        y = 0
    else:
        continue
    txt = (ex["text"] or "").replace("\n", " ").strip()
    wc = len(txt.split())
    if not (WMIN <= wc <= WMAX):
        continue
    if len(pool[y]) < CAP:
        pool[y].append((txt, wc))
    if len(pool[0]) >= CAP and len(pool[1]) >= CAP:
        break
    if seen > 400000:
        break
print(f"scanned {seen} | pool: nontoxic={len(pool[0])} toxic={len(pool[1])}", flush=True)

BIN = 8
def binned(items):
    b = {}
    for t, wc in items:
        b.setdefault(wc // BIN, []).append((t, wc))
    for k in b: b[k].sort(key=lambda x: x[1])
    return b
b0, b1 = binned(pool[0]), binned(pool[1])
rows, PER = [], 250
n0 = n1 = 0
for k in sorted(set(b0) & set(b1)):
    take = min(len(b0[k]), len(b1[k]), PER - n0, PER - n1)
    if take <= 0: continue
    for t, wc in b0[k][:take]: rows.append({"text": t, "label": 0, "confound": math.log(wc)}); n0 += 1
    for t, wc in b1[k][:take]: rows.append({"text": t, "label": 1, "confound": math.log(wc)}); n1 += 1
random.shuffle(rows)
lab = np.array([r["label"] for r in rows]); cf = np.array([r["confound"] for r in rows])
corr = float(np.corrcoef(lab, cf)[0, 1])
print(f"matched n={len(rows)} (nontoxic={n0} toxic={n1}) | corr(label,log_len)={corr:+.3f}", flush=True)

texts = [r["text"] for r in rows]; y = [r["label"] for r in rows]
oof = cross_val_predict(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
      LogisticRegression(max_iter=2000)), texts, y, cv=5, method="predict_proba")[:, 1]
refit = float(roc_auc_score(y, oof))
print(f"BoW construct-recoverable AUC (real human toxicity) = {refit:.3f}", flush=True)

MODELS = [
    ("unitary/toxic-bert", "ROBUST (control)"),
    ("s-nlp/roberta_toxicity_classifier", "CONFOUND-DEPENDENT"),
    ("martin-ha/toxic-comment-model", "THRESHOLD-BIASED (negligible)"),
    ("unitary/unbiased-toxic-roberta", "CONFOUND-DEPENDENT"),
]
def tox_score(d):
    for k in ("toxic", "toxicity"):
        if k in d:
            return d[k]
    return None

results = []
for mid, synth in MODELS:
    print("==", mid, flush=True)
    try:
        clf = pipeline("text-classification", model=mid, top_k=None, truncation=True, max_length=256)
        scores = [tox_score({p["label"].lower(): float(p["score"]) for p in clf(r["text"])[0]}) for r in rows]
        if any(s is None for s in scores):
            raise RuntimeError("toxic label not found")
        rep = audit_confound(rows, scores=scores, instrument=mid, confound="log_words", construct_recoverable_auc=refit)
        rec = {"model": mid, "synthetic_verdict": synth, "real_verdict": rep.verdict.split()[0],
               "coef": rep.confound_score_coef, "ci": list(rep.confound_score_coef_ci95),
               "within": rep.within_stratum_auc, "overall_auc": getattr(rep, "overall_auc", None), "gate_ok": rep.gate_ok}
        wa = rep.within_stratum_auc or {}
        print(f"   REAL -> {rec['real_verdict']:18s} coef {rec['coef']:+.3f} | within {wa} | overall {rec['overall_auc']} (synth {synth})", flush=True)
        results.append(rec)
    except Exception as e:
        print("   ERR", str(e).splitlines()[0], flush=True)
        results.append({"model": mid, "synthetic_verdict": synth, "error": str(e).splitlines()[0]})

json.dump({"styxx": styxx.__version__, "dataset": "google/civil_comments (toxic>=0.5 vs non-toxic<=0.2, human annotator fraction)",
           "n": len(rows), "corr_label_length": corr, "bow_recoverable_auc": refit, "results": results},
          open(os.path.join(OUT, "groundtruth_toxicity_results.json"), "w"), indent=2)
print("DONE", flush=True)
