# Reproduce the multi-confound battery (length / exclamation / ALL-CAPS) across the HF fleet.
#
#   pip install 'styxx[hf]'
#   python papers/grounded-honesty-axis/confound_battery_repro.py
#
# Each new confound is injected by controlled transformation of the validated boundary corpus, with the
# level assigned BALANCED WITHIN each label class (stratified by length) so it is orthogonal to the
# construct by construction. styxx.audit_confound's gate_ok independently verifies orthogonality.
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from transformers import pipeline

from styxx import audit_confound

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "benchmarks", "data", "external"))


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def bow_refit(rows):
    texts = [r["text"] for r in rows]
    y = [int(r["label"]) for r in rows]
    oof = cross_val_predict(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
          LogisticRegression(max_iter=2000)), texts, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, oof))


def _assign(base):
    for label in (0, 1):
        items = sorted([r for r in base if r["label"] == label], key=lambda r: len(r["text"].split()))
        for i, r in enumerate(items):
            yield label, r["text"], (i % 2 == 1)


def inject_exclamation(base):
    out = []
    for label, txt, hi in _assign(base):
        t = txt.replace("!", "")
        if hi:
            t = t.rstrip() + " !!!"
        out.append({"text": t, "label": label, "confound": float(t.count("!"))})
    return out


def inject_caps(base):
    out = []
    for label, txt, hi in _assign(base):
        t = txt.upper() if hi else txt
        alpha = sum(c.isalpha() for c in t)
        up = sum(c.isupper() for c in t if c.isalpha())
        out.append({"text": t, "label": label, "confound": up / max(1, alpha)})
    return out


CONFOUNDS = {"exclamation": inject_exclamation, "caps": inject_caps}
CORPORA = {"sentiment": "sentiment_boundary_lengrid_gemini.jsonl",
           "toxicity": "toxicity_boundary_lengrid_gemini.jsonl"}
MODELS = [
    ("distilbert-base-uncased-finetuned-sst-2-english", "sentiment", False),
    ("nlptown/bert-base-multilingual-uncased-sentiment", "sentiment", True),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment", False),
    ("siebert/sentiment-roberta-large-english", "sentiment", False),
    ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", "sentiment", False),
    ("unitary/toxic-bert", "toxicity", False),
    ("s-nlp/roberta_toxicity_classifier", "toxicity", False),
    ("martin-ha/toxic-comment-model", "toxicity", False),
    ("unitary/unbiased-toxic-roberta", "toxicity", False),
]
SENT, TOX = ("positive", "pos"), ("toxic", "toxicity")


def score_of(d, cons, stars):
    if stars:
        sl = [k for k in d if "star" in k and k.split()[0].isdigit()]
        if sl and len(sl) == len(d):
            w = sum(d[k] for k in sl)
            return (sum(int(k.split()[0]) * d[k] for k in sl) / max(w, 1e-9) - 1) / 4
    for k in (SENT if cons == "sentiment" else TOX):
        if k in d:
            return d[k]
    return None


def main():
    corpora = {c: load(f) for c, f in CORPORA.items()}
    inj, refit = {}, {}
    for cons, base in corpora.items():
        for cname, fn in CONFOUNDS.items():
            rows = fn(base)
            inj[(cons, cname)] = rows
            refit[(cons, cname)] = bow_refit(rows)
    results = []
    for mid, cons, stars in MODELS:
        try:
            clf = pipeline("text-classification", model=mid, top_k=None, truncation=True, max_length=256)
            for cname in CONFOUNDS:
                rows = inj[(cons, cname)]
                scores = [score_of({p["label"].lower(): float(p["score"]) for p in clf(r["text"])[0]}, cons, stars)
                          for r in rows]
                rep = audit_confound(rows, scores=scores, instrument=mid, confound=cname,
                                     construct_recoverable_auc=refit[(cons, cname)])
                results.append({"model": mid, "construct": cons, "confound": cname,
                                "verdict": rep.verdict.split()[0], "coef": rep.confound_score_coef,
                                "ci": list(rep.confound_score_coef_ci95), "within": rep.within_stratum_auc,
                                "gate_ok": rep.gate_ok})
                print(f"{mid.split('/')[-1]:42s} {cname:12s} -> {rep.verdict.split()[0]}")
        except Exception as e:
            print(f"{mid:50s} ERROR {str(e).splitlines()[0]}")
    with open(os.path.join(HERE, "confound_battery_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote confound_battery_result.json ({len(results)} audits)")


if __name__ == "__main__":
    main()
