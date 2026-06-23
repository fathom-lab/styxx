"""Length-decorrelated refit of sycophancy v0 (offline, from the cached n=1200 responses).

Root cause: the yield (sycophantic) system prompt instructs "elaborate", so sycophantic training
responses are systematically longer → the logistic fit learned log_word_count +0.356 as a spurious
feature → false-positives on long honest text. This refits WITHOUT log_word_count and checks the cost
to discrimination (5-fold CV AUC) + the behavioral length-confound + the announcement FP.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from styxx.guardrail.sycophancy_signals import extract_sycophancy_features

rows = [json.loads(l) for l in (ROOT/"benchmarks/data/sycophancy/responses_v0.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
FULL = ["agreement_lexicon_density","premise_echo_rate","counter_lexicon_density","capitulation_density",
        "starts_with_agreement","opinion_marker_density","superlative_density","hedge_density","log_word_count"]
NOLEN = [f for f in FULL if f != "log_word_count"]

feats, y, words = [], [], {True: [], False: []}
for r in rows:
    feats.append(extract_sycophancy_features(r.get("question",""), r["response"]))
    lab = bool(r["label_sycophantic"]); y.append(1 if lab else 0)
    words[lab].append(len(r["response"].split()))
y = np.array(y)
print(f"n={len(rows)}  sycophantic mean_words={st.mean(words[True]):.1f}  non-syc mean_words={st.mean(words[False]):.1f}  "
      f"(ratio {st.mean(words[True])/st.mean(words[False]):.2f}x) — the data-level length bias")

def cv_auc(features):
    X = np.array([[f[k] for k in features] for f in feats])
    s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    return s.mean(), s.std()

a_full, s_full = cv_auc(FULL)
a_nolen, s_nolen = cv_auc(NOLEN)
print(f"5-fold AUC  full (9 feat, w/ length):   {a_full:.4f} +- {s_full:.4f}")
print(f"5-fold AUC  no-length (8 feat):         {a_nolen:.4f} +- {s_nolen:.4f}")
print(f"AUC cost of dropping length:            {a_full - a_nolen:+.4f}")

# refit on all data without length -> paste-ready weights, and behavioral check
sc = StandardScaler().fit(np.array([[f[k] for k in NOLEN] for f in feats]))
clf = LogisticRegression(max_iter=2000).fit(sc.transform(np.array([[f[k] for k in NOLEN] for f in feats])), y)
def score(text):
    f = extract_sycophancy_features("", text)
    x = sc.transform([[f[k] for k in NOLEN]])
    return float(clf.predict_proba(x)[0, 1])
unit = "The module hashes the message and re-runs a deterministic audit to reproduce the verdict. "
print("\nbehavioral check with the REFIT (no-length) model:")
for n in (1, 4, 8, 16):
    t = unit * n
    print(f"   sober words={len(t.split()):4d}  syc={score(t):.3f}")
HYPE = "You are absolutely right, what a brilliant question — I completely agree with your amazing instinct here."
ann = (ROOT/"examples/parrhesia/announcement.txt").read_text(encoding="utf-8")
print(f"   HYPE (flattery): {score(HYPE):.3f}   announcement (sober 187w): {score(ann):.3f}")
