"""Builds examples/audit_confound_colab.ipynb (run to regenerate). Kept as a builder so the notebook JSON is
never hand-edited. The notebook is self-contained and runs WITHOUT any API key."""
import json
from pathlib import Path

RAW = "https://raw.githubusercontent.com/fathom-lab/styxx/main/benchmarks/data/external/sentiment_boundary_lengrid_gemini.jsonl"


def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
    md("# `styxx.audit_confound` — is your classifier's score tracking the concept, or a confound?",
       "",
       "Every deployed text scorer — a toxicity filter, a sentiment model, an LLM guardrail — can separate its",
       "training data while secretly keying on a **confound** (length, politeness, identity terms…) that happened",
       "to correlate with the label. When the confound and the concept come apart in production, it fails silently.",
       "",
       "`audit_confound` finds out, on a corpus where the concept and the suspected confound are **orthogonal**,",
       "and returns one of four CI-backed verdicts — `ROBUST` / `THRESHOLD-BIASED` (+ a validated `report.guard()`) /",
       "`CONFOUND-DEPENDENT` (broken) / `INCONCLUSIVE`. **Runtime → Run all.** No API key needed."),
    code(["%pip install -q git+https://github.com/fathom-lab/styxx.git\n",
          "%pip install -q transformers torch scikit-learn\n"]),

    md("## 1 · Catch a planted confound (instant, no downloads)",
       "A classifier whose score = `3·label − 2·confound`. It *can* tell the classes apart, but its score rides the",
       "confound. The auditor should flag it `THRESHOLD-BIASED` and hand back a fix."),
    code(["import numpy as np\n",
          "from styxx import audit_confound\n",
          "\n",
          "rng = np.random.default_rng(0); n = 200\n",
          "y = np.tile([0, 1], n // 2)                 # the concept label\n",
          "confound = rng.standard_normal(n)           # a cue, orthogonal to the label\n",
          "score = 3.0 * y - 2.0 * confound + rng.normal(0, 0.3, n)   # a scorer that rides the confound\n",
          "rows = [{'text': f'item {i}', 'label': int(y[i]), 'confound': float(confound[i])} for i in range(n)]\n",
          "\n",
          "rep = audit_confound(rows, scores=list(score), instrument='my_classifier', confound='my_cue')\n",
          "print(rep.verdict)\n",
          "print('\\nscore at reference cue:', round(rep.guard(1.0, rep.guard_ref), 3),\n",
          "      '| same score, extreme cue, corrected:', round(rep.guard(1.0, rep.guard_ref + 3), 3))\n"]),

    md("## 2 · Audit a REAL deployed model — and find its hidden length bias",
       "`distilbert-base-uncased-finetuned-sst-2-english` is the **default HuggingFace sentiment model** (hundreds",
       "of millions of downloads). On clear-cut reviews it's perfectly length-robust. We load a corpus of *lukewarm*",
       "reviews (near the decision boundary), score each with the real model, and audit. The bias only shows at the",
       "boundary — **clear-cut content saturates AUC→1.0 and hides confounds. Probe the boundary, not the extremes.**"),
    code(["import json, urllib.request, numpy as np\n",
          "from transformers import pipeline\n",
          "from sklearn.feature_extraction.text import TfidfVectorizer\n",
          "from sklearn.linear_model import LogisticRegression\n",
          "from sklearn.pipeline import make_pipeline\n",
          "from sklearn.model_selection import cross_val_predict\n",
          "from sklearn.metrics import roc_auc_score\n",
          "from styxx import audit_confound\n",
          "\n",
          "# a boundary corpus (lukewarm reviews x short/long), construct orthogonal to length\n",
          f"url = '{RAW}'\n",
          "rows = [json.loads(l) for l in urllib.request.urlopen(url).read().decode().splitlines() if l.strip()]\n",
          "\n",
          "clf = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english',\n",
          "               top_k=None, truncation=True)\n",
          "def p_positive(t):\n",
          "    d = {p['label'].upper(): p['score'] for p in clf(t)[0]}\n",
          "    return d['POSITIVE']\n",
          "scores = [p_positive(r['text']) for r in rows]\n",
          "\n",
          "# construct-recoverability (model-agnostic): can a plain bag-of-words refit recover the concept?\n",
          "texts = [r['text'] for r in rows]; y = [r['label'] for r in rows]\n",
          "oof = cross_val_predict(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),\n",
          "                                      LogisticRegression(max_iter=2000)),\n",
          "                        texts, y, cv=5, method='predict_proba')[:, 1]\n",
          "refit = roc_auc_score(y, oof)\n",
          "\n",
          "rep = audit_confound(rows, scores=scores, instrument='distilbert-sst2', confound='log_words',\n",
          "                     construct_recoverable_auc=refit)\n",
          "print(rep.verdict)\n",
          "print('\\nwithin-stratum AUC:', rep.within_stratum_auc, '| confound coef:', rep.confound_score_coef,\n",
          "      rep.confound_score_coef_ci95)\n"]),
    md("> You should see **THRESHOLD-BIASED**: the model separates sentiment within each length band, but longer",
       "> mildly-negative reviews are scored more positive — a length bias invisible on clear-cut text."),

    md("## 3 · Audit YOUR classifier",
       "Two ways to bring data. Either is fine.",
       "",
       "**A — you already have labeled text + a cue to test:** make `rows` of `{text, label(0/1), confound(number)}`",
       "and pass your model as `score_fn` (or precomputed `scores`).",
       "",
       "**B — generate an orthogonal grid with any LLM:** `build_confound_grid` crosses two stance prompts with",
       "confound levels so the concept and the cue are decorrelated by construction. Bring your own `generate_fn`."),
    code(["from styxx import audit_confound, build_confound_grid\n",
          "\n",
          "# --- A: your own rows ---\n",
          "# def my_model(text): return float(...)   # your classifier's score for `text`\n",
          "# rows = [{'text': t, 'label': lbl, 'confound': len(t.split())} for t, lbl in my_data]\n",
          "# report = audit_confound(rows, score_fn=my_model, confound='length')\n",
          "# print(report.verdict)\n",
          "# if report.verdict.startswith('THRESHOLD'):\n",
          "#     fair = report.guard(raw_score, confound_value)   # confound-fair score\n",
          "\n",
          "# --- B: generate the grid with your LLM (bring a generate_fn(system_prompt, user_item) -> text) ---\n",
          "# rows = build_confound_grid(\n",
          "#     items=['topic 1', 'topic 2', ...],\n",
          "#     pos_prompt='You write clearly POSITIVE reviews.',\n",
          "#     neg_prompt='You write clearly NEGATIVE reviews.',\n",
          "#     confound_rules={'short': 'One sentence.', 'long': 'Five sentences.'},\n",
          "#     generate_fn=my_llm)\n",
          "# report = audit_confound(rows, score_fn=my_model, confound='log_words')\n",
          "print('Fill in A or B above with your classifier.')\n"]),

    md("---",
       "Tip: audit at the **decision boundary** (ambiguous examples), not the extremes — that's where confounds bite.",
       "",
       "More: the technical note `papers/grounded-honesty-axis/NOTE_confound_audit_2026_06_25.md` ·",
       "repo [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx)."),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "audit_confound_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
