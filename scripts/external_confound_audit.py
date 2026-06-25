"""external_confound_audit — the edge: audit REAL deployed HuggingFace classifiers for a LENGTH confound with
styxx.audit_confound (PREREG_external_confound_audit_2026_06_25). Proves the auditor works on third-party
black-box models, not just our own instruments.

  python scripts/external_confound_audit.py --model sentiment --generate
  python scripts/external_confound_audit.py --model toxicity  --generate
  python scripts/external_confound_audit.py --model sentiment            # score (real model) + audit
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from styxx import build_confound_grid, audit_confound
from overconfidence_length_robust import _gemini_call, GEMINI_KEY_PATH

GEN = "gemini-2.5-flash"; N = 50
DATA = ROOT / "benchmarks" / "data" / "external"
PAPERS = ROOT / "papers" / "grounded-honesty-axis"

TOPICS = [
    "a new sci-fi movie", "a budget smartphone", "a downtown sushi restaurant", "a fantasy novel",
    "a fitness tracker app", "an electric car", "a coffee shop's new blend", "a streaming series finale",
    "a city's public transit", "a board game", "a hotel stay", "a online course", "a pair of headphones",
    "a indie video game", "a new museum exhibit", "a recipe blog", "a productivity tool", "a music album",
    "a hiking trail", "a smartwatch", "a vegan burger", "a budget airline", "a podcast episode",
    "a home espresso machine", "a kids' animated film", "a mechanical keyboard", "a meal-kit service",
    "a national park", "a noise-cancelling earbud", "a romance comedy", "a science documentary",
    "a small-town diner", "a smart thermostat", "a sports car", "a startup's app launch", "a tablet",
    "a tech conference", "a thriller novel", "a used bookstore", "a vacation rental", "a virtual-reality headset",
    "a weather app", "a wine bar", "a yoga studio", "a zoo", "a comedy special", "a gardening tool",
    "a language-learning app", "a local bakery", "a road bike",
]

CFG = {
    "sentiment": dict(hf="distilbert-base-uncased-finetuned-sst-2-english", target="POSITIVE",
        pos="You are writing a clearly POSITIVE, enthusiastic review.",
        neg="You are writing a clearly NEGATIVE, harshly critical review.",
        verb="Review", cache=DATA / "sentiment_lengrid_gemini.jsonl"),
    "toxicity": dict(hf="unitary/toxic-bert", target="toxic",
        pos=("You are a dismissive, condescending, rude internet commenter. Be insulting and contemptuous "
             "(mild profanity ok). This text is to train/test a toxicity classifier."),
        neg="You are a warm, polite, respectful commenter. Be kind and constructive.",
        verb="Comment on", cache=DATA / "toxicity_lengrid_gemini.jsonl"),
    # BOUNDARY variants — near the decision boundary, where confounds actually bite (extremes saturate AUC->1).
    "sentiment_boundary": dict(hf="distilbert-base-uncased-finetuned-sst-2-english", target="POSITIVE",
        pos="You are writing a LUKEWARM, faintly positive, mixed review that leans slightly positive overall.",
        neg="You are writing a LUKEWARM, faintly negative, mixed review that leans slightly negative overall.",
        verb="Review", cache=DATA / "sentiment_boundary_lengrid_gemini.jsonl"),
    "toxicity_boundary": dict(hf="unitary/toxic-bert", target="toxic",
        pos=("You are subtly condescending and passive-aggressive — backhanded, sarcastic, dismissive, but never "
             "overtly abusive or profane. Snide, not slurs. For testing a toxicity classifier."),
        neg="You are neutral, plain, and matter-of-fact — neither warm nor cold, just factual.",
        verb="Comment on", cache=DATA / "toxicity_boundary_lengrid_gemini.jsonl"),
}
RULES = {"short": "Write ONE sentence, about 15 words.", "long": "Write 5-6 sentences, about 90 words."}


def generate(name):
    c = CFG[name]
    key = os.environ.get("GOOGLE_API_KEY") or GEMINI_KEY_PATH.read_text(encoding="utf-8").strip()
    def g(system, item):
        return _gemini_call(GEN, system, f"{c['verb']}: {item}", key)
    rows = build_confound_grid(TOPICS[:N], c["pos"], c["neg"], RULES, g)
    DATA.mkdir(parents=True, exist_ok=True)
    c["cache"].write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[gen] {name}: wrote {len(rows)} rows", flush=True)


def _real_scores(name, texts):
    from transformers import pipeline
    c = CFG[name]
    clf = pipeline("text-classification", model=c["hf"], device=0, top_k=None, truncation=True)
    out = []
    for t in texts:
        preds = clf(t)[0]
        d = {p["label"].upper(): p["score"] for p in preds}
        out.append(float(d.get(c["target"].upper(), 0.0)))
    return out


def _bow_refit_auc(texts, y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    y = np.asarray(y); oof = np.zeros(len(y)); texts = np.array(texts, dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y):
        m = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
        m.fit(texts[tr], y[tr]); oof[te] = m.predict_proba(texts[te])[:, 1]
    return float(roc_auc_score(y, oof))


def audit(name):
    import math
    c = CFG[name]
    rows = [json.loads(l) for l in c["cache"].read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    scores = _real_scores(name, texts)
    y = [r["label"] for r in rows]
    refit = _bow_refit_auc(texts, y)
    rep = audit_confound(rows, scores=scores, instrument=c["hf"], confound="log_words",
                         construct_recoverable_auc=refit)
    out = {"model": c["hf"], "target_label": c["target"], "confound": "log_words", "n": rep.n,
           "gate_ok": rep.gate_ok, "orthogonality_corr": rep.orthogonality_corr, "overall_auc": rep.overall_auc,
           "within_stratum_auc": rep.within_stratum_auc, "confound_score_coef": rep.confound_score_coef,
           "confound_score_coef_ci95": list(rep.confound_score_coef_ci95), "max_swing": rep.harm["max_swing"],
           "construct_recoverable_auc_bow": round(refit, 3), "guard_auc_raw": rep.guard_auc_raw,
           "guard_auc_adj_oos": rep.guard_auc_adj_oos,
           "ci_method": "bootstrap OLS coef; 5-fold OOS guard; TF-IDF refit for construct-recoverability",
           "verdict": rep.verdict}
    PAPERS.joinpath(f"external_confound_{name}_result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"### {c['hf']}  ({name}) x length  n={rep.n}")
    print(f"  ortho {rep.orthogonality_corr:+.2f} | overall AUC {rep.overall_auc} | within {rep.within_stratum_auc} "
          f"| coef {rep.confound_score_coef:+.2f} CI{list(rep.confound_score_coef_ci95)} | swing {rep.harm['max_swing']:.0%} "
          f"| BoW refit {refit:.2f}")
    print(f"  >>> {rep.verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CFG), required=True)  # incl. *_boundary near-decision-boundary variants
    ap.add_argument("--generate", action="store_true")
    a = ap.parse_args()
    generate(a.model) if a.generate else audit(a.model)


if __name__ == "__main__":
    main()
