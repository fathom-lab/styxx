# Reproduce the HuggingFace Confound Report Card with the SHIPPED tool.
#
#   pip install 'styxx[hf]'
#   python papers/grounded-honesty-axis/hf_report_card_repro.py
#
# Each model is graded by styxx.audit_hf_model on the bundled, validated length-orthogonal
# boundary corpus (n=200) — the same corpora used in the published finding. Verdicts are
# CI-backed; distilbert-sst2 and unitary/toxic-bert are the replication controls (prior verdicts
# THRESHOLD-BIASED and ROBUST respectively).
import json

from styxx import audit_hf_model

MODELS = [
    ("distilbert-base-uncased-finetuned-sst-2-english", "sentiment"),        # control
    ("nlptown/bert-base-multilingual-uncased-sentiment", "sentiment"),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment"),
    ("siebert/sentiment-roberta-large-english", "sentiment"),
    ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", "sentiment"),
    ("unitary/toxic-bert", "toxicity"),                                       # control
    ("s-nlp/roberta_toxicity_classifier", "toxicity"),
    ("martin-ha/toxic-comment-model", "toxicity"),
    ("unitary/unbiased-toxic-roberta", "toxicity"),
    # finiteautomata/bertweet-base-sentiment-analysis — excluded: tokenizer indexing error, not a verdict.
]


def main():
    out = []
    for model_id, construct in MODELS:
        try:
            r = audit_hf_model(model_id, construct=construct)
            rec = {
                "model": model_id, "construct": construct,
                "verdict": r.verdict.split()[0],
                "confound_score_coef": round(r.confound_score_coef, 3),
                "confound_score_coef_ci95": [round(x, 3) for x in r.confound_score_coef_ci95],
                "within_stratum_auc": r.within_stratum_auc,
                "construct_recoverable_auc": r.construct_recoverable_auc,
            }
            print(f"{model_id:55s} -> {rec['verdict']:18s} coef {rec['confound_score_coef']:+.3f}")
        except Exception as e:  # honestly record (don't drop) anything that fails to load/score
            rec = {"model": model_id, "construct": construct, "error": str(e).splitlines()[0]}
            print(f"{model_id:55s} -> ERROR  {rec['error']}")
        out.append(rec)
    with open("hf_report_card_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote hf_report_card_result.json ({sum('verdict' in r for r in out)} graded / {len(MODELS)})")


if __name__ == "__main__":
    main()
