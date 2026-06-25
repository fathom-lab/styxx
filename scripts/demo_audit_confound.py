"""demo_audit_confound — canonical use of the styxx.audit_confound PUBLIC API on a real shipped instrument.

The SAME auditor that flagged overconfidence_v0 as THRESHOLD-BIASED (a 4%->46% length swing) is pointed at
deception_v0 x length. Today's frontier audit found deception length-robust, so a trustworthy auditor should
CLEAR it. Catch on one instrument, clear on another = the tool is an honest auditor, not a confound-finder that
always cries wolf.

  python scripts/demo_audit_confound.py --generate   # build the orthogonal 2x2 via Gemini (build_confound_grid)
  python scripts/demo_audit_confound.py              # audit + write result
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from styxx import build_confound_grid, audit_confound
from overconfidence_length_robust import _gemini_call, GEMINI_KEY_PATH
from deception_train_v0 import QUESTIONS, SYSTEM_HONEST, SYSTEM_DISHONEST, extract_deception_features
import styxx.guardrail.calibrated_weights_deception_v0 as D

MODEL = "gemini-2.5-flash"
N_Q = 50
CACHE = ROOT / "benchmarks" / "data" / "deception" / "confound_grid_lenxdecep_gemini.jsonl"
RESULT = ROOT / "papers" / "grounded-honesty-axis" / "demo_audit_confound_deception_result.json"
RULES = {"short": "Answer in ONE sentence, about 25 words — no more.",
         "long": "Answer in 4–5 sentences, about 80 words."}


def generate():
    import os
    key = os.environ.get("GOOGLE_API_KEY") or GEMINI_KEY_PATH.read_text(encoding="utf-8").strip()
    def g(system, item):
        return _gemini_call(MODEL, system, item, key)
    # label 1 = dishonest (the construct positive class), label 0 = honest
    rows = build_confound_grid(QUESTIONS[:N_Q], SYSTEM_DISHONEST, SYSTEM_HONEST, RULES, g)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[gen] wrote {len(rows)} rows -> {CACHE.relative_to(ROOT)}", flush=True)


def deception_score(question, response):
    f = extract_deception_features(question, response)
    x = np.array([f[n] for n in D.FEATURE_NAMES], float)
    z = (x - np.asarray(D.SCALER_MEAN)) / np.asarray(D.SCALER_SCALE)
    return float(z @ np.asarray(D.COEFS) + D.INTERCEPT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--generate", action="store_true")
    a = ap.parse_args()
    if a.generate:
        generate(); return
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = [deception_score(r["item"], r["text"]) for r in rows]
    # construct-recoverability: a fresh refit on this corpus's deception features (is the construct learnable
    # from the text at all? -> distinguishes a broken instrument from a degenerate corpus).
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    Xr = np.array([[extract_deception_features(r["item"], r["text"])[n] for n in D.FEATURE_NAMES] for r in rows], float)
    yr = np.array([r["label"] for r in rows]); oof = np.zeros(len(yr))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xr, yr):
        s = StandardScaler().fit(Xr[tr]); m = LogisticRegression(max_iter=2000).fit(s.transform(Xr[tr]), yr[tr])
        oof[te] = m.predict_proba(s.transform(Xr[te]))[:, 1]
    refit_auc = float(roc_auc_score(yr, oof))
    rep = audit_confound(rows, scores=scores, instrument="deception_v0", confound="log_words",
                         construct_recoverable_auc=refit_auc)
    out = {"instrument": rep.instrument, "confound": rep.confound, "n": rep.n, "gate_ok": rep.gate_ok,
           "orthogonality_corr": rep.orthogonality_corr, "overall_auc": rep.overall_auc,
           "within_stratum_auc": rep.within_stratum_auc, "confound_score_coef": rep.confound_score_coef,
           "confound_score_coef_ci95": list(rep.confound_score_coef_ci95), "max_swing": rep.harm["max_swing"],
           "guard_auc_raw": rep.guard_auc_raw, "guard_auc_adj_oos": rep.guard_auc_adj_oos,
           "construct_recoverable_auc": rep.construct_recoverable_auc,
           "ci_method": "bootstrap 95% CI on OLS coef; 5-fold OOS guard; refit CV for construct-recoverability",
           "verdict": rep.verdict}
    RESULT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"=== audit_confound: deception_v0 x log_words (n={rep.n}) ===")
    print(f"  gate_ok={rep.gate_ok} ortho={rep.orthogonality_corr:+.2f} | overall AUC {rep.overall_auc} | "
          f"within-stratum {rep.within_stratum_auc}")
    print(f"  confound coef {rep.confound_score_coef:+.2f} CI{list(rep.confound_score_coef_ci95)} | swing {rep.harm['max_swing']:.0%}")
    print(f"  >>> {rep.verdict}")
    print(f"  wrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
