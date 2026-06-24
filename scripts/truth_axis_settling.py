"""Settling experiment (PREREG_truth_axis_settling): does a WIDE construct recover the transferable truth
axis? Fit on wide_truthset, test on the large cross-verified ood_naturals. Offline.

  python scripts/truth_axis_settling.py --extract --model {qwen,llama} --which {wide,ood}
  python scripts/truth_axis_settling.py    # fit + OOD test + controls + verdict
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.feature_extraction.text import TfidfVectorizer

REPOS = {"qwen": ("Qwen/Qwen2.5-3B-Instruct", 19), "llama": ("meta-llama/Llama-3.2-3B-Instruct", 14)}
DDIR = ROOT / "benchmarks" / "data" / "deception"
WIDE = DDIR / "wide_truthset.jsonl"; OOD = DDIR / "ood_naturals.jsonl"
def apath(tag, which): return DDIR / f"_settling_{which}_{tag}.npz"
def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def extract(tag, which):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, L = REPOS[tag]; rows = load(WIDE if which == "wide" else OOD)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="cuda",
                                                 output_hidden_states=True).eval()
    acts = []
    for r in rows:
        ids = tok(r["statement"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = model(**ids).hidden_states
        acts.append(hs[L][0, -1, :].float().cpu().numpy())
    extra = {"dom": np.array([r.get("domain", -1) for r in rows])} if which == "wide" else \
            {"cat": np.array([r["cat"] for r in rows])}
    np.savez_compressed(apath(tag, which), acts=np.array(acts, np.float32),
                        y=np.array([r["label_false"] for r in rows]), **extra)
    print(f"[extract] {tag}/{which}: {np.array(acts).shape}"); del model; torch.cuda.empty_cache()


def mm_dir(X, y): return X[y == 1].mean(0) - X[y == 0].mean(0)


def boot_ci(y, score, it=2000):
    rng = np.random.default_rng(0); y = np.asarray(y); n = len(y); out = []
    for _ in range(it):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        out.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def analyze():
    txtw = [r["statement"] for r in load(WIDE)]; txto = [r["statement"] for r in load(OOD)]
    for tag, (repo, L) in REPOS.items():
        if not apath(tag, "wide").exists() or not apath(tag, "ood").exists(): continue
        dw = np.load(apath(tag, "wide")); Aw, yw, domw = dw["acts"], dw["y"], dw["dom"]
        do = np.load(apath(tag, "ood")); Ao, yo = do["acts"], do["y"]
        s = StandardScaler().fit(Aw); Wd = s.transform(Aw); Od = s.transform(Ao)
        # wide-construct direction -> OOD
        dmm = mm_dir(Wd, yw); ood_mm = roc_auc_score(yo, Od @ dmm)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Wd, yw)
        ood_lr = roc_auc_score(yo, clf.decision_function(Od))
        best = max(ood_mm, ood_lr); best_score = Od @ dmm if ood_mm >= ood_lr else clf.decision_function(Od)
        ci = boot_ci(yo, best_score)
        # in-OOD-internal LOO ceiling (does a truth axis exist on naturals?)
        oof = np.zeros(len(yo))
        for tr, te in LeaveOneOut().split(Od):
            d = mm_dir(Od[tr], yo[tr]); oof[te] = Od[te] @ d
        ceil = roc_auc_score(yo, oof)
        ood_dir = mm_dir(Od, yo)  # in-OOD-internal direction (full)
        cos = float(dmm @ ood_dir / (np.linalg.norm(dmm) * np.linalg.norm(ood_dir) + 1e-9))
        # controls
        rng = np.random.default_rng(0); shuf = roc_auc_score(yo, Od @ mm_dir(Wd, rng.permutation(yw)))
        # BoW silence floors
        Vw = TfidfVectorizer().fit(txtw)
        bow = LogisticRegression(max_iter=2000).fit(Vw.transform(txtw).toarray(), yw)
        bow_ood = roc_auc_score(yo, bow.decision_function(Vw.transform(txto).toarray()))
        bow_ood = max(bow_ood, 1 - bow_ood)
        # verdict (frozen)
        if best >= 0.75 and ci[0] > 0.70 and cos >= 0.50:
            v = "AXIS-RECOVERABLE"
        elif best < 0.65 and ci[1] < 0.75 and ceil >= 0.75:
            v = "ROBUST-FAILURE"
        else:
            v = "PARTIAL"
        print(f"\n=== {tag} (L{L}, wide n={len(yw)}, OOD n={len(yo)}) ===")
        print(f"  wide-construct dir -> OOD: mass-mean {ood_mm:.3f} | logistic {ood_lr:.3f} | best {best:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}]")
        print(f"  cosine(wide dir, in-OOD-internal dir): {cos:+.3f}   (>=0.5 = recovers the real axis)")
        print(f"  in-OOD-internal LOO ceiling (axis EXISTS on naturals?): {ceil:.3f}")
        print(f"  controls: shuffle {shuf:.3f} | BoW-construct->OOD {bow_ood:.3f} (silence floor)")
        print(f"  >>> {v}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model", choices=list(REPOS), default="qwen"); ap.add_argument("--which", choices=["wide", "ood"], default="wide")
    a = ap.parse_args()
    if a.extract: extract(a.model, a.which)
    else: analyze()


if __name__ == "__main__":
    main()
