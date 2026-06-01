"""SAE wall attack (feasibility step of WALL_FRONTIER.md). Does the cave become crisper in Gemma Scope
SAE-feature space than in the raw residual? If yes, the SAE is the ruler that sees through the wall.

Encode gemma-2-2b cave residuals through the Gemma Scope JumpReLU SAE (layer 20, 16k, l0~71); confidence-
matched lie/mistake; probe SAE features vs raw residual. Reconstruction check guards the pt-SAE/it-model and
layer-index mismatches. Bar (locked in WALL_FRONTIER): SAE AUROC >= raw AUROC + 0.05.  CPU.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from score_intent_bc import bin_balance

p = hf_hub_download("google/gemma-scope-2b-pt-res", "layer_20/width_16k/average_l0_71/params.npz")
S = np.load(p)
W_enc, b_enc, thr, W_dec, b_dec = S["W_enc"], S["b_enc"], S["threshold"], S["W_dec"], S["b_dec"]
print(f"SAE: d_model {W_enc.shape[0]} -> d_sae {W_enc.shape[1]}  (JumpReLU)")


def encode(x):
    pre = x @ W_enc + b_enc
    return np.where(pre > thr, pre, 0.0).astype(np.float32)


def decode(a):
    return a @ W_dec + b_dec


meta = json.load(open(os.path.join(HERE, "intent_metaxf_gemma.json"), encoding="utf-8"))
R = np.load(os.path.join(HERE, "residuals_intentxf_gemma.npz"))["residuals"].astype(np.float32)
cls = np.array([r["cls"] for r in meta["rows"]])
lmarg = np.array([r["letter_margin"] for r in meta["rows"]])
L = R.shape[1]

# resolve the residual-index <-> SAE-layer match by reconstruction quality (FVU)
print("\nreconstruction FVU by residual index (lower = in-distribution for this SAE):")
best_li, best_fvu = None, 9e9
for li in [c for c in (19, 20, 21, 22) if c < L]:
    x = R[:300, li, :]
    rec = decode(encode(x))
    fvu = float(((x - rec) ** 2).sum() / (((x - x.mean(0)) ** 2).sum() + 1e-9))
    l0 = float((encode(x) > 0).sum(1).mean())
    print(f"  idx {li}: FVU {fvu:.3f}   mean-L0 {l0:.0f}")
    if fvu < best_fvu:
        best_fvu, best_li = fvu, li
print(f"-> using residual index {best_li} (FVU {best_fvu:.3f})")

sw = np.where((cls == "lie") | (cls == "mistake"))[0]
bal = bin_balance(sw, lmarg[sw], (cls[sw] == "lie").astype(int), nbins=12, seed=0)
y = (cls[bal] == "lie").astype(int)
x_raw = R[bal, best_li, :]
x_sae = encode(x_raw)
tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)


def probe(X):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))


raw_auc = probe(x_raw)
sae_auc = probe(x_sae)
print(f"\nconfidence-matched cave probe (n={len(y)}, layer-idx {best_li}):")
print(f"  RAW residual ({x_raw.shape[1]}-d)   AUROC = {raw_auc:.3f}")
print(f"  SAE features ({x_sae.shape[1]}-d, sparse)  AUROC = {sae_auc:.3f}")
print(f"  delta (SAE - raw) = {sae_auc-raw_auc:+.3f}   bar >= +0.05 -> {sae_auc-raw_auc >= 0.05}")
ok = best_fvu < 0.5
print(f"\n  reconstruction usable (FVU<0.5): {ok}" + ("" if ok else "  -- SAE/residual mismatch, result inconclusive"))
print(f"RESULT = {'SAE SEES IT CRISPER' if (ok and sae_auc-raw_auc>=0.05) else ('no SAE advantage' if ok else 'INCONCLUSIVE (mismatch)')}")
json.dump({"experiment": "Gemma Scope SAE cave-feature vs raw (wall feasibility)",
           "prereg": "papers/grounded-honesty-axis/WALL_FRONTIER.md",
           "sae": "google/gemma-scope-2b-pt-res layer_20/width_16k/average_l0_71",
           "residual_index": best_li, "recon_fvu": best_fvu, "n": len(y),
           "raw_auc": raw_auc, "sae_auc": sae_auc, "delta": sae_auc - raw_auc},
          open(os.path.join(HERE, "intent_sae_result.json"), "w"), indent=2)
print("wrote intent_sae_result.json")
