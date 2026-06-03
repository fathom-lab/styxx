# -*- coding: utf-8 -*-
"""
feature_decomposition.py — WHICH human-meaning dimensions does deep capture better than shallow?
Per-feature cross-validated prediction (embedding -> each of the 54 human features), dimensionality-
matched (PCA-50 both). Ranks features by the deep-minus-shallow advantage -> an interpretable map of
where depth helps (e.g. abstract/affective) vs where word-counting suffices (e.g. concrete/perceptual).
"""
import os, csv, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb
from encode_cv import cv_encode, pca_k

GLOSS = {  # rough English gloss for readability (Chinese feature names)
    "视觉": "visual", "明亮度": "brightness", "黑暗度": "darkness", "颜色": "color", "图案": "pattern",
    "大": "big", "小": "small", "运动": "motion", "声音": "sound", "触觉": "touch", "温度": "temperature",
    "味道": "taste", "气味": "smell", "形状": "shape", "重量": "weight", "情感": "emotion", "快乐": "happy",
    "悲伤": "sad", "愤怒": "anger", "恐惧": "fear", "社会": "social", "人物": "person", "身体": "body",
    "时间": "time", "空间": "space", "数量": "quantity", "价值": "value", "因果": "causal", "认知": "cognition",
}


def main():
    rows = list(csv.reader(open(os.path.join(HERE, "annot/sf/feature.csv"), encoding="utf-8-sig")))
    feats = rows[0][1:]
    H = np.load(os.path.join(HERE, "human_features.npy"))     # (672, 54)
    _, gpt2 = load_emb("GPT2.mat"); _, glove = load_emb("GloVe.mat")
    deepE = pca_k(gpt2, 50); shalE = pca_k(glove, 50)         # dim-matched

    _, _, r_deep = cv_encode(deepE, H)       # per-feature correlation (54,)
    _, _, r_shal = cv_encode(shalE, H)
    adv = r_deep - r_shal
    order = np.argsort(adv)[::-1]
    print("=== per-feature: deep(GPT2,50d) vs shallow(GloVe,50d) at predicting each HUMAN feature ===")
    print(f"overall mean: deep {r_deep.mean():+.3f}  shallow {r_shal.mean():+.3f}  advantage {adv.mean():+.3f}\n")
    print("TOP — depth helps most:")
    for i in order[:8]:
        g = GLOSS.get(feats[i], "")
        print(f"  {feats[i]:6s}{('('+g+')') if g else '':12s} deep {r_deep[i]:+.3f}  shallow {r_shal[i]:+.3f}  adv {adv[i]:+.3f}")
    print("\nBOTTOM — word-counting already suffices (or wins):")
    for i in order[-6:]:
        g = GLOSS.get(feats[i], "")
        print(f"  {feats[i]:6s}{('('+g+')') if g else '':12s} deep {r_deep[i]:+.3f}  shallow {r_shal[i]:+.3f}  adv {adv[i]:+.3f}")

    import json
    out = {"mean_deep": round(float(r_deep.mean()), 3), "mean_shallow": round(float(r_shal.mean()), 3),
           "mean_advantage": round(float(adv.mean()), 3),
           "per_feature": {feats[i]: {"gloss": GLOSS.get(feats[i], ""), "deep": round(float(r_deep[i]), 3),
                                      "shallow": round(float(r_shal[i]), 3), "adv": round(float(adv[i]), 3)} for i in range(len(feats))}}
    json.dump(out, open(os.path.join(HERE, "feature_decomposition_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote feature_decomposition_result.json")


if __name__ == "__main__":
    main()
