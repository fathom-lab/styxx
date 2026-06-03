# -*- coding: utf-8 -*-
"""
run_cn_deflation.py — THE decisive test, on real high-power neural data.

Builds the group brain RDM (+ noise ceiling) from the per-subject GLM RDMs (rdm_sub-*.npz), then runs
the deflation over 672 concepts: does the DEEP model (GPT2/BERT) predict the human brain BEYOND shallow
co-occurrence (GloVe) and vision (ResNet/ViT)? This is the Chinese, auditory (no visual confound),
672-concept, high-SNR resolution of measurement-vs-substance the Mitchell-60 data couldn't reach.
"""
import os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))


def partial_corr(a, b, Z):
    if Z is None or (hasattr(Z, "shape") and Z.shape[1] == 0):
        return float(np.corrcoef(a, b)[0, 1])
    X = np.column_stack([np.ones(len(a)), Z])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(y, *cols):
    X = np.column_stack([np.ones_like(y)] + list(cols))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    return float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))


def main():
    subs = sorted(glob.glob(os.path.join(HERE, "rdm_sub-*.npz")))
    if not subs:
        print("no subject RDMs yet"); return
    persub = [np.load(s, allow_pickle=True) for s in subs]
    # common concept set across subjects
    common = set(persub[0]["concept_index"].tolist())
    for p in persub[1:]:
        common &= set(p["concept_index"].tolist())
    common = sorted(common)
    print(f"{len(subs)} subjects, {len(common)} common concepts", flush=True)

    # each subject's RDM restricted to common concepts
    sub_rdms = []
    for p in persub:
        ci = p["concept_index"].tolist()
        pos = [ci.index(c) for c in common]
        sub_rdms.append(p["rdm"][np.ix_(pos, pos)])
    sub_rdms = np.array(sub_rdms)
    group = sub_rdms.mean(0)
    N = len(common); IU = np.triu_indices(N, 1)
    rel = [float(p["reliability"]) for p in persub]
    print(f"per-subject split-half RDM reliability: {[round(r,3) for r in rel]}", flush=True)

    # noise ceiling (inter-subject)
    if len(subs) >= 2:
        up, lo = [], []
        for i in range(len(sub_rdms)):
            others = np.delete(sub_rdms, i, axis=0).mean(0)
            up.append(np.corrcoef(sub_rdms[i][IU], group[IU])[0, 1])
            lo.append(np.corrcoef(sub_rdms[i][IU], others[IU])[0, 1])
        ceil = (float(np.mean(lo)), float(np.mean(up)))
    else:
        ceil = (rel[0], rel[0])  # single subject: use its split-half reliability as the ceiling proxy
    print(f"noise ceiling: [{ceil[0]:.3f}, {ceil[1]:.3f}]", flush=True)

    # predictors over the common concepts
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    pred = {t.replace("rdm__", ""): P[t][np.ix_(common, common)] for t in P.files if t.startswith("rdm__")}
    # lexical control: Chinese character count
    cl = np.array([len(words[c]) for c in common], float)
    zc = (cl - cl.mean()) / (cl.std() + 1e-9)
    L = np.abs(zc[:, None] - zc[None, :])[IU][:, None]

    g = group[IU]
    deep = {k: pred[k][IU] for k in ["GPT2", "BERT"] if k in pred}
    shallow = {k: pred[k][IU] for k in ["GloVe", "fastText"] if k in pred}
    vis = {k: pred[k][IU] for k in ["ResNet", "ViT"] if k in pred}
    deep_cons = np.mean(list(deep.values()), axis=0)
    vis_cons = np.mean(list(vis.values()), axis=0)
    glove = shallow["GloVe"]

    cl_lo = ceil[0]
    print("\n=== partial-lexical RSA to the HUMAN BRAIN (672-concept, auditory, high-SNR) ===")
    rows = {}
    for tag, v in [("GloVe(shallow)", glove), ("GPT2(deep)", deep.get("GPT2")), ("BERT(deep)", deep.get("BERT")),
                   ("deep-consensus", deep_cons), ("ResNet(vision)", vis.get("ResNet")), ("ViT(vision)", vis.get("ViT"))]:
        if v is None:
            continue
        r = partial_corr(v, g, L); rows[tag] = r
        print(f"  {tag:16s} {r:+.3f}   ({100*r/cl_lo:.0f}% of ceiling-lo)")

    # variance partition: does DEEP add beyond shallow + vision?
    R = {
        "lex": r2(g, L),
        "lex+glove": r2(g, glove, L),
        "lex+glove+vis": r2(g, glove, vis_cons, L),
        "lex+glove+vis+deep": r2(g, glove, vis_cons, deep_cons, L),
        "lex+vis+deep": r2(g, vis_cons, deep_cons, L),
    }
    unique_deep = R["lex+glove+vis+deep"] - R["lex+glove+vis"]
    unique_glove = R["lex+glove+vis+deep"] - R["lex+vis+deep"]
    print(f"\nvariance partition (brain R^2):")
    print(f"  GloVe total over lex            : {100*(R['lex+glove']-R['lex']):+.2f}%")
    print(f"  DEEP unique beyond GloVe+vision : {100*unique_deep:+.2f}%   <-- the test")
    print(f"  GloVe unique beyond deep+vision : {100*unique_glove:+.2f}%")
    print(f"  deep-vs-GloVe geometry RSA      : {np.corrcoef(deep_cons, glove)[0,1]:+.3f}")

    deep_r = rows.get("deep-consensus", 0); glove_r = rows.get("GloVe(shallow)", 0)
    if deep_r > glove_r + 0.02 and unique_deep > 0.005:
        verdict = (f"DEPTH REACHES THE BRAIN: deep models beat shallow co-occurrence at the human brain "
                   f"(deep {deep_r:.3f} > GloVe {glove_r:.3f}), uniquely adding {100*unique_deep:.1f}% beyond GloVe+vision. "
                   f"At 672-concept high-SNR resolution, the deep advantage real in behavior IS in the brain -> reading (a) MEASUREMENT "
                   f"(the Mitchell-60 deflation was underpowered). Replicates in Chinese -> not English-specific.")
    elif abs(deep_r - glove_r) <= 0.02:
        verdict = (f"DEFLATION HOLDS at high SNR: GloVe ({glove_r:.3f}) ~ deep ({deep_r:.3f}) even with 672 concepts, "
                   f"deep-unique {100*unique_deep:.1f}%. The brain's concept geometry really is shallow co-occurrence -> reading (b) "
                   f"SUBSTANCE. Strong, cross-lingual (Chinese) confirmation.")
    else:
        verdict = f"MIXED: deep {deep_r:.3f} vs GloVe {glove_r:.3f}, deep-unique {100*unique_deep:.1f}%."
    print(f"\n>>> {verdict}")
    out = {"n_subjects": len(subs), "n_concepts": N, "noise_ceiling": ceil, "reliability": rel,
           "partial_lexical_RSA": {k: round(v, 3) for k, v in rows.items()},
           "deep_unique_beyond_glove_vision_pct": round(100 * unique_deep, 2),
           "glove_unique_pct": round(100 * unique_glove, 2), "verdict": verdict}
    import json
    open(os.path.join(HERE, "cn_deflation_result.json"), "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote cn_deflation_result.json")


if __name__ == "__main__":
    main()
