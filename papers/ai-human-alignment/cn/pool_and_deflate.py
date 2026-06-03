# -*- coding: utf-8 -*-
"""
pool_and_deflate.py — pool per-concept betas across subjects (common voxel space), select voxels by
INTER-SUBJECT reliability, build ONE group brain RDM + noise ceiling, then run the GloVe-vs-deep-vs-
vision deflation. The correct, powerful method (fixes the v1 per-subject-voxel failure).
"""
import os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def distmat(P):
    P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    return 1.0 - P @ P.T


def partial_corr(a, b, Z):
    X = np.column_stack([np.ones(len(a)), Z])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(y, *cols):
    X = np.column_stack([np.ones_like(y)] + list(cols))
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))


def main():
    files = sorted(glob.glob(os.path.join(HERE, "betas_sub-*.npz")))
    if not files:
        print("no betas yet"); return
    dat = [np.load(f, allow_pickle=True) for f in files]
    common = sorted(set.intersection(*[set(d["concept_index"].tolist()) for d in dat]))
    print(f"{len(files)} subjects, {len(common)} common concepts", flush=True)
    # per-subject mean betas aligned to common concepts (S, Nc, K) float32
    M, ODD, EVEN = [], [], []
    for d in dat:
        ci = d["concept_index"].tolist(); pos = [ci.index(c) for c in common]
        M.append(d["mean"][pos].astype(np.float32))
        ODD.append(d["odd"][pos].astype(np.float32)); EVEN.append(d["even"][pos].astype(np.float32))
    M = np.stack(M); ODD = np.stack(ODD); EVEN = np.stack(EVEN)
    S, Nc, K = M.shape

    # INTER-SUBJECT voxel reliability over concepts
    Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    rel = np.zeros(K); cnt = 0
    for i in range(S):
        for j in range(i + 1, S):
            rel += (Z[i] * Z[j]).mean(0); cnt += 1
    rel /= max(cnt, 1)
    npos = int((rel > 0).sum())
    Ksel = min(10000, max(1000, npos))
    sel = np.argsort(rel)[-Ksel:]
    print(f"inter-subject voxel reliability: {npos}/{K} voxels >0, max {rel.max():.3f}, "
          f"selected top {Ksel} (mean rel {rel[sel].mean():.3f})", flush=True)

    group = M.mean(0)[:, sel]
    IU = np.triu_indices(Nc, 1)
    Rg = distmat(group)
    # noise ceiling (leave-one-subject-out) + within-subject split-half
    sub_rdms = [distmat(M[i][:, sel]) for i in range(S)]
    if S >= 2:
        lo = float(np.mean([np.corrcoef(sub_rdms[i][IU], distmat(np.delete(M, i, 0).mean(0)[:, sel])[IU])[0, 1] for i in range(S)]))
        up = float(np.mean([np.corrcoef(sub_rdms[i][IU], Rg[IU])[0, 1] for i in range(S)]))
    else:
        lo = up = float("nan")
    sh = np.mean([np.corrcoef(distmat(ODD[i][:, sel])[IU], distmat(EVEN[i][:, sel])[IU])[0, 1] for i in range(S)])
    print(f"noise ceiling [lo {lo:.3f}, up {up:.3f}]; within-subject split-half RDM reliab {sh:.3f}", flush=True)

    # predictors
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    pr = {t.replace("rdm__", ""): P[t][np.ix_(common, common)][IU] for t in P.files if t.startswith("rdm__")}
    cl = np.array([len(words[c]) for c in common], float); zc = (cl - cl.mean()) / (cl.std() + 1e-9)
    L = np.abs(zc[:, None] - zc[None, :])[IU][:, None]
    g = Rg[IU]
    glove = pr["GloVe"]; deep = np.mean([pr[k] for k in ["GPT2", "BERT"] if k in pr], 0)
    vis = np.mean([pr[k] for k in ["ResNet", "ViT"] if k in pr], 0)

    rows = {}
    print("\n=== partial-lexical RSA to the GROUP BRAIN RDM (672-concept, auditory, pooled) ===")
    for tag, v in [("GloVe(shallow)", glove), ("GPT2", pr.get("GPT2")), ("BERT", pr.get("BERT")),
                   ("deep-consensus", deep), ("ResNet(vis)", pr.get("ResNet")), ("ViT(vis)", pr.get("ViT"))]:
        if v is None:
            continue
        rows[tag] = partial_corr(v, g, L)
        pct = 100 * rows[tag] / lo if lo and lo > 0.02 else float("nan")
        print(f"  {tag:16s} {rows[tag]:+.3f}   ({pct:.0f}% of ceiling-lo)")

    Rr = {"lex": r2(g, L), "lex+gl": r2(g, glove, L), "lex+gl+vi": r2(g, glove, vis, L),
          "lex+gl+vi+dp": r2(g, glove, vis, deep, L), "lex+vi+dp": r2(g, vis, deep, L)}
    ud = Rr["lex+gl+vi+dp"] - Rr["lex+gl+vi"]; ug = Rr["lex+gl+vi+dp"] - Rr["lex+vi+dp"]
    print(f"\nvariance partition: GloVe over lex {100*(Rr['lex+gl']-Rr['lex']):+.2f}% | "
          f"DEEP unique beyond GloVe+vision {100*ud:+.2f}% <-- the test | GloVe unique {100*ug:+.2f}%")

    best = max(rows.values()); dr = rows.get("deep-consensus", 0); gr = rows.get("GloVe(shallow)", 0)
    if (lo is None) or (lo < 0.04) or best < 0.04:
        verdict = (f"STILL TOO NOISY: noise ceiling {lo:.3f}, best predictor {best:.3f}. The pooled brain RDM does not "
                   f"carry enough semantic signal to test the deflation. Need more subjects or a semantic ROI. Not a result.")
    elif dr > gr + 0.02 and ud > 0.005:
        verdict = (f"DEPTH REACHES THE BRAIN: deep {dr:.3f} > GloVe {gr:.3f}, deep-unique {100*ud:.1f}% beyond GloVe+vision. "
                   f"At 672-concept pooled high-SNR, the deep advantage IS in the brain -> reading (a) MEASUREMENT (Mitchell was "
                   f"underpowered). Cross-lingual (Chinese, auditory) -> not English/visual-specific.")
    elif abs(dr - gr) <= 0.02:
        verdict = (f"DEFLATION HOLDS at pooled high-SNR: GloVe {gr:.3f} ~ deep {dr:.3f} over 672 concepts (ceiling {lo:.3f}), "
                   f"deep-unique {100*ud:.1f}%. The brain's concept geometry IS shallow co-occurrence -> reading (b) SUBSTANCE. "
                   f"Cross-lingual confirmation.")
    else:
        verdict = f"MIXED: deep {dr:.3f} vs GloVe {gr:.3f}, deep-unique {100*ud:.1f}%, ceiling {lo:.3f}."
    print(f"\n>>> {verdict}")
    out = {"n_subjects": S, "n_concepts": Nc, "n_voxels_selected": int(Ksel),
           "noise_ceiling": [round(float(lo), 3), round(float(up), 3)], "within_subject_reliability": round(float(sh), 3),
           "partial_lexical_RSA": {k: round(v, 3) for k, v in rows.items()},
           "deep_unique_pct": round(100 * ud, 2), "glove_unique_pct": round(100 * ug, 2), "verdict": verdict}
    open(os.path.join(HERE, "cn_pooled_result.json"), "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote cn_pooled_result.json")


if __name__ == "__main__":
    main()
