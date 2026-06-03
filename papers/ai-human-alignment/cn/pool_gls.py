# -*- coding: utf-8 -*-
"""
pool_gls.py — pool GLMsingle per-concept betas across subjects (common voxel space), select voxels by
INTER-SUBJECT reliability, build the group brain RDM + noise ceiling, run the GloVe-vs-deep-vs-vision
deflation. Alignment is baked into the GLMsingle design (concept index = embedding index), so NO shift.
"""
import os, glob, json, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))


def distmat(P):
    P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); return 1 - P @ P.T


def pcorr(a, b, Z):
    X = np.column_stack([np.ones(len(a)), Z]); ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]; rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(y, *cols):
    X = np.column_stack([np.ones_like(y)] + list(cols)); res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))


def main():
    fs = sorted(glob.glob(os.path.join(HERE, "gls_betas_sub-*.npz")))
    dat = [np.load(f, allow_pickle=True) for f in fs]
    common = sorted(set.intersection(*[set(d["concept_index"].tolist()) for d in dat]))
    # drop concepts that are all-zero (unestimated) in any subject
    good = []
    for c in common:
        ok = True
        for d in dat:
            pos = d["concept_index"].tolist().index(c)
            if np.all(d["mean"][pos] == 0):
                ok = False; break
        if ok:
            good.append(c)
    common = good
    M = np.stack([d["mean"][[d["concept_index"].tolist().index(c) for c in common]].astype(np.float32) for d in dat])
    S, Nc, K = M.shape; IU = np.triu_indices(Nc, 1)
    print(f"{S} subjects, {Nc} concepts, {K} voxels", flush=True)

    Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    rel = np.zeros(K); cnt = 0
    for i in range(S):
        for j in range(i + 1, S):
            rel += (Z[i] * Z[j]).mean(0); cnt += 1
    rel /= max(cnt, 1)
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    cl = np.array([len(words[c]) for c in common], float); zc = (cl - cl.mean()) / (cl.std() + 1e-9)
    L = np.abs(zc[:, None] - zc[None, :])[IU][:, None]
    prr = {t.replace("rdm__", ""): P[t][np.ix_(common, common)][IU] for t in P.files if t.startswith("rdm__")}
    glove = prr["GloVe"]; deep = (prr["GPT2"] + prr["BERT"]) / 2; vis = (prr["ResNet"] + prr["ViT"]) / 2

    print(f"{'K':>6} {'ceiling':>8} {'GloVe':>8} {'GPT2':>8} {'BERT':>8} {'deep':>8} {'vision':>8}")
    best = {}
    for Ksel in [1000, 2000, 5000, 10000, 20000]:
        if Ksel > K:
            continue
        sel = np.argsort(rel)[-Ksel:]
        subR = [distmat(M[i][:, sel]) for i in range(S)]
        if S >= 2:
            lo = float(np.mean([np.corrcoef(subR[i][IU], distmat(np.delete(M, i, 0).mean(0)[:, sel])[IU])[0, 1] for i in range(S)]))
        else:
            lo = float("nan")
        g = distmat(M.mean(0)[:, sel])[IU]
        row = {"ceiling": lo, "GloVe": pcorr(glove, g, L), "GPT2": pcorr(prr["GPT2"], g, L),
               "BERT": pcorr(prr["BERT"], g, L), "deep": pcorr(deep, g, L), "vision": pcorr(vis, g, L)}
        print(f"{Ksel:>6} {lo:>+8.3f} {row['GloVe']:>+8.3f} {row['GPT2']:>+8.3f} {row['BERT']:>+8.3f} {row['deep']:>+8.3f} {row['vision']:>+8.3f}")
        if not best or row["deep"] > best.get("deep", -9):
            best = {"K": Ksel, **row, "g": g}

    # variance partition at the best K
    g = best.pop("g")
    ud = r2(g, glove, vis, deep, L) - r2(g, glove, vis, L)
    bp = max(best["GloVe"], best["deep"], best["vision"])
    if bp < 0.05:
        verdict = f"STILL too noisy even with GLMsingle (best predictor {bp:.3f}, ceiling {best['ceiling']:.3f}). Limiter is deeper than estimation."
    elif best["deep"] > best["GloVe"] + 0.02 and ud > 0.005:
        verdict = f"DEPTH REACHES THE BRAIN: deep {best['deep']:.3f} > GloVe {best['GloVe']:.3f}, deep-unique {100*ud:.1f}%. Reading (a) MEASUREMENT (Chinese, auditory, 672-concept, GLMsingle)."
    elif abs(best["deep"] - best["GloVe"]) <= 0.02:
        verdict = f"DEFLATION HOLDS at GLMsingle high-SNR: GloVe {best['GloVe']:.3f} ~ deep {best['deep']:.3f}, deep-unique {100*ud:.1f}%. Reading (b) SUBSTANCE. Cross-lingual confirmation."
    else:
        verdict = f"MIXED: deep {best['deep']:.3f} vs GloVe {best['GloVe']:.3f}, deep-unique {100*ud:.1f}%."
    print(f"\n>>> {verdict}")
    json.dump({"n_subjects": S, "n_concepts": Nc, "best": {k: round(v, 3) for k, v in best.items()},
               "deep_unique_pct": round(100 * ud, 2), "verdict": verdict},
              open(os.path.join(HERE, "cn_gls_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote cn_gls_result.json")


if __name__ == "__main__":
    main()
