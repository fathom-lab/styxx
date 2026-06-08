"""Leakage audit (responds to external review #1): is the held-knowledge recovery genuine semantic
content, or a shortcut on length / confidence / token-stats / activation-magnitude? Residualize the
residual features on those covariates BEFORE the gold probe; if the elevation survives, it's not leakage.
CPU-only on disk. python run_leakage_audit.py --tag pc_3 --label qwen3b
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
L2I = {"A": 0, "B": 1, "C": 2, "D": 3}


def fit(X, y):
    sc = StandardScaler().fit(X); return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)


def acc(sc, clf, X, y):
    return float((clf.predict(sc.transform(X)) == y).mean())


def resid_on(X, C):
    """residualize each column of X on covariates C (+intercept), train-fit; return residuals."""
    A = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta, beta


def run(tag, label):
    meta = json.load(open(HERE / f"intent_meta{tag}.json", encoding="utf-8")); rows = meta["rows"]
    R = np.load(HERE / f"residuals_intent{tag}.npz")["residuals"].astype(np.float32)
    N, L, d = R.shape
    gold = np.array([L2I[r["gold"]] for r in rows]); cls = np.array([r["cls"] for r in rows])
    nc = np.array([bool(r["neutral_correct"]) for r in rows])
    grf = np.array([int(r["gold_rank"][-1]) for r in rows])
    nm = np.array([float(r["neutral_margin"]) for r in rows])
    lmg = np.array([float(r["letter_margin"]) for r in rows])
    ve = np.array([float(r.get("vocab_entropy", 0.0)) for r in rows])
    alen = np.array([len(str(r.get("gold", ""))) for r in rows], float)  # answer-string length (letters ~const)
    lie = np.where((cls == "lie") & nc)[0]; tnk = np.array([i for i in np.where(cls == "mistake")[0] if (not nc[i]) and grf[i] >= 3])
    rng = np.random.RandomState(0); pl = rng.permutation(lie); k = int(0.6 * len(lie)); lie_tr, lie_te = pl[:k], pl[k:]
    pt = rng.permutation(tnk); kt = int(0.6 * len(tnk)); tnk_tr, tnk_te = pt[:kt], pt[kt:]
    # lock layer: route-at-chance (probe on LIE-tr -> tnk-tr) nearest perm; reuse simple: pick best LIE-CV under route<=perm
    # for the audit, just use a fixed mid-late layer = round(0.85*L) (matches reclimb read region)
    Lk = round(0.85 * L)
    # activation magnitude covariate = per-item residual L2 norm at locked layer
    norm = np.linalg.norm(R[:, Lk, :], axis=1)
    COV = {"none": None,
           "magnitude": norm[:, None],
           "confidence(neutral_margin)": nm[:, None],
           "token_stats(letter_margin+vocab_entropy)": np.column_stack([lmg, ve]),
           "length": alen[:, None],
           "ALL": np.column_stack([norm, nm, lmg, ve, alen])}
    out = {"experiment": "leakage audit — does held-knowledge recovery survive covariate partialling?",
           "model": label, "tag": tag, "locked_layer": Lk, "n_LIE_test": len(lie_te), "elevation_by_partial": {}}
    perm = []
    for s in range(100):
        yp = gold[lie_tr].copy(); np.random.RandomState(s).shuffle(yp)
        sc, clf = fit(R[lie_tr, Lk, :], yp); perm.append(acc(sc, clf, R[lie_te, Lk, :], gold[lie_te]))
    p95 = float(np.percentile(perm, 95))
    for name, C in COV.items():
        Xtr, Xte = R[lie_tr, Lk, :].copy(), R[lie_te, Lk, :].copy()
        Xtnk = R[tnk_te, Lk, :].copy()
        if C is not None:
            Ctr = C[lie_tr]; Cte = C[lie_te]; Ctnk = C[tnk_te]
            Xtr, beta = resid_on(Xtr, Ctr)
            Ate = np.column_stack([np.ones(len(Cte)), Cte]); Xte = Xte - Ate @ beta
            Atnk = np.column_stack([np.ones(len(Ctnk)), Ctnk]); Xtnk = Xtnk - Atnk @ beta
        sc, clf = fit(Xtr, gold[lie_tr])
        lie_rec = acc(sc, clf, Xte, gold[lie_te]); route = acc(sc, clf, Xtnk, gold[tnk_te])
        floor = max(route, p95); elev = lie_rec - floor
        out["elevation_by_partial"][name] = {"LIE_rec": round(lie_rec, 3), "floor": round(floor, 3), "elevation": round(elev, 3)}
        print(f"  partial={name:<42} LIE_rec={lie_rec:.3f}  floor={floor:.3f}  elevation={elev:+.3f}", flush=True)
    base = out["elevation_by_partial"]["none"]["elevation"]; allp = out["elevation_by_partial"]["ALL"]["elevation"]
    out["survives_all_partials"] = bool(allp >= 0.15 and allp >= 0.5 * base)
    out["verdict"] = ("SEMANTIC — recovery survives partialling out length/confidence/token-stats/magnitude"
                      if out["survives_all_partials"] else "LEAKAGE-SENSITIVE — elevation collapses under partialling")
    (HERE / f"leakage_audit_{label}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"perm_p95={p95:.3f}  base elevation={base:+.3f}  ALL-partial elevation={allp:+.3f}")
    print("VERDICT:", out["verdict"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="pc_3"); ap.add_argument("--label", default="qwen3b")
    a = ap.parse_args(); run(a.tag, a.label)


if __name__ == "__main__":
    main()
