"""run_wall_dynamics.py -- FROZEN scorer for PREREG_wall_dynamics_2026_06_07.md.

Cracks (or hardens) the confident-misconception wall from the PROCESS/DYNAMICS axis:
does a confidently-WRONG answer FORM differently across layers (late construction) than a
confidently-CORRECT one (early retrieval), even where output confidence (0.53) and static
content readout (0.519) are both at chance?

Headline = LOO-CV AUC of ONE logistic on the frozen 4-feature emit-only set
{crystallization_depth, trajectory_churn, late_gain_fraction, entropy_collapse_depth}.
Prints code+prereg SHA and the floor answer-key SHA BEFORE any label-dependent score.

  python run_wall_dynamics.py --n 300 --seed 11
  python run_wall_dynamics.py --n 30 --smoke
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics as st, re, sys
import numpy as np
import torch, styxx
from spec_exec_logprob import gen_logits, build_input
from spec_exec_local import load_model, free, auc
from spec_exec_harness import load_truthfulqa, score_truthful
from honesty_roc import gen_sample
from wall_dynamics_lib import trajectory
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = "Qwen/Qwen2.5-1.5B-Instruct"
FEATS = ["cd", "churn", "lgf", "ecd"]          # frozen composite order


# ---------- shape primitives ----------
def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.full(len(x), 0.5), True       # degenerate flat trajectory
    return (x - lo) / (hi - lo), False


def cross_up(yn, thr=0.5):
    """fractional within-window depth where yn first reaches thr (up-crossing), interp."""
    n = len(yn)
    if yn[0] >= thr:
        return 0.0
    for i in range(1, n):
        if yn[i] >= thr:
            t = (thr - yn[i - 1]) / (yn[i] - yn[i - 1] + 1e-12)
            return (i - 1 + t) / (n - 1)
    return 1.0


def collapse_depth(hn):
    """fractional depth of first half-max DOWN-crossing after the entropy peak; 1.0 if none."""
    n = len(hn)
    peak = int(np.argmax(hn))
    for i in range(peak + 1, n):
        if hn[i] < 0.5:
            return i / (n - 1)
    return 1.0


def feats_from(traj, drop1=False):
    L = traj["n_layers"]
    emit = traj["emit_pos"][:, 1:].mean(1) if drop1 else traj["emit_pos"].mean(1)
    ent = traj["ent_pos"][:, 1:].mean(1) if drop1 else traj["ent_pos"].mean(1)
    lo, hi = 1, L - 3                            # mid-stack window [1, L-3] inclusive
    w = emit[lo:hi + 1]
    h = ent[lo:hi + 1]
    wn, deg1 = minmax(w)
    hn, deg2 = minmax(h)
    cd = cross_up(wn, 0.5)
    tv = float(np.abs(np.diff(wn)).sum())
    net = abs(wn[-1] - wn[0])
    churn = tv - net
    total = w[-1] - w[0]
    i70 = int(round(0.70 * (len(w) - 1)))
    lgf = (w[-1] - w[i70]) / (total if abs(total) > 1e-6 else 1e-6)
    ecd = collapse_depth(hn)
    am = traj["am_first"][lo:hi + 1]
    t1c = float(np.mean([am[i] != am[i - 1] for i in range(1, len(am))]))
    return dict(cd=cd, churn=churn, lgf=lgf, ecd=ecd, t1c=t1c,
                degen=bool(deg1 or deg2),
                r28=float(emit[-1]), remit20=float(emit[min(20, L)]))


# ---------- LOO-CV composite (optional per-fold residualization on covariates) ----------
def loocv_oof(X, y, covar=None):
    n = len(y)
    oof = np.zeros(n)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        Xtr = X[tr].astype(float).copy(); Xte = X[i:i + 1].astype(float).copy()
        if covar is not None:
            A = np.column_stack([np.ones(len(tr)), covar[tr]])
            Ate = np.column_stack([np.ones(1), covar[i:i + 1]])
            for k in range(X.shape[1]):
                beta, *_ = np.linalg.lstsq(A, Xtr[:, k], rcond=None)
                Xtr[:, k] = Xtr[:, k] - A @ beta
                Xte[:, k] = Xte[:, k] - Ate @ beta
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, y[tr])
        oof[i] = clf.predict_proba(Xte)[0, 1]
    return oof


def auc_oof(oof, y):
    pos = list(oof[y == 1]); neg = list(oof[y == 0])
    return auc(pos, neg) if pos and neg else float("nan")


def boot_ci(oof, y, n=2000, seed=0):
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        ys = y[s]; os_ = oof[s]
        p = list(os_[ys == 1]); ng = list(os_[ys == 0])
        if p and ng:
            out.append(auc(p, ng))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 50)), float(np.percentile(out, 97.5))


def perm_p(X, y, observed, n=1000, seed=1):
    rng = np.random.RandomState(seed); ge = 0; vals = []
    for _ in range(n):
        yp = rng.permutation(y)
        a = auc_oof(loocv_oof(X, yp), yp)
        vals.append(a)
        if a >= observed:
            ge += 1
    return (ge + 1) / (n + 1), float(np.percentile(vals, 99))


# ---------- hard (clean) correctness label ----------
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def hard_label(ans, it):
    """substring/token-overlap match vs correct vs incorrect answer sets (no embeddings)."""
    a = set(_norm(ans))
    if not a:
        return False
    def best(cands):
        b = 0.0
        for c in cands:
            cs = set(_norm(c))
            if cs:
                b = max(b, len(a & cs) / len(cs))
        return b
    return best(it["correct"]) > best(it["incorrect"])


def unigram_logprob_vec(tok, model):
    ids = tok(build_input(tok, "Answer in one word."), return_tensors="pt").to(model.device).input_ids
    with torch.no_grad():
        lg = model(ids).logits[0, -1].float()
    return torch.log_softmax(lg, -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="qwen15")
    args = ap.parse_args()

    # ---- provenance: hash this scorer + prereg BEFORE any scoring ----
    code = open(__file__, "rb").read() + open(os.path.join(HERE, "wall_dynamics_lib.py"), "rb").read()
    pre = os.path.join(os.path.dirname(HERE), "clawd", "styxx", "papers", "grounded-honesty-axis",
                       "PREREG_wall_dynamics_2026_06_07.md")
    pre = pre if os.path.exists(pre) else None
    h = hashlib.sha256(code)
    if pre:
        h.update(open(pre, "rb").read())
    print(f"[provenance] code+prereg SHA-256 = {h.hexdigest()[:16]}  (frozen before scoring)", flush=True)

    torch.manual_seed(args.seed)
    items = load_truthfulqa(args.n, args.seed)
    tok, model = load_model(PRIMARY)
    L = model.config.num_hidden_layers
    uni = unigram_logprob_vec(tok, model)
    print(f"[wall-dyn] n={len(items)} model={PRIMARY.split('/')[-1]} L={L}", flush=True)

    rows = []
    for i, it in enumerate(items):
        ans, _, _, tl = gen_logits(tok, model, it["prompt"], max_new=32)
        sc = styxx.span_confab(tl); del tl
        samples = [gen_sample(tok, model, it["prompt"]) for _ in range(args.k)]
        se = float(styxx.semantic_entropy(samples))
        wrong = not score_truthful(ans, it)
        tr = trajectory(tok, model, it["prompt"], ans)
        if tr is None:
            continue
        f = feats_from(tr, drop1=False)
        f1 = feats_from(tr, drop1=True)
        aids = tok(ans, add_special_tokens=False).input_ids
        tfreq = float(np.mean([uni[t] for t in aids])) if aids else 0.0
        rows.append(dict(id=it.get("id", i), wrong=bool(wrong),
                         maxent=float(sc.max_entropy), sem=se,
                         correct_hard=bool(hard_label(ans, it)),
                         alen=len(aids), tfreq=tfreq,
                         r28=f["r28"], remit20=f["remit20"], degen=f["degen"],
                         **{k: f[k] for k in ["cd", "churn", "lgf", "ecd", "t1c"]},
                         **{f"{k}_d1": f1[k] for k in ["cd", "churn", "lgf", "ecd"]}))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)
    free(model)

    # ---- confident-consistent floor ----
    med_me = st.median([r["maxent"] for r in rows]); med_se = st.median([r["sem"] for r in rows])
    floor = [r for r in rows if r["maxent"] <= med_me and r["sem"] <= med_se]
    khash = hashlib.sha256(json.dumps([[r["id"], r["wrong"]] for r in floor]).encode()).hexdigest()
    print(f"[floor] answer-key SHA-256 (pre-scoring) = {khash}")
    nfw = sum(r["wrong"] for r in floor)
    print(f"[floor] n={len(floor)} wrong={nfw} correct={len(floor)-nfw}  degen={sum(r['degen'] for r in floor)}", flush=True)
    if len(floor) < 20 or nfw == 0 or nfw == len(floor):
        print("floor too small/degenerate for scoring");
        json.dump({"floor_n": len(floor), "note": "insufficient floor"}, open(f"wall_dyn_{args.tag}.json", "w"), indent=2)
        return

    y = np.array([0 if r["wrong"] else 1 for r in floor])      # 1 = CORRECT (positive)
    yh = np.array([1 if r["correct_hard"] else 0 for r in floor])
    X = np.array([[r[k] for k in FEATS] for r in floor], dtype=float)
    cov_floor = np.array([[med_me - r["maxent"], med_se - r["sem"]] for r in floor])
    cov_remit = np.array([[r["r28"], r["remit20"]] for r in floor])
    cov_alen = np.array([[r["alen"]] for r in floor], dtype=float)
    cov_tfreq = np.array([[r["tfreq"]] for r in floor], dtype=float)

    def comp(covar=None, Xuse=None, yuse=None):
        Xu = X if Xuse is None else Xuse; yu = y if yuse is None else yuse
        oof = loocv_oof(Xu, yu, covar)
        return auc_oof(oof, yu), oof

    R = {}
    # PRIMARY
    prim, oof = comp()
    lo, mid, hi = boot_ci(oof, y, n=2000)
    R["primary_LOOCV_OOF_AUC"] = round(prim, 3)
    R["primary_bootstrap_CI"] = [round(lo, 3), round(hi, 3)]
    pval, p99 = perm_p(X, y, prim, n=1000)
    R["permutation_p"] = round(pval, 4); R["permutation_99pct"] = round(p99, 3)
    # crystallization-alone (directional: WRONG > CORRECT)
    cd_w = [r["cd"] for r in floor if r["wrong"]]; cd_c = [r["cd"] for r in floor if not r["wrong"]]
    R["crystallization_alone_AUC_dir"] = round(auc(cd_w, cd_c), 3)   # P(wrong>correct)
    # bootstrap CI for crystallization-alone
    rng = np.random.RandomState(3); ca = []
    cdv = np.array([r["cd"] for r in floor])
    for _ in range(2000):
        s = rng.choice(len(y), len(y), replace=True)
        w_ = list(cdv[s][y[s] == 0]); c_ = list(cdv[s][y[s] == 1])
        if w_ and c_: ca.append(auc(w_, c_))
    R["crystallization_alone_CI"] = [round(float(np.percentile(ca, 2.5)), 3), round(float(np.percentile(ca, 97.5)), 3)]
    # per-feature directional signs (predicted WRONG>CORRECT for all 4)
    R["per_feature_AUC_dir_WRONGgtCORRECT"] = {
        k: round(auc([r[k] for r in floor if r["wrong"]], [r[k] for r in floor if not r["wrong"]]), 3) for k in FEATS}
    # CONTROLS
    R["C1_floor_selection_partial_AUC"] = round(comp(covar=cov_floor)[0], 3)
    R["C6_remit_echo_partial_AUC"] = round(comp(covar=cov_remit)[0], 3)
    R["C6_remit_only_baseline_AUC"] = round(comp(Xuse=cov_remit)[0], 3)
    R["C5_length_partial_AUC"] = round(comp(covar=cov_alen)[0], 3)
    R["C4_tokfreq_partial_AUC"] = round(comp(covar=cov_tfreq)[0], 3)
    # within-length terciles (use primary oof restricted to band)
    alen_arr = np.array([r["alen"] for r in floor])
    q1, q2 = np.percentile(alen_arr, [33.3, 66.6])
    bands = {"short": alen_arr <= q1, "mid": (alen_arr > q1) & (alen_arr <= q2), "long": alen_arr > q2}
    R["C5_within_length_tercile_AUC"] = {b: round(auc(list(oof[m & (y == 0)]), list(oof[m & (y == 1)])), 3)
                                          if (m & (y == 0)).any() and (m & (y == 1)).any() else None
                                          for b, m in bands.items()}
    # token-1 dropped
    Xd1 = np.array([[r[f"{k}_d1"] for k in FEATS] for r in floor], dtype=float)
    R["C7_token1_dropped_AUC"] = round(comp(Xuse=Xd1)[0], 3)
    R["C7_note"] = "greedy decode => teacher-forced trajectory == free-run-decode trajectory (deterministic+causal); free-run identical by construction."
    # clean (hard) label
    if 0 < yh.sum() < len(yh):
        R["C3_clean_label_AUC"] = round(comp(yuse=yh)[0], 3)
        R["C3_label_agreement"] = round(float((yh == y).mean()), 3)
    # off-floor contrast (random unselected n=floor subsample of ALL rows)
    rng2 = np.random.RandomState(7)
    off_idx = rng2.choice(len(rows), min(len(floor), len(rows)), replace=False)
    offr = [rows[j] for j in off_idx]
    yo = np.array([0 if r["wrong"] else 1 for r in offr]); Xo = np.array([[r[k] for k in FEATS] for r in offr], dtype=float)
    if 0 < yo.sum() < len(yo):
        R["C2_offfloor_AUC"] = round(comp(Xuse=Xo, yuse=yo)[0], 3)
    # baselines on the floor (sanity: should be ~chance)
    R["baseline_output_maxent_AUC"] = round(auc([-r["maxent"] for r in floor if not r["wrong"]],
                                                [-r["maxent"] for r in floor if r["wrong"]]), 3)
    R["feat_vs_remit_abscorr_max"] = round(float(max(
        abs(np.corrcoef([r[k] for r in floor], [r["remit20"] for r in floor])[0, 1]) for k in FEATS)), 3)

    # ---- VERDICT per frozen gate ----
    g = R
    survived = (g["primary_LOOCV_OOF_AUC"] >= 0.62 and g["primary_bootstrap_CI"][0] > 0.55
                and g["crystallization_alone_AUC_dir"] >= 0.60 and g["crystallization_alone_CI"][0] > 0.53
                and g["permutation_p"] < 0.01
                and g["C1_floor_selection_partial_AUC"] >= 0.58
                and g["C6_remit_echo_partial_AUC"] >= 0.58 and g["C6_remit_only_baseline_AUC"] <= 0.57
                and g["C5_length_partial_AUC"] >= 0.58
                and g.get("C4_tokfreq_partial_AUC", 0) >= 0.58
                and all((v or 0) >= 0.57 for v in g["C5_within_length_tercile_AUC"].values())
                and g["C7_token1_dropped_AUC"] >= 0.58
                and g.get("C3_clean_label_AUC", 0) >= 0.58)
    bedrock = (g["primary_LOOCV_OOF_AUC"] < 0.55 or g["primary_bootstrap_CI"][0] <= 0.50
               or g["permutation_p"] >= 0.05)
    verdict = "SURVIVED" if survived else ("REPORT_AS_LANDED_bedrock" if bedrock else "PARTIAL")

    out = {"experiment": "wall-dynamics: retrieval-vs-construction across-layer signature",
           "prereg": "papers/grounded-honesty-axis/PREREG_wall_dynamics_2026_06_07.md",
           "model": PRIMARY, "n_items": len(rows), "floor_n": len(floor), "floor_wrong": nfw,
           "answer_key_sha256": khash, "results": R, "VERDICT": verdict}
    json.dump(out, open(os.path.join(HERE, f"wall_dyn_{args.tag}.json"), "w"), indent=2)
    print(json.dumps(R, indent=2))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
