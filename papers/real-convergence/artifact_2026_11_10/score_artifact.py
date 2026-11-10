"""score_artifact.py: score the sealed predictions of PREREG_start_token_artifact_2026_11_10.md.

Lead-written, 2026-11-10, sealed by hash with the prereg and extract_small.py before any real extraction.
Reads the per-template states written by extract_small.py, the kit copy of the lab's committed files, and
nothing else. numpy only. Thresholds are the constants below and are not to be edited after the seal.

Usage: py -3.12 -X utf8 -I -B score_artifact.py <kit_dir> <states_dir> <out.json>
"""
import json
import os
import sys

import numpy as np

KIT, SDIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, KIT)
import reproduce as kit  # distmat, partial_corr, lex_design, flatten: the kit copy of the lab's method

# ---- sealed thresholds ----
T_NORM_RATIO_MIN = 20.0        # P1, P6: models without a start token
T_NORM_RATIO_MAX_CONTROL = 3.0  # P1: model with a start token
T_SHARE8_MIN = 0.90            # P2, P6
T_AUC_MIN = 0.99               # P2, P6
T_SHARE7_MAX = 0.30            # P3, P6
T_GAIN_MIN = 0.30              # P4
T_R7_MIN = 0.50                # P4
T_SELF_RSA_MIN = 0.95          # P5
T_ROWS_OK_MIN = 90             # P7 (of 96)
T_ROW_REL_ERR = 0.05           # P7
T_CONFIRM_PAIR_TOL = 0.02      # P8
MIN_CLASS = 10                 # P2 testable only if at least this many words in each class
NULL_DRAWS, NULL_SEED = 200, 0

A, B, C = "Qwen2.5-1.5B", "Llama-3.2-1B", "Qwen2.5-0.5B"   # A, C: no start token expected. B: start token.


def load(setname, model):
    p = os.path.join(SDIR, "states__%s__%s.npz" % (setname, model))
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {"states": z["states"].astype(np.float64), "seq_len": z["seq_len"],
            "words": [str(w) for w in z["words"]], "cats": np.array([str(c) for c in z["cats"]])}


def pc1(x):
    xc = x - x.mean(0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    return float(s[0] ** 2 / (s ** 2).sum()), u[:, 0] * s[0]


def auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    a = float(gt / (len(pos) * len(neg)))
    return max(a, 1.0 - a)


def up(x):
    n = x.shape[0]
    return kit.distmat(x)[np.triu_indices(n, k=1)]


def rsa(x, y):
    return float(np.corrcoef(up(x), up(y))[0, 1])


def spearman(x, y):
    a, b = up(x), up(y)
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def within_cat_null(x, y, cats):
    rng = np.random.default_rng(NULL_SEED)
    a = up(x)
    groups = [np.where(cats == c)[0] for c in dict.fromkeys(cats.tolist())]
    vals = []
    for _ in range(NULL_DRAWS):
        perm = np.arange(len(cats))
        for g in groups:
            perm[g] = rng.permutation(g)
        vals.append(np.corrcoef(a, up(y[perm]))[0, 1])
    return float(np.mean(vals)), float(np.std(vals))


def describe(d):
    """everything computed for one model on one set"""
    st, sl = d["states"], d["seq_len"]
    norms = np.linalg.norm(st, axis=2)                       # words x templates
    bare = sl[:, 0]
    alone = (bare == 1).astype(int)
    single = (bare == bare.min()).astype(int)
    ratio = norms[:, 0] / np.median(norms[:, 1:], axis=1)
    rep8, rep7 = st.mean(1), st[:, 1:].mean(1)
    s8, p8 = pc1(rep8)
    s7, p7 = pc1(rep7)
    return {
        "n_words": int(st.shape[0]), "n_alone_at_position_0": int(alone.sum()),
        "n_single_token_words": int(single.sum()), "bare_len_min": int(bare.min()),
        "adds_start_token": bool(bare.min() >= 2),
        "norm_ratio_median_alone": None if alone.sum() == 0 else float(np.median(ratio[alone == 1])),
        "norm_ratio_median_single": float(np.median(ratio[single == 1])),
        "norm_ratio_median_not_single": None if single.sum() == len(single) else float(np.median(ratio[single == 0])),
        "pc1_share_8_templates": s8, "auc_pc1_vs_alone_8": auc(p8, alone),
        "pc1_share_7_templates": s7, "auc_pc1_vs_alone_7": auc(p7, alone),
        "self_rsa_8_vs_7": rsa(rep8, rep7),
        "_rep8": rep8, "_rep7": rep7, "_alone": alone,
    }


def verdict(ok):
    return "PASS" if ok else "FAIL"


out = {"prereg": "PREREG_start_token_artifact_2026_11_10.md", "sets": {}, "predictions": []}
toks = json.load(open(os.path.join(KIT, "lex_tokens.json"), encoding="utf-8"))
stored = np.load(os.path.join(KIT, "contextual_reps.npz"))
confirm = json.load(open(os.path.join(KIT, "real_convergence_confirm_result.json"), encoding="utf-8"))
desc = {}
for setname in ("original", "fresh"):
    desc[setname] = {}
    for m in (A, B, C):
        d = load(setname, m)
        desc[setname][m] = None if d is None else dict(describe(d), words=d["words"], cats=d["cats"])

# ---- P7: provenance on the original set (positive control for the run itself) ----
prov = {}
for m in (A, B):
    d = desc["original"][m]
    if d is None:
        prov[m] = None
        continue
    new, old = d["_rep8"], stored["fixed__" + m].astype(np.float64)
    rel = np.linalg.norm(new - old, axis=1) / (np.linalg.norm(old, axis=1) + 1e-12)
    prov[m] = {"rows_within_tolerance": int((rel <= T_ROW_REL_ERR).sum()), "of": int(len(rel)),
               "median_relative_error": float(np.median(rel)), "max_relative_error": float(rel.max()),
               "rsa_new_vs_stored": rsa(new, old)}
    prov[m]["ok"] = prov[m]["rows_within_tolerance"] >= T_ROWS_OK_MIN
    out["predictions"].append({"id": "P7", "model": m, "set": "original",
                               "claim": "at least %d of 96 re-extracted rows within %.2f relative error of the stored array" % (T_ROWS_OK_MIN, T_ROW_REL_ERR),
                               "got": "%d of %d (median %.4f, max %.4f)" % (prov[m]["rows_within_tolerance"], prov[m]["of"], prov[m]["median_relative_error"], prov[m]["max_relative_error"]),
                               "verdict": verdict(prov[m]["ok"])})
out["provenance"] = prov


def void(*models):
    """a line is VOID if P7 failed for a model it uses; NOT RUN if a model was not extracted"""
    for m in models:
        if m in prov and prov[m] is not None and not prov[m]["ok"]:
            return "VOID"
    return None


def add(pid, setname, models, claim, got, ok, testable=True):
    v = void(*models)
    if any(desc[setname][m] is None for m in models):
        v = "NOT RUN"
    elif v is None:
        v = "NOT TESTABLE" if not testable else verdict(ok)
    out["predictions"].append({"id": pid, "set": setname, "model": " x ".join(models), "claim": claim, "got": got, "verdict": v})


for setname in ("fresh", "original"):
    dA, dB, dC = desc[setname][A], desc[setname][B], desc[setname][C]
    words = (dA or dB or dC or {}).get("words")
    cats = (dA or dB or dC or {}).get("cats")
    rec = {}
    for m, d in ((A, dA), (B, dB), (C, dC)):
        rec[m] = None if d is None else {k: v for k, v in d.items() if not k.startswith("_") and k not in ("words", "cats")}
    # P1
    if dA is not None:
        r = dA["norm_ratio_median_alone"]
        add("P1a", setname, [A], "bare-template norm at least %gx the other templates, median over words alone at position 0" % T_NORM_RATIO_MIN,
            "no word alone" if r is None else "%.1fx over %d words" % (r, dA["n_alone_at_position_0"]), r is not None and r >= T_NORM_RATIO_MIN,
            testable=r is not None)
    else:
        add("P1a", setname, [A], "norm ratio", "", False)
    if dB is not None:
        r = dB["norm_ratio_median_single"]
        add("P1b", setname, [B], "control with a start token: the same ratio at most %gx for single-token words" % T_NORM_RATIO_MAX_CONTROL,
            "%.2fx over %d words; start token added: %s" % (r, dB["n_single_token_words"], dB["adds_start_token"]),
            r <= T_NORM_RATIO_MAX_CONTROL and dB["adds_start_token"])
    else:
        add("P1b", setname, [B], "control norm ratio", "", False)
    # P2, P3
    if dA is not None:
        n1 = dA["n_alone_at_position_0"]
        testable = n1 >= MIN_CLASS and dA["n_words"] - n1 >= MIN_CLASS
        a8 = dA["auc_pc1_vs_alone_8"]
        add("P2", setname, [A], "8 templates: top direction holds at least %.2f of variance and separates alone words at AUC %.2f or more" % (T_SHARE8_MIN, T_AUC_MIN),
            "share %.4f, AUC %s, alone %d of %d" % (dA["pc1_share_8_templates"], "n/a" if a8 is None else "%.4f" % a8, n1, dA["n_words"]),
            dA["pc1_share_8_templates"] >= T_SHARE8_MIN and a8 is not None and a8 >= T_AUC_MIN, testable=testable)
        add("P3", setname, [A], "bare template dropped (7 templates): top direction holds at most %.2f" % T_SHARE7_MAX,
            "share %.4f" % dA["pc1_share_7_templates"], dA["pc1_share_7_templates"] <= T_SHARE7_MAX)
    else:
        add("P2", setname, [A], "dominant direction", "", False)
        add("P3", setname, [A], "intervention", "", False)
    # P4, P5, P8 and descriptive extras for the pair A x B
    if dA is not None and dB is not None:
        r8, r7 = rsa(dA["_rep8"], dB["_rep8"]), rsa(dA["_rep7"], dB["_rep7"])
        add("P4", setname, [A, B], "cross-family RSA rises by at least %.2f when the bare template is dropped, and reaches %.2f" % (T_GAIN_MIN, T_R7_MIN),
            "8 templates %.4f, 7 templates %.4f, gain %+.4f" % (r8, r7, r7 - r8), (r7 - r8) >= T_GAIN_MIN and r7 >= T_R7_MIN)
        iu = np.triu_indices(len(words), k=1)
        Z = kit.lex_design(words, toks[setname]["counts"], iu)
        pl8 = kit.partial_corr(up(dA["_rep8"]), up(dB["_rep8"]), Z)
        pl7 = kit.partial_corr(up(dA["_rep7"]), up(dB["_rep7"]), Z)
        n8, n8sd = within_cat_null(dA["_rep8"], dB["_rep8"], cats)
        n7, n7sd = within_cat_null(dA["_rep7"], dB["_rep7"], cats)
        rec["pair_A_x_B"] = {"raw_8": r8, "raw_7": r7, "partial_lex_8": pl8, "partial_lex_7": pl7,
                             "spearman_8": spearman(dA["_rep8"], dB["_rep8"]), "spearman_7": spearman(dA["_rep7"], dB["_rep7"]),
                             "within_category_null_8": [n8, n8sd], "within_category_null_7": [n7, n7sd],
                             "excess_over_within_category_null_8": r8 - n8, "excess_over_within_category_null_7": r7 - n7}
        if setname == "fresh":
            cj = [p for p in confirm["pairs"] if {p["a"], p["b"]} == {A, B}]
            claimed = cj[0]["partial_lex"] if cj else None
            add("P8", setname, [A, B], "re-derived partial-lexical value within %.2f of the June result file for this pair" % T_CONFIRM_PAIR_TOL,
                "June %s, re-derived %.4f" % (claimed, pl8), claimed is not None and abs(pl8 - claimed) <= T_CONFIRM_PAIR_TOL,
                testable=claimed is not None)
    else:
        add("P4", setname, [A, B], "cross-family consequence", "", False)
        if setname == "fresh":
            add("P8", setname, [A, B], "June pair value", "", False)
    if dB is not None:
        add("P5", setname, [B], "control: dropping the bare template leaves this model's own geometry at RSA %.2f or more" % T_SELF_RSA_MIN,
            "%.4f" % dB["self_rsa_8_vs_7"], dB["self_rsa_8_vs_7"] >= T_SELF_RSA_MIN)
    else:
        add("P5", setname, [B], "control self RSA", "", False)
    # P6: the model the lab never stored
    if dC is not None:
        r = dC["norm_ratio_median_alone"]
        n1 = dC["n_alone_at_position_0"]
        testable = n1 >= MIN_CLASS and dC["n_words"] - n1 >= MIN_CLASS
        a8 = dC["auc_pc1_vs_alone_8"]
        ok = (r is not None and r >= T_NORM_RATIO_MIN and dC["pc1_share_8_templates"] >= T_SHARE8_MIN
              and a8 is not None and a8 >= T_AUC_MIN and dC["pc1_share_7_templates"] <= T_SHARE7_MAX)
        add("P6", setname, [C], "secondary, a model with no stored arrays: ratio, share and AUC as P1a and P2, and P3 after the drop",
            "ratio %s, share8 %.4f, AUC %s, share7 %.4f, alone %d of %d" % ("n/a" if r is None else "%.1fx" % r, dC["pc1_share_8_templates"],
                                                                         "n/a" if a8 is None else "%.4f" % a8, dC["pc1_share_7_templates"], n1, dC["n_words"]),
            ok, testable=testable)
        if dB is not None:
            rec["pair_C_x_B"] = {"raw_8": rsa(dC["_rep8"], dB["_rep8"]), "raw_7": rsa(dC["_rep7"], dB["_rep7"])}
        if dA is not None:
            rec["pair_A_x_C_same_family"] = {"raw_8": rsa(dA["_rep8"], dC["_rep8"]), "raw_7": rsa(dA["_rep7"], dC["_rep7"])}
    else:
        add("P6", setname, [C], "secondary model", "", False)
    out["sets"][setname] = rec

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
order = {"fresh": 0, "original": 1}
for p in sorted(out["predictions"], key=lambda p: (order[p["set"]], p["id"], p["model"])):
    print("%-12s %-4s %-8s %-27s %s" % (p["verdict"], p["id"], p["set"], p["model"], p["got"]))
print("wrote", OUT)
