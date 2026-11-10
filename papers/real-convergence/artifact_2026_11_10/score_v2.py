"""score_v2.py: score the sealed predictions of PREREG_corrected_geometry_five_models_2026_11_10.md.

Lead-written, 2026-11-10, rewritten after a pre-seal review by two independent reviewers, and sealed by hash with
the prereg and extract_v2.py before any fresh-word state of the three larger models exists. Reads the states and
logs written by extract_v2.py, the first test's states (as a control), and the kit copy of the lab's committed
June files. numpy only. The thresholds below are not to be edited after the seal.

Order of work, fixed:
  1. every states file is checked against the kit's word order and the token-count file;
  2. the controls are scored. if ANY control fails, only control lines are printed and written, nothing else is
     computed, and the exit code is 3;
  3. only then are the prediction lines scored.
It is to be run ONCE, after every model has either its states or a refusal log.

Usage: py -3.12 -X utf8 -I -B score_v2.py <kit_dir> <v1_states_dir> <v2_states_dir> <out.json>
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

KIT, V1DIR, V2DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, KIT)
import reproduce as kit  # distmat, partial_corr, lex_design, flatten: the kit copy of the lab's method

# ---- sealed thresholds ----
T_PATH_REL = 1e-4            # C1: every state within this relative error of the first test's saved state
T_C2_ROW_REL = 0.01          # C2a: rep8 rows that were NOT alone at position 0, against the stored June row
T_C2_ALONE_REL = 0.05        # C2b: rows that were alone: |new rep8 - June rep8| against |(7/8) new rep7|
T_C2_MAX_BAD_ROWS = 2        # C2a and C2b: at most this many rows may miss
T_PAD = 1e-4                 # C3: the extractor's own pad check, re-read from its log
T_NORM_RATIO_MIN = 10.0      # A1
T_SHARE8_MIN = 0.85          # A2
T_AUC_MIN = 0.99             # A2
T_SHARE7_MAX = 0.30          # A3
T_NORM_RATIO_MAX_CONTROL = 3.0   # A4
T_SELF_RSA_MIN = 0.95        # A4
T_QL_PAIR_MIN = 0.85         # G1: each Qwen x Llama pair, bare template dropped
T_PHI_PAIR_MIN = 0.70        # G2: each pair with Phi, bare template dropped
T_MEAN8_MIN = 0.80           # G3: mean over the 8 cross-family pairs
T_EXCESS_MIN = 0.15          # G4: RSA minus the mean of the within-category shuffle null, on the r scale
T_JUNE_TOL = 0.02            # G5: re-derived June pair values
T_LOO_TOL = 0.05             # G6: dropping any one of the 7 kept templates moves a pair's RSA by at most this
T_PHI_GAP = 0.10             # the fixed sentence about Phi is printed if its pairs average this much below Qwen x Llama
MIN_CLASS = 10
NULL_DRAWS, NULL_SEED = 200, 0

FAMILY = {"Qwen2.5-1.5B": "qwen", "Qwen2.5-3B": "qwen", "Llama-3.2-1B": "llama", "Llama-3.2-3B": "llama",
          "Phi-3.5-mini": "phi", "Qwen2.5-0.5B": "qwen"}
ARCH = {"qwen": "Qwen2", "llama": "Llama", "phi": "Phi3"}
FIVE = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Llama-3.2-1B", "Llama-3.2-3B", "Phi-3.5-mini"]   # of the June six
CONTROL_PATH = ["Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B"]   # also extracted by the first test
NEW = ["Qwen2.5-3B", "Llama-3.2-3B", "Phi-3.5-mini"]            # never extracted per template before
NO_START = ["Qwen2.5-3B", "Phi-3.5-mini"]
WITH_START = ["Llama-3.2-3B"]
XF8 = [(a, b) for a, b in itertools.combinations(FIVE, 2) if FAMILY[a] != FAMILY[b]]
JUNE10 = list(itertools.combinations(FIVE, 2))
assert len(XF8) == 8 and len(JUNE10) == 10

toks = json.load(open(os.path.join(KIT, "lex_tokens.json"), encoding="utf-8"))
stored = np.load(os.path.join(KIT, "contextual_reps.npz"))
confirm = json.load(open(os.path.join(KIT, "real_convergence_confirm_result.json"), encoding="utf-8"))
SETS = {"original": kit.flatten(kit.ORIGINAL), "fresh": kit.flatten(kit.FRESH)}


def load(d, pattern, setname, model):
    p = os.path.join(d, pattern % (setname, model))
    if not os.path.exists(p):
        return None
    z = np.load(p)
    words = [str(w) for w in z["words"]]
    cats = [str(c) for c in z["cats"]]
    kw, kc = SETS[setname]
    if words != list(kw) or cats != [str(c) for c in kc] or words != list(toks[setname]["words"]):
        raise SystemExit("REFUSED: %s has a word or category order that differs from the kit or the token-count file" % os.path.basename(p))
    if z["states"].shape[:2] != (96, 8) or z["seq_len"].shape != (96, 8):
        raise SystemExit("REFUSED: %s has an unexpected shape" % os.path.basename(p))
    return {"states": z["states"].astype(np.float64), "seq_len": z["seq_len"], "words": words, "cats": np.array(cats)}


def logs_for(model):
    out = []
    for p in sorted(glob.glob(os.path.join(V2DIR, "extract2_log__%s__run*.json" % model))):
        try:
            out.append(json.load(open(p, encoding="utf-8")))
        except Exception:
            out.append({"status": "UNREADABLE LOG"})
    return out


def presence(model):
    """OK: both sets present with an OK log. REFUSED: no states, and a log says memory. MISSING: anything else."""
    have = [os.path.exists(os.path.join(V2DIR, "states2__%s__%s.npz" % (s, model))) for s in ("original", "fresh")]
    lg = logs_for(model)
    if all(have):
        return "OK"
    if not any(have) and lg and all(str(l.get("status", "")).startswith("REFUSED: not enough free memory") for l in lg):
        return "REFUSED"
    return "MISSING"


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


def up(x, keep=None):
    if keep is not None:
        x = x[keep]
    return kit.distmat(x)[np.triu_indices(x.shape[0], k=1)]


def rsa(x, y, keep=None):
    return float(np.corrcoef(up(x, keep), up(y, keep))[0, 1])


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


def same_category_rsa(x, y, cats):
    """Pearson r over only the word pairs that share a category (8 x 66 = 528 entries)"""
    iu = np.triu_indices(len(cats), k=1)
    same = (cats[:, None] == cats[None, :])[iu]
    return float(np.corrcoef(kit.distmat(x)[iu][same], kit.distmat(y)[iu][same])[0, 1]), int(same.sum())


def describe(d):
    st, sl = d["states"], d["seq_len"]
    norms = np.linalg.norm(st, axis=2)
    bare = sl[:, 0]
    alone = (bare == 1).astype(int)
    single = (bare == bare.min()).astype(int)
    ratio = norms[:, 0] / np.median(norms[:, 1:], axis=1)
    rep8, rep7 = st.mean(1), st[:, 1:].mean(1)
    s8, p8 = pc1(rep8)
    s7, _ = pc1(rep7)
    return {"n_words": int(st.shape[0]), "n_alone": int(alone.sum()), "n_single": int(single.sum()),
            "adds_start_token": bool(bare.min() >= 2),
            "ratio_alone": None if alone.sum() == 0 else float(np.median(ratio[alone == 1])),
            "ratio_single": float(np.median(ratio[single == 1])),
            "share8": s8, "auc8": auc(p8, alone), "share7": s7, "self_rsa_8_vs_7": rsa(rep8, rep7),
            "rep8": rep8, "rep7": rep7, "single": single.astype(bool), "alone": alone.astype(bool), "states": st}


out = {"prereg": "PREREG_corrected_geometry_five_models_2026_11_10.md", "controls": [], "lines": [], "tables": {}}
PRES = {m: presence(m) for m in FAMILY}
out["presence"] = PRES
D = {s: {m: (load(V2DIR, "states2__%s__%s.npz", s, m) if PRES[m] == "OK" else None) for m in FAMILY} for s in ("original", "fresh")}
failed = []          # control failures; any entry stops the scoring


def control(cid, setname, model, claim, got, ok):
    out["controls"].append({"id": cid, "set": setname, "model": model, "claim": claim, "got": got, "verdict": "PASS" if ok else "FAIL"})
    if not ok:
        failed.append((cid, model))


# ---- C3: the run log of every model that has states ----
for m in FAMILY:
    if PRES[m] != "OK":
        continue
    oks = [l for l in logs_for(m) if l.get("status") == "OK"]
    worst = min((l.get("pad_check_worst_relative_difference", 9.0) for l in oks), default=None)
    ok = len(oks) == 1 and worst is not None and worst <= T_PAD
    control("C3", "original", m, "exactly one OK run log, with the pad check at most %.0e" % T_PAD,
            "%d OK log(s); pad check worst %s over %s rows" % (len(oks), "n/a" if worst is None else "%.2e" % worst,
                                                              oks[0].get("pad_check_rows") if oks else "n/a"), ok)

# ---- C1: the new code path against the first test's saved states ----
for m in CONTROL_PATH:
    for s in ("original", "fresh"):
        a = D[s][m]
        if a is None:
            control("C1", s, m, "new code path equals the first test", "no states from this run (%s)" % PRES[m], False)
            continue
        b = load(V1DIR, "states__%s__%s.npz", s, m)
        if b is None:
            control("C1", s, m, "new code path equals the first test", "the first test's file was not found", False)
            continue
        rel = np.linalg.norm(a["states"] - b["states"], axis=2) / (np.linalg.norm(b["states"], axis=2) + 1e-12)
        same_len = bool((a["seq_len"] == b["seq_len"]).all())
        control("C1", s, m, "all 768 states within %.0e of the first test, token lengths equal" % T_PATH_REL,
                "%d of %d states within (max relative error %.2e), lengths equal %s" % (int((rel <= T_PATH_REL).sum()), rel.size, rel.max(), same_len),
                bool((rel <= T_PATH_REL).all()) and same_len)

# ---- C2: the three larger models against the stored June arrays (original set) ----
for m in NEW:
    d0 = D["original"][m]
    if d0 is None:
        if PRES[m] == "REFUSED":
            out["controls"].append({"id": "C2", "set": "original", "model": m, "claim": "re-extraction reproduces the stored June array", "got": "refused for memory", "verdict": "NOT RUN"})
        else:
            control("C2", "original", m, "re-extraction reproduces the stored June array", "states missing without a memory refusal", False)
        continue
    st = d0["states"]
    alone = d0["seq_len"][:, 0] == 1
    rep8, rep7 = st.mean(1), st[:, 1:].mean(1)
    old = stored["fixed__" + m].astype(np.float64)
    rel = np.linalg.norm(rep8 - old, axis=1) / (np.linalg.norm(old, axis=1) + 1e-12)
    bad_a = int((rel[~alone] > T_C2_ROW_REL).sum())
    control("C2a", "original", m, "rows not alone at position 0: rep8 within %.2f of the stored June row, at most %d rows missing" % (T_C2_ROW_REL, T_C2_MAX_BAD_ROWS),
            "%d of %d rows within (median %.5f, max %.5f)" % (int((~alone).sum()) - bad_a, int((~alone).sum()), float(np.median(rel[~alone])), float(rel[~alone].max())),
            bad_a <= T_C2_MAX_BAD_ROWS)
    if alone.any():
        scaled = np.linalg.norm(rep8 - old, axis=1) / (np.linalg.norm(0.875 * rep7, axis=1) + 1e-12)
        bad_b = int((scaled[alone] > T_C2_ALONE_REL).sum())
        control("C2b", "original", m, "rows alone at position 0: the rep8 difference from June is at most %.2f of that row's seven-template part, at most %d rows missing" % (T_C2_ALONE_REL, T_C2_MAX_BAD_ROWS),
                "%d of %d rows within (median %.5f, max %.5f)" % (int(alone.sum()) - bad_b, int(alone.sum()), float(np.median(scaled[alone])), float(scaled[alone].max())),
                bad_b <= T_C2_MAX_BAD_ROWS)

order = {"fresh": 0, "original": 1}


def show(items):
    for p in items:
        print("%-9s %-4s %-8s %-30s %s" % (p["verdict"], p["id"], p["set"], p["model"], p["got"]))


if failed:
    out["stopped"] = "a control failed; nothing else was computed. a rerun needs an amendment sealed first, naming the change and justified by control values only."
    out["failed_controls"] = ["%s %s" % f for f in failed]
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    show(out["controls"])
    print("STOPPED: %d control line(s) failed. no prediction line was computed." % len(failed))
    print("wrote", OUT)
    sys.exit(3)

# ================= every control passed: the prediction lines =================
DESC = {s: {m: (None if D[s][m] is None else describe(D[s][m])) for m in FAMILY} for s in D}


def status_of(models):
    if any(PRES[m] == "MISSING" for m in models):
        return "MISSING"
    if any(PRES[m] == "REFUSED" for m in models):
        return "NOT RUN"
    return None


def line(pid, setname, models, claim, got, ok, testable=True):
    v = status_of(models)
    if v is None:
        v = "NOT TESTABLE" if not testable else ("PASS" if ok else "FAIL")
    else:
        got = ""
    out["lines"].append({"id": pid, "set": setname, "model": " x ".join(models), "claim": claim, "got": got, "verdict": v})


for s in ("fresh", "original"):
    words, cats = list(SETS[s][0]), np.array([str(c) for c in SETS[s][1]])
    iu = np.triu_indices(len(words), k=1)
    Z = kit.lex_design(words, toks[s]["counts"], iu)
    # ---- A1 to A3: the artifact in the two larger models without a start token ----
    for m in NO_START:
        d = DESC[s][m]
        if d is None:
            for pid in ("A1", "A2", "A3"):
                line(pid, s, [m], "artifact", "", False)
            continue
        testable = d["n_alone"] >= MIN_CLASS and d["n_words"] - d["n_alone"] >= MIN_CLASS
        r = d["ratio_alone"]
        line("A1", s, [m], "bare-template norm at least %gx the other templates, median over words alone at position 0" % T_NORM_RATIO_MIN,
             "no word alone" if r is None else "%.1fx over %d words" % (r, d["n_alone"]), r is not None and r >= T_NORM_RATIO_MIN, testable=r is not None)
        line("A2", s, [m], "8 templates: top direction share at least %.2f and AUC against alone at least %.2f" % (T_SHARE8_MIN, T_AUC_MIN),
             "share %.4f, AUC %s, alone %d of %d" % (d["share8"], "n/a" if d["auc8"] is None else "%.4f" % d["auc8"], d["n_alone"], d["n_words"]),
             d["share8"] >= T_SHARE8_MIN and d["auc8"] is not None and d["auc8"] >= T_AUC_MIN, testable=testable)
        line("A3", s, [m], "bare template dropped: top direction share at most %.2f" % T_SHARE7_MAX, "share %.4f" % d["share7"], d["share7"] <= T_SHARE7_MAX)
    # ---- A4: the control with a start token ----
    for m in WITH_START:
        d = DESC[s][m]
        if d is None:
            line("A4", s, [m], "control", "", False)
            continue
        line("A4", s, [m], "start token added, norm ratio at most %gx, own geometry 8 against 7 templates at least %.2f" % (T_NORM_RATIO_MAX_CONTROL, T_SELF_RSA_MIN),
             "start token %s, ratio %.2fx, self RSA %.4f" % (d["adds_start_token"], d["ratio_single"], d["self_rsa_8_vs_7"]),
             d["adds_start_token"] and d["ratio_single"] <= T_NORM_RATIO_MAX_CONTROL and d["self_rsa_8_vs_7"] >= T_SELF_RSA_MIN)
    # ---- the pair table ----
    rows = {}
    for a, b in itertools.combinations(list(FAMILY), 2):
        da, db = DESC[s][a], DESC[s][b]
        if da is None or db is None:
            continue
        xf = FAMILY[a] != FAMILY[b]
        both_single = da["single"] & db["single"]
        row = {"a": a, "b": b, "cross_family": xf, "in_june_six": a in FIVE and b in FIVE,
               "raw_8": rsa(da["rep8"], db["rep8"]), "raw_7": rsa(da["rep7"], db["rep7"]),
               "partial_lex_8": kit.partial_corr(up(da["rep8"]), up(db["rep8"]), Z),
               "partial_lex_7": kit.partial_corr(up(da["rep7"]), up(db["rep7"]), Z),
               "spearman_7": spearman(da["rep7"], db["rep7"]),
               "n_words_single_token_in_both": int(both_single.sum())}
        sc, n_same = same_category_rsa(da["rep7"], db["rep7"], cats)
        row["same_category_pairs_rsa_7"], row["same_category_pairs_n"] = sc, n_same
        if xf:
            n7, n7sd = within_cat_null(da["rep7"], db["rep7"], cats)
            row.update({"within_category_null_7": n7, "within_category_null_7_sd": n7sd, "excess_7": row["raw_7"] - n7,
                        "excess_7_in_null_sd": (row["raw_7"] - n7) / (n7sd + 1e-12)})
            if both_single.sum() >= 24:
                row["raw_7_single_token_words_only"] = rsa(da["rep7"], db["rep7"], keep=both_single)
                ns, _ = within_cat_null(da["rep7"][both_single], db["rep7"][both_single], cats[both_single])
                row["excess_7_single_token_words_only"] = row["raw_7_single_token_words_only"] - ns
            # leave one of the seven kept templates out, in both models
            loo = []
            for j in range(1, 8):
                keep = [k for k in range(1, 8) if k != j]
                loo.append(rsa(da["states"][:, keep].mean(1), db["states"][:, keep].mean(1)))
            row["leave_one_template_out_7"] = loo
            row["leave_one_template_out_max_move"] = float(max(abs(v - row["raw_7"]) for v in loo))
        rows[(a, b)] = row
    out["tables"][s] = list(rows.values())

    def get(a, b):
        return rows.get((a, b)) or rows.get((b, a))

    for a, b in XF8:
        r = get(a, b)
        phi = "phi" in (FAMILY[a], FAMILY[b])
        pid, bar = ("G2", T_PHI_PAIR_MIN) if phi else ("G1", T_QL_PAIR_MIN)
        label = "pair with Phi" if phi else "Qwen with Llama"
        line(pid, s, [a, b], "%s, bare template dropped: RSA at least %.2f" % (label, bar),
             "" if r is None else "%.4f (8 templates %.4f)" % (r["raw_7"], r["raw_8"]), r is not None and r["raw_7"] >= bar)
        line("G4", s, [a, b], "RSA exceeds the within-category shuffle null by at least %.2f on the r scale" % T_EXCESS_MIN,
             "" if r is None else "%.4f over a null of %.4f (%.0f null sd); same-category pairs only: r %.4f" % (r["excess_7"], r["within_category_null_7"], r["excess_7_in_null_sd"], r["same_category_pairs_rsa_7"]),
             r is not None and r["excess_7"] >= T_EXCESS_MIN)
        line("G6", s, [a, b], "dropping any one of the seven kept templates moves the RSA by at most %.2f" % T_LOO_TOL,
             "" if r is None else "largest move %.4f" % r["leave_one_template_out_max_move"], r is not None and r["leave_one_template_out_max_move"] <= T_LOO_TOL)
    # ---- G3 ----
    st5 = status_of(FIVE)
    have = [get(a, b) for a, b in XF8]
    if st5 is None and all(h is not None for h in have):
        v7 = [h["raw_7"] for h in have]
        ql = [get(a, b)["raw_7"] for a, b in XF8 if "phi" not in (FAMILY[a], FAMILY[b])]
        ph = [get(a, b)["raw_7"] for a, b in XF8 if "phi" in (FAMILY[a], FAMILY[b])]
        gap = float(np.mean(ql) - np.mean(ph))
        out["lines"].append({"id": "G3", "set": s, "model": "8 cross-family pairs among five models",
                             "claim": "mean RSA with the bare template dropped at least %.2f" % T_MEAN8_MIN,
                             "got": "mean %.4f, median %.4f, min %.4f (8 templates: mean %.4f)" % (float(np.mean(v7)), float(np.median(v7)), min(v7), float(np.mean([h["raw_8"] for h in have]))),
                             "verdict": "PASS" if float(np.mean(v7)) >= T_MEAN8_MIN else "FAIL"})
        out.setdefault("phi_gap", {})[s] = {"qwen_llama_mean": float(np.mean(ql)), "phi_pairs_mean": float(np.mean(ph)), "gap": gap,
                                           "sentence_required": gap > T_PHI_GAP,
                                           "sentence": "Phi remains lower with the bare template dropped; cause not established."}
    else:
        out["lines"].append({"id": "G3", "set": s, "model": "8 cross-family pairs among five models",
                             "claim": "mean RSA with the bare template dropped at least %.2f" % T_MEAN8_MIN, "got": "",
                             "verdict": "INVALID" if st5 == "MISSING" else "NOT RUN"})
    # ---- G5: the June result file, re-derived under the published pipeline (fresh set only) ----
    if s == "fresh":
        for a, b in JUNE10:
            r = get(a, b)
            cj = [p for p in confirm["pairs"] if {p["a"], p["b"]} == {a, b}]
            claimed = cj[0]["partial_lex"] if cj else None
            line("G5", s, [a, b], "June partial-lexical value re-derived within %.2f under the published 8-template pipeline" % T_JUNE_TOL,
                 "" if r is None else "June %s, re-derived %.4f" % (claimed, r["partial_lex_8"]),
                 r is not None and claimed is not None and abs(r["partial_lex_8"] - claimed) <= T_JUNE_TOL, testable=claimed is not None)

out["per_model"] = {s: {m: (None if DESC[s][m] is None else {k: v for k, v in DESC[s][m].items() if k not in ("rep8", "rep7", "single", "alone", "states")}) for m in FAMILY} for s in DESC}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
show(out["controls"])
show(sorted(out["lines"], key=lambda p: (order[p["set"]], p["id"], p["model"])))
for s in ("fresh", "original"):
    g = (out.get("phi_gap") or {}).get(s)
    if g:
        print("PHI GAP %-8s Qwen x Llama mean %.4f, Phi pairs mean %.4f, gap %.4f -> sentence required: %s" % (s, g["qwen_llama_mean"], g["phi_pairs_mean"], g["gap"], g["sentence_required"]))
counts = {}
for p in out["lines"]:
    counts[(p["set"], p["verdict"])] = counts.get((p["set"], p["verdict"]), 0) + 1
print("lines:", ", ".join("%s %s %d" % (k[0], k[1], v) for k, v in sorted(counts.items())))
print("wrote", OUT)
