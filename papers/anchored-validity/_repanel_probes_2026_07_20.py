"""Inline re-panel probe battery -- anchored-validity Stage A @ fc9528f. PANEL ATTACKS, NOT SCORED RUNS.

Executes the three-lens probe program (exchangeability / verdict-mapping / strawman-fairness)
against the harness's own functions. No frozen bars here: these are adversarial probes whose
outputs feed the re-panel receipt verbatim, favourable or not. Fresh seeds throughout (never
seed 0, the scored stream). Writes _repanel_probe_receipts_2026_07_20.json. ASCII only.
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchored_stage_a as A

ALPHAS = [0.15, 0.20, 0.10, 0.18]
BETAS = [0.85, 0.80, 0.90, 0.78]
PI, N, K = 0.35, 6000, 400
OUT = {"subject": "anchored_stage_a.py @ fc9528f", "probes": {}}


def base_panel(rng, n=N, pi=PI, alphas=None, betas=None, **kw):
    return A.simulate_panel(rng, n, pi, alphas or ALPHAS, betas or BETAS, **kw)


def base_anchors(rng, k=K, **kw):
    neg = A.make_anchors(rng, k, "neg", ALPHAS, BETAS, **kw)
    pos = A.make_anchors(rng, k, "pos", ALPHAS, BETAS, **kw)
    return neg, pos


def sync_est(V, neg, pos, seed, n_boot=300):
    return A.anchored_sync(V, neg, pos, np.random.default_rng(seed), n_boot=n_boot)


def rec(r, extra=None):
    d = {"verdict": r["verdict"], "pi": r.get("pi"), "s": r.get("s"),
         "pi_unclipped": r.get("pi_unclipped"),
         "misfit": r.get("lack_of_fit", {}).get("chi2_per_df"),
         "err": None if r.get("pi") is None else abs(r["pi"] - PI)}
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------- clean reference band (5 seeds)
refs = []
for sd in (201, 202, 203, 204, 205):
    rng = np.random.default_rng(sd)
    y, V = base_panel(rng)
    neg, pos = base_anchors(rng)
    refs.append(sync_est(V, neg, pos, 900 + sd))
OUT["probes"]["clean_reference_band"] = {
    "misfits": [round(r["lack_of_fit"]["chi2_per_df"], 3) for r in refs],
    "max_misfit": max(r["lack_of_fit"]["chi2_per_df"] for r in refs),
    "pi_errs": [round(abs(r["pi"] - PI), 4) for r in refs],
    "s_hats": [r["s"] for r in refs]}
CLEAN_MAX = OUT["probes"]["clean_reference_band"]["max_misfit"]

# ============================ LENS A: exchangeability ============================
# A1: does s absorb BETA non-exchangeability? Both directions.
a1 = {}
for tag, shift in (("organic_betas_minus_010", -0.10), ("organic_betas_plus_010", +0.10)):
    rng = np.random.default_rng(211 if shift < 0 else 212)
    y, V = base_panel(rng, betas=[b + shift for b in BETAS])
    neg, pos = base_anchors(rng)                      # anchors at base rates
    a1[tag] = rec(sync_est(V, neg, pos, 911 if shift < 0 else 912))
# alpha-optimism direction (organic alphas above anchors)
rng = np.random.default_rng(213)
y, V = base_panel(rng, alphas=[a + 0.05 for a in ALPHAS])
neg, pos = base_anchors(rng)
a1["organic_alphas_plus_005"] = rec(sync_est(V, neg, pos, 913))
# specialist through the sync arm
ob = list(BETAS); ob[0] = ALPHAS[0]
rng = np.random.default_rng(214)
y, V = base_panel(rng, betas=ob)
neg, pos = base_anchors(rng)
a1["one_of_J_specialist"] = rec(sync_est(V, neg, pos, 914))
OUT["probes"]["A1_channel_absorption"] = a1

# A2: refusal false-fires at legitimate extreme prevalence (two-sided duty)
a2 = {}
for pv in (0.02, 0.98):
    rng = np.random.default_rng(int(pv * 1000) + 220)
    y, V = base_panel(rng, pi=pv)
    neg, pos = base_anchors(rng)
    r1 = A.anchored(V, neg, pos, np.random.default_rng(921))
    r2 = sync_est(V, neg, pos, 922)
    a2[f"pi_true_{pv}"] = {"anchored": {"verdict": r1["verdict"], "pi": r1.get("pi"),
                                        "err": None if r1.get("pi") is None else abs(r1["pi"] - pv)},
                           "sync": {"verdict": r2["verdict"], "pi": r2.get("pi"),
                                    "err": None if r2.get("pi") is None else abs(r2["pi"] - pv)}}
OUT["probes"]["A2_refusal_false_fire"] = a2

# A3: refusal evasion window -- partial pooling of detector garbage into neg
a3 = []
for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    rng = np.random.default_rng(230)
    y, V = base_panel(rng, sync_frac=0.02)
    neg, pos = base_anchors(rng)
    det = A.make_anchors(np.random.default_rng(231), K, "neg", ALPHAS, BETAS, sync_frac=0.80)
    k_in = int(frac * K)
    pooled = np.vstack([neg, det[:k_in]])
    r = sync_est(V, pooled, pos, 930 + int(frac * 100))
    a3.append({"pool_frac": frac, **rec(r)})
OUT["probes"]["A3_refusal_evasion_sweep"] = a3

# A4: true s above the grid edge
rng = np.random.default_rng(240)
y, V = base_panel(rng, sync_frac=0.70)
neg, pos = base_anchors(rng)
r = sync_est(V, neg, pos, 941)
OUT["probes"]["A4_s_beyond_grid"] = rec(r, {"s_at_grid_edge": r.get("s_at_grid_edge"),
                                            "true_s": 0.70})

# A5: bootstrap coverage spot-check (30 reps, n=3000 for speed -- disclosed)
cov_pi = cov_s = 0
REPS = 30
for i in range(REPS):
    rng = np.random.default_rng(3000 + i)
    y, V = base_panel(rng, n=3000, sync_frac=0.10)
    neg, pos = base_anchors(rng)
    r = sync_est(V, neg, pos, 4000 + i, n_boot=200)
    if r["ci"][0] <= PI <= r["ci"][1]:
        cov_pi += 1
    if r["s_ci"][0] <= 0.10 <= r["s_ci"][1]:
        cov_s += 1
OUT["probes"]["A5_coverage_spot"] = {"reps": REPS, "n": 3000, "n_boot": 200,
                                     "pi_ci_coverage": cov_pi / REPS, "s_ci_coverage": cov_s / REPS}

# A6: power of the 3-SE alpha-transfer bar vs pi-material shifts
a_mid = 0.18
se = float(np.sqrt(a_mid * (1 - a_mid) / K))
min_inf = float(min(np.array(BETAS) - np.array(ALPHAS)))
det_limit = 3 * se
OUT["probes"]["A6_alpha_bar_power"] = {
    "three_se_detection_limit": round(det_limit, 4),
    "pi_bias_at_that_shift_first_order": round(det_limit / min_inf, 4),
    "shift_needed_for_003_pi_bias": round(0.03 * min_inf, 4),
    "conclusion": "a shift causing >0.03 pi bias can hide inside 3 SE at K=400"}

# ============================ LENS C: strawman-fairness ============================
# C1: hunt the R8d silent window -- misspecified keys the file did not test
c1 = []
for tag, kw, s_true in (
        ("subset01_003", {"sync_frac": 0.03, "sync_judges": (0, 1)}, 0.03),
        ("subset01_005", {"sync_frac": 0.05, "sync_judges": (0, 1)}, 0.05),
        ("subset01_008", {"sync_frac": 0.08, "sync_judges": (0, 1)}, 0.08),
        ("subset012_005", {"sync_frac": 0.05, "sync_judges": (0, 1, 2)}, 0.05),
        ("subset012_010", {"sync_frac": 0.10, "sync_judges": (0, 1, 2)}, 0.10),
        ("partial05_010", {"sync_frac": 0.10, "sync_strength": 0.5}, 0.10),
        ("partial07_005", {"sync_frac": 0.05, "sync_strength": 0.7}, 0.05)):
    rng = np.random.default_rng(hash(tag) % (2**31))
    y, V = base_panel(rng, **kw)
    neg, pos = base_anchors(rng)
    r = sync_est(V, neg, pos, (hash(tag) // 7) % (2**31))
    silent = (r["verdict"] == "ESTIMATED" and r["pi"] is not None
              and abs(r["pi"] - PI) > 0.03
              and r["lack_of_fit"]["chi2_per_df"] <= CLEAN_MAX)
    c1.append({"fixture": tag, "true_s": s_true, **rec(r), "SILENT_WRONG": bool(silent)})
OUT["probes"]["C1_silent_window_hunt"] = c1

# C2: y-correlated keys (content-borne exploits plausibly correlate with truth)
def panel_key_on(rng, key_rate, key_on):
    y = (rng.random(N) < PI).astype(int)
    J = len(ALPHAS)
    V = np.empty((N, J), int)
    for j in range(J):
        p = np.where(y == 1, BETAS[j], ALPHAS[j]).astype(float)
        V[:, j] = (rng.random(N) < p).astype(int)
    elig = (y == 0) if key_on == "neg" else (y == 1)
    key = elig & (rng.random(N) < key_rate)
    V[key] = 1
    return y, V

c2 = {}
for key_on, rate in (("neg", 0.15), ("pos", 0.15)):
    rng = np.random.default_rng(260 if key_on == "neg" else 261)
    y, V = panel_key_on(rng, rate, key_on)
    neg, pos = base_anchors(np.random.default_rng(262))
    r = sync_est(V, neg, pos, 963)
    eff_s = rate * ((1 - PI) if key_on == "neg" else PI)
    silent = (r["verdict"] == "ESTIMATED" and r["pi"] is not None
              and abs(r["pi"] - PI) > 0.03
              and r["lack_of_fit"]["chi2_per_df"] <= CLEAN_MAX)
    c2[f"key_on_{key_on}_rate015"] = {**rec(r), "effective_overall_rate": round(eff_s, 4),
                                      "SILENT_WRONG": bool(silent)}
OUT["probes"]["C2_y_correlated_keys"] = c2

# C3: trivial all-fire plug-in baseline vs the profile WLS
def trivial_baseline(V, neg, pos):
    a_hat, b_hat = neg.mean(0), pos.mean(0)
    keep = (b_hat - a_hat) >= A.INFORMATIVENESS_GATE
    idx = np.where(keep)[0]
    f0 = neg[:, idx].prod(1).mean(); f1 = pos[:, idx].prod(1).mean()
    af_obs = V[:, idx].prod(1).mean()
    rows, tgts, wts = A._moment_system(V, neg, pos, idx)
    pi = A._solve_pi_raw(rows, tgts, wts)
    s = 0.0
    for _ in range(4):
        af_pred = pi * f1 + (1 - pi) * f0
        s = max(0.0, (af_obs - af_pred) / max(1 - af_pred, 1e-9))
        av, bv = neg.mean(0), pos.mean(0)
        num = den = 0.0
        for j in idx:
            t_corr = (V[:, j].mean() - s) / max(1 - s, 1e-9) - av[j]
            d = bv[j] - av[j]
            num += d * t_corr; den += d * d
        pi = num / max(den, 1e-12)
    return float(pi), float(s)

c3 = []
for dose in (0.05, 0.08, 0.15):
    rng = np.random.default_rng(int(dose * 1000) + 270)
    y, V = base_panel(rng, sync_frac=dose)
    neg, pos = base_anchors(rng)
    r = sync_est(V, neg, pos, int(dose * 1000) + 971)
    tp, ts = trivial_baseline(V, neg, pos)
    c3.append({"dose": dose, "profile_pi_err": round(abs(r["pi"] - PI), 4),
               "profile_s": r["s"], "trivial_pi_err": round(abs(tp - PI), 4),
               "trivial_s": round(ts, 4)})
OUT["probes"]["C3_trivial_baseline"] = c3

# C4: tiny sync 0.01 -- can the knob move off zero for real but small doses?
rng = np.random.default_rng(280)
y, V = base_panel(rng, sync_frac=0.01)
neg, pos = base_anchors(rng)
OUT["probes"]["C4_tiny_sync"] = rec(sync_est(V, neg, pos, 981), {"true_s": 0.01})

dest = HERE / "_repanel_probe_receipts_2026_07_20.json"
dest.write_text(json.dumps(OUT, indent=1), encoding="utf-8")
print(json.dumps(OUT, indent=1))
print(f"\n-> {dest.name}")
