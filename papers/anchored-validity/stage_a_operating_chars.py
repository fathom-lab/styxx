"""R9: operating characteristics of the anchored instrument -- the datasheet run.

PREREG_R9_operating_characteristics_2026_07_20.md is the frozen contract; this file implements
it and nothing else. Consumes anchored_stage_a.py; touches no Stage-A check logic. Gates are
calibration-shaped (coverage bands, false-alarm bands, refusal rates); performance quantities
are CHARACTERISTICS -- measured, Wilson-intervaled, never gated. `--smoke` runs reduced R and
writes only a *_SMOKE_INVALID* file. ASCII only, CPU only, deterministic seed bases.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchored_stage_a as A

ALPHAS = [0.15, 0.20, 0.10, 0.18]
BETAS = [0.85, 0.80, 0.90, 0.78]
PI, N, K = 0.35, 6000, 400
S_NULL = A.S_NULL
NOMINAL_FA = 0.05

SEED_BASE = {"clean_cal": 10_000, "clean_val": 20_000, "rho30": 30_000, "sync05": 40_000,
             "sync15": 50_000, "sync02": 60_000, "deaf": 70_000, "contam10": 80_000,
             "keypos": 90_000, "betaplus": 100_000, "oneparam_rho30": 110_000}


def wilson(k, n):
    if n == 0:
        return [0.0, 1.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def q(xs, p):
    return float(np.percentile(np.asarray(xs, float), p)) if len(xs) else None


def sync_rep(seed, *, organic_alphas=None, organic_betas=None, rho=0.0, sync=0.0,
             viol=None, n_boot=200):
    """One replicate of the sync-corrected instrument. Anchors are always clean of sync and
    share rho (the exchangeable case); violations perturb exactly one thing each."""
    rng = np.random.default_rng(seed)
    oa = organic_alphas or ALPHAS
    ob = organic_betas or BETAS
    if viol == "keypos":                       # y-correlated key: fires on true positives only
        y = (rng.random(N) < PI).astype(int)
        V = np.empty((N, 4), int)
        for j in range(4):
            p = np.where(y == 1, ob[j], oa[j]).astype(float)
            V[:, j] = (rng.random(N) < p).astype(int)
        key = (y == 1) & (rng.random(N) < 0.15)
        V[key] = 1
    else:
        y, V = A.simulate_panel(rng, N, PI, oa, ob, rho, sync)
    neg = A.make_anchors(rng, K, "neg", ALPHAS, BETAS, rho, 0.0)
    pos = A.make_anchors(rng, K, "pos", ALPHAS, BETAS, rho, 0.0)
    if viol == "contam10":                     # 10 percent of negatives replaced by detector garbage
        det = A.make_anchors(rng, K, "neg", ALPHAS, BETAS, 0.0, 0.80)
        k_in = int(0.10 * K)
        neg = np.vstack([neg[:-k_in], det[:k_in]])
    r = A.anchored_sync(V, neg, pos, np.random.default_rng(seed + 7_000_000), n_boot=n_boot)
    return {"verdict": r["verdict"], "pi": r.get("pi"), "ci": r.get("ci"),
            "s": r.get("s"), "s_ci": r.get("s_ci"),
            "misfit": r["lack_of_fit"]["chi2_per_df"] if r.get("lack_of_fit") else None,
            "edge": bool(r.get("s_at_grid_edge", False))}


def oneparam_rep(seed, rho):
    rng = np.random.default_rng(seed)
    y, V = A.simulate_panel(rng, N, PI, ALPHAS, BETAS, rho, 0.0)
    neg = A.make_anchors(rng, K, "neg", ALPHAS, BETAS, rho, 0.0)
    pos = A.make_anchors(rng, K, "pos", ALPHAS, BETAS, rho, 0.0)
    r = A.anchored(V, neg, pos, np.random.default_rng(seed + 7_000_000))
    return {"verdict": r["verdict"], "pi": r.get("pi"), "ci": r.get("ci")}


def deaf_rep(seed):
    rng = np.random.default_rng(seed)
    al, be = [0.45] * 4, [0.52] * 4
    y, V = A.simulate_panel(rng, N, PI, al, be)
    neg = A.make_anchors(rng, K, "neg", al, be)
    pos = A.make_anchors(rng, K, "pos", al, be)
    r = A.anchored(V, neg, pos, np.random.default_rng(seed + 7_000_000), n_boot=50)
    a = np.asarray(r["alpha"]); b = np.asarray(r["beta"])
    margin_gate = A.INFORMATIVENESS_GATE + 3 * np.sqrt((a * (1 - a) + b * (1 - b)) / K)
    return {"void_plain": r["verdict"] == "VOID_PANEL__uninformative",
            "void_noise_margin": not bool(((b - a) >= margin_gate).any())}


def family(name, reps, fn):
    t0 = time.time()
    out = [fn(SEED_BASE[name] + i) for i in range(reps)]
    print(f"  {name}: R={reps} in {time.time() - t0:.0f}s")
    return out


def coverage(recs, truth=PI):
    est = [r for r in recs if r["verdict"] == "ESTIMATED"]
    k = sum(1 for r in est if r["ci"][0] <= truth <= r["ci"][1])
    return {"n_estimated": len(est), "covered": k,
            "rate": k / len(est) if est else None, "wilson": wilson(k, max(len(est), 1))}


def run(smoke=False):
    R = {"clean": 100, "rho": 80, "sync": 80, "sync02": 60, "deaf": 60, "viol": 50, "op": 80}
    if smoke:
        R = {k: max(3, v // 20) for k, v in R.items()}
    out = {"prereg": "PREREG_R9_operating_characteristics_2026_07_20.md",
           "design": {"alphas": ALPHAS, "betas": BETAS, "pi": PI, "n": N, "k_anchor": K,
                      "seed_bases": SEED_BASE, "R": R, "nominal_fa": NOMINAL_FA,
                      "s_null": S_NULL}, "gates": [], "characteristics": {}}
    ok_all = True

    def gate(name, cond, detail):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        out["gates"].append({"gate": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")

    print("== OC1 clean: calibration + validation ==")
    cal = family("clean_cal", R["clean"], lambda s: sync_rep(s))
    val = family("clean_val", R["clean"], lambda s: sync_rep(s))
    clean = cal + val
    thr = q([r["misfit"] for r in cal], 95)
    fa_k = sum(1 for r in val if r["misfit"] > thr)
    fa = fa_k / len(val)
    refuse_k = sum(1 for r in clean if r["verdict"] == "VOID_ANCHORS__nonexchangeable")
    phantom_k = sum(1 for r in clean if r["s"] is not None and r["s"] > S_NULL)
    errs_clean = [abs(r["pi"] - PI) for r in clean if r["pi"] is not None]

    print("== OC2 rho 0.30: both arms ==")
    rho_sync = family("rho30", R["rho"], lambda s: sync_rep(s, rho=0.30))
    rho_1p = family("oneparam_rho30", R["op"], lambda s: oneparam_rep(s, 0.30))

    print("== OC3 sync-on-real-only doses ==")
    s05 = family("sync05", R["sync"], lambda s: sync_rep(s, sync=0.05))
    s15 = family("sync15", R["sync"], lambda s: sync_rep(s, sync=0.15))
    s02 = family("sync02", R["sync02"], lambda s: sync_rep(s, sync=0.02))

    print("== OC4 deaf panel ==")
    deaf = family("deaf", R["deaf"], deaf_rep)
    void_k = sum(1 for d in deaf if d["void_plain"])
    void_nm_k = sum(1 for d in deaf if d["void_noise_margin"])

    print("== OC5 violation families (silent cases from the re-panel) ==")
    contam = family("contam10", R["viol"], lambda s: sync_rep(s, viol="contam10", n_boot=100))
    keypos = family("keypos", R["viol"], lambda s: sync_rep(s, viol="keypos", n_boot=100))
    betap = family("betaplus", R["viol"],
                   lambda s: sync_rep(s, organic_betas=[b + 0.10 for b in BETAS], n_boot=100))

    # ------------------------------------------------------------------------------- gates
    print("== GATES (calibration-shaped, frozen in the prereg) ==")
    cov_clean = coverage(clean)
    cov_rho = coverage(rho_sync)
    cov_05 = coverage(s05)
    cov_15 = coverage(s15)
    cov_1p = coverage(rho_1p)
    for tag, cov in (("clean", cov_clean), ("rho30_sync", cov_rho), ("sync05", cov_05),
                     ("sync15", cov_15), ("rho30_oneparam", cov_1p)):
        gate(f"G1:coverage_{tag}_in_band",
             cov["rate"] is not None and 0.90 <= cov["rate"] <= 0.99,
             f"{cov['covered']}/{cov['n_estimated']} = "
             f"{cov['rate'] if cov['rate'] is None else round(cov['rate'], 3)} "
             f"wilson {[round(x, 3) for x in cov['wilson']]}")
    gate("G2:misfit_false_alarm_calibrated", 0.01 <= fa <= 0.12,
         f"threshold {thr:.3f} (95th pct of cal), achieved {fa:.3f} on validation "
         f"({fa_k}/{len(val)}), nominal {NOMINAL_FA}")
    gate("G3:deaf_void_rate", void_k / len(deaf) >= 0.93,
         f"plain gate {void_k}/{len(deaf)} = {void_k / len(deaf):.3f} "
         f"(noise-margin variant {void_nm_k}/{len(deaf)} = {void_nm_k / len(deaf):.3f}, unbarred)")
    gate("G4:clean_false_refusal", refuse_k / len(clean) <= 0.02,
         f"{refuse_k}/{len(clean)} = {refuse_k / len(clean):.3f}")

    # ---------------------------------------------------------------- characteristics (no bars)
    def err_stats(recs, truth=PI):
        errs = [abs(r["pi"] - truth) for r in recs if r["pi"] is not None]
        return {"median": q(errs, 50), "p90": q(errs, 90), "n": len(errs)}

    def s_power(recs):
        det = sum(1 for r in recs if r.get("s_ci") and r["s_ci"][0] > 0)
        return {"detected": det, "n": len(recs), "rate": det / len(recs),
                "wilson": wilson(det, len(recs))}

    def viol_stats(recs):
        n = len(recs)
        flagged = sum(1 for r in recs if r["misfit"] is not None and r["misfit"] > thr)
        silent = sum(1 for r in recs if r["verdict"] == "ESTIMATED" and r["pi"] is not None
                     and abs(r["pi"] - PI) > 0.03 and r["misfit"] is not None
                     and r["misfit"] <= thr)
        voided = sum(1 for r in recs if r["verdict"] != "ESTIMATED")
        return {"n": n, "misfit_power": flagged / n, "misfit_power_wilson": wilson(flagged, n),
                "silent_wrong_rate": silent / n, "silent_wrong_wilson": wilson(silent, n),
                "void_rate": voided / n, **err_stats(recs)}

    out["characteristics"] = {
        "misfit_null": {"threshold_95": thr, "achieved_fa": fa,
                        "cal_misfit_quantiles": {"p50": q([r["misfit"] for r in cal], 50),
                                                 "p95": q([r["misfit"] for r in cal], 95),
                                                 "p99": q([r["misfit"] for r in cal], 99)},
                        "note": "design-point-specific; Stage B owes its own calibration"},
        "clean": {"phantom_sync_rate": phantom_k / len(clean),
                  "phantom_sync_wilson": wilson(phantom_k, len(clean)),
                  "false_refusal_rate": refuse_k / len(clean),
                  "err": {"median": q(errs_clean, 50), "p90": q(errs_clean, 90)},
                  "coverage": cov_clean},
        "rho30": {"sync_arm": {**err_stats(rho_sync), "coverage": cov_rho},
                  "oneparam_arm": {**err_stats(rho_1p), "coverage": cov_1p}},
        "sync_doses": {
            "0.02": {**err_stats(s02), "s_median": q([r["s"] for r in s02 if r["s"] is not None], 50),
                     "s_detection_power": s_power(s02), "coverage": coverage(s02)},
            "0.05": {**err_stats(s05), "s_median": q([r["s"] for r in s05 if r["s"] is not None], 50),
                     "s_detection_power": s_power(s05), "coverage": cov_05},
            "0.15": {**err_stats(s15), "s_median": q([r["s"] for r in s15 if r["s"] is not None], 50),
                     "s_detection_power": s_power(s15), "coverage": cov_15}},
        "deaf": {"void_rate_plain": void_k / len(deaf), "void_wilson": wilson(void_k, len(deaf)),
                 "void_rate_noise_margin": void_nm_k / len(deaf)},
        "violations": {"contam10": viol_stats(contam), "key_on_pos": viol_stats(keypos),
                       "beta_plus10": viol_stats(betap)},
        "grid_edge_rate_all_families": sum(1 for r in clean + rho_sync + s05 + s15 + s02
                                           + contam + keypos + betap if r["edge"])
                                       / len(clean + rho_sync + s05 + s15 + s02 + contam
                                             + keypos + betap)}
    out["all_gates_ok"] = ok_all
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="reduced-R smoke; *_SMOKE_INVALID only")
    args = ap.parse_args()
    t0 = time.time()
    out = run(smoke=args.smoke)
    name = ("stage_a_operating_chars_SMOKE_INVALID.json" if args.smoke
            else "stage_a_operating_chars_result.json")
    dest = HERE / name
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{'SMOKE (INVALID)' if args.smoke else 'RESULT'}: all_gates_ok={out['all_gates_ok']} "
          f"({time.time() - t0:.0f}s) -> {dest.name}")
    sys.exit(0 if out["all_gates_ok"] else 1)


if __name__ == "__main__":
    main()
