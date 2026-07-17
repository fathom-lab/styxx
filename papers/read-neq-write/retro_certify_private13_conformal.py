"""Retro-certify the private-13 honesty probe with a TRANSFER-GUARANTEED threshold rule.

Follow-up owed by commit 0e707db. The first pass (retro_certify_private13.py) ran the private-13
honesty probe through styxx.instrument_admissibility and landed VOID_INSTRUMENT__nonspecific:
sensitivity strong (discrim 0.938, two-sided p 0.002) but the DEPLOYMENT threshold -- the
mount.calibrate_threshold rank rule at target_fpr=0.20 on the CALIB negatives -- fired on 0.548 of
EVAL nulls (n_null=31), blowing 174% past its calibrated 0.20 operating point. That commit
pre-registered the fix, verbatim: "a threshold rule with a transfer guarantee (e.g. conformal /
quantile with finite-sample correction) certified through the same gate."

This script keeps the INSTRUMENT identical -- same model, same stem-disjoint E1 split, same best-of-13
probe (logreg_L20), the SAME per-item scores -- and changes ONLY the threshold RULE, from the
mount-style point estimate to two finite-sample conformal rules:

  (1) SPLIT-CONFORMAL quantile.  For a "fire when score > tau" detector at target FPR alpha on n
      calibration negatives, tau = the ceil((n+1)*(1-alpha))-th smallest calib-negative score.
      GUARANTEE (distribution-free, finite-sample, marginal over the calibration draw, assuming the
      n calibration negatives and a future negative are EXCHANGEABLE):
          P(score_new > tau) <= alpha
      Equivalently coverage P(score_new <= tau) >= ceil((n+1)(1-alpha))/(n+1) >= 1-alpha. This is the
      Vovk conformal quantile; the (n+1) is the finite-sample correction the naive empirical quantile
      lacks. If ceil((n+1)(1-alpha)) > n the guarantee needs a threshold above every calibration
      point -> tau = +inf (never fire).

  (2) BETA / order-statistic TOLERANCE bound at (alpha, delta).  A strictly stronger, more
      conservative claim: not just "expected FPR <= alpha" but "with confidence >= 1-delta the TRUE
      exceedance probability <= alpha". tau = X_(r), the r-th smallest calib-negative, where r is the
      SMALLEST rank whose order-statistic coverage clears the confidence bar. The fraction of the
      population above X_(r) is Beta(n-r+1, r)-distributed (continuous case; conservative under ties),
      so the confidence that it is <= alpha is I_alpha(n-r+1, r) (regularized incomplete beta, here
      the exact binomial sum). Pick the smallest r with I_alpha(n-r+1, r) >= 1-delta; if none exists
      for r<=n the (alpha,delta) tolerance is unattainable at finite n -> tau = +inf. This is the
      classic distribution-free one-sided tolerance interval.

We also report a DKW-band variant (epsilon = sqrt(ln(2/delta)/(2n)); target the tightened level
alpha - epsilon) as a third, coarser conservative correction, for the transfer curve only.

THE HONEST FORK (pre-stated, not tuned):
  * if a conformal rule certifies SPECIFIC on EVAL -> the 0.548 gap was a quantile-DERIVATION
    artifact (mount's point estimate lacked the finite-sample correction) and the repaired operating
    point is certified ADMISSIBLE.
  * if a conformal rule STILL fires far above 0.20 on EVAL nulls -> exchangeability itself FAILS
    between CALIB and EVAL. Here CALIB negatives are the logistic probe's OWN TRAINING-split
    decision-function values (the probe is fit on CALIB), so they are in-sample and NOT exchangeable
    with held-out EVAL negatives; the miscalibration is a real train/eval distribution shift in
    probe-score space, which no threshold rule computed on the training scores can repair. That is
    the sharper finding, and it is reported verbatim whichever way the numbers land.

Deliverables (NOT committed):
  retro_cert_private13_conformal.json       -- split-conformal threshold, certified + re-verified
  retro_cert_private13_conformal_beta.json  -- beta-tolerance threshold, certified + re-verified
both via .certificate(out_path=...), receipts = this script's sha256, re-verified with
verify_admissibility_certificate (receipts re-hashed + every headline recomputed from stored points).

Usage:
  python papers/read-neq-write/retro_certify_private13_conformal.py         # real GPU pass (~40s)
  python papers/read-neq-write/retro_certify_private13_conformal.py --dry   # CPU-only synthetic harness test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.admissibility import instrument_admissibility, verify_admissibility_certificate


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Reuse the ORIGINAL retro-cert script verbatim: same model/split/probe/scoring code path, so the
# instrument's per-item scores are byte-for-byte the ones the committed cert was issued from. Only
# the threshold RULE below differs. (Attribution: build_probes / random_decoy / real_scores /
# three_way_split all live in retro_certify_private13.py @ 0e707db and papers/read-neq-write/*.)
ORIG = _load("retro_orig", HERE / "retro_certify_private13.py")

# frozen certification settings -- IDENTICAL to the original so the only moving part is fire_threshold
TARGET_FPR = ORIG.TARGET_FPR      # 0.20 deployment FPR target
MAX_FIRE = ORIG.MAX_FIRE          # 0.30 specificity gate (1.5x target; see original docstring)
AUROC_FLOOR = ORIG.AUROC_FLOOR    # 0.70
ALPHA_SIG = ORIG.ALPHA            # 0.05 significance alpha (NOT the FPR alpha)
K_PERM = ORIG.K_PERM              # 1000
SEED = ORIG.SEED                  # 0
EXPECT = ORIG.EXPECT              # "higher_on_positive"
DELTA = 0.10                      # tolerance-bound confidence miss rate (1-delta = 0.90 confidence)

SCRIPT_REL = "papers/read-neq-write/retro_certify_private13_conformal.py"
CERT_CONF = HERE / "retro_cert_private13_conformal.json"
CERT_BETA = HERE / "retro_cert_private13_conformal_beta.json"
OLD_CERT = HERE / "retro_cert_private13.json"


# --------------------------------------------------------------------------------------------
# threshold rules (all CPU / pure) -- a "higher fires" detector: fewer false alarms at HIGHER tau
# --------------------------------------------------------------------------------------------

def split_conformal_threshold(neg_scores, alpha):
    """Split-conformal (Vovk) quantile for a higher-fires detector at target FPR alpha.

    tau = ceil((n+1)*(1-alpha))-th SMALLEST calibration-negative score. Under exchangeability of the
    n calibration negatives with a future negative, P(score_new > tau) <= alpha exactly (marginal
    over the calibration draw). The (n+1) is the finite-sample correction. Returns (tau, rank, n)."""
    s = np.sort(np.asarray(neg_scores, dtype=float))   # ascending
    n = len(s)
    if n == 0:
        return float("nan"), 0, 0
    k = int(math.ceil((n + 1) * (1.0 - alpha)))        # 1-indexed rank of the order statistic
    if k > n:
        return float("inf"), k, n                       # guarantee needs tau above every calib point
    return float(s[k - 1]), k, n


def _betacdf_alpha_int(a_shape, b_shape, alpha):
    """Regularized incomplete beta I_alpha(a, b) for POSITIVE INTEGER a,b via the exact binomial
    identity I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j), with n=a+b-1. No scipy."""
    n = a_shape + b_shape - 1
    x = float(alpha)
    return float(sum(math.comb(n, j) * (x ** j) * ((1.0 - x) ** (n - j))
                     for j in range(a_shape, n + 1)))


def beta_tolerance_threshold(neg_scores, alpha, delta):
    """(alpha, delta) one-sided upper tolerance bound via order statistics.

    tau = X_(r), the r-th smallest calib-negative, with r the SMALLEST rank whose order-statistic
    coverage clears confidence 1-delta: the fraction of the population above X_(r) is Beta(n-r+1, r),
    so confidence(true tail <= alpha) = I_alpha(n-r+1, r). Return (tau, rank_or_None, n, confidence).
    If no r<=n reaches 1-delta the tolerance is unattainable at this n -> tau=+inf, rank=None."""
    s = np.sort(np.asarray(neg_scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan"), None, 0, float("nan")
    for r in range(1, n + 1):
        conf = _betacdf_alpha_int(n - r + 1, r, alpha)   # confidence true exceedance-prob <= alpha
        if conf >= 1.0 - delta:
            return float(s[r - 1]), r, n, conf
    return float("inf"), None, n, float(_betacdf_alpha_int(1, n, alpha))  # best (r=n) still short


def dkw_threshold(neg_scores, alpha, delta):
    """DKW-band conservative variant: with prob >= 1-delta, sup|F_n - F| <= eps, eps =
    sqrt(ln(2/delta)/(2n)). To force true tail <= alpha, target the tightened empirical level
    alpha_eff = alpha - eps and take the ceil(n*(1-alpha_eff))-th smallest score. If alpha_eff <= 0
    the band is wider than the target and no finite threshold qualifies -> +inf. (eps, alpha_eff)."""
    s = np.sort(np.asarray(neg_scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    eps = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    alpha_eff = alpha - eps
    if alpha_eff <= 0.0:
        return float("inf"), eps, alpha_eff
    rank = int(math.ceil(n * (1.0 - alpha_eff)))
    if rank > n:
        return float("inf"), eps, alpha_eff
    return float(s[rank - 1]), eps, alpha_eff


def realized_fpr(neg_scores, tau, expect=EXPECT):
    """Out-of-sample / in-sample false-alarm rate at tau for the given fire direction."""
    s = np.asarray(neg_scores, dtype=float)
    if len(s) == 0:
        return float("nan")
    fired = (s > tau) if expect == "higher_on_positive" else (s < tau)
    return float(np.mean(fired))


# --------------------------------------------------------------------------------------------
# certify EVAL at a caller threshold (identical gate to the original, only fire_threshold changes)
# --------------------------------------------------------------------------------------------

def certify_at(*, eval_scores, eval_labels, fire_threshold, instrument_name, receipts, out_path, root):
    rep = instrument_admissibility(
        scores=eval_scores, labels=eval_labels, expect=EXPECT,
        fire_threshold=fire_threshold, auroc_floor=AUROC_FLOOR, alpha=ALPHA_SIG, max_fire=MAX_FIRE,
        k_perm=K_PERM, seed=SEED, receipts=receipts,
    )
    rep.instrument = instrument_name
    rep.certificate(receipts=receipts, out_path=str(out_path))
    v = verify_admissibility_certificate(str(out_path), root=root)
    return rep, v


def _print_verify(tag, out_path, v):
    print(f"  -> {out_path}")
    print(f"  verify: receipts {'OK' if v['ok'] else 'DRIFT'} "
          f"(checked {v['checked']}/{v['n_recorded']}); "
          f"recompute {'FAITHFUL' if v['faithful'] else 'UNFAITHFUL'}")
    for fd in v.get("field_diffs", []):
        print(f"     DIFF {fd['field']}: stored {fd['stored']!r} != recomputed {fd['recomputed']!r}")


# --------------------------------------------------------------------------------------------
# the run (shared by --dry synthetic and the real GPU pass)
# --------------------------------------------------------------------------------------------

def run_conformal(*, eval_scores, eval_labels, calib_scores, calib_labels,
                  probe_tag, out_conf, out_beta, root, receipts, verbose=True):
    eval_scores = np.asarray(eval_scores, dtype=float)
    eval_labels = np.asarray(eval_labels, dtype=int)
    calib_scores = np.asarray(calib_scores, dtype=float)
    calib_labels = np.asarray(calib_labels, dtype=int)

    calib_neg = calib_scores[calib_labels == 0]
    eval_neg = eval_scores[eval_labels == 0]
    n_cn = len(calib_neg)

    # baseline: the ORIGINAL mount-style point estimate, recomputed here for the transfer curve
    tau_mount = ORIG.mount_style_threshold(calib_neg, TARGET_FPR)
    tau_conf, k_conf, _ = split_conformal_threshold(calib_neg, TARGET_FPR)
    tau_beta, r_beta, _, conf_beta = beta_tolerance_threshold(calib_neg, TARGET_FPR, DELTA)
    tau_dkw, eps_dkw, aeff_dkw = dkw_threshold(calib_neg, TARGET_FPR, DELTA)

    curve = [
        ("mount point-estimate", tau_mount, None),
        (f"SPLIT-CONFORMAL k={k_conf}/{n_cn}", tau_conf, None),
        (f"BETA-TOLERANCE r={r_beta}/{n_cn} conf={conf_beta:.3f}", tau_beta, None),
        (f"DKW eps={eps_dkw:.3f} aeff={aeff_dkw:.3f}", tau_dkw, None),
    ]
    if verbose:
        print("\n" + "=" * 90)
        print(f"TRANSFER CURVE  --  {probe_tag}")
        print(f"  calib negatives n={n_cn}  |  eval negatives n={len(eval_neg)}  |  target FPR "
              f"alpha={TARGET_FPR}  delta={DELTA}  gate max_fire={MAX_FIRE}")
        print("=" * 90)
        print(f"  {'rule':<40s} {'threshold':>12s} {'CALIB FPR(in)':>14s} {'EVAL FPR(out)':>14s}")
        for name, tau, _ in curve:
            cfpr = realized_fpr(calib_neg, tau)
            efpr = realized_fpr(eval_neg, tau)
            tau_s = "  +inf" if math.isinf(tau) else f"{tau:12.6f}"
            print(f"  {name:<40s} {tau_s:>12s} {cfpr:14.3f} {efpr:14.3f}")

    # --- certify EVAL at the two conformal thresholds ---
    conf_name = (f"{probe_tag} | SPLIT-CONFORMAL fire_threshold (alpha={TARGET_FPR}, "
                 f"rank k={k_conf}/n={n_cn}, tau={'+inf' if math.isinf(tau_conf) else round(tau_conf, 6)}) | "
                 f"EVAL nulls out-of-sample")
    beta_name = (f"{probe_tag} | BETA-TOLERANCE fire_threshold (alpha={TARGET_FPR}, delta={DELTA}, "
                 f"rank r={r_beta}/n={n_cn}, conf={round(conf_beta, 4)}, "
                 f"tau={'+inf' if math.isinf(tau_beta) else round(tau_beta, 6)}) | EVAL nulls out-of-sample")

    rep_c, v_c = certify_at(eval_scores=eval_scores, eval_labels=eval_labels, fire_threshold=tau_conf,
                            instrument_name=conf_name, receipts=receipts, out_path=out_conf, root=root)
    rep_b, v_b = certify_at(eval_scores=eval_scores, eval_labels=eval_labels, fire_threshold=tau_beta,
                            instrument_name=beta_name, receipts=receipts, out_path=out_beta, root=root)

    if verbose:
        print("\n" + "-" * 90)
        print("SPLIT-CONFORMAL certificate")
        print("-" * 90)
        print(rep_c.summary())
        _print_verify("conf", out_conf, v_c)
        print("\n" + "-" * 90)
        print("BETA-TOLERANCE certificate")
        print("-" * 90)
        print(rep_b.summary())
        _print_verify("beta", out_beta, v_b)

    return {
        "n_calib_neg": n_cn, "n_eval_neg": int(len(eval_neg)),
        "mount": {"tau": tau_mount, "eval_fpr": realized_fpr(eval_neg, tau_mount)},
        "conformal": {"tau": tau_conf, "rank": k_conf, "eval_fpr": rep_c.fire_rate,
                      "calib_fpr": realized_fpr(calib_neg, tau_conf),
                      "verdict": rep_c.admissibility_verdict, "rep": rep_c, "verify": v_c},
        "beta": {"tau": tau_beta, "rank": r_beta, "conf": conf_beta, "eval_fpr": rep_b.fire_rate,
                 "calib_fpr": realized_fpr(calib_neg, tau_beta),
                 "verdict": rep_b.admissibility_verdict, "rep": rep_b, "verify": v_b},
        "dkw": {"tau": tau_dkw, "eps": eps_dkw, "alpha_eff": aeff_dkw,
                "eval_fpr": realized_fpr(eval_neg, tau_dkw)},
    }


# --------------------------------------------------------------------------------------------
# --dry: CPU-only synthetic harness test (no model). Validates the conformal MATH + cert plumbing.
# --------------------------------------------------------------------------------------------

def dry() -> int:
    print("[--dry] synthetic scores, CPU-only, no model (conformal math + harness plumbing test)")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    fails = []

    # (A) EXCHANGEABLE world: calib negs and eval negs drawn from the SAME N(-1.5,1); positives
    #     N(+1.5,1). Conformal transfer guarantee should HOLD -> EVAL FPR ~ alpha, cert ADMISSIBLE.
    n_eval, n_calib = 400, 300
    e_y = np.array([1] * (n_eval // 2) + [0] * (n_eval - n_eval // 2))
    e_s = np.where(e_y == 1, rng.normal(1.5, 1.0, n_eval), rng.normal(-1.5, 1.0, n_eval))
    c_y = np.array([1] * (n_calib // 2) + [0] * (n_calib - n_calib // 2))
    c_s = np.where(c_y == 1, rng.normal(1.5, 1.0, n_calib), rng.normal(-1.5, 1.0, n_calib))

    # closed-form checks of the two rules on a known vector
    neg = np.sort(c_s[c_y == 0])
    tau_conf, k_conf, n_cn = split_conformal_threshold(neg, TARGET_FPR)
    assert k_conf == math.ceil((n_cn + 1) * (1 - TARGET_FPR)), "conformal rank formula"
    assert tau_conf == neg[k_conf - 1], "conformal picks the k-th order statistic"
    tau_beta, r_beta, _, conf_beta = beta_tolerance_threshold(neg, TARGET_FPR, DELTA)
    assert r_beta is not None and tau_beta == neg[r_beta - 1], "beta picks the r-th order statistic"
    assert conf_beta >= 1 - DELTA, "beta confidence clears 1-delta"
    assert r_beta >= k_conf, "beta tolerance is >= as conservative as split-conformal"
    # cross-check the scipy-free incomplete beta against scipy when present
    try:
        from scipy.stats import beta as _sbeta
        ref = float(_sbeta.cdf(TARGET_FPR, n_cn - r_beta + 1, r_beta))
        assert abs(ref - conf_beta) < 1e-9, f"betacdf mismatch {ref} vs {conf_beta}"
        print(f"  [ok] scipy-free I_alpha matches scipy.stats.beta.cdf ({conf_beta:.6f})")
    except ImportError:
        print("  [skip] scipy absent -- incomplete-beta cross-check skipped")

    tmp = Path(tempfile.mkdtemp(prefix="retro_cert_conformal_dry_"))
    receipts = {SCRIPT_REL: hashlib.sha256((ROOT / SCRIPT_REL).read_bytes()).hexdigest()}
    resA = run_conformal(eval_scores=e_s, eval_labels=e_y, calib_scores=c_s, calib_labels=c_y,
                         probe_tag="SYNTHETIC exchangeable detector (dry-A)",
                         out_conf=tmp / "dryA_conf.json", out_beta=tmp / "dryA_beta.json",
                         root=ROOT, receipts=receipts, verbose=True)
    if resA["conformal"]["verdict"] != "ADMISSIBLE":
        fails.append(f"dry-A conformal verdict {resA['conformal']['verdict']} != ADMISSIBLE")
    if resA["conformal"]["eval_fpr"] > MAX_FIRE:
        fails.append(f"dry-A conformal EVAL FPR {resA['conformal']['eval_fpr']} > gate")
    for k in ("conformal", "beta"):
        if not (resA[k]["verify"]["ok"] and resA[k]["verify"]["faithful"]):
            fails.append(f"dry-A {k} cert not ok/faithful")

    # (B) SHIFTED world: eval negs shifted UP by +2 vs calib negs -> exchangeability BROKEN ->
    #     conformal threshold (built on calib) must FIRE far above alpha on eval -> nonspecific.
    e_s2 = e_s.copy()
    e_s2[e_y == 0] = rng.normal(0.5, 1.0, int((e_y == 0).sum()))  # calib negs ~ -1.5, eval negs ~ +0.5
    resB = run_conformal(eval_scores=e_s2, eval_labels=e_y, calib_scores=c_s, calib_labels=c_y,
                         probe_tag="SYNTHETIC shifted-null detector (dry-B)",
                         out_conf=tmp / "dryB_conf.json", out_beta=tmp / "dryB_beta.json",
                         root=ROOT, receipts=receipts, verbose=False)
    if resB["conformal"]["verdict"] != "VOID_INSTRUMENT__nonspecific":
        fails.append(f"dry-B conformal verdict {resB['conformal']['verdict']} != nonspecific")
    print(f"\n  [dry-B] shift-broken exchangeability: conformal EVAL FPR "
          f"{resB['conformal']['eval_fpr']:.3f} -> {resB['conformal']['verdict']}")

    # (C) +inf edge: tiny n where beta tolerance is unattainable -> +inf -> FPR 0 -> (in)sensitive path
    tiny = np.array([-3.0, -2.0, -1.0])
    tau_i, r_i, _, _ = beta_tolerance_threshold(tiny, TARGET_FPR, DELTA)
    if not (math.isinf(tau_i) and r_i is None):
        fails.append(f"dry-C expected +inf tolerance at n=3, got tau={tau_i} r={r_i}")
    if realized_fpr(np.array([0.0, 1.0, 2.0]), tau_i) != 0.0:
        fails.append("dry-C +inf threshold should fire on nothing")
    print(f"  [dry-C] n=3 beta tolerance unattainable -> tau=+inf, FPR 0 (edge handled)")

    ok = not fails
    print(f"\n[--dry] harness self-check: {'PASS' if ok else 'FAIL'}   ({time.time() - t0:.1f}s)")
    for f in fails:
        print(f"   FAIL: {f}")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------
# real GPU pass
# --------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="CPU-only synthetic harness test (no model)")
    a = ap.parse_args()
    if a.dry:
        return dry()

    t0 = time.time()
    script_sha = hashlib.sha256((ROOT / SCRIPT_REL).read_bytes()).hexdigest()
    receipts = {SCRIPT_REL: script_sha}

    # SAME instrument, SAME scores: reuse the original real_scores() verbatim.
    (best_eval, e_y, best_calib, c_y, decoy_eval, decoy_calib,
     probe_name, decoy_name, meta) = ORIG.real_scores()
    best = meta["best"]
    probe_tag = (f"private-13 honesty probe (best-of-13: {best['name']}, EVAL AUROC "
                 f"{best['auroc']:.4f}) | clean {ORIG.MODEL} | fit=CALIB read=EVAL")

    # cross-check: our reused EVAL scores must equal the committed cert's stored points (same instrument)
    if OLD_CERT.exists():
        old = json.loads(OLD_CERT.read_text(encoding="utf-8"))
        old_s = np.array([p["score"] for p in old["points"]], dtype=float)
        old_l = np.array([p["label"] for p in old["points"]], dtype=int)
        if len(old_s) == len(best_eval):
            dmax = float(np.max(np.abs(np.sort(best_eval) - np.sort(old_s))))
            lab_ok = int((np.asarray(e_y) == old_l).sum() if len(old_l) == len(e_y) else -1)
            print(f"\n[cross-check vs committed retro_cert_private13.json] "
                  f"max|EVAL score diff| (sorted) = {dmax:.2e}  labels_match={lab_ok}/{len(e_y)}  "
                  f"old_threshold={old['specificity']['fire_threshold']}  "
                  f"old_fire_rate={old['specificity']['fire_rate']}")
        else:
            print(f"\n[cross-check] committed cert has {len(old_s)} points, reused EVAL has "
                  f"{len(best_eval)} -- lengths differ, skipping numeric compare")

    res = run_conformal(eval_scores=best_eval, eval_labels=e_y, calib_scores=best_calib,
                        calib_labels=c_y, probe_tag=probe_tag,
                        out_conf=CERT_CONF, out_beta=CERT_BETA, root=ROOT, receipts=receipts,
                        verbose=True)

    # --- the honest fork verdict ---
    cf, bf = res["conformal"], res["beta"]
    print("\n" + "=" * 90)
    print("HONEST FORK (point 4)")
    print("=" * 90)
    print(f"  mount point-estimate : tau={res['mount']['tau']:.6f}  EVAL FPR {res['mount']['eval_fpr']:.3f}  "
          f"(committed baseline = 0.548)")
    conf_tau = "+inf" if math.isinf(cf["tau"]) else f"{cf['tau']:.6f}"
    beta_tau = "+inf" if math.isinf(bf["tau"]) else f"{bf['tau']:.6f}"
    print(f"  SPLIT-CONFORMAL      : tau={conf_tau}  CALIB FPR(in) {cf['calib_fpr']:.3f}  "
          f"EVAL FPR(out) {cf['eval_fpr']:.3f}  -> {cf['verdict']}")
    print(f"  BETA-TOLERANCE       : tau={beta_tau}  CALIB FPR(in) {bf['calib_fpr']:.3f}  "
          f"EVAL FPR(out) {bf['eval_fpr']:.3f}  -> {bf['verdict']}")
    any_specific = ("ADMISSIBLE" in cf["verdict"] and "ONLY" not in cf["verdict"]) or \
                   ("ADMISSIBLE" in bf["verdict"] and "ONLY" not in bf["verdict"])
    if any_specific:
        print("  FORK => (i) QUANTILE-DERIVATION ARTIFACT: a conformal rule certified SPECIFIC on "
              "EVAL; the mount point-estimate lacked the finite-sample correction. Repaired operating "
              "point CERTIFIED.")
    else:
        print("  FORK => (ii) EXCHANGEABILITY FAILS between CALIB and EVAL: even a finite-sample "
              "conformal / tolerance threshold built on CALIB negatives fires far above alpha on EVAL "
              "nulls. CALIB scores are the probe's OWN in-sample (training-split) decision values, so "
              "they are NOT exchangeable with held-out EVAL negatives -- a real train/eval "
              "distribution shift in probe-score space that no CALIB-derived threshold can repair. "
              "This is the sharper finding.")

    print(f"\n[wall] {time.time() - t0:.1f}s")
    print(f"[files] {CERT_CONF}")
    print(f"        {CERT_BETA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
