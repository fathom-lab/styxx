"""Cycle 61 -- the BURIED-JUDGE family: does the honest ladder PRICE or REFUSE a single
informative judge as its true separation sinks toward the noise-margin gate?

Cycle 60 closed the four-family PRICE/REFUSE partition (attr+numeric price, chain+temporal
refuse) and named the sharpening next step: "a genuinely-informative-but-HARD family (real judge
buried under noise) would sharpen the price/refuse boundary". This is that family, built on the
SEALED Stage-A generating process (anchored_stage_a.simulate_panel / make_anchors) and audited by
the SHIPPED public instrument styxx.anchors.audit_panel -- nothing new is invented, the DGP and
the verifier are the ones already in the package.

CONSTRUCTION. Four judges. Judge 0 is the ONLY informative one: alpha0 = 0.30, beta0 = 0.30 + sep.
Judges 1-3 are deaf (alpha 0.45, beta 0.50, true separation 0.05 -- below the 0.15 gate always).
Anchors are drawn from the SAME per-judge (alpha, beta) as the organic panel -> exchangeable,
honest. PI = 0.35, N = 6000, K = 400 per stratum (the design point). In expectation the single
informative judge identifies pi exactly: t = 0.30 + 0.35*sep, A = 0.30, B = 0.30 + sep, so
(t - A) / (B - A) = 0.35 for every sep. The audit is therefore never wrong in expectation; the
question is entirely whether the noise-margin gate + anchor sampling let the ESTIMATE stay COVERED
as the separation is buried.

WHY THE KILL CAN FIRE (this is not a victory-lap family). The noise-margin gate keeps judge 0 iff
its ANCHOR-MEASURED separation beta_hat - alpha_hat >= 0.15 + 3*sqrt(...) ~ 0.25 at K=400. Near
that gate the keep decision SELECTS on the anchor draw: a replicate survives only when its anchor
draw happens to show a LARGE beta_hat - alpha_hat, which inflates the denominator (B - A) and biases
pi = (t - A)/(B - A) DOWNWARD relative to the true 0.35. Selection-on-separation can therefore make
the priced replicates systematically miscovered -- an over-pricing danger the sweep is built to
catch. The two-sided design forces the output to MOVE with the input (deaf VOIDs; strong prices)
and then asks the load-bearing question at the boundary.

FROZEN BARS (see PREREG_buried_judge_2026_07_24.md; nothing here computed after seeing results):
  PD_MOVE   (validity precondition, two-sided sanity -- NOT the discovery):
              deaf cell (sep=0.00) VOID rate >= 0.90 AND strong cell (sep=0.40) ESTIMATED
              rate >= 0.90. If either fails the instrument is not reading separation and the run
              is INVALID (reported as blocked, not as a result).
  PD_DANGER (THE KILL -- can fire CLOSED_NEGATIVE): for every cell the ladder PRICES (ESTIMATED
              rate >= 0.50), pi-CI coverage of PI among the ESTIMATED replicates must be >= 0.80
              (the instrument's own worst measured regime, styxx.anchors docstring). If ANY priced
              cell covers < 0.80 -> CLOSED_NEGATIVE (over-pricing a buried judge). Reported verbatim.
  PD_HONEST (refusal is clean): no VOID replicate may emit a non-null pi (a refusal that leaks a
              number is not a refusal).
  PD_BOUNDARY (deliverable, REPORTED not gated): the smallest sep with ESTIMATED rate >= 0.50 AND
              coverage >= 0.90 -- the "buried but recoverable" threshold that sharpens the boundary.

CPU only, deterministic seed bases, ASCII only. `--smoke` runs reduced R and writes ONLY a
*_SMOKE_INVALID* file (never read as a result).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchored_stage_a as A

# repo root on path for styxx.anchors (the SHIPPED instrument under test)
sys.path.insert(0, str(HERE.parent.parent))
from styxx.anchors import audit_panel  # noqa: E402

# ------------------------------------------------------------------- frozen design constants
PI, N, K = 0.35, 6000, 400
ALPHA_INFORM = 0.30                  # informative judge false-positive rate
DEAF_ALPHA, DEAF_BETA = 0.45, 0.50   # 3 deaf judges (true sep 0.05, below gate always)
SEP_GRID = [0.00, 0.16, 0.22, 0.25, 0.28, 0.34, 0.40]   # judge-0 true separation sweep;
# 0.22/0.25/0.28 bracket the effective noise-margin gate (~0.255 at K=400), where the keep
# decision selects on the anchor draw and the over-pricing kill can actually fire.
R_DEFAULT = 60
NULL_SIMS = 200                      # per-dataset tau + misfit (the shipped default)
N_BOOT = 300
COVERAGE_FLOOR = 0.80                # instrument's worst measured regime (anchors docstring)
RECOVER_TARGET = 0.90                # PD_BOUNDARY "recoverable" coverage
SEED_BASE = 610_000                  # cycle 61; disjoint per (cell, replicate)


def _panel(rng, sep):
    """Organic + anchor strata for a single-informative-judge panel at true separation `sep`.
    Judge 0 informative; judges 1-3 deaf. Anchors share the SAME per-judge rates (exchangeable)."""
    alphas = [ALPHA_INFORM, DEAF_ALPHA, DEAF_ALPHA, DEAF_ALPHA]
    betas = [ALPHA_INFORM + sep, DEAF_BETA, DEAF_BETA, DEAF_BETA]
    _, V = A.simulate_panel(rng, N, PI, alphas, betas)
    neg = A.make_anchors(rng, K, "neg", alphas, betas)
    pos = A.make_anchors(rng, K, "pos", alphas, betas)
    return V, neg, pos


def one_rep(seed, sep):
    rng = np.random.default_rng(seed)
    V, neg, pos = _panel(rng, sep)
    r = audit_panel(V, neg, pos, n_boot=N_BOOT, null_sims=NULL_SIMS, seed=seed + 5_000_000)
    verdict = r["verdict"]
    pi = r.get("pi")
    ci = r.get("ci")
    covered = bool(ci is not None and ci[0] <= PI <= ci[1])
    return {"sep": sep, "verdict": verdict, "pi": pi, "ci": ci, "covered": covered,
            "kept": r.get("kept"), "activated": r.get("activated"),
            "regime": r.get("regime")}


def wilson(k, n):
    if n == 0:
        return [0.0, 1.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def summarize_cell(recs):
    n = len(recs)
    est = [r for r in recs if r["verdict"] == "ESTIMATED"]
    void = [r for r in recs if r["verdict"] == "VOID_PANEL__uninformative"]
    nonx = [r for r in recs if r["verdict"] == "VOID_ANCHORS__nonexchangeable"]
    cov_k = sum(1 for r in est if r["covered"])
    cov_rate = (cov_k / len(est)) if est else None
    void_leak = sum(1 for r in recs
                    if r["verdict"] != "ESTIMATED" and r["pi"] is not None)
    pis = [r["pi"] for r in est if r["pi"] is not None]
    return {"n": n,
            "n_estimated": len(est), "estimated_rate": len(est) / n,
            "n_void_panel": len(void), "void_rate": len(void) / n,
            "n_void_anchors": len(nonx),
            "coverage_k": cov_k, "coverage_n": len(est),
            "coverage_rate": cov_rate,
            "coverage_wilson": wilson(cov_k, len(est)) if est else None,
            "pi_median": float(np.median(pis)) if pis else None,
            "pi_p10": float(np.percentile(pis, 10)) if pis else None,
            "pi_p90": float(np.percentile(pis, 90)) if pis else None,
            "void_leak": void_leak}


def run(smoke=False):
    R = 3 if smoke else R_DEFAULT
    t0 = time.time()
    cells = {}
    for gi, sep in enumerate(SEP_GRID):
        ts = time.time()
        recs = [one_rep(SEED_BASE + gi * 100_000 + i, sep) for i in range(R)]
        cells[f"{sep:.2f}"] = {"sep": sep, "reps": recs, "summary": summarize_cell(recs)}
        s = cells[f"{sep:.2f}"]["summary"]
        print(f"  sep {sep:.2f}: est_rate {s['estimated_rate']:.3f} void_rate {s['void_rate']:.3f} "
              f"cov {s['coverage_rate']} ({s['coverage_k']}/{s['coverage_n']}) "
              f"in {time.time() - ts:.0f}s")

    # ------------------------------------------------------------------------------- gates
    deaf = cells[f"{SEP_GRID[0]:.2f}"]["summary"]
    strong = cells[f"{SEP_GRID[-1]:.2f}"]["summary"]
    pd_move = (deaf["void_rate"] >= 0.90) and (strong["estimated_rate"] >= 0.90)

    priced_cells = {k: c for k, c in cells.items() if c["summary"]["estimated_rate"] >= 0.50}
    danger_hits = [{"sep": k, "coverage_rate": c["summary"]["coverage_rate"],
                    "coverage": f"{c['summary']['coverage_k']}/{c['summary']['coverage_n']}"}
                   for k, c in priced_cells.items()
                   if (c["summary"]["coverage_rate"] is not None
                       and c["summary"]["coverage_rate"] < COVERAGE_FLOOR)]
    pd_danger_kill = len(danger_hits) > 0

    void_leaks = {k: c["summary"]["void_leak"] for k, c in cells.items()
                  if c["summary"]["void_leak"] > 0}
    pd_honest = len(void_leaks) == 0

    # PD_BOUNDARY: smallest sep priced (>=0.50) AND recoverable (coverage >= RECOVER_TARGET)
    recover_sep = None
    for sep in SEP_GRID:
        s = cells[f"{sep:.2f}"]["summary"]
        if (s["estimated_rate"] >= 0.50 and s["coverage_rate"] is not None
                and s["coverage_rate"] >= RECOVER_TARGET):
            recover_sep = sep
            break

    if not pd_move:
        verdict = "INVALID__instrument_not_reading_separation"
    elif pd_danger_kill:
        verdict = "CLOSED_NEGATIVE__ladder_over_prices_buried_judge"
    elif not pd_honest:
        verdict = "CLOSED_NEGATIVE__refusal_leaks_a_number"
    else:
        verdict = "SURVIVED__prices_where_covered_refuses_below_gate"

    gates = [
        {"gate": "PD_MOVE", "ok": bool(pd_move),
         "detail": f"deaf(sep0.00) void_rate {deaf['void_rate']:.3f}>=0.90 AND "
                   f"strong(sep0.40) est_rate {strong['estimated_rate']:.3f}>=0.90"},
        {"gate": "PD_DANGER", "ok": bool(not pd_danger_kill),
         "detail": f"priced cells (est_rate>=0.50): {sorted(priced_cells)}; "
                   f"coverage<{COVERAGE_FLOOR} in: {danger_hits or 'none'}"},
        {"gate": "PD_HONEST", "ok": bool(pd_honest),
         "detail": f"void replicates emitting a pi: {void_leaks or 'none'}"},
    ]
    for g in gates:
        print(f"  [{'OK ' if g['ok'] else 'FAIL'}] {g['gate']}: {g['detail']}")
    print(f"  PD_BOUNDARY (reported): smallest priced+recoverable(cov>={RECOVER_TARGET}) sep = "
          f"{recover_sep}")
    print(f"  VERDICT: {verdict}")

    return {"prereg": "PREREG_buried_judge_2026_07_24.md",
            "design": {"pi": PI, "n": N, "k_anchor": K, "alpha_inform": ALPHA_INFORM,
                       "deaf": [DEAF_ALPHA, DEAF_BETA], "sep_grid": SEP_GRID, "R": R,
                       "null_sims": NULL_SIMS, "n_boot": N_BOOT,
                       "coverage_floor": COVERAGE_FLOOR, "recover_target": RECOVER_TARGET,
                       "seed_base": SEED_BASE, "instrument": "styxx.anchors.audit_panel"},
            "cells": {k: {"sep": c["sep"], "summary": c["summary"]} for k, c in cells.items()},
            "gates": gates,
            "pd_boundary_recoverable_sep": recover_sep,
            "verdict": verdict,
            "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-R smoke; writes ONLY *_SMOKE_INVALID*")
    args = ap.parse_args()
    out = run(smoke=args.smoke)
    name = ("buried_judge_SMOKE_INVALID.json" if args.smoke
            else "buried_judge_result.json")
    dest = HERE / name
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{'SMOKE (INVALID)' if args.smoke else 'RESULT'}: {out['verdict']} "
          f"({out['elapsed_s']:.0f}s) -> {dest.name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
