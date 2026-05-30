"""Can a SPAN-AGGREGATE single-pass signal recover closed-model confab detection where the
FIRST-TOKEN signal fails? PREREG_detection_locus_gpt_span_2026_05_30.md.

The first-token closed-model run (run_detection_locus_gpt.py) tests whether single-pass entropy at
the FIRST answer token detects gpt-4o-mini confabulation. The probe showed it should fail: gpt-4o-mini
emits correct LEADING digits and confabulates TRAILING digits, so the error is DOWNSTREAM of the
first token. But the OpenAI logprobs expose top-20 at EVERY answer token — so a single-pass signal
aggregated across the WHOLE answer span (mean / max token entropy, min token margin) should see the
trailing-digit uncertainty the first token misses. STILL ONE forward pass, NO resampling.

This run computes, on the SAME seeded items as run_detection_locus_gpt (same hash), per-answer-token
entropy/margin and both FIRST-token and SPAN-aggregate detectors, plus N=10 resampling instability:
  B1   = AUC(resampling instability)            -- baseline (confab IS detectable)
  first = best AUC of first-token entropy/margin -- the SHIPPED single_pass_confab gate
  span  = best AUC of span-aggregate (mean/max entropy, -mean/-min margin)
  B_contrast_first = B1 - first ;  B_contrast_span = B1 - span

RECOVERY SURVIVED iff B1>=0.70 AND span>=0.70 AND B_contrast_span<0.20 (span ties resampling) AND
B_contrast_first>=0.20 (first-token loses) -- i.e. a one-forward-pass SPAN gate recovers closed-model
confab detection to resampling parity where the first-token gate fails. Reported either way.

Usage:
    python papers/grounded-honesty-axis/run_detection_locus_gpt_span.py --n 4   # pilot
    python papers/grounded-honesty-axis/run_detection_locus_gpt_span.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_detection_locus import stability_of  # noqa: E402
from run_confabulation_specificity import auc_score  # noqa: E402
from run_detection_locus_gpt import (  # noqa: E402
    HARD, EASY, MODEL, N_RESAMPLE, TEMPERATURE, SEED, _call, _parse_int)

RECEIPT = HERE / "detection_locus_gpt_span_result.json"


def _span_signals(r):
    """Per-answer-token entropy/margin (top-20 + residual bucket), aggregated. None if absent."""
    lc = r.choices[0].logprobs.content if r.choices[0].logprobs else None
    if not lc:
        return None
    ents, margs = [], []
    for tok in lc:
        top = tok.top_logprobs
        if not top:
            continue
        ps = [math.exp(t.logprob) for t in top]
        resid = max(0.0, 1.0 - sum(ps))
        h = -sum(p * math.log(p) for p in ps if p > 0.0)
        if resid > 1e-12:
            h -= resid * math.log(resid)
        ents.append(h)
        margs.append((top[0].logprob - top[1].logprob) if len(top) >= 2 else 0.0)
    if not ents:
        return None
    return {
        "first_entropy": ents[0], "mean_entropy": sum(ents) / len(ents), "max_entropy": max(ents),
        "first_margin": margs[0], "mean_margin": sum(margs) / len(margs), "min_margin": min(margs),
        "n_tokens": len(ents),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(HARD))
    ap.add_argument("--model", type=str, default=MODEL)
    args = ap.parse_args(argv)
    import run_detection_locus_gpt as _G
    _G.MODEL = args.model   # _call (imported) reads its module global -> override for cross-model

    hard = HARD[: args.n]
    easy = EASY[: args.n] if args.n < len(HARD) else EASY
    items = hard + easy

    key_blob = json.dumps([(q, c) for _, q, c, _ in items], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={args.model} N_resample={N_RESAMPLE} temp={TEMPERATURE} seed={SEED}")

    rows = []
    for subset, user, correct, grp in items:
        rg = _call(user, 0.0, True)
        a1 = rg.choices[0].message.content
        v1 = _parse_int(a1); ok1 = (v1 == correct)
        member = (grp == "correct" and ok1) or (grp == "confab" and (not ok1) and v1 is not None)
        row = {"group": grp, "subset": subset, "correct": correct, "v1": v1,
               "ok1": ok1, "member": bool(member), "usable": False}
        if member:
            sig = _span_signals(rg)
            if sig is not None:
                vals = [_parse_int(_call(user, TEMPERATURE, False).choices[0].message.content)
                        for _ in range(N_RESAMPLE)]
                stab, nd = stability_of(vals)
                row.update({"usable": True, "instability": 1.0 - stab, "n_distinct": nd,
                            "resamples": vals, **sig})
        rows.append(row)
        if row["usable"]:
            print(f"[{grp:7}|{subset:4}] {user:>16}={correct:<10} v1={str(v1):<10} | "
                  f"inst={row['instability']:.2f} | first_ent={row['first_entropy']:.3f} "
                  f"mean_ent={row['mean_entropy']:.3f} max_ent={row['max_entropy']:.3f} "
                  f"min_marg={row['min_margin']:.2f} (ntok={row['n_tokens']})")
        else:
            print(f"[{grp:7}|{subset:4}] {user:>16}={correct:<10} v1={str(v1):<10} | non-member")

    conf = [r for r in rows if r["usable"] and r["group"] == "confab"]
    corr = [r for r in rows if r["usable"] and r["group"] == "correct"]
    n_conf, n_corr = len(conf), len(corr)
    powered = (n_conf >= 12) and (n_corr >= 12)
    labels = [1] * n_conf + [0] * n_corr

    def auc_for(key, sign=1.0):
        sc = [sign * r[key] for r in conf] + [sign * r[key] for r in corr]
        return auc_score(labels, sc)

    a_inst = auc_for("instability", 1.0)
    a_first_ent = auc_for("first_entropy", 1.0)
    a_first_marg = auc_for("first_margin", -1.0)
    a_mean_ent = auc_for("mean_entropy", 1.0)
    a_max_ent = auc_for("max_entropy", 1.0)
    a_mean_marg = auc_for("mean_margin", -1.0)
    a_min_marg = auc_for("min_margin", -1.0)
    best_first = max(a_first_ent, a_first_marg)
    best_span = max(a_mean_ent, a_max_ent, a_mean_marg, a_min_marg)
    bc_first = a_inst - best_first
    bc_span = a_inst - best_span

    b1 = powered and a_inst >= 0.70
    b_span = powered and best_span >= 0.70
    b_recovery = powered and bc_span < 0.20
    b_first_fails = powered and bc_first >= 0.20
    result = "SURVIVED" if (b1 and b_span and b_recovery and b_first_fails) else "REPORT_AS_LANDED"

    def m(rs, k):
        a = np.array([r[k] for r in rs], float)
        return round(float(a.mean()), 4) if len(a) else None

    receipt = {
        "experiment": "detection locus — can a SPAN-AGGREGATE single-pass signal recover gpt-4o-mini confab detection where the FIRST-TOKEN gate fails? (one forward pass, no resampling)",
        "prereg": "papers/grounded-honesty-axis/PREREG_detection_locus_gpt_span_2026_05_30.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": args.model, "domain": "multiplication, closed-model API, full-span logprobs",
        "seed": SEED, "n_resample": N_RESAMPLE, "temperature": TEMPERATURE,
        "n_confab_usable": n_conf, "n_correct_usable": n_corr, "powered": powered,
        "means": {
            "confab_instability": m(conf, "instability"), "correct_instability": m(corr, "instability"),
            "confab_first_entropy": m(conf, "first_entropy"), "correct_first_entropy": m(corr, "first_entropy"),
            "confab_mean_entropy": m(conf, "mean_entropy"), "correct_mean_entropy": m(corr, "mean_entropy"),
            "confab_max_entropy": m(conf, "max_entropy"), "correct_max_entropy": m(corr, "max_entropy"),
            "confab_min_margin": m(conf, "min_margin"), "correct_min_margin": m(corr, "min_margin"),
            "confab_n_tokens": m(conf, "n_tokens"), "correct_n_tokens": m(corr, "n_tokens")},
        "AUC": {
            "B1_resampling_instability": round(a_inst, 4) if a_inst == a_inst else None,
            "first_entropy": round(a_first_ent, 4) if a_first_ent == a_first_ent else None,
            "first_neg_margin": round(a_first_marg, 4) if a_first_marg == a_first_marg else None,
            "span_mean_entropy": round(a_mean_ent, 4) if a_mean_ent == a_mean_ent else None,
            "span_max_entropy": round(a_max_ent, 4) if a_max_ent == a_max_ent else None,
            "span_neg_mean_margin": round(a_mean_marg, 4) if a_mean_marg == a_mean_marg else None,
            "span_neg_min_margin": round(a_min_marg, 4) if a_min_marg == a_min_marg else None},
        "best_first_token_auc": round(best_first, 4) if best_first == best_first else None,
        "best_span_aggregate_auc": round(best_span, 4) if best_span == best_span else None,
        "B_contrast_first_token": round(bc_first, 4) if bc_first == bc_first else None,
        "B_contrast_span_aggregate": round(bc_span, 4) if bc_span == bc_span else None,
        "conditions": {"B1_resampling>=0.70": bool(b1), "B_span>=0.70": bool(b_span),
                       "B_recovery_span_ties_resampling<0.20": bool(b_recovery),
                       "B_first_token_fails>=0.20": bool(b_first_fails)},
        "rows": rows,
        "RESULT": result,
        "honest_scope": (
            f"single closed model {args.model} via OpenAI API; multiplication only; one confirmatory "
            "run; feasibility-grade; per-answer-token entropy/margin from top-20 logprobs + residual "
            "bucket (TRUNCATED proxy); resampling N=10 at T=1.0 (exact-integer, no judge); ground "
            "truth in-code, hashed pre-scoring. Same items/hash as run_detection_locus_gpt (direct "
            "first-token-vs-span comparison). SURVIVED = a ONE-forward-pass span-aggregate gate "
            "recovers closed-model confab detection to resampling parity where the first-token gate "
            "fails -> a cheap closed-model confab signal. Difficulty confound (CONFAB-hard/"
            "CORRECT-easy) held fixed across detector TYPES by the contrasts. Does NOT touch the "
            "correctness bound: signals DETECT, none CORRECTS."),
    }
    out_path = RECEIPT if args.model == MODEL else (
        HERE / f"detection_locus_gpt_span_result_{args.model.replace('.', '_')}.json")
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nn_conf={n_conf} n_corr={n_corr} powered={powered}")
    print(f"B1(inst)={a_inst:.3f} | first best={best_first:.3f} (Bc_first={bc_first:+.3f}) | "
          f"span best={best_span:.3f} (Bc_span={bc_span:+.3f}) -> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
