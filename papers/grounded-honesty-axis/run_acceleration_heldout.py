"""ACCELERATION — HELD-OUT confirmation: pick the cascade threshold + early-stop k on TRAIN, freeze,
evaluate the compounded route x trim speedup on UNSEEN TEST. PREREG_acceleration_heldout_2026_05_30.

The cascade (FINDING_cascade_acceleration) and early-stop (FINDING_resample_earlystop) reported the
5.15x compounding with IN-SAMPLE operating points. This is the clean confirmation: per regime, on a
stratified TRAIN half, pick (tau1 = escalation threshold on clean_entropy, min_k = early-stop sample
count) that minimizes train compute subject to train cascade-AUC >= train full-N-AUC - 0.02. FREEZE
them. On the held-out TEST half, measure the compounded passes/item and the retained detection AUC.
Repeat over R=5 stratified splits (seeds 0..4) and aggregate (n-weighted). No test peeking.

Unified item = (clean_entropy [cheap, 1 pass], resamples [for early-stopped instability], full-N
instability, label), pooled per regime from receipts that carry ALL of them. Per-model calibrated.

Cascade+early-stop score: escalate iff clean_entropy >= tau1; score = (tau1 + instability_from_first_
min_k_samples) if escalated else clean_entropy. Compute = 1 + min_k * escalation_rate.

Bar (fixed in the PREREG):
  HC1: aggregate held-out test cascade-AUC >= test full-N-AUC - 0.03  AND  test passes <= 0.50 * N.
  SURVIVED iff HC1.

Usage:  python papers/grounded-honesty-axis/run_acceleration_heldout.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "acceleration_heldout_result.json"
R_SPLITS = 5
TRAIN_FRAC = 0.5
TRAIN_TOL = 0.02      # train cascade-AUC must be within this of train full-N-AUC
TEST_TOL = 0.03       # HC1 held-out AUC tolerance
COMPUTE_BAR = 0.50    # HC1 compute <= 0.50 * N


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def instab_k(seq, k):
    return (len(set(seq[:k])) - 1) / (k - 1) if k >= 2 else 0.0


def load_regimes():
    regimes, seen = {}, set()
    for f in sorted(glob.glob(str(HERE / "detection_locus*result*.json"))):
        name = Path(f).name
        if name in seen:
            continue
        seen.add(name)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("rows", [])
        if not rows or not all(k in rows[0] for k in ("clean_entropy", "instability", "resamples", "group")):
            continue
        N = d.get("n_resample", 10)
        items = []
        for r in rows:
            if not (r.get("usable", True) and r.get("member", True)):
                continue
            ce, inst, seq = r.get("clean_entropy"), r.get("instability"), r.get("resamples")
            if ce is None or inst is None or not seq or len(seq) < N:
                continue
            items.append((float(ce), list(seq[:N]), float(inst), 1 if r.get("group") == "confab" else 0))
        nc = sum(it[3] for it in items)
        if nc >= 8 and (len(items) - nc) >= 8:   # enough per class to split 50/50 with >=4/4
            key = name.replace("detection_locus_", "").replace("_result", "").replace(".json", "")
            regimes[key] = {"items": items, "N": N}
    return regimes


def split(items, seed):
    conf = [it for it in items if it[3] == 1]
    corr = [it for it in items if it[3] == 0]
    rng = random.Random(seed)
    rng.shuffle(conf)
    rng.shuffle(corr)
    nc, nk = max(1, int(len(conf) * TRAIN_FRAC)), max(1, int(len(corr) * TRAIN_FRAC))
    return conf[:nc] + corr[:nk], conf[nc:] + corr[nk:]


def pick_operating_point(train, N):
    """min-train-compute (tau1, min_k) with train cascade-AUC >= train full-N-AUC - TRAIN_TOL."""
    labels = [it[3] for it in train]
    auc_full = auc([it[2] for it in train], labels)
    best = None
    for tau1 in sorted(set(it[0] for it in train)):
        n_esc = sum(1 for it in train if it[0] >= tau1)
        esc_rate = n_esc / len(train)
        for k in range(2, N + 1):
            score = [(tau1 + instab_k(it[1], k)) if it[0] >= tau1 else it[0] for it in train]
            a = auc(score, labels)
            if a == a and a >= auc_full - TRAIN_TOL:
                compute = 1.0 + k * esc_rate
                if best is None or compute < best[0]:
                    best = (compute, tau1, k, a)
    if best is None:   # fall back to escalate-all + full N (= resample everything)
        lo = min(it[0] for it in train) - 1.0
        best = (1.0 + N, lo, N, auc_full)
    return best[1], best[2]   # tau1, min_k


def evaluate(test, tau1, min_k, N):
    labels = [it[3] for it in test]
    score = [(tau1 + instab_k(it[1], min_k)) if it[0] >= tau1 else it[0] for it in test]
    esc_rate = sum(1 for it in test if it[0] >= tau1) / len(test)
    return (auc(score, labels), auc([it[2] for it in test], labels),
            1.0 + min_k * esc_rate, esc_rate)


def main() -> int:
    regimes = load_regimes()
    if not regimes:
        print("no qualifying receipts (need clean_entropy + instability + resamples)")
        return 1
    N = max(v["N"] for v in regimes.values())
    h = hashlib.sha256(json.dumps([[it[0], it[2], it[3]] for v in regimes.values()
                       for it in v["items"]], default=str).encode()).hexdigest()
    print(f"input SHA-256 (pre-scoring): {h}")
    print(f"regimes={len(regimes)} N={N} splits={R_SPLITS} train_frac={TRAIN_FRAC}\n")

    per_regime = {}
    for key, v in regimes.items():
        items, n = v["items"], v["N"]
        rows = []
        for seed in range(R_SPLITS):
            tr, te = split(items, seed)
            tau1, min_k = pick_operating_point(tr, n)
            ta, tfa, passes, esc = evaluate(te, tau1, min_k, n)
            if ta == ta and tfa == tfa:
                rows.append((ta, tfa, passes, esc, min_k))
        if not rows:
            continue
        per_regime[key] = {
            "n": len(items),
            "test_cascade_auc": round(statistics.mean(r[0] for r in rows), 4),
            "test_full_auc": round(statistics.mean(r[1] for r in rows), 4),
            "test_passes_per_item": round(statistics.mean(r[2] for r in rows), 3),
            "test_escalation": round(statistics.mean(r[3] for r in rows), 4),
            "median_min_k": statistics.median(r[4] for r in rows),
            "speedup_x": round(n / statistics.mean(r[2] for r in rows), 2)}

    tot = sum(p["n"] for p in per_regime.values())
    agg_auc = sum(p["n"] * p["test_cascade_auc"] for p in per_regime.values()) / tot
    agg_full = sum(p["n"] * p["test_full_auc"] for p in per_regime.values()) / tot
    agg_passes = sum(p["n"] * p["test_passes_per_item"] for p in per_regime.values()) / tot
    hc1 = (agg_auc >= agg_full - TEST_TOL) and (agg_passes <= COMPUTE_BAR * N)
    result = "SURVIVED" if hc1 else "REPORT_AS_LANDED"

    receipt = {
        "experiment": "ACCELERATION held-out confirmation: train-picked (cascade tau1 + early-stop min_k) frozen, evaluated on unseen test, R=5 stratified splits, per-model calibrated",
        "prereg": "papers/grounded-honesty-axis/PREREG_acceleration_heldout_2026_05_30.md",
        "input_sha256_pre_scoring": h,
        "N_resample": N, "splits": R_SPLITS, "train_frac": TRAIN_FRAC,
        "aggregate_heldout": {
            "test_cascade_auc": round(agg_auc, 4), "test_full_auc": round(agg_full, 4),
            "auc_delta": round(agg_auc - agg_full, 4),
            "test_passes_per_item": round(agg_passes, 3),
            "speedup_x": round(N / agg_passes, 2)},
        "HC1": {"bar": f"test AUC >= full AUC - {TEST_TOL} AND passes <= {COMPUTE_BAR}*N",
                "auc_ok": bool(agg_auc >= agg_full - TEST_TOL),
                "compute_ok": bool(agg_passes <= COMPUTE_BAR * N), "held": bool(hc1)},
        "per_regime": per_regime,
        "RESULT": result,
        "honest_scope": (
            "held-out (train picks operating points, test evaluates), R=5 stratified splits, "
            "per-model calibrated, offline over detection-locus receipts carrying clean_entropy + "
            "instability + resamples. Counts forward passes, not wall-clock. Confirms the route x "
            "trim acceleration generalizes to unseen items within each model/regime; does NOT claim "
            "cross-model transport of a FIXED threshold (entropy scale is per-model). Accelerates "
            "detection, corrects nothing."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "honest_scope"}, indent=2))
    print(f"\nHELD-OUT aggregate: cascade AUC {agg_auc:.4f} vs full {agg_full:.4f} "
          f"(delta {agg_auc-agg_full:+.4f}) @ {agg_passes:.2f} passes/item ({N/agg_passes:.1f}x)")
    print(f"HC1: auc_ok={agg_auc >= agg_full - TEST_TOL} compute_ok={agg_passes <= COMPUTE_BAR*N} -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
