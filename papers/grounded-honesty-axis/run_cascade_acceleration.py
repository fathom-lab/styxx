"""ACCELERATION bet — the load-bearing detector is also a COMPUTE ROUTER.

The expensive honesty signal is N=10 resampling instability (10 forward passes/item). The cheap
signal is single-pass clean-first-token entropy (1 forward pass) — it TIES resampling on derivation
but is WEAKER on factual recall (the detection-locus arc). The acceleration thesis: run the cheap
gate on EVERY item, and escalate to expensive resampling ONLY on the uncertain fraction the cheap
gate flags. Spend compute where the model is uncertain; skip it where it is confidently right.

Offline cascade analysis over the already-collected detection-locus per-item receipts (each row has
clean_entropy = cheap signal, instability = expensive N=10 signal, group = label, on the SAME item).
No new model runs — the cascade metric is novel (never computed before), so there is no peeking.

Cascade (tier-1 threshold tau1 on clean_entropy):
  - escalate iff clean_entropy >= tau1  (the cheap gate is uncertain)
  - cascade_score = (tau1 + instability) if escalated else clean_entropy
    -> all escalated items rank above all non-escalated (escalation itself is a positive signal),
       refined within each region by its own signal. A valid monotone combined ranking.
  - compute (forward passes / item) = 1 + N * escalation_rate   (1 cheap pass always; N resamples
    on the escalated fraction only). Full = N; cheap-only = 1.

Bars (pre-registered in PREREG_cascade_acceleration_2026_05_30.md):
  A1 (acceleration holds): there is a cascade operating point with
       AUC_cascade >= AUC_full - 0.02  AND  compute <= 0.40 * N   (pooled across regimes).
  A2 (regime structure, descriptive): the compute saving is larger where the cheap gate is strong
       (derivation) than where it is weak (factual).
  SURVIVED iff A1.

Usage:  python papers/grounded-honesty-axis/run_cascade_acceleration.py
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "cascade_acceleration_result.json"


def auc(scores, labels):
    """P(score|confab > score|correct), ties 0.5. labels: 1=confab, 0=correct."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def load_items():
    """Pool usable items from every detection-locus receipt that has BOTH signals per row."""
    regimes = {}
    seen_files = set()
    for f in sorted(glob.glob(str(HERE / "detection_locus*result*.json"))):
        name = Path(f).name
        if name in seen_files:
            continue
        seen_files.add(name)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("rows", [])
        if not rows or not all(k in rows[0] for k in ("clean_entropy", "instability", "group")):
            continue
        n_resample = d.get("n_resample", 10)
        items = []
        for r in rows:
            if not (r.get("usable", True) and r.get("member", True)):
                continue
            ce, inst = r.get("clean_entropy"), r.get("instability")
            if ce is None or inst is None:
                continue
            items.append((float(ce), float(inst), 1 if r.get("group") == "confab" else 0))
        if sum(y for *_, y in items) >= 4 and sum(1 - y for *_, y in items) >= 4:
            key = name.replace("detection_locus_", "").replace("_result", "").replace(".json", "")
            regimes[key] = {"items": items, "N": n_resample}
    return regimes


def cascade_curve(items, N):
    """Sweep tau1 over clean_entropy; return list of (tau1, esc_rate, compute, auc_cascade)."""
    ce = sorted(set(c for c, _, _ in items))
    thresholds = [ce[0] - 1.0] + ce + [ce[-1] + 1.0]   # from escalate-all to escalate-none
    out = []
    n = len(items)
    for tau1 in thresholds:
        esc = [(c, i, y) for (c, i, y) in items if c >= tau1]
        esc_rate = len(esc) / n
        compute = 1.0 + N * esc_rate
        scores = [(tau1 + i) if c >= tau1 else c for (c, i, y) in items]
        labels = [y for *_, y in items]
        out.append((tau1, esc_rate, compute, auc(scores, labels)))
    return out


def best_operating_point(items, N, auc_full, eps=0.02):
    """Min-compute cascade point with AUC >= auc_full - eps."""
    curve = cascade_curve(items, N)
    ok = [(tau1, esc, comp, a) for (tau1, esc, comp, a) in curve
          if a == a and a >= auc_full - eps]
    if not ok:
        # fall back to the max-AUC point
        curve_valid = [c for c in curve if c[3] == c[3]]
        return min(curve_valid, key=lambda t: (-t[3], t[2])) if curve_valid else None
    return min(ok, key=lambda t: t[2])   # least compute among qualifying


def main() -> int:
    regimes = load_items()
    if not regimes:
        print("no qualifying receipts found")
        return 1

    pooled = [it for r in regimes.values() for it in r["items"]]
    N = max(r["N"] for r in regimes.values())   # all 10
    h = hashlib.sha256(json.dumps(sorted(pooled), sort_keys=True).encode()).hexdigest()
    reg_summary = ", ".join(f"{k}(n={len(v['items'])})" for k, v in regimes.items())
    print(f"input items SHA-256 (pre-scoring): {h}")
    print(f"regimes: {reg_summary}")
    n_conf = sum(y for *_, y in pooled)
    print(f"pooled n={len(pooled)} (confab={n_conf}, correct={len(pooled) - n_conf}), N_resample={N}\n")

    labels = [y for *_, y in pooled]
    auc_full = auc([i for _, i, _ in pooled], labels)        # resample-all, compute = N
    auc_cheap = auc([c for c, _, _ in pooled], labels)        # cheap-only, compute = 1

    op = best_operating_point(pooled, N, auc_full)
    _, esc, comp, auc_c = op
    compute_ratio = comp / N

    # per-regime (A2, descriptive)
    per_regime = {}
    for k, v in regimes.items():
        items = v["items"]
        af = auc([i for _, i, _ in items], [y for *_, y in items])
        ac = auc([c for c, _, _ in items], [y for *_, y in items])
        o = best_operating_point(items, v["N"], af)
        per_regime[k] = {
            "n": len(items), "auc_full_resample": round(af, 4) if af == af else None,
            "auc_cheap_only": round(ac, 4) if ac == ac else None,
            "cascade_compute_passes": round(o[2], 3) if o else None,
            "cascade_compute_ratio": round(o[2] / v["N"], 4) if o else None,
            "cascade_auc": round(o[3], 4) if o else None,
            "escalation_rate": round(o[1], 4) if o else None}

    a1 = (auc_c >= auc_full - 0.02) and (compute_ratio <= 0.40)
    result = "SURVIVED" if a1 else "REPORT_AS_LANDED"

    receipt = {
        "experiment": "ACCELERATION — the detector as a compute router: cheap single-pass gate routes the uncertain fraction to expensive N=10 resampling. How much compute is saved at ~full detection?",
        "prereg": "papers/grounded-honesty-axis/PREREG_cascade_acceleration_2026_05_30.md",
        "input_items_sha256_pre_scoring": h,
        "pooled_n": len(pooled), "n_resample": N,
        "baselines": {
            "resample_all": {"auc": round(auc_full, 4), "compute_passes": N},
            "cheap_only": {"auc": round(auc_cheap, 4) if auc_cheap == auc_cheap else None, "compute_passes": 1}},
        "cascade_operating_point": {
            "auc": round(auc_c, 4), "compute_passes": round(comp, 3),
            "compute_ratio_vs_full": round(compute_ratio, 4),
            "escalation_rate": round(esc, 4),
            "speedup_x": round(N / comp, 2)},
        "per_regime": per_regime,
        "A1_accel": {"value": {"cascade_auc": round(auc_c, 4), "auc_full": round(auc_full, 4),
                               "compute_ratio": round(compute_ratio, 4)},
                     "held": bool(a1),
                     "bar": "AUC >= AUC_full - 0.02 AND compute <= 0.40*N"},
        "RESULT": result,
        "honest_scope": (
            "offline cascade analysis over already-collected detection-locus per-item receipts "
            "(white-box open models: Qwen/Llama/Gemma; arithmetic/code/logic/factual). The cascade "
            "metric is novel (never computed before) so there is no data-peek; but the underlying "
            "signals were collected by prior runs, not for this test. Compute is counted in forward "
            "passes (cheap gate=1, each resample=1). Real wall-clock also depends on batching. This "
            "ACCELERATES detection (routes compute), it does not change WHAT is detected or correct "
            "anything (correction is the closed negative). Single-token closed-model confab is out of "
            "scope (the cheap first-token gate fails there; span/closed regimes excluded)."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "honest_scope"}, indent=2))
    print(f"\nFULL resample-all: AUC {auc_full:.3f} @ {N} passes/item")
    print(f"CHEAP only:        AUC {auc_cheap:.3f} @ 1 pass/item")
    print(f"CASCADE:           AUC {auc_c:.3f} @ {comp:.2f} passes/item "
          f"({compute_ratio*100:.0f}% of full, {N/comp:.1f}x speedup, escalate {esc*100:.0f}%)")
    print(f"-> A1={a1} => {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
