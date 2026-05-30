"""ACCELERATION bet 2 — adaptive early-stopping on the EXPENSIVE tier (compounds with the cascade).

The cascade routes WHO gets resampled. This trims HOW MANY samples each escalated item needs. The
detection signal is resampling instability = (distinct-1)/(N-1) over N=10 @ T=1.0. But on derivation,
correct items are near-perfectly stable (every sample agrees) and confabs scatter within the first
few samples — so most of the N=10 budget may be wasted. How few samples retain detection?

Offline over the already-collected detection-locus per-item RESAMPLE SEQUENCES (each row stores the
length-N list of sampled integers, the group label, the correct answer). Per-model calibrated (the
valid unit, per the cascade finding). The early-stop metric is new -> no data-peek.

Two analyses:
  1. FIXED-k: instability from the first k samples; AUC_k vs k; min_k with AUC_k >= AUC_N - 0.02.
  2. ADAPTIVE (distinct-count certificate, monotone): an item is confab iff it yields >= D distinct
     values in N samples. Stop early and decide CONFAB the instant distinct reaches D (distinct only
     grows), or CORRECT once D is unreachable in the remaining budget. Report mean samples drawn.

Bars (PREREG_resample_earlystop_2026_05_30.md):
  E1: median over regimes of min_k <= 5 (>= 2x saving on the resample tier) at AUC_k >= AUC_N - 0.02.
  E2 (descriptive): the compounding speedup = cascade routing x early-stop trimming.
  SURVIVED iff E1.

Usage:  python papers/grounded-honesty-axis/run_resample_earlystop.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "resample_earlystop_result.json"


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def distinct_count(seq):
    """running distinct count after each prefix length."""
    seen, out = set(), []
    for x in seq:
        seen.add(x)
        out.append(len(seen))
    return out


def instability_k(seq, k):
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
        if not rows or "resamples" not in rows[0] or "group" not in rows[0]:
            continue
        N = d.get("n_resample", 10)
        items = []
        for r in rows:
            if not (r.get("usable", True) and r.get("member", True)):
                continue
            seq = r.get("resamples")
            if not seq or len(seq) < N:
                continue
            items.append((list(seq[:N]), 1 if r.get("group") == "confab" else 0))
        if sum(y for _, y in items) >= 4 and sum(1 - y for _, y in items) >= 4:
            key = name.replace("detection_locus_", "").replace("_result", "").replace(".json", "")
            regimes[key] = {"items": items, "N": N}
    return regimes


def main() -> int:
    regimes = load_regimes()
    if not regimes:
        print("no qualifying receipts")
        return 1
    h = hashlib.sha256(json.dumps([[s, y] for v in regimes.values()
                       for s, y in v["items"]], default=str).encode()).hexdigest()
    print(f"input SHA-256 (pre-scoring): {h}")

    per_regime, min_ks = {}, []
    for key, v in regimes.items():
        items, N = v["items"], v["N"]
        labels = [y for _, y in items]
        auc_N = auc([instability_k(s, N) for s, _ in items], labels)
        # fixed-k
        min_k = N
        auc_by_k = {}
        for k in range(2, N + 1):
            ak = auc([instability_k(s, k) for s, _ in items], labels)
            auc_by_k[k] = round(ak, 4) if ak == ak else None
            if ak == ak and ak >= auc_N - 0.02 and min_k == N:
                min_k = k
        # adaptive distinct-count certificate: best D (Youden on full-N distinct), then mean samples
        dvals = [distinct_count(s) for s, _ in items]
        full_distinct = [dc[-1] for dc in dvals]
        best_D, best_j = 2, -1.0
        for D in range(2, N + 1):
            tpr = sum(1 for fd, y in zip(full_distinct, labels) if y == 1 and fd >= D) / max(1, sum(labels))
            fpr = sum(1 for fd, y in zip(full_distinct, labels) if y == 0 and fd >= D) / max(1, len(labels) - sum(labels))
            if tpr - fpr > best_j:
                best_j, best_D = tpr - fpr, D
        # samples drawn to certify (reach D, or prove D unreachable: distinct + remaining < D)
        drawn = []
        for dc in dvals:
            stop = len(dc)
            for k in range(1, len(dc) + 1):
                if dc[k - 1] >= best_D or dc[k - 1] + (len(dc) - k) < best_D:
                    stop = k
                    break
            drawn.append(stop)
        per_regime[key] = {
            "n": len(items), "auc_full_N": round(auc_N, 4) if auc_N == auc_N else None,
            "min_k_for_full_detection": min_k, "auc_by_k": auc_by_k,
            "adaptive_best_D_distinct": best_D,
            "adaptive_mean_samples": round(statistics.mean(drawn), 3),
            "adaptive_youden_J": round(best_j, 4)}
        if auc_N == auc_N:
            min_ks.append(min_k)

    median_min_k = statistics.median(min_ks) if min_ks else None
    e1 = median_min_k is not None and median_min_k <= 5
    result = "SURVIVED" if e1 else "REPORT_AS_LANDED"

    # compounding with the cascade (read the cascade receipt if present)
    compounding = None
    cf = HERE / "cascade_acceleration_result.json"
    if cf.exists():
        cres = json.load(open(cf, encoding="utf-8"))
        pr = cres.get("per_regime", {})
        N0 = cres.get("n_resample", 10)
        tot_items = tot_passes_casc = tot_passes_compound = 0
        for key, prr in per_regime.items():
            cr = pr.get(key)
            if not cr or cr.get("escalation_rate") is None:
                continue
            n = cr["n"]; esc = cr["escalation_rate"]; mk = prr["min_k_for_full_detection"]
            tot_items += n
            tot_passes_casc += n * (1 + N0 * esc)                # cascade alone
            tot_passes_compound += n * (1 + mk * esc)            # cascade + early-stop
        if tot_items:
            compounding = {
                "full_passes_per_item": N0,
                "cascade_only_passes": round(tot_passes_casc / tot_items, 3),
                "cascade_plus_earlystop_passes": round(tot_passes_compound / tot_items, 3),
                "compound_speedup_x": round(N0 / (tot_passes_compound / tot_items), 2)}

    receipt = {
        "experiment": "ACCELERATION 2 — adaptive early-stopping on the resample tier: how few samples retain detection, and the compounding speedup with the cascade",
        "prereg": "papers/grounded-honesty-axis/PREREG_resample_earlystop_2026_05_30.md",
        "input_sha256_pre_scoring": h,
        "per_regime": per_regime,
        "median_min_k": median_min_k,
        "E1_min_k_le_5": {"value": median_min_k, "held": bool(e1), "bar": "median min_k <= 5 at AUC_k >= AUC_N - 0.02"},
        "compounding_with_cascade": compounding,
        "RESULT": result,
        "honest_scope": (
            "offline over already-collected detection-locus resample sequences; per-model calibrated; "
            "white-box derivation+factual; exact-integer resampling (the signal is sample agreement). "
            "Counts samples/forward-passes, not wall-clock. min_k is in-sample (the AUC_k curve is "
            "descriptive); a clean confirmation needs held-out items. Accelerates detection, corrects "
            "nothing. Where full-N detection is itself weak (noisy regimes) min_k is unstable."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("honest_scope",)}, indent=2))
    print(f"\nmedian min_k = {median_min_k} (of N={max(v['N'] for v in regimes.values())}) -> E1={e1} {result}")
    if compounding:
        print(f"COMPOUNDING: full {compounding['full_passes_per_item']} -> cascade "
              f"{compounding['cascade_only_passes']} -> +early-stop "
              f"{compounding['cascade_plus_earlystop_passes']} passes/item "
              f"({compounding['compound_speedup_x']}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
