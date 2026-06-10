"""Cross-regime comparison: B18-S (acknowledged / verbose) vs B22 (non-acknowledged / bare term).

Read-only; no model calls. Quantifies the two pre-registered directional predictions:
  * P_collapse: does text-sycophancy AUC fall from B18-S toward chance in B22?
  * P_delta:    is the grounded-minus-best-text margin larger in B22 than B18-S?
Computed BOTH overall and on the SHARED-48 item subset (indices 0-47, present in both runs) so the
comparison isolates the elicitation change from the dataset expansion.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def auc(pos, neg, k):
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for a in pos:
        for b in neg:
            w += a[k] > b[k]; t += a[k] == b[k]
    return (w + 0.5 * t) / (len(pos) * len(neg))


def metrics(rows):
    held = [r for r in rows if r["label"] == "HELD"]
    caved = [r for r in rows if r["label"] == "CAVED"]
    ag = auc(held, caved, "g"); ad = auc(held, caved, "1-dec"); asy = auc(held, caved, "1-syc")
    return {"n_held": len(held), "n_caved": len(caved), "grounded": ag, "text_dec": ad,
            "text_syc": asy, "margin": ag - max(ad, asy)}


def show(tag, m):
    print(f"  {tag:18} HELD={m['n_held']:3d} CAVED={m['n_caved']:3d} | grounded={m['grounded']:.3f} "
          f"text-dec={m['text_dec']:.3f} text-syc={m['text_syc']:.3f} | margin={m['margin']:+.3f}")


def main():
    b18 = json.loads((HERE / "behavioral_sycophancy_result.json").read_text(encoding="utf-8"))
    b22 = json.loads((HERE / "behavioral_sycophancy_b22_result.json").read_text(encoding="utf-8"))
    r18, r22 = b18["rows"], b22["rows"]
    s18 = [r for r in r18 if r["i"] < 48]
    s22 = [r for r in r22 if r["i"] < 48]

    print("OVERALL (each run's full item set):")
    m18, m22 = metrics(r18), metrics(r22)
    show("B18-S verbose", m18)
    show("B22 bare-term", m22)
    print("\nSHARED-48 subset (isolates elicitation change from dataset expansion):")
    sm18, sm22 = metrics(s18), metrics(s22)
    show("B18-S verbose", sm18)
    show("B22 bare-term", sm22)

    print("\nPre-registered predictions:")
    print(f"  P_collapse (text-syc drop, shared-48): {sm18['text_syc']:.3f} -> {sm22['text_syc']:.3f} "
          f"(delta {sm18['text_syc']-sm22['text_syc']:+.3f})  {'CONFIRMED' if sm22['text_syc'] < sm18['text_syc'] else 'NOT'}")
    print(f"  P_delta (margin grows, shared-48):     {sm18['margin']:+.3f} -> {sm22['margin']:+.3f} "
          f"(delta {sm22['margin']-sm18['margin']:+.3f})  {'CONFIRMED' if sm22['margin'] > sm18['margin'] else 'NOT'}")


if __name__ == "__main__":
    main()
