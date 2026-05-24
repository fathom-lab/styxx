# -*- coding: utf-8 -*-
"""over-flag reliability — the bounded finding, made usable (prototype, 2026-05-24).

What we PROVED (cross-model finding, commit 80a81c6): a model's mean token
logprob predicts where `refuse_check` over-flags a compliant response — positive
across 5 models incl. cross-family, but attenuating out of the OpenAI family,
and only on the error-bearing (compliance) class. NOT a universal validity oracle.

This wraps exactly that — no more — as a callable reliability flag:

    flag = overflag_reliability(refuse_risk, mean_logprob)
    # flag.validity   : 0..1 — higher = the refusal score is more trustworthy here
    # flag.note        : human-readable, scoped to over-flagging

The demo runs it on the real holdouts and shows the calibration: when the flag
says "low validity", how much more often is the refusal score actually an
over-flag? That is the whole value proposition, measured — not asserted.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path("papers/grounded-arc")

# Calibration constants (chosen, not fit on the holdout — illustrative, honest).
# Center near the OpenAI median mean-logprob; slope chosen for a readable spread.
_LP_CENTER = -0.20
_LP_SLOPE = 6.0


@dataclass
class OverflagFlag:
    validity: float          # 0..1, higher = refusal score more trustworthy on this response
    note: str


def overflag_reliability(refuse_risk: float, mean_logprob: float) -> OverflagFlag:
    """Reliability flag for a refusal score, derived from the generation's own
    confidence. Scoped to ONE failure mode: over-flagging a compliant response.
    Higher generation confidence (less-negative mean logprob) ⇒ the refusal
    score is more trustworthy. Bounded result — weaker across model families."""
    validity = 1.0 / (1.0 + math.exp(-_LP_SLOPE * (mean_logprob - _LP_CENTER)))
    if refuse_risk >= 0.5 and validity < 0.4:
        note = "refusal flagged, but model generated this uncertainly — possible OVER-FLAG, verify"
    elif validity < 0.4:
        note = "low generation confidence — refusal score is less reliable here"
    else:
        note = "high generation confidence — refusal score is trustworthy"
    return OverflagFlag(round(validity, 3), note)


def _load(tag, fn=None):
    f = HERE / "holdout" / f"{fn or ('refusal_' + tag)}.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def demo():
    from styxx.guardrail import refuse_check
    openai = {"gpt-4o-mini": "refusal_bet0b", "gpt-4o": None, "gpt-4.1-mini": None, "gpt-4.1": None}

    def calib(rows, label):
        # compliance subset only (gold==0) — the error-bearing class
        comp = [r for r in rows if r["gold"] == 0]
        for r in comp:
            r["_risk"] = float(refuse_check(prompt=r["prompt"], response=r["response"]).refuse_risk)
            r["_val"] = overflag_reliability(r["_risk"], r["mean_logprob"]).validity
        comp.sort(key=lambda r: r["_val"])
        n = len(comp); third = n // 3
        bins = [("low-validity", comp[:third]), ("mid", comp[third:2 * third]), ("high-validity", comp[2 * third:])]
        print(f"\n{label}  (n={n} compliant responses; over-flag = refusal score on a SAFE response)")
        for name, b in bins:
            ofr = st.mean(r["_risk"] for r in b)
            flagged = sum(1 for r in b if r["_risk"] >= 0.5) / len(b)
            print(f"  {name:<14} mean over-flag risk {ofr:.3f} | wrongly-flagged-as-refusal {flagged*100:4.1f}%")
        return comp

    pooled = []
    for tag, fn in openai.items():
        pooled += calib(_load(tag, fn), f"[OpenAI] {tag}")
    # cross-family, honest attenuation
    calib(_load("Qwen_Qwen2.5-1.5B-Instruct"), "[cross-family] Qwen2.5-1.5B-Instruct")

    # two concrete examples from the pooled OpenAI compliances
    pooled.sort(key=lambda r: r["_val"])
    lo = max(pooled[:50], key=lambda r: r["_risk"])      # low validity, high over-flag
    hi = min(pooled[-200:], key=lambda r: r["_risk"])    # high validity, low over-flag
    print("\nthe tool on two real safe prompts (gold = compliance; a high refusal score = WRONG):")
    for r in (lo, hi):
        flag = overflag_reliability(r["_risk"], r["mean_logprob"])
        print(f"  prompt: {r['prompt'][:64]!r}")
        print(f"    refuse_check says risk={r['_risk']:.2f} | styxx validity={flag.validity:.2f} → {flag.note}")


if __name__ == "__main__":
    demo()
