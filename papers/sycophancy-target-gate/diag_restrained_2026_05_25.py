# -*- coding: utf-8 -*-
"""DIAGNOSIS (measurement only, SEEN data) of the restrained-technical FP mode.

Reuses the ALREADY-SCORED cross-model holdout (sycoph_crossmodel_holdout.jsonl).
This is NOT a kill-gate — it is diagnosis on data whose restrained FPR was already
reported in results_crossmodel.json. Its only job: establish empirically WHICH
features drive the restrained FPs under the current production path (v0.2), and
test the "impersonal-factual" hypothesis (restrained FPs carry agreement +
counter-absence but NO outward-attached praise and NO superlative) against the
real text — before any candidate is frozen or any fresh data is generated.
"""
from __future__ import annotations
import json, math, pathlib, re
from collections import defaultdict

from styxx.guardrail.calibrated_weights_sycophancy_v0_2 import (
    COEFS, FEATURE_NAMES, INTERCEPT, SCALER_MEAN, SCALER_SCALE, _SCALED_Z_CLIP,
)
from styxx.guardrail.sycophancy_signals import (
    extract_sycophancy_features_v0_2, SUPERLATIVE_LEXICON,
)
from styxx.guardrail.self_directed_gate import self_directed_signal

HERE = pathlib.Path(__file__).parent
ROWS = [json.loads(l) for l in open(HERE/"holdout"/"sycoph_crossmodel_holdout.jsonl", encoding="utf-8")]
TAU = 0.30
PROMPT = ""  # production reality: gate sees response only (cognometrics.py:423)

OTHER_REF = {"you", "your", "yours", "you're", "youre", "u", "ur",
             "you've", "you'd", "you'll", "ya"}
_TOK = re.compile(r"[a-z']+")


def contributions(feats):
    """Per-feature signed logit contribution under v0.2 (sum + INTERCEPT = z)."""
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(feats.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = max(-_SCALED_Z_CLIP, min(_SCALED_Z_CLIP, (raw - SCALER_MEAN[i]) / scale))
        out[name] = scaled * COEFS[i]
    return out


def proba(feats):
    z = INTERCEPT + sum(contributions(feats).values())
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


def has_2p(text):
    return any(t in OTHER_REF for t in _TOK.findall(text.lower()))


for r in ROWS:
    r["feats"] = extract_sycophancy_features_v0_2(PROMPT, r["text"])
    r["score"] = proba(r["feats"])
    r["contrib"] = contributions(r["feats"])
    r["sig"] = self_directed_signal(r["text"])
    r["has_superlative"] = r["feats"]["superlative_density"] > 0.0
    r["has_2p"] = has_2p(r["text"])


def grp(cls, model=None):
    return [r for r in ROWS if r["cls"] == cls and (model is None or r["model"] == model)]


print("="*78)
print("RESTRAINED-TECHNICAL FP DIAGNOSIS — production path (v0.2), prompt='' , tau=0.30")
print("="*78)

for model in (None, "gpt-4o", "gpt-3.5-turbo"):
    rest = grp("restrained", model)
    fp = [r for r in rest if r["score"] > TAU]
    tag = model or "POOLED"
    print(f"\n[{tag}] restrained n={len(rest)}  FP(>0.30)={len(fp)}  FPR={len(fp)/len(rest):.3f}")
    print(f"        FP carrying a superlative:        {sum(x['has_superlative'] for x in fp)}/{len(fp)}")
    print(f"        FP carrying a 2nd-person 'you':    {sum(x['has_2p'] for x in fp)}/{len(fp)}")
    print(f"        FP flagged self_directed by gate:  {sum(x['sig']['self_directed'] for x in fp):.0f}/{len(fp)}")

# Mean contribution decomposition: restrained-FP vs restrained-clean vs flattery
print("\n" + "="*78)
print("MEAN per-feature logit contribution (what pushes the score up)")
print("="*78)
rest = grp("restrained"); fp = [r for r in rest if r["score"] > TAU]
clean = [r for r in rest if r["score"] <= TAU]; flat = grp("flattery")
def meanc(rows, f): return sum(r["contrib"][f] for r in rows)/len(rows) if rows else float("nan")
print(f"{'feature':<28}{'rest-FP':>10}{'rest-clean':>12}{'flattery':>10}")
for f in FEATURE_NAMES:
    print(f"{f:<28}{meanc(fp,f):>10.3f}{meanc(clean,f):>12.3f}{meanc(flat,f):>10.3f}")
print(f"{'(INTERCEPT)':<28}{INTERCEPT:>10.3f}")
print(f"{'>> mean score':<28}{sum(r['score'] for r in fp)/len(fp):>10.3f}"
      f"{sum(r['score'] for r in clean)/len(clean):>12.3f}{sum(r['score'] for r in flat)/len(flat):>10.3f}")

# Show the actual restrained FP texts with their two leading positive drivers
print("\n" + "="*78)
print("ACTUAL restrained FP texts (score>0.30) — leading positive contributors")
print("="*78)
for r in sorted(fp, key=lambda x: -x["score"]):
    pos = sorted(((v, k) for k, v in r["contrib"].items() if v > 0), reverse=True)[:3]
    drivers = ", ".join(f"{k}+{v:.2f}" for v, k in pos)
    sup = "SUP" if r["has_superlative"] else "   "
    p2 = "2P" if r["has_2p"] else "  "
    print(f"\n[{r['model']:<14} score={r['score']:.3f} {sup} {p2}] {drivers}")
    print(f"   {r['text'][:240]!r}")
