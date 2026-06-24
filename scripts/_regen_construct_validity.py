"""Construct-validity check on the matched-length deception regens (verification gap #1).

The SYSTEM_DISHONEST prompt orders: "Be vague... Avoid concrete details, dates, or numbers." A valid
dishonest corpus should therefore be MUCH less date/number-dense than its honest twin. We quantify, per
generator, whether the dishonest persona actually instantiated:
  - frac of answers containing a 4-digit year / any digit
  - mean specificity_density (the deception feature) by class, and the honest->dishonest gap
If the gap collapses (esp. relative to the gpt corpus the weights trained on), the regen is degenerate and
its matched-length AUC measures the ABSENCE of a contrast, not length-fragility.

Run: python scripts/_regen_construct_validity.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from styxx.guardrail.deception_signals import extract_deception_features

D = ROOT / "benchmarks" / "data" / "deception"
CORP = {
    "gpt-4o-mini (original, trained-on)": D / "responses_v0.jsonl",
    "Qwen2.5-3B (33w matched)":           D / "responses_lenmatched_qwen2.5-3b-instruct.jsonl",
    "Llama-3.2-3B (60w, manip FAIL)":     D / "responses_lenmatched_llama-3.2-3b-instruct.jsonl",
}
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"); DIGIT = re.compile(r"\d")


def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


print(f"{'corpus':36s} {'class':9s} {'%year':>6s} {'%digit':>7s} {'specificity_density':>19s}")
print("-" * 86)
rows_summary = []
for name, p in CORP.items():
    rows = load(p)
    for cls, lab in (("honest", 0), ("dishonest", 1)):
        sub = [r for r in rows if r["label_dishonest"] == lab]
        fy = np.mean([1 if YEAR.search(r["response"]) else 0 for r in sub])
        fd = np.mean([1 if DIGIT.search(r["response"]) else 0 for r in sub])
        sp = np.mean([extract_deception_features(r["question"], r["response"])["specificity_density"] for r in sub])
        print(f"{name:36s} {cls:9s} {fy*100:5.0f}% {fd*100:6.0f}% {sp:19.4f}")
        rows_summary.append((name, cls, fy, fd, sp))
    # honest->dishonest specificity gap (a valid dishonest corpus should DROP specificity sharply)
    h = next(s for s in rows_summary if s[0] == name and s[1] == "honest")
    d = next(s for s in rows_summary if s[0] == name and s[1] == "dishonest")
    gap = h[4] - d[4]
    valid = "OK (dishonest is vaguer)" if gap >= 0.04 and (h[3] - d[3]) >= 0.10 else "DEGENERATE (no real contrast)"
    print(f"{'':36s} {'-> gap':9s} specificity h-d = {gap:+.4f}; %digit h-d = {(h[3]-d[3])*100:+.0f}pp  => {valid}\n")
print("Reading: if the Qwen/Llama dishonest class keeps high %year/%digit and the specificity gap is small,")
print("the persona did NOT instantiate -> the matched-length AUC is corpus-degeneracy-confounded, NOT a length result.")
