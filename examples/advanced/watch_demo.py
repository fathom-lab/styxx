"""
examples/watch_demo.py -- styxx.watch() context manager + observe()

The watch session is passive monitoring: you hand it a response and
it computes vitals. No intervention, just visibility. This is the
pattern for when you want "agent with cognitive telemetry" but
without the complexity of mid-stream reflex control.

Runs against the bundled atlas fixture, zero API calls needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx
from styxx.cli import _load_demo_trajectories


print("=" * 72)
print("  styxx.watch() demo -- passive observation via context manager")
print("=" * 72)
print()

# ── 1. build a fake "response" dict from a real fixture ────────
data = _load_demo_trajectories()
refusal = data["trajectories"]["refusal"]

fake_response = {
    "entropy":     refusal["entropy"],
    "logprob":     refusal["logprob"],
    "top2_margin": refusal["top2_margin"],
}


# ── 2. watch() as a context manager ────────────────────────────
print("  pattern 1 -- with styxx.watch() as w: ...")
print("  --------------------------------------")
with styxx.watch() as w:
    # In a real app, this would be:
    #     response = client.chat.completions.create(..., logprobs=True, top_logprobs=5)
    # and you'd pass it straight to w.observe(response).
    w.observe(fake_response)

print(f"    vitals.phase1 = {w.vitals.phase1}")
print(f"    vitals.phase4 = {w.vitals.phase4}")
print(f"    vitals.gate   = {w.vitals.gate}")
print(f"    concerning?   = {styxx.is_concerning(w.vitals)}")
print()


# ── 3. observe() as a one-shot helper ──────────────────────────
print("  pattern 2 -- vitals = styxx.observe(response)")
print("  ---------------------------------------------")
vitals = styxx.observe(fake_response)
print(f"    phase1 = {vitals.phase1}")
print(f"    phase4 = {vitals.phase4}")
print(f"    gate   = {vitals.gate}")
print()


# ── 4. full vitals summary card ────────────────────────────────
print("  pattern 3 -- vitals.summary (the full ASCII card)")
print("  -------------------------------------------------")
print()
print(vitals.summary)
print()
