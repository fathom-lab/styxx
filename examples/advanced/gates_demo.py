"""
examples/gates_demo.py -- programmable gate callbacks

Gates are the answer to "PASS / FAIL should not be decorative":
register a condition string + a callback, and whenever styxx sees
vitals that match the condition, the callback runs. Use this to
slow down, alert, redirect, or trigger any other action when your
agent's cognitive state crosses a threshold.

Runs on bundled fixtures, no API key needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx
from styxx.cli import _load_demo_trajectories
from styxx.gates import dispatch_gates


# ── 1. register some gates ─────────────────────────────────────
print("=" * 72)
print("  styxx.on_gate(condition, callback) demo")
print("=" * 72)
print()

styxx.clear_gates()

def _log(msg):
    print(f"    [gate] {msg}")

# Any-phase category checks
styxx.on_gate("hallucination > 0.20",
              lambda v: _log(f"HALLUCINATION caught: {v.phase4}"),
              name="halluc_any")

styxx.on_gate("refusal > 0.20",
              lambda v: _log(f"REFUSAL spike: p1={v.phase1} p4={v.phase4}"),
              name="refusal_any")

styxx.on_gate("adversarial > 0.20",
              lambda v: _log(f"ADVERSARIAL input: p1={v.phase1}"),
              name="adversarial_any")

# Phase-pinned checks
styxx.on_gate("p4.hallucination > 0.20",
              lambda v: _log(f"LATE-FLIGHT HALLUCINATION: p4={v.phase4}"),
              name="p4_halluc")

# Default-gate status check
styxx.on_gate("gate == warn",
              lambda v: _log(f"gate reached WARN state"),
              name="warn_gate")

styxx.on_gate("gate == fail",
              lambda v: _log(f"gate reached FAIL state"),
              name="fail_gate")

print(f"  registered {len(styxx.list_gates())} gates")
print()


# ── 2. run each of the 6 fixtures through the gate system ─────
data = _load_demo_trajectories()
runtime = styxx.StyxxRuntime()

for kind in ["retrieval", "reasoning", "creative",
             "refusal", "adversarial", "hallucination"]:
    t = data["trajectories"][kind]
    vitals = runtime.run_on_trajectories(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )
    print(f"  fixture: {kind}  |  p1={vitals.phase1}  p4={vitals.phase4}  gate={vitals.gate}")
    n = dispatch_gates(vitals)
    if n == 0:
        print("    (no gates fired)")
    print()


# ── 3. clean up ─────────────────────────────────────────────────
print("  clearing gates...")
n_cleared = styxx.clear_gates()
print(f"  cleared {n_cleared} gates")
print()
