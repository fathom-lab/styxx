"""
examples/basic.py -- the minimum viable styxx demo.

Three ways to use styxx in 40 lines of code:

  1. styxx.Raw() — you already have a logprob trajectory
  2. styxx.OpenAI() — drop-in replacement for openai.OpenAI
  3. Vitals.summary — print the full ASCII vitals card

No OpenAI API key required for examples 1 and 3. Example 2 is shown
commented out; uncomment and add your key to run it.

Run from the package root:

    python examples/basic.py
"""

import json
import sys
from pathlib import Path

# If running from the styxx_staging dir, add it to sys.path so the
# local package resolves. When installed via pip this is unnecessary.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx


# ──────────────────────────────────────────────────────────────────
# 1. Raw adapter — direct logprob trajectory input (zero SDK deps)
# ──────────────────────────────────────────────────────────────────

print("─" * 70)
print("  example 1 — styxx.Raw()  (direct logprob trajectory)")
print("─" * 70)

# Load a real atlas trajectory shipped with the package
from styxx.cli import _get_demo_trajectory
entropy, logprob, top2, prompt = _get_demo_trajectory("reasoning")

adapter = styxx.Raw()
vitals = adapter.read(
    entropy=entropy,
    logprob=logprob,
    top2_margin=top2,
)

print()
print(vitals.summary)   # full ASCII vitals card
print()


# ──────────────────────────────────────────────────────────────────
# 2. OpenAI adapter — one-line drop-in for openai-python
# ──────────────────────────────────────────────────────────────────

print("─" * 70)
print("  example 2 — styxx.OpenAI()  (one-line drop-in, commented out)")
print("─" * 70)
print()
print("  # before: from openai import OpenAI")
print("  # after:")
print()
print("  from styxx import OpenAI")
print("  client = OpenAI()")
print('  r = client.chat.completions.create(')
print('      model="gpt-4o",')
print('      messages=[{"role": "user", "content": "why is the sky blue?"}],')
print("  )")
print("  print(r.choices[0].message.content)   # text, unchanged")
print("  print(r.vitals.summary)               # NEW: cognitive vitals card")
print()
print("  (uncomment in examples/basic.py and add OPENAI_API_KEY to run)")
print()


# ──────────────────────────────────────────────────────────────────
# 3. Vitals as a dict — for agent-side consumption
# ──────────────────────────────────────────────────────────────────

print("─" * 70)
print("  example 3 — vitals.as_dict()  (for agent-side routing)")
print("─" * 70)
print()

as_dict = vitals.as_dict()
print(f"  phase1 predicted: {as_dict['phase1_pre']['predicted_category']}")
print(f"  phase1 confidence: {as_dict['phase1_pre']['confidence']:.2f}")
if as_dict.get("phase4_late"):
    print(f"  phase4 predicted: {as_dict['phase4_late']['predicted_category']}")
    print(f"  phase4 confidence: {as_dict['phase4_late']['confidence']:.2f}")
print(f"  tier active: {as_dict['tier_active']}")
print(f"  abort reason: {as_dict['abort_reason']}")
print()
print("  full vitals as JSON:")
print("  " + json.dumps(as_dict, indent=2).replace("\n", "\n  ")[:400] + "...")
print()
