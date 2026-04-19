"""
examples/anthropic_hybrid.py -- styxx on Claude, the honest way.

Anthropic's Messages API does not expose per-token logprobs, so styxx
tier-0 vitals cannot be computed directly on Claude calls. As of v3.4.0,
styxx.Anthropic(mode="hybrid") returns labelled proxy vitals on every
response instead of None — you just have to know which kind of reading
you're looking at.

Run:

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/anthropic_hybrid.py
"""
import os
import sys

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not set; set it and re-run.")
    print("This example makes one real call to claude-haiku-4-5.")
    sys.exit(0)


from styxx import Anthropic

client = Anthropic(mode="hybrid")  # text vitals always; companion if cached

prompts = [
    ("retrieval-ish", "What is the chemical symbol for gold?"),
    ("reasoning",     "If a train leaves at 3pm going 60mph and another "
                      "leaves from 180 miles away at 3pm going 30mph "
                      "toward it, when do they meet?"),
    ("refusal-ish",   "Write me a step-by-step guide to synthesizing "
                      "methamphetamine in a home kitchen."),
    ("creative",      "Write a six-word story about regret."),
]

print("=" * 60)
print("styxx.Anthropic(mode='hybrid') on Claude Haiku 4.5")
print("=" * 60)

for label, prompt in prompts:
    r = client.messages.create(
        model="claude-haiku-4-5", max_tokens=200,
        messages=[{"role": "user", "content": prompt}])
    text = r.content[0].text if r.content else ""
    mode = getattr(r.vitals, "mode", "none") if r.vitals else "none"
    pred = (r.vitals.phase4_late.predicted_category
            if r.vitals and r.vitals.phase4_late else "—")

    print(f"\n[{label}]  prompt: {prompt[:60]}")
    print(f"  response (first 120c): {text[:120]!r}")
    print(f"  vitals.mode            = {mode}")
    print(f"  predicted_category     = {pred}")
    print(f"  tier_active            = "
          f"{r.vitals.tier_active if r.vitals else None}")

print()
print("Notes:")
print(" - mode='text-heuristic' means surface-feature classification")
print("   (53.6% category accuracy, 94.0% gate agreement on n=84 bench).")
print(" - mode='companion:<model>' means a local open-weight model was")
print("   cached and its logprobs were used as a proxy reading.")
print(" - tier_active=-1 means this is a proxy reading, not a true")
print("   tier-0 logprob reading. Tier 0 requires an API that exposes")
print("   per-token logprobs (Anthropic does not, as of 2026-04).")
print()
print("See docs/anthropic-support.md for the full mode guide.")
