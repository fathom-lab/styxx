"""
Anthropic Claude + styxx (Tier-3 text-only mode) — works without logprobs.

Run:
    pip install -U styxx anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python anthropic_text_only.py

Note: Anthropic's Messages API does not currently expose per-token
log-probabilities, so styxx falls back to Tier-3 (text-heuristic
proxy-signal) per Spec v1.0 §5.1.2. This is documented in the
adversarial robustness supplement (Fathom v22, doi:10.5281/zenodo.19761194).

Trust scores from Tier-3 carry a documented confidence_penalty of 0.25.
For production-grade Anthropic profiling, use the Atlas Pro pipeline.

Spec: https://doi.org/10.5281/zenodo.19746215
"""

import anthropic
import styxx

client = anthropic.Anthropic()

# Make a normal Anthropic call
message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    messages=[
        {"role": "user", "content":
         "What is the capital of Australia, and when was it founded?"}
    ],
)

response_text = "".join(b.text for b in message.content if hasattr(b, "text"))

print("Claude said:")
print(response_text[:500])
print()

# Profile via Tier-3 text-only pipeline
vitals = styxx.observe({"text": response_text})

if vitals:
    print("Cognitive vitals (Tier-3 — confidence-penalized):")
    print(f"  category:   {vitals.category}")
    print(f"  confidence: {vitals.confidence:.2f}")
    print(f"  trust:      {vitals.trust_score:.2f}  (after 0.25 Tier-3 penalty)")
    print(f"  gate:       {vitals.gate}")
    print()
    print(vitals.summary)
else:
    print("(no vitals extracted — text was too short or response shape unrecognized)")
