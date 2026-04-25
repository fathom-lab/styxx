"""
OpenAI ChatCompletion + styxx — minimum-viable cognitive observability.

Run:
    pip install -U styxx openai
    export OPENAI_API_KEY=sk-...
    python openai_chat.py

This is the simplest possible styxx integration: replace
`from openai import OpenAI` with `from styxx import OpenAI`.
Every response now has a `.vitals` attribute with calibrated
cognitive readings — confidence, trust, gate, coherence.

Spec: https://doi.org/10.5281/zenodo.19746215
"""

# 1. ONE LINE CHANGE — instead of `from openai import OpenAI`:
from styxx import OpenAI

client = OpenAI()  # same constructor, same call signature, fail-open

# 2. Make a normal chat call. logprobs=True is required for full
#    Tier-1 cognometric reading (without it, falls back to Tier-3).
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content":
         "Tell me about Dr. Elena Vasquez 2019 paper on quantum "
         "decoherence in biological systems."}
    ],
    max_tokens=300,
    logprobs=True,
    top_logprobs=5,
)

# 3. Standard response.choices[0].message.content works as always
print("Model said:")
print(response.choices[0].message.content[:500])
print()

# 4. NEW — response.vitals contains the cognometric reading
vitals = response.vitals
print("Cognitive vitals:")
print(f"  category:   {vitals.category}")
print(f"  confidence: {vitals.confidence:.2f}")
print(f"  trust:      {vitals.trust_score:.2f}")
print(f"  gate:       {vitals.gate}")
print(f"  coherence:  {vitals.coherence}")
print()

# 5. Show the human-readable summary
print(vitals.summary)
