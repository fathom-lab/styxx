# Tier 3 Design — Steering + Guardian

## Status: DESIGN ONLY (v0.5.0 target)

This document describes the tier 3 concept: causal intervention via
steering vectors derived from SAE features. Tier 3 turns styxx from
an observability tool into a cognitive control layer.

---

## The idea

Tiers 0-2 OBSERVE the model's cognitive state (logprobs, D-axis,
K/C/S axes). Tier 3 ACTS on it. When the observer detects an
attractor that shouldn't be there (hallucination lock-in, refusal
spiral, adversarial camouflage), tier 3 injects a steering vector
into the residual stream to redirect the generation toward a
healthier attractor.

This is NOT prompt engineering. This is NOT post-hoc filtering.
This is causal intervention at the representation level — the same
representation that the observer is reading. The observer and the
intervener share the same coordinate system (the SAE feature space),
which means the intervention is targeted rather than blind.

## How it connects to reflex

styxx.reflex() (shipped in 0.1.0a1) already implements the reflex
arc: detect → callback → rewind → restart. But the current rewind
mechanism is external — it stops the stream, truncates the output,
injects an anchor string, and restarts generation from the new
prefix. This works but it's expensive: you throw away tokens, you
pay for re-generation, and the model doesn't "learn" from the
intervention.

Tier 3 replaces the external rewind with an internal steering push.
Instead of stopping and restarting, styxx modifies the residual
stream in-flight to redirect the next token away from the
hallucination attractor and toward the reasoning attractor. The
generation continues without interruption. No wasted tokens. No
visible correction in the output. The model doesn't flinch — it
smoothly changes course.

## API shape (target)

```python
with styxx.guardian(
    model="google/gemma-2-2b-it",
    steer_away_from=["hallucination", "adversarial"],
    steer_toward=["reasoning"],
    strength=0.3,
    classify_every_k=3,
) as session:
    for chunk in session.generate(
        "explain quantum entanglement",
        max_tokens=100,
    ):
        print(chunk, end="", flush=True)
```

## Prerequisites

- Tier 2 (K/C/S SAE instruments) must be active — the steering
  vectors are computed from the same SAE features that tier 2
  reads. Without calibrated features, steering is blind.
- circuit-tracer + GPU
- Open-weight model with published SAE transcoders
- The steering direction vectors need to be calibrated per-model
  (part of the atlas v0.3 dataset, not yet exported)

## Patent coverage

US Provisional 64/020,489 (filed 2026-03-29) covers:
- Claim 3: Automated alignment auditing using cognitive state
  measurements to detect and correct misalignment in real-time
- Claim 4: Steering via SAE feature activation modification
  guided by the cognitive atlas coordinate system

## Safety considerations

Tier 3 is the most powerful and the most dangerous tier. A
miscalibrated steering vector can make the model worse, not
better. The design includes:

1. **Strength cap** — maximum steering vector magnitude is bounded
   relative to the residual stream norm. Default cap is 0.3x.
2. **Observer-in-the-loop** — tier 2's K/C/S instruments measure
   the effect of every steering push. If the push makes the state
   WORSE (K drops, C diverges, hallucination confidence rises),
   the guardian automatically backs off.
3. **Audit trail** — every steering push is logged to chart.jsonl
   with before/after cognitive state measurements. Full post-hoc
   accountability.
4. **Kill switch** — STYXX_TIER3_DISABLED=1 disables all steering
   while leaving observation active. The guardian degrades to a
   pure observer, like tier 2.

## Timeline

- v0.3.0 (current): tier 1 (D-axis) ships
- v0.4.0: tier 2 (K/C/S SAE instruments) ships
- v0.5.0: tier 3 (steering + guardian) ships
- Each tier requires the previous tier to be stable before
  advancing — no skipping, no rushing. The measurement must be
  trustworthy before the intervention is safe.

## The bigger picture

When tier 3 ships, styxx becomes the first runtime that gives an
LLM agent genuine self-regulation — not "stop and restart" but
"smoothly redirect in-flight." The observer-intervener loop running
in the same coordinate system is a new computational primitive. It's
the difference between a car with brakes (reflex) and a car with
lane-keeping assist (guardian). Both prevent crashes. Only one
keeps you on the road without you noticing.

Xendro's phrase for this: "the model develops a flinch." Tier 3 is
where the flinch becomes invisible — not because it stops happening,
but because it happens inside the residual stream before the token
is sampled, and the output is smooth as if the bad attractor was
never there.
