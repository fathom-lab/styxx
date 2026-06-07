# FINDING — The Inaccessible Thought: a planted concept is legible to a lens but not to the mind

**2026-06-06. Fathom Lab / styxx. PREREG_introspection_fc_2026_06_06.md.**
Verdict: **REPORT_AS_LANDED_null** on grounded introspective access — *with the apparatus
validated* (the abort gate passed on every model). Answer keys SHA-256-hashed pre-scoring.

## Setup

A genuine concept direction (paired template-cancelling diff-of-means, **steering-validated** —
injecting it measurably moves free generation toward the concept) is covertly injected into the
residual stream (rms all-position hook, 0.60-depth layer, α=10; α=16 robustness). The model is
then asked, via a **forced choice that the RLHF persona cannot refuse**, to identify which of 8
concepts was injected — answering with an **arbitrary letter code** so that a raw logit-lens on
the injected vector cannot win.

## Result (3 instruct models; chance = 0.125 for the 8-way symbol code)

| model | symbol-code inject | random | **prime (abort)** | concept-**word** (lens) | clean-readpos | wrong-layer |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.083 | 0.083 | **1.00** | 0.333 | 0.104 | 0.021 |
| Llama-3.2-3B-Instruct | 0.167 | 0.167 | **0.98** | **0.958** | 0.167 | 0.146 |
| gemma-2-2b-it | 0.104 | 0.083 | **1.00** | **1.000** | 0.083 | 0.229 |
| Qwen2.5-3B-Instruct (α=16) | 0.146 | 0.125 | **1.00** | 0.313 | — | 0.042 |

2-AFC (chance 0.5): inject 0.50–0.65, prime 1.00.

## The dissociation (the actual finding)

Every gate tells the same story, and the controls make it airtight:

- **The forced channel works perfectly when the answer is handed over** (prime ≈ 1.00). So a null
  is *not* a dead channel or a persona artifact — the apparatus is validated. This is what v1's
  free-response framing could never establish (there the persona just said "NO").
- **The injected thought is present in the activations**: reading over the concept *words*
  (a logit-lens on the injection) recovers it — **model-dependent: ~0.33 on Qwen-3B (weak,
  ~2.7× chance), ~0.96–1.00 on Llama-3B/gemma-2b.** The "up to 100%" rides Llama/gemma.
- **Yet the model cannot use it.** Asked to map that same internal state to an arbitrary letter,
  it scores **at chance** (0.08–0.17), at **every dose**, on **every model**.
- **The symbol-code signal is not a live lens**: it stays at chance even at the at-injection
  position, while the concept-word lens dissociates from it. Wrong-layer injection collapses to
  chance on Qwen/Llama (0.02 / 0.15) and stays *below the 0.30 gate but above chance* on gemma
  (0.23) — i.e. it passed the gate, it did not fully collapse on gemma.

**A concept planted in a 2–3B model is read by an external probe — on Qwen-3B the processed-state
probe (read layer 31, never directly injected) decodes it at 1.00 (clean control 0.00,
perm-null 0.10) — while the mind that holds it identifies it at chance.** The white-box readout
that hits 1.00 is the *processed-state probe* (validated on Qwen at one saturating dose, α=10,
where skip-last clean-read-position and at-position both saturate at 1.00 — a ceiling, not a graded
propagation curve); the *logit-lens* readout is the model-dependent 0.33–1.00 above.

## Why it matters

1. **It draws the behavior-vs-mind line with teeth.** styxx's thesis is "we measure behavior,
   never minds." This is the sharpest demonstration yet: the *same* internal state is present
   (lens = 1.0) and inaccessible to self-report (forced choice = chance), with persona,
   generic-perturbation, logit-lens, live-lens, and layer confounds all controlled.
2. **It bounds a load-bearing assumption — for IDENTIFICATION specifically.** A model cannot
   *name* (forced-choice identify) an injected concept even though an external probe decodes it.
   **Important limit (from our own same-night dose/tamper files):** this is about *content
   identification*, NOT detection-of-presence. For merely *detecting that something is off*, a
   forced YES/NO self-report works fine at the deployment dose (self-report DETECTION AUROC ≈ 0.99
   at α=8, 1.00 at α=16) — there the white-box probe has **no** advantage. The self-report failure
   is specific to *what* was injected, and to *subtle* doses (α≤2). So the honest claim is narrow:
   "models can't introspectively identify their injected content," **not** a blanket "probe-based
   monitors beat self-report monitors."
3. **It is a controlled open-model counterpoint** to the frontier-only "emergent introspection"
   claim, with the random-direction, text-priming, symbol-code, clean-read-position, and
   wrong-layer controls that prior demonstrations lacked — and an abort gate proving the channel
   was alive.

## Honest scope

≤3B open instruct models; one injected linear concept direction; 8-concept set; forced-choice
read-out through the LM head. A null here does **not** prove introspection is impossible at scale,
nor that the model "has no inner state" — only that, at 2–3B, an injected concept is **not
forced-choosable through the model's own output**, while being trivially lens-readable. This
measures a read-out channel, never a mind.

## Companion

This is the self-legibility half of the night's program. The mutual-legibility half
(`papers/disjoint-worlds/run_real_legibility.py`) asks the dual question across minds: can a
concept in one model be aligned to another with zero paired data — and, critically, validates the
aligner with a positive-control calibration the prior run lacked.
