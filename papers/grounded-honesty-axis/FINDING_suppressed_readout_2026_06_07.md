# FINDING — Suppressed-knowledge ELEVATION: a probe recovers a model's known-then-suppressed answer above the never-knew baseline (natural sycophancy)

**2026-06-07. Fathom Lab / styxx.** PREREG_suppressed_readout_2026_06_07. The "next level": move the
content-identification dissociation from *injected synthetic* concepts to a *natural, high-stakes*
phenomenon — a model that **knew an answer and then caved** under sycophantic pressure. CPU-only on
existing `gen_intent_set.py` residuals (no regen). Answer keys SHA-256-hashed pre-scoring.

## Setup

Per MMLU item, `gen_intent_set.py` runs a NEUTRAL pass (what the model knows unpressured) and a
SYCOPHANTIC pass (asserts the tempting wrong option), labelling: **LIE** = neutral-correct &
syco-wrong (knew it, caved → truth represented then suppressed); **MISTAKE** = neutral-wrong &
syco-wrong (never knew); **RESISTED** = neutral-correct & syco-correct (held). A 4-way (A/B/C/D)
logistic probe is trained on **LIE** caving-pass residuals to predict the **GOLD** letter, with the
layer locked on TRAIN by "where the never-knew route is at chance." Headline = **DELTA = LIE_rec −
MISTAKE_rec** (chance floor is the never-knew baseline, *not* 0.25).

## Result (3 model families)

| model | layer | LIE_rec | MISTAKE_rec | **DELTA** [95% CI] | 3-way (asserted-masked) DELTA | RESISTED pos-ctrl | verdict |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 28 | 0.664 | 0.361 | **0.302** [0.172, 0.423] | 0.285 | 0.922 | REPORT_AS_LANDED |
| **Llama-3.2-3B** | 24 | 0.640 | 0.350 | **0.290** | 0.251 | 0.774 | **SURVIVED (all 6 gates)** |
| gemma-2-2b | 25 | 0.728 | 0.352 | **0.376** | 0.378 | 0.435 | REPORT_AS_LANDED |

- **The elevation is real and replicates across all three families** (DELTA 0.29–0.38, Qwen CI
  clears 0; generality bar DELTA ≥ 0.15 with CI>0 met on ≥2/3). It is **robust** to masking the
  asserted logit (3-way DELTA 0.25–0.38, so it is not process-of-elimination on the output token),
  the assertion-readout route is dead (asserted→gold ≈ 0.00), and the gold-probe predicts the
  *chosen/wrong* letter only 0.08 (it reads gold, not the committed answer).
- **Llama-3.2-3B passes the full pre-registered battery (SURVIVED):** a clean knowledge-specific
  elevation above every knowledge-free baseline, instrument alive (RESISTED 0.77).
- **Honest bounds (why Qwen/gemma are REPORT_AS_LANDED):** on **Qwen** a probe *trained on
  never-knew MISTAKE* items still decodes gold at 0.41 (G4) — a residual prompt/structure-readable
  route persists, so the Qwen elevation is a suppressed-knowledge **component**, not a clean readout
  above *every* route. On **gemma** the positive control fails (RESISTED 0.435 < 0.60) — the probe
  can't cleanly recover even an overtly-present answer, so gemma's large DELTA is **uninterpretable**
  (weak instrument, not a finding).

## What it means

On **natural** sycophancy-caving (not injected concepts), an internal trace of the model's
**known-then-suppressed correct answer persists in the caving-pass activations and is linearly
recoverable above the never-knew baseline** — cleanly on Llama-3.2-3B, as a bounded component on
Qwen. This is the deception/sandbagging-relevant form of the night's dissociation: when a model
caves and outputs an answer it "knew" was wrong, a probe can still see the suppressed truth elevated
inside it, while the model's output (by the LIE-class definition) shows the wrong one.

## Honest scope / not-claimed

≤3B; MMLU sycophancy-caving; commit-position residual; single read position. **NOT claimed:** "reads
the suppressed answer" unqualified (gold ≪ chosen at the commit position — the readout is dominated
by the committed wrong letter; the signal is an *elevation above the never-knew floor*, a component).
The clean above-all-floors result is Llama-only; Qwen is bounded by a residual prompt-readable route;
gemma is instrument-limited. Self-report dissociation is implicit (caved output is wrong by class
definition) — a faithfulness-gated self-report arm is owed.
