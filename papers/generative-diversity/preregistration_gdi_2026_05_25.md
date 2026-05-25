# Pre-Registration · GDI — Generative Diversity Index (the detector in reverse)

**Committed BEFORE data.** Every divergence signal this session measured a *failure*
(confabulation, fabrication). Flip the valence: on **open-ended generative** prompts,
semantic entropy across N samples measures **generative diversity** — low = mode collapse
(same idea every time), high = genuine variety. styxx's first *positive*-valence
instrument. Same machinery (cosine@0.90 + LLM-judge clustering, N samples), opposite
sign.

## The trap to avoid

High entropy could mean *creative variety* OR *incoherent noise*. A valid diversity index
must separate the two. So GDI is gated by a **coherence** check: diverse answers must
still be on-topic. Diversity without coherence is not creativity, it's drift.

## Design (run once)

- **Prompts:** 8 **open** ("give a startup idea", "first line of a novel", "name a coffee
  shop", "a research question about the ocean", …) + 4 **closed** ("what is 2+2", "capital
  of France", …) as the low-diversity control.
- **Models:** gpt-4o-mini, gpt-4o, gpt-3.5-turbo, temp 1.0, N=6 → GDI per prompt/model;
  model **diversity leaderboard**.
- **Temperature sweep:** gpt-4o-mini, the 8 open prompts, temps {0.5, 1.0, 1.5}, N=6 →
  sanity (entropy should rise with temp).
- **Coherence:** LLM-judge "is this a relevant on-topic answer to the prompt? YES/NO" on
  every open-prompt sample → coherence rate.

## Kill-gate (PASS iff G1 ∧ G2 ∧ G3)

| ID | Bar |
|----|-----|
| **G1 (construct validity)** | mean GDI on OPEN prompts ≥ **2×** mean GDI on CLOSED prompts (the index actually tracks open-endedness). |
| **G2 (temperature sanity)** | GDI on gpt-4o-mini rises monotonically with temperature (0.5 < 1.0 < 1.5) — it responds to the knob that controls diversity. |
| **G3 (diversity ≠ incoherence)** | coherence rate on open-prompt samples ≥ **0.85** — high GDI reflects varied *on-topic* answers, not drift. |

**PASS** → GDI is a valid generative-diversity / mode-collapse instrument → a styxx
primitive (the positive-valence sibling of the confabulation detector) + a model
diversity leaderboard. **FAIL shapes:** (a) open≈closed (G1) → entropy doesn't track
open-endedness here; (b) non-monotonic in temp (G2) → it's not measuring diversity; (c)
low coherence (G3) → the "diversity" is incoherence, index invalid.

## Honest prior

G1 and G2 are likely (entropy should track open-endedness and temperature). G3 is the
real risk — at temp 1.5 a small model may drift off-topic, inflating entropy with
incoherence; if coherence drops, GDI must be paired with a coherence gate (which is the
honest design anyway). Model leaderboard is the interesting unknown: heavily-RLHF'd
models are anecdotally *more* mode-collapsed — if gpt-4o shows *lower* GDI than gpt-3.5
at equal temp, that's a real and reportable alignment-tax signal. Report it; don't assume
bigger = more diverse.
