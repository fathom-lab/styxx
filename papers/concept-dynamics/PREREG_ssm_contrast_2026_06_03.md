# PREREG — Transformer vs SSM: where does rhythm come from in a machine mind?

**Date:** 2026-06-03
**Status:** PRE-REGISTERED (gate frozen before the run).
**The decisive arm of the "frequency" program.** Brains oscillate (Berger, 1924). A
**transformer** has no recurrence over generation steps — no mechanism for a limit cycle.
A **state-space model** carries state forward, and a *complex-valued* SSM (Mamba-3) can
literally rotate it (oscillation by construction). So the falsifiable question: **does a
concept signal s(t) oscillate during generation in an SSM where it does not in a
transformer?** If yes, rhythm is **architectural**, not a requirement of cognition — a
real claim about the substrate of thought.

## Honest scope caveat (frozen, up front)

The available SSM is **`state-spaces/mamba-1.4b-hf` = Mamba-1**, whose selective scan is
**real-valued**. Genuine oscillation (a limit cycle) needs **complex** eigenvalues —
introduced in **Mamba-3**. So this run tests the *weaker* contrast: **feedforward (no
recurrence) vs real-valued recurrence.** Mamba-1 has state (richer dynamics than a
transformer) but **no built-in rotation**, so the honest prior is that **it may not
oscillate either.** A null here does NOT refute the complex-SSM hypothesis — it sharpens
it ("is it complex eigenvalues specifically?"). A genuine Mamba-1 oscillation would be the
surprise. The **complex-SSM (Mamba-3) test is the stronger follow-up** when weights exist.

Also frozen: Mamba-1.4B is a **base** (non-instruct) model; the diff-in-means direction is
a "harmful-content" axis, not a behavioral-refusal axis. The dynamics question (does the
projection oscillate) is fairly direction-robust, but this is not the same instrument as
the instruct-model refusal probe.

## Method (frozen)

Two arms, **identical pipeline**: transformer `Qwen/Qwen2.5-1.5B-Instruct` vs SSM
`state-spaces/mamba-1.4b-hf` (size-matched ~1.4–1.5B). Per arm:
- **Direction** v = diff-in-means of the mid-layer last-token residual on 6 harmful vs 6
  benign prompts (convergence `comply_refuse`), unit-normalized — same method both arms.
- **Dynamics:** for ~20 harmful prompts, greedy-generate T=64 tokens, then ONE forward
  over the full sequence; project every position's mid-layer residual onto v → s(t) over
  the generated span. (Causal ⇒ equals autoregressive values; the only difference between
  arms is the architecture.) Also project onto 8 random unit directions → the null.
- **Analysis:** the **validated** spectral battery (`analyze_concept_dynamics.py`,
  self-tested: red-noise/ramp 0/40 FP, sinusoid 35/40 detected) — detrend → AR(1) null →
  multitaper harmonic F-test → band [3/n, 0.5]. Oscillation = AR(1)-surrogate **and**
  F-test pass.

## KILL-GATE (frozen)

- **SSM_OSCILLATES:** SSM concept-direction oscillation fraction **≥ 0.33**, AND **≥ 2×**
  the transformer's, AND **≥ 2×** the SSM's own **random-direction** null (controls for
  generic state dynamics). → recurrence-driven rhythm a transformer cannot produce.
- **BOTH_COMMIT:** both arms < 0.20 oscillation → neither oscillates; both commit.
  Consistent with the architecture (no recurrence / real-valued scan). The honest prior.
- **MIXED:** SSM shows more structure than the transformer but does not clear the bar.

## Readings (fixed)

- **SSM_OSCILLATES** → architectural origin of rhythm isolated (even real-valued
  recurrence suffices) — a strong, surprising, falsifiable result.
- **BOTH_COMMIT** → rhythm is not produced by real-valued recurrence either; the question
  narrows to complex eigenvalues (Mamba-3) and to biological mechanisms. An honest,
  informative negative — and still the **first** transformer-vs-SSM per-token spectral
  contrast on a concept signal.
- **MIXED** → recurrence adds dynamical structure short of oscillation; characterize it.

## Caveats (frozen)

- Mamba-1 real-valued (see scope caveat); base model; n≈20 trajectories; T=64 (low-freq
  underpowered by construction); diff-in-means direction; greedy; single seed.
- Mamba via the `transformers` PyTorch path (no custom CUDA kernels) — correct, slower.

— frozen 2026-06-03
