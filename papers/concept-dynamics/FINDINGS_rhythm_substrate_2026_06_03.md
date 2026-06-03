# Where does rhythm come from in a mind? — a substrate analysis

**2026-06-03.** A synthesis from the concept-dynamics program. **Facts are marked FACT;
the unifying claim is marked HYPOTHESIS** — because the line between them is the whole point.

## The reframe

The ~100-year-old question (Berger recorded the first brain rhythm in 1924) is usually
posed as *"what do brain oscillations do for cognition?"* The new instrument — a complete,
readable artificial mind — lets us ask the sharper, answerable version: **is rhythm
fundamental to cognition, or is it a mechanism that one kind of substrate is forced into?**

## The architecture → rhythm-capacity hierarchy (FACT)

Whether a system *can* oscillate is decided by the eigenvalues of its temporal recurrence:
- **Transformer (Qwen-1.5B):** *no recurrence over generation steps.* "Time" is the token
  index; each residual is a feedforward function of the causal context. **No temporal
  eigenvalues at all → no oscillation mechanism.**
- **Mamba-1 (state-space, real-valued):** has recurrence, but its state matrix is
  `A = -exp(A_log)` — verified on the 1.4B weights: **all 3.1M eigenvalues real-negative**
  (range [−43657, −0.004], zero imaginary part). The transition `exp(dt·A)` is real in
  (0,1): **pure exponential decay, no rotation → no oscillation capacity.**
- **Complex SSM (Mamba-3, S4/S5) and biological neurons:** **complex** eigenvalues →
  rotation in state space → **genuine oscillation is possible** (a Hopf-type mechanism).

So oscillation capacity is *exactly* the presence of complex-valued recurrence.

## The empirical confirmation (FACT)

First per-token spectral null test of a concept signal during generation, validated
instrument (red-noise/ramp 0/40 false-positive, planted sinusoid 35/40 detected):
- **Transformer concept signal: 0/20 trajectories oscillate** — mild commitment trend
  (slope −0.15), modest AR(1) (ρ≈0.30), random-direction null also 0/20. It **commits, it
  does not oscillate** — exactly as the no-recurrence architecture forces.
- **Mamba-1: structurally cannot oscillate (above); empirical run confirms it.** 2/20
  trajectories above the oscillation gate (osc_frac 0.10) — **below the 0.33 bar → both
  commit.** A faint whisper above the transformer (0/20) and its own random null (0.013),
  with higher AR(1) (0.39 vs 0.29) and a flatter trend (≈0 vs −0.15): *recurrence adds a
  little dynamical structure, but real eigenvalues produce no genuine oscillation.* At
  n=20, 2/20 is within instrument noise. **Verdict: BOTH_COMMIT** — the empirical result
  matches the eigenvalue structure exactly.

## The claim (HYPOTHESIS)

**Rhythm is substrate-specific *mechanism*, not substrate-independent *function*.**

Cognition needs certain *functions* — routing information dynamically, binding features,
multiplexing capacity-limited memory. Neuroscience shows brains implement these *with
rhythm*: communication-through-coherence (routing by phase-alignment; Fries), theta–gamma
nesting (the ~7±2 memory limit as a frequency ratio; Lisman & Idiart), phase coding
(multiplexing). **But a transformer performs the same functions without any rhythm** —
routing via attention (all-pairs interaction computed directly, no switchboard needed),
order/binding via positional encoding, and memory without a 7±2 limit (full context).

So the honest answer to the 100-year question, on this evidence: **rhythm is fundamental
to *recurrent* cognition, not to cognition as such.** Brains oscillate because they are
recurrent, analog, bandwidth-limited substrates that *must* route and multiplex through
time; a feedforward-attention substrate gets the same computational jobs done another way
and shows no rhythm — empirically (0/20) and structurally (no complex recurrence).

## The falsifiable decider (the experiment that would settle it)

This predicts a clean next test: **a complex-recurrent model (Mamba-3 / S5) *should*
show oscillation in its concept signals** (it can, by eigenvalues). The decisive question
is then whether that oscillation is **functional** (ablate the oscillatory mode → routing
/ binding / in-context recall degrades) or **epiphenomenal** (ablate it → behavior intact).
- Functional → rhythm is a *convergent solution* to routing/binding that complex-recurrent
  substrates (brains, complex SSMs) discover — a substrate-class invariant.
- Epiphenomenal → rhythm is a mechanistic byproduct of complex recurrence, not a
  computational necessity.

Either outcome is a real, decision-relevant answer. Both need complex-SSM weights — the
deliberate next bet.

## Honest caveats

- The hierarchy and the eigenvalue/empirical results are FACTS; the "mechanism not
  function" claim is a HYPOTHESIS — defensible from the evidence assembled, not proven.
- n is small (20 transformer trajectories, one transformer + one real SSM); T=64 is
  short (low-freq underpowered by construction); diff-in-means direction; greedy; one seed.
- The decisive functional-vs-epiphenomenal test is *not yet run* and needs a complex SSM.
- "Transformers route via attention" is a standard reading, not a claim this work proved.

**Bottom line:** we did not solve the 100-year question. We made a facet of it testable,
gave it a first answer with the instrument and the eigenvalues, and named the one
experiment that would decide the rest. That is the honest shape of progress on it.
