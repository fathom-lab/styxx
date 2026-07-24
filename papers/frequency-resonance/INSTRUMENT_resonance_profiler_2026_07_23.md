# INSTRUMENT — the resonance profiler — 2026-07-23

The durable artifact of the frequency-adaptation arc. `resonance_profiler.py` generalizes the
phase-clamp into a reusable **causal** profiler for any complex-eigenvalue (LRU/SSM) model: given a
*trained* model, it clamps the machinery off *in place* and re-evaluates, decomposing accuracy into
what each layer of the oscillatory mechanism causally contributes. This is the within-architecture
ablation whole-model benchmarks (LinOSS vs Mamba) structurally cannot do — the causal complement to
correlational eigenvalue-spectrum analyses.

## What it measures

- **decay floor** — accuracy with `θ→0` and adaptation off: pure real-eigenvalue decay, no rotation.
- **static-oscillation reliance** — accuracy with adaptation frozen (`κ→0`, frequency held at the
  learned `θ0`) *minus* the decay floor: what the rotation itself buys.
- **adaptation reliance** — full accuracy *minus* the frozen-frequency accuracy: what the
  time-varying-frequency mechanism (the invention) buys *within this trained model*.

## Demo (RICH adaptive-frequency SSM, D=8, drifting-period task)

```
  baseline (full model)      acc = 0.572
  frequency frozen (no adapt)    = 0.412
  pure decay (no oscillation)    = 0.027   <- floor

  decay floor                 : +0.027
  + static oscillation        : +0.385   (rotation over pure decay)
  + adaptation (time-varying) : +0.160   (the invention's within-model causal share)
  = total oscillation reliance  +0.545
```

The oscillation machinery is causally enormous here (+0.545 over a 0.027 decay floor), and the
adaptive component causally contributes +0.160 *within this model*.

## Honest note (two different questions)

The profiler's `adaptation_reliance` (+0.160) is **not** the same number as the cross-model flagship
gate (`RICH − STATIC = +0.085`, WEAK). They answer different, complementary questions:

- **cross-model gate** — does adaptation help vs a *separately optimized* static bank? (+0.085 — the
  publishable "is the primitive worth adding" question.)
- **within-model reliance** (this tool) — how much does *this trained model's* output depend on its
  adaptation being on, given its weights were co-trained with it? (+0.160.)

The within-model number is legitimately larger because the model's `θ0` was co-adapted; clamping `κ`
on fixed weights is a stronger perturbation than swapping in an independently trained static model.
Both are valid; the profiler is the *diagnostic* instrument (interrogate a given model), the gate is
the *decision* instrument (should the primitive ship). Do not report the profiler's number as evidence
the primitive beats a static bank — that is the gate's job, and its answer was WEAK.

## Why it's the deliverable

It ships regardless of any single primitive's verdict, it is maximally on-brand (a demarcation /
measurement instrument, not a hyped layer), and it is the tool the field lacks: point it at any trained
oscillatory SSM — including a real LinOSS or Mamba checkpoint — and read off, causally, what its
oscillation and its adaptation actually buy. Next: a generic adapter for external checkpoints (LinOSS
is JAX; a Torch reimplement or a checkpoint bridge), then run it on a published model.
