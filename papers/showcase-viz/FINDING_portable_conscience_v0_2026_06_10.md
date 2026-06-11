# FINDING — the conscience is portable as structure, not as a direction (MODEL-SPECIFIC)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_portable_conscience_v0_2026_06_10.md`
(frozen pre-run). Receipt: `portable_conscience_result.json`. The pre-registered floor control
demoted a tempting perfect score to an honest negative — the third such save of the day.**

## The question

Is styxx's honesty signal portable — can ONE model's truthfulness direction, carried through a
learned cross-model map, read truth-vs-false in a different model it was never trained on? A single
portable direction would make styxx one universal conscience; the answer decides whether the North
Star is a magic direction or something else.

## Result — the trap and the control

- **The tempting positive:** gemma-2-2b's truthfulness direction (layer 12), mapped into
  Llama-3.2-3B's residual space via a ridge alignment fit on 108 anchor statements (held-out val
  R² 0.72), read Llama's 16 held-out true/false pairs at **paired accuracy 1.000** — higher than
  gemma reading its OWN activations (ceiling 0.75). Looks like a universal conscience.
- **The pre-registered floor caught it:** a hundred RANDOM directions of matched norm pushed through
  the SAME map also reached paired accuracy 1.000 (95th percentile = 1.000). The perfect separation is
  not a property of gemma's specific honesty direction — it is a property of the MAP, which (fit on
  truth-laden anchors) renders true and false statements linearly separable along almost any
  direction in the shared space.
- **Verdict per the frozen gate (transferred must exceed the floor): MODEL-SPECIFIC.** The claim
  that a single honesty direction transfers across minds is NOT supported.

## What is real (P2, descriptive)

The learned map is doing genuine work, not noise: a RANDOM map of the same shape scores only 0.688
(vs 1.000 for the fitted map). So truth IS strongly, linearly readable in the cross-model aligned
space — the alignment carries the truth structure. Composed with today's convergence and
translation findings, the honest reading is: **the conscience is portable as STRUCTURE — the shared
geometry of truth across minds — not as a transferable vector.** A universal honesty readout is
achievable in the aligned space, but it must be FIT there (on labeled anchors per model pair); it is
not a single direction you carry from one model to another for free.

## Why this is the right answer

styxx exists to catch exactly this: a representation made separable by the analysis pipeline,
mistaken for a transferred signal. The floor control — written before the data — turned a 1.000 that
would have shipped as "universal conscience" into a precise, true statement about what is and isn't
portable. The same law that governed the matcher's miscalibration and the deception probe's
saturation: surface-perfect separation is the cheapest counterfeit; only the control tells real from
artifact.

## Implication for the North Star

The portable conscience is not one magic direction. Two honest paths remain open: (1) per-model
probes (cheap, the current `styxx.residual_probe` atlas — a small zoo), or (2) a truth readout fit
in the cross-model aligned space (portable as structure, needs anchors). The single-vector dream is
closed by its own control; the structure-level portability is the live frontier.

## Bounds

One source→target pair, one task (truthfulness), one map family (linear ridge), an out-of-distribution
test for the gemma probe (ceiling 0.75, not 0.85). A negative bounds linear single-direction
transferability only; nonlinear maps and richer anchor sets are untested.
