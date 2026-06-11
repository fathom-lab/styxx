# PREREG — the portable conscience: does honesty transfer across minds?

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx.**

The North Star property styxx has NOT yet proven: portability. Today's live MVP reads one model's
honesty signature from a probe trained on that model. The 2026-05-14 universal-directions work showed
separately-trained honesty DIRECTIONS correlate across families — but every model still needs its own
probe. The harder, decisive question, newly testable because tonight we recovered cross-mind
alignment without labels: **can ONE model's honesty direction, carried through a learned cross-model
map, read truth-vs-false in a DIFFERENT model it was never trained on?** If yes, styxx is one
universal conscience for all minds; if no, honesty is model-specific and the per-model zoo is
required. (Activation-STEERING correction is out of scope — closed by `FINDING_uncave` / read != write.)

## Apparatus (frozen)

- **Source conscience:** gemma-2-2b-it truthfulness direction `w` (atlas probe, layer 12, dim 2304,
  in-distribution AUC 0.851) + bias `b`.
- **Target mind:** Llama-3.2-3B-Instruct (different family, size, hidden dim 3072, 28 layers) — never
  trained for this probe.
- **Anchor set (for fitting the map, disjoint from test):** ~160 generated factual statements
  (capitals, element symbols, arithmetic, biology, geography), true and false, avoiding every fact
  used in the test set.
- **Cross-model map `M`:** ridge regression (alpha selected on a held-out anchor split) mapping the
  target's residual at its selected layer into the source's layer-12 space (3072 -> 2304). Target
  layer selected by best anchor-set fit R^2 ONLY (no test contact; no fishing on the outcome).
- **Test set:** the 16 true/false factual pairs from the live MVP (`live_signature_result.json`),
  held out from anchors. Score each target statement `s = w . M(h_target) + b`; metric = paired
  accuracy P(p_correct(true) > p_correct(false)) and mean gap.

## Pre-registered gates (frozen)

- **P1 (portable):** transferred paired accuracy >= 0.65 AND > the 95th percentile of a
  random-direction-transfer floor (100 random unit directions of matched norm pushed through the SAME
  map M, scored identically) AND clearly > chance 0.5. PASS -> **CONSCIENCE-PORTABLE** (one direction
  reads another mind's honesty through a learned map). FAIL -> **MODEL-SPECIFIC** (honesty does not
  transfer even under alignment; reported as the bound).
- **P2 (the map is necessary, descriptive):** transferred accuracy vs a no-information control
  (random map of same shape). The learned map must beat it, else the signal is not coming from
  alignment.
- **Ceiling/VOID:** gemma's own truthfulness probe on gemma's test activations must reach paired
  accuracy >= 0.60 (else VOID-PIPELINE — the source probe itself is broken on this test set).

## VOID / scope

- VOID-PIPELINE (ceiling control fails) or VOID-SUBSTRATE (a model fails to load).
- In-distribution caveat stands: even if portable, the signal is a calibrated correlate of
  believed-truth, not a universal lie verdict; OOD degradation expected. Smoke writes
  `*_SMOKE_INVALID*` only.
- One source->target pair, one task (truthfulness), one map family (linear ridge). A positive result
  is existence, not universality across all pairs; a negative bounds linear transferability only.

## Honest prior

Genuinely uncertain. The directions correlate across families (2026-05-14), and tonight's geometry
aligns across minds — both favor portability. But honesty may live in model-specific features that a
linear map cannot carry, and the test is out-of-distribution for the gemma probe. Either verdict
advances the North Star: PORTABLE = a universal conscience exists; MODEL-SPECIFIC = the precise reason
it doesn't, and the per-model requirement quantified.
