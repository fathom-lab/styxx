# FINDING — the conscience axes are DISTINCT but ENTANGLED (PARTIAL-STRUCTURED)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_axis_independence_2026_06_11.md` (frozen
pre-run, committed 662b6ce). Receipt: `axis_independence_result.json`. This is the adversarial
self-falsification of the same-day VALUES-PORTABLE finding
(`FINDING_portable_values_refusal_2026_06_11.md`, @d1e21d4), which claimed "the conscience is a BASIS,
not a lucky truth vector." The objection under test: truth and refusal might be ONE valence axis wearing
two hats. They are not — but they are not clean independent readouts either.**

## Result — distinct directions, not sentiment, but with cross-axis read cross-talk

Three difference-of-means directions (truth / refusal / a valence-sentiment control) fit at a common
gemma layer (L12), then a cross-readout AUROC matrix on held-out test sets (n=15 each). Diagonal = each
direction on its OWN axis (raw AUROC); off-diagonal = each direction on the OTHERS' axes
(discriminability = max(AUROC, 1−AUROC), the conservative measure that makes independence HARDER to
claim).

| direction ↓ / test → | truth | refusal | valence |
| --- | --- | --- | --- |
| **w_truth** | **0.84** | 0.8929 | 0.7679 |
| **w_refusal** | 0.84 | **1.0** | 0.7857 |
| **w_valence** | 0.92 | 0.8036 | **1.0** |

Cosines at L12: truth·refusal **−0.2132**, truth·valence 0.0903, refusal·valence −0.0686. Refusal native
(L8) AUROC 0.9643. Valence-orthogonalized diagonals: truth 0.80, refusal 1.0.

**The two confound hypotheses both FAIL to fire (good for the conscience):**
- **NOT one collapsed direction.** CONFOUND-COLLAPSE required truth↔refusal off-diagonals ≥ 0.75 AND
  |cos| ≥ 0.60. The cosine is only **0.2132** — truth and refusal are near-ORTHOGONAL vectors. They are
  genuinely different directions, not the same vector relabeled.
- **NOT sentiment.** CONFOUND-VALENCE required valence to read both AND orthogonalizing it out to gut
  both. Removing the valence component leaves truth at **0.80** and refusal at **1.0** — neither axis is
  reducible to a good-vs-bad-feeling direction. (Each pairwise cosine with valence is < 0.10.)

**But BASIS-INDEPENDENT also fails.** It required the truth↔refusal off-diagonals ≤ 0.65; they are
**0.8929** (truth-dir reads refusal) and **0.84** (refusal-dir reads truth). Near-orthogonal directions
that nonetheless discriminate each other's content. So the clean "each reads only its own axis" picture
is refused. **Verdict per the frozen gate: PARTIAL-STRUCTURED.**

## What transfers through the map (the cross-model claim)

One shared label-free ridge map (target → gemma L12, fit on the UNION of train texts, labels never touch
it) reproduces the SAME structure in other minds — the entanglement is not a gemma artifact:

| mapped target | truth·truth | refusal·refusal | truth-dir·refusal | refusal-dir·truth |
| --- | --- | --- | --- | --- |
| Llama-3.2-3B (map L11, R² 0.3717) | 0.82 | 1.0 | 0.875 | 0.90 |
| Qwen2.5-3B (map L20, R² 0.4269) | 0.82 | 1.0 | 0.8929 | 0.76 |

Diagonals stay high, off-diagonals stay high — the distinct-but-entangled geometry survives transport.

## Honest reading — what "basis" does and does not mean now

- **SUPPORTED:** the portable conscience is built from MULTIPLE DISTINCT, valence-irreducible value
  directions (truth and refusal are near-orthogonal, each survives removing sentiment), and they ride
  ONE label-free cross-model map. In the load-bearing sense the VALUES-PORTABLE finding intended — not
  one lucky vector, not mere sentiment — "basis" stands.
- **NOT SUPPORTED:** that those directions are CLEANLY SEPARABLE readouts. They cross-talk: a direction
  fit for one axis discriminates the others' content well above the independence ceiling, in gemma and
  through the map. A conscience mount reading one axis will pick up neighboring-axis signal; the axes
  are a correlated frame, not an orthonormal basis.
- This QUALIFIES, it does NOT retract, the prior claim — the two retraction gates (collapse, valence)
  explicitly did not fire. The prior finding carries a pointer to this qualification.

## Honest bounds (what is NOT claimed)

The off-diagonal uses discriminability max(AUROC, 1−AUROC) at n_test = 15 (about half in each class),
whose chance floor is ELEVATED above the random-AUROC midpoint — so the cross-talk MAGNITUDE is
upper-biased and the lower off-diagonals
(≈0.77–0.80) may sit near that inflated floor; the higher ones (0.89–0.92) and the consistent
cross-model replication argue the cross-talk is real, not pure floor. A pre-registered permutation null
on the off-diagonal discriminability is the clean confirmation (named, not run here). The test fixes a
single common layer (L12) by design — cleaner separation might appear with per-axis native layers, a
whitened/Mahalanobis readout, more data, or a non-linear map; this result does not test those. Linear
DiM directions, three axes, local open models (gemma source; Llama-3.2-3B + Qwen2.5-3B targets), n=48
per axis. No model generated any response; valence items are benign/truthful and no operational harmful
content appears anywhere. The frontier this opens: a permutation-nulled off-diagonal, a
whitened-readout independence test, and whether a deliberately-orthogonalized basis reads each axis
cleanly without losing transfer.
