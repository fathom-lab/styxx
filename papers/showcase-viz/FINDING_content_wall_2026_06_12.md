# FINDING — the "content wall" is a WHITENING-READOUT artifact: content identity DOES transport cross-model (CONTENT-WALL per the frozen gate, interpretation corrected)

**2026-06-12 · Fathom Lab / styxx. Pre-registered: `PREREG_content_wall_2026_06_12.md` (frozen pre-run,
committed fa7c4ee). Receipt: `content_wall_result.json`; figure `content_wall.png`. Backlog B31. Cycle 6
(`FINDING_concept_decode_2026_06_12.md`, CONTENT-WEAK) concluded cross-model content identity does not
transport — "the channel is a value thermometer, not a content transcript." This scaled the anchor data
~18× to test whether that wall was a power problem. It is — AND more: the wall was also the wrong readout
metric. Content identity transports near-perfectly once the map is well-fit and read in the right space.**

## Result — frozen verdict CONTENT-WALL, but the wall is in the metric, not the channel

Only the anchor count changed from cycle six: 720 per-instance anchor points (120 concepts × six
templates) versus cycle six's 40 concept centroids; the same 20 held-out test concepts; the gate readout
identical to cycle six (label-free ridge map target→gemma, gemma-ZCA whitening, cosine retrieval). The
map is now extremely well-fit: held-out R² = **0.9095** (Llama) and **0.8654** (Qwen), versus the
near-zero R² of the cycle-six map.

| readout (Llama-3.2-3B → gemma, 20-way, chance 0.05) | top-1 | top-5 | MRR |
| --- | --- | --- | --- |
| **gate: gemma-ZCA-whitened (cycle-6-identical)** | **0.05** | 0.35 | 0.2478 |
| random-map floor (p95) | 0.10 | — | — |
| descriptive: RAW cosine (no whitening) | **0.90** | 1.0 | 0.9417 |
| descriptive: mapped-space whitened (B29/B32 recipe) | **1.0** | 1.0 | 1.0 |
| gemma in-model ceiling | 1.0 | — | — |

Qwen-3B agrees: gate-whitened 0.1 (chance), RAW 0.85, mapped-whitened 1.0. **Per the frozen gate (the
gemma-whitened readout < 3× chance with a well-fit map), the verdict is CONTENT-WALL.** But that gate
readout was mis-specified, and its pre-registered interpretation ("content does not transport even with a
well-fit map") is FALSIFIED by the descriptive readouts: with the SAME well-fit map, raw cosine retrieves
content at 0.90 and the mapped-space-whitened readout at 1.0. **Content identity transports near-perfectly
cross-model. The wall is in the whitening metric, not the channel.**

## Why the gemma-whitening readout destroys content (the mechanism)

The gemma ZCA whitening is fit on the anchor distribution and up-weights its low-variance directions
(division by sqrt of small eigenvalues + eps). In gemma's native space the content signal lives in the
high-variance directions; but the ridge map's residual error concentrates in the many near-null
directions, and whitening amplifies exactly those — swamping the content signal that raw cosine reads
cleanly. This is the B29 lesson at full strength: reading mapped points in the source metric
mis-calibrates them; for a 1-D value axis that cost 0.0062, for high-D content identity it costs
everything (1.0 → 0.05). The mapped-space whitening (B29/B32) re-estimates the metric on the mapped
distribution and recovers content perfectly.

## What this corrects — cycle 6's conclusion was an underpowered map read in the wrong metric

Cycle 6's CONTENT-WEAK verdict was honestly earned at the time (40 anchors, gemma-whitened, top-1 0.0),
but its broad reading — "the cross-model channel is a value thermometer, not a content transcript" — is
RETRACTED. It rested on two stacked problems: an underpowered map (R² 0.06) AND a whitening readout that
destroys content. Remove both — scale the anchors (R² 0.91) and read raw or mapped-whitened — and the
transcript transports near-perfectly, Llama and Qwen. The honest, corrected statement: **cross-model
content identity DOES transport through a label-free linear map, given enough anchor data and the right
readout space.** The value/content asymmetry cycle 6 reported was a readout artifact, not a property of
the channel.

## Discipline — the frozen verdict stands; CONTENT-TRANSPORTS is owed a fresh prereg

The frozen gate was defined on the cycle-6-identical gemma-whitened readout, and it gives chance, so the
verdict label is CONTENT-WALL — not retroactively switched to the descriptive readout that would pass. The
raw and mapped-whitened readouts were pre-registered as DESCRIPTIVE, so they correct the INTERPRETATION
but cannot be claimed as the verdict (no goalpost-moving — same rule that kept cycle 5 at PARTIAL). This
is methodological self-correction of a mis-specified gate readout, the same class as the cycle-2→3
covariance-artifact correction and the OOD v1→v2 null. The confirmatory **CONTENT-TRANSPORTS** verdict
gets a FRESH pre-registration with the raw/mapped-whitened retrieval as the frozen gate (backlog B33); the
descriptive numbers here (0.90 / 1.0, both targets) predict it passes overwhelmingly.

## The telepathy ledger, revised

Cycle 6 closed the content-transport door; B31 reopens it. With a well-fit map, cross-model content
identity decodes near-perfectly (20-way, 0.90 raw / 1.0 mapped-whitened, Llama AND Qwen) — much closer to
"reading what a model represents" than cycle 6 implied. The operator's "we are close" intuition gains real
support on THIS sub-question. The categorical gaps to telepathy still stand and are NOT touched: this needs
white-box activations AND a paired anchor corpus (zero-paired is a closed negative), it is a fixed 20-way
set (not open vocabulary), and the targets are local same-cluster models (cross-vendor universality is
killed). So: the content channel is NOT walled, but reading a sealed, uncooperative, arbitrary-vocabulary
mind remains several categorical rungs away. Honest bounds: linear ridge map, mid-layer, template-mean
centroids, n_test=20, neutral concepts, last-token pre-output, no generation. The figure shows all
readouts so the gate-vs-descriptive gap is visible, not hidden.
