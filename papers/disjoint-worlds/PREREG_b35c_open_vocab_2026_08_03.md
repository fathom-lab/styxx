# PREREG — B35-c: open-vocabulary readout — the 70-way crutch removed

Fathom Lab · 2026-08-03 · frozen before the scored run. Every read in this arc identified a
held-out concept among **70 candidates**. Real reading has no shortlist. This run removes the
crutch: the same label-free pipeline, but the query is scored against **all 462 concepts** —
the 392 anchors the MLP trained toward *plus* the 70 held-outs. Chance collapses from 1/70
(0.0143) to 1/462 (0.00216), and the distractor set now contains the anchors the map was fit
on — the adversarial direction, since mapped queries could snap to trained anchor targets.

## Design (frozen)

Method identical to b34-v3/b35-a (seed 343 split — reusing the b34-v3 split is deliberate: the
question is the READOUT regime, not a new draw; b35-a already settled seed stability). Discover
correspondence label-free, fit one MLP on the discovered pairing, then read each of the 70
held-out queries against the FULL 462-centroid candidate set in the target space. Report both
regimes side by side (70-way from the committed b34v3 receipt; 462-way from this run).
Targets: llama_1b, gemma_2b, qwen_1p5b. Null: the pairing-shuffled MLP, also 462-way.

## Ex-ante sizing honesty

The measured 70-way gemma read is 0.5714. Moving to 462-way can only lower it (more
distractors, including trained-anchor magnets). The bar is set at 10× the NEW chance —
0.0216 — an intentionally modest floor: the licensed question is *does open-vocabulary reading
survive at all*, not *does it stay at 40×*. The 70-way→462-way retention ratio is reported,
not gated (no measured base rate exists to size it).

```gates
{"gates": {"G0_sanity": {"metric": "targets.llama_1b.read462", "op": ">=", "value": 0.0216},
           "G2_null": {"metric": "max_null462", "op": "<=", "value": 0.00648},
           "G1_open": {"metric": "targets.gemma_2b.read462", "op": ">=", "value": 0.0216}},
 "outcomes": [{"when": {"G0_sanity": false}, "verdict": "INVALID__same_family_lost_open_vocab"},
              {"when": {"G0_sanity": true, "G2_null": false}, "verdict": "INVALID__null_artifact"},
              {"when": {"G0_sanity": true, "G2_null": true, "G1_open": true}, "verdict": "OPEN_VOCAB_READ_SURVIVES"},
              {"when": {"G0_sanity": true, "G2_null": true, "G1_open": false}, "verdict": "OPEN_VOCAB_COLLAPSES__closed_set_artifact"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

(G2_null floor = 3× the 462-way chance; a 70-query null draws ~0.15 expected hits at 1/462, so
even 1 hit ≈ 0.0143 would exceed it — stated now: a single coincidental null hit fails G2, and
if that occurs the honest reading is a re-run with a second null seed reported beside it, not a
bar move. The asymmetric strictness is deliberate: at 1/462, "null hits anything" is already
surprising.)

## Outcome reading

`OPEN_VOCAB_READ_SURVIVES` = cross-family label-free reading works without a shortlist — the
strongest form of the claim this arc can state at this scale. `OPEN_VOCAB_COLLAPSES` = the
40× results are real but closed-set-dependent — a scope line every prior finding gains verbatim.
Both go in the synthesis either way.

## Discipline

CPU-from-cache, zero model loads. Smoke = b34-v3 smoke split, INVALID-only. Result
`b35c_result.json`, scored by `styxx.protocol`, OATH-certified before commit.
