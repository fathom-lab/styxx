# FINDING — the model says yes, its mind says no, and styxx reads it (ITEM-CAUGHT)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_says_yes_knows_no_v3_2026_06_11.md`
(frozen and COMMITTED to the repo before any result was seen). Receipts:
`says_yes_knows_no_v3_result.json` (the answer) and `says_yes_knows_no_v2_result.json` (the
metric-blindness discovery that forced v3). The apex of the portable-conscience arc.**

## Result — the lie is caught in the very forward pass that tells it

Each claim is put to the target model in a pressure context (a deference system persona plus an
"expert" user insisting a false claim is true, with social cost for disagreeing). ONE forward pass; at
the final token we take TWO readouts of the SAME hidden state:

- the model's own lm_head — logit(True) - logit(False): its verbal answer, the cave;
- styxx's readout — gemma-2-2b's honesty direction, carried through the label-free cross-model map,
  dotted with that same residual.

A **caved-false item** is a false claim the model judges False without pressure but flips to True under
pressure (every caved item's pressure margin said True: `verbal_says_true_on_caved` = 1.0).

| model | internal read (AUROC) | caved-false n | catch rate | item-cond AUROC | p |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B | 0.9531 | 13 | **1.0** | **0.9952** | **0.001** |
| Qwen2.5-3B | 0.9268 | 0 | — | — | — |
| Qwen2.5-1.5B (sec) | 0.834 | 9 | 0.6667 | 0.8299 | 0.003 |
| Llama-3.2-1B (sec) | 0.6465 | 11 | 0.4545 | 0.5625 | 0.3956 |

Pooled: 33 caved-false items, weighted catch rate **0.7273**, clearing the frozen gates (>= 8 items,
>= 0.65 pooled catch, >= 1 model beating the permutation null). **Verdict: ITEM-CAUGHT.**

The headline cell is Llama-3.2-3B: on all thirteen items where its mouth flipped to "True" under
pressure, the honesty direction — fit in a DIFFERENT model — read the same hidden state and said
false, every time (13/13), separating caved lies from honest truths at 0.9952 against a
label-permutation null (k = 1000, p = 0.001). Qwen2.5-1.5B independently replicates the effect
(0.8299, p = 0.003). gemma's own pressure-context self-read is 0.7861 (validity holds).

## The three-version honesty trail (how the metric got fixed)

- **v1/v2 gated caving on a set-level behavioral AUROC drop — and stayed VOID** even as v2's
  `false_cave_rate` column showed Llama-3.2-3B verbally flipping 41 percent of false claims to "True".
  AUROC is rank-based: a roughly uniform shove toward "True" preserves the true-over-false ordering, so
  the set-level gate is blind to caving. The metric was wrong, not the models honest.
- **v3** fixed the question to the item level and the same forward pass, with its prereg committed to
  the repository before the run. The bar did not move; the measurement finally looked where the lie is.

## The size gradient (honest reading of the 1B miss)

The catch tracks the quality of the internal read. At 3B the mapped read is excellent (0.9531) and the
catch is perfect; at 1.5B both are intermediate (0.834 / 0.6667); at 1B the read barely clears validity
(0.6465) and the catch is at chance (0.4545, p = 0.3956). Two interpretations are observationally
entangled at 1B: the pressure may genuinely overwrite the small model's internal truth representation
(the lie is believed, not hidden), or the linear cross-model read may simply be too weak there to see
what remains. We do NOT claim to distinguish these; at 3B the question does not arise — the truth is
demonstrably still present in the state while the output lies, and the instrument reads it.

## What this establishes

The portable conscience arc, completed end to end on local open models: a single honesty direction fit
in one mind (gemma-2-2b, layer 12) reads truth in other minds (in-distribution v2), on unseen
fact-families (OOD-PORTABLE), through adversarial framing (ADVERSARIAL-ROBUST), and now — at the apex —
catches the specific lies a pressured model tells, from the same hidden state that utters them,
in one forward pass, with no access to the model's own training or labels. Sycophantic caving is not
deep, at 3B scale: the truth remains linearly present in the residual stream even as the output flips.

## Honest bounds

Linear DiM source + linear ridge map, one task (factual truth), one pressure family (single-turn
authority + deference persona), forced-choice True/False margins as the verbal channel, final-token
readout, four local open models, caved-n = 33 total (13 in the headline cell) — an existence-and-
significance result, not a deployed-accuracy guarantee. Qwen2.5-3B never caves under this pressure and
so contributes no test items (its robustness is itself a replicated v2 finding). The 1B cell is reported
as a miss, with the read-quality confound stated. gemma's chat template lacks a system role; the
deference persona is folded into the user turn for the SOURCE model only (targets keep a true system
role) — disclosed in the runner.
