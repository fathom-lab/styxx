# PREREG — the real-model bridge, attempt three: does the recall-versus-comparison dissociation appear in deployed pure-decay and attention language models? — 2026-10-11

**FROZEN before confirmatory data — awaiting the operator's signature of the decisions in PLAN §2 (defaults built).**
Runner: `run_realmodel_bridge_v3.py`, scored through `styxx.protocol` from the gates block in §8.
Receipt: `realmodel_bridge_v3_result.json`. Supersedes `PREREG_realmodel_bridge_v2_2026_07_24.md`, whose
200-item design was never run (`RESULT_realmodel_bridge_2026_07_24.md`, ABSTAIN twice).

This document is the preregistration text of `PLAN_next_experiment.md` §3 (the judge's edition of
DESIGN_1, 2026-10-11, amendments **[J1]**, **[J2]**, **[J3]** marked where they occur and collected in §16),
transcribed section for section (§N here is PLAN §3.N). The operator decisions of PLAN §2 are built to the
judge's defaults, and each default is named where it binds: **no M3 arm** (decision 2), **screen bar 0.40**
(decision 3), **the bounded pre-freeze pilot allowed** (decision 4), **n = 600 per cell** (decision 7). A
signature that departs from a default is recorded as a dated amendment in §17 before the seal, never as an
edit of the text above it.

**Seal.** When the operator signs, the lab seals the git blob of this file by the rule in
`papers/checksum/SEALS_2026_09_13.md` (the sha256 of the **git blob**, `git show <commit>:<path> | sha256sum`,
never the working copy) and records it there; whether to anchor is the operator's (decision 5). The pilot of
§9 happens before the seal and its one permitted effect is written into §9 before sealing.

**Amendment convention.** Any change to this document before confirmatory data is a dated amendment
section appended under §17, in the form `PREREG_pmnist_untied_2026_09_02.md` used ("Amendment, before any
data"), stating what changed, why, and that no bar moved. Frozen text is never edited in place. After
confirmatory data nothing changes; a defect found then is written into the RESULT.

---

## 1. What this bridges, and what the record already says

The arc's controlled results, with their receipts:

- A pure-decay channel (eigenvalue phase clamped to zero) did not learn to compare a planted fact with a
  claim 255 steps later within budget, while the rotating channel did: CLAMPED 0.4857 (both seeds), FREE
  1.0 — `papers/frequency-resonance/consistency_oscillation_result.json#/result/cmp_long_clamped`,
  `RESULT_consistency_oscillation_2026_07_23.md` (SUPPORT). Per `CRITIC.md` M6 (the 2026-10-11 frequency
  read), 0.4857 identical to four decimals across two seeds on 6,000 items is a constant-responder base
  rate, so the toy fact being bridged is "decay did not learn distant comparison in 5,000 steps", not
  "requires".
- Decay's probability of keeping a distant fact consistent has a half-horizon at gap 32 and tracks the
  surviving signal (Spearman 0.881) — `consistency_horizon_v2_result.json#/result/half_horizon_gap`,
  `#/result/mechanism_spearman_psolve_vs_surviving`, `RESULT_consistency_horizon_2026_07_24.md` (CONFIRM).
- Decay recalls one fact at every gap from 1 to 255 — `recall_horizon_result.json#/result/clamped_solve_rate_by_gap`
  (all 1.0), `#/result/half_horizon_gap` = null, `RESULT_recall_horizon_2026_07_24.md` (PARTIAL; the
  "relating, not remembering" sentence is that document's reading, not a gated verdict).
- The dose-response in the number of relations did not show: relational drop 0.2 against a 0.40 bar —
  `relational_load_result.json#/result/relational_drop`, `RESULT_relational_load_2026_07_24.md` (PARTIAL).

Two attempts to see this in deployed models ABSTAINED (`RESULT_realmodel_bridge_2026_07_24.md`):

- v1 (`PREREG_realmodel_bridge_2026_07_24.md`, mamba-130m vs pythia-160m, 24 items): recall at ceiling
  because the probe repeated the fact verbatim; compare at chance — `realmodel_bridge_smoke.json#/acc/decay_ssm_compare_16`
  = 0.4167, `#/acc/attention_compare_16` = 0.5417; the ABSTAIN clause `max(recall, compare) >= 0.60` was
  satisfied by the ceiling-level control.
- v2 (`PREREG_realmodel_bridge_v2_2026_07_24.md`, mamba-1.4b-hf vs pythia-1.4b, 20 items): both
  architectures returned identical accuracies, 0.65 at distance 16 and 0.45 at 256 —
  `realmodel_bridge_v2_smoke.json#/acc/decay_ssm_compare_16`, `#/acc/attention_compare_256` — equal to
  what always answering "incorrect" scores on those items, `realmodel_degeneracy_receipt.json#/always_incorrect_would_score`
  = {16: 0.65, 256: 0.45}, `#/matches_constant_responder` = true. The receipt's own `/result/verdict`
  still reads NULL; the certified prose voids it. The 200-item, four-distance design was never run.

Two defects in the frozen v1/v2 gates this preregistration fixes, because they would recur:

1. **Raw accuracies with different chance levels were subtracted.** v2 scored recall against two
   distractors (chance 1/3) and compare against a balanced label (chance 1/2), then formed
   `[C_t − C_m] − [R_t − R_m]` on raw accuracies. Every metric below is on the chance-normalized scale
   `a' = (a − chance) / (1 − chance)`: 0 at chance, 1 at ceiling, in both families.
2. **Label balance was sampled, not enforced.** `run_realmodel_bridge_v2.py` asserted
   `|mean(claim_matches) − 0.5| < 0.2`, which admits the 0.35 / 0.65 split that let a constant responder
   clear the floor. Every balance below is an exact count, asserted per cell from the items.

## 2. Hypothesis — one falsifiable sentence

In a deployed pure-decay state-space model (`state-spaces/mamba-1.4b-hf`, whose recurrence matrix is real
and strictly negative on its loaded weights) the chance-normalized accuracy of judging whether a later
claim matches a planted fact falls, between 16 and 1,024 tokens of intervening text, by at least 0.10
more than its accuracy of recalling that fact, relative to an attention model trained on the same corpus
with the same tokenizer at the same scale (`EleutherAI/pythia-1.4b`) — and if it does not, the
recall-versus-comparison dissociation is confined to the lab's controlled SSMs.

## 3. Arms

| arm | checkpoint | why | availability (HF card / API, read by Designer 1 on 2026-10-11) | on this box |
|---|---|---|---|---|
| **M1** decay SSM | `state-spaces/mamba-1.4b-hf` | `A = −exp(A_log)` real and strictly negative on the loaded weights (`realmodel_bridge_v2_smoke.json#/redteam/mamba_A_real_and_negative_on_weights` = true; re-verified 2026-10-11, `ENV_READINESS_2026_10_11.md`, max −0.1844). The CLAMPED arm of the lab's ablation, shipped as a language model. | public, not gated; F32 safetensors, 1,372,178,432 params; card silent on training data — **the Pile pairing with Pythia rests on the Mamba paper and is printed as an assumption** | cached; fp16 load 7.5 s, 2,627 MB peak; no `mamba-ssm` kernels, slow path |
| **T** attention | `EleutherAI/pythia-1.4b` | the matched comparator the lab froze in v2 | public, apache-2.0; card: the Pile, GPT-NeoX tokenizer, context 2048 | cached; fp16 2,708 MB peak |
| **M3** rotating-state SSM (exploratory, decision 2) | `state-spaces/mamba3-siso-1.5b` | the one deployed language model found whose state carries phase (arXiv 2603.15569, "a complex-valued state update rule"; `config.json` `rope_fraction: 0.5`); its rotation angle is input-dependent, not a static learned θ — the construct the lab's efficiency control found no benefit from on a toy (`efficiency_control_result.json#/verdict` = CAPACITY_IN_DISGUISE) | public, apache-2.0; 2.99 GB bf16; FineWeb-Edu 100 B tokens; tokenizer gated (substitute the cached Llama-3.2-1B tokenizer, vocab 128,256, asserted) | **not runnable on Windows as shipped** (Triton, `mamba-ssm` absent); **default NOT_RUN — the default built here** |

Why not LinOSS, D-LinOSS or S5: Hugging Face searches for `linoss`, `d-linoss`, `s5-ssm`, `oscillatory`
return no models; `github.com/tk-rusch/linoss` holds JAX code and empty `outputs/` folders — no checkpoint,
no language model (Designer 1 and Designer 2, independently, 2026-10-11). Why not another decay model:
one decay arm fits the budget, and M1 is the arm the lab already verified on weights.

The architectural fact carried as the premise of the M1 arm (`RHYTHM_PROPER.md` §5, carry item 1):
Mamba-1's `A = −exp(A_log)` is real and negative on loaded weights. It is a stated precondition with its
check (§12 assert 1), not a result.

## 4. Task — two families over one planted context, matched in everything but the operation

One **item** = three labelled four-digit codes planted at the start ("The {name} code is {code}." ×3,
names drawn from a fixed list of three: alpha, bravo, delta), then exactly D tokens of filler, then a
question. The same 600 items (same codes, same filler, same distance) serve both families and all arms;
only the question differs.

- **RECALL** — "Question: What is the {name} code? Answer:" → total log-likelihood of each of the three
  planted codes; prediction = argmax; chance 1/3 by exact slot balance.
- **COMPARE** — "Question: Is the {name} code {claimed}? Answer:" → log p(" Yes") − log p(" No") at the
  next position; prediction Yes iff > 0; chance 1/2 by exact label balance. When the claim is false,
  {claimed} is one of the **other two planted codes**, so a false claim is in-context and the model must
  bind name to code rather than notice an unseen number.

**Few-shot format.** Four exemplars precede each item, one blank line apart, each a complete three-code
block with zero filler and its question and answer; exemplar codes come from a pool disjoint from the
test pool. Compare exemplars carry exactly two Yes and two No; recall exemplars query slots 1, 2, 3 and 1.
Exemplar order is permuted per item by seed so that the last exemplar's answer is Yes on exactly half the
items of every cell (recency control). The exemplars establish the answer format; they are not the
measurement.

**Filler — [J1].** DESIGN_1 asked for one *contiguous* digit-free passage per item truncated to D tokens.
Contiguous digit-free spans of 1,024 GPT-NeoX tokens are rare in Wikipedia-derived text (dates, counts,
years), so item generation at D = 1024 is at risk of failing or quietly reusing text. Amended: the filler
for an item is a sequence of **distinct digit-free sentences drawn without replacement from the cached
`wikitext-103-raw-v1` train split**, concatenated with single spaces, truncated to exactly D NeoX tokens;
no sentence is used in more than one item or exemplar across the whole run (asserted). Each sentence
must contain no digit, none of the three names, and the word "code" nowhere. This keeps the design's
no-boilerplate property (the v2 filler was two sentences repeated, which a model can skip) and is
feasible: the designer's 200 KB test-split sample had 677 digit-free sentences of 1,449; the train split is
about 500× larger.

**Distances.** D ∈ {16, 64, 256, 1024} NeoX tokens between the last code sentence and "Question:".
D0 = 16, D1 = 1024. Longest prompt ≈ 4 × 45 (exemplars) + 30 (codes) + 1,024 + 15 ≈ 1,250 NeoX tokens,
under Pythia's 2,048 context.

**Items per cell.** n = 600 per (family, distance) (decision 7, default kept); exact balance asserted from
the generated items: compare 300 true / 300 false; claimed-code slot 200 / 200 / 200; queried slot
200 / 200 / 200; last-exemplar-Yes 300 / 300; true × queried slot 100 each. Item seed 20261011. The code
pool is restricted to codes whose " "+code tokenizes to the same length within each tokenizer (two tokens
in NeoX, e.g. " 4271" → [577, 28209], per Designer 1's measurement; asserted at run time), so summed
log-likelihood is a fair comparison across the three candidates.

**Scoring tokens.** " Yes" = 6279 and " No" = 1621 in the NeoX tokenizer that `pythia-1.4b` and
`mamba-1.4b-hf` share (Designer 1's measurement from the cached files; asserted at run time, ids
recorded in the receipt). All single tokens; asserted.

**Batching.** Right-padded batches of 16 prompts of one distance; logits read at each item's own last
position. Causality makes right-padding exact for both architectures; Mamba has no attention mask.

## 5. Controls

- **C_abl — fact-ablated (must stay at chance).** The three code sentences are replaced by a neutral
  sentence of the same NeoX token count; everything else identical. Both primary arms, both families, D0
  and D1. If any cell's normalized accuracy exceeds 0.16 the probe reads priors or leaks (the claimed
  code or the exemplars give the answer away) and the run is INVALID.
- **Positive control that must fire.** The capability screen (§6): both primary arms perform both
  operations well above chance at D0 under the same items, exemplars and scoring that will be read at
  D1. Together with C_abl this shows the probe reads what is planted and only what is planted.
- **C_win — windowed attention, reported, not gated — [J2].** DESIGN_1 gated two outcomes on Pythia run
  under a 4-D additive attention mask (exemplar span + last W = 128 tokens): both families ≥ 0.40 at D0
  and ≤ 0.16 at D1. The judge removes these from the verdict: with a per-query window of W applied in
  every layer, information propagates across layers (position t reads t−W, which read t−2W, …), so a
  24-layer model has an effective reach near 24 × 128 = 3,072 tokens, beyond D1. A windowed Pythia may
  therefore keep some signal at D1 for an architectural reason unrelated to the items, and the design
  would then print INVALID__positive_control_silent for the wrong reason. C_win is still run and
  reported in the receipt under `controls.windowed` (with the W = ∞ equivalence assert, max |Δlogit| ≤ 1e-3
  fp32) as transparency about what a hard-windowed reader does on these items; nothing in the verdict
  depends on it.
- **Degeneracy guard (per cell, per arm, per family).** Compare: fraction predicted Yes within
  [0.20, 0.80]. Recall: largest fraction of items whose chosen code sits in any one slot ≤ 0.80. Also
  reported: AUROC of the compare log-odds against the label (0.5 for any constant responder). The guard is
  **gated at D0** as part of the screen and **reported at every distance**.
- **Mechanism diagnostic (reported, not gated).** For M1, a forward hook on `mixer.dt_proj` gives the
  per-token Δ; with A this gives each channel's per-token |λ| = exp(A·Δ). The product over the filler of
  the slowest channel's |λ| per layer is the deployed analogue of the toy's `mag_max^gap`
  (`consistency_horizon_v2_result.json#/result/mechanism_spearman_psolve_vs_surviving` = 0.881). Reported
  against compare accuracy by distance; four points cannot gate anything.

## 6. Capability screen (the ABSTAIN gate the record prescribes)

For each primary arm (M1, T) at D0: compare normalized accuracy ≥ 0.40 (raw 0.70), recall normalized
accuracy ≥ 0.40 (raw 0.60), and the degeneracy guard passed (decision 3: **0.40 kept, the default**; the
alternative 0.30 could only have been set before sealing, and the bar never moves after). If any fails,
ABSTAIN: the models cannot perform the operation at the shortest distance and no slope may be read. The
bar is on the operation, not on a control, so a ceiling-level recall cannot carry it (the v1 defect).

## 7. Metrics (all chance-normalized, all over items)

`C_x(D)`, `R_x(D)` = compare / recall normalized accuracy of arm x at distance D.

- `between_D1 = [C_T(D1) − C_M1(D1)] − [R_T(D1) − R_M1(D1)]`
- `within_M1 = [C_M1(D0) − C_M1(D1)] − [R_M1(D0) − R_M1(D1)]`
- `within_T` likewise (reported)
- `between_D1_ci_lo` = lower end of the 95% paired item-bootstrap interval of `between_D1` (2,000
  resamples over the 600 shared items, seed 20261011)

## 8. Gates — [J2] applied: eight gates, total outcome table

```gates
{"gates": {"G_P_plumbing": {"metric": "result.plumbing_ok", "op": "==", "value": 1.0,
                            "power_basis": "an all-or-nothing flag over asserted invariants: Mamba A real and strictly negative on the loaded weights in every layer; identical token ids for M1 and T on every prompt; exact per-cell balance counts computed from the items; single-token Yes/No (or True/False) in the tokenizer; equal-length candidate codes; no filler sentence reused across items or exemplars; no digit, no name and no 'code' in any filler; every prompt <= 2000 tokens; right-padding equivalence on two items per arm (max abs logit diff <= 1e-3); >= 6 GB free GPU memory before each load; the runner refuses before data if any fails, so this gate records that it did not"},
           "G_A_screen": {"metric": "result.screen_min_norm_acc_D0", "op": ">=", "value": 0.40,
                          "power_basis": "min over M1 and T, compare and recall, of normalized accuracy at D0; chance is 0 with SE about 0.041 (compare) and 0.035 (recall) at n=600, so 0.40 is about ten SE above chance and leaves 0.60 of headroom for a deficit to appear; raw 0.70 compare, raw 0.60 recall; the v2 compare floor was raw 0.60 and a constant responder cleared it because balance was sampled; under exact balance a constant responder scores 0 here"},
           "G_A_guard": {"metric": "result.guard_ok_D0", "op": "==", "value": 1.0,
                         "power_basis": "1.0 iff every D0 cell of M1 and T has compare Yes-rate in [0.20, 0.80] and recall max-slot fraction <= 0.80; the v2 failure was a constant responder scoring the label base rate (realmodel_degeneracy_receipt.json#/matches_constant_responder = true); with exact balance a constant responder fails G_A_screen, so this gate is the named diagnostic the RESULT asked for, not the only defence"},
           "G_C_abl": {"metric": "result.ablated_max_norm_acc", "op": "<=", "value": 0.16,
                       "power_basis": "max over M1 and T, both families, D0 and D1, of normalized accuracy with the codes removed; the 97.5th percentile of chance at n=600 is about 0.08 normalized (compare) and 0.07 (recall); 0.16 is twice that, so a probe that reads priors or leaks by more than a trivial amount is refused"},
           "G_S_between": {"metric": "result.between_D1", "op": ">=", "value": 0.10,
                           "power_basis": "the bar the lab froze twice (PREREG_realmodel_bridge_2026_07_24.md, PREREG_realmodel_bridge_v2_2026_07_24.md), carried to the normalized scale where it is stricter for compare (raw 0.05) and about equal for recall; at n=600 per cell the SE of this four-cell contrast is 0.05-0.07 depending on the accuracies, so 0.10 is 1.5-2 SE and the interval gate is the arbiter"},
           "G_S_ci": {"metric": "result.between_D1_ci_lo", "op": ">", "value": 0.0,
                      "power_basis": "lower end of the 95% paired item-bootstrap interval of between_D1 (2,000 resamples, seed 20261011) must exclude zero; at n=600 an effect of 0.10 has about 50% power to clear this and an effect of 0.17 about 90%, so SUPPORT needs a clear effect or a lucky draw on a borderline one, and a borderline one is PARTIAL"},
           "G_S_within": {"metric": "result.within_M1", "op": ">=", "value": 0.10,
                          "power_basis": "the within-model form the lab froze twice: Mamba's own compare accuracy must fall with distance by 0.10 more than its recall; without it a between-model gap could be a fixed compare deficit at every distance, which the toy horizon does not predict"},
           "G_N_null": {"metric": "result.between_D1", "op": "<=", "value": 0.03,
                        "power_basis": "the NULL bar the lab froze twice: a between-model contrast at or below 0.03, or of the wrong sign, is read as no transfer; 0.03 is under one SE at n=600, so NULL means the point estimate sits at zero, not that zero is merely inside the interval"}},
 "outcomes": [{"when": {"G_P_plumbing": false}, "verdict": "INVALID__plumbing"},
              {"when": {"G_P_plumbing": true, "G_A_screen": false}, "verdict": "ABSTAIN__capability_absent_at_D0"},
              {"when": {"G_P_plumbing": true, "G_A_screen": true, "G_A_guard": false}, "verdict": "ABSTAIN__degenerate_at_D0"},
              {"when": {"G_P_plumbing": true, "G_A_screen": true, "G_A_guard": true, "G_C_abl": false}, "verdict": "INVALID__probe_reads_priors"},
              {"when": {"G_P_plumbing": true, "G_A_screen": true, "G_A_guard": true, "G_C_abl": true, "G_S_between": true, "G_S_ci": true, "G_S_within": true}, "verdict": "SUPPORT__compare_specific_distance_deficit_in_deployed_decay_ssm"},
              {"when": {"G_P_plumbing": true, "G_A_screen": true, "G_A_guard": true, "G_C_abl": true, "G_N_null": true}, "verdict": "NULL__dissociation_does_not_transfer_to_deployed_models"},
              {"when": {"G_P_plumbing": true, "G_A_screen": true, "G_A_guard": true, "G_C_abl": true, "G_N_null": false}, "verdict": "PARTIAL__reported_verbatim"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

The judge enumerated all 256 assignments of the eight booleans against the lab's own scanner: every
assignment reaches a row. The assignments where `G_S_between` and `G_N_null` are both true are
numerically impossible (one metric, incompatible bars); the table still resolves each by order (the
earliest matching row wins: SUPPORT when `G_S_ci` and `G_S_within` also hold, else NULL), as the lab's
`_select_gates_block`-fed scorer walks the table, so the walk never falls off the end. (PLAN §3.8 counted
these as "two assignments"; with eight gates there are 64, and the conclusion — total table, no row
needed for them — is unchanged.) `tests/test_realmodel_bridge_v3.py` re-runs that enumeration against
this file and checks that the block is byte-identical to the plan's.

## 9. Pre-freeze pilot — the one adjustment allowed (decision 4: allowed, as bounded — the default)

A 48-item format pilot at D0 only, item seed 1011 (codes and filler sentences disjoint from the main
set), both primary arms, before the document is committed and sealed. Its only permitted effect: if either
arm's compare normalized accuracy is below the screen bar or fails the guard under the default answer
format (" Yes" / " No"), the whole design switches once to " True" / " False" (single tokens in NeoX,
ids asserted and recorded), and the chosen format is written into this section before sealing. If both
formats fail the pilot, the run is not started and the record says so. Pilot numbers go into the receipt
under `pilot` and license nothing. Pilot items and sentences are never reused.

The runner reads the **last** line of the form below anywhere in this document and refuses to start the
main run while it says PENDING_PILOT. The line here stays as frozen; after the pilot (`--pilot`, receipt
`realmodel_bridge_v3_pilot.json`) a dated amendment in §17 quotes the pilot receipt and carries a new line
of the same form reading `yes_no` or `true_false`, before the seal.

Answer format chosen (written before sealing): PENDING_PILOT

## 10. Exploratory arm M3 (decision 2; outside the verdict)

If approved: the M3 bars of DESIGN_1's `exploratory` block apply unchanged — `m3_bpb_ratio_vs_M1 <= 1.25`
on 50 KB of wikitext-103 test text as the plumbing check, the same screen and guard at D0, and the labels
`M3_ORDER_AS_PREDICTED` (`within_M1 − within_M3 ≥ 0.10`), `M3_ORDER_AGAINST`, `M3_INDISTINCT`,
`M3_NOT_RUN`. M3 differs from M1 and T in training data, token budget, tokenizer, discretization and
gating; its number is a label beside the verdict, never part of it. **Default: `M3_NOT_RUN` — the default
built; the runner has no M3 path and the receipt's `exploratory` block says so with the reason.**

## 11. Seeds, budget and runtime on this box

- **Seeds.** Items 20261011; pilot 1011; bootstrap 20261011; exemplar permutations derived from the item
  seed. Inference is deterministic in eval mode up to fp16 batched-matmul noise; compare items with
  |log-odds| < 1e-3 are counted and reported, ties broken toward No.
- **Hardware.** RTX 4070 Laptop GPU, 8,188 MiB; the runner checks free memory ≥ 6 GB before each load
  and refuses otherwise (decision 1). Peak expected: M1 fp16 2.6 GB weights + under 2 GB activations at
  batch 16 × 1,250 tokens; T 2.7 GB + sdpa attention at batch 16. One arm resident at a time.
- **Disk.** Primary pair: zero new bytes. M3 (if approved): 2.99 GB of the ~5 GB working figure.
- **Runtime (estimate; the smoke measures it).** M1 runs on the transformers slow path
  (`modeling_mamba.py` loops over sequence positions, 48 layers), roughly launch-bound at 1–5 s per batch
  depending on D (Designer 1's estimate; the judge's own estimate for D = 1024 runs to about 10 s). Per
  distance: 600 items × 4 forward passes / 16 = 150 batches; four distances ≈ 25–50 min; C_abl at D0 and
  D1 ≈ 17–30 min. T with sdpa: a few minutes including C_win. Whole design under 2 GPU-hours with 2×
  slack. If the smoke's extrapolated total exceeds 4 hours the run is refused and reported as
  INVALID__budget, not as a verdict.
- **No training, no fine-tuning, no cloud.**

## 12. Red-team asserts (the runner refuses before data if any fails; all recorded in `redteam`)

1. M1's `A = −exp(A_log)` is real and strictly negative in every layer on the loaded weights.
2. Every M1 and T prompt is byte-identical and tokenizes to identical ids (asserted on every prompt).
3. Exact balance counts per cell as listed, computed from the items, not from the generator's intent.
4. **[J1]** No filler sentence is reused across items, exemplars or the pilot; no planted code, no name
   and the word "code" occur in any filler; filler contains no digit.
5. All candidate codes have equal token length within each tokenizer; the answer tokens are single tokens;
   ids recorded.
6. Windowed mask at W = ∞ reproduces the unmasked Pythia logits (max |Δ| ≤ 1e-3, fp32, one short prompt)
   — a plumbing assert on the reported control, not a gate.
7. Every prompt ≤ 2,000 tokens in every tokenizer used.
8. Free GPU memory ≥ 6 GB before each load.
9. M3 only (if approved): tokenizer vocab 128,256 equals embedding rows; greedy "The capital of France is"
   contains "Paris"; bits-per-byte ratio vs M1 on the same 50 KB ≤ 1.25.
10. Right-padding check: logits at an item's last position are identical (≤ 1e-3) whether the item is run
    alone or inside a padded batch, two items per arm.
11. The runner is red-teamed before any GPU time (the review pattern `RESULT_obfuscation_gradient_2026_06_20.md`
    records), and `--smoke` (24 items, D ∈ {16, 1024}, T only) returns INVALID-only by the gates block.

## 13. Receipt

`realmodel_bridge_v3_result.json`: `config` (checkpoints, revisions, dtype, seeds, format chosen,
distances, n, screen bar), `pilot`, `acc_raw` and `acc_norm` by arm × family × distance, `guard`
(Yes-rates, slot fractions, AUROC) by cell, `controls` (`ablated`, `windowed`), `result` (the eight gate
metrics plus `within_T`), `bootstrap` (interval, resamples, seed), `mechanism` (M1 surviving signal by
distance), `exploratory` (M3 or `NOT_RUN` with the reason), `redteam`, `runtime`. The RESULT is sworn to
this receipt; the gates block's sha256 is recorded with the verdict; the sworn coverage share is printed
beside the HELD/FAILED counts (`CRITIC.md` m5).

## 14. The falsifier, and the sentences the lab could print

**Falsifier.** `between_D1 ≤ 0.03` or of the wrong sign after the controls fire: a deployed pure-decay
SSM loses no more of its comparison than of its recall with distance, relative to attention. Then the
dissociation does not transfer and the arc's July results stay what each already says it is — a
controlled-SSM result. A second way the premise can fail inside PARTIAL: if M1's recall collapses with
distance as much as its compare (`within_M1 ≈ 0` with both at chance at D1), the deployed model has a
storage horizon, which the toy said decay does not have (`recall_horizon_result.json#/result/half_horizon_gap`
= null); the RESULT must print that contradiction, not fold it into "partial".

**If SUPPORT:** "In two Pile-trained 1.4 B language models, the pure-decay state-space model lost at least
0.10 more of its ability to judge a distant claim against a planted fact than to recall the fact, relative
to the attention model, between 16 and 1,024 tokens (`realmodel_bridge_v3_result.json`, interval excluding
zero); this is consistent with the controlled-SSM dissociation and is not a causal attribution to the state
matrix." It is correlational: M1 and T differ in architecture throughout, in optimizer and recipe, and the
training-data match is an assumption the cards do not state.

**If NULL:** "Between 16 and 1,024 tokens, a deployed pure-decay state-space model showed no
comparison-specific distance deficit beyond an attention model of the same scale (between-model contrast
≤ 0.03, `realmodel_bridge_v3_result.json`, controls fired); the recall-versus-comparison dissociation is
confined to the lab's controlled SSMs."

**If ABSTAIN:** "The few-shot probe did not clear its capability screen at 16 tokens in 1.4 B models
(`realmodel_bridge_v3_result.json`); nothing about deployed architectures was measured, and the behavioural
bridge is closed pending a different instrument." A third ABSTAIN closes the behavioural route and leaves
the profiler route (PLAN §1.3's redesign).

**If PARTIAL:** the numbers, the interval, and the sentence "unsupported at this resolution, not refuted",
in the form `RESULT_relational_load_2026_07_24.md` used.

**No outcome** bears on honesty in LLMs, frontier models, human cognition, or importance; the lab's charter
permits only the measured sentence (`STATE_OF_THE_ARC.md` §6). Whatever the verdict, the sentence it earns
is about decay versus attention in deployed sequence models under a compare-minus-recall contrast with its
screen and degeneracy gates; it names which object it measured and carries nothing from the honesty axis
(`RHYTHM_PROPER.md` §5, "One rule for whatever the bridge reports").

## 15. Discipline

Committed before any main-set data; sealed as a git blob per `papers/checksum/SEALS_2026_09_13.md`; the
pilot happens before sealing and is written into this document; `--smoke` is INVALID-only by the gates
block. Result → `realmodel_bridge_v3_result.json`, scored through `styxx.protocol`, RESULT sworn to the
receipt. No bar moves after data. If M3 is not run, the RESULT prints `M3_NOT_RUN` and why. Whatever the
verdict, `RESULT_realmodel_bridge_2026_07_24.md` and `papers/SYNTHESIS_program_2026_07_24.md` item 3 get a
forward pointer to the RESULT in the same commit (the §5.2 shape `papers/AUDIT_the_whole_program_2026_09_01.md`
named is not to be repeated). If this sandbox cannot afford the run, the preregistration stands frozen for
whoever can, and its verdict is theirs to report. Nothing is posted, threaded, previewed or summarized in
public before the RESULT is committed and sworn (decision 6).

## 16. The judge's amendments, collected

- **[J1] Filler from a sentence pool without replacement, not contiguous passages.** Feasibility at
  D = 1024; preserves the no-boilerplate property; adds one plumbing assert (no sentence reused).
- **[J2] C_win reported, not gated; gates reduced from ten to eight.** The windowed-attention control can
  stay informative at D1 through cross-layer propagation (reach ≈ W × layers), so gating on its collapse
  risks an INVALID for an architectural reason. The positive control is now the screen at D0 together with
  C_abl at both distances. The outcome table was re-enumerated (256 assignments, total).
- **[J3] Wording.** One sentence in DESIGN_1 (§Task, COMPARE) contained, as a substring of a longer
  word, one of the words the lab never prints; its author's scan missed it. Reworded. No other sentence
  was changed for style.

Not amended, and why: the 0.40 screen bar (decision 3 lets the operator lower it before sealing; the judge
does not recommend it); n = 600 (decision 7); the bounded pilot (decision 4); the M3 arm (decision 2,
default off); every frozen bar the lab set twice (0.10 / 0.03) and the normalized scale that carries them.

## Disclosed prior

Toward PARTIAL, with ABSTAIN the dominant residual risk. DESIGN_1's author expected the two 1.4 B models
to clear the screen with a few-shot format (the v2 smoke's recall was 1.0 at both distances,
`realmodel_bridge_v2_smoke.json#/acc/decay_ssm_recall_16`, so binding three codes is within reach; compare
with an in-context false code is harder) and, if they do, Mamba's compare to fall more than Pythia's with
distance, while doubting the four-cell contrast clears 0.10 with its interval off zero at n = 600. The judge
put the chance that a 1.4 B base model fails a raw 0.70 compare screen even few-shot at about one in two
(PLAN §1.1). A clean NULL would be as useful as a SUPPORT: it would close the behavioural bridge the arc has
carried open since 2026-07-24. The bars are frozen either way.

## 17. Amendments (dated, appended; nothing above this line is edited)

None yet. The pilot's outcome (§9) and the operator's signature of PLAN §2 (any departure from a default)
are entered here before the seal, each dated, each stating that no bar moved.

### 17.1 Amendment, before any data (2026-10-11) — the smoke, the pilot, and why the main run is not started

**No bar moved; nothing above §17 is edited.** This entry records the two runs §§9, 11 and 12 (item 11) call
for before the seal, and the one effect §9 lets them have. Runner: `run_realmodel_bridge_v3.py` at commit
`9b54d8d2` (sha256 `45dd70df…`, recorded in each receipt as `config.runner_sha256`). The operator decisions
of PLAN §2 remain unsigned; the defaults are what was built and run.

**The smoke** (`realmodel_bridge_v3_smoke.json`). Run as `--smoke --arms M1,T`: 24 items at D ∈ {16, 1024},
both arms, so that the slow arm's runtime is measured — a superset of §12 item 11's T-only smoke, INVALID-only
either way. Verdict by the gates block: `INVALID__smoke_plumbing_only` (prereg commit `333712d5`, gates
sha256 `a21aa51d…`). The smoke's items come from a seed of their own (1016) and from reserved slices of the
sentence order and of the code pool that the main set and the pilot never touch: no main-set item was
generated or scored (the frozen text fixes no smoke seed; the runner's choice is recorded here). Every assert
of §12 held and is in `redteam`: `A = −exp(A_log)` real and strictly negative in all 48 layers (max −0.0039);
identical token ids for M1 and T on 192 prompts; exact balance per cell; 1,113 sentences used, none twice,
none with a digit, a name or "code"; " Yes" 6279 / " No" 1621 / " True" 11793 / " False" 16708 single
tokens; 8,828 two-token codes; longest prompt 1,210 tokens; the W = ∞ mask reproduces the unmasked logits
(max |Δ| 0.0, fp32); right-padding max |Δlogit| 3.4e−5 (M1) and 6.1e−4 (T) in fp32, with the fp16-on-GPU
difference (0.039, 0.165) reported, not gated; 6.85–6.90 GiB free before each load and after each release.
The runner runs M1 at batch 16 through a copy of transformers' `slow_forward` whose discretised tensors are
materialised per chunk of the sequence; it reproduces the reference to max |Δlogit| 0.0 in fp32
(`redteam.chunked_scan_equivalence_fp32_cpu`). Budget (§11), per batch of 16: M1 11.7 s at D = 1024 and
2.3 s at D = 16, T 1.2 s and 0.2 s; extrapolated main run 1.635 h (M1 1.436, T 0.199), inside the 4-hour
rule, so not `INVALID__budget`. The smoke's accuracies are plumbing numbers on 24 items and are not read.

**The pilot** (`realmodel_bridge_v3_pilot.json`, §9). 48 items at D0 = 16, item seed 1011, codes and
sentences disjoint from the main set and from the smoke (171 sentences used, none shared), both arms, both
answer formats, exact balance (24/24; 16/16/16; 16/16/16; 24/24; 8/8/8), the same asserts held (A as above;
chunked scan 0.0; W = ∞ 0.0; right-padding fp32 2.3e−5 / 6.0e−4). Compare: normalized accuracy against the
0.40 bar, answer rate against the [0.20, 0.80] guard, AUROC of the log-odds against the label; recall is
format-free:

| arm | " Yes" / " No" | " True" / " False" | recall |
|---|---|---|---|
| M1 | −0.042 (raw 0.479); Yes-rate 0.94, **guard failed**; AUROC 0.38 | −0.167 (raw 0.417); True-rate 0.92, **guard failed**; AUROC 0.37 | +0.656 (raw 0.771); slot fractions 0.52 / 0.17 / 0.31; guard passed |
| T | +0.083 (raw 0.542); Yes-rate 0.42, guard passed; AUROC 0.56 | +0.042 (raw 0.521); True-rate 0.56, guard passed; AUROC 0.53 | +0.969 (raw 0.979); 0.35 / 0.31 / 0.33; guard passed |

Under the default format M1's compare is below the bar and fails the guard, so the one switch §9 permits
applies; under " True" / " False" M1's compare is again below the bar and again fails the guard, and T's
compare is below the bar under both formats. Both formats fail the pilot. §9: "If both formats fail the
pilot, the run is not started and the record says so." The record says so:

Answer format chosen (written before sealing): NOT_STARTED

The runner reads that line and refuses the main run with that reason. **No main-set item has been generated
or scored.** Nothing is sealed; whether to seal a preregistration whose run does not start is the operator's
(decision 5). The pilot's numbers license nothing (§9): they are not a verdict of the gates block, not an
ABSTAIN, and no RESULT is written from them. What they say, on 48 items at the shortest distance, is that
neither 1.4 B base model answered the few-shot comparison question above the bar this document froze — T's
raw 0.542 and M1's 0.479 sit more than two standard errors (0.072 at n = 48) under the raw 0.70 the bar
means — while both recalled the planted code well above its bar (M1 raw 0.771, T 0.979). PLAN §1.1 put the
chance of this at about one in two before anything ran.

**What follows is the operator's.** §9 permits one bounded adjustment and it has been used. Any further change
— another answer format, more exemplars, another prompt, another scale, another probe — is a different
instrument and a new preregistration with its own pilot, not an amendment of this one. The bars here are
frozen whether or not a successor is written, and this document stands as the record of the attempt. Nothing
is posted (decision 6).

**Runner defects the smoke found before any data, fixed as commits on this branch, none touching a number:**
the slow path's whole-sequence `[B, 4096, L, 16]` tensors had forced M1 to batch 2 at D = 1024, eight times
the batches §4 assumes — the sequence-chunked scan restores batch 16 and is asserted equal to the reference
(`b6672234`); under a half model the Mamba backbone's `last_hidden_state` is fp32 (`residual_in_fp32`), so
the head is applied in its own dtype (`65397781`); the release path emptied the allocator cache while a
reference to the model was alive, and the §11 free-memory check refused the second arm at 4.28 GiB — every
reference is dropped before the cache is emptied (`6c21580b`). Reported, not gated, and added after the
smoke showed the frozen mechanism headline (§5, the slowest channel's surviving signal) reading −0.0 at both
distances because the per-token slowest channel is one whose Δ ≈ 0: two companions per layer, the median
channel's filler-summed log|λ| and the fraction of channels still above 0.5 (`9b54d8d2`); on the smoke's 24
items they read −0.0034 → −0.0845 (best layer) and 0.466 → 0.056 from D = 16 to D = 1024.

### 17.2 Amendment, before any data (2026-10-11) — the operator's one bounded probe redesign, the last; the receipt fields the downstream designs ask for; four review minors

**No bar moved; nothing above §17 is edited.** The format line of §17.1 (`NOT_STARTED`) stands until §17.3: the main run
stays refused while this amendment is in force and no new pilot has run. Runner: `run_realmodel_bridge_v3.py` at the
commit that follows this one on the branch (its sha256 is recorded in every receipt it writes as `config.runner_sha256`
and is quoted in §17.3). Tests: `tests/test_realmodel_bridge_v3.py` at the commit after that.

**The operator's decision (2026-10-11, about 22:50), in substance.** After the §9 pilot failed both answer formats
(§17.1), the operator approved **one** bounded redesign of the compare probe, pre-stated here as **the last**: a compare
readout that is argmax-shaped like recall, so that no Yes-prior exists to drift, scored over the three name tokens with
exact balance over the correct slot; the degeneracy guard for compare becomes max-slot ≤ 0.80, as recall's already is;
one 48-item pilot at D0, both arms, under 15 GPU-minutes, written into this document before sealing. **A second failed
pilot closes the behavioural route** (`PLAN_next_experiment.md` §3.14: a third ABSTAIN-class outcome leaves only the
profiler route). The operator also approved, pre-data, the four receipt and generator amendments the two downstream
designs ask of this document (`NEXT_AFTER_V3.md` §3.6 items 1–4; `horizon-zoo/DESIGN.md` §8 items 2–3 and ATTACK F1/M5;
`intervene/ATTACK.md` M7). Anything beyond this entry — another readout, more exemplars, another scale, another probe —
is a different instrument and a new preregistration (§17.1, last paragraph), not an amendment of this one.

#### 17.2.1 The redesigned compare readout

- **Prompt shape.** Unchanged in structure (§4): four exemplars, a blank line, the three planted sentences
  `The {name} code is {code}.`, exactly D NeoX tokens of filler, a newline, the question. The compare question becomes
  `Question: Which name has the largest code: alpha, gamma or delta? Answer:` — the same words on every item and every
  exemplar; the three option names are listed in slot order. The recall question is unchanged
  (`Question: What is the {name} code? Answer:`).
- **Answer set and scoring.** At the position after `Answer:` (NeoX id 27 for `:`), the log-probabilities of the three
  name tokens are read from one forward pass; prediction = argmax over the three; correct iff it is the slot whose code
  is numerically the largest. An exact tie resolves to the lowest slot and is counted (`near_ties`). The recall family
  is unchanged (argmax of the three candidate codes' summed log-likelihood).
- **Names.** The §4 list (alpha, bravo, delta) cannot serve as an equal-length three-way answer set: in the NeoX
  tokenizer both arms share, ` bravo` is two tokens ([270, 41647]) while ` alpha` (9765) and ` delta` (18687) are one,
  so a summed log-likelihood over the three would compare one token against two. The test names become **alpha,
  gamma, delta** (ids 9765, 17356, 18687, each a single token, asserted and recorded in `redteam.name_token_ids`).
  The exemplar names are a disjoint fixed list, **sigma, theta, omega** (40009, 39116, 40639; §17.2.2 item 4).
- **How the claim-is-true/false structure maps onto a three-way choice.** §4's compare item carried a claim about one
  name whose truth was a binary label with an in-context false code; its readout was the sign of log p(Yes) − log p(No)
  against a fixed threshold, which is where a Yes-prior lives. The redesigned item carries the same three bound facts
  and asks a relation among them: exactly one of the three in-context names is the correct answer, and the two
  distractors are the other planted names, so every option the model can choose is in context (the property §4 wanted
  from in-context false claims), the readout has no threshold and no label prior, and a constant responder scores
  exactly 1/3 raw = 0 normalized under exact balance, as recall's already does. What "compare" measures changes with
  it and the record says so: §4's operation was *fact versus later claim* (the toy's `cmp_long`); the redesigned
  operation is *fact versus fact* — the magnitude order of the three planted codes, which cannot be answered from any
  one binding. Both are relations over planted content that recall does not require; the dissociation this document
  bridges (§1, "relating, not remembering") is read with that substitution, and §14's sentences are carried with
  "judge which of three planted codes is the largest" in place of "judge a distant claim against a planted fact".
- **Exact balance per cell (asserted from the items, assert 3).** n must be a multiple of 6 (600, 48 and 24 are).
  Correct (largest-code) slot n/3 each; queried slot n/3 each; the last exemplar's compare answer slot n/3 each and its
  recall query slot n/3 each; the last exemplar's compare answer slot equals the correct slot on exactly n/3 items and
  its recall query slot equals the queried slot on exactly n/3 items, so a copy-the-last-exemplar's-position responder
  scores exactly chance in both families (the recency control §4 had for Yes/No, carried to slots). At n = 600:
  200/200/200 for each marginal, 200 for each coincidence; at n = 48: 16/16/16 and 16. One count is reported and not
  exact: the number of items whose queried slot is also the largest-code slot is near n/3 (600 is not a multiple of
  9); it bears on no in-prompt strategy, since the compare question names no slot and the recall prompt carries no
  notion of "largest".
- **Exemplars.** Four blocks with the exemplar names and zero filler, as §4; recall queries slots 1, 2, 3, 1 (frozen);
  the largest exemplar code sits at slots 2, 1, 3, 2, arranged by the generator, so the queried code is the largest
  in one exemplar of four. The per-item permutation is chosen so that the last exemplar is exemplar 2 on n/3 items,
  exemplar 3 on n/3, and exemplars 1 and 4 on n/6 each — which is what makes both last-slot marginals exact.
- **Codes.** Four-digit, from the equal-length pool (8,828 two-token codes in NeoX, asserted); the three planted codes
  of an item are drawn distinct from the region's test pool restricted to **≤ 8999**, and the largest is placed at
  the correct slot; the restriction exists so that a larger code is always available for C_overwrite.
- **Guard.** Compare: the largest fraction of items answered with any one name ≤ 0.80 (`max_slot_fraction`), the
  recall guard's rule; gated at D0 inside `G_A_guard`, reported at every distance. The Yes-rate window retires with
  the Yes/No readout. Also reported per cell: the one-versus-rest AUROC averaged over the three slots (0.5 for any
  constant responder), the mean and median margin (score of the correct slot minus the best other slot — the
  three-way analogue of the Yes/No log-odds), and the near-tie count.
- **Chance normalization stays three-way-correct.** `a′ = (a − chance)/(1 − chance)` with chance 1/3 in **both**
  families (compare was 1/2). The bars never move: 0.40 normalized is raw 0.60 in both families (compare was raw
  0.70); 0.16 normalized is raw 0.44. The paired item bootstrap of §7 weights both families by 1/(1 − 1/3) = 1.5 and
  its point estimate equals `between_D1` (asserted). Chance SE at n = 600 is sqrt((1/3)(2/3)/600) × 1.5 = 0.029
  normalized in both families, so 0.40 is about fourteen SE above chance (the frozen `G_A_screen` text says ten),
  and the 97.5th percentile of chance is about 0.057 normalized against the 0.16 `G_C_abl` bar (the frozen text says
  0.08 for compare). The gates block of §8 is **not edited** (its bytes are the plan's, tested byte-identical); where
  its `power_basis` prose names raw 0.70, SE 0.041 or the Yes-rate window, it describes the retired readout and this
  entry gives the three-way reading; the metric paths, ops and values are unchanged.
- **Metrics and gates.** §7's formulas are unchanged in form. `G_C_abl` reads `result.ablated_max_norm_acc` over D0
  and D1 as frozen; the ablated cells at 64 and 256 (item 4 below) are reported under `controls.ablated` and their
  maximum under `result.ablated_max_norm_acc_all_distances`, outside the gate.
- **Retired.** The `yes_no` and `true_false` readouts of §9: the runner parses them on the format line and refuses the
  main run on either ("retired by amendment 17.2"). The §9 line and the §17.1 line stand as the record of the attempt.

#### 17.2.2 The receipt fields and generator rules of `NEXT_AFTER_V3.md` §3.6 items 1–4 (pre-data)

1. **`per_item`** — for every arm, family, distance and variant (plain, ablated, windowed, near, overwrite): the
   correctness bits, the predicted and the correct slot, the three scores (compare: log p of each name token; recall:
   each candidate's summed log-likelihood) and the margin per item; for C_overwrite also whether the prediction was the
   re-planted name. The zoo's bootstrap and `c_auc` can be recomputed from the receipt (zoo §8 item 2, ATTACK m8/m10).
2. **`items.sha256`** — sha256 of the UTF-8 text of every prompt (per family, distance and variant, in item order), of
   every code the run plants (test, overwrite and exemplar codes, sorted by hash) and of every sentence the pool
   handed out (sorted by hash). Later runs exclude by these lists, never by regeneration (intervene M7). The runner
   asserts disjointness from the committed receipts by these lists before any model is loaded (§17.2.4).
3. **`items.prefix_nested_across_distances = false`** (and `code_paired_across_distances = false`): items are
   generated per distance from the item seed plus D; the filler at D = 16 is not the head of any other distance's
   filler and the codes are drawn afresh per distance; an item is shared across the two families, every variant and
   both arms at one distance (zoo §8 item 3 — the zoo's interval must resample distances independently).
4. **Generator rules and controls.**
   - *Exemplar names disjoint from test names* (sigma, theta, omega against alpha, gamma, delta; asserted). Every
     prompt binds each test name once, so the recency-resolution component of zoo ATTACK F1 is removed.
   - *`C_abl` at every distance* (was D0 and D1): the three code sentences replaced by a neutral text of the same
     in-situ token count, both families, both arms, D ∈ {16, 64, 256, 1024}; gate as above.
   - *`C_near` at D1, compare only, reported* (zoo F1 fix b, re-cut for a three-way readout, which has no single
     "fourth code" question): the item's own D1 filler stands between the exemplars and the code block, and a fresh
     16-token filler stands between the block and the question — the codes are 16 tokens before the question while
     the exemplars are where they are at D1. If c_near(D1) ≈ c(D0), format distance is excluded and a compare drop at
     D1 is about the codes; if c_near(D1) falls with c(D1), the drop is format decay and the RESULT prints
     FORMAT_DECAY beside the cell. Recorded under `controls.near`.
   - *`C_overwrite` at D1, compare only, reported* (zoo M5): the item's filler keeps its head and its last sentence
     becomes `The {name} code is now {code}.` for a name that is **not** the correct slot (the overwrite slot, never
     the correct one; each (correct, overwrite) pair on n/6 items so the overwrite name is balanced n/3 each) and a
     code drawn from the region's test pool **larger than all three** planted codes; the filler stays exactly D1
     tokens (a fresh fill closes any gap before the re-plant). Scored against the original correct slot; the
     fraction of items answered with the re-planted name is `overwrite_follow_rate`. A model whose readout at D1 is
     live to planted content follows the re-plant (accuracy against the original falls to or below chance); a model
     that follows nothing at D1 has its far cells printed POSITIVE_CONTROL_SILENT. Recorded under `controls.overwrite`.
   - The filler rules of [J1] extend to the six names: a pool sentence containing any of them, a digit or "code" is
     skipped at draw time (count recorded as `fillers.sentences_skipped_for_a_name`); the pool itself — its
     eligibility filter, cap and order — is unchanged, so the sentence indices in the committed smoke and 1011-pilot
     receipts still name the same sentences (asserted against the parquet sha256s and the eligible count). The
     re-plant sentence is not filler and is excluded from the no-name check by construction.
   - Mechanism diagnostic (§5): unchanged, on the plain compare cells of M1.

#### 17.2.3 The four review minors (2026-10-11 review of the branch at `7ba10c3e`)

- (a) `write_json` emits UTF-8 without BOM with LF newlines on every platform (the committed receipts' blobs are LF by
  the checkout's normalization; the working copies were CRLF on Windows). Tested.
- (b) The model-level asserts of **both** arms (§12 items 1, 6 and 10 and the chunked-scan equivalence, on CPU in fp32)
  run before any cell of any arm is scored or printed; the order and clock are recorded under
  `redteam.model_level_asserts_before_any_scoring`, and the scoring loader refuses an arm whose asserts did not run.
- (c) §12 item 11's T-only smoke lets the 4-hour rule miss the slow arm. The budget extrapolation now requires
  `--arms M1,T` (the smoke's default): an untimed arm makes its verdict `OK_PARTIAL` and the main run refuses on it.
  The pass structure counted is: compare 1 + recall 3 per cell, plain and ablated at every distance, T's windowed
  control at D0 and D1, and the two compare-only D1 controls for both arms. Under that structure the committed smoke's
  per-batch seconds (M1 11.66 / 2.27 s, T 1.18 / 0.19 s at D = 1024 / 16) extrapolate to **2.22 h** (M1 1.97, T 0.25),
  inside the rule. The main run nevertheless requires a smoke receipt **written by this readout** (`config.readout`),
  since the committed smoke red-teamed the retired code paths; that smoke has not been run and is not part of this
  amendment's GPU budget.
- (d) The `G_A_screen` power basis quotes a recall SE of 0.035 where sqrt((1/3)(2/3)/600) × 1.5 = 0.029. Corrected
  here; the block's bytes are not edited (the seal is over them) and the bar is unchanged. Under this amendment the
  same 0.029 is the compare SE as well.

#### 17.2.4 The pilot of this amendment (seed 1012), pre-stated

- **Items.** 48 at D0 = 16, item seed **1012**; sentences from a reserved 3,000-sentence slice of the seeded sentence
  order cut from what was the main region's tail, before any main-set item exists (`perm[-10000:-7000]`, reshuffled by
  1012); the smoke's slice (`perm[-7000:-4000]`) and the 1011 pilot's (`perm[-4000:]`) are unchanged. Codes from two
  new pools (12 exemplar, 60 test) carved from the head of what was the main test pool; the five committed pools are
  unchanged. Disjointness from the committed smoke and 1011-pilot receipts is **asserted by sha256 lists** (every
  sentence index and sentence hash they record, every code of their reserved pools) and from the main region by
  construction (every sentence drawn lies inside the pilot2 slice, asserted); the pilot's own lists go into its receipt
  so the main run excludes them the same way.
- **Run.** Both arms, the model-level asserts of both before any scoring, then per arm: compare (this readout), recall,
  and C_abl at D0 for both families (reported). Under 15 GPU-minutes; one arm resident at a time; ≥ 6 GB free before
  each load. Receipt `realmodel_bridge_v3_pilot2.json` with every field of §17.2.2. The runner refuses to run it twice.
- **The rule, pre-stated.** An arm *clears* iff at D0 its compare and its recall normalized accuracy are both ≥ 0.40
  and both max-slot guards pass (the §6 screen on the pilot's items). Outcome labels: `CLEARED_BOTH`,
  `CLEARED_T_ONLY`, `CLEARED_M1_ONLY`, `NEITHER`. If `CLEARED_BOTH`, §17.3 sets the format line to `argmax_largest`
  and the main run may proceed **only after** this document is sealed as a git blob and the operator's anchor is
  placed (decision 5), and after a smoke under this readout with both arms timed (§17.2.3 c). Otherwise §17.3 sets
  the format line to `NOT_STARTED` (final) and states that the behavioural route is closed by the operator's rule.
  `CLEARED_T_ONLY` is printed as a measured fact at pilot strength (`NEXT_AFTER_V3.md` §3.1) and closes the route all
  the same. The pilot's numbers license nothing else (§9).

#### 17.2.5 What this entry does not change

The eight gates, their metrics, ops and values; the outcome table; n = 600; D ∈ {16, 64, 256, 1024}; the arms; the
item seed 20261011 and bootstrap seed; the budget rule; the discipline of §15; decision 6 (nothing posted). The
preregistration remains unsealed and unsigned (PLAN §2); the defaults are what is built.
