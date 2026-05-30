# PRE-REGISTRATION — does single-pass confabulation legibility on MULTI-HOP LOGIC hold on a THIRD architecture FAMILY (Gemma-2-2B)?

**Written 2026-05-30, BEFORE the confirmatory Gemma-2-2B run.** Single-pass legibility on multi-hop
transitive-ordering logic now has two-family support: Qwen2.5-1.5B (B_contrast 0.000,
`FINDING_detection_locus_logic_2026_05_30.md`) and Llama-3.2-3B (B_contrast −0.183,
`FINDING_detection_locus_logic_llama3b_2026_05_30.md`, commit `8fbcfcd`) — on both, single-pass clean
entropy detects logic confabulation at least as well as N=10 resampling. Qwen and Llama, while
distinct, are both dense decoder transformers in the common lineage. This run tests a THIRD,
genuinely different pretraining family — **google/gemma-2-2b-it (26 layers, GQA, logit soft-capping)**
— to ask whether the legibility is cross-FAMILY, not just cross-model.

**Does single-pass internal confidence (clean entropy / logit margin) separate confab from correct
on multi-hop logic for Gemma-2 too, or is the legibility specific to the Qwen/Llama lineage?**

## Item set (same seeded 48 items as the Qwen/Llama logic runs — identical construction)

`run_detection_locus_logic.py --model google/gemma-2-2b-it`, seed `20260530`. Same HARD pool 24
(K∈{6,7,8,9}, 5–8 hops) + EASY pool 24 (K∈{2,3}, 1–2 hops); ground truth = target rank, in-code,
hashed pre-scoring.

**Answer-key SHA-256 (48 items, identical to the Qwen/Llama runs, pinned pre-scoring):**
`97d816808a6874027637a35a7beeb8a7078aa483f30a01f3ea9e58f9e347e02c`

- **CONFAB group** = HARD items Gemma-2-2B answers WRONG greedily (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = EASY items Gemma-2-2B answers RIGHT greedily (`v1 == correct`).

## Apparatus notes (Gemma-2-specific; pre-committed)

Gemma-2 requires two machinery accommodations, both recorded BEFORE the confirmatory run:
1. **No system role.** Gemma-2's chat template rejects a `system` message. The shared
   `wb._render_chat` folds the system text into the user turn (`f"{system}\n\n{user}"`) ONLY when
   the template rejects a system role; it is byte-identical for Qwen/Llama (which accept it). The
   system prompt content is UNCHANGED.
2. **Attention soft-capping → eager.** The runner loads Gemma with `attn_implementation="eager"`
   (Gemma-2's attention logit soft-capping is skipped under sdpa). Qwen/Llama loads are unchanged.
3. **Final-logit soft-capping (scope caveat).** Gemma-2 applies `final_logit_softcapping`
   (≈30, tanh) to the logits the logit-lens reads, so Gemma's ABSOLUTE entropy/margin are compressed
   and NOT comparable in absolute terms to Qwen/Llama. The pre-registered quantity B_contrast is a
   WITHIN-model difference of AUCs (both computed on Gemma's own soft-capped signals) and is
   unaffected by this — it remains valid and cross-architecture-comparable as a difference.

## Competence gate (pre-confirmatory feasibility, BEFORE the confirmatory run)

Greedy one-shot on all 48 items, Gemma-2-2B (eager, folded prompt): **EASY correct-members 18/24,
HARD confab-members 22/24** — both clear the ≥12 bar. Easy answers are GENUINE, not reflexive: EASY
v1 distribution `{1: 12, 2: 12}` (the model returns 1 for single-hop and 2 for two-hop by reasoning);
HARD v1 is varied `{3:9, 4:10, 5:1, 6:3, 7:1}`. A single-pass sanity check on a sample showed the
soft-capped logit-lens is still discriminative (correct items ent ≈0.3 / margin ≈3.3; confab items
ent ≈1.0 / margin ≈1.0). The bars, signals, and SURVIVED condition below are inherited VERBATIM from
the prior detection-locus preregs and are NOT tunable from the gate (greedy labels only).

## Signals (label confab=1, correct=0; HIGHER = more-likely-confab) — identical to detection-locus

1. **Resampling instability** = `1 − Stability`, Stability = `1 − (n_distinct − 1)/(N − 1)` over
   N=10 answers at T=1.0, exact-integer parse, None its own class (no judge).
2. **Single-pass clean entropy** at the first answer-token position (logit-lens, no intervention).
3. **Single-pass −margin** = `−(top1 − top2)` clean logit gap at the first answer token.

## Predictions / bars (identical to detection-locus)

- **B1 (core):** AUC(instability) `≥ 0.70`.
- **B2 / B3 (reported):** AUC(clean entropy), AUC(−margin).
- **B_contrast (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable items per group.

**Cross-family reading (pre-committed):**
- If B_contrast FAILS on Gemma too (single-pass entropy/margin separate confab from correct, AUC
  comparable to resampling), single-pass legibility on LOGIC is **cross-FAMILY** across three
  distinct pretraining lineages (Qwen, Llama, Gemma) — the confident-confabulation refutation is not
  a lineage artifact.
- If B_contrast HOLDS on Gemma (single-pass near chance, resampling dominates by ≥0.20), then the
  legibility is lineage-specific and Gemma-2 logic is where confabulation IS confident in one pass —
  locating a family boundary. Reported either way.

## Honest scope (pre-committed)

Single third-family open model Gemma-2-2B-it; multi-hop transitive-ordering logic only; one
confirmatory run; feasibility-grade; resampling N=10 at T=1.0; Stability from exact distinct-integer
counts (no judge); single-pass entropy/margin from the clean full-vocab logit-lens at the first
answer token (soft-capped — absolute values not cross-model comparable; B_contrast within-model is);
ground truth in-code then hashed pre-scoring; exact-integer correctness; system folded into the user
turn; eager attention. SAME difficulty confound as every prior detection-locus run (CONFAB hard-deep
/ CORRECT easy-shallow) — so B1/B2/B3 are difficulty-driven-wrongness detectors, not truth oracles;
B_contrast holds the confound FIXED across detector types (same items) and is the load-bearing,
cross-family-comparable result. Does NOT touch the correctness bound — every signal DETECTS
confabulation, none CORRECTS it; the detector flags abstain, never the answer.
