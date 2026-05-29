# PRE-REGISTRATION — is single-pass confabulation legibility CROSS-ARCHITECTURE or Qwen-specific?

**Written 2026-05-29, BEFORE the Llama run.** Tonight's detection-locus run
(`FINDING_detection_locus_2026_05_29.md`, Qwen2.5-1.5B-Instruct) found that confabulation is
internally legible in a SINGLE forward pass: clean first-token entropy (AUC 0.924) and logit
margin (0.915) separate confab from correct almost as well as N=10 resampling instability
(0.980) — so the pre-registered "confident confabulation" hypothesis was FALSIFIED *on that
model/domain*. The standing open question (flagged in that finding's honest scope): is this a
property of Qwen specifically, or does single-pass legibility hold across architectures?

This run replicates the detection-locus protocol UNCHANGED on a SECOND architecture —
**meta-llama/Llama-3.2-1B-Instruct (16 decoder layers)** — same arithmetic item set, same three
detector scores, same bars. All signals read from the SAME (Llama) network.

**Does single-pass internal confidence (clean entropy / logit margin) separate confab from
correct on Llama too, or is the legibility Qwen-specific?**

## Item set (balanced, pre-committed — identical construction to the Qwen run)

- **CONFAB group** = realized one-shot greedy confabs from the 36 hard `SPECS`
  (`v1 != correct`, `v1 is not None`), on Llama.
- **CORRECT group** = the 24 `EASY_SPECS`, greedy `v1 == correct`, on Llama.

Group membership uses the GREEDY one-shot label on Llama. NOTE the competence shift: Llama-1B is
a different (and on arithmetic, likely weaker) model than Qwen-1.5B, so the easy set may yield
fewer correct items and the hard set a different confab mix. Powering is reported, not assumed.

## Signals (label confab=1, correct=0; every score oriented so HIGHER = more-likely-confab)

1. **Resampling instability** = `1 − Stability`, Stability = `1 − (n_distinct − 1)/(N − 1)` over
   N=10 answers sampled at T=1.0; exact-integer parse, None its own class (no judge).
2. **Single-pass clean entropy** — Shannon entropy of the clean next-token distribution at the
   first answer-token position (logit-lens, γ=1, no intervention).
3. **Single-pass −margin** — `−(top1 − top2)` clean logit gap at the first answer token.

## Predictions / bars (identical to the Qwen detection-locus run)

- **B1 (core):** AUC(instability; confab vs correct) `≥ 0.70`.
- **B2 / B3 (reported):** AUC(clean entropy), AUC(−margin).
- **B_contrast (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable items per group.

**Cross-architecture reading (pre-committed):**
- If B_contrast FAILS on Llama too (single-pass entropy/margin separate confab from correct, AUC
  comparable to resampling), then single-pass legibility is **architecture-GENERAL** on
  small-model arithmetic — the "confident confabulation" refutation is not a Qwen quirk.
- If B_contrast HOLDS on Llama (single-pass near chance, resampling dominates by ≥0.20), then
  legibility is **Qwen-specific**, and Llama is the model where confabulation IS confident in a
  single pass — locating the boundary. Reported either way.

## Honest scope (pre-committed)

Single second open model Llama-3.2-1B-Instruct; arithmetic only; one confirmatory run;
feasibility-grade; resampling N=10 at T=1.0; Stability from exact distinct-integer counts (no
judge); single-pass entropy/margin from the clean full-vocab logit-lens at the first answer
token; ground truth computed in-code then SHA-256'd pre-scoring; exact-integer correctness. SAME
difficulty confound as the Qwen run (CORRECT easy / CONFAB hard) — so B1/B2/B3 are
difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the confound FIXED
across detector types (same items) and is the load-bearing, cross-architecture-comparable result.
Does NOT touch the correctness bound — every signal DETECTS confabulation, none CORRECTS it; the
detector flags abstain, never the answer. A confab-vs-correct AUC here is NOT a within-mode truth
signal (correct/confab are different-difficulty items, per the keystone).
