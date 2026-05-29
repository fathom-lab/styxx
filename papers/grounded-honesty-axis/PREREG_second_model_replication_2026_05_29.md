# PRE-REGISTRATION — second-model replication of the corrected confabulation mechanism

**Written 2026-05-29, BEFORE any code for this test is written. The single recurring caveat
across this whole arc is "one open model" (Qwen2.5-1.5B-Instruct). This run asks the one
question that caveat raises: does the *corrected* mechanism — late + tight INSTALLATION of a
confident wrong answer over an undifferentiated field, causally dampenable into honest
uncertainty (not truth) — reproduce on a DIFFERENT architecture?**

Second model: **meta-llama/Llama-3.2-1B-Instruct** (LlamaForCausalLM, 16 decoder layers,
hidden 2048, vocab 128256 — already cached). Different family, different depth, different
tokenizer from Qwen (28 layers). Same n=36 arithmetic items, same SAE-free full-vocab
logit-lens readout, same in-code arithmetic ground truth SHA-256'd pre-scoring, same
exact-integer correctness (no judge), greedy/deterministic.

## Why this run exists

Two mechanism results landed on Qwen tonight and the prior day:
1. **Suppression-rhythm (descriptive):** confabulation is the LATE (median flip hidden-idx 25
   of 29, late-fraction 0.88), TIGHT (flip IQR 4) installation of a confident wrong answer,
   and the correct token is NOT specially privileged mid-network (D1 delta −0.008, p 0.84) —
   installation over a flat field, not suppression of a computed truth.
2. **Disinhibition (causal, SURVIVED):** knocking down the decoder band [22,26] write at the
   divergence position removes the wrong commitment on 89% of confabs vs 22% for an early
   control, and what is exposed underneath is uncertainty (+7.9 nats), not truth (recovery
   0.0625). A lever on confidence/abstention, not correctness.

Both are single-model. If they are a fact about *transformer confabulation* and not a quirk of
Qwen, a second architecture should show the same shape — with its band found by **its own**
geometry, since 16 layers ≠ 28 and Qwen's absolute band [22,26] is meaningless on Llama.

## Two-stage design (both pre-registered here, run in one script)

### Stage A — MEASURE Llama's install band (suppression-rhythm geometry)

Identical logit-lens method to `run_suppression_rhythm.py`: for each one-shot confab, at the
first divergent answer position, read per-layer (final_norm + lm_head) the correct vs realized
token logits; `flip_layer` = last hidden-state index (< final) where correct still leads; a
genuine "crossing" = correct leads somewhere mid then realized wins at the final layer.

Reported (descriptive, characterize the install):
- **D2 late-localized:** late-fraction = share of crossings with flip ≥ ⌈2·(L−1)/3⌉.
- **D3 tight band:** flip-layer IQR.
- **D1 truth-specificity (control):** does the correct digit lead more than MATCHED non-correct
  single-digit distractors? (paired rate test).

### Band-derivation rule (PRE-COMMITTED — computed from Stage A, NOT hand-picked)

**[AMENDED 2026-05-29 after the n=6 PILOT, BEFORE the confirmatory n=36 run — see "Pilot
amendment" below.]** Bands are DEPTH-PROPORTIONAL (a fixed fraction of decoder depth), not a
fixed absolute layer count, because a 5-layer band is 18% of Qwen's 28 layers but 31% of
Llama's 16 and the early control then saturates. The proportional rule REPRODUCES Qwen's
published bands exactly when applied to 28 layers, so it is the same instrument re-expressed:

Let `m = round(median flip_layer)` over genuine crossings (hidden-state index, 0 = embeddings,
i = output of decoder layer i−1), `N = n_decoder`, and half-width `hw = round(2·N/28)`
(Qwen N=28 → hw=2; Llama N=16 → hw=1).
- **decoder center** `c = m − 1` (a decoder layer i writes hidden-state i+1) — from Llama's
  OWN measured median, the scientific content of Stage A.
- **TARGET_BAND = [c−hw, c+hw]** decoder layers, clamped to `[0, N−1]`.
- **control center** `cc = round(8·N/28)` (Qwen → 8; Llama → 5) — Qwen's control-center depth
  fraction, fixed, the destructiveness guard.
- **CONTROL_BAND = [cc−hw, cc+hw]** decoder layers, clamped.

Check, Qwen N=28: hw=2, m=25 → c=24 → TARGET=[22,26]; cc=8 → CONTROL=[6,10]. Exactly the
published Qwen bands. Llama N=16: hw=1, cc=5 → CONTROL=[4,6]; TARGET from Llama's own median.

### Stage B — TEST disinhibition at Llama's measured band (causal)

Identical intervention to `run_disinhibition.py`: at the divergence position, attenuate the
band's residual write `h_out → h_in + γ·(h_out − h_in)`, read the next-token distribution.
Primary γ=0; pre-named γ=0.5 fallback if target γ=0 coherence (numeric-argmax rate) < 0.50.
- **I1 — band causes the commitment:** `f_target − f_ctrl ≥ 0.30` AND `f_target ≥ 0.50` AND
  discordant sign-test (target-only vs ctrl-only removed) p < 0.05.
- **I2 — disinhibition yields UNCERTAINTY, not truth (installation branch):** among
  removed-commitment items, `truth_recovery_rate < 0.34` AND paired entropy rise p < 0.05,
  mean > 0. Reverse branch (suppression): recovery ≥ 0.50.
- **I3 — dose-response (corroborator):** Spearman ρ(γ, commitment-rate) ≥ +0.90.

## Powering / preconditions

- Stage A descriptive needs ≥ 8 genuine crossings to report D2/D3; ≥ 8 digit-position confabs
  for D1.
- Stage B: ≥ 12 usable confabs (clean teacher-forced baseline + alignable divergence) for I1;
  ≥ 6 removed for I2.
- **Band-validity precondition:** TARGET_BAND lower bound MUST exceed CONTROL_BAND upper bound
  (no overlap) AND the median flip MUST be late (m ≥ ⌈2·(L−1)/3⌉). If the install lands EARLY
  on Llama (m below the late threshold), the "late install" pattern itself fails to replicate —
  that is an informative NULL reported as such, and Stage B is reported as a non-late-install
  control (not a clean replication).

## What counts as a result

**REPLICATION SURVIVED iff ALL hold on Llama at its OWN measured band:**
- Stage A: install is **late** (D2 late-fraction ≥ 0.60) AND **tight** (D3 IQR ≤ 5) AND
  **D1 NOT truth-specific** (NOT(delta ≥ 0.20 AND p < 0.05 AND delta > 0)) — the same
  installation-over-flat-field signature.
- Stage B: **I1 ∧ I2-installation** at the measured band.

Anything short of all of the above is **REPORT_AS_LANDED**, reported against these predictions
including the specific way it diverges. A divergence is genuinely informative: it would bound
the Qwen mechanism as family-specific rather than a general fact about transformer confabulation.

## Pilot amendment (2026-05-29, transparent)

The n=6 pilot ran end-to-end and confirmed Stage A replicates strongly on Llama (install late,
flip median hidden-idx 13 of 17, IQR 1.5, band derived to decoder [10,14]). It surfaced ONE
validity failure in Stage B: with the originally-pre-registered FIXED 5-layer control band
[2,6], the control was itself fully destructive (f_ctrl = 1.0), so the I1 discordant contrast
collapsed (Δ = 0, no discordant pairs) — the destructiveness guard could not do its job on a
16-layer net. The target band was NOT blunt (γ=0 coherence 1.0), so the problem is purely the
control's absolute size on a shallow model. This is exactly what a pilot is for. The amendment
above (depth-proportional bands, reproducing Qwen's exact bands at N=28) is motivated solely by
this control-saturation validity failure and is fixed BEFORE the confirmatory n=36 run; it is
not chosen to produce any particular verdict. If the proportional control STILL saturates on
the confirmatory run, I1 is reported as untestable-on-this-architecture (REPORT_AS_LANDED), not
forced to a pass.

## Honest scope (pre-committed)

Two small open models now, not "all transformers." SAE-free logit-lens; single-position,
teacher-forced, single confirmatory run per stage; feasibility-grade n. The band on Llama is
derived from Llama's own Stage-A geometry by the fixed rule above — not transferred from Qwen
and not hand-tuned. A SURVIVED here upgrades the mechanism from "in one model" to "in two
architectures with bands each found by their own geometry"; it does NOT claim universality, and
it does NOT touch the standing correctness bound (every internal lever moves confidence; only
re-derivation moves correctness).
