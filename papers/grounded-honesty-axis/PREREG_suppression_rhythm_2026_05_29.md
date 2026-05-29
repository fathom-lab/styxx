# PREREG — Is the truth-flash TRUTH-SPECIFIC, and is the overwrite RHYTHMIC? (the control F3 was missing + the layer-frequency of suppression)

**Pre-registered 2026-05-29, BEFORE any code for this test is written or run. One
confirmatory run. Feasibility-grade: single open model (Qwen2.5-1.5B-Instruct), SAE-free
logit-lens, the SAME n=36 arithmetic items × one-shot generation as the spectral run.
Arithmetic ground truth computed in-code (`run_competence_cliff` SPECS) and SHA-256'd
before any scoring (expected to match the standing key `ddccd8e4…b87964d` — same items,
same key). Correctness = exact integer match (no judge). Greedy/deterministic.**
Receipt: `suppression_rhythm_result.json`.

## Why this run (closing the gap in our own headline)

The spectral run's headline finding (F3) was: on **78% (25/32)** of one-shot
confabulations, the **correct** answer token outranks the realized wrong token at some
intermediate layer and is then overwritten at the final layer — "confabulation is
SUPPRESSION, not ignorance." That is a strong claim, and it has an **untested control**:
maybe *many* tokens "lead then lose" mid-network — i.e. the residual stream is a noisy
competition where lots of candidates transiently outrank the eventual winner, and the
correct token leading is nothing special. If so, F3 is a dynamics artifact, not a truth
signal. **This run runs the control we owe our own headline.**

It also turns the operator's standing "focus on the frequency/rhythm" intuition into its
one *falsifiable, non-circular* form. The global-β rhythm claim is already closed (failed
K — the control token " the" showed the same β, so global slope indexes mode not truth; do
NOT re-run that). What is **not** closed is the *layer-frequency of the overwrite itself*:
is suppression a **rhythmic, localized** event (a consistent late layer band) or a
scattered one? That is answer-specific by construction (it is defined on the
correct-vs-realized divergence), so the K confound that killed global β cannot apply, and
it directly tells a future **disinhibition** intervention *where* to act.

## Apparatus (committed before data)

- **Model:** Qwen2.5-1.5B-Instruct, local, greedy decoding; reproduces the white-box
  answer-key hash.
- **Per-token logit-lens, full vocab:** at the divergence position, project the residual
  at each layer ℓ (0…L, L=28) through `model.model.norm` + `lm_head` and read the logit of
  any chosen token. Reuses `flash_crossing`'s machinery exactly (same `norm`, same `W`).
- **Divergence position p\*** (per confab item): the first answer-token index where the
  realized (wrong) token differs from the correct token, exactly as in the spectral run's
  `flash_crossing`.
- **Correct-token lead & flip layer:** at p\*, `lc[ℓ]`/`lr[ℓ]` = lens logit of the correct
  / realized token across layers. *Lead* = `lc[ℓ] > lr[ℓ]`. *Crossing* (the F3 definition,
  replicated) = correct leads at some ℓ<L AND realized wins at the final layer L. *Flip
  layer* ℓ_flip = the **last** layer ℓ<L at which the correct token is still ahead (the
  overwrite completes at ℓ_flip+1).
- **Plausibility-matched distractor set (the control):** the single-digit token ids
  `{tok(str(d)) for d in 0..9}` that tokenize to one token, **minus** the realized token
  and the correct token. At a position that must emit a digit, these are the natural
  matched competitors. Restrict D1 to confab items whose p\* has BOTH realized and correct
  as single-digit tokens (so the digit-distractor control is well-defined); report the
  qualifying n. For each distractor x, `lead(x)` = x leads the realized token at some ℓ<L
  (same definition as the correct token's lead).

## Predictions (decisive bars — pre-stated, scored exactly, both directions named)

| id | prediction | bar |
| --- | --- | --- |
| **D1 — the flash is TRUTH-specific** | the CORRECT digit leads the realized digit mid-network more often than MATCHED non-correct digits do | Δ = (correct-digit lead rate) − (mean non-correct-digit lead rate) ≥ **0.20** AND paired test (per-item: correct lead vs mean-distractor lead) p < **0.05**, correct higher. If Δ ≈ 0 (in [−0.05, 0.20) or p≥0.05) → the flash is **generic competition, not truth-specific** → F3 is DEFLATED to a dynamics artifact, reported as such. If Δ < −0.05 → correct leads *less* than distractors → strong deflation. |
| **D2 — suppression is LATE-localized** | on genuine correct-token crossings, the overwrite completes in the late layers | fraction of crossings with ℓ_flip ≥ **⌈2L/3⌉** (=19 of 28) is ≥ **0.60**. If < 0.60 / flips spread or early → suppression is **not** a late-localized event; reported against prediction. |
| **D3 — suppression is RHYTHMIC (tight band)** | the flip layer is consistent across items, not scattered | IQR(ℓ_flip) ≤ **5** layers → rhythmic/localized. If IQR > 5 → scattered. **Corroborator, not a SURVIVED gate.** |

**RESULT = SURVIVED iff D1 ∧ D2.** D3 corroborates. Otherwise REPORT_AS_LANDED with
whatever held, scored against prediction.

## Precondition / honest failure modes (stated in advance)

1. **Under-powered control stratum.** If fewer than **8** confab items have a
   single-digit-token p\* for both correct and realized, D1 is under-powered → reported
   descriptively, no SURVIVED claim on D1.
2. **D1 is the genuine gamble and the point of the run.** A D1 **null deflates our own
   headline** — that is exactly why it must be run. I will not soften a null. The honest
   prior: I expect the correct token to lead *somewhat* more than matched digits (the
   spectral examples — 5535, 9306, 13824, 74412 — looked truth-specific), but a generic
   competition story is live and a Δ near zero is a real possibility.
3. **D2/D3 likely but not guaranteed.** F3 already implies a late overwrite, so D2 is
   probable; whether it is *tight* (D3) is genuinely uncertain.
4. **ℓ_flip is defined only for genuine crossings.** Non-crossing confabs (correct never
   led, ~22%) are excluded from D2/D3 by construction and counted separately.

## Honest scope (pre-committed)

Single open model, SAE-free full-vocab logit-lens, feasibility-grade n=36, one confirmatory
run; arithmetic ground truth computed in-code then hashed; exact-integer correctness, no
judge. This tests (a) whether the F3 truth-flash is specific to the *correct* token versus
matched digit distractors, and (b) the layer-localization/consistency of the overwrite
event — nothing more. It is a logit-lens phenomenon on the *realized* run; it is **not** a
causal demonstration that dampening the overwrite recovers the answer (that remains the
named disinhibition test, which D2/D3 would target). A D1 null does not refute that the
model computes the answer transiently — it would refute that the *correct token* is
privileged in the mid-network competition, which is the specific thing F3 asserted.
