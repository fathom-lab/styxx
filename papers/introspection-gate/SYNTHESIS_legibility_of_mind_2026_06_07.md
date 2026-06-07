# The Legibility of Mind — what we found (2026-06-06 → 07)

**Fathom Lab / styxx.** A night mapping *who can read a thought inside an AI* — the mind itself,
an external probe, another mind — and whether you can write one across. Every claim pre-registered,
hashed before scoring, with positive controls that could (and did) catch broken instruments.

## One picture

Inject a concept directly into a model's residual stream (steering-validated: it measurably moves
generation toward the concept). Then ask every possible reader what's there.

| reader | can it read the planted thought? |
|---|---|
| the **unembedding lens** | yes, ≈1.00 |
| an **external probe** of the model's *clean processed downstream* state | **yes, 1.00** (0.5B/1.5B/3B; perm-null chance) |
| **another model**, zero-anchor (no shared key) | **partially** — top1 0.15–0.48, graded by isometry |
| the **mind itself**, forced-choice (can't dodge) | **no, ≈chance** (0.08–0.21, 0.5B→7B) |

**The mind is the least reliable reader of itself.** A thought can be 100% legible to outside
instruments — and even partially legible to a *different* mind — while the mind that holds it is
blind. This is scale-robust across a 14× parameter range.

## The four results

1. **Self-legibility — the inaccessible thought (ROBUST WIN).** Forced-choice symbol-code (arbitrary
   letters, so a logit-lens can't win): the model identifies its own injected concept at **chance**,
   while an external probe on a *clean, un-injected read position* hits **1.00**. Controls all fire:
   prime ≈1.00 (channel works when handed the answer → apparatus valid), clean-read-position and
   wrong-layer collapse, concept-word lens dissociates. Holds 0.5B→7B. *The white-box edge is in
   reading CONTENT/IDENTITY the model cannot report.*

2. **Mutual legibility — zero-anchor cross-model reading (REAL POSITIVE; corrected a prior null).**
   With a Wasserstein-Procrustes aligner *validated by a positive-control calibration the prior run
   lacked*, real differently-trained models are partially alignable with no paired data, far above
   chance (1/160): Llama-3B↔1B (RSA 0.97) top1 0.48; Llama-3B↔gemma-2b (cross-family) top1 0.15 /
   top5 0.43. *Caveat:* the same-family 0.48 is **at** its RSA-matched calibration floor (0.51) —
   as alignable as its RSA predicts, not beyond-noise; the **cross-family** pair is the one that
   **beats** its floor (0.15 vs 0.089) = real structure beyond noise. WProc-specific (entropic-GW
   gives ~5× lower); chance below RSA≈0.7. The prior "not alignable (0.05)" was an un-validated-GW
   artifact — the correction stands.

3. **No clean *writing* — control transfer is instrument-limited / suggestive (NOT a landed
   falsification).** The same map that partially *reads* did not measurably *write*: a transferred
   steering direction steers the target at NTE 0.22 (≈4× weaker than native), beating random 14/21
   — **but that is not significant** (sign-test p=0.19; only paired-t p=0.03). And per the frozen
   prereg the result is **ALIGNER_LIMITED**: the A→A-rot concept-direction positive control reached
   only 0.74, *below* the pre-registered G0 = 0.80, so the cross-model null is not cleanly
   interpretable. So **"reading ≠ writing" is a SUGGESTIVE hypothesis here, not a landed result** —
   the honest read is "writing was not demonstrated; the instrument fell short of its own gate."
   (The mechanism — steering is more sensitive to map imprecision than reading — remains the
   leading explanation, to be confirmed by raising the instrument past G0.)

4. **The tamper detector is a perturbation meter (HONEST NULL, precisely characterized).** Dose-
   response: the probe detects steering across all strengths (0.98–1.00) and *beats self-report at
   subtle doses* (gap +0.31 at α=1, where the model can't feel it) — BUT in exactly that subtle
   regime it **cannot distinguish a real injection from a benign "be concise" instruction**
   (attack-vs-benign 0.00 at α=1). The two needed properties (beat self-report; separate
   attack-from-benign) are **anti-correlated across dose with no sweet spot**. So white-box
   *perturbation-presence* detection is **not** a clean attack discriminator — the geometry-
   detector's grave, now mapped. (The genuine edge remains CONTENT-IDENTITY, result #1, not flagging.)
   *Note:* in the tamper JSON, probe AUROC 1.00 and benign-meta FP 0.95 coexist **by design** —
   AUROC is rank-based (attack ranks above meta) while the FP uses a train-fixed absolute threshold;
   the dose-response, not the single-threshold FP, is the load-bearing read.

## The unifying law: READ ≠ WRITE (within a mind and across minds)

The night's results cohere into one dissociation. **Legibility and controllability come apart:**

| | READING (recover the content) | WRITING (install / restore / transfer) |
|---|---|---|
| within a model | ✓ probe 1.00 (injected); DELTA 0.30 (suppressed knowledge) | ✗ un-cave null (correction closed even for suppression) |
| across models | ✓ zero-anchor partial, isometry-graded (0.15–0.48) | ✗ control-transfer null |

You can *see* what a mind holds — an injected state, its known-then-suppressed answer, even
(partially) another mind's concepts with no shared key — but you **cannot write it back** with a
general linear intervention, whether *within* a mind (restoring a caved answer) or *across* minds
(transferring a control). Reading tolerates an imprecise map/representation; writing does not. Both
write-side nulls hold with the apparatus *demonstrably alive* (it perturbs outputs, breaks held-correct
items) and the content *demonstrably present* (the read side is positive) — so this is a real
dissociation, not a dead instrument. (Findings: `papers/grounded-honesty-axis/FINDING_uncave_2026_06_07.md`,
`papers/disjoint-worlds/FINDING_thought_transfer_2026_06_07.md`.)

*Hardest test of the law:* even **coupling the read to the write** (probe decodes the suppressed
letter per item, then steer toward it) does not break it cleanly — read-coupled write beats the
*unconditional* direction (0.40 vs 0.00) but is only **+0.12 over generic random-letter steering** and
**destroys half the held-correct answers** (RESISTED-break 0.53). The suppressed answer is readable
but not cleanly or safely writable, even conditionally. read ≠ write stands; no sycophancy antidote
(`papers/grounded-honesty-axis/FINDING_readwrite_2026_06_07.md`).

## Dogfood — the instrument on its maker

styxx's shipped honesty audit was run on Claude's *own verbatim claims from this session*
(`FINDING_dogfood_self_audit_2026_06_07.md`). It flagged the **hype framing** hardest (overconfidence
0.922 — the most "revolutionary"-sounding sentence) while the calibrated `needs_revision` gate passed
all scoped claims and sycophancy stayed low. Action taken: the public framing here is the
honest-scoped version, not the hype the instrument flagged. The standard, run on its author, and obeyed.

## The strike at the named-impossible: the wall is INTERNAL

We pointed the whole program (knowledge-presence read + dynamics) at styxx's deepest open problem —
the **confident-misconception wall** (where a model believes a falsehood and all output-based
detection fails). On a powered, grown confident-consistent floor (n=121), a label-free mid-layer
logit-lens read of the teacher-forced **gold** answer is at **chance (R_gold AUC 0.519)** — the
suppressed truth is **not internally favored** for confident misconceptions. Output is at chance
(0.53) as always; the only faint signal (K=0.62, borderline, at the bar) is the emitted-wrong-answer
being mid-layer-*weaker*, not recovered knowledge. **Verdict: REPORT_AS_LANDED — the wall is *internal*,
not merely behavioral.** The field couldn't crack it from the output because the error is internally
confident; we now have evidence it can't be cracked from the *inside* either, because for a believed
misconception **there is nothing to recover** (the myth is encoded *as* knowledge). Bedrock, not a
crack — and refusing to call the borderline 0.62 a "crack" is the dogfood lesson lived
(`papers/grounded-honesty-axis/FINDING_wall_internal_2026_06_07.md`).

## The honest claim (vs the hype)

- **Real:** a model cannot **introspectively identify** an injected concept (forced-choice ≈ chance)
  while an external probe decodes it (Qwen-3B 1.00) — a content-*identification* gap, scale-robust on
  the forced-choice side (0.5B→7B), probe side validated on Qwen. *Scoped honestly:* this is about
  identifying *what* is in the state, NOT detecting *that* something is off — for mere detection a
  forced YES/NO self-report works at deployment dose. It still cuts against the *ask-the-model*
  paradigm for any task needing the model to report *what* it represents (and maps to EU AI Act
  Art. 15 — oversight can't rest on the system identifying its own state).
- **Real:** minds are *partially mutually readable* with no shared key (legibility, isometry-graded;
  cross-family beats its noise floor).
- **NOT claimed (the controls / the prereg killed these):** "we can install a thought in another
  mind" (writing **not demonstrated** — instrument fell below its own prereg gate); "we built a
  white-box jailbreak/tamper detector" (it's a perturbation meter); "small models are
  introspectively aware" (they can't identify their own injected content).

## Discipline receipts

Every positive control earned its place: the WProc aligner was caught returning 0.00 on a known
rotation (random-direction artifact) before any cross-model claim; the prior legibility null was
caught as instrument-limited; the tamper detector was caught as a magnitude meter by the
benign-instruction control; the transfer was caught as underpowered (floored dose) before being
fairly re-run. **The instrument that draws the line is the product.**

## Scope

≤7B open models; injected *linear* concept directions; MiniLM steering metric; shared training data
not controlled (cross-model is the practical, not data-independent, frontier). Local, $0, one 8GB GPU.

## Status

All local on `feat/spec-exec-router`, uncommitted. Findings: `papers/introspection-gate/FINDING_v1/v2/v3`,
`papers/disjoint-worlds/FINDING_real_legibility`, `FINDING_thought_transfer`, this synthesis.
