# Per-Call Validity for Cognometric Instruments: A Pre-Registered Map of Where Model-Internal Confidence Predicts Reliability — and Where It Doesn't

**Alex Rodabaugh · Fathom Lab · 2026-05-24**
*Reference implementation: `fathom-lab/styxx`. All hypotheses pre-registered and
hashed before data; commit chain cited inline.*

---

## Abstract

A cognometric instrument scores a cognitive state — refusal, hallucination,
sycophancy — from a model's output. But a score with no error bar silently
extrapolates beyond its calibration domain: the user cannot tell whether to
trust it *on this input*. We ask whether an instrument's score can carry a
calibrated **per-call validity** — a 0–1 estimate of its own reliability — and,
if so, from what substrate. Across four pre-registered hypotheses (Spearman
ρ(validity, −error) ≥ 0.40, p < 0.01, holdouts hashed before scoring, each run
once), we find: **(1)** validity from prompt-embedding distance fails (ρ = 0.30);
**(2)** validity from the model's own generation logprobs works for the *refusal*
instrument's over-flagging, generalizes across five models including one
cross-family open model, but **attenuates** out of family (ρ 0.58 → 0.29);
**(3)** it does **not** generalize to the *hallucination* instrument (ρ = −0.18).
The negative is mechanistic, not incidental: it is **confident confabulation** —
models are frequently confident *when they are wrong*, so generation confidence
cannot flag the failure mode one most needs caught. The result is a sharp,
pre-registered map: **model-internal confidence predicts a cognometric
instrument's reliability iff that instrument's errors are driven by generation
uncertainty, and not where they are driven by confident error.** We release the
one green-cell capability — a logprob-grounded over-flag reliability flag for
refusal — and decline the universal claim the evidence does not support.

---

## 1 · The problem

styxx (and any observability layer) emits numbers: `audit.refusal = 0.87`. A
user quotes that as a fact about the model. But the threshold-law for
cognometric transport [TL] shows reliability is domain-bounded — above a
corpus-overlap threshold a score is reliable; below it, it degrades along a
measured curve. Today that information lives in a paper, not in the API. Every
call silently extrapolates whenever the input leaves the calibration domain.

The prior release (styxx 7.4.1) fixed the *language* of scope-honesty — the
README discloses each instrument's construct ceiling. This work asks the next
question: can scope-honesty **compile into runtime**, as a per-call `(score,
validity)` pair? The answer turns out to be *partly* — and the boundary is the
contribution.

## 2 · What was already settled (not under test)

- **Threshold-law transport** [TL]: same-family cognometric transport is governed
  by a vendor-agnostic corpus↔domain-overlap threshold (τ ≈ 0.31). We replicated
  the curve from committed data before relying on it: cross-family Spearman
  +0.6865 (published +0.6923), τ = 0.31 exact, scipy-confirmed (`04a8904`).
- **Construct ceilings**: styxx's text-only deception and overconfidence
  instruments are pre-registration-killed closed negatives — they read register,
  not the construct. The repeatedly-named escape was *model-internal signal*.
  This work is the first clean test of that escape for the validity problem.

## 3 · Method and pre-registration discipline

Four hypotheses, each committed to git **before** its holdout existed; each
holdout's SHA-256 committed **before** the instrument scored it; each analysis
run **once** (no peeking, no optional stopping); the kill-gate bar (ρ ≥ 0.40,
permutation p < 0.01) never moved. The commit chain is the credibility: the
boundary is trustworthy because we tried to cross it and reported, each time,
that we could or couldn't.

For each instrument *I*: `error_I(prompt, response) = |score_I − gold_I|`;
`validity` is a monotone function of the candidate substrate; the test is
Spearman ρ(validity, −error). Because validity is monotone in its substrate, the
Spearman is invariant to the exact transform — the test is whether the substrate
*rank-predicts* instrument error.

A **pre-declared confound control** runs throughout: response classes can carry
near-zero instrument error (nothing to predict), so we report ρ **within class**
and require the signal to survive there, not merely pool.

## 4 · Results

| # | substrate | instrument | n | pooled ρ | verdict | commit |
|---|---|---|---|---|---|---|
| H1 | embedding distance → calibration corpus | refusal | 450 | +0.30 (p<10⁻⁴) | **CLOSED NEGATIVE** | `ebe0475` |
| H1b | mean token logprob (generation confidence) | refusal (gpt-4o-mini) | 450 | +0.73 | PASS | `c3b1b33` |
| H1c | mean token logprob | refusal × 5 models | 5×450 | 0.58–0.75 | **BOUNDED** (see §4.3) | `80a81c6` |
| H1d | mean token logprob | hallucination | 450 | −0.18 | **CLOSED NEGATIVE** | `44b9c5c` |

### 4.1 Embedding distance fails (H1)

Validity computed from a prompt's k-NN distance to the refusal calibration corpus
(the threshold-law domain pool), gold via a vendor-robust labeler (cross-checked
against XSTest human labels, 0.944 agreement). ρ = 0.3024 (labeler) / 0.3118
(human), both p < 10⁻⁴, both below the 0.40 bar. The signal is real but
insufficient: the corpus-level threshold-law does not transfer to the prompt
level. Static text geometry is too weak a substrate.

### 4.2 Generation logprobs work for refusal (H1b)

Validity from the model's mean token logprob (its own generation confidence)
clears the bar on gpt-4o-mini: pooled ρ = 0.734, and it survives the confound
control (within-compliance 0.575, within-refusal 0.437).

### 4.3 …but only for over-flagging, and it attenuates cross-family (H1c)

Across gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-4.1, and the cross-family
Qwen2.5-1.5B-Instruct, the **pooled** ρ replicates (0.58–0.75) — but the
confound control shows it is class-mediated. The within-**refusal** class is
uninformative on every model (`refuse_check` is near-perfect on refusals, mean
|error| 0.01–0.04 — no error variance to predict; gpt-4o-mini's within-refusal
0.44 was a low-variance fluke the cross-model test exposed). The real signal
lives in the within-**compliance** class — where the instrument *over-flags*
safe prompts — and there it is positive on all five models: **0.575 / 0.460 /
0.506 / 0.440 / 0.294**. It is genuinely model-general, including cross-family,
but it **attenuates** in the small open model. Under the locked rule (both
within-classes ≥ 0.20), 0/4 additional models pass; the strict "clean both-class"
replication does not hold and was not claimed.

### 4.4 It does not generalize to hallucination (H1d)

On HaluEval-QA with closed-book gpt-4o-mini responses (gold by a gpt-4o judge
validated at 0.90 against known right/hallucinated pairs), the hallucination
instrument's reliability is **not** predicted by generation confidence: pooled
ρ = −0.184 (the opposite of the gate). Within correct answers the refusal
pattern recurs (ρ = +0.476), but within hallucinations confidence predicts
nothing (ρ = −0.027), and the confident-but-wrong cases drive the pooled
correlation negative.

## 5 · The mechanism: confident confabulation

The negative is the finding. Generation confidence predicts an instrument's
reliability **only where that instrument's errors are driven by generation
uncertainty.** For refusal, an uncertain generation produces ambiguous text the
detector mishandles — low confidence flags low reliability. Hallucination breaks
this: **the model is confident when it is wrong.** The failure mode one most
wants a validity signal for — confident falsehood — is exactly where confidence
carries no information, *because* the model is confident in it.

This is a clean limit on a tempting idea. "Use the model's own logprobs as a
universal reliability oracle" is intuitive and, for a class of failures, correct.
It is also wrong for the most dangerous class, and wrong in a way no amount of
calibration repairs, because the substrate itself is decoupled from correctness
there.

## 6 · What is closed-negative

The **universal reliability oracle** — one validity signal across cognometry — is
pre-registration-killed across *both* substrates tested (embedding distance,
model logprobs) *and* both instruments (refusal, hallucination). It is not
claimed. This joins styxx's pre-registered negative chain (text-only deception;
text-only overconfidence; cross-vendor transport universality).

## 7 · What ships

One green cell yields a real capability: a **logprob-grounded over-flag
reliability flag for refusal** — given a refusal score and the generation's
confidence, it estimates whether the score is a false positive on a safe prompt.
On held-out data the over-flag risk drops monotonically with the flag's validity
(e.g., gpt-4o 0.33 → 0.18 low→high validity; it catches `refuse_check`
mislabeling a benign cartoon question). It scopes to that one failure mode — not
universal validity. Reference prototype: `papers/grounded-arc/overflag_validity.py`.

## 8 · Limitations

- **One green instrument.** Only refusal over-flagging cleared the bar; one
  positive cell does not establish a class.
- **Family attenuation.** The signal weakens from OpenAI (ρ ≈ 0.5) to a small
  open model (0.29); behavior on larger open models is untested.
- **Gold dependence.** Refusal gold used an offline labeler; hallucination gold
  used an LLM judge (validated 0.90) — judge error is a noise floor on H1d.
- **Pooling artifact (methodological).** A near-perfect response class both
  inflated the pooled headline (via class mediation) and failed the within-class
  control (via noise). Future tests should be defined on the **error-bearing
  subset** of each instrument, not pooled across a near-perfect class.
- **Single session.** These are first results; the map's green cell warrants
  independent replication before a product depends on it.
- **The negative is substrate-specific.** H1d rules out *generation-confidence
  (token logprobs)* as a validity substrate for hallucination — not all
  model-internal signal. Hidden-state probes and self-consistency sampling are
  untested and are the natural next substrate. This work does not claim
  hallucination-validity is impossible, only that token logprobs cannot supply it.

## 9 · Implications

For the field: per-call validity from cheap surface signals (embedding distance)
is a mirage; from model-internal confidence it is real but narrow and
mechanism-bounded. Anyone shipping a "confidence = reliability" layer should
expect it to go silent exactly on confident errors.

For self-monitoring agents: an agent's own confidence **cannot** be a universal
self-check, because its most dangerous errors (confident hallucination) are the
ones it is confident about. Targeted self-validation where errors track
uncertainty is viable; universal self-trust from logprobs is not.

## 10 · Reproducibility

Pre-registrations (each committed before its data): `preregistration_2026_05_19.md`
(`29874f2`; operator decisions amended/locked `53269f6`, holdout hashed `060ef7a`),
`preregistration_bet0b_2026_05_24.md` (`5ef0dc2`),
`preregistration_crossmodel_2026_05_24.md` (`e580964`),
`preregistration_crossinstrument_2026_05_24.md` (`2c71aa0`). Holdout hashes,
per-finding notes, and run scripts under `papers/grounded-arc/`. Data: XSTest-v2
(refusal), HaluEval-QA (hallucination). Models: gpt-4o-mini / gpt-4o /
gpt-4.1-mini / gpt-4.1 (logprobs via the OpenAI API), Qwen2.5-1.5B-Instruct
(local, true token logprobs).

## 11 · Conclusion

We set out to make styxx's scores self-disclose their reliability. The honest
result is a map, not a magic number: model-internal confidence is a real
per-call validity substrate for refusal over-flagging, generalizing across model
families but attenuating, and it does **not** extend to hallucination because of
confident confabulation. We ship the narrow capability and decline the universal
claim. The four held kill-gates are the point: in a field that ships validity
from cosine distance and calls it honesty, a pre-registered demonstration of
*where per-call validity is real and where it is a mirage* — with the mechanism
named — is the contribution worth standing behind.

---

*[TL] Threshold-law for cognometric transport, Zenodo `10.5281/zenodo.20278945`.*
*styxx is MIT-licensed; calibrated atlas data CC-BY-4.0.*
