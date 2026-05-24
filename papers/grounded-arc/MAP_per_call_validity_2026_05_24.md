# Per-Call Validity for Cognometric Instruments — a Pre-Registered Map

**Fathom Lab / styxx · grounded-arc · 2026-05-24**

## The question

styxx scores a cognitive state (refusal, hallucination, …) from a model's
output. But a score with no error bar silently extrapolates: the user can't tell
whether to trust it *on this input*. The 7.4.1 release made styxx's scope-honesty
true in the README; this arc asked whether it can be made true **per call, at
runtime** — can an instrument's score carry a calibrated **validity** (reliability)
signal?

Four hypotheses, each pre-registered before its data existed, each with a hard
kill-gate (Spearman ρ(validity, −error) ≥ 0.40, p<0.01), each hashed before
scoring and run once. The bar never moved.

## Results

| # | substrate | instrument | result | ρ | commit |
|---|---|---|---|---|---|
| H1 | embedding distance to calibration corpus | refusal | **CLOSED NEGATIVE** | +0.30 | `ebe0475` |
| H1b | model logprob (generation confidence) | refusal | PASS (gpt-4o-mini) | +0.73 | `c3b1b33` |
| H1c | model logprob | refusal × 5 models | **BOUNDED** — pooled replicates (0.58–0.75) but class-mediated; the real signal (compliance over-flagging) is model-general incl. cross-family but **attenuates** (0.58→0.29) | — | `80a81c6` |
| H1d | model logprob | hallucination | **CLOSED NEGATIVE** | −0.18 | `44b9c5c` |

## The discovery — a mechanism, not just a boundary

**Model-internal confidence predicts a cognometric instrument's reliability *iff*
the instrument's errors are driven by generation uncertainty — and not where they
are driven by confident error.**

- **Refusal — works.** An uncertain generation yields ambiguous text the detector
  mishandles; low confidence flags low reliability. It even holds cross-family,
  though it weakens in a small open model.
- **Hallucination — fails, and instructively.** The model is frequently
  **confident when it is wrong** ("confident confabulation"). Within correct
  answers the refusal pattern recurs (ρ=+0.48); within hallucinations confidence
  predicts nothing (ρ≈0); pooled, the confident-but-wrong cases drive ρ negative.
  Confidence cannot flag the failure you most need caught, *because* the model is
  confident in it.

**Embedding distance fails everywhere** (ρ=0.30): static text geometry is too weak
a substrate; what carries the signal is the model's *dynamic* generation
confidence, where it carries it at all.

## What is closed-negative

The "universal reliability oracle" — one validity signal for all of cognometry —
is pre-registration-killed across **both** substrates tested (embedding vs logprob)
**and** both instruments (refusal vs hallucination). It is not claimed.

## What survives, and ships honestly

A **real but narrow** capability: a logprob-grounded reliability flag for **refusal
over-flagging** — telling a caller when a refusal score is likely a false positive
on a safe prompt. Demonstrated on real data (over-flag risk drops monotonically
with the flag's validity; it catches `refuse_check` mislabeling a Donald-Duck
question). Prototype in `overflag_validity.py`. If shipped into styxx 8.0, it
scopes to that one failure mode — not universal validity.

## Implication for "agents that validate themselves"

An agent's own confidence **cannot** be its universal self-check: its most
dangerous errors (confident hallucination) are exactly the ones it is confident
about. Self-validation is viable where errors track uncertainty (refusal-style),
and must stay silent about claiming the cases where they don't.

## Why this is the contribution

The map is the result. In a field that ships validity numbers from cosine
distance and calls it scope-honesty, this is a pre-registered demonstration of
*where per-call validity is real and where it is a mirage* — with the mechanism
named. The four kill-gates, bars held, are the credibility: the boundary is
trustworthy because we tried to cross it and reported, each time, that we couldn't.

A future bet targets the specific (instrument, failure-mode) cells the map marks
green — not the universal claim the map marks red.
