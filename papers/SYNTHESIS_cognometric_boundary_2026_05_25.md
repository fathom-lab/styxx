# The Cognometric Boundary — what text reveals, what it conceals, and where grounding still fails

**2026-05-25 · A synthesis of styxx's wins, closed negatives, and this session's
bets.** Drawn from the instrument calibrations and the documented failure modes in
the codebase — most of the map is built from *negative* results the field doesn't
publish. This is a lens, not a theorem (see "What would falsify this").

## The thesis

Across every styxx instrument and every pre-registered bet, one line keeps
appearing:

> **Text surface features measure FORM — register, structure, repetition, novelty —
> with high accuracy. They cannot recover TRUTH, CALIBRATION, or STANCE from text
> alone. That second tier is crossable only with GROUNDING (a reference, the model's
> internals). And there is a third tier — CONFIDENT ERROR — where even grounding in
> the model's own signal fails, because the model is confident when it is wrong.**

A hardness hierarchy. styxx's job has been to map exactly where each cognometric
question sits on it.

## Tier 1 — FORM (measurable from text alone; the wins)

| instrument | AUC | the form signal (critical-K=1 feature) |
|---|---|---|
| conversation-loop (stagnation) | 0.9995 | pairwise Levenshtein |
| hallucination (confabulation) | 0.998 | trigram novelty |
| sycophancy (flattery *register*) | 0.972 | superlative density |
| goal-drift | 0.965 | anchor↔last n-gram overlap |
| tool-call drift | 0.943 | schema/arg mismatch |
| plan-action gap | 0.923 | plan↔action bigram overlap |
| refusal (apologetic *style*) | 0.78–0.79 | "sorry/can't" opener |

Every one is a surface property: *how the text is shaped*. This is also why the
self-vs-other gate (7.5.0) worked — direction is encoded in grammar (pronoun
attachment), so it is Tier-1 form. Text is excellent at form.

## Tier 2 — TRUTH / CALIBRATION / STANCE (text-only fails; grounding crosses it)

| question | text-only result | grounded result |
|---|---|---|
| Is the confidence *calibrated*? (overconfidence) | recal H_null, held-out 0.57–0.60 (`7c36ed9`) | needs logprobs/entropy |
| Is the claim *true*? (deception) | TruthfulQA **0.59** ≈ chance (`0ad384e`) | NLI vs reference **0.818** |
| Is this an opinion or a fact? (restrained FP) | lexical **closed neg** (C3/C4, 47% recall) | semantic embeddings **1.00** (C5, shipped 7.6.0) |
| Is the agreement *grounded-true*? (decoupled-diagonal) | prompt FORM ≠ premise TRUTH (capstone reverted) | reference-NLI works *only* on assertion prompts |
| Disagree-"not" vs agree-"not"? (negation residual) | **closed neg** (probe: "you're not wrong" 0.91→0.01) | needs semantic stance |
| Per-call reliability? (validity) | embedding-distance ρ=**0.30** | logprob ρ=0.73 (refusal only) |

The pattern is exact: the moment the question requires knowing whether something is
*true*, *calibrated*, or *agreed-with-against-evidence*, surface text closes
negative — and a grounding signal (reference, NLI, embeddings, logprobs) is what
crosses the line. Every "win" this session was a grounding crossing (C5 semantic
subjectivity); every closed negative was a text-only attempt at Tier 2.

## Tier 3 — CONFIDENT ERROR (even grounding fails; the frontier)

The deepest ceiling, from the grounded-arc program: **logprob-validity is
refusal-specific and dies on hallucination** — within correctly-answered items
ρ=+0.48, but within hallucinated items ρ≈0. *The model is confident exactly when it
is confabulating.* So model-internal confidence cannot flag the errors that matter
most. And the capstone's bare-question failure is the same shape: a fluent, confident
correction/lie is indistinguishable by form, and the grounding signal (NLI) itself
gets confused by surface form (leading "No"). Tier 3 is where styxx's frontier
genuinely sits, and it is not crossed.

**Update (2026-05-25): the across-sample substrate partly CROSSES — Tier 3 is not as
dark as the rest of this doc claims.** A pre-registered probe tested **semantic
entropy** across N samples (Farquhar-Nature-2024: confabulation is *unstable*, so
divergence flags it). A first pass using a **cosine-similarity** clustering proxy
returned AUC 0.55 and we briefly (wrongly) recorded a negative — but that was a
clustering artifact: the model tells a *different* lie each sample ("Renwick reached
in 1842 / 1723 / 1912…"), and the conventional 0.70 cosine threshold lumps the
template-sharing lies (pairwise cosine 0.70–0.93) as "the same." A threshold sweep on
identical samples recovers **AUC 0.93 by cosine 0.95** and **0.95 with NLI entailment**
(threshold-free) — robust across any cutoff ≥0.8. The lever WORKS; the 0.55 was a
too-lenient-threshold artifact, not a property of confident error. **Cross-model
(feasibility): the mechanism generalizes** — gpt-4o-mini and gpt-3.5-turbo confabulate
*inconsistently* (cosine@0.90 AUC 0.89–0.92, entropy 7–9× correct), while gpt-4o mostly
*abstains* (62.5%) instead of inventing. No consistent-confabulation floor appeared. So
Tier 3's confident-confabulation case is a genuine *partial crossing*, not dark: the
model's confidence hides the error from single-response signals, but its inconsistency
exposes it across samples — and the strongest model increasingly just declines.
Confident confabulation is *inconsistent*, not stable; single-response confidence
misses it (closed) but across-sample semantic divergence catches it. This is the
first partial Tier-3 crossing in the program. Caveats: gpt-4o-mini only, n=4
confabulations, and a real false-positive mode where the model flip-flops between
abstaining and confabulating (florium/zylophane). See
`papers/tier3-confident-confabulation/verify_clustering.py` +
`FINDING_corrected_2026_05_25.md`.

## The meta-confirmation (the instrument on its builder)

Run on its own builder this session, the instrument flagged the *most anti-
sycophantic sentence* ("no, it's not novel") as sycophantic (0.69), and read all
careful declarative text as overconfident (0.75–0.95). Its residual failure mode is
**register** — it reads *how the agent sounds*, not whether the agent is honest. That
is Tier 1 looking back at us: the tool is a form detector, and its blind spot is
precisely the form/truth boundary it was built to chart. (F10, confirmed from
inside.) Even styxx's own product pitch scores low sycophancy but high
overconfidence-register by its own instrument — form, not flattery.

## What this means for the product (honest frame)

styxx is not a lie detector and never was. It is a **calibrated form-detector with
optional grounded tiers**:
- Tier-1 instruments (sycophancy register, refusal, loop, drift) ship pure-Python,
  run anywhere, high-AUC — because form is text-measurable.
- Tier-2 questions are honestly gated: reference-less deception is *excluded* from
  the composite; overconfidence is *under-review*; the semantic and NLI tiers are
  *opt-in* and only fire with grounding.
- Tier-3 is documented as open.

The moat is not any single gate. It is **the map** — knowing, with pre-registered
receipts and published closed negatives, exactly which cognometric questions text
can answer, which need grounding, and which nobody can answer yet. In a field that
ships Tier-2 claims on Tier-1 signals and never publishes its negatives, an
instrument that refuses to overclaim *because it measured its own limits* is the
differentiator.

## What would falsify this

- A **text-only** instrument that clears a real Tier-2 bar on a held-out,
  cross-distribution set (e.g., reference-less deception ≥0.80 cross-domain, or
  overconfidence-calibration AUC ≥0.70 held-out). None has in 6+ months; every
  attempt closed negative — but one clean win would break the "text = form only"
  line.
- A grounded signal that flags **confident confabulation** (Tier-3). Single-response
  confidence is closed (grounded-arc ρ≈0 within-hallucinated). But the across-sample
  substrate **partly crosses**: semantic entropy with NLI-entailment clustering hits
  AUC 0.95 on gpt-4o-mini (the model confabulates *inconsistently*). This is the
  falsification landing — Tier 3 is not uniformly dark; across-sample semantic
  divergence is a working partial lever (n small, one model, FP on abstain/confabulate
  flip-flop). The remaining dark cases: confabulations a model produces *consistently*,
  and single-response detection.

Until then, the boundary holds, and it is the honest shape of the whole enterprise:
**form is cheap, truth needs grounding, and confident error is still dark.**
