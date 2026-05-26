# Black-Box Cognometrics: Measuring AI Cognition Across Vendors, and the Honest Map of Its Limits

**Fathom Lab · styxx · 2026-05-25 · technical report (feasibility-grade; all claims tagged)**

## Abstract

White-box interpretability (reading a model's own weights/features) belongs to the labs
that own the weights. **This report is about the other 99%: the closed models running the
world's agents, which no one can open.** We ask what an AI's cognitive state —
*confabulation, the boundary of its knowledge, fabrication, diversity, calibration* — can
be measured **from behavior alone, across vendors, with no weight access and no reference
answer**, and — equally — *where that measurement provably fails*. The contribution is
two things a single lab is structurally unlikely to build: (1) a **cross-vendor,
black-box** instrument layer, and (2) the **published map of its limits**, including the
negatives. Every result here is feasibility-grade (small n, single pre-registered runs);
the discipline — pre-register, run once, publish the floor — is the method, not a
footnote.

## The thesis (one line)

> A model's self-report is dark exactly when it's wrong — but its **divergence** is bright:
> across its own samples, and across independent vendors. Fact is a shared attractor
> (convergent); fabrication has none (divergent). What divergence cannot see — error that
> is *consistent across vendors* — is the floor.

## Results (each tagged: PROVEN-feasibility / BENCHMARK / NEGATIVE / FLOOR / PENDING)

| # | finding | evidence | tag |
|---|---|---|---|
| 1 | Confident confabulation is detectable from across-sample divergence | TriviaQA n=150 AUC **0.785** (judge clustering); cross-model 0.88–0.95 | BENCHMARK |
| 1a | …but it does **not** beat single-response logprob where logprobs exist | TriviaQA logprob 0.817 > 0.785 | NEGATIVE (corrected a shipped overclaim → 7.7.1) |
| 2 | The clustering step is the crux: cosine@0.70 manufactures a null; ≥0.9/entailment recovers it; cosine has documented failure modes | threshold sweep + self-audit | PROVEN + NEGATIVE |
| 3 | Confabulation-inconsistency generalizes across models | gpt-4o-mini / gpt-4o / gpt-3.5 | PROVEN-feasibility |
| 4 | Reference-free fabrication detection via inter-model agreement | AUC 1.0 real-vs-fake; truth-tracking (fame rejected) | PROVEN-feasibility |
| 5 | **Cross-vendor** agreement tracks truth, not OpenAI-consensus; **beats** same-vendor by breaking correlated confabulation | OpenAI+Alibaba+Google, AUC **0.917**, 0/8 shared fabrications | PROVEN-feasibility |
| 6 | Epistemic humility is **prompt-elastic** (a clause flips abstention 0%→97%) | KBC | NEGATIVE/caution |
| 7 | The knowledge boundary as a psychometric curve (admits vs betrays) | per-model curves | PROVEN-feasibility |
| 8 | Diversity index (detector-in-reverse); alignment-tax **null** | open 14× closed, coherence-gated | PROVEN + NEGATIVE |
| 9 | **Security model:** detectors are injection-blind (robust to instruction, not to context-injection) | red-team | FLOOR (deployment boundary) |
| 10 | Consensus answer layer: calibrated **abstention** works; accuracy-boost does **not** (weak members drag) | TriviaQA, Δ 0.253 abstention / −0.051 accuracy | PROVEN (abstention) + NEGATIVE (accuracy) |
| 11 | **Consensus hallucination** via perturbation-fragility — catches *fragile* shared misconceptions (0 false-positives on truth), misses *stubborn* ones | n=10/14, AUC **0.70** (at bar), recall 40%, truth-flip 0% | **FIRST + FLOOR (marginal, tiny-n)** |
| 12 | The *sharper* swing (CVPD): continuous **cross-vendor agreement-fracture** under challenge is a **worse** detector than the binary flip — because misconception correction is **coordinated** (vendors move to truth together, preserving agreement), so Δagreement is blind to it | n=25/25, AUC **0.52**, fracture-recall 0.12, **lift −0.32**; flip replicates at **0.44** | **NEGATIVE + FLOOR (mechanism falsified, n=25)** |

## The negatives are the moat (the asymmetric method)

A neutral indie cannot out-compute the giants; it can only build where their incentives
can't follow — **between** vendors, and at the **floor**. This report leads with its
wrong turns: a public claim retracted, an overclaim caught *inside a shipped PyPI wheel*
and corrected within the hour (7.7.0→7.7.1), two VOIDs and several PASS=FALSEs kept as
written. In a field where the most capable labs say in public that their systems are
"mysterious," a measurement layer whose claims survive scrutiny — because it publishes the
floor — is the rare, load-bearing thing.

## Shipped

`styxx` **7.7.2** (PyPI): `semantic_entropy(samples)` (confabulation signal; logprob-less
niche) and `council_agreement(answers)` (reference-free, cross-vendor, calibrated
abstention). Pure functions; cosine default documented-flawed, judge backend recommended;
**injection-blind by design** — do not run on poisoned context.

## The frontier

The collective blind spots of the *entire* model generation — facts every vendor
confidently agrees on but gets wrong — are visible only cross-vendor (which only a neutral
party builds) and are systemically important (a shared misconception propagates to every
agent on any model). Detecting them reference-free is the dark-matter swing (#11). A
working detector is a first; a proven floor is a boundary the field can cite. **The gate
has resolved — a first *and* a floor.** Perturbation-fragility (a real fact survives a
neutral "reconsider" challenge; a misconception may not) **does** flag *fragile* shared
misconceptions with **zero false-positives on truth** (0/14 truths flipped; flips
self-correct 3/4) — a reference-free signal where divergence, confidence, and agreement
all go blind. **But recall is 40%**: the 60% it misses are the *stubborn* misconceptions
that resist reflection — the most dangerous ones. At **n=10**, AUC exactly on the 0.70
bar: a *suggestive first signal*, not a validated result. The honest landing: **the dark
matter is *partly* visible, and its dangerous core is proven (by this method) to stay
dark.** A candle in the cave; most of the cave is still black — and both are the
contribution.

We then took a **sharper swing** at that dark core (#12, CVPD), and it is the cleanest
negative in the report. Instead of asking whether the *majority answer* flips, CVPD
measured the continuous **drop in cross-vendor agreement** under the same challenge,
betting that a shared misconception would *fracture* — one vendor defecting, re-exposing
divergence. Pre-registered, run once at **n=25/25**, it **fails decisively** (AUC 0.52 ≈
chance; fracture-recall 0.12) and, more tellingly, is **worse than the binary flip it tried
to beat (lift −0.32)**. The data falsifies the mechanism: **9 of 11 misconception
corrections preserved agreement** — vendors move toward truth *together*, in lockstep, so
agreement-fracture is structurally blind to exactly the corrections the simpler signal
catches. Correction of a shared misconception is **coordinated, not fractured.** The same
run **replicated** the first swing's signal (flip-recall 0.44 at n=25 vs 0.40 at n=10), so
the candle is robust — but the lesson for the field is now mapped, not guessed: *don't chase
cross-vendor agreement-fracture for consensus hallucination; the dangerous myth-class core
stays dark, and the cleanest reference-free signal for the fragile shell is the cruder one.*

## Reproducibility

Every probe in `papers/` is pre-registered before data, run once, with the script and
results JSON committed. Bars were fixed in commits that precede the data commits. The
wrong turns are in the git history on purpose.
