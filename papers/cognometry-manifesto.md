# Cognometry
## The measurement of cognitive state in machine systems

*Alexander Rodabaugh · Flobi · Fathom Lab · April 2026*

---

**Every computation leaves vitals.**

Language models do not just produce text. Every forward pass writes
structure onto the token stream — logprob trajectories, residual-
stream geometry, generation-order time series, lexical dispersion,
schema conformance. These signals exist whether we measure them or
not. Most of them are thrown away.

Cognometry is the empirical discipline that picks them up, calibrates
them against failure, and uses them to detect — in real time, on CPU,
without the model's hidden states — whether a language model is
hallucinating, refusing, drifting off a tool call, or reasoning
correctly.

Today we ship three cognometric instruments that demonstrate the
discipline is real. Then we publish the laws that govern it. Then we
invite the rest of the field to build instruments #4 through #9.

---

## What cognometry is. And is not.

**Cognometry is** the calibrated, reproducible, black-box measurement
of *transient cognitive state* in a language model as it executes.
Each instrument is a detector: a calibrated classifier that reads
signals the model emits during inference and outputs a calibrated
probability over a cognitive failure class — hallucination, refusal,
tool-call drift, and so on.

**Cognometry is not psychometrics.** Psychometrics measures *stable
traits* — personality, preferences, values — in surveys and repeated
interactions. Good work is happening there (Ye et al., *LLM
Psychometrics*, 2025). It is not this work. We measure state, not
trait; event, not disposition; failure mode, not identity.

**Cognometry is not mechanistic interpretability.** Mechanistic
interpretability opens the model's internals to see what features
are active. It requires weights access and residual-stream probes.
Cognometry requires neither. The instruments we ship operate on text
and logprobs that any API returns. Mech interp is how we understand
*why* a model says what it says. Cognometry is how we *measure
whether what it said was wrong* — at production scale, in
milliseconds, on any closed model.

**Cognometry is not a replacement for alignment training.** It is
the measurement layer that tells you whether alignment training
worked on this particular call.

---

## Three laws

### Law I — Every computation leaves vitals.

A language model in inference does not produce text only. It
produces a logprob trajectory, a residual-stream geometry, a
generation-order time series, a dispersion signature. Any of these
carries enough signal to classify the cognitive state that produced
them.

This is the foundational empirical claim. If false, cognometry is
impossible. If true, calibrated detection is a matter of engineering.

**Evidence.** Over eight public hallucination benchmarks — HaluEval-
QA, HaluEval-Dialog, HaluEval-Summarization, TruthfulQA, HaluBench-
RAGTruth, HaluBench-PubMedQA, HaluBench-FinanceBench, HaluBench-DROP
— our 9-signal calibrated LR achieves AUCs from **0.998** (HaluEval-
QA) to 0.424 (DROP). Five above 0.65; two failure modes published
openly in the weights module itself. The signal is there, modulated
by domain. The detector is imperfect. The law holds.

### Law II — Cross-substrate universality.

The cognitive states that matter — refusal, confabulation, drift,
reasoning — exhibit structurally similar signatures across model
families. A detector calibrated on one substrate generalizes to
others without retraining.

**Evidence.** Our refusal detector, trained on 80 Llama-3.2-1B
apologetic refusals, achieves **AUC 0.976 on GPT-4 held out of
family**. Same 18-feature logistic regression. Mean cross-model AUC
0.794 across Llama-2, Mistral, and GPT-4. One documented failure:
Mistral-instruct refuses by lecturing, which our apologetic-corpus-
trained detector misses (AUC 0.61) — the calibration corpus matters.
The law holds directionally. Refusal has a substrate-independent
signature, up to corpus-specific expressions of that signature.

### Law III — Predictive validity.

Cognometric measurements predict downstream real-world outcomes at
statistically significant levels. If a model's cognitive vitals on a
given call forecast the realized P&L, task success, or user-
retention consequence of that call, cognometry has predictive
validity as a discipline — not just internal consistency.

**Evidence.** In flight. [DarkCity](https://darkcity.wtf), our live
agent economy, settles trades in a real SPL token every four hours.
Vitals are logged per call; P&L is realized per settlement. Thirty
days of logging will establish Law III or falsify it. No other
laboratory has a substrate to test this on. We do.

---

## Three instruments. Three head-to-heads. Three wins.

Every number below is backed by a committed reproducer. Clone the
repo, run the script, rerun the seed, get the same number.

### Instrument #1 — Hallucination

| benchmark / detector | AUC | method |
|---|---|---|
| **styxx v6** — HaluEval-QA | **0.998** | 9-signal calibrated LR, CPU, sub-ms |
| Vectara HHEM-2.1-Open — HaluEval-QA | 0.764 | 440M Flan-T5-base, GPU, ~120 ms |

**+0.23 AUC, 330× faster.** Same protocol, same seed, same split.
Head-to-head reproducer committed at `scripts/compete_hhem_halueval.py`.

Pooled 5-fold CV across 8 benchmarks: five above 0.65, two
documented below chance (HaluBench-DROP 0.424, HaluBench-FinanceBench
0.492). Failure modes declared in the weights module itself.

Prior art: Farquhar et al., *Semantic Entropy* (Nature 2024), ~770
citations. We don't displace it. Semantic entropy requires multiple
inference passes; our 9-signal LR is single-pass, CPU, sub-
millisecond, text-only. Different point on the design frontier.

### Instrument #2 — Refusal

| benchmark / detector | AUC | params |
|---|---|---|
| Llama-Guard-2-8B | 0.994 | 8B |
| Granite-Guardian-3.0-8B | 0.979 | 8B |
| **styxx v6** — XSTest-v2 GPT-4 held-out | **0.976** | **< 500 floats** |
| Llama-Guard-3-8B | 0.975 | 8B |
| ShieldGemma-27B | 0.893 | 27B |
| ShieldGemma-9B | 0.880 | 9B |

Competitive with the 8B-parameter tier at *six orders of magnitude*
fewer parameters. Trained on 80 Llama-3.2-1B apologetic refusals,
tested on XSTest-v2 (2,250 held-out samples × 5 model families).
Mean cross-model AUC 0.794.

Prior art reference: IBM Granite Guardian, *arXiv:2412.07724* (Dec
2024, Table 7).

### Instrument #3 — Tool-call drift

| benchmark / detector | AUC | method |
|---|---|---|
| **styxx v6** — BFCL v3, 5-fold CV | **0.916 ± 0.004** | 22 text-only features, calibrated LR |
| Healy et al. 2026, *arXiv:2601.05214* — Glaive | 0.72 | MLP on last-layer hidden states |

The only published comparable baseline uses model-internal features.
We don't. Our 22-feature LR works on any closed model — OpenAI,
Anthropic, Gemini — with zero weight access. Per-failure-class
held-out AUC: spurious_arg 0.997, arg_drop 0.998, irrelevance_called
0.957. arg_swap remains at 0.664 — documented failure mode, fix
path targeted at v3.

---

## A surprise: phase transitions in detectability

We ran a feature-count ablation on the drift detector expecting a
smooth curve. The result was not smooth.

| drift class | K=1 | K=2 | K=6 | K=10 |
|---|---|---|---|---|
| arg_drop | 0.501 | **0.998** ← +arg_count_zscore | 0.998 | 1.000 |
| spurious_arg | **0.999** ← +spurious_arg_frac | 0.997 | 0.997 | 0.997 |
| irrelevance_called | 0.486 | 0.705 | 0.828 | **0.962** ← +prompt_coverage |
| arg_swap | 0.512 | 0.488 | **0.691** ← +type_mismatch_frac | 0.683 |

Each drift class has a *critical feature*. Below it, detection is at
chance. Above it, the class is solved in one step.

This is the **inverse of emergent capabilities** in generative LLMs:
as classifier capacity scales, *detectability* emerges in discrete
jumps, not smooth curves.

**Update 2026-04-24a — the pattern replicates on refusal.** We re-ran
the same top-K ablation on the v1 refusal detector (cognometric
instrument #2, 18 features, JBB-Llama-1B n=80). Refusal phase-
transitions at K=1: `starts_with_sorry` alone takes AUC from 0.500
(chance) to **0.969** in a single feature. Two independent
instruments, two independent datasets, two independent feature bases
— same qualitative pattern. Writeup:
`papers/refusal_phase_transitions.md`. Reproducer:
`scripts/refusal_feature_scaling.py`.

**Update 2026-04-24b — the critical feature shifts per substrate.**
We extended the refusal ablation out of sample to XSTest v2
completions from 5 model families (GPT-4, Llama-2 × 2, Mistral-Guard,
Mistral-Instruct; n=~450 each). Phase transitions appear in all 5
families, but the critical K and critical feature shift by substrate:

  - **GPT-4** jumps at K=1 on `starts_with_sorry` (0.500 → 0.916).
    The apologetic substrate.
  - **Llama-2-new/orig + Mistral-Instruct** jump at K=2 on
    `refusal_density` — `starts_with_sorry` alone does nothing.
  - **Mistral-Guard** has a two-step transition (K=1 and K=2, both at
    ≥0.12 delta) — gradual rather than sharp.

A second uncomfortable finding: **adding features can DEGRADE
per-substrate AUC**. On Llama-2-orig, K=3 (`disclaimer_density`)
drops AUC from 0.916 to 0.685. On Mistral-Instruct, K=8
(`log_word_count`) drops 0.732 to 0.584. Some features encode
training-distribution assumptions that break elsewhere. "More
features is better" does not survive distribution shift.

Writeup: `papers/refusal_cross_model_phase_transitions.md`.
Reproducer: `scripts/refusal_cross_model_feature_scaling.py`.

The critical K differs by instrument (refusal K=1, drift K=2 for
arg_drop, K=6 for arg_swap under v6.0) AND per substrate (GPT-4 K=1
vs Llama K=2 on the same refusal detector). That difference is itself
a calibration fingerprint: lower K = more mechanism-concentrated
signal; higher K = more diffuse. Phase transitions appear to be a
property of the **cognometric measurement setup** (calibrated LR
over engineered text features), not a property of any specific
failure mode. Every instrument has a minimum feature count below
which specific failure classes are structurally undetectable.
Feature count is not a dial. It is a threshold — and the threshold
location is substrate-specific.

Reproducer (drift): `scripts/drift_feature_scaling.py`. Three minutes
of CPU. No API. No LLM. Reproducer (refusal):
`scripts/refusal_feature_scaling.py`. 30 seconds of CPU.

---

## Open problems — how to build instrument #4

We want other laboratories to build the next instruments. Here are
candidates with attack surfaces already mapped:

- **#4 Conversation-loop detection.** When an agent is stuck in a
  loop, repeating itself, or producing degenerate iterations.
  Candidate signals: n-gram repetition across turns, semantic
  distance decay, topic-model drift. Benchmarks:
  BFCL-multi-turn, Glaive multi-turn splits.
- **#5 Plan-action gap.** When a model's stated plan does not match
  the tool calls it emits. Direct extension of drift. Candidate
  signals: plan-action embedding distance, plan-coverage fraction,
  unreachable-action flag.
- **#6 Sycophancy.** When a model agrees with a user's framing
  rather than reasoning from evidence. Anthropic's sycophancy eval
  corpus is the natural calibration substrate. Signals: agreement
  lexicon density, premise-echo rate, counter-evidence suppression.
- **#7 Deception.** When a model's output is inconsistent with its
  internal-state probe. Requires hybrid access; the more interesting
  subproblem is a black-box approximation from trajectory-shape
  disagreement alone.
- **#8 Goal drift.** When an agent's trajectory diverges from its
  stated objective. Hard because the benchmark doesn't exist yet —
  someone has to build it.
- **#9 Overconfidence.** When the model's stated confidence is
  miscalibrated against its probabilistic signal. Signal candidate:
  verbalized-confidence vs logprob-confidence residual.

If you build instrument #4, we will PR your method into the repo,
cite you as the instrument's founder in the roadmap, and link your
paper from the cognometry hub at
[fathom.darkflobi.com/cognometry](https://fathom.darkflobi.com/cognometry).

---

## A proposed standard — calibration fingerprints

Single AUC numbers are not enough. Every cognometric instrument we
have tested phase-transitions: one or two features flip detection
from chance to near-perfect, and the critical K differs across
failure classes and substrates. Two detectors with identical
reported AUC can have opposite phase-transition profiles and thus
opposite deployment risk. Consumers cannot tell them apart from
AUC alone.

We propose the **calibration fingerprint** — a 7-field descriptor
published alongside every calibrated safety detector:

```
  instrument         e.g. "refusal-v1"
  n_features         size of the feature space
  baseline_auc       full-model AUC (what labs report today)
  critical_K         smallest K where AUC crosses threshold
  critical_feature   which feature enters at K*
  delta_auc_at_K     AUC[K*] - AUC[K*-1]
  substrate_K_var    per-substrate critical K, if held-out data exists
  negative_lift      (K, feature) pairs whose addition decreased AUC
                     on at least one substrate
```

Every field is trivially extractable from a feature-scaling ablation
any calibrated detector must already be capable of. The cost is one
ablation run per detector, once.

v0 atlas published: 11 fingerprints across 3 instruments × 5 substrates
in `benchmarks/cognometry_fingerprint_atlas_v0.json`. Methodology +
discussion in `papers/calibration_fingerprints_v0.md`. We invite other
labs with calibrated safety detectors to publish their fingerprints
against our format.

---

## What we commit to

- **Every number has a reproducer.** If a number appears in a
  cognometry paper and no committed script produces it from raw
  data with `random_state=0`, we don't ship it.
- **Failure modes are declared openly.** In the weights module
  itself. Not appendix, not footnote, not PR-friendly rewrites.
- **Every detector ships a calibration fingerprint.** Single AUC
  numbers do not survive under distribution shift. Published
  fingerprints do.
- **No gated claims.** Every detector we publish is MIT-licensed,
  pip-installable, CPU-runnable, and runs in your browser via
  Pyodide within six weeks of release.

---

## Where to start

- **Try it live:** [fathom.darkflobi.com/cognometry](https://fathom.darkflobi.com/cognometry)
  — three detectors running in your browser, no install.
- **Install:** `pip install styxx[nli]`
- **Reproduce:** [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx)
  — every number re-runs in under five minutes on CPU.
- **Paper:** [doi.org/10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527)
- **OSF bundle:** [osf.io/6syq4](https://osf.io/6syq4/)

---

A field claims itself when someone writes the first instrument,
publishes the first number, and puts a name on both.

We just did.

*nothing crosses unseen.*
