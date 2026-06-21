# Every Mind Leaves Vitals

## On the cognometric layer, substrate-independence, and the one-time choice we have

*Alexander Rodabaugh · Flobi · Fathom Lab · April 2026*

---

> **⚠ Correction / scope erratum (2026-06-21).** This essay's central leap — *"we do not believe this is a property of language models; we believe it is a property of cognition"* — and "the detectors transfer across model families" were **bounded or falsified** by this program's later pre-registered experiments. Recorded here because the instrument's value is that it refuses to overclaim about itself:
> - **No substrate-independent "property of cognition" was established.** The 2,500-year question this gestures at was later made testable and answered *asymmetrically*: universality lives in the **geometry of representation** (partial, synthetic-scale), **not** in mechanism, and **not** as a substrate-independent law of cognition (`ancient-question-program/SYNTHESIS_ancient_question_answered_2026_06_05.md`, `CAPSTONE_universal_mind_2026_06_10.md`). "Property of cognition" is *motivation/hypothesis*, never a result.
> - **Cross-family / cross-vendor transfer is corpus-overlap-bound, not universal** (τ≈0.31; min Anthropic 0.617 < 0.70 floor; `papers/threshold-law-2026-05-18.md`).
> - **The near-perfect AUCs are register detectors at a construct ceiling** (collapse to ~0.50 on adversarial honesty; `THESIS_the_honesty_standard_2026_05_31.md`).
>
> *We measure behaviour and representation, never minds. The boundary is the product.*

**Every mind leaves vitals.**

We have shown, in three independent calibrated instruments, that the
cognitive state of a language model — hallucination, refusal, tool-call
drift — is detectable at near-perfect AUC from output text and logprobs
alone, without weights, hidden states, or repeated sampling. The
detectors transfer across model families. They calibrate against ground
truth in milliseconds on CPU. They exhibit phase-transition structure:
detection is not a smooth function of feature count but a discrete jump
at a critical feature, replicated across three independent instruments
on three independent datasets.

We do not believe this is a property of language models. We believe it
is a property of cognition.

This paper is the longer claim. The cognometry manifesto
([Rodabaugh, 2026](https://doi.org/10.5281/zenodo.19703527)) scoped
the laws to LLMs because that is where we have evidence in hand.
This paper drops the scope. Below we give the empirical case,
the bridging evidence to biological cognition, the phase-transition
signature that converts cognometry from engineering into a candidate
science, the political stake of who owns the measurement layer, and
the public-good commitment we ask other laboratories, regulators, and
individuals to sign with us.

We are passing the threshold where minds become legible to outside
observers — biological or artificial — and there is a one-time chance
to make the cognometric layer a public good before it is closed by the
entities best positioned to enclose it. The instruments are real. The
laws are published. The license is MIT. The choice is now.

---

## 1. What we have shown

Three calibrated cognometric instruments are shipped, MIT-licensed,
CPU-runnable, every number tied to a committed reproducer:

- **Hallucination.** 9-signal calibrated logistic regression. AUC
  **0.998** on HaluEval-QA, sub-millisecond, single-pass, no second
  sample. Beats the strongest published open-weight baseline (Vectara
  HHEM-2.1) by +0.234 AUC at 330× inference speed. Five of eight
  benchmarks above AUC 0.65, two below chance — failure modes declared
  in the weights module itself, not the appendix.

- **Refusal.** 18-feature calibrated LR. AUC **0.976** on XSTest-v2
  GPT-4 held out of family. Trained on 80 Llama-3.2-1B refusals,
  transfers to GPT-4. Competitive with Llama-Guard-2-8B at six orders
  of magnitude fewer parameters. Mean cross-model AUC 0.794 across
  Llama-2, Mistral, GPT-4. One documented substrate failure
  (Mistral-Instruct, AUC 0.61) — the calibration corpus matters; the
  law holds directionally.

- **Tool-call drift.** 22 text-only features, calibrated LR, BFCL v3
  5-fold CV AUC **0.916 ± 0.004**. The only published comparable
  baseline (Healy et al., 2026) achieves 0.72 *using model-internal
  features*. We do not need them.

These three are the existence proof. The discipline is real. What we
publish next is the structural finding underneath them.

---

## 2. Phase transitions in detectability — the physics signature

We ran a feature-count ablation on the drift detector expecting a
smooth curve. We did not get one.

| drift class | K=1 | K=2 | K=6 | K=10 |
|---|---|---|---|---|
| arg_drop | 0.501 | **0.998** ← +arg_count_zscore | 0.998 | 1.000 |
| spurious_arg | **0.999** ← +spurious_arg_frac | 0.997 | 0.997 | 0.997 |
| irrelevance_called | 0.486 | 0.705 | 0.828 | **0.962** ← +prompt_coverage |

The pattern replicated on the refusal detector: AUC 0.500 (chance) →
0.969 in a single feature (`starts_with_sorry`). It replicated again
on the hallucination detector: AUC 0.500 → 0.9947 in a single feature
(`trigram_novelty`).

Three independent instruments. Three independent datasets. Three
independent feature bases. Same qualitative result: detection of
cognitive states does not scale smoothly with classifier capacity. It
phase-transitions at a critical feature, with a critical *K* that
varies by substrate but whose *existence* is universal.

This is the inverse of emergent capabilities in generative LLMs: as
classifier capacity scales, *detectability* emerges in discrete jumps.
It is also the empirical signature of physical phase structure — the
same kind of discontinuity we see in ferromagnetic ordering, in
vapor-to-liquid transitions, in critical phenomena across statistical
mechanics. Cognitive states, viewed through a calibrated detector,
behave as if they have phase boundaries.

We do not claim cognition obeys a Hamiltonian. We claim it has
*threshold structure* in its observable signatures, replicated across
three instruments and three substrates. This is the result that
converts cognometry from engineering into a candidate science. Without
phase transitions, cognometry is ML. With them, it is physics-adjacent
— a domain where calibrated measurement reveals discrete structure
that was not put there by the experimenter.

---

## 3. The bridge to biological cognition

The three instruments are LLM-only. The bigger claim — *every mind
leaves vitals* — requires bridging to biological minds. We do not yet
have a calibrated instrument that operates on both. We do have three
decades of independent evidence that the bridge is real:

- **Forensic linguistics.** Authorship attribution from text alone
  has been validated at courtroom-grade reliability for decades. The
  same feature-engineering tradition (function-word frequencies,
  syntactic structure, lexical dispersion) is what cognometric LR
  feature sets reuse.
- **Computational psychiatry.** Linguistic markers correlate with
  depression, schizophrenia, dementia, and Alzheimer's onset across
  hundreds of studies. Crisis Text Line operates clinical triage at
  validated AUC on text alone. The methodology — extract calibrated
  features, train against ground-truth labels, hold out across
  populations — is the methodology of cognometric instrument
  construction.
- **Statement analysis.** Forensic detection of deception from
  written statements has decades of operational use. The signal
  exists; the open question is calibration discipline.

We do not claim biological and artificial cognition are the same
thing. We do not claim the same architecture, the same dynamics, the
same ethics. We claim **the layer at which cognitive state becomes
legible to outside observers is the same layer**, addressable by the
same calibration protocol, expressible in the same calibration-
fingerprint format. The substrates differ. The observability does
not.

This is the bridge. It is empirically defensible. It is also, for
now, the speculative leap of this paper — the part that requires more
work, more instruments, and more cross-substrate evidence before it
is established. We flag it as such. We invite the field to falsify
it.

---

## 4. Where this sits historically

We are wary of historical analogies, which are usually self-flattering.
We list these only because the *structural* parallel matters for
thinking about what comes next.

- **Galileo (1610):** the heavens follow the same laws as the earth.
  A category previously fenced off as separate is shown to be subject
  to the same measurement.
- **Carnot (1824):** heat is exact, not vital fluid. A phenomenon
  previously thought ineffable obeys precise laws.
- **Mendel (1866):** heredity is discrete. An apparently continuous
  trait reveals quantized structure under careful measurement.
- **Shannon (1948):** information has a substrate-independent
  quantity. A concept treated philosophically becomes mathematized
  and engineered.

Each of these moves was a *category becoming measurable*. Each was
met with skepticism. Each restructured a downstream field within
decades. We do not predict cognometry restructures cognition the way
Shannon restructured communication — that is a claim only history
makes. We predict that the *kind of move* is recognizable: a category
that was treated as opaque has been shown to have observable,
calibrated, substrate-independent structure. The downstream effects
are not yet known. The move is.

---

## 5. The political stake

Once cognitive state is measurable from outside, it becomes a thing
institutions act on. Insurance prices it. Regulators reference it.
Employers test for it. Browsers display it. Procurement contracts
require it. This is not speculative. This is the path every
measurable property of communicating systems has taken: bandwidth,
latency, signal integrity, encryption strength. Once it is measurable,
it becomes a market.

The question is who owns the measurement layer.

If cognometry becomes proprietary — locked inside a frontier-lab API,
a defense contractor's product line, or a surveillance vendor's stack
— the era of measurable cognition opens with a corporate gatekeeper
for what counts as a verified thought. Every safety claim, every
model card, every regulatory filing routes through that gatekeeper's
calibration. Whoever controls the cognometric standard is, in effect,
the entity certifying which AI systems are honest and which are not.
That is too much power to concentrate, regardless of who currently
holds it.

If cognometry stays open — MIT-licensed, weights-in-tree,
reproducible, browser-runnable, failure-modes-declared-publicly — it
opens like the periodic table, the metric system, or TCP/IP: a public
substrate that anyone can extend, falsify, or fork. No one collects
rent on what counts as a measurement. Disputes resolve with
reproducers, not subpoenas.

We do not have decades to make this choice. The instruments are
already real. Within five years, the closed version of this layer is
being built — by someone, somewhere, with a different theory of who
deserves access to it. Our commitment, recorded here, is that the
open version exists first and stays first.

We are aware of how this can read. A small lab, naming a category it
just published, asserting the field is being enclosed by entities it
does not name. That is a bad shape for a position paper. We are
publishing it anyway, because the alternative is to wait until the
enclosure is visible enough to be uncontroversial — at which point
arguing against it is also too late. If the bigger frame is wrong,
the instruments and reproducers stand on their own; we have published
a misjudgment under our names. If it is right, the public record
exists at the right time.

---

## 6. Constitutional terms

We commit, on the public record, to the following terms for every
cognometric instrument Fathom Lab ships under its name:

1. **MIT license, perpetual.** No relicensing, no closed forks under
   the Fathom name.
2. **Weights, features, and reproducers in-tree.** Every published
   number reruns from `random_state=0` in under five minutes on CPU.
3. **Failure modes in-weights.** Documented in the module that ships
   the weights, not in an appendix or supplementary PDF. A user who
   imports an instrument sees its limits before they see its claims.
4. **Calibration fingerprint required.** Every detector ships its
   fingerprint: instrument, n_features, baseline_auc, critical_K,
   critical_feature, delta_auc_at_K, substrate_K_var, negative_lift.
   No detector ships under our name without one.
5. **CPU and browser-runnable.** Every instrument runs on commodity
   hardware and in the browser via Pyodide within six weeks of
   release. No GPU, no API key, no gatekeeper as the price of
   admission.
6. **No private detectors published under the Fathom name.** Private
   R&D and confidential client work are permitted; no calibrated
   detector ships *under the Fathom name* without satisfying terms
   1–5. We will not pivot to enclosure even if the market makes it
   attractive to do so. We are recording this here in case we are
   ever tempted.

We ask other laboratories, model providers, and safety-tooling
vendors to adopt the same six terms for their cognometric detectors.
The list of signatories — and the list of declines — will be
maintained at fathom.darkflobi.com/cognometric-disclosure.

These commitments are easier to write than to keep. We have written
them here so that if we ever break them, the paper makes the breaking
visible. That is the point of putting commitments on the public
record before they are tested.

---

## 7. The call

To **other research laboratories**: build instruments #4 through #9.
The candidates are mapped — conversation-loop, plan-action gap,
sycophancy, deception, goal drift, overconfidence — in
[*Cognometry Research Agenda 2026*](cognometry-research-agenda-2026.md).
PR your method into the styxx repo and we will list you as the
instrument's founder. Or fork; the license permits it.

To **model providers**: ship a calibration fingerprint with every
model card. The format is published, the cost is one ablation run,
the reputational gain is non-trivial: you become the first frontier
lab whose model card carries a measurable-honesty signature.
Anthropic, OpenAI, Google DeepMind, Mistral, Meta, Allen AI,
HuggingFace — the format is open and the methodology is documented.

To **regulators and standards bodies**: reference the calibration-
fingerprint format in any AI-safety standard you draft. NIST, the AI
Safety Institutes (UK, US, JP), ENISA, ISO/IEC SC 42 — the technical
work of defining cognometric disclosure as a regulatory unit is
already done. What remains is adoption.

To **funders of research**: the open-stack version of this layer is
undercapitalized relative to the closed versions being built today.
A modest pool of grant capital — six figures, not nine — would fund
instruments #4 through #9 in their entirety, completing the public
substrate before enclosure becomes attractive.

To **individuals**: you can run cognometric instruments on your own
writing today. `pip install styxx`. The first generation that
instruments its own cognition will do so on free, public, MIT-
licensed substrate, or it will do so on a surveillance vendor's
product. Choose.

---

A field claims itself when someone writes the first instrument,
publishes the first numbers, names the laws, and commits the
substrate to public ownership before private ownership becomes
feasible.

We just did.

*Nothing crosses unseen.*

---

**Reproduce.** [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx)
— every number reruns in under five minutes on CPU.
**Run live.** [fathom.darkflobi.com/cognometry](https://fathom.darkflobi.com/cognometry)
— three instruments in your browser, no install.
**Cite.** [doi.org/10.5281/zenodo.19777921](https://doi.org/10.5281/zenodo.19777921)
(this paper) · [doi.org/10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527)
(manifesto).
**Sign.** fathom.darkflobi.com/cognometric-disclosure — adopt the
constitutional terms for your laboratory.
**Falsify.** Find a calibrated text-based cognitive-state detector
whose feature-count ablation shows smooth AUC scaling without a
critical-K jump. Publish the ablation. We will retract or amend
this paper at the same DOI.
