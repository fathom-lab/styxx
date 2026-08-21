# recon — who is working near us, and where we actually stand

*2026-08-21. Written to be useful, which means saying where we lose.*

---

## the finding that matters most

An arXiv study this year — *Towards Evaluation Engineering: An Empirical Study
of ML Evaluation Harnesses in the Wild* (2605.24213) — reports, independently of
us:

> "Algorithmic errors in evaluation harnesses often **fail silently**, with the
> harness producing plausible output without throwing exceptions, allowing
> defects to escape normal testing."

Their worked example: **LM Eval reported ROUGE-L near 1.0 for LLaMA-3.1-8B
across every LongBench summarization task** because of a metric computation bug.
Near 1.0 is the *maximum of the range* — the flattering end — and it was caught
only when a user cross-referenced published paper results.

That is SILENT-PASS, in someone else's harness, found the hard way. Our class is
not a styxx idiosyncrasy; it is an observed property of measurement software,
now documented by people who have never heard of us.

The same paper names the second-order version precisely:

> "This manifests the **test oracle problem**: without an independent reference,
> the harness's own output becomes the implicit ground truth."

That is our SP-7 self-confirming loop, arrived at from a different direction.
Two independent observations of one phenomenon is the beginning of a real
finding.

---

## where we lose, plainly

**The guardrails / hallucination-detection product category is saturated and
better funded.** NVIDIA NeMo Guardrails, OpenAI's own Guardrails Python,
Patronus AI (Lynx, open-weights), Giskard, Arize Phoenix, Braintrust, Maxim's
Bifrost, promptfoo. Distribution alone decides that fight: OpenAI ships a
guardrails library with the API.

**We should not fight there.** styxx's `@trust` / guardrail surface is a fine
feature and a hopeless product wedge. Anyone comparing "hallucination detectors"
in a table will pick a funded vendor, and they will usually be right to.

The common methods in that category — LLM-as-judge, semantic entailment,
embedding similarity — are also methods we deliberately refuse inside the
detectors, because a judge is itself a measurement that can fail silently.

---

## where nobody is standing

Searching the space for our actual subject returns the gap described, not filled:

- On static detection of this class: *"static analysis tools often struggle to
  detect the subtle logic errors where fallback mechanisms mask underlying
  failures by returning healthy/default values."* Described as an open
  difficulty. No tool named.
- On who checks the checker: the honest answer found in the literature is
  *"largely a shared responsibility between teams themselves, researchers who
  discover bugs post-deployment, and the open-source community."* Which is to
  say: nobody, systematically.

So the unoccupied position is not "a better hallucination detector". It is:

> **measurement integrity as a software-engineering discipline** — does the
> measurement layer of an AI system actually measure, and can a consumer tell when
> it didn't.

We hold four things there that no one else appears to:

1. **A named, taxonomized class** — SILENT-PASS, 8 subtypes, with the structural
   explanation (the inert default and the flattering default are the same value).
2. **Detectors** — `styxx.absence` (static, per-instance) and `styxx.loops`
   (static, self-confirmation), both characterized, both with published recall
   and published blind spots.
3. **A benchmark** — 20 real defects, commit-cited so it cannot drift, scoring
   recall only and *saying so*, with a localization sweep that separates
   detection from proximity.
4. **74 confirmed instances** in a real codebase, fixed, with receipts.

---

## the adjacent work, and why it is allied rather than rival

- **Data-level contamination** — *Training Data Self-Poisoning* (2026) reports
  medical models retrained on AI-generated notes scoring *better* on
  contaminated evals while degrading 45-fold on authentic notes. ACL 2026
  publishes FDR-controlled detection of contaminated evaluation data.
  **They work on the corpus; we work on the code that writes it.** Our
  `loops` finds the pipe before any data flows through it. Those are two halves
  of one problem and the code half is empty.
- **Defect-dataset mining** — a mature field (D2A, ReDef, RegMiner, BugsPHP,
  Defectors; survey at arXiv 2504.17977). **Our mining method is standard, and
  that is good**: it means SILENT-PASS is built the accepted way. What is new is
  the *class*, not the technique. Existing datasets target generic defects via
  revert commits or analyzer diffs; none targets measurement integrity.

---

## the strategic read

Every vendor in the crowded category ships an evaluation harness, a scoring
pipeline, and a trace store. The arXiv finding says those harnesses fail
silently. **Our tools screen exactly that layer.**

That reframes the entire competitive set: they are not who we beat, they are
who has this problem. The question the literature leaves open — *who audits the
evaluator?* — is a category with one credible answer available and nobody
sitting in it.

The credibility asset is the uncomfortable one. Patronus publishes *"alarming
safety gaps in leading AI systems"* — pointed outward, at others. We published a
census in which **we rank last on our own scale**, a detector scoring **45% on
our own benchmark**, and six occasions on which our own tools committed the
defect they hunt. Nobody in this space is doing that, and it is the only thing
here that cannot be replicated by spending money.

## what to do about it

1. **Stop positioning as a guardrail.** Reposition as measurement integrity.
   Same code, different claim, uncontested ground.
2. **Ship bet 1** (runtime contract, `DESIGN_next_level_2026_08_21.md`). It
   closes SP-6, which is the class's biggest hole and the part no static tool
   reaches.
3. **Write the finding up properly**, citing 2605.24213 as independent
   observation. Two independent sightings plus a detector plus a benchmark is a
   paper; one codebase's bug list is not.
4. **Offer, do not accuse.** Run `styxx-absence` over the open-source harnesses
   in the category and take anything confirmed to the maintainers privately
   first. The census methodology already binds us to hand-verification before
   any named claim, and being the lab that reports quietly is worth more than
   being the lab that posts a table.

---

*Sources: arXiv 2605.24213 (evaluation harnesses fail silently; LM Eval ROUGE-L
bug; test oracle problem), arXiv 2504.17977 (defect-dataset survey), ACL 2026
(contamination detection with statistical guarantees), TianPan 2026-05-06
(training data self-poisoning), vendor documentation for NeMo Guardrails,
OpenAI Guardrails Python, Patronus, Giskard, Arize Phoenix, Braintrust.*
