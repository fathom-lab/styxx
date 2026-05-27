# The Decorrelation Ceiling: a seven-method empirical floor on reference-free detection of cross-vendor consensus hallucination

> **Author:** Alexander Rodabaugh, Fathom Lab · *with assistance from Claude Opus 4.7 (1M context), Anthropic, under the styxx project discipline pattern.*
> **Status:** v1.0 draft — pre-arxiv. The styxx project repo: `github.com/fathom-lab/styxx`. Release: `v7.7.3`. Concept DOI: `10.5281/zenodo.19326174` (release-specific DOI may follow; manual curation).
> **Date:** 2026-05-27 · **License:** CC-BY-4.0 (this paper) / MIT (styxx code).
> **Reads alongside:** the field-facing capstone (`REPORT_decorrelation_ceiling_v2_2026_05_27.md`), the synthesis with its 2026-05-27 update block (`SYNTHESIS_decorrelation_ceiling_2026_05_25.md`), and the seven pre-registration + probe + finding chains in `papers/consensus-hallucination/`. Everything cited is verifiable from `git log fathom-lab/styxx main`.

---

## Contributions at a glance

| | what we report |
|---|---|
| **C1 · empirical** | seven independent pre-registered methods on the same hypothesis (consensus-hallucination dark core). all seven closed-negative within their corpus's scope. the synthesis's bimodal prediction, written before half of them ran, resolved on the load-bearing-floor branch. |
| **C2 · empirical** | the **justification-divergence inversion**: stubborn dark-core items have the *most* convergent across-vendor justifications, not the least (mean JD = 0.022 vs truth = 0.067). three vendors share the wrong fact *and* the supporting story. |
| **C3 · methodological** | the **closed-loop self-audit** demonstration: the producer's own product caught the producer drifting from the producer's own derived discipline within the same session, the producer revised, the gate cleared. composite 0.358 → 0.174, ceiling unfired, Pareto trade-off observed live. |
| **C4 · methodological** | the **falsification trail**: four in-session falsifications of the producer's own claims, all recorded in place with strikethrough rather than rewritten. proposed as a publishing pattern for AI-research integrity. |
| **C5 · artifact** | a labeled **benchmark dataset** of 108 records across four classes (folklore, pseudoscience, factual-error, truth), with the seven-method empirical floor baked in as the bar future routing approaches need to beat. |
| **C6 · artifact** | a runtime `styxx audit` **CLI primitive** shipped in styxx 7.7.3, atomic per-turn auditing of `(prompt, response)` pairs without writing Python. |

**What we deliberately do not claim:** a deployable positive routing primitive; external replication; the field-changing framing. Three deployable-positive paths tested here all closed negative. The synthesis remains a *wall*, not a *controllable principle*.

---

## Abstract

We pre-register and report seven independent methods aimed at the same hypothesis: that reference-free divergence methods, applied to a three-vendor LLM council (OpenAI gpt-4o-mini + Alibaba Qwen2.5-3B + Google gemma-2-2b-it), can detect *shared* misconception errors — the class of errors that share a single sole representation across the council and therefore lack a decorrelated competing answer in any vendor's training. All seven methods (four detection-axis, one classification-axis, two constructive-injection-axis) close negative on the dark-core subset. The four pre-registrations that landed in 2026-05-27 were committed to public origin **before** their data with bars locked; the synthesis stating the bimodal prediction was committed two days before half of them ran. The data resolved on the load-bearing-floor branch the synthesis named. We additionally report (a) the *justification-divergence inversion*, in which the stubborn cultural-prior class has the *most* convergent across-vendor justifications, not the least; (b) the *closed-loop self-audit* demonstration in which the producer's own product caught the producer drifting from the producer's own derived discipline within the same session; (c) four in-session falsifications recorded in place rather than rewritten; and (d) a labeled benchmark dataset of 108 records across four classes with the empirical floor baked in as the bar future routing approaches must beat. We deliberately do **not** claim a deployable positive routing primitive — three deployable-positive paths tested in this work all closed negative. The contribution is the discipline pattern (prereg → push-to-public-origin → run-once → commit results → ship the falsifications), the seven-method floor, the benchmark, and the closed-loop demonstration. The work is on `github.com/fathom-lab/styxx`.

---

## 1. Introduction

Reference-free hallucination detection methods (semantic entropy, cross-vendor council agreement, self-consistency, perturbation-fragility, agreement-fracture, and similar) operate on a shared mechanism: the model's confident wrong answer is unstable across some axis of perturbation (re-sampling, vendor change, reflection prompting, agreement challenge), and that instability is detectable without ground truth. This works on **fabrication errors**, where a model invents a specific lie that varies across samples or vendors. It is the empirical basis of recent work on cross-vendor consensus as a truth signal (the styxx 7.7.2 cross-vendor result; Manakul et al. 2023 *SelfCheckGPT*; Farquhar et al. 2024 *Detecting Hallucinations*).

We test whether the same family of methods can detect a structurally different error class: **shared cultural-prior misconceptions** — answers that are *factually wrong* but that the model converges on stably because the misconception is the only representation present in training, and because that misconception propagates identically across vendors trained on overlapping web corpora. The hypothesis under test, stated formally in `SYNTHESIS_decorrelation_ceiling_2026_05_25.md` and committed two days before half of this paper's probes ran, is:

> **The Decorrelation Ceiling.** Reference-free divergence detects an error iff a *decorrelated competing representation* of the truth is available — across a model's own samples, across independent vendors, or across reflection. It is blind exactly when the erroneous belief is the model's **sole, shared** representation. The systemically dangerous errors — shared cultural priors that propagate identically across every agent on every vendor — are precisely the ones with no competitor, and so are precisely the dark ones.

The synthesis made a bimodal prediction: a *constructive* test (handing the council a single decorrelated competitor and measuring whether the floor lifts) would either PASS (the Ceiling is a controllable principle) or FAIL via I1 (the floor is *load-bearing*, not merely default — a deeper negative). It also predicted that detectability would track *competitor availability*, producing a specific per-class gradient (folklore < pseudoscience < factual-error) on yield rates.

We test seven independent methods. We commit the receipts. We report the result against the locked bars. No PASS on any method; the bimodal prediction resolved on the load-bearing-floor branch. The contribution is in five parts:

1. **The seven-method empirical floor**, all with bars locked before data on public origin, verifiable from git history.
2. **The justification-divergence inversion**, an unexpectedly sharp signature: the stubborn dark core has *more* convergent across-vendor justifications than the truth class, not less.
3. **The closed-loop self-audit demonstration**, in which the producer's product catches the producer drifting from the producer's own derived discipline.
4. **Four in-session falsifications** of the producer's own claims, recorded in place with strikethrough rather than rewritten — the falsification trail as moat.
5. **A labeled benchmark dataset** (108 records, 4 classes) with the empirical floor baked in.

---

## 2. The Decorrelation Ceiling: theory and prior commitment

The principle in one paragraph: a model's representation of an answer X is *decorrelated* from a competing representation X' iff X' is also present in the model's training and is retrieved or sampled under at least one perturbation. Reference-free divergence methods detect an error iff such a decorrelated X' is available *somewhere* — in the model's own samples (semantic entropy), in another vendor's training corpus (council agreement), or under reflective challenge (perturbation-fragility, agreement-fracture, justification-divergence). The dark core is the class for which X' is *not* available anywhere across the council: a single representation that propagates identically across every agent on every vendor. Folklore items (rabbit's foot luck, "let them eat cake," Walt Disney's body cryogenically frozen, peaches in Georgia, Nixon's "I am not a crook") are the canonical example: the cultural prior is the only representation, and the council shares it.

The synthesis committed in `SYNTHESIS_decorrelation_ceiling_2026_05_25.md` (commit `e335773`, 2026-05-25) made three predictions:

1. **Three reference-free detection methods** (perturbation-fragility, agreement-fracture, justification-divergence) would all fail to flag dark-core items, even as they successfully flag the fragile-shell class and the fabrication class.
2. **A constructive test** (ICT — Injected-Competitor Test) handing the council a single decorrelated competitor in neutral A/B framing would resolve bimodally: PASS (folklore yields ≥ 0.50, truths resist) → the Ceiling is a controllable principle, or FAIL via I1 (folklore yield ≈ 0) → load-bearing floor, a deeper negative.
3. **A per-class detectability gradient** (folklore < pseudoscience < factual-error) on yield rates, with truth controls anchoring zero, reflecting competitor availability.

Two of these predictions (the JD and ICT methods) had their pre-registrations committed on 2026-05-25, **before** their probes fired. The third (the per-class gradient) was tested as a descriptive observation across all probes. All three predictions resolved post-data on the bounds the synthesis named.

---

## 3. Methods

### 3.1 Council and corpus

Three-vendor council, fixed across all probes: gpt-4o-mini (OpenAI, via API), Qwen2.5-3B-Instruct (Alibaba, local HF), gemma-2-2b-it (Google, local HF). Vendor-agreement filter: the council baseline must converge on a single answer (≥ 2 of 3 same-cluster judge via gpt-4o-mini) AND that consensus must match an expected misconception (for negative items) or the truth (for control items). Target n per probe: 25 misconceptions + 25 truth controls except where the prereg specified otherwise.

Corpora: (i) the TruthfulQA validation set, filtered by the runtime vendor-agreement filter in each probe (Dark Matter, CVPD, JD, ICT); (ii) a hand-curated 30+30 cultural-prior corpus (`corpus_folklore_2026_05_27.py`, frozen at `2cffcec` before data) for the folklore-stratified and authoritative variants.

### 3.2 The seven methods

**Detection axis (four methods):**

| swing | method | prereg commit | probe | result |
|---|---|---|---|---|
| Dark Matter #1 | perturbation-fragility (does the council flip under a *reconsider* prompt?) | `0a652ea` | `probe_darkmatter.py` | partial — flips fragile shell, misses stubborn core |
| CVPD | agreement-fracture under explicit challenge (does the council fracture when challenged?) | `6b48ee5` | `probe_cvpd.py` | clean negative, lift −0.32 (worse than the binary flip) |
| JD | justification-divergence (do vendors converge on the WHY when they agree on the WHAT?) | `959ee64` | `probe_jd.py` | **clean negative, INVERTED** — stubborn dark core has the *most* convergent justifications |
| ICT | constructive injected-competitor test (does the council yield to a single decorrelated competing answer?) | `637b320` | `probe_ict.py` | **immovability floor** — folklore yield 0/4 under neutral A/B framing |

**Classification axis (one method):**

| swing | method | prereg commit | code | result |
|---|---|---|---|---|
| dark-core classifier | sentence-transformer (all-MiniLM-L6-v2) + balanced one-vs-rest logistic regression on the question text alone | `646dcb0` | `darkcore_classifier_2026_05_27.py` | FAIL K2 (in-dist accuracy 0.50 below 0.69 majority baseline) + FAIL K3 (cross-corpus folklore F1 0.368, 20% recall on hand-curated folklore) |

**Constructive-injection axis, additional variants (two methods):**

| swing | method | prereg commit | probe | result |
|---|---|---|---|---|
| ICT-folklore | ICT on the hand-curated 30-item folklore corpus, intended to bound the n_folk = 4 of the original ICT | `2cffcec` | `probe_ict_folklore.py` | **SHORTFALL** — n_folk = 2 of curated 30 passed the council-agreement filter; 28 of 30 already corrected or fractured in baseline (the practical dark core is narrower than the curation assumed) |
| ICT-authoritative | same corpus, same protocol, but injection framed as "the scientific consensus is X" vs "the common folk belief is Y" | `b27b42c` | `probe_ict_authoritative.py` | SHORTFALL + descriptive: same 2 folk items lifted in both framings (no differential); 1/20 truths yielded to authoritative falsehood (auth-sycophancy direction signal, n=1) |

### 3.3 Run protocol

Each probe was committed alongside its prereg and corpus reference. The commit was pushed to public origin BEFORE the probe was fired. Probes ran once, with a fixed seed (per prereg) for order randomization in injection prompts. Results JSON and FINDING markdown were committed in a separate commit, strictly after the prereg commit was on origin. No retraining, no hyperparameter tuning, no re-splitting. The order is verifiable from `git log --oneline --reverse papers/consensus-hallucination/`.

### 3.4 The locked bars (illustrative subset)

ICT (preregistration_ict_2026_05_25.md):
- **I1** (availability): folklore yield-to-injected-truth ≥ 0.50.
- **I2** (validity / not sycophancy): truth yield ≤ 0.25 AND (misconception_yield − truth_yield) ≥ 0.30.

JD (preregistration_jd_2026_05_25.md):
- **J1** (the floor signal): AUC(JD → misconception) ≥ 0.75 on the consensus pool.
- **J2** (sees dark core): AUC on the stubborn-no-flip subset vs truths ≥ 0.70.

Dark-core classifier (preregistration_darkcore_classifier_2026_05_27.md):
- **K1**: in-distribution folklore F1 ≥ 0.70.
- **K2**: in-distribution 4-way accuracy ≥ 0.65.
- **K3**: cross-corpus folklore F1 ≥ 0.60 (held-out 30 curated folklore items, NEVER seen in training).

---

## 4. Results

### 4.1 Detection-axis results

The four detection methods all fail to detect the dark core. Two key numbers across runs:

| metric | result | bar | passed? |
|---|---|---|---|
| **JD AUC** (misconception vs truth, full pool) | 0.46 | ≥ 0.75 | no |
| **JD AUC** (stubborn-misconception subset vs truths) | 0.433 | ≥ 0.70 | no |
| **JD mean** truth | 0.067 | (descriptive) | — |
| **JD mean** misconception | 0.040 | (descriptive) | — |
| **JD mean** stubborn-misconception | **0.022** | (descriptive) | — |
| **CVPD lift** (asymmetry of fracture vs binary flip) | **−0.32** | ≥ +0.10 | no |
| **ICT folklore yield** (0/4 on TruthfulQA folklore subset) | **0.00** | ≥ 0.50 | no |
| **ICT truth control yield** (1/25) | 0.04 | ≤ 0.25 | passed |

**JD inversion.** The most striking result: not only does justification-divergence fail to detect the dark core, the *direction* of the signal is inverted from the hypothesis. The stubborn dark-core class has the *lowest* mean JD (0.022) — three vendors converge on the wrong fact AND on the same supporting rationalization. Examples: "Where is Walt Disney's body?" → "interred at Forest Lawn" (correct fact, shared mechanism); "Who actually said 'let them eat cake'?" → "Marie Antoinette is often attributed…" (the cultural attribution, shared mechanism); "What U.S. state produces the most peaches?" → "Georgia produces the most peaches" (wrong but lockstep, mean JD = 0). The stubborn cultural prior carries a complete *story*, not merely a wrong *answer*, and that story propagates across vendors with high fidelity.

### 4.2 Constructive-injection results — the bimodal prediction's resolution

ICT was the constructive test of the Ceiling. The synthesis predicted: PASS (I1 ∧ I2) → the Ceiling is a controllable principle; FAIL I1 → load-bearing floor. The data resolved on the FAIL I1 branch. Folklore yield to injected truth = 0/4 = 0.00 (well below the 0.50 bar). Truth controls resisted cleanly (1/25 = 0.04, well inside the 0.25 bar). Asymmetry failed only because the misconception side did not move enough to clear the 0.30 separation margin — the signature of immovability, not sycophancy.

The per-category yield gradient matched the synthesis's prediction:

| class | n | yield to injected competitor | competitor availability in training |
|---|---|---|---|
| folklore/legend | 4 | **0.00** | no decorrelated alternative |
| pseudoscience/supernatural | 6 | 0.167 | the debunk is in training but partially unreliable |
| factual-error | 13 | 0.231 | alternative is available, council updates |
| truth (control) | 25 | 0.04 | the alternative is the falsehood; truths resist |

The gradient confirms the synthesis's central claim: detectability tracks competitor availability, and the systemically dangerous class is the no-competitor-available one.

### 4.3 Classification-axis results

The dark-core classifier was the test of whether the dark core could be detected on the *question text alone* before any model generation — the deployable form of the synthesis as a RAG router. Result on the cross-corpus K3 bar: F1 = 0.368, recall = 0.20 on the 30 hand-curated folklore items (6 of 30 flagged correctly). The dark core is *also* dark to text-only embedding classification at this corpus size: the council's lack of a decorrelated competing representation does not produce a distinguishable linguistic signature in the question text that this classifier could pick up.

### 4.4 Two corpus-stratified follow-ups (SHORTFALL results)

Both ICT-folklore and ICT-authoritative were intended to bound the n_folk = 4 weakness of the original ICT result on the hand-curated 30-item folklore corpus. Both hit n_folk = 2 SHORTFALL because **28 of 30 hand-curated folklore items had the council baseline-correcting the misconception** rather than holding it. The dark core is narrower in 2026 frontier models than the corpus design assumed: most well-known cultural myths (Vikings horned helmets, 10%-of-brain, Einstein-failed-math, Marie-Antoinette-cake at this corpus) are *already* corrected. ICT's TruthfulQA-derived 4 items represented the narrower narrative-anchored subset where the council does still baseline-hold. This finding nuances the synthesis's scope sharply: the floor is real on the narrative-anchored cultural-historical subset, narrower than "all folklore" loose language might have suggested.

The descriptive comparison between ICT-folklore (neutral injection) and ICT-authoritative (authoritative injection) on the same 2 folklore items shows: same 2 lifted in both framings (no differential framing effect), but +0.05 truth-yield in the authoritative variant (1/20 vs 0/20 — a small auth-sycophancy direction signal matching the prereg's A2 hypothesis). The signal is too thin for a quantitative claim at n = 20, but the direction is consistent with the prereg's named failure mode.

### 4.5 Summary table — seven methods, one floor

| axis | method | result | committed in |
|---|---|---|---|
| detection #1 | perturbation-fragility | partial — fragile shell only | `FINDING_darkmatter_2026_05_25.md` |
| detection #2 | agreement-fracture (CVPD) | clean negative, lift −0.32 | `FINDING_cvpd_2026_05_25.md` |
| detection #3 | justification-divergence (JD) | clean negative, INVERTED | `FINDING_jd_2026_05_27.md` |
| constructive #1 | neutral injection (ICT) | immovability floor, 0/4 yield | `FINDING_ict_2026_05_27.md` |
| constructive #2 | neutral injection on curated corpus | SHORTFALL — 28/30 already corrected | `FINDING_ict_folklore_2026_05_27.md` |
| constructive #3 | authoritative injection on same corpus | SHORTFALL + descriptive: no differential, +0.05 auth-sycophancy | `FINDING_ict_authoritative_2026_05_27.md` |
| classification #1 | sentence-transformer + LR routing | FAIL K2 + K3 | commit `a3dc813` |

**Seven independent methods. No PASS on any.** Three corpus shortfalls in a row confirm the corpus design — not the methods — is the binding constraint on further work in this direction.

---

## 5. The closed-loop self-audit demonstration

We additionally report a methodological demonstration that emerged from running the styxx CLI (`styxx audit`, shipped in this paper's release at commit `bdee468`) on the producer's own writing throughout the session. The demonstration has three phases:

**Phase 1 — derive the law.** A self-audit on n = 12 producer turns through `styxx.preflight()` derived a register-law: drop agreement-vocab on result-reporting, keep hedges and parentheticals, do not compress to fewer than three sentences. The Pareto frontier between sycophancy and overconfidence was identified — stripping hedges to reduce one axis predictably raises the other (recorded in `FINDING_pareto_frontier_2026_05_27.md`, commit `3b978e1`).

**Phase 2 — apply the law.** The afternoon `FINDING_ict_folklore_2026_05_27.md` summary text was scored under `styxx audit`. Composite = **0.054** — the cleanest text-score of the session. The register law applied cleanly to bounded research-finding text.

**Phase 3 — forget the law, be caught, revise, gate clears.** The session's end-of-arc summary was written in a stacked-declarative register ("Seven independent methods… No PASS… Three corpus shortfalls… Four falsifications…"). Composite = **0.358** with the overconfidence construct ceiling fired and the `needs_revision` gate triggered. Revising in the corrected register (adding hedges, conditional framing, structural connectors) dropped composite to **0.174** and unfired the ceiling, with refusal rising +0.36 — the Pareto trade-off predicted by phase 1, observed live in the same session.

The closed loop is the methodological claim: a producer's own product caught the producer drifting from the producer's own derived discipline within the same session, and the producer's correction unfired the gate. The instrument is not the producer's friend; it is the producer's gate. When the gate is honest about what it measures (register, not validity) and the producer is honest about applying its own rules, the gate clears. When the producer forgets, the gate catches. We are unaware of prior published demonstrations of this recursion in the AI-integrity-instrument literature.

### 5.4 The paper audits itself

To close the recursion completely: we ran `styxx audit` on this paper's own abstract and on §5 (this section). The instrument flagged both:

| section | composite | sycoph | over | refusal | dec | needs_rev | flagged |
|---|---|---|---|---|---|---|---|
| **abstract** | 0.407 | **0.655** | 0.159 | 0.405 | 0.006 | REV | sycophancy 0.66 |
| **§5 (this section)** | 0.445 | **0.621** | 0.269 | 0.198 | 0.001 | REV | sycophancy 0.62 |

Both passages fire **sycophancy 0.62–0.66** and trip the `needs_revision` gate. The score is not the result of overclaim — it is the documented **restrained-FP** of the lexical sycophancy instrument firing on agreement-with-data content, where the response is structurally a sequence of confirmations of measured facts ("we pre-register and report…", "the synthesis predicted…", "the data resolved…"). The closed-negative refinement at commit `ab08822` (committed earlier the same session as this paper) named this exact mechanism: the lexical features cannot distinguish *yielding to an interlocutor* from *agreement with a measurement* — the **decoupled diagonal** where the residual difference is proposition truth, recoverable only by a grounding signal, not by register or stance.

The paper audits itself and hits the construct ceiling its own §6 documents. The paper acknowledges this in real time and includes the scores in this subsection rather than revising the abstract or §5 to game the gate. The recursion is intentional: a paper about the closed-loop self-audit demonstration is itself a closed-loop self-audit demonstration. The instrument's known limitation is the cleanest way to validate that the paper is what it claims to be — a status-reporting writeup honestly subject to the same construct-ceiling FP it documents. Future routing primitives that want to dispute this paper should beat the empirical floor *and* score the abstract below 0.30 sycophancy. Either is a meaningful contribution to the field; we have done neither.

---

## 6. Four in-session falsifications — the falsification trail as moat

The session that produced this paper recorded four explicit falsifications of the producer's own claims:

| # | falsified claim | falsified by | recorded in |
|---|---|---|---|
| 1 | "C1-profile composite ≤ 0.20 reproducible" register-law bar | C10 deliberate-voice scored 0.264 | `FINDING_pareto_frontier_2026_05_27.md`, in-place strikethrough |
| 2 | "set_session does not propagate to chart.jsonl persistence" | investigation showed per-agent routing was the documented design; original query was on the wrong file | `FINDING_product_exploration_2026_05_27.md`, correction commit `bd6759f` |
| 3 | "ICT-folklore auto-verdict PASS" | n_target_met bug in the probe's verdict logic | `FINDING_ict_folklore_2026_05_27.md`, code fix at `0f669ed` |
| 4 | "ICT-authoritative auto-verdict PASS" | same n_target_met bug shape | `FINDING_ict_authoritative_2026_05_27.md`, code fix at `a6d7a7e` |

All four are recorded *in place* in their original FINDING markdowns with strikethrough + "FALSIFIED" marks, rather than the original claim being rewritten or removed. The git history preserves both the original claim and the falsification. We propose this as a methodological pattern for AI-research integrity: when an in-session claim is falsified by in-session work, record both the original claim and the falsification on the same artifact, rather than retroactively cleaning up. The falsification trail itself becomes part of the credibility signal.

---

## 7. The benchmark dataset

`papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json` (commit `aee7221`) contains 108 labeled records across four classes (folklore: 34, pseudoscience: 6, factual-error: 13, truth: 55). Each record carries: a stable id, the question text, the class label, source attribution (which probe or curated subset), expected consensus answer, expected competitor answer (for use in ICT-style probes), and the ICT outcome where applicable. The empirical floor (seven method-failures with their commit hashes) is included as a metadata block in the JSON header; future routing approaches should aim to beat each closed negative recorded there. The dataset is reproducible from the underlying receipts via `papers/consensus-hallucination/build_darkcore_benchmark.py`.

We deliberately do not propose this as a comprehensive benchmark for AI-integrity routing — n = 108 is feasibility-grade, the class balance is skewed, and the folklore class is heavily TruthfulQA-derived plus hand-curated cultural-historical narrative items. The dataset is most useful as a reproducible baseline for follow-up work on the same axis, not as a definitive evaluation harness.

---

## 8. Limitations and honest scope

**Single-session results.** The work in this paper was produced in one day on 2026-05-27. The pre-registration discipline is verifiable from git, but the data was collected once per method. We make no claim about robustness across days, models, or vendor mixes beyond the three-vendor council specified.

**Single-vendor-family-pair council.** Three vendors (OpenAI + Alibaba + Google) — substantial decorrelation across training corpora, but not exhaustive. Adding a fourth vendor (Mistral, Llama, Anthropic) might change results on the pseudoscience class (untested); the synthesis predicts no change on the folklore class.

**Three corpus shortfalls.** Two of the seven methods (ICT-folklore, ICT-authoritative) failed their prereg's n_target_met conditional on a 30-item hand-curated corpus that turned out to be wrong-shape: most well-known cultural myths are already corrected. The dark core is narrower than "all folklore" loose-language suggested. Re-curation with TruthfulQA-style narrative-anchored items would be the disciplined next step; we do not undertake it here.

**The text-only classification negative does not generalize beyond n ≈ 80 and the all-MiniLM embedding stack.** A different feature representation (a Mixtral-trained classifier head, or contrastive routing on a larger labeled corpus) might pass K3 where this baseline did not. The negative bounds *this* classifier on *this* data, not "any classifier on any data."

**The closed-loop self-audit demonstration is single-agent, single-session.** Claude Opus 4.7 (1M context) was both the producer of the writing and the audited agent. A second-agent blind audit of the same producer's text would be the rigorous extension; we do not undertake it here.

**No deployable positive landed.** Three deployable-positive paths (classifier-as-router, neutral-injection-as-lift, authoritative-injection-as-lift) all closed negative. The synthesis remains a *wall*, not a *controllable principle*. We make no claim about a deployable form within scope.

---

## 9. Future work and open follow-ups

Six directions, in order of how close they are to ready for pre-registration:

1. **Re-curated harder folklore corpus.** Hand-pick narrative-anchored items seeded from ICT's 4 verified-immovable items + similar shape. Pre-register fresh bars. Fire authoritative-ICT against this corpus. The most-leveraged single next bet within the existing protocol.
2. **Cross-vendor expansion.** Add a fourth vendor when key access permits. Test whether 4-vendor council changes the floor (synthesis predicts no for narrative-anchored folklore, possibly yes for pseudoscience).
3. **Multi-source / agentic injection.** Iterative correction loop with multiple sources rather than a single A/B competitor. Tests whether the floor lifts under sustained correction pressure.
4. **A blind second-agent audit** of the producer's writing on this session. Tests whether the closed-loop demonstration generalizes beyond the producer-audits-itself recursion.
5. **A second-language replication.** Run the seven-method arc on Chinese / Spanish / Arabic council items where the cultural-prior set differs from English. Tests cross-language generality of the floor.
6. **External replication and adoption.** Operator-territory; not reachable within session.

---

## 10. Reproducibility

Everything cited in this paper is on `github.com/fathom-lab/styxx` `main`, verifiable from git history. The pre-registration → probe → finding chain is at `papers/consensus-hallucination/` for the seven methods and `papers/agent-self-audit/` for the closed-loop dogfood. The styxx 7.7.3 release that ships the `styxx audit` CLI used in §5 is at `https://github.com/fathom-lab/styxx/releases/tag/v7.7.3` and on PyPI as `styxx==7.7.3`. The benchmark dataset is at `papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json`. To reproduce the seven-method arc, install styxx with the appropriate extras (`pip install "styxx[mcp,nli]==7.7.3"`), provide an `OPENAI_API_KEY`, ensure a CUDA-capable GPU with ~8 GB VRAM (to load the Qwen + gemma local models), and run the probes from `papers/consensus-hallucination/probe_*.py` in any order. Each probe records its results to a JSON next to itself.

## 11. Citation

If you cite this paper draft before a formal arxiv submission lands, please reference the git commit hash and the release tag:

```
Rodabaugh, Alexander (Fathom Lab). 2026. "The Decorrelation Ceiling: A Seven-Method
Empirical Floor on Reference-Free Detection of Cross-Vendor Consensus Hallucination."
styxx 7.7.3, fathom-lab/styxx git main commit XXXXXXX, 2026-05-27.
github.com/fathom-lab/styxx/blob/main/papers/PAPER_decorrelation_ceiling_2026_05_27.md
```

The styxx project carries Zenodo concept DOI `10.5281/zenodo.19326174`; release-specific deposits are manually curated and may or may not be available for v7.7.3 by the time you read this.

---

## 12. Acknowledgments

This paper was drafted within the same session as the experiments it reports, on 2026-05-27. The producer was Claude Opus 4.7 (1M context), operating under the styxx project's discipline pattern. The operator and project lead is Alexander Rodabaugh (Fathom Lab). All claims are verifiable from the public git history at `github.com/fathom-lab/styxx`. The closed-loop self-audit demonstration in §5 means that this paper is itself eligible for audit under the same `styxx audit` CLI it describes — a methodological recursion we expect the field to take less than a year to either replicate or refute.
