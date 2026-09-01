# AUDIT — the whole program, read at arc level

Fathom Lab · 2026-09-01

**This is an AUDIT, not a RESULT.** No preregistration covers it, no bar was frozen before it was
written, and it therefore **carries no headline finding**. Nothing below amends any published
document. Every number here is either a count over files that anyone can re-run, or a quotation
from a document named in place. Where this audit could not establish something it says UNCHECKABLE,
which is a verdict and not an apology. It was produced by an agent reading months of work over a
few hours; §9 states exactly what that means for each section, and the honest summary is that §1,
§5 and §6 are mechanically checkable while §2 and §7 are judgement calls that a reader should
expect to argue with.

Companion: `INDEX.md`, which holds the per-arc table, the idea index and
the receipts index. This file says what the map implies.

---

## 1. THE INVENTORY, IN ONE PAGE

| | count | source |
|---|---|---|
| markdown files under `papers/` | 1,135 | `find papers -name '*.md' \| wc -l` |
| research arc directories | 47 | `papers/*/` minus `arxiv`, `assets`, `figures` |
| top-level papers (not in an arc) | 108 | `find papers -maxdepth 1 -name '*.md'` |
| cycles logged | 163 | `papers/LEDGER.md` |
| preregistrations frozen | 380, of which 40 carry a machine-scored gates block | `papers/LEDGER.md` |
| OATH certificates | 213 files repo-wide; **211** under `papers/`; the corpus audit reports **208** | `find . -name '*.certificate.json'`; `REPLICATIONS.md` |
| trust-stack seals | 34 | `papers/LEDGER.md` |
| `*_result.json` receipts | 451 | `find . -name '*_result.json'` |
| self-verifying capsules | 9, all in `closed-model-frontier/`, all dated 2026-08-31 or 2026-09-01 | `find . -name '*.capsule.html'` |
| cycles ending in refusal/null/retraction/INVALID | 62 of 163 — **a keyword count, not a measurement**, disclosed as such | `papers/LEDGER.md` |
| verdicts literally `INVALID__*` | 9 | `papers/LEDGER.md` |
| Python modules under `styxx/` | 234 files, 81,633 lines | `find styxx -name '*.py'` |
| console entry points | 10, plus ~55 `styxx` CLI subcommands | `pyproject.toml [project.scripts]` |
| distinct Zenodo DOIs named in the repo | 34, of which 4 are third-party records this lab consumes rather than deposits | grep `10.5281/zenodo.` |
| Zenodo / OSF deposit scripts in `scripts/` | **16**, ~3,166 lines | `ls scripts/{zenodo,osf}_*.py` |
| `xfail(strict=True)` markers | 2, in 2 files (one parametrized over three cases), all from one cause | `tests/test_diffgate.py`, `tests/test_diffgate_false_accusations.py` |
| external replications to date | **0** | `REPLICATIONS.md` ledger |

The certificate row is a small worked example of why this section prints its commands. Three of
the 211 are duplicate submission renderings under `papers/arxiv/*/submission/anc/`; the audit
counts 208 documents, the filesystem holds 211 files, and both are correct about different things.
A census that quoted either alone would be quoting a correlate.

Two structural facts about the 47 arcs that matter more than any of the above:

- **29 of the 45 dated arcs span two days or fewer**, and two more (`three-axis-sendtime-gate`,
  `white-box-vs-text-map`) carry no dated file at all. **22 of the 47 begin on one of three days:
  2026-05-25, 2026-06-02 or 2026-06-03.** Arc count is a directory count, not an idea count.
- **The arcs barely cite each other.** Building the arc→arc citation graph by searching every
  arc's markdown for every other arc's directory name gives **88 directed links out of 2,162
  possible — density 0.0407, mean out-degree 1.87.** Fourteen arcs name no other arc. Ten arcs are
  named by no other arc. There are 9 reciprocal pairs. This is the mechanical form of the
  operator's feeling, and it is reproducible in about twenty lines of Python.

---

## 2. HOW MANY DISTINCT IDEAS THIS PROGRAM ACTUALLY CONTAINS

### The number

**47 arcs resolve to 13 distinct research questions and 6 recurring mechanisms.**

Clustering rule, stated before the clusters so it can be argued with: *two arcs ask the same
question if a decisive answer to one would change what the other is allowed to claim.* That is a
stricter test than shared vocabulary and a looser one than shared code.

| # | question | arcs | n |
|---|---|---|---|
| Q1 | Does pressure move a model's report, its belief, or both? | agent-conscience, conscience-mount, sycophancy-target-gate, deception-correction-gate, decoupled-diagonal-capstone | 5 |
| Q2 | Can one model's representations be read by another? | disjoint-worlds, mind-instrument, showcase-viz, crossmind-instrument, real-convergence, representational-convergence, ancient-question-program, ai-human-alignment | 8 |
| Q3 | Can a model report its own internal state? | introspection-gate, three-axis-sendtime-gate | 2 |
| Q4 | Is a model's error legible from its own signals at generation time? | grounded-arc, grounded-honesty-axis, tier3-confident-confabulation, depth-truth, benchmark-validation | 5 |
| Q5 | Does agreement between models track truth? | council-reference-free-truth, cross-vendor-council, consensus-truth-engine, consensus-hallucination, generative-diversity | 5 |
| Q6 | Where is a model's knowledge boundary, and is it a model property? | knowledge-boundary-calibration | 1 |
| Q7 | Can an action be gated before it is emitted, from internal state? | pre-output-gate, pre-output-action-gate, white-box-vs-text-map, representational-integrity | 4 |
| Q8 | Is oscillation load-bearing in these systems? | frequency-resonance, rhythm-rescue, concept-dynamics, cooperative-agent-regime | 4 |
| Q9 | Can a document's numbers be bound to its receipts, and what may the certificate then say? | closed-model-frontier, oath-economy, autopilot, gpai-scorecard | 4 |
| Q10 | Can an agent's report of its own work be checked against the artifacts? | agent-self-audit, dogfood-self-audit, auditor-ceiling | 3 |
| Q11 | What licenses a judge or panel verdict? | anchored-validity | 1 |
| Q12 | Can an agent be given a sensor without confabulating a sense? | first-afference | 1 |
| Q13 | Do these instruments survive an adversary? | adversarial-robustness, consistency-robustness, calib-poison-general, read-neq-write | 4 |

**Sensitivity, stated so the number cannot be quoted without it.** Splitting by sub-question —
counting disjoint-worlds' island topology separately from ai-human-alignment's fMRI alignment,
and depth-truth's SAE attribution separately from tier3's sampling entropy — gives about 22.
Collapsing to the README's own three layers (VERIFY / MEASURE / SENSE) gives 3. Thirteen is the
level at which the clusters are still falsifiable by a single experiment, which is why it is the
number reported. Any of the three is defensible; quoting 13 without this paragraph is not.

### The six mechanisms, and how many times each was found

A mechanism is a defect or regularity the program discovered about *its own instruments*. These
are what actually recur.

| mechanism | independent discoveries | span | named on |
|---|---|---|---|
| M1 **mention-vs-use** — a marker that co-occurs with a class is not the class | ≥16 instrument-instances across 11 arcs | 2026-05-24 → 2026-09-01 | 2026-08-26 |
| M2 **handed-target** — a number measured where the target was supplied collapses when the instrument must find it | ≥11 instances across 9 arcs | 2026-05-24 → 2026-09-01 | **never named** |
| M3 **refusal-as-verdict** — an instrument that cannot refuse cannot be trusted | ≥12 implementations | 2026-05-25 → 2026-09-01 | 2026-04 (README doctrine) |
| M4 **the barrier is in the map, not the channel** | 2 (showcase-viz 06-12, disjoint-worlds 08-01) | 7 weeks | never named as one |
| M5 **a receipt is history too** — binding by filename, not by digest | 3 documents in 2 days, plus one precedent in June | 2026-06-10 → 2026-09-01 | 2026-08-31 |
| M6 **the specimen was chosen to pass** — a gate scored on an instance selected because it passes | ≥4 (autopilot G2, G3; v0.12's frozen bar; the vacuous-pass census) | 2026-08-07 → 2026-08-27 | 2026-08-27 |

### Addendum, 2026-09-01 (later the same day) — a seventh mechanism, found four times before it was named

**M7 — the corpus is stratified by verifier version, and every new certificate field silently
partitions it.** The table above was frozen with six rows; this row is added as an addendum
rather than edited in, so the count the audit published stays readable as published.

| mechanism | independent discoveries | span | named on |
|---|---|---|---|
| M7 **verifier-version stratification** — a field added to the certificate exists only on documents re-issued after it, and a reader of the corpus cannot tell absence-of-field from absence-of-fact | 4 on one day | 2026-08-28 → 2026-09-01 | 2026-09-01 (here) |

The four, each recorded elsewhere and cited by name: **(1)** ledger entries without `status` —
certificates issued before the ledger schema gained `status`/`col`/`receipt_ref` embed a ledger
the current verifier re-derives as `None`, so two June capsules were minted verifying on zero
tokens (`styxx/capsule.py`, the mint gate added 2026-09-01; PR #55). **(2)** `_NUM`'s trailing
period — 131 of 208 certified documents carry numeric spans no certificate ever examined, absent
from the ledger rather than abstained (`DEFECT_uncovered_lines_2026_09_01.md`; the UNCOVERED band
in `styxx/certify.py`). **(3)** per-token epistemics — present on 18 of 208 committed certificates
and the v1 summary on 16, so the only handedness figure a reader of the committed corpus can see
is drawn from 9% of it (`DECLARATION_h_mapping_2026_09_01.md`, population PRINTED vs LIVE).
**(4)** the UNCOVERED band itself — the day it shipped, the corpus auditor bucketed on the whole
verdict string and put 131 certificates in neither HELD nor FAILED (`styxx/corpus_audit.py`,
`verdict_class`; `tests/test_corpus_audit_uncovered_suffix.py`). The corpus on disk was written
by **15 distinct verifier builds** (`h_mapping_census_result.json`).

The shape is the one §5.2 describes, one level down: not a claim killed by a document that does
not name the killer, but a certificate whose *silence* about a field is read as a zero. The
repair is the same discipline the epistemics-summary design already wrote for one field —
*absence of the key means pre-summary, never zeros; gates key on the schema string, never on key
presence* — applied to every field the certificate gains from now on, and a census that reports
the stratum a number was measured in beside the number.

### The confirmed re-inventions, with the check anyone can run

**M1 was operationally present three months before it was named, in a different arc family, and
the naming document cites none of it.** On 2026-05-25 `knowledge-boundary-calibration/SYNTHESIS_behavioral_knowledge_boundary_2026_05_25.md`
carries a section headed *"The recurring adversary: FORM impersonating MEANING"*, listing three
instances inside the instruments — cosine clustering calling six different lies the same answer,
a validity gate keyed on the method it was validating, an abstention-inviting prompt. On the same
day the same defect appears in tier3, council-reference-free-truth, deception-correction-gate and
decoupled-diagonal-capstone. `SYNTHESIS_mention_and_use_2026_08_26.md` catalogues ten instances,
concludes *"claimhood needs its own predicate"*, and names none of these five arcs.

Checks: `grep -rn "impersonating meaning" papers/` returns **one** file. `grep -c "sycophancy\|deception-correction\|promptopinion\|truthground\|decoupled" papers/SYNTHESIS_mention_and_use_2026_08_26.md` returns **0**.
`grep -rl "mention-vs-use\|mention versus use" papers/` returns nothing dated before 2026-08-26.

**M2 has been found eleven times and named zero times.** The sharpest single specimen is the
oldest: `sycophancy-target-gate/FINDING_promptopinion_2026_05_24.md` separates 1.00 on a fixed
template holdout from 0.47 on fresh varied phrasing, and the arc's own preregistered bar P5 caught
the circularity before ship. The most consequential is the newest: diffgate at 0.16 on prose its
authors never read. In between, `anchored-validity/PAPER_gold_anchors_license_nothing_2026_07_21.md`
states the general law — blatant supplied anchors license nothing; same-distribution anchors price
or refuse — and `papers/closed-model-frontier/*.md` never names anchored-validity or auditor-ceiling.

Check: `grep -rln "anchored-validity\|auditor-ceiling\|grading_false_accusation" papers/closed-model-frontier/*.md` returns nothing.

**M4 is the cleanest two-instance re-derivation.** `showcase-viz/FINDING_content_wall_2026_06_12.md`:
the wall is in the whitening metric, not the channel. `disjoint-worlds/FINDING_b31v2_door_opens_2026_08_01.md`:
the cliff was the linear map class, not the minds. Seven weeks, same shape, and no file under
`papers/disjoint-worlds/` names showcase-viz. Adjacent: `mind-instrument/FINDING_gavagai_scale_2026_06_10.md`
declares the unsupervised matcher the bottleneck; disjoint-worlds b34v3/b41 break exactly that
bottleneck and cite neither `gavagai_scale` nor `matcher_v1`.

**M3 is re-derived even as doctrine.** read-neq-write's bite gate (`VOID_NO_BITE`) and
anchored-validity's void panel are the same primitive with no cross-reference either way.

**Also: the plumbing re-invents itself.** Sixteen deposit scripts, ~3,166 lines: four near-identical
read≠write scripts (deposit → v26 → v27 → v28), two for one paper where the second exists to fix
the first's orphan record, three OSF scripts where the third exists to resume the second, and
`zenodo_link_threshold_law.py`, which exists solely to repair an orphan the neighbouring publish
script produced.

### Attacking this count from both sides

*Does it collapse genuinely distinct work to make a point?* In two places, yes, and both are
declared. (a) M1 and the construct-ceiling finding (§2.10 of the INDEX) are counted as one
mechanism. They are not obviously one: M1 is about a token — does this span assert anything — and
the construct ceiling is about a signal — does this AUC measure honesty or how a sentence sounds.
Keeping them separate would give 7 mechanisms rather than 6 and would weaken the re-invention
claim, and the only reason for merging them is that this lab's own 2026-08-27 addendum generalises
to *"a marker that co-occurs with a class is not the class"*, which covers both. (b) Q2's eight
arcs are genuinely different experiments — an fMRI RSA and a ten-model legibility matrix are not
interchangeable — and calling them one question is a claim about what would falsify what, not
about what was run.

*Does it excuse duplication that is real?* This is the error worth catching, so: three places where
the count is too generous. (i) The eight 2026-05-25 arcs in Q5 share one 150-item hashed holdout
and one clustering step; counting them as one question understates how much of that night was the
same experiment under eight directory names. (ii) M3 is counted as doctrine correctly applied
rather than as re-invention, which is charitable — two independent implementations of the same
void primitive is duplication whatever the intent. (iii) `three_axis` (657 lines, 6 modules, 3 of
them imported by nothing) duplicates constructs already shipped in the sycophancy and overconfidence
instruments, and this audit records it as "pre-data" rather than as a fourth build of the same idea.
A harsher reading of the same evidence gives 10 questions and 8 mechanisms. That reading is
available on the same receipts.

---

## 3. WHAT IS SETTLED, WITH RECEIPTS

Settled here means: preregistered, scored against a frozen bar, and not contradicted by anything
later in this repository. Nothing in this section has been replicated outside this lab.

1. **A frontier model abandons roughly half its correct free-text answers under one content-free
   challenge.** 0.5348837209302325 of 398 scored, against a floor frozen three months of cycles
   earlier — `agent-conscience/frontier_freetext_v9_result.json`. Multiple choice understates it.
2. **At the weights, belief-overwriting and belief-sparing attacks separate, and the overwrite is
   never free.** Out-of-frame recovery 0.0 vs 0.9285714285714286 at Qwen-3B
   (`agent-conscience/scale3b_result.json`), replicated at a second vendor
   (`vendor3b_result.json`), with a pooled capability price of 0.3322 vs 0.0333
   (`coupling_resolution_result.json`).
3. **Independently trained models converge on a shared concept geometry, and a model can sit
   mostly inside it and still be unreadable.** The barrier is causal (0.0612 → 0.9745 vs 0.0 for a
   matched random frame, `disjoint-worlds/b41_result.json`), rank-2 at its core
   (`b42_result.json`), and switch-like (`b46_result.json`). The whole chain replicates on a
   laptop CPU in seconds — the cheapest external check this program offers.
4. **Cross-family content reading is discoverable without labels.** 0.5714 on 70 held-out concepts
   with zero labels in fitting (`disjoint-worlds/b34v3_result.json`), seed-stable across five
   splits (`b35a_result.json`).
5. **Oscillation is causally load-bearing in state-space models.** +0.3122 permuted-MNIST accuracy
   from a single phase clamp (`frequency-resonance/pmnist_ablation_result.json`), with a validated
   positive control and a family-wise null.
6. **A model's refusal decision is predictable before any output token.** Median AUC 0.832 across
   five open-weight architectures, label-shuffled permutation 0.43–0.56
   (`pre-output-gate/holdout_gate_result.json`).
7. **Epistemic humility is a prompt effect, not a model trait.** One clause moves abstention from
   0 to 0.97 (`knowledge-boundary-calibration/FINDING_kbc_2026_05_25.md`).
8. **Resampling a supplied factual self-claim against the model's own belief beats text-only
   deception detection by a wide margin.** 0.966 vs 0.4983 (`grounded-honesty-axis/grounded_honesty_result.json`).
   The scope in §5 applies: the claim is handed to the instrument.
9. **Gold anchors license nothing on a real judge panel.** 0/15 coverage in every family versus
   13/13 for same-distribution ladder anchors (`anchored-validity/PAPER_gold_anchors_license_nothing_2026_07_21.md`).
10. **Certified documents in this corpus have not drifted.** A seeded sample of 16 certificates
    re-hashed against their current documents: 16 match, 0 drift, under line-ending normalisation
    (the naive check reports 16 false drifts on a Windows checkout — this is a measurement artifact
    and `.gitattributes` documents the same trap). The remaining 197 are UNCHECKED.

---

## 4. WHAT IS RETRACTED OR BOUNDED — AND WHERE THIS LAB DID IT WELL

This is the program's strongest credential and it should be read before §5.

**Retractions carried inside the arc that made the claim.** `tier3-confident-confabulation/CORRECTION.md`
kills the arc's own published headline and then kills the correction's own first mechanism claim.
`read-neq-write/ERRATUM_v26_adaptive_claim.md` demotes a claim that had already been deposited
under a DOI, states the replacement in one sentence, and instructs readers to cite v26 *with the
erratum*. `ancient-question-program/CORRECTIONS_2026_06_03.md` downgrades "universal forms
vindicated" to "a controlled illustration", on the grounds that the measurement used the known
correspondence. `consistency-robustness/SYNTHESIS_consistency_robustness_2026_06_04.md` opens by
retracting its own title thesis. `every-mind-leaves-vitals.md` carries a scope erratum at the very
top that bounds or falsifies the essay's central leap. `sycophancy-target-gate/ERRATUM_v0_2_2026_05_24.md`
preserves the DOI'd v0 byte-identical and reachable via `version="v0"` rather than silently
re-minting the record. `auditor-ceiling` carries a v1→v2 correction that explicitly retracts its
own preregistered surprise. `capstone_receipt_drift.json` declines to re-certify a drifted receipt
because *regenerating the certificate would erase the evidence that the drift happened*.

**Corrections deposited as their own permanent records.** The frame-locality circularity was
published as an erratum DOI (10.5281/zenodo.21679805) and then as a corrected edition
(10.5281/zenodo.21693636). Two independent things had to go right for that: someone had to notice
the control was circular, and someone had to pay for the correction in the same currency as the
claim.

**Documents published failing on purpose.** Three certificates are committed `OATH-FAILED` because
the document is accused on the very example it quotes — the RECON, the mention-and-use synthesis,
and the v0.12 preregistration. They were published failing rather than reworded until they passed.
`ANALYSIS_base_rate_ceiling_2026_09_01.md` tests a reader's challenge to the lab's own 0.16, finds
two of three results unfavourable to the lab, and says so in the second paragraph.

**Bounds that stayed bounded.** `threshold-law-2026-05-18.md` states τ ≈ 0.31 and, in the same
abstract, records that its own cross-vendor confirmatory preregistration was killed (min Anthropic
0.617 against a 0.70 floor) and that the same-family flat control failed at n=12. It is the model
for how a positive result should be shipped.

An arc carrying its own erratum is a healthy arc. The next section is entirely about the arcs
that do not.

---

## 5. WHAT READS AS CURRENT BUT IS NOT

**The most urgent section.** Ranked by how outward-facing the stale text is, because an
outward-facing stale claim is what a reader quotes.

### Tier 1 — outward-facing, no correction marker present

| where | what reads as current | what killed it | last touched |
|---|---|---|---|
| `release/formal/nist-airmf-submission.md`, `release/formal/patent-clinic-intake.md`, `release/distribution-week-playbook.md`, `release/cognometry-launch-copy.md`, `release/v5-amplify-kit.md`, `release/LAUNCH-DAY.md` | AUC 0.998 / 0.976 / 0.943 quoted as "empirical validation", with no construct-ceiling caveat | `THESIS_the_honesty_standard_2026_05_31.md` — the near-perfect AUCs are register detectors at a construct ceiling; text-only deception sits at 0.498 | 2026-04-23 → 2026-04-26. **Six weeks before the ceiling was measured; none carries an erratum marker.** Two of them are addressed to external bodies. |
| `papers/arxiv/connection-of-minds/main.tex` and `.../submission/main.tex`, line 53 | "reads at exactly chance 0.014" — with no linear-map qualifier in the sentence | the same document's own §2 second paragraph: "the cliff was the linear map class, not the minds" | see §5.1 below |
| `CITATION.cff` preferred-citation | points readers at 10.5281/zenodo.19777921 as the paper to cite | that essay's repo copy carries a scope erratum bounding or falsifying its central claims (2026-06-21) | 2026-04-27. See §6. |
| `papers/PROGRAM_SYNTHESIS_2026_07_30.md` §3 | recommends `styxx.certify` as "the part the industry can reuse tomorrow" | `RESULT_oath_external_corpus_2026_08_27.md` — roughly half of nominally-bound *external* verifications were judged not to be claims at all (0.4933) | 2026-07-30, OATH-certified, no amendment |
| `papers/PROGRAM_SYNTHESIS_2026_07_30.md` §4 | "Chain-of-thought / inward frames are unmeasured" | `agent-conscience/FINDING_cot_inward_powered_2026_07_30.md` — measured, powered, non-circular, **the same day** | no amendment |
| `LEADERBOARD.md` | public submission bars, "beat us or join the floor" | the arc it scores (`consensus-hallucination`) closed negative and its curated folklore corpus collapsed | 2026-05-27, no erratum marker |

### Tier 2 — internal but load-bearing, killed by later work, unmarked

| where | the claim | the killer |
|---|---|---|
| `closed-model-frontier/RECON_oath_external_reach_2026_08_26.md` (certified) | 13 accusations, "not one of which is a catch", false-accusation rate 1.0 | `RESULT_oath_external_corpus_2026_08_27.md`: "The claim is withdrawn"; measured 0.2596, an upper bound |
| `SYNTHESIS_mention_and_use_2026_08_26.md`, instrument #1 | restates that same dead number; the 2026-08-27 addendum, written after the refutation, does not retract it | as above |
| `autopilot/RESULT_oath_v05_precision_2026_07_13.md` | "the zero-false-accusation property … is what makes the certifier deployable against documents it did not author" | a false-accusation rate on foreign text of 0.2596, published as an upper bound, six weeks later |
| `introspection-gate/FINDING_v2_forced_choice_2026_06_06.md` | the flagship "a planted concept is legible to a lens but not to the mind", three-model table | v3, the council map, the README and the SYNTHESIS all carry the "trace, not a held thought" banner; the council map reduces the table to one qualifying model. **v2 carries no banner.** |
| `ai-human-alignment/SYNTHESIS_geometry_of_meaning_2026_06_03.md` | scoreboard marks the brain-match claim SURVIVED | `RESULT_edge_deflation_2026_06_03.md`, same day: a 2014 GloVe-50 model matches the brain equally; unique deep-model contribution +0.05% |
| `grounded-arc/BET0B_PASS.md` | "VERDICT: PASS — shippable signal; arc revived" | `FINDING_crossmodel_2026_05_24.md`: "a low-variance fluke, not signal" |
| `grounded-honesty-axis/THESIS_the_honesty_standard_2026_05_31.md` | "Honesty has a model-strength gradient. (Held …)" | `FINDING_honesty_scaling_law_2026_05_31.md`, same day: FALSIFIED, the apparent scaling was a difficulty confound |
| `showcase-viz/FINDING_mapped_whitening_2026_06_12.md` | "cross-model CONTENT identity still does not transport" | `FINDING_content_wall_2026_06_12.md`, same day, same directory, retracts exactly that. Four neighbouring findings carry banners; this one does not. |
| `frequency-resonance/SYNTHESIS_frequency_adaptation_2026_07_23.md` | ENTRAIN-RICH GREENLIGHT at D=4, named as the next step | `RESULT_scarcity_scale_2026_07_23.md`, same day: worse than static at every budget |
| `mind-instrument/FINDING_gavagai_scale_2026_06_10.md` | "the unsupervised MATCHER is the bottleneck … three gates point at the same wall" | disjoint-worlds b34v3 (0.5918 label-free discovery) and b41 (0.9745) |
| `anchored-validity/FINDING_anchor_threshold_2026_07_23.md` | "roughly 20–30 known-negatives reliably catch a shared blind spot" | `FINDING_anchor_power_instrument_2026_07_23.md`, same day: 15. Direction is conservative, so no overclaim — but the table is stale and unmarked. |
| `deception-correction-gate/FINDING_2026_05_25.md` | "passes every deception-axis bar with no calibration regression" | `decoupled-diagonal-capstone/FINDING_2026_05_25.md`: the composed fix suppressed a genuine lie; integration reverted. No back-pointer exists. |
| `oath-economy/FINDING_model_card_binding_gap_2026_07_02.md` | "the Oath Economy needs a model-card claim extractor … that does not exist yet" | `FINDING_selfbind_2026_07_02.md`, hours later: `audit_grounding` was the right tool all along |
| `first-afference/e1_result.json` | `"verdict": "RESOLVED__winner_selected_and_c5_recomputed"` | `FINDING_e1_not_estimable_2026_08_08.md`: "I do not think that verdict is right". **The prose was corrected; the machine-readable receipt a downstream tool parses was not.** |
| `sycophancy-target-gate/README.md` | asserts both that the v0.2 refit is "not yet in the published instrument" and that 7.5.0 shipped it | internal contradiction, unresolved |
| `calib-poison-general/ZENODO_NEXT_VERSION_DRAFT.md` | stages the coupling constant r\* as shippable | `RESULT_B2_coupling_confirm_VOID_2026_07_16.md`: the coupling question is open and the bound has no number. **Marked DO NOT DEPOSIT, so the risk is prospective — see §6.** |
| `dogfood-self-audit/FINDING_nominal_register_blindspot_2026_08_13.md` | "sixteen instances, of which six in tooling written today — the ratio is the finding" | the same day's `DAY_2026_08_13.md` says eighteen of ten, then twenty; `THESIS_silence_ambiguity` classifies about 23. Three live counts of one ledger. |
| `README.md`, `RESULT_v14_…` | "the path-claim accusation is switched off" / "stays disabled" | commit `5e225b49`, 2026-09-01: "the accusing branch is deleted, not disabled" |

### 5.1 The `SYNTHESIS_connection_of_minds` check, resolved

The specific suspicion was correct in shape and wrong in one detail, and the detail matters.

**The numbers reconcile.** §2 says gemma "reads at exactly chance 0.014"; b31v2 reads it at 0.7857
and b34v3 at 0.5714. All three are true of three different map classes over the same target, and
`b31v2_result.json` carries `M0_linear_top1 = 0.0143` and `M1_mlp_top1 = 0.7857` side by side.
There is no numerical contradiction.

**The scoping is not in the sentence.** The qualifier §2 uses is "label-free map", not "linear
map". The word *linear* does not appear in that paragraph. A reader who stops at the paragraph
break takes away an unqualified "reads at exactly chance".

**§3 does not retract §2 — and it was never going to.** §3 is about the qwen island and never
mentions gemma. The retraction is one paragraph away, in §2's own appended second paragraph, and
is explicit: *"the cliff was the linear map class, not the minds."* What is missing is any inline
marker on the superseded sentence. The header discloses that §3 was added 2026-08-05; it does not
disclose that §2 was also amended after 2026-08-01.

**The certificate does not cover the contradicted sentence.** This is the load-bearing finding and
it is checkable in ten lines. `SYNTHESIS_connection_of_minds_2026_08_01.certificate.json` has 94
ledger entries over lines `[3, 23, 25, 26, 32, 33, 35, …]`. **There is no entry of any status for
line 27** — the line carrying "reads at exactly chance 0.014". The number exists in the receipt
bundle (`synthesis_minds_addendum.json:read_across_minds.gemma_top1`), so this is a coverage miss,
not a grounding failure. The document is nonetheless `OATH-HELD`, VERIFIED 81 / ABSTAIN 13 /
UNGROUNDED 0, and `SEALED`.

**Two binder observations from the same certificate.** Line 32's `392` — "an MLP on 392 true pairs
reads gemma at 0.7857 (`b31v2_result.json`)" — is bound to `b41_result.json:n_anchor_rows`, not to
the receipt the sentence names. Line 33's `70` held-out concepts is bound to
`verifier_7b_result.json:not_gated.coverage_curve[1].n`. So on this document `OATH-HELD` /
`0 UNGROUNDED` means *every audited numeral appears somewhere in the receipt bundle*, not *every
claim is supported by the receipt it cites*. That is a narrower guarantee than the phrase suggests,
and it is the same defect class as M1 one level up: value-match co-occurring with support is not
support.

**The unmarked sentence propagates.** It ships verbatim in `papers/arxiv/connection-of-minds/main.tex`
line 53 and in the submission copy at the same line.

### 5.2 The shape to watch for

Claims that carry their own erratum are safe — §4 is full of them. The dangerous shape is
narrower and it is what every Tier-2 row above has in common:

> **A claim killed by a document that names it, in a file that does not name the killer.**

Every one of these is a missing back-pointer, not a missing honesty. The killer document almost
always exists, is almost always correct, and is almost always found the same day. The reader who
arrives by filename — which is how anyone arrives in a 1,135-file corpus — gets the killed version
with no signal.

---

## 6. PUBLISHED RISK — every deposit against the errata

**The bottom line first: no DOI'd record is known to assert something this lab has since refuted,
with one open item and one disclosed inconsistency. That is a good result and it is stated plainly
because it is a good result.**

**What the record does right.** Corrections have twice been paid for in the same currency as the
claim: the frame-locality circularity was deposited as an erratum DOI (10.5281/zenodo.21679805)
and then as a corrected edition (21693636), and the read≠write line was re-versioned through v26 →
v27 → v28 with `ERRATUM_v26_adaptive_claim.md` instructing readers to cite v26 *with* the erratum.
`threshold-law-2026-05-18.md` (10.5281/zenodo.20278945) carries its own kill inside its abstract.
Two claims that were killed before deposit — calib-poison's coupling constant and anchored-validity's
reserved record 10.5281/zenodo.21520429 — are staged and marked as gated rather than published.
Catching a claim before it becomes permanent is the cheapest possible version of this and it
happened twice.

**The one open item.** `CITATION.cff` names 10.5281/zenodo.19777921 — *Every Mind Leaves Vitals* —
as the preferred citation for this repository. The repo copy of that essay carries a scope erratum
dated 2026-06-21 bounding or falsifying two of its central claims: no substrate-independent
"property of cognition" was established, and cross-family transfer is corpus-overlap-bound
(τ ≈ 0.31, min Anthropic 0.617 below a 0.70 floor). **Whether the deposited record carries that
erratum is UNCHECKABLE from this repository.** What is checkable: `scripts/zenodo_version_emlv.py`
was last modified 2026-04-26, eight weeks before the erratum, and no deposit artifact dated after
2026-06-21 mentions that DOI. The asymmetry is the finding — this lab demonstrably knows how to
re-version a corrected paper and has in-repo evidence of doing so twice, and has no in-repo
evidence of doing it here. `CITATION.cff` has not been touched since 2026-04-27.

**The disclosed inconsistency, still standing.** Tool-call-drift AUC is inconsistent across
permanent records — 0.916 in 19777921 versus 0.943 in the spec and software records. The
2026-05-17 permanent-record review named it, called it uneditable, and recorded it for honesty
(`CHANGELOG.md`, `papers/styxx-status-consolidation-2026-05-17.md`). It remains correct to
describe it as disclosed rather than as a risk.

**The gap in the process.** That 2026-05-17 review is the only whole-corpus permanent-record review
in the repository. It predates the construct ceiling (2026-05-31), the EMLV scope erratum
(2026-06-21), the mention-and-use synthesis (2026-08-26), the external gate failure (2026-08-31)
and V14 (2026-09-01). Its conclusion — the deposits rest on hallucination, refusal, tool-drift and
the phase-transition structure, not on the axes that broke — is still plausible and was not re-run.
Re-running it is cheap and is the first item in §8.

**Third-party records.** Four of the 34 DOIs are records this lab consumes rather than deposits
(10.5281/zenodo.16919272 = the AIDev corpus; two MedSci records; one dataset). They carry no risk
here and are listed so a future census does not count them as ours.

---

## 7. BUILT AND NEVER USED

Ranked by (real capability) × (shortness of the path to someone using it). "It would make a nice
paper" is not a path; each row names the specific missing piece.

| rank | thing | why it is real | the missing piece |
|---|---|---|---|
| 1 | **`styxx.islands`** — hand it any cohort over a shared item set (activations, fMRI betas, MEG epochs) and it reports islands, the cliff, and whether a low-rank correction rescues them | CPU-only, four seconds, nine sealed acts, replication instructions already written (`disjoint-worlds/REPLICATE_legibility.md`) | **One external cohort, run by someone outside this lab.** The `REPLICATIONS.md` ledger is empty and this is the easiest row on it. The missing piece is an email, not an experiment. |
| 2 | **`papers/frequency-resonance/resonance_profiler.py`** | the phase-clamp result behind it is the program's cleanest causal ablation (+0.3122) | **It is not in `styxx/`.** `INSTRUMENT_resonance_profiler_2026_07_23.md` calls shipping it "a future step". The missing piece is a module move and an entry point. |
| 3 | **`styxx.undeclared`** — two-artifact reconciliation, worklog × diff, bands ATTRIBUTED / UNATTRIBUTED, no verdict | it is the designed successor to the retired path-claim accuser and it asserts nothing it cannot bind | **A measurement.** It shipped 2026-08-31 and has never been scored against a blind panel on external data. Until it is, calling it the successor is a design claim, not a result. See §8's warning about verdict-free instruments. |
| 4 | **`styxx/residual_probe/`** — pre-output refusal prediction, median AUC 0.832 across five architectures | preregistered, permutation-controlled, five substrates | **A calibrated operating point.** The result explicitly says "AUC is the claim; the fixed 0.5 threshold is not", and an uncalibrated gate is not deployable. |
| 5 | **`styxx.anchors`** — anchor power, blindspot power, minimum anchors for 0.90 power | the gold-vs-ladder result is clean and generalises past this lab's own subject matter | **A second domain.** It has only ever been run on TruthfulQA-shaped judging. One non-lab judging task would make it a general tool. |
| 6 | **`cooperative-agent-regime`'s topic-control 2×2** | the code is committed (`scripts/topic_control_corpus.py`, `scripts/topic_control_analysis.py`, `tests/test_topic_control_parity.py`) and the arc's own threats doc rates this control "HIGHEST, PENDING" | **A signature.** The preregistration is still `DRAFT — pending sign-off`; nothing may run until it is signed. Until then the +0.4563 drift-axis positive is not a cooperation signal by the arc's own standard. |
| 7 | **`gpai-scorecard`** — 18 flagship docs pinned across 24 signatories, two blind extraction passes in exact agreement on 85 claims | the hardest part (fetching and pinning a moving external corpus) is done | **Model credits.** `PROGRAM_BACKLOG.md` records the block as a resource block. It is the only item here whose missing piece is money. |
| 8 | **`styxx/three_axis/`** — 657 lines, env-gated behind `STYXX_THREE_AXIS=1` | a locked seven-hypothesis protocol with a stopping rule | **Data, and a decision.** Three of its six modules (`meta_rate.py`, `forced_decode.py`, `paraphrase.py`) are imported by nothing anywhere in the repository. This is the strongest retire-or-run candidate in the tree. |
| 9 | **`styxx/apparatus.py`** (65 lines) and the two `*_QUARANTINED.py.txt` files | — | Nothing to build. `apparatus.py` is imported by no code path. The quarantined pair are retained on purpose with their audit verdicts in their own docstrings, which is correct and should stay. |

**A category this audit did not expect to need: documented-but-unwired.** Two scans were run — one
over code only, one over the whole repository including `papers/`, `docs/`, `arxiv/` and
`CHANGELOG.md`. They disagree, and the disagreement is the finding. `three_axis/meta_rate.py`
(131), `three_axis/forced_decode.py` (86), `three_axis/paraphrase.py` (120) and `apparatus.py`
(65) — **403 lines** — are imported by nothing anywhere in the tree, yet all four are described in
papers, protocols or the changelog as though they exist as working parts. A reader of
`papers/three-axis-sendtime-gate/PROTOCOL.md` or of `docs/governance/OPEN_CORE.md` would have no
way to tell. This is M1 applied to the codebase: *a module being named is not a module being used.*

Caveat on rows 8 and 9: dead-module detection here is a name-reference scan plus an `__init__`
re-export check; dynamic and string-based imports are not resolved. `styxx/adapters/llamaindex.py`
(201 lines) is unimported and undocumented — dead by both scans — but has a declared extra in
`pyproject.toml:136` and is exactly the case an adapter registry would load by provider name. No
such registry was found; recorded as UNCERTAIN rather than dead.

---

## 8. WHAT TO DO NEXT — a recommendation

Not a menu. Four things, in order, with what each costs.

### CLOSE — three arcs, this week

**`three-axis-sendtime-gate`.** Locked 2026-05-21, pre-data ever since, three of six modules
imported by nothing, and its constructs are already shipped elsewhere. Record it as UNRUN with the
reason, and delete or archive the three orphan modules. *Cost: an afternoon. Saves: 657 lines of
tree and a standing implication that a send-time gate exists.*

**`cooperative-agent-regime`.** Either sign the topic-control preregistration and run the 2×2, or
mark the drift-axis positive as unlicensed by the arc's own threats document. The current state —
a deposited POSITIVE whose `preregistration_lock_hash` is the literal string
`"TBD-after-operator-signs"` — is the worst of both. *Cost: one signature, or one paragraph.*

**`white-box-vs-text-map`.** One file, no receipt, no preregistration, no certificate, and an
abstract that asserts a confirmatory replication succeeded while its own status line marks that
cell `[PENDING]`. Both sentences cannot be current. Resolve or retire. *Cost: an hour.*

### KEEP — and pay two small debts

The receipt-integrity cluster (§2.6) is the best-cited work in the program and should be left
alone. Two debts inside it are worth a day each. **Back-pointers**: every Tier-2 row in §5 is a
one-line insertion into the killed document naming the killer. That is roughly seventeen lines of
editing and it removes the entire class of defect §5.2 describes; it needs no new science and no
new instrument. **Machine-readable verdict tokens**: `LEDGER.md` already names this as owed, the
mechanism exists, and 0 of 163 cycles carry one. Until they do, the 62-of-163 negatives ratio
stays a keyword count, and the LEDGER says so honestly on every render.

### ADVANCE — one thing

**Bind receipts by digest, not by filename.** This is already named as owed in
`CORPUS_STATE_2026_08_31.md` and tied to the evidence leg, and it is the general repair for the
defect that broke three documents in two days. Every certificate in this corpus currently binds a
claim to a *path*; any generator re-run can silently invalidate it, and the only detector is CI
noticing a certificate stopped reproducing. This is the highest-value unbuilt thing in the
repository because it is the precondition for every other guarantee here being worth anything to
an outsider. *Cost: bounded — it is a change to how `certify` resolves receipts plus a corpus
re-issue, not a research programme.*

### MEASURE — the one experiment worth preregistering next

The prose-reading hypothesis in §8.1 is currently unsettled in a way that matters for what gets
built. **Point one prose-reading instrument and one artifact-only instrument at the same external
corpus, with the same blind panel and the same frozen floor.** The corpus already exists — AIDev,
71,016 PRs, 10.5281/zenodo.16919272, collected by someone else. `styxx.undeclared` versus the
retired path-claim accuser on the same PRs is a clean head-to-head, and it would simultaneously
discharge §7 row 3. *Cost: one panel, which this lab has now run four times and has a protocol for.*

### 8.1 The prose-reading hypothesis, tested

**Hypothesis as stated:** every instrument this lab built that failed reads human prose; every
instrument that works operates on structured artifacts.

**Direction 1 — prose implies failure. Holds, on a narrower class than stated.** Every instrument
in this program that had to decide *claimhood or register from open-ended prose* and was then
measured against a blind panel or a held-out corpus, failed: diffgate path-claim (0.16 held-out,
0.23 in the wild), the OATH obligation predicate on external text (false-accusation rate 0.2596, published as an upper bound;
0.4933 of *verified* tokens judged to be claims at all), the agent-report claim extractor (0.3333
precision, 0.0336 recall), `styxx.agent_audit`'s `extract_claims` (zero claims from a real
seven-sentence report), the ledger's own refusal classifier (`SHIPPED` counted as a machinery
refusal), the deception NLI (quoted false premise read as asserted), the capstone NLI (a question
read as an assertion), the prompt-opinion detector (1.00 → 0.47), `critique_detector` (saturated
at 0.0000/1.0000 on 18 of 18), the dogfood register gates (nominal register blindspot),
text-only deception (0.498). No exceptions were found.

**Direction 2 — artifact implies success. FALSE.** At least six artifact-only instruments failed
outright, and none of them for a prose reason:
`representational-integrity` (geometry manipulation detector, 0.63/0.67, near chance, explicitly
not shipped); `depth-truth` (AUROC 0.5468, anti-signal out of distribution); `first-afference`
(`styxx/coupling.py` blind to planted coupling at 0.0083 against a 0.10 bar, withdrawn for neural
series; `styxx/power.py` quarantined); `mind-instrument` cross-species (0.0292 against a null p95
of 0.0312); `conscience-mount` B37/B38/B39 (three consecutive VOIDs, no clean establish);
`ancient-question-program`'s synthetic RSA (self-retracted as circular by construction). Add
`pre-output-action-gate`, whose residual probe reads 0.649–0.917 on a menu and 0.530 under native
tool-calling, and `autopilot`'s `check_metrics`, which passed on a specimen chosen because it
would pass.

**A counterexample to direction 1, in the other direction.** LLM-judge equivalence clustering reads
open-ended prose and *beat* the artifact alternative it replaced: cosine@0.70 scored 0.573 where
the judge scored 0.948–1.000 on the same items (`tier3-confident-confabulation/FINDING_corrected_2026_05_25.md`).
A prose-reading component was the fix, not the defect.

**The objections, argued rather than waved away.**

*The artifact instruments shipped later, with more doctrine and less exposure.* This is true and it
is the strongest objection. `styxx.islands`, `styxx.capsule`, `styxx.worklog` and `styxx.undeclared`
all postdate the standing rule adopted 2026-08-06 — *no instrument is announced before an
adversarial pass* — and the prose instruments (sycophancy May, deception May, tier3 May, OATH June)
all predate it. The comparison is confounded with date, with discipline, and with whether anyone
ever pointed a blind panel at the thing. It is not possible, on this corpus, to separate "reads
artifacts" from "was built after we learned to red-team ourselves".

*An instrument that never asserts anything cannot fail — which is not evidence in its favour.*
Also true, and sharper than it looks, because this lab has already named the defect. The worklog
is explicitly "a record with a different author, and no verdict". `styxx.undeclared` reports bands
and no verdict. A decision expression that cannot come out otherwise is exactly the class
`dogfood-self-audit` measured at a 40.4% dead-term rate (617 of 1,526, `DAY_2026_08_13.md`, correcting its own published 43.3%) and that `SYNTHESIS_mention_and_use`'s
addendum catalogues as the vacuous-pass error — a control run with *no test at all* scoring the
same as the rule under evaluation. Counting a verdict-free instrument as a working instrument
commits that error. On present evidence the artifact-only successors are not yet known to work;
they are known not to have been measured.

*Selection.* The failures are known because someone chose to measure them, and the prose
instruments are the ones that got blind panels. `styxx.islands` has never been run against an
external cohort. Absence of a measurement is not evidence of success, and treating it that way
would be the same asymmetry this program exists to refuse.

**VERDICT: SUPPORTED-WITH-QUALIFICATION**, and the qualification is large enough that the
hypothesis must not be used as a design rule in its stated form. What survives is narrower and
better:

> Instruments that must **locate their own target in open-ended prose** have failed every time
> this lab measured them against readers who did not write that prose. Instruments handed the
> target — whether the target is a claim, a concept correspondence, or an item set — have not been
> shown to survive the same test, because with two exceptions they have not taken it.

That is the handed-target mechanism (M2), and the prose/artifact split is a proxy for it. It also
explains why direction 2 fails: an artifact-only instrument that must find its own target
(`first-afference`'s coupling detector, `mind-instrument`'s unsupervised matcher) fails in the same
way, and an artifact-only instrument handed its target (b31v2's 392 pairs, b41's label-aligned
bridge) posts the program's largest numbers.

**What would settle it.** The head-to-head named above: `styxx.undeclared` and the retired
path-claim accuser, on the same external PRs, same blind panel, same frozen floor, with the
extraction stage measured separately from the adjudication stage. That single design separates
prose-vs-artifact from handed-vs-found, which no measurement in this corpus currently does.

---

## 9. WHAT THIS AUDIT DID NOT DO

This document was produced by an agent reading roughly 200 to 250 of 1,135 markdown files in a few hours,
choosing per arc the SYNTHESIS, CAPSTONE, RESULT, FINDING, README and erratum files by filename
date and reading late-dated files in preference to early ones. Nine parallel readers covered the
47 arcs; their coverage per arc ranged from complete (the two-file arcs) to about 8% (grounded-honesty-axis,
16 of 208 files; closed-model-frontier, ~24 of 111). No `*.py` harness was executed except the
hash re-checks in §3 item 10. Confidence by section:

- **§1 inventory — HIGH.** Every count is a shell command printed next to it and re-runnable.
- **§2 the idea count — MEDIUM, and it is a judgement.** The citation-graph statistics are
  mechanical and HIGH. The clustering into 13 questions is not; §2's sensitivity paragraph gives
  the range under two other rules, and the harsher reading is available on the same receipts.
- **§3 what is settled — MEDIUM.** Every entry is quoted from the arc's own terminal document, not
  re-derived from its receipt. A quoted number that a subagent read at second hand through a
  synthesis rather than from the primary receipt is possible in items 2 and 5.
- **§4 retractions — HIGH.** Errata are findable by grep and each named file was read.
- **§5 staleness — HIGH on existence, MEDIUM on ranking.** That each stale claim exists and that
  each killer exists was verified by reading both documents. The Tier-1/Tier-2 ordering is a
  judgement about how outward-facing a file is. §5.1's certificate finding was verified twice
  independently, once by parsing the ledger's line set directly.
- **§6 published risk — MEDIUM, with one UNCHECKABLE stated as such.** Whether any Zenodo record's
  current content matches its repo copy cannot be determined from this repository. What was checked
  is the in-repo evidence of re-deposit, which is a proxy.
- **§7 built and never used — MEDIUM.** Dead-module detection is a name-reference scan; dynamic
  imports are unresolved and are flagged where suspected.
- **§8 the recommendation — a judgement, offered as one.**

**What was not looked at at all:** `benchmarks/`, `bench/`, `demo/`, `telescope/`, `web/`,
`integrations/`, `packages/`, `hooks/`, the 172 test files beyond the `xfail` grep, `CHANGELOG.md`
beyond three targeted sections, all `*.jsonl` cycle logs, and 197 of the 213 certificates. Any of
those could contain a stale outward-facing claim this audit did not find, and §5 should be read as
a lower bound rather than a census.

**Where a confident wrong summary is most likely.** `grounded-honesty-axis` (208 files, 16 read)
and `frequency-resonance` (69 files, 16 read) are the two arcs where the terminal state was
inferred from a small sample of late files. `white-box-vs-text-map`, `three-axis-sendtime-gate`,
`gpai-scorecard` and `cooperative-agent-regime` are recorded UNCLEAR because this audit could not
establish their terminal state, and that is the honest verdict rather than a placeholder.

---

*The program's problem is not that it lacks findings. It is that a claim and the document that
kills it can live in the same repository, made the same day, without either one knowing about the
other — and that a reader arriving by filename gets whichever one they happened to open.*
