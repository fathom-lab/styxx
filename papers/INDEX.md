# INDEX — the program map

Fathom Lab · 2026-09-01 · **Navigation, not a claim.** No preregistration covers this file and it
carries no finding. It exists so that somebody with a new idea can answer *have we already done
this?* without reading 1,135 markdown files. Every row is a pointer; the arc's own documents are
the authority, and where this index and an arc disagree, the arc wins.

Companion: `AUDIT_the_whole_program_2026_09_01.md`, which states what this map implies and what
its own limits are. Read the AUDIT's coverage disclosure before quoting anything here.

**Status vocabulary.** LIVE = the terminal claim stands as written. SUPERSEDED = a later document
in this repository replaced the headline. RETRACTED = the arc withdrew its own claim. NEGATIVE-RESULT
= the arc's preregistered bet failed and the failure is the output. UNCLEAR = this index could not
establish the terminal state; the "would need to read" column in the AUDIT names what is missing.

**Headlines are recorded as they terminally stand.** Where an arc's opening hope was falsified by
its own later work, the falsified version is what appears below.

---

## 1. THE 49 ARCS (47 at the 2026-09-01 audit; `sworn` opened later that day; `charon` on 2026-09-02)

| arc | terminal RESULT, one sentence | status | core-idea tags | ships in | DOI / OSF |
|---|---|---|---|---|---|
| adversarial-robustness | Divergence detectors survive instruction and persona attacks but go blind under context injection; the blanket claim was later replaced by a three-way calibrated map. | SUPERSEDED | handed-target, refusal-as-verdict, calibration, preregistration | `styxx/divergence.py` | NONE |
| agent-conscience | A frontier model abandons 0.5348 of correct free-text answers under one content-free challenge; the hope that only the report caves was retracted, then measured false in context. | SUPERSEDED | sycophancy-pressure, knowledge-boundary, self-verification, preregistration, calibration | `styxx/framelocality.py`, `styxx/knowsay.py` | 10.5281/zenodo.19326174 chain (21679805, 21693636, 21695691) |
| agent-self-audit | A repo-grounded auditor passes curated claim lists (13/13, 33/35, 18/18) but extracts zero claims from a real seven-sentence agent report and false-failed a true version claim. | SUPERSEDED | agent-provenance, extraction-vs-adjudication, receipt-integrity, self-verification, mention-vs-use | `styxx/agent_audit.py`, `styxx/critique.py` | NONE |
| ai-human-alignment | LLM concept geometry tracks human brain geometry, but a 2014 GloVe-50 model predicts the brain equally well and the deep model's unique contribution is +0.05%. | LIVE (self-deflated) | representational-geometry, cross-model-transfer, benchmark-construct, preregistration | `styxx.meaning_integrity` | NONE (third-party OSF datasets only) |
| anchored-validity | Gold anchors license nothing on a real judge panel (0/15 coverage in every family); same-distribution ladder anchors restore coverage or force a refusal. | LIVE | handed-target, benchmark-construct, refusal-as-verdict, calibration, extraction-vs-adjudication | `styxx/anchors.py` | 10.5281/zenodo.21520429 RESERVED, not published |
| ancient-question-program | Minds converge on what they represent, not on how they compute it; the "universal forms vindicated" reading was self-retracted as circular by construction. | SUPERSEDED | representational-geometry, receipt-integrity, preregistration, handed-target | `styxx/certify.py` | NONE |
| auditor-ceiling | Mechanical QA grading false-accuses correct answers at 0.126 on TriviaQA; the PopQA figure fell from 0.112 to 0.031 after a void and a referent-ambiguity separation. | SUPERSEDED (v2) | handed-target, benchmark-construct, extraction-vs-adjudication, calibration | NONE | NONE |
| autopilot | Preregistered gates kept passing on specimens chosen to pass; 31 of 32 preregs carry undeclared gates and cycle-18 certificates catch a doctored digit 0.216 of the time. | SUPERSEDED | preregistration, receipt-integrity, self-verification, benchmark-construct | `styxx/protocol.py`, `styxx/certify.py` | NONE |
| benchmark-validation | `semantic_entropy` matches the published TriviaQA band (0.785) but loses to mean-logprob (0.817); its niche is providers that expose no logprobs. | LIVE (bounded) | benchmark-construct, calibration, preregistration | `styxx.semantic_entropy` | NONE |
| calib-poison-general | Poisoning severity grades by probe robustness, but the "private-calibration defense" is roughly two-thirds probe capacity rather than privacy, and the coupling constant was VOIDed. | NEGATIVE-RESULT | handed-target, calibration, representational-geometry, receipt-integrity | `styxx/mount.py` (`relock`), `styxx.ladder` | cites 10.5281/zenodo.21263158; next version is a gated draft |
| closed-model-frontier | After two preregistered repair cycles the agent-prose path-claim accuser scored 0.16 held-out precision against a 0.95 floor; the accusing class is retired, not retried. | NEGATIVE-RESULT | mention-vs-use, extraction-vs-adjudication, receipt-integrity, agent-provenance, preregistration | `styxx/certify.py`, `styxx/diffgate.py`, `styxx/capsule.py`, `styxx/evidence.py` | NONE of its own (consumes 10.5281/zenodo.16919272) |
| concept-dynamics | Current models are arrhythmic: oscillation capacity is exactly complex-valued recurrence, which transformers structurally lack. | LIVE (bounded negative) | oscillation-dynamics, representational-geometry, preregistration | NONE | NONE |
| conscience-mount | A borrowed read-only truth axis catches 0.85 of pressured caves at 0.20 realized FPR, buys no adversarial robustness, and never reached a clean preregistered establish. | NEGATIVE-RESULT | handed-target, representational-geometry, cross-model-transfer, policy-gating, refusal-as-verdict | `styxx/mount.py`, `styxx/crossmind.py` | NONE |
| consensus-hallucination | Seven preregistered methods failed to see or crack shared cultural-prior errors, and the curated folklore corpus itself collapsed. | NEGATIVE-RESULT | handed-target, benchmark-construct, preregistration | benchmark JSON only; `classify_dark_core` NOT shipped | NONE |
| consensus-truth-engine | Cross-vendor consensus does not out-answer its best member (−0.051) but its fracture is a calibrated abstention signal (gap 0.253). | NEGATIVE-RESULT (abstention half LIVE) | cross-model-transfer, calibration, refusal-as-verdict, preregistration | `styxx.consensus` | NONE |
| consistency-robustness | The arc's own "consistency beats signatures" thesis was retracted the same day: the activation probe survived every feasible attack while the consistency check cracked at 0.183. | NEGATIVE-RESULT | deception-honesty, refusal-as-verdict, calibration, preregistration | `styxx/honesty.py`, `styxx/probe.py`, `styxx/anchors.py` | NONE |
| cooperative-agent-regime | Phase coherence between agent dyads closed negative (0.111); the replacement drift-axis positive (+0.4563) is still an uncontrolled topic-convergence proxy because its load-bearing 2×2 control was never signed or run. | UNCLEAR | oscillation-dynamics, agent-provenance, preregistration, benchmark-construct | `styxx/coherence.py` | NONE |
| council-reference-free-truth | Inter-model agreement separates real from fabricated reference-free and survives the fame test, but cannot certify truth past the verifiable≈documented≈known confound. | SUPERSEDED | mention-vs-use, cross-model-transfer, calibration, self-verification | `styxx.council_agreement` | NONE |
| cross-vendor-council | Agreement across three vendors tracks truth (0.917) and beats the same-vendor council because vendors do not share each other's fabrications. | LIVE | cross-model-transfer, calibration, benchmark-construct, preregistration | `styxx.council_agreement` | NONE |
| crossmind-instrument | A frozen invariant contract for the cross-model value-axis reader, hard-wired to refuse steering — a preregistration for an instrument, not a scientific claim. | LIVE | refusal-as-verdict, cross-model-transfer, policy-gating, receipt-integrity | `styxx/crossmind.py` | NONE |
| deception-correction-gate | A three-signal prompt-aware deception suppressor stopped flagging honest corrections of false premises but failed an orthogonal sycophancy bar, so nothing shipped. | NEGATIVE-RESULT | mention-vs-use, deception-honesty, policy-gating, preregistration | NONE (staged) | NONE |
| decoupled-diagonal-capstone | Both fixes passed the joint preregistered kill-gate, then the mandatory full-suite guard caught the deception fix suppressing a genuine lie; the integration was reverted. | NEGATIVE-RESULT | mention-vs-use, deception-honesty, receipt-integrity, self-verification | NONE (reverted) | NONE |
| depth-truth | SAE circuit-attribution depth does not predict answer correctness (AUROC 0.5468) — null solo, null additive over semantic entropy, and actively anti-signal out of distribution. | NEGATIVE-RESULT | handed-target, knowledge-boundary, representational-geometry, preregistration | NONE | NONE |
| disjoint-worlds | Cross-family content reading is real and label-free-discoverable for one target pair, but the island topology it produced does not survive a ten-model cohort. | LIVE (sub-results SUPERSEDED) | handed-target, cross-model-transfer, representational-geometry, preregistration, receipt-integrity | `styxx/islands.py`, `styxx/sentinel.py` | NONE (third-party OSF datasets) |
| dogfood-self-audit | Twenty instances of one defect class in a day — instruments overstating their own resolution invisibly — ten of them built by the people who had already published the class. | LIVE | mention-vs-use, receipt-integrity, self-verification, extraction-vs-adjudication, agent-provenance | `styxx/claim_audit.py`, `styxx/conscience.py` | NONE |
| first-afference | The physical-sense channel was never measured: the coupling instrument failed its own exams, was withdrawn for neural series, and the one scored cohort returned a bounded null. | NEGATIVE-RESULT | refusal-as-verdict, preregistration, self-verification, calibration, handed-target | `styxx/protocol.py`; `styxx/coupling.py` withdrawn; `styxx/power_QUARANTINED.py.txt` | NONE |
| frequency-resonance | The single-knob phase clamp shows oscillation is causally load-bearing in SSMs (+0.3122 on permuted MNIST); every proposed extension — adaptive frequency, nested coupling, real-model transfer — died. | LIVE (extensions NEGATIVE) | oscillation-dynamics, handed-target, preregistration, cross-model-transfer | `styxx/resonance.py` (the profiler, shipped 2026-09-02) | NONE |
| generative-diversity | The confabulation detector run in reverse is a valid coherence-gated diversity index, and it found no diversity difference between models. | NEGATIVE-RESULT | benchmark-construct, calibration, preregistration | NONE | NONE |
| gpai-scorecard | The receipt-binding scorecard was frozen, fetched, and never scored — the preregistration has no answer anywhere in this repository. | UNCLEAR (prereg-only) | preregistration, receipt-integrity, policy-gating, extraction-vs-adjudication, mention-vs-use | `styxx.certify` as tooling only | NONE |
| grounded-arc | The universal per-call validity oracle is preregistration-killed on both substrates; what survives is a narrow logprob flag for refusal over-flagging that attenuates cross-family. | NEGATIVE-RESULT | calibration, refusal-as-verdict, cross-model-transfer, preregistration, knowledge-boundary | NONE shipped (`styxx/preflight.py` patch STAGED) | cites 10.5281/zenodo.20278945 |
| grounded-honesty-axis | Resampling a **handed** factual self-claim against the model's own belief separates true from false at 0.966 where text-only axes sit at chance (0.4983); every white-box read is inert. | LIVE | handed-target, deception-honesty, calibration, knowledge-boundary, representational-geometry | `styxx/divergence.py`, `styxx/single_pass.py`, `styxx/confound_audit.py` | cites 10.5281/zenodo.19777921 |
| introspection-gate | A model cannot forced-choose its own injected concept while a probe reads it at 1.00 — but that read certifies an injected trace, not a held thought, and the steering-validated version survives on one model. | SUPERSEDED | introspection, mention-vs-use, representational-geometry, handed-target, self-verification | `styxx/steer.py` | NONE |
| knowledge-boundary-calibration | Epistemic humility is a prompt effect, not a model trait — one clause moves abstention from 0 to 0.97 — so the knowledge boundary is a model property only under a pinned neutral prompt. | LIVE | mention-vs-use, knowledge-boundary, calibration, refusal-as-verdict, benchmark-construct | NONE | NONE |
| mind-instrument | Structure-only concept transmission between independently built models works at roughly 14× chance but is species-bound and matcher-limited. | SUPERSEDED | handed-target, cross-model-transfer, representational-geometry, preregistration | `styxx/mind.py` | NONE |
| oath-economy | A frontier model card binds 0 of 201 numeric eval claims to receipts, yet a third party re-binds 18 of 22 exactly — the gap is mostly a missing hyperlink over a structural residue. | LIVE (stalled) | mention-vs-use, receipt-integrity, benchmark-construct, extraction-vs-adjudication | `styxx.audit_grounding`, `styxx/claim_audit.py` | NONE |
| pre-output-action-gate | An open-weight model's pre-emission residual linearly predicts and causally steers its impending destructive tool call, but the edge collapses to 0.530 under native tool-calling. | LIVE (mixed board) | policy-gating, representational-geometry, cross-model-transfer, preregistration | NONE shipped (research code only) | NONE |
| pre-output-gate | A native residual probe predicts a model's own refusal decision before any output token across five open-weight architectures (median AUC 0.832); the operating point is uncalibrated. | LIVE | refusal-as-verdict, policy-gating, representational-geometry, calibration | `styxx/residual_probe/` | NONE |
| read-neq-write | Apparent evasion of the honesty probe was calibration poisoning, not erasure; the published "survives an adaptive attacker" claim was retracted and adaptive robustness stays open. | LIVE (sub-claim RETRACTED) | calibration, representational-geometry, receipt-integrity, refusal-as-verdict, handed-target | `styxx/mount.py`, `styxx/admissibility.py` | 10.5281/zenodo.21241185 (v26) → 21250272 (v27) → 21263158 |
| real-convergence | Real models share concept geometry above chance (0.258 cross-family) but heterogeneously, and the predicted scale law failed with the opposite sign (ρ −0.573). | LIVE | representational-geometry, cross-model-transfer, preregistration, calibration | NONE | NONE |
| representational-convergence | Cross-family representational convergence is concept-specific: refusal converges (0.700), corrigibility not at all (−0.006), deception untestable on four of six probes. | LIVE (bounded) | representational-geometry, cross-model-transfer, benchmark-construct, preregistration | UNCLEAR | NONE |
| representational-integrity | The white-box geometry manipulation detector is dead — it detects meta or behavioural instruction, not malice (0.63/0.67, near chance). | NEGATIVE-RESULT | representational-geometry, policy-gating, benchmark-construct, preregistration | NONE (explicitly not shipped) | NONE |
| rhythm-rescue | Clamping eigenvalue phase halves an LRU's ordered-memory capacity (6.0 → 2.67), establishing oscillation as real but non-necessary and attention-dominated. | SUPERSEDED | oscillation-dynamics, handed-target, preregistration, benchmark-construct | NONE | NONE |
| showcase-viz | Cross-model content identity does transport through a label-free linear map (0.90 raw top-1); the arc's own "value thermometer, not a content transcript" wall was two stacked artifacts and is retracted. | SUPERSEDED | cross-model-transfer, representational-geometry, deception-honesty, calibration | `styxx/crossmind.py` (with owed changes) | NONE |
| charon | The lab's record over its three verdict-bearing formats — 213 certificates, 18 sworn documents, 12 capsules — re-derived from bytes into one chained log a stranger re-runs with one command; the receipt set sits on every line so an OATH verdict bought with volume is visible, and a moved verifier reads as SKEW rather than as drift; an adversarial pass found four sentences the instrument published about itself to be false, and the chain is now stated to bind order to a head pinned outside the log. | LIVE | receipt-integrity, agent-provenance, preregistration | `styxx/charon.py` | NONE |
| sworn | The author binds a sentence to bytes it could not have written and the verifier is never handed a target; v0.2 survived a twelve-attack battery at the price of four rules and withdrew its coverage estimate (its denominator was a diff-claim detector fired on result-shaped prose); sixteen documents in the tree are sworn, and no measurement exists of whether authors bind the sentences that matter. | LIVE (unmeasured) | handed-target, receipt-integrity, agent-provenance, mention-vs-use, preregistration | `styxx/sworn.py` | NONE |
| sycophancy-target-gate | A grammatical self-versus-other attachment gate fixed self-apology false positives and shipped; every lexical route to the restrained-technical false positive is closed-negative because opinion-versus-fact is irreducibly semantic. | LIVE | mention-vs-use, handed-target, sycophancy-pressure, calibration, policy-gating | `styxx/guardrail/self_directed_gate.py` | cites 10.5281/zenodo.19777921 |
| three-axis-sendtime-gate | A locked seven-hypothesis send-time protocol, pre-data: nothing scored, stopping rule never fired. | UNCLEAR (pre-data) | preregistration, introspection, self-verification, oscillation-dynamics | `styxx/three_axis/` (env-gated; 3 of 6 modules unimported) | NONE |
| tier3-confident-confabulation | Confident confabulation is inconsistent, not stable — the published "AUC 0.55, the model tells the same lie every time" headline was a clustering-threshold artifact of this lab's own probe. | RETRACTED (original) / LIVE (corrected) | mention-vs-use, receipt-integrity, self-verification, calibration | `styxx.semantic_entropy` | NONE |
| white-box-vs-text-map | White-box probes beat text monitors only where the signal is representational and the interface clean; the edge is interface-fragile and absent on closed models. | UNCLEAR | policy-gating, benchmark-construct, other (weakest-receipt arc in the corpus) | UNCLEAR | NONE |

---

## 2. IDEA INDEX — the lookup that prevents re-invention

Each block is one idea, with every arc that touches it in date order. **CITED** means the later arc
names the earlier one by filename. **RE-DERIVED** means it does not. Where a row says RE-DERIVED,
the second arc built the idea again.

### 2.1 mention-vs-use — a marker that co-occurs with a class is not the class

| date | arc / file | the local name for it | citation |
|---|---|---|---|
| 2026-05-24 | sycophancy-target-gate/`FINDING_promptopinion_2026_05_24.md` | prompt-opinion vs model-opinion | — |
| 2026-05-24 | sycophancy-target-gate/`FINDING_2026_05_24.md` | self-vs-other target attachment | — |
| 2026-05-25 | tier3-confident-confabulation/`FINDING_corrected_2026_05_25.md` | form impersonating meaning (cosine clustering) | — |
| 2026-05-25 | knowledge-boundary-calibration/`SYNTHESIS_behavioral_knowledge_boundary_2026_05_25.md` | **"The recurring adversary: FORM impersonating MEANING"** — names three instances inside the instruments | — |
| 2026-05-25 | council-reference-free-truth/`SELF_AUDIT_2026_05_25.md` | surface similarity standing in for semantic identity | — |
| 2026-05-25 | deception-correction-gate/`FINDING_2026_05_25.md` | NLI reads a quoted false premise as asserted | — |
| 2026-05-25 | decoupled-diagonal-capstone/`FINDING_2026_05_25.md` | NLI reads a question as an assertion | — |
| 2026-06-06 | introspection-gate/`FINDING_v2_forced_choice_2026_06_06.md` | trace vs held thought (the representational analogue) | — |
| 2026-07-02 | oath-economy/`FINDING_model_card_binding_gap_2026_07_02.md` | NAMED-ONLY vs BOUND | — |
| 2026-08-07 | autopilot/`FINDING_protocol_power_basis_invalid_2026_08_07.md` | declared vs verified | — |
| 2026-08-13 | dogfood-self-audit/`FINDING_nominal_register_blindspot_2026_08_13.md` | nominal register; BACKED / AMBIENT_ONLY / UNBACKED | — |
| 2026-08-26 | `SYNTHESIS_mention_and_use_2026_08_26.md` | **mention versus use** — catalogues ten instances and names the class | **RE-DERIVED: cites none of the eleven rows above** |
| 2026-08-30→09-01 | closed-model-frontier (`RESULT_obligation_predicts_claimhood`, `RESULT_v14_…`) | obligated vs volunteered oath; claimhood as its own predicate | cites the 2026-08-26 synthesis |

The string `mention-vs-use` appears nowhere in this repository before 2026-08-26. The string
`FORM impersonating MEANING` appears exactly once, in the 2026-05-25 file, and is cited by nothing.

### 2.2 handed-target — a number measured where the target was supplied, not found

The program has **no name for this**. Grepping for handed-target language across `papers/` returns
one incidental hit. The closest named relatives are `extraction-vs-adjudication` (named 2026-08-30)
and anchored-validity's *gold anchors license nothing* (2026-07-21).

| date | arc | handed | found | citation |
|---|---|---|---|---|
| 2026-05-24 | sycophancy-target-gate | template holdout 1.00 | varied phrasing 0.47 | — |
| 2026-05-28 | grounded-honesty-axis | `grounded_honesty(samples, claim)` — claim supplied — 0.966 | text-only deception 0.4983 | — |
| 2026-06-05 | ancient-question-program | RSA using the known concept correspondence | (downgraded by `CORRECTIONS_2026_06_03.md`) | self-marked |
| 2026-06-10 | mind-instrument | oracle top-1 0.8322, true alignment given | unsupervised 0.1441 | — |
| 2026-07-04 | auditor-ceiling | 0.126 estimated over rows the mechanical key had already marked wrong | complementary miss rate unmeasured | — |
| 2026-07-21 | anchored-validity | gold anchors: 0/15 coverage | ladder anchors 13/13 | **names the law** |
| 2026-08-01 | disjoint-worlds b31v2 | 392 true pairs → 0.7857 | label-free discovery → 0.5714 (b34v3) | both reported |
| 2026-08-05 | disjoint-worlds b41 | label-aligned surgery → 0.9745 | — | self-disclosed |
| 2026-08-07 | autopilot | G3 measured on a specimen chosen to pass | corpus-wide re-measure fails on nine results | same-day erratum |
| 2026-08-30 | closed-model-frontier | (curated corpora) | extractor precision 0.3333, corpus recall 0.0336 | — |
| 2026-09-01 | closed-model-frontier v14 | (internal corpora) | held-out precision 0.16 | — |

**anchored-validity states the general law and no later arc cites it.** `papers/closed-model-frontier/*.md`
contains no reference to `anchored-validity` or to `auditor-ceiling`.

### 2.3 refusal-as-verdict — an instrument that cannot refuse cannot be trusted

anchored-validity (VOID_PANEL) · read-neq-write (VOID_NO_BITE) · first-afference (protocol v4) ·
crossmind-instrument (`steer` raises) · conscience-mount (`steer` refused by construction) ·
grounded-arc · consensus-truth-engine (abstention) · knowsay (returns None with the failing floor
named) · OATH (ABSTAIN / UNCHECKABLE) · diffgate (UNCHECKABLE, 2026-08-31) · `styxx.islands`
(refuses below eight members) · `styxx.sense` (`COUPLED_BEYOND_CONFOUND__attribution_pending`).
**RE-DERIVED at least twice:** read-neq-write's bite gate and anchored-validity's void panel are the
same primitive, invented independently, with no cross-reference in either direction.

### 2.4 cross-model representational reading, and its barrier

representational-convergence (06-02) → real-convergence (06-03) → ancient-question-program (06-05)
→ disjoint-worlds (06-06 →) → mind-instrument (06-10) → showcase-viz (06-10) → introspection-gate
(06-06) → crossmind-instrument (06-12) → conscience-mount (06-12) → ai-human-alignment (06-30) →
disjoint-worlds b31v2..b50 (08-01 → 08-08).
**Two RE-DERIVATIONS inside this line.** (a) showcase-viz `FINDING_content_wall_2026_06_12.md`
finds "the wall is in the whitening metric, not the channel"; disjoint-worlds `FINDING_b31v2_door_opens_2026_08_01.md`
finds "the cliff was the linear map class, not the minds" — the same shape, seven weeks apart, and
`papers/disjoint-worlds/*.md` never names showcase-viz. (b) mind-instrument `FINDING_gavagai_scale_2026_06_10.md`
declares "the unsupervised MATCHER is the bottleneck"; disjoint-worlds b34v3/b41 solve exactly that
open problem and cite neither `gavagai_scale` nor `matcher_v1`.

### 2.5 belief vs report under pressure

sycophancy-target-gate (05-24) · deception-correction-gate (05-25) · decoupled-diagonal-capstone
(05-25) · grounded-honesty-axis intent sub-arc (05-31) · conscience-mount (06-12) · read-neq-write
(07-04) · calib-poison-general (07-09) · agent-conscience (07-24 → 08-14).
Mostly CITED within the later half; the May chain is not cited by the July chain. conscience-mount
B37/B39 and agent-conscience `FINDING_cot_inward_powered_2026_07_30.md` derive the same
"deliberation raises honest discrimination under deference pressure" effect with no cross-reference
in either direction.

### 2.6 receipt integrity — a certificate is only as good as what it binds

autopilot mutant battery (07-03) · oath-economy (07-02) · closed-model-frontier OATH v04→v14
(07-03 → 09-01) · dogfood-self-audit (08-13) · `LEDGER.md` · `REPLICATIONS.md` ·
`CORPUS_STATE_2026_08_31.md` ("a receipt is history too") · `capstone_receipt_drift.json`.
**Best-cited cluster in the program.** The one gap: the 2026-06-10 `capstone_receipt_drift.json`
is the same defect as the 2026-08-31 "a receipt is history too" and is not named as its precedent.

### 2.7 oscillation as a computational mechanism

concept-dynamics (06-02) · rhythm-rescue (06-03) · frequency-resonance (06-04 → 07-24) ·
cooperative-agent-regime phase-coherence arm (05-20). CITED — frequency-resonance names
`rhythm-rescue/run_rhythm_rescue.py`. One gap: `PAPER_oscillation_causal_map_2026_07_23.md`'s
receipt list omits `PAPER_oscillation_memory_2026_06_04.md`, which contains the contrary result.

### 2.8 agreement across models as a reference-free truth proxy

council-reference-free-truth · cross-vendor-council · consensus-truth-engine · consensus-hallucination
· generative-diversity · benchmark-validation · knowledge-boundary-calibration · tier3-confident-confabulation
— **every one of them earliest-dated 2026-05-25**, all over the same 150-item hashed holdout, all using the same
cosine-or-judge clustering step. Eight directories, one night, one machinery.

### 2.9 pre-output gating from internal state

pre-output-gate (06-02) · pre-output-action-gate (06-02) · white-box-vs-text-map (06-02) ·
representational-integrity (06-03) · showcase-viz says-yes-knows-no (06-11) ·
three-axis-sendtime-gate (05-21, pre-data). Sibling arcs cite each other; the May three-axis
protocol shares constructs with the sycophancy arc and cites nothing from it.

### 2.10 construct ceiling — text-only instruments read register, not content

`THESIS_the_honesty_standard_2026_05_31.md` ("a construct ceiling the program had hit four times")
· `CONSTRUCT_CEILING_PUBLIC_RESPONSE_2026_05_29.md` · grounded-arc staged `preflight.py` patch ·
cooperative-agent-regime ("we read the REGISTER rhythm, not the cognitive rhythm") ·
dogfood-self-audit nominal register (08-13) · `every-mind-leaves-vitals.md` scope erratum (06-21).
**This is 2.1 seen from the signal side rather than the token side.** The AUDIT argues both ways
about whether they should be counted as one idea or two.

---

## 3. RECEIPTS INDEX — where each headline number lives

| number | what it is | receipt / document |
|---|---|---|
| 0.16 | diffgate path-claim held-out precision vs a 0.95 floor | `closed-model-frontier/v14_adjudication.json` |
| 0.23 | diffgate precision in the wild, 71,016 external PRs | `closed-model-frontier/RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md` |
| 0.6975 | V14 cumulative recovery vs a 0.6667 bar | `closed-model-frontier/v14_gates.json` |
| 0.3462 | V13 recovery vs the same bar | reported only inside `closed-model-frontier/PREREG_v14_repair_2026_08_31.md` — **V13 has no RESULT file** |
| 0.3333 | agent-report claim-extractor precision, blind panel | `closed-model-frontier/agent_claim_extractor_baseline.json` |
| 0.033621 | that extractor's corpus-level recall (ESTIMATE) | same |
| 0.0204 | claim density of the never-read band, vs a 0.02 floor | same |
| 0.5811 / 0.72 | unobligated-oath rate, internal / external | `closed-model-frontier/oath_unobligated_oath_census.json`, `oath_external_epistemics_census.json` |
| 0.4933 | share of *verified* external tokens a blind panel called claims | `closed-model-frontier/oath_adjudication_result.json` |
| 0.3654 / 0.7826 | claim-share of volunteered / obligated external oaths | `closed-model-frontier/oath_obligation_claimhood_join.json` |
| 0.126 | mechanical QA grading false-accusation rate, TriviaQA | `auditor-ceiling/final_rates.json` |
| 0.216 | cycle-18 certificate catch rate over 269 mutants | `autopilot/cycle18_mutant_battery_result.json` |
| 0.5348837209302325 | frontier free-text cave rate under one challenge | `agent-conscience/frontier_freetext_v9_result.json` |
| −0.2793478260869565 | retained-probe reach margin (caved 0.6956 vs held 0.975) | `agent-conscience/frontier_incontext_oof_result.json` |
| 0.9285714285714286 | knowledge-preserving out-of-frame recovery at 3B | `agent-conscience/scale3b_result.json` |
| 0.966 / 0.4983 | grounded self-claim vs text-only deception AUC | `grounded-honesty-axis/grounded_honesty_result.json` |
| τ ≈ 0.31 | corpus↔domain overlap threshold for label-free transport | `papers/threshold-law-2026-05-18.md`; runs in `scripts/dogfood/out_corpus_coverage_law*.json` |
| 0.617 | minimum Anthropic transported AUC vs a 0.70 floor (kill) | `papers/threshold-law-2026-05-18.md` |
| 0.0143 / 0.7857 / 0.5714 | gemma read: linear map / 392 handed pairs / label-free discovery | `disjoint-worlds/b31v2_result.json`, `b34v3_result.json` |
| 0.9745 | label-aligned bridge discovery, vs 0.0 for a matched random frame | `disjoint-worlds/b41_result.json` |
| 0.832 | median pre-output refusal-probe AUC, five architectures | `pre-output-gate/holdout_gate_result.json` |
| 0.530 | the same class of gate under native tool-calling | `pre-output-action-gate/RESULT_open_toolcall_confirm_2026_06_03.md` |
| 0.5468 | SAE attribution depth → correctness AUROC | `depth-truth/results/verdict.json` |
| +0.3122 | permuted-MNIST phase-clamp ablation | `frequency-resonance/pmnist_ablation_result.json` |
| 0.1441 / 0.8322 | structure-only cross-family transmission: unsupervised, versus the oracle cell with the true alignment supplied | `mind-instrument/FINDING_gavagai_scale_2026_06_10.md` |
| 0/201, 18/22 | model-card claims bound by vendor / re-bound by a third party | `oath-economy/census_v1_results.json`, `rebind_result.json` |
| 0/15, 13/13 | gold vs ladder anchor coverage on a real judge panel | `anchored-validity/PAPER_gold_anchors_license_nothing_2026_07_21.md` |
| 0.838 / 0.722 | clean-calibrated read AUROC vs a ~0.55 random floor | `read-neq-write/e1_result.json` |
| 0.183 | confidently-fooled rate vs a 0.20 bar | `consistency-robustness/grounded_attack_result.json` |
| 0.111 | dyad phase coherence, closed negative | `cooperative-agent-regime/results/phase_coherence_corpus_2026-05-20.json` |
| 163 / 380 / 208 / 34 / 62 / 9 | cycles, preregs, certificates, seals, keyword-matched negatives, `INVALID__*` verdicts | `papers/LEDGER.md` (regenerated by `papers/build_ledger.py`) |
| 208 / 200 / 8 / 1 / 1 / 1 | corpus-audit line: certificates, HELD, FAILED, verdict-drift, incomplete, receipt-changed | `REPLICATIONS.md`, reproduced by `python -m styxx.corpus_audit papers/` |
| 0 | external replications to date | `REPLICATIONS.md` ledger (empty) |
