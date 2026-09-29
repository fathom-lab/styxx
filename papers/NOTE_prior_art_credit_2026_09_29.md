# Note — prior art credited, and the priority sentences that were not earned

2026-09-29. The lab-wide record of one correction pass. It edits no frozen document. For each
sentence it names where the sentence is, who did the thing earlier and when, and the reading that
stands from today. Code and documents that are not frozen were corrected in place on the same branch
and are listed in section 3.

**Where the facts come from.** A landscape synthesis dated 2026-09-28 and its critic pass, kept
outside the repository: one synthesizer, five single-agent sweeps (107 entries) merged with the
lab's earlier surveys, and a critic pass that re-fetched 27 sources through a summarising web
fetcher and the GitHub, Hugging Face and PyPI APIs under the lab's research user agent. No hashed
bytes, no frozen procedure, no human review. The operator approved on 2026-09-29 that the lab
correct its own priority text and credit its neighbours. Nothing was re-fetched for this note, so
every entry below corrects toward less and none adds a claim about styxx. Neighbours' dates are
their own: arXiv v1, release notes, or repository creation per the GitHub API. Swarm Orchestrator's
dates and numbers are self-reported.

**The rule applied.** A sentence that says or implies nobody else did something, that the lab did
it before others, or that it is the only one, is kept only if a survey under a frozen procedure
priced it and nothing read since contradicts it. Otherwise it is withdrawn and, where a neighbour is
known, the neighbour is named with its date.

## 1. The lane notes

| lane | note | what it covers |
|---|---|---|
| diffgate, evidence leg, BENCH-1, DECIDE-1, COMPAT-2, the OATH prior-art survey, the capsule, DECLARE-1, EXTERNAL-1 | `closed-model-frontier/NOTE_prior_art_credit_2026_09_29.md` | the "we know of no one who pointed it at prose" sentence and the evidence-leg conjunction (PR-MCI 2026-01-08, AgentLiar 2026-05-20, Swarm Orchestrator 2026-05-23, backcheck 2026-08-04, DeerFlow 2026-08-03); BENCH-1's "no PR-level benchmark" (PR-MCI's 974 labelled PRs); the OATH survey's negative result (statcheck, PR-MCI, Deterministic Integrity Gates, metacheck); DECLARE-1's credits (NabaOS, DeerFlow, readback, Swarm Orchestrator, commitlint-scope, Agent Trace) |
| sworn | `sworn/NOTE_prior_art_credit_2026_09_29.md` | the second-question sentence (metacheck, 2026-08-16); the surviving sentence's `UNSWORN` clause, occupied on a narrower object by DeerFlow's zero-citation verdict (RFC 2026-08-03, merged 2026-08-29); Inline XBRL (2013), showyourwork! (2021), VeriFin (2026-08-10), Trusty URIs (2014) |
| charon | `charon/NOTE_prior_art_credit_2026_09_29.md` | SKEW versus DRIFT (Proof-Carrying Agent Actions §8.3, 2026-06-02); signed, chained receipts (Microsoft agent-governance-toolkit, 2026-04-27); Swarm Orchestrator's ledger (2026-05-23) and re-derivation (2026-09-02 / 09-07); Agentic Witnessing; Sello; witnessed logs |

## 2. Documents at `papers/` root (frozen; not edited)

**`PLAN_the_next_level_2026_09_02.md`, lines 119-125** (a sworn document). *"We know of no other
format that binds whole sentences of a free-text retrospective report … with a distinct verdict for
a document that bound nothing …"* and *"Each clause is occupied; only the conjunction is not."* The
2026-09-05 sworn survey supplied a qualified successor and was never adopted into the plan. Since
then one clause was found occupied before sworn on a narrower object: DeerFlow renders a report with
action claims and no receipt citations `UNVERIFIED`, "not a clean bill" (RFC 2026-08-03, merged
2026-08-29). **Reading:** at most a residual, pending the re-pricing the sworn note lists.

**`RECON_landscape_2026_08_21.md`, lines 74-84**, *"We hold four things there that no one else
appears to"*: a named class (SILENT-PASS), detectors, a benchmark, and confirmed instances. A
success report that hid a failure was named, measured and detected in CI research years earlier:
Gallaba, Macho, Pinzger and McIntosh (ASE 2018: 12% of passing Travis builds actively ignore a
failure; the Hansel detector and the Gretel fixer, with accepted pull requests); CI-Odor (Vassallo,
Proksch, Gall, Di Penta, 2019); Zampetti et al.'s catalogue of CI bad practices (2020); CD-Linter's
Fake Success smell (Vassallo, Proksch, Jancso, Gall, Di Penta, 2020: 633 instances in 5.4% of 5,312
GitLab projects, and 145 issues followed for six months); and "silent failures", a green job that
failed at its task, measured in an industrial pipeline by Aïdasso, Bordeleau and Tizghadam (arXiv
2509.14347, 2025-09-17). The pseudo-tested-methods line (Niedermayr et al., 2016; Vera-Pérez,
Danglot, Monperrus, Baudry, 2018) is the origin of "remove the work and demand that something
fails". SILENT-PASS is a code-level member of that family rather than the CI form, so the class is
adjacent, not identical. **Lines 99-101**, *"What is new is the class, not the technique … none
targets measurement integrity"*, and **line 96**, *"the code half is empty"*, fall with it.
**Lines 113-114** (*"a category with one credible answer available and nobody sitting in it"*)
and **lines 120-121** (*"Nobody in this space is doing that"*, of publishing one's own failures):
Swarm Orchestrator's `docs/claims.md` (from 2026-08-18) maps every public claim to an artifact,
lists what may not be said, and withdrew a published number; honest-signal (2026-07-09) publishes
16 of 18 hypotheses falsified; COMPare published every assessment including its own coding error
rate. **Reading:** the four holdings are the lab's work on a family others named earlier; the
"no one else" and "nobody" sentences are withdrawn.

**`THESIS_the_honesty_standard_2026_05_31.md`.** No priority sentence was found, but it cites
neither of the two benchmarks that made honesty measurable apart from accuracy before it: MASK
(Ren, Agarwal, Mazeika and others, Center for AI Safety and Scale AI, arXiv 2503.03750, 2025) and
Liars' Bench (Kretschmar, Laurito, Maiya, Marks, arXiv 2511.16035, 2025). They are credited here.

## 3. Code and non-frozen documents corrected in place on this branch

| file | what it said | what it says now |
|---|---|---|
| `styxx/analytics.py` docstring (lines 6 and 31-36) | primitives "nobody else has shipped"; no other observability tool computes a personality profile | the 0.1.0a3 claims are named as withdrawn and never surveyed; line count unchanged |
| `styxx/dashboard.py` docstring | nobody had visualised cognitive state in real time; a priority claim for the implementation | a live view of the package's own stream; the priority claim withdrawn |
| `styxx/forecast.py` docstring | a priority claim for predicting failure before it happens; every other AI safety system reactive | both withdrawn; representation engineering (Zou et al., 2023) read and steered internal state during generation earlier |
| `styxx/intercept.py` docstring and demo banner | a priority claim for catching a model mid-generation; every other system reactive | both withdrawn, with the same credit; the banner prints what the demo does |
| `styxx/probe.py` docstring | "Nobody offers this." | withdrawn, never surveyed; line count unchanged |
| `styxx/verify.py` docstring | "the answer no one else can" | withdrawn; semantic entropy (Farquhar et al., Nature 2024, already cited in `cognometry-manifesto.md`) credited |
| `styxx/hallucination.py` docstring | production tooling had no inference-time, per-token reader | withdrawn; representation engineering (Zou et al., 2023) credited |
| `styxx/declare.py` docstring | no credit | NabaOS, DeerFlow, readback, commitlint-scope, Swarm Orchestrator, Agent Trace, with dates |
| `styxx/admissibility.py`, `styxx/adapters/guardrails.py` docstrings | charter words for the lab's own certificate and score | plain descriptions |
| `examples/synth_preference_pairs.py` | printed "nobody else can build this" | prints what it uses and that others were never surveyed |
| `integrations/openclaw/styxx-capsule/SKILL.md` | a charter word for the capsule | "an HTML file that carries its bytes and a checker for them"; Swarm Orchestrator's verifier-carrying bundles (evidence 2026-08-18) credited |
| `integrations/inspect/refusal_probe_gate/README.md`, `eval.yaml` | a priority claim within Inspect Evals | withdrawn: the catalogue was not surveyed |
| `docs/research/cognitive-dynamics-v0.md`, `docs/research/cognitive-metrology-charter.md` | nobody had a readout of cognitive state; nothing could intervene during generation | withdrawn; representation engineering (Zou et al., 2023) credited |
| `README.md` (VERIFY) | no credit | a paragraph naming who did this earlier, with dates |
| `web/gate/README.md` | no credit | a section crediting the diff gate's neighbours, with dates |
| `sworn/README.md` | no credit | a section crediting sworn's neighbours, with dates |
| `benchmarks/silent_pass/CORPUS.md` | no credit | a section crediting the CI silent-failure research, with dates |
| `CHANGELOG.md` | released entries untouched | one `[Unreleased]` entry listing every correction here |

## 4. Modules not edited, and why

`styxx/sworn.py`, `charon.py`, `capsule.py`, `certify.py`, `diffgate.py`, `evidence.py`,
`claimdetect.py`, `attestation.py` and `corpus_audit.py` are not touched. Their bytes are digested
into receipts: `sworn_sha256` in every sworn verdict receipt and in the conformance vectors, and
charon's `_MODULES` in every charon line; `web/gate` pins `diffgate.py`'s hash. A docstring edit
would move digests that committed receipts name. None of them carries a priority sentence this pass
found; their credits live in the lane notes and the READMEs.

## 5. Sentences outside the landscape's lanes: withdrawn as never priced

The landscape did not refresh these lanes (cognometrics, the white-box honesty arcs, compliance
bridges), so no neighbour is named unless this repository already cites one. Each sentence below
claims or implies priority and was never priced by any survey. **Reading for every row:**
withdrawn as a priority claim; what the document measured stands on its own receipts.

| where | the claim, paraphrased |
|---|---|
| `cognometry-v0.md` lines 17 and 245; `cognometry-v0.5.md` and `cognometry-v0.5-pdfsafe.md` lines 16 and 495 | priority for styxx as an open-source instrument suite, and "we are" ahead of the field on a regression |
| `cognitive-instruction-set-v0.md` and `-filled.md` line 21 | priority "to our knowledge" for an open runtime of that kind (representation engineering, Zou et al., 2023, cited in this repository, steered residual streams earlier) |
| `cognometric-fingerprint-spec-v1.0.md` line 14; `spec-v1.0-robustness-supplement.md` line 336 | priority for an open reference framework and a public reference suite |
| `cognometry-research-agenda-2026.md` line 12 | "Nobody ships" runtime detection of prompt injection or adversarial input |
| `cognometry-manifesto.md` lines 112-113 | no other laboratory has a substrate to test Law III on |
| `EU_AI_ACT_COMPLIANCE_2026.md` lines 213 and 303 | priority for the Article 15 bridge; no other observability vendor publishes the artifact |
| `SYNTHESIS_program_2026_07_24.md` line 42 | the part "no other eval tooling ships" |
| `logprob-trajectory-confabulation.md` line 125 | a conditional priority claim for a real-time per-token trust signal (semantic entropy, Farquhar et al., Nature 2024, is cited in this repository) |
| `universal-cognitive-basis-phase2.md` line 164; `styxx_dogfood_claude_2026_05_14.md` line 170 | priority for a public measurement and a publicly issued artifact |
| `agent-conscience/PAPER_knowsay_gap_2026_07_27.md` line 270 | priority "to our knowledge" for a non-circular inference-time probe |
| `agent-conscience/PREREG_voice_lora_honesty_2026_08_11.md` line 25; `pre-output-gate/PREREG_holdout_gate_2026_06_02.md` line 15 | "Nobody has measured" the question the PREREG asks |
| `grounded-honesty-axis/PREREG_cross_model_belief_topography_2026_05_30.md` line 15 | a measurement "nobody else can" make |
| `oath-economy/SYNTHESIS_the_binding_stack_2026_07_02.md` line 36 | a quantity "nobody has ever quantified" |
| `introspection-gate/README_legibility_of_mind.md` line 11 | "a tool no one else has" |
| `showcase-viz/FINDING_live_signature_mvp_2026_06_10.md` line 36; `showcase-viz/SIGNATURE_VIZ_BUILD_PLAN_2026_06_10.md` line 47 | nobody else would build or ship the demo that way |
| `consensus-truth-engine/preregistration_consensus_2026_05_25.md` line 7 | "Nobody ships" calibrated reference-free abstention |
| `autopilot/STRATEGY_edge_panel_2026_07_13.md` lines 60-61 | the parity control "nobody else ships", which "styxx alone" can operate |
| `ai-human-alignment/en/meaning_agreement_demo.py` line 4 | use cases "nobody offers a tool for" (the same sentence is in `CHANGELOG.md` at 4094 and 4163) |
| `../scripts/dogfood/PREREG_verifiable_attestation.md` line 27 | "Nobody ships" an agent-produced, third-party-reproducible honesty attestation |

Found and **not** withdrawn, because no neighbour the landscape read contradicts them and each is
already worded as "we know of no other" with the size of its look stated: the extraction-term
sentences (`closed-model-frontier/PREREG_extraction_ceiling_2026_09_01.md` lines 34 and 184,
`ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md` line 130) and the cross-model read
sentences in `disjoint-worlds/` (`CENSUS_read_extraction_2026_09_01.md` line 222,
`PREREG_b52_pooled_battery_2026_09_01.md` line 119, `PREREG_open_set_read_2026_09_01.md` lines 37
and 162, `PREREG_read_extraction_ceiling_2026_09_01.md` line 318). They are still unpriced and
still owe a frozen procedure.

## 6. Copies of released text, not edited

Released `CHANGELOG.md` entries carry the same kind of sentence at lines 4094, 4163, 4713 and 4717,
5814, 6492, 7685, 7755, 7769, 7809, 8855 and 8935-8936; the `[Unreleased]` entry added today lists
each with its reading. `release/`, `zenodo/`, `arxiv/`, `drafts/`, `benchmarks/cognitive_bench/`,
`benchmarks/cogvm_demo/`, `benchmarks/darkcity_csv/` and the outreach and deposit scripts under
`scripts/` hold historical copies of released text, submission drafts and sent or unsent outreach;
they carry the same priority claims (and, in the outreach scripts, a charter phrase) and are not
edited, because they record what was said. They are read the same way: withdrawn, never priced.
The sand lane (`plates/`, `checksum/`) was not opened, by the landscape's handling note.

## 7. Owed by the open ci-audit pull requests (not touched here)

The ci-audit and SWALLOW documents live on an open branch from another session and are not edited
here. Before any of them merges they owe credit, with dates, to: CD-Linter (2020), Gallaba et al.
(2018), Zampetti et al. (2020), CI-Odor (2019), Aïdasso, Bordeleau and Tizghadam (2025-09-17, a year
before SWALLOW-1), the pseudo-tested-methods line (2016, 2018), Alshammari et al. (arXiv 2401.15788,
2024), Lazarek et al. (HotOS 2025), ShellCheck SC2312 and actionlint's ShellCheck integration,
Swarm Orchestrator's defect-injection oracle (2026-06-02) and `--challenges` (2026-09-07),
backcheck's *qualified* verdict for a `|| true` that swallowed a failure (2026-08-04), and
i-dont-believe-you (2026-09-15, six days before SWALLOW-1).

## 8. What this note does not do

It prices nothing and is not a survey under a frozen procedure. Every "we know of no other" left in
the tree still owes one, and the landscape lists the candidates for the diffgate conjunction and
the sworn re-pricing. A human-reviewed pass over the landscape's closest rows is still owed. No
neighbour was contacted.
