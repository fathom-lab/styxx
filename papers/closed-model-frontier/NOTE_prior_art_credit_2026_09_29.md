# Note — prior art for the diff gate, the evidence leg, BENCH-1, the OATH survey, the capsule and DECLARE-1, and the priority sentences here that were not earned

2026-09-29. This note sits beside frozen documents in this directory. None of them is edited: the
PREREGs and RESULTs are history, and two of them (`RESULT_oath_prior_art_survey_2026_08_26.md`,
`RECON_oath_prior_art_2026_08_26.md`) carry OATH certificates. What follows names each sentence,
the neighbour that did the thing earlier, the dates on both sides, and the reading of the sentence
that stands from today.

**Where the facts come from.** A landscape synthesis dated 2026-09-28 and its critic pass (one
synthesizer, five single-agent sweeps, 27 critic re-fetches through a summarising web reader and
the GitHub, Hugging Face and PyPI APIs under the lab's research user agent; no hashed bytes, no
frozen procedure, no human review). The operator approved these corrections on 2026-09-29. Nothing
here was re-fetched for this note, so every sentence below corrects toward less; none of it is a
new claim about styxx. Neighbours' dates are their own (arXiv v1, release notes, repository
creation per the GitHub API). Swarm Orchestrator's dates and numbers are self-reported by its
release notes and evidence-directory names; its commit history was not audited.

**The lab's dates used**, from `origin/main`'s log: OATH `certify` v0 2026-06-10; diffgate
2026-08-01 (7.29.0); the OATH capsule 2026-08-31; the evidence-leg PREREG 2026-09-01; sworn
2026-09-01; charon 2026-09-02; BENCH-1 and DECIDE-1 2026-09-17; DECLARE-1 2026-09-18.

## The neighbours, once, with dates

| neighbour | what it did | its date | against styxx | where |
|---|---|---|---|---|
| **Swarm Orchestrator / swarm-verify**, Brad Kinnard (ISC) | A merge gate for AI-written pull requests: `swarm audit` runs ten cheat detectors over the diff (they do not read the PR description) and writes a hash-chained audit ledger, shipped as a GitHub Action (v10.0.0). A defect-injection oracle catching 253 of 300 planted cheats (v11.1.0). v2.0.1 already "verifies results using transcript evidence before merging". Run bundles that carry their own dependency-free verifier. | v2.0.1 2026-01-26; v10.0.0 2026-05-23; v11.1.0 2026-06-02; bundles with evidence dated 2026-08-18 and 2026-08-23 | about ten weeks before diffgate; the verifier-carrying bundle before the capsule | https://github.com/moonrunnerkc/swarm-orchestrator |
| **AgentLiar**, Daksh Jain (MIT per its README; the GitHub API finds no licence file) | Takes a task description, an agent's completion claim (a JSON object with free-text summary and details plus structured files_modified / tests_added / tests_passed fields) and its file changes; runs a file check, a test-quality check (assertion-free tests), a scope-narrowing check ("only", "for now") and an optional LLM judge; one 0-100 score per PR; CLI, Python, GitHub Action, HTTP API. No accuracy published. | repository 2026-05-20 | about ten weeks before diffgate, as a checker of an agent's completion claim against its changes | https://github.com/dakshjain-1616/AgentLiar |
| **PR-MCI**, Jingzhi Gong, Giovanni Pinna, Yixin Bian, Jie M. Zhang (MSR '26 Mining Challenge) | Message-code inconsistency on 23,247 AIDev agent PRs; 406 (1.7%) highly inconsistent, 45.4% of them descriptions claiming unimplemented changes; accepted 28.3% vs 80.0%, merged in 55.8 h vs 16.0 h. 974 hand-annotated PRs (a 600-PR validation sample, κ 0.892, plus 374 high-inconsistency PRs). A heuristic similarity detector at P 0.742, R 0.548, F1 0.630 on the 600, in the replication repository `gjz78910/PR-MCI` (no licence per the GitHub API). | arXiv v1 2026-01-08 | about seven months before diffgate; they measured the phenomenon on AIDev before this lab did | https://arxiv.org/abs/2601.04886 |
| **backcheck**, Vector Institute (Apache-2.0) | Reads a coding-agent transcript, extracts the closing claims (tests pass, lint, build, commits, files written) and checks each against the tool-execution records in the same transcript; no model in the verdict path; supported / inconclusive / contradicted / unsupported / *qualified*. Reports 36 of 36 "tests pass" claims agreeing with an independent scan over 81 sessions. | repository 2026-08-04 | three days after diffgate; ahead of it on binding a test claim to the run it names | https://github.com/VectorInstitute/backcheck |
| **DeerFlow tool receipts**, bytedance/deer-flow (MIT) | The runtime stamps each tool result (name, status, argument and output sha256, bytes, time); the model never writes it. Report claims cite `[rN]`; an uncited claim is UNVERIFIED. Layer 2 anchors `tests_passed:<command>` to the recorded exit status. | RFC #4651 opened 2026-08-03 (the no-citation UNVERIFIED rule and the `tests_passed` binding were added in its revision 2, after reviews of 2026-08-03/04; the date of that edit is not recorded); receipts #4659 merged 2026-08-23; citation verification #5076 merged 2026-08-29; checklist #5109 merged 2026-09-01 | the RFC opened two days after diffgate; citation verification merged about three weeks before DECLARE-1 | https://github.com/bytedance/deer-flow/issues/4651 |
| **NabaOS tool receipts**, Abhinaba Basu | HMAC-signed per-call receipts; the model tags each claim by its evidence source; count and absence claims checked against result counts; 1,800 synthetic scenarios; self-tag compliance about 92% (Claude). Code promised, URL withheld for anonymous review. | arXiv 2026-03-09 | before diffgate and DECLARE-1 | https://arxiv.org/abs/2603.10060 |
| **readback**, Josh Duffy (MIT) | An agent declares claims as JSON in a `readback-claims` fenced block; six types; verified / contradicted / indeterminate; no LLM; exit 2 when nothing could be checked or no usable claims. | repository 2026-09-13 | five days before DECLARE-1, with the same three-verdict shape | https://github.com/joshduffy/readback |
| **AgentLTL**, Laila Elkoussy, Julien Perez (EPITA) | Temporal-logic (LTL) properties over tool-call traces, scored without a judge; a grounding predicate requires final-answer entities to appear in tool outputs. | arXiv 2026-07-01 | before diffgate | https://arxiv.org/abs/2607.02599 |
| **How Coding Agents Fail Their Users**, Ningzhi Tang et al. | 20,574 sessions; inaccurate self-reporting a growing share of misalignment episodes. | arXiv 2026-05-28 | before diffgate | https://arxiv.org/abs/2605.29442 |
| **Qodo Merge ticket compliance** (Qodo blog, Elana Krasner) | An LLM judges whether a PR's changes meet the linked ticket's requirements; no accuracy published. | 2024-11-26 | about twenty months before diffgate | https://www.qodo.ai/blog/qodo-merge-jira-ensuring-code-quality-through-ticket-compliance/ |
| **DOCER**, Wen Siang Tan, Markus Wagner, Christoph Treude | A GitHub Action on PRs flagging code references in documentation that no longer exist; more than a quarter of the 1,000 most popular projects had one. | arXiv 2023-07-10 | three years before diffgate | https://arxiv.org/abs/2307.04291 |
| **iComment**, Lin Tan, Ding Yuan, Gopal Krishna, Yuanyuan Zhou | Comments turned into checkable rules against code, with developer-confirmed findings. | SOSP 2007 | nineteen years before | https://www.eecg.utoronto.ca/~yuan/papers/icomment.html |
| **linux-next `Fixes:` tag checks**, Stephen Rothwell | Checks the claim in `Fixes: <sha> ("subject")` against the repository and mails problems to maintainers. | since at least 2019-02 | seven years before | https://lkml.iu.edu/hypermail/linux/kernel/1902.1/01841.html |
| **commitlint-scope**, thumbrise (Apache-2.0) | Lints changed paths against a declared conventional-commit scope: the comparison `only_touches` makes, on a structured header. | repository 2026-05-25 | about nine weeks before diffgate | https://github.com/thumbrise/commitlint-scope |
| **FEVER**, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal | Supported / Refuted / NotEnoughInfo over 185,445 claims. | 2018 | the three-way claim verdict with abstention, eight years before | https://arxiv.org/abs/1803.05355 |
| **Pham & Ghaleb; Ogenrwot & Businge** (MSR 2026) | Similarity-based alignment of agent PR descriptions against diffs, with human baselines. | arXiv 2026-01 | before diffgate | https://arxiv.org/abs/2601.17627 ; https://arxiv.org/abs/2601.17581 |
| **Sello (Notarized Agents)**, Juan Figuera | The receiving service signs a COSE receipt per agent action. | arXiv v1 2026-06-02 | before the capsule | https://arxiv.org/abs/2606.04193 |
| **GitHub artifact attestations and PEP 740** | Provenance bound to a package release. styxx 7.48.0 on PyPI carries none (the integrity endpoint returned 404 on 2026-09-28). | 2024 | earlier, and adopted by neither styxx release | https://peps.python.org/pep-0740/ |

Later than the styxx part, and recorded so no one reads them as followers of it: Transluce Docent's
"Measuring coding agent misalignment in the wild" (2026-08-04, contemporaneous with diffgate),
i-dont-believe-you (Leonard Leroy, 2026-09-15), OverclaimBench (arXiv 2609.20812, 2026-09-17),
agent-acceptance (Stackbilt, 2026-09-24), agent-claim-verifier (chiragborse1, 2026-09-27), and
Kraishan and Jitkajornwanich's *Plans They Abandon, Reports They Author* (arXiv 2609.12205, 2026-09).
Each was reached independently of this lab as far as anything read shows.

## The sentences

### `PREREG_evidence_leg_2026_09_01.md`

**Line 347**, *"Our contribution is that we know of no one who pointed it at prose and measured
what happened."* PR-MCI pointed a checker at 23,247 agent PR descriptions against their diffs and
measured it against 974 hand-labelled PRs, published 2026-01-08. DOCER ran its documentation check
over the 1,000 most popular GitHub projects in 2023, and iComment reported developer-confirmed
findings in 2007. The lab's own plan of the day before (`PLAN_prior_art_and_the_next_move_2026_08_31.md`)
already described the MSR study. **Reading from today:** others pointed checkers at prose and
measured what happened, earlier. What this leg went on to measure is narrower: per-accusation
precision of a deterministic gate against a blind panel with sealed decoys (EXTERNAL-1,
2026-08-31, precision 0.23 against a 0.95 floor). Among the sources the landscape read, no
neighbour publishes that kind of number (Swarm Orchestrator publishes a false-alarm rate, a
planted-cheat recall and a false-green rate, and withdrew one number); that is bounded by the
sources read and is not a survey result.

**Lines 355-361, the conjunction**, *"We know of no other tool that adjudicates an author's
free-form claim about a change they just made … against evidence bytes produced by a party other
than the claimant, deterministically, in CI … we know of none that combine the two, and none that
adjudicate a retrospective report."* It was never priced under a frozen procedure, and the
neighbours above occupy it. AgentLiar (2026-05-20) scores an agent's completion claim, a JSON
object with free-text summary and details plus structured files_modified / tests_added /
tests_passed fields, against its file changes and a task description, in a GitHub Action, with
deterministic checks beside an optional LLM judge; its output is one 0-100 score per PR. Swarm
Orchestrator (2026-05-23) gates AI-written pull requests on their diffs in a GitHub Action; its
detectors do not read the description, so it occupies the "deterministically, in CI, on agent
diffs" part, not the free-form-claim part. backcheck (2026-08-04) adjudicates the closing claims
of a session, a retrospective report, against tool records the claimant did not write,
deterministically, though where the agent works rather than in CI. DeerFlow renders a report with
action claims and no receipt citations UNVERIFIED against runtime-stamped receipts the model never
writes (a rule added in revision 2 of RFC #4651, date of the edit not recorded; merged in #5076 on
2026-08-29, three days before this PREREG). **Reading from today:** withdrawn as a
priority sentence. What is left after every neighbour is named is at most a residual, and it may
not carry any sentence until the frozen survey the landscape asks for prices it (candidates:
Swarm Orchestrator, AgentLiar, DOCER, backcheck, readback, DeerFlow, NabaOS, AgentLTL,
agent-claim-verifier, PR-MCI, Qodo).

**Lines 373-374**, *"… we can find none that verify the sentence is true against a test
attestation the claimant did not author."* backcheck verifies a "tests pass" sentence against the
transcript's own tool-execution record (2026-08-04). DeerFlow's layer 2 anchors
`tests_passed:<command>` to the recorded exit status (a binding added in revision 2 of RFC #4651,
date of the edit not recorded; merged 2026-09-01, the day of this PREREG). **Reading from today:** others verify that sentence against a run record the
claimant did not write, and on this class they are ahead of diffgate, whose accusing `tests_pass`
branch was deleted (`5e225b49`).

**Line 416**, on Doc Detective, *"We know of no earlier deployment of that idea."* DOCER (2023) is a
further deployment: a GitHub Action that checks prose about code in CI. Whether it precedes Doc
Detective was not checked, so line 416's credit to Doc Detective stands, unpriced.

### `PLAN_prior_art_and_the_next_move_2026_08_31.md`

**Lines 12-14**, *"The gap is real … No shipping tool deterministically gates natural-language
claims against evidence bytes and fails a build on contradiction."* The evidence-leg PREREG
retired this sentence itself the next day (its lines 331-337) against Cucumber/Gherkin, Doc
Detective and Jdoctor/Toradocu. The agent-specific neighbours are older still, both GitHub
Actions: AgentLiar (2026-05-20) scores a partly free-text completion claim against the changes,
and Swarm Orchestrator (2026-05-23) gates agent diffs without reading the description;
commitlint-scope (2026-05-25) fails a lint on a declared scope; DOCER (2023) runs on pull requests.
**Reading:** withdrawn.

**Line 21**, *"Qodo's 'ticket compliance' is the nearest neighbour."* Swarm Orchestrator,
AgentLiar and backcheck are nearer on the landscape's scale (closeness 4 against Qodo's 3), and
two of them predate diffgate. **Reading:** Qodo is one neighbour of several, and not the nearest.

**Lines 53-54 and 84-85**, *"The study calls for automated verification of this in CI and releases
no tool."* PR-MCI's replication repository carries its heuristic detector and its validation
metrics (read by `PREREG_third_party_precision_2026_09_01.md`). **Reading:** the study released a
detector measured against human labels, not a CI tool.

**Lines 83-85**, *"No shipping tool we could find performs deterministic, claim-level gating of an
author's natural-language summary against the diff it describes, failing CI on contradiction."*
AgentLiar (a PR-level score over a partly free-text claim) and Swarm Orchestrator (a diff gate
that does not read the summary), both earlier and both GitHub Actions. **Reading:** withdrawn as
never priced.

**Lines 86-89**, *"No project we could find ships a single self-contained file that seals its
evidence and re-verifies itself offline in a reader's browser, with a local command re-deriving
the verdict."* Swarm Orchestrator's run bundles carry their own dependency-free verifier, with
evidence dated 2026-08-18 and 2026-08-23, before the capsule (2026-08-31). Whether they are one
file verified in a browser was not read. **Reading:** unpriced, with an earlier near neighbour;
the capsule sentence still owes the frozen survey the landscape lists (Sello, Kraishan et al.,
Swarm Orchestrator, and the two schemas it names).

**Lines 90-91** describe the capsule corpus with a word the lab's charter forbids. **Reading:** an
HTML file that carries its bytes and a checker for them, as `integrations/openclaw/styxx-capsule/SKILL.md`
now says.

**Line 98 and the closing lines (121-122)** license a priority claim "with the qualifier attached". Under the
charter as it stands, the qualifier is not enough: a "we know of no other" needs a priced survey
behind it, and none of this plan's has one.

### `PREREG_bench1_pr_claim_benchmark_2026_09_17.md`

**Line 10**, *"There is no PR-level benchmark for 'does this pull request's description match its
diff'."* PR-MCI's 974 hand-annotated PRs (2026-01-08) are a PR-level labelled set for exactly that
question, and the PREREG's next sentences describe them. Sphinx, SWE-PRBench and BulkPR-Bench were
not read. **Reading:** a PR-level labelled set existed; BENCH-1's object is narrower, claim kinds
the diff settles without an annotator, and that is what it set out to build.

**Lines 12-13**, *"… and stated that detection tooling does not yet exist."* PR-MCI's statement at
its own date (2026-01-08); adjacent tools already existed then (Qodo ticket compliance 2024-11-26,
DOCER 2023). By BENCH-1's date AgentLiar, Swarm Orchestrator, backcheck and readback
existed, as did diffgate. **Reading:** read it as PR-MCI's statement at its own date.

**Lines 25-26**, *"no existing benchmark reports the second"* (the false discovery rate at the base
rate). Unpriced. **Reading:** "we know of none among the sources read", no stronger.

**Line 101**, *"Nobody has built the benchmark the field was told it needed."* Same correction as
line 10. **Reading:** withdrawn.

### `PREREG_decide1_decidable_fraction_2026_09_17.md`, line 17

*"… and nobody has published a number for it."* Among the sources the landscape read, DeerFlow and
readback emit UNVERIFIED or indeterminate without publishing a rate, agent-claim-verifier's 0
VERIFIED on 62 local sessions is not a rate, and Swarm Orchestrator reports `unmeasured` per run
with no population rate. VeriFin reports coverage on financial filings (from its body, not
re-checked). **Reading:** among the agent-claim checkers read, none publishes a decided fraction;
outside them, VeriFin reports one on filings. "Nobody" is not licensed.

### `PREREG_compat2_surface_and_panel_2026_09_16.md`, line 12

*"The sentence agents write most and nobody checks is 'no breaking changes'."* cargo-semver-checks,
japicmp and revapi check API breakage against a declared version, and the evidence-leg PREREG
credits them for exactly that. **Reading:** tools check the property against a structured claim;
whether any reads the prose sentence was not surveyed.

### `PREREG_collateral_census_2026_08_31.md`, line 22

*"The number that move needs and nobody has is the collateral floor."* No neighbour the landscape
read measured it. **Reading:** "we know of no one who has published it", unpriced.

### `RESULT_oath_prior_art_survey_2026_08_26.md` and `RECON_oath_prior_art_2026_08_26.md` (both certified)

**RESULT lines 124-129 and RECON lines 93-96**: *"Nobody found in this survey pointed such
machinery at prose that was never written to carry receipts and reported what happened"*, carried
as the survey's surviving "negative result … still … unoccupied". As a sentence about that
survey's thirteen queries it was true. As a claim about the field it is not: statcheck, which the
survey itself credits, recomputes the statistics in published articles as delivered; PR-MCI
(2026-01-08) measured a checker on agent PR descriptions; Deterministic Integrity Gates (Nam,
Jeong, Kim, arXiv 2606.09500, 2026-06-08) reconciles a manuscript's numbers against locked analysis
tables; and metacheck's `reproducibility_check` (ScienceVerse: DeBruine, Lakens, Mesquida, Werner;
committed to its `dev` branch 2026-08-16, ten days before the survey) re-executes a paper's shared
code and matches every reported test against the output, contradicting the survey's null that no
package checks a prose document's numbers against machine-readable results (on a development
branch, not a release).

**RESULT lines 143-144**, *"others built the parts; nobody has run them against documents that were
not written for them, and nobody has applied them to themselves at this scale"*, with lines
130-133 (*"a practice at a size nobody surveyed matches"*). The first half
falls to the same neighbours. The second half was never measured: honest-signal (repository
2026-07-09) gates its own claims on preregistrations and publishes 16 of 18 hypotheses falsified;
Swarm Orchestrator's `docs/claims.md` (from 2026-08-18) maps every public claim to an artifact,
lists what may not be said, and withdrew a published number; COMPare published every trial
assessment it made, including its own coding error rate (2 of 756). **Reading:** others run such machinery on documents not written for it, and others
turn it on themselves; the scale comparison is unmeasured and may not be said.

### DECLARE-1 (`PREREG_declare1_the_toll_2026_09_18.md`, `RESULT_declare1_the_toll_2026_09_18.md`)

Neither document makes a priority claim, and neither credits the neighbours that declared claims
earlier. They are: NabaOS (2026-03-09: claims tagged by evidence source, checked against signed
receipts, with a measured self-tag compliance DECLARE-1 does not have); DeerFlow's receipt
citations (RFC #4651, its UNVERIFIED rule added in revision 2, date of the edit not recorded;
merged in #5076 2026-08-29); Swarm Orchestrator's declared-file-set check
(evidence dated 2026-08-18); readback (2026-09-13, a fenced claim block and the same three
verdicts, five days earlier); commitlint-scope (2026-05-25, a declared scope against changed
paths); and Agent Trace (v0.1.0 RFC, 2026-01, attribution records that verify nothing). Later:
agent-acceptance (2026-09-24). DECLARE-1 measured adoption at zero; NabaOS measured compliance.
`styxx/declare.py` now carries this credit in its docstring.

### The capsule (`SPEC_oath_capsule_v01_2026_08_31.md`, `SPEC_oath_capsule_v02_2026_08_31.md`, `HANDOFF_capsule_v02_2026_08_31.md`)

No priority sentence; the credit they lack is Swarm Orchestrator's verifier-carrying run bundles
(evidence 2026-08-18 and 2026-08-23) and Sello's receiver-signed action receipts (arXiv v1 2026-06-02).
Swarm Orchestrator also signs its verdicts as DSSE / in-toto attestations (v14.0.0, 2026-09-07);
nothing styxx ships is signed, and its 7.48.0 release carries no PEP 740 provenance.

### EXTERNAL-1 (`RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md`, certified)

No priority sentence. For the record: measuring a claim checker against human labels on AIDev is
PR-MCI's, earlier. What the landscape found specific to EXTERNAL-1 among the sources it read is the
per-accusation precision with sealed decoys for a gate, and the number is unflattering (0.23; the
path class is disabled).

## What this note does not do

It prices nothing, re-fetches nothing, and is not a survey under a frozen procedure. A
human-reviewed pass over the landscape's closeness-4 rows is still owed. The lab-wide record,
including the code and documents corrected in place today, is `../NOTE_prior_art_credit_2026_09_29.md`;
sworn's sentences are in `../sworn/NOTE_prior_art_credit_2026_09_29.md` and charon's in
`../charon/NOTE_prior_art_credit_2026_09_29.md`.
