# PLAN — the next level: the format leaves the builder's hands, and every sentence about it waits for a number

Fathom Lab · 2026-09-02 · **A plan, not a result.** Decided by a six-lens design panel and nine
adversarial reviews run the same day (workflow `wf_3d0d4bca-ea3`; the reviews are the record, the
plan is the author's reading of them). It makes no numeric claim of its own: every number below
is sworn to the receipt that already holds it, and every sentence about what the lab may say is
written so it can be argued with. Successor to `closed-model-frontier/PLAN_prior_art_and_the_next_move_2026_08_31.md`,
which it does not edit.

## The directive, and what the receipts allow it to mean

The operator asked for the utmost ambition: breakthroughs in science and technology as a whole,
and a lab that leads the agentic race. This plan takes that the only way this program takes
anything — by receipts — and the receipts say where the ceiling is.

Every instrument this lab built to *find* a claim in a stranger's prose died against a blind
panel. <sworn r="path:papers/closed-model-frontier/external1_adjudication.json#/precision" k="numeric">The shipped gate scored 0.23 precision on agent pull requests it had never seen.</sworn>
<sworn r="path:papers/closed-model-frontier/v14_adjudication.json#/precision" k="numeric">After three repair cycles the held-out figure was 0.16.</sworn>
<sworn r="path:papers/closed-model-frontier/oath_adjudication_result.json#/false_accusation_rate/rate" k="numeric">The document verifier's false-accusation rate abroad was 0.2596</sworn>,
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/false_accusation_rate_after" k="numeric">and 0.2323 after its range rule was turned into a reporter.</sworn>
<sworn r="path:papers/closed-model-frontier/stage2_result.json#/arms/flagged/A_share" k="numeric">The best structural claim detector reached 0.4211 precision</sworn>
<sworn r="path:papers/closed-model-frontier/stage2_result.json#/arms/flagged/total" k="numeric">on 38 sentences.</sworn>
And the account that would have excused the adjudicator was tested and lost:
<sworn r="path:papers/closed-model-frontier/extraction_panel_result.json#/decomposition/E" k="numeric">the extraction term came back at 0.55</sworn>,
<sworn r="path:papers/closed-model-frontier/extraction_panel_result.json#/decomposition/A" k="numeric">the adjudication term at 0.20</sworn>,
<sworn r="path:papers/closed-model-frontier/extraction_panel_result.json#/decomposition/G_E3_verdict" k="quote">and the frozen gate read `REFUTED`</sworn> —
<sworn r="path:papers/closed-model-frontier/extraction_panel_result.json#/decomposition/upheld_among_claim" k="numeric">with only 11 upheld accusations sitting among the sentences the panel called claims</sworn>
<sworn r="path:papers/closed-model-frontier/extraction_panel_result.json#/decomposition/n_claim" k="numeric">out of 55</sworn>,
so the identity that named the term does not close, and the correct form carries a third term
for accusations upheld against sentences that made no claim.

What held against strangers was a target the author had committed to by structure:
<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/header/genuine_share" k="numeric">header-handed accusations were genuine at 0.9515</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/line/genuine_share" k="numeric">against 0.6391 for line-handed ones</sworn> —
read with the confound now receipted rather than rumoured:
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/kind_adjusted_gap" k="numeric">reweighted to the line cell's token-kind mix the gap is 0.1695</sworn>,
exploratory, one family, one repository supplying most rows. Sworn output is that surviving
mechanism's limiting case — the author hands the verifier a commitment, never a target — and the
receipts on sworn itself say two things. The verifier keeps the author's word on the spans bound:
<sworn r="r1" k="numeric">358 tests passed in the harness run this plan swears to.</sworn>
And nothing yet shows the author binds the spans that matter:
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/documents" k="numeric">across the 12 sworn documents committed before this cycle</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/v01_estimate_min" k="numeric">the coverage number the verifier printed ran from 0.6667</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/v01_estimate_max" k="numeric">to 1.0</sworn>,
and it meant nothing, because its denominator was a diff-claim detector that never reads a
measured rate as a claim;
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/floor_min" k="numeric">the floor that replaces it runs from 0.0156</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/floor_max" k="numeric">to 0.2185</sworn>
on the same documents, and it is a floor, not coverage.

So the ceiling is not a smarter detector, not a new attestation rail (in-toto, SLSA, sigstore,
C2PA occupy those), not a "wire" or a "protocol" (nothing outside this lab has minted a manifest),
and not a number about strangers (no panel of two model families has read a sworn document). The
ceiling is a format that has been attacked, a measurement that has been designed so it cannot be
passed by the failure it exists to detect, and engineering shipped under labels narrow enough to
survive the next reviewer. That is the whole move, and it is stated in the lab's own words:
**maximal in scope, minimal in claim.**

## The move, in four legs

**Leg 1 — the format hardens before it leaves (done this cycle).** Twelve attacks, four rules,
the coverage number withdrawn, provenance and rungs printed on every span, receipts re-issued
under a digest that no longer depends on the observer. `SPEC_sworn_output_v02_2026_09_02.md`,
`ATTACKS_sworn_v01_battery_2026_09_02.md`, `RESULT_sworn_v02_ships_2026_09_02.md`. The attacker
was the builder, which the battery says in its last paragraph; the next attacker should not be.

**Leg 2 — the measurement, repaired, preregistered, run on this box.** The v0.1 design is
redesigned in `DESIGN_sworn_measurement_v2_2026_09_02.md` with every repair the reviews named:
seats from two model families, both available on this machine without API credits (Claude through
the subscription CLI transport already used in `run_b23_fable.py`; a local open-weight instruct
model on CPU), so the credit block the backlog records no longer binds a panel; per-family
two-sided sealed decoys; the headline reported in three cells — load-bearing and bindable under
the kinds, load-bearing but unbindable, not load-bearing — so a miss is titled *authors leave
bindable sentences unbound* and never more; a receipt-seeing panel so the pairing question (does
the leaf evidence the sentence?) has a producer, which the v0.1 design's false-accusation gate
lacked; a seeded-canary arm of harness-inserted known-false spans so the verifier's FAILED
precision is a count with a denominator rather than a vacuous zero; the HELD-but-false cell
preregistered as a quantity of its own, because the extraction RESULT showed the lab had
forgotten that term once already; the in-house arm first (eight sworn RESULTs already in the tree
were written about other arcs by authors who had read the bars, and the design says so), a
CI-bound pilot second where receipts other than a diff exist, and the diff-only external arm
last, titled as binding behaviour under a diff-only receipt pool with the receipt-availability
ceiling reported first. **What blocks it is one signature on the bars.** A lock hash that reads
"TBD" is the shape the audit named the worst of both, so the design stays a DESIGN until the
operator signs, and then it is a PREREG in a commit and the panel runs the same week.

**Leg 3 — engineering, each piece under its own narrow label, in this order.**

| # | artifact | path | label it ships under | days |
|---|---|---|---|---|
| 1 | OATH certificates resolve receipts by digest at the issuing commit; `corpus_audit` distinguishes *receipt regenerated under a certificate* from *certificate wrong*; a census over the 213 tracked certificates with *issuing commit unrecoverable* as its own cell; no corpus re-issue | `styxx/certify.py`, `styxx/corpus_audit.py`, `tests/test_certify_by_digest.py` | the audit's one ADVANCE, and the general repair of "a receipt is history too" | 2 |
| 2 | conformance vectors generated from `tests/test_sworn.py` — document bytes, manifest, tree snapshot, expected receipt core — content-addressed by one digest | `conformance/sworn/` | the precondition for any second verifier; no claim | 1 |
| 3 | harness adapters at L1 and L2: a Claude Code hook pair (PostToolUse and Stop) minting `tool_stdout` / `file_read` receipts and recording `authored_sha256` for Write and Edit, documented as blind to shell-written files; a JUnit adapter through `styxx.evidence`; a GitHub event adapter | `integrations/claude-code/sworn-hooks/`, `styxx/harness/` | adapters, never a recorder (in-toto Witness records); L1 printed as weak on every manifest | 3 |
| 4 | a report-only sworn action in its own subdirectory (the root action is diffgate's), the runner minting the manifest after the turn at L2, exit 0 on every verdict, the README stating that on pull requests from forks the minting job is the claimant's | `sworn/action.yml`, `sworn/sworn_action.py` | report-only until the measurement prices FAILED; dogfooded only after the operator merges the workflow | 2 |
| 5 | a browser verifier held to the vectors for `rN` and embedded blobs only, and a capsule profile that seals document, manifest and receipt and fails closed | `styxx/_data/sworn_verify.js`, `styxx/capsule.py` | "re-derives sworn span verdicts offline; a forger controlling the whole file passes both browser layers; the package at the named commit is the check" — never *self-verifying* | 3 |
| 6 | the prior-art survey the v0.1 spec owes, with Proof-Carrying Numbers, Self-Verifying Measurement Records, Deterministic Integrity Gates, nanopublications, knitr/Quarto inline values, Registered Reports and honest-signal read in full | `papers/sworn/SURVEY_sworn_neighbours_2026_09.md` | the gate on every "we know of no other" | 2 |
| 7 | a release carrying `styxx.sworn`, and a cold-start target that says *clone at a pinned commit*, not *pip install*, until it does | `styxx/_version.py`, `REPLICATIONS.md` | operator-gated | — |

**Leg 4 — the science move, preregistered and cheap.** The provenance reading of the handedness
result — that who handed the verifier its target bounds what a verdict can mean, independently of
the token's form — is now a testable law with its contaminated prior committed
(`EXPLORATORY_handedness_by_kind_2026_09_02.md`). Its preregistration needs a fresh corpus disjoint
from the repositories already used, integer cells sized above thirty, two families of seats, and
publication with and without the dominant repository. If it holds, the field gets a number it does
not have: the extraction ceiling of any free-prose claim verifier is predictable from its
target-provenance mix before a panel is convened, and cannot be raised by a better judge, only by
moving the target up the ladder — which is what a format the author commits to does. If it fails,
the grain synthesis loses its mechanism in public. Either way it is a result about verification
itself, and it costs a corpus fetch and one panel.

## The claim ledger

**May be said now.** That sworn output survived a twelve-attack battery at the price of four rules,
with the six unrepaired attacks named. That the coverage estimate v0.1 printed was near-vacuous
beside result-shaped documents, with the census. That the verifier keeps an author's word on the
spans it bound, re-derivable offline at a named commit. That every number the lab has published
about finding claims in strangers' prose failed a frozen bar, with the receipts.

**Only with the qualifier, and only after leg 3 item 6.** *We know of no other format that binds
whole sentences of a free-text retrospective report — numbers, quotes, hashes and negatives — to
bytes minted by a party other than the author, with a distinct verdict for a document that bound
nothing and the unbound sentences counted beside every verdict.* Proof-Carrying Numbers does this
for numerals in a renderer; Cucumber for controlled steps the author executes; in-toto and Witness
for artifacts, not sentences; Registered Reports bind the bar, not the sentence. Each clause is
occupied; only the conjunction is not, and the survey prices it.

**Must not be said.** "First." "Nobody." "Wire", "protocol", "the contract every agent's claims
are checked against" — until a harness outside this lab mints a manifest. "Self-verifying." "Zero
false accusations." "Sworn makes agents honest." Any bound-recall number before the panel runs.
Any "0.117". Any use of the sworn documents in this tree as evidence that the format works — they
are specimens chosen to pass, written by the builder or by authors who had read the bars.

## What "lead the agentic race" and "breakthroughs in science" mean here, honestly

They are conditionals, and the plan states the condition. If the measurement clears its bars with
two families of seats, the lab may say that an agent's report can arrive with its load-bearing
sentences bound to bytes the agent could not have written, the unbound share printed beside the
verdict, and every verdict re-derivable by a stranger — and that it measured how much of what
mattered such reports leave in narrative, which is the number any adoption argument has to cite.
If the provenance law holds, the lab may say that verification of claims made by minds has a
measurable ceiling set by who handed the target, and that the only way past it is a format the
author commits to. Neither sentence is licensed today. The plan's ambition is that both are
falsifiable within the month on hardware this lab already owns.

## What this plan does not say

That any of it is replicated outside the lab (external replications remain zero). That the two
model families on this machine are independent judges (correlated error across families is the
ceiling, and only an external seat moves it). That the measurement will pass — the design's own
first paragraph says the most likely outcome of the external arm is that agents leave what
matters in narrative, and that result publishes under that title. That the audit's other
recommendations are discharged: the three arcs it said to close, the back-pointers it said to
add, and the machine-readable verdict tokens it said the ledger owes are not in this plan and
remain owed.

---

*The panel wanted a wire, a science, a price list, an adversary, a pipe and a law, and the nine
reviews that followed took a clause from each until what was left could be sworn. What is left is
a format that has been attacked, a measurement that cannot be passed by the failure it exists to
detect, and a ledger of what the lab may say. That is the level the receipts allow. The next
level is a number.*
