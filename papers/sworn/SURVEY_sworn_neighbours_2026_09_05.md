# SURVEY — the neighbours of sworn output, priced against nineteen fetches

Fathom Lab · 2026-09-05 · **A survey, not a result.** It runs the procedure frozen on 2026-09-02
in `papers/sworn/PROTOCOL_sworn_prior_art_2026_09_02.md`, which named its sources, its questions
and its outcome table before anything was fetched, and it prices exactly one sentence: the one
`papers/PLAN_the_next_level_2026_09_02.md` holds under *only with the qualifier, and only after
leg 3 item 6*. It is the owed item 5 of `SPEC_sworn_output_v02_2026_09_02.md` and the discharge of
the line the changelog has carried since v0.1. Every count below is sworn to
`papers/sworn/sworn_prior_art_survey.json`, which holds the per-source answers and the derivation.
This document is itself sworn.

## What was read

<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/counts/sources_in_list" k="numeric">The frozen list named 19 sources and closed itself against additions.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/counts/read" k="numeric">All of them were fetched and read end to end on one date: 19 READ.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/counts/skimmed" k="numeric">0 sources were recorded as SKIMMED.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/counts/unfetchable" k="numeric">0 were UNFETCHABLE, so no clause is UNPRICED and no clause is unpriced by silence.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/counts/leads_not_scored" k="numeric">7 leads surfaced while fetching and are recorded without being scored, because the list was closed.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json" k="hash">The receipt these verdicts are read from is sha256 67a09996f79a7fd83f4c3b19346513257c858f47897046ace818842324f53347 at the commit this document's sidecar names.</sworn>

Every record carries the URLs as fetched, the time of the fetch and the sha256 of the bytes saved
from the page. Two readings the 2026-08-26 OATH survey left owed are discharged here: arXiv
2606.27934 was read in full rather than from its abstract, and arXiv 2509.06902 was fetched under
a procedure rather than recalled from a one-afternoon pilot.

## The sentence under test, and its clauses

> We know of no other format that binds whole sentences of a free-text retrospective report —
> numbers, quotes, hashes and negatives — to bytes minted by a party other than the author, with
> a distinct verdict for a document that bound nothing and the unbound sentences counted beside
> every verdict.

The frozen clause table splits it into C1 *binds whole sentences*, C2 *of a free-text retrospective
report*, C3 *numbers, quotes, hashes and negatives*, C4 *to bytes minted by a party other than the
author*, C5 *a distinct verdict for a document that bound nothing*, C6 *the unbound sentences
counted beside every verdict*, and C7 the conjunction. Each READ source scores RETIRES (same thing,
same object), OCCUPIES (same thing, narrower or different object) or SILENT.

## The verdicts

<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/clause_status_counts/retired" k="numeric">0 clauses are RETIRED: no source read does this to a whole sentence of a free-text report.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/clause_status_counts/free" k="numeric">0 clauses are FREE.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/clause_status_counts/occupied" k="numeric">All 6 come back OCCUPIED, so every clause survives only with a neighbour named beside it.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/clauses/C1/status" k="quote">The clause about binding whole sentences comes back `OCCUPIED`.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/conjunction/status" k="quote">The conjunction's status in the receipt is `OCCUPIED`, not retired and not free.</sworn>
<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/conjunction/max_clauses_occupied_by_one_source" k="numeric">No single source does all six; the most any one of them does is 5.</sworn>

| clause | status | nearest occupants | what they do instead |
|---|---|---|---|
| C1 binds whole sentences | OCCUPIED | Cited but Not Verified (2605.06635); Deterministic Integrity Gates | reach the sentence by segmenting or extracting it, then judge — the writer bound nothing |
| C2 free-text retrospective report | OCCUPIED | statcheck; *Self-Verifying Measurement Records*; knitr, Quarto, MyST-NB | already live inside the report's own prose, around numerals rather than sentences |
| C3 numbers, quotes, hashes, negatives | OCCUPIED | Deterministic Integrity Gates | all four kinds, aimed at claims a detector chose and at artifacts |
| C4 bytes minted by another party | OCCUPIED | in-toto Witness; SCITT; Verifiable Credentials; C2PA | mint who signed and when, for an artifact, never for a statement's content |
| C5 distinct verdict for binding nothing | OCCUPIED | C2PA; Proof-Carrying Certificates for LLM Pipelines | a status for content with no manifest; a four-token assurance card for a pipeline |
| C6 unbound counted beside the verdict | OCCUPIED | Deterministic Integrity Gates; honest-signal; Registered Reports | an untraced share from a detector's denominator; a coverage statement written by hand; a section heading |

### C1 — the clause the whole reading turns on

Every one of the nineteen scored OCCUPIES here, and two came close enough to matter.
Deterministic Integrity Gates extracts candidate claims from a clinical manuscript by script and
reconciles the numerical ones three ways: <sworn r="path:papers/sworn/sworn_prior_art_survey.json#/sources/2/quote" k="quote">its own report is that of 136 numerical claims, `74 (54%) exact-match a manifest-locked analysis table`</sworn>.
*Cited but Not Verified* segments a model-written report into sentence-level claims, retrieves each
cited URL and judges the pair. Both reach the object this clause names — a sentence of a free-text
report — and neither binds: a detector chooses the unit, and in the second case a model decides the
verdict. Under the frozen rule neither can RETIRE, because RETIRES asks for the same thing done to
the same object. The clause therefore survives on one word, *binds*, and the survey says so plainly
rather than letting the word do quiet work.

### C2 — the host was never the hard part

statcheck reads a published article as delivered, in PDF, HTML or DOCX, and recomputes p-values
from the statistics reported in its sentences. *Self-Verifying Measurement Records* binds every
quantity in the running text of its own paper by content hash. knitr, Sweave, Quarto and MyST-NB
put computed values in narrative prose by design. The retrospective report as host is crowded.

### C3 — one source has all four kinds

Deterministic Integrity Gates carries exact numerals against a manifest-locked table,
citation-key and cross-reference substring checks, SHA-256 over every input and derived table, and
absence-shaped guards for a missing citation. The kinds are not the distinguishing part of the
sentence. The reproducible-report family scores SILENT here and the reason is worth stating: an
inline value is substituted, not compared, so nothing can disagree with it and there is no verdict
to report.

### C4 — the rails are older and better at their own job

Witness mints attestations as a pipeline runs, signed without the developer holding a key and
timestamped by an outside authority. SCITT adds a transparency service that registers a signed
statement and returns a receipt, and its own security considerations draw the boundary this survey
turns on: <sworn r="path:papers/sworn/sworn_prior_art_survey.json#/sources/17/quote" k="quote">registering a statement, it says, `only proves it was produced by an Issuer`</sworn>.
Verifiable Credentials makes the same separation an ecosystem of issuers, holders and verifiers.
The lab credits all three as the prior art for the rung ladder and claims none of it.

### C5 and C6 — the two the plan named no occupant for

<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/clauses/C5/occupied_count" k="numeric">The distinct-verdict clause is occupied by 2 sources.</sworn>
C2PA's status vocabulary carries `manifest.unknownProvenance` and `ingredient.unknownProvenance`
for content that has no manifest at all, and they sit outside the failure-code table — a token for
having bound nothing, for an asset rather than a document. Proof-Carrying Certificates for LLM
Pipelines reports one of four verdicts on an assurance card, one of which is Abstain.

<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/clauses/C6/occupied_count" k="numeric">The counting clause is occupied by 4.</sworn>
Deterministic Integrity Gates prints an untraced share beside the traced one, from the denominator
its own detector chose — which is the shape sworn v0.2 withdrew when it replaced the coverage
estimate with a floor. honest-signal states, row by row rather than as an average, how much of its
own record sits outside its gate. Registered Reports puts every unregistered analysis into a
section named as such, in every Stage 2 report. Proof-Carrying Certificates audit-logs the claims
its residue operator dropped.

The near-miss that belongs in the record: honest-signal's gate, given a repository with no claims,
prints that there is nothing to verify and returns the success code. That is the outcome a distinct
verdict for a document that bound nothing exists to prevent, reached by the closest neighbour the
lab has.

### C7 — and the two that came within one clause

Deterministic Integrity Gates does C1, C2, C3, C4 and C6 — everything but the distinct verdict.
Proof-Carrying Certificates for LLM Pipelines does C1, C3, C4, C5 and C6 — everything but the
free-text host, which it declines on purpose, drawing its trust boundary above prose. The
conjunction is not occupied by anything read here, and it is closer to occupied than the plan's
map suggested.

## The surviving sentence

Rule 1 does not fire: neither C1 nor C4 is retired. No clause is deleted. Every clause survives
with its neighbour named, and by rule 4 the two paragraphs below travel together — the second is
not commentary, it is part of what may be said.

> Among the 19 sources read for this survey on 2026-09-05, we know of no other format that binds
> whole sentences of a free-text retrospective report - numbers, quotes, hashes and negatives - to
> bytes minted by a party other than the author, with a distinct verdict for a document that bound
> nothing and the unbound sentences counted beside every verdict.

> Every one of those six clauses is occupied separately: sentences of a free-text report are
> reached by the citation-attribution evaluator of Onweller et al. and by Deterministic Integrity
> Gates, both of which segment or extract rather than bind; the retrospective report is already the
> host in statcheck and in *Self-Verifying Measurement Records*; all four kinds appear in
> Deterministic Integrity Gates; non-author bytes are the whole subject of in-toto Witness, SCITT
> and Verifiable Credentials; a distinct state for having bound nothing exists as C2PA's
> unknownProvenance status and as the Abstain verdict of Proof-Carrying Certificates for LLM
> Pipelines; and what falls outside the binding is accounted for by Deterministic Integrity Gates'
> untraced share, by honest-signal's row-by-row coverage statement and by the Exploratory Analyses
> section a Registered Report requires. Two of them come within one clause of the whole
> conjunction: Deterministic Integrity Gates lacks only the distinct verdict, and Proof-Carrying
> Certificates lacks only the free-text host.

<sworn r="path:papers/sworn/sworn_prior_art_survey.json#/sentence/surviving_sentence" k="quote">The receipt's own copy of that sentence still contains `bytes minted by a party other than the author`.</sworn>

## What the tree may now say

Nothing below is edited into a committed document. Sworn documents are history; the wording is
proposed here and the next plan, spec or changelog entry adopts it by citing this survey.

**1. The claim-ledger sentence — `PLAN_the_next_level_2026_09_02.md`, under *Only with the
qualifier, and only after leg 3 item 6*.** The qualifier is now satisfiable. Replace the italic
sentence and the four-clause gloss that follows it with the two paragraphs quoted above, and
replace *Each clause is occupied; only the conjunction is not, and the survey prices it* with:
*Each clause is occupied and the survey names by whom; two sources come within one clause of the
conjunction; the survey is `papers/sworn/SURVEY_sworn_neighbours_2026_09_05.md`.* The plan's own
gloss survives the fetch — Proof-Carrying Numbers does check numerals in the renderer, Cucumber
does bind steps of a controlled language the author executes, in-toto and Witness do attest
artifacts, Registered Reports does bind the bar — and it is incomplete on two clauses it left
without an occupant.

**2. `SPEC_sworn_output_v01_2026_09_01.md`, the second-question sentence.** *We know of no other
format that makes the second question answerable at all* is not defended by this survey and not
retired by it: the procedure prices one sentence, and that one is not it. What can be said is that
the clause the second question corresponds to — the unbound counted beside the verdict — is
OCCUPIED, by an untraced share in Deterministic Integrity Gates, a hand-written coverage statement
in honest-signal, and a section heading in every Registered Report. The v0.1 sentence was written
about the coverage estimate v0.2 withdrew, so any successor should read: *Among the sources read
for the 2026-09-05 survey, we know of no other format that answers the second question with a
count whose denominator the checker did not choose.* That sentence is not priced here either; it
is offered as the honest shape, and pricing it would need its own frozen procedure.

**3. `SPEC_sworn_output_v02_2026_09_02.md` owed item 5 and `RESULT_sworn_v02_ships_2026_09_02.md`
owed item — the survey behind any "we know of no other".** Discharged by this document. The next
changelog entry may say so and cite the receipt.

**4. Credit that should appear in the next spec, not only in a survey.** C2PA for a status token
that names content carrying no manifest; Registered Reports for putting the unregistered part of a
report where a reader cannot miss it; Deterministic Integrity Gates for the widest kind coverage
over a manuscript and for printing what it could not trace; SCITT for stating the boundary between
who produced a statement and whether it is true; honest-signal, already credited for precedence,
also for saying row by row how much of its own record its gate does not cover.

**5. Sentences this survey did not price, and may not be read as clearing.** The diffgate
conjunction in `PREREG_evidence_leg_2026_09_01.md`; the capsule sentence in
`PLAN_prior_art_and_the_next_move_2026_08_31.md`; the unqualified *no other tool in the space*
sentences in the July cognometrics entries of `CHANGELOG.md`, `styxx/analytics.py` and
`styxx/three_axis/meta_rate.py`, which predate the standing rule and are outside every lane this
procedure covers. The charon name check is discharged elsewhere, in that spec's own errata, and is
not redone here.

## What this survey does not say

It does not say the sentence is true. Nineteen named sources are not the literature, and a clause
with no occupant would have been a fact about nineteen fetches on one date rather than about the
world — no clause turned out that way, which is the more useful outcome. It was run by one agent in
one pass, and unlike the 2026-08-26 OATH survey it has no independent re-fetch, so a source
recorded from memory would not have been caught; the saved bytes and their digests are what stands
in for that, and they prove the page was fetched, not that it was read well. The rows are authored:
the script checks their shape and derives the statuses from the frozen tables, and it does not read
the pages. It ranks no one and credits by name: a neighbour that occupies a clause for a narrower
object built that thing before this lab did.

## Owed

1. A human-reviewed pass before any of this goes outward, which the OATH survey also left owed.
2. An independent re-fetch of these nineteen by an agent that did not record them.
3. A frozen procedure for the two conjunctions this one did not price — the diffgate sentence and
   the capsule sentence — if the tree intends to keep saying them.
4. The v0.1 second-question sentence, priced properly or retired.

*The neighbours were named from memory for five weeks. They are named from fetches now, and two of
them are closer than the memory said.*
