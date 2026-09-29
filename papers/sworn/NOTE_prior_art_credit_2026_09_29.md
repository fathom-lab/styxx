# Note — sworn's neighbours after the 2026-09-05 survey, and the two sentences they re-price

2026-09-29. This note sits beside frozen documents in this directory and edits none of them: the
spec, the survey and the RESULTs are history, and most of them are sworn documents with receipts.
It names the sentences, the neighbours the 2026-09-05 survey did not read, their dates, and the
reading that stands from today.

**Where the facts come from.** The landscape synthesis of 2026-09-28 and its critic pass: single
agents reading through a summarising web fetcher and the GitHub API under the lab's research user
agent, no hashed bytes, no frozen procedure, no human review. Nothing was re-fetched for this note.
It does not re-run `PROTOCOL_sworn_prior_art_2026_09_02.md` and does not price anything; it records
what a re-pricing must add. Dates are the neighbours' own. sworn landed on `main` on 2026-09-01;
the browser verifier and its conformance vectors on 2026-09-05.

## Neighbours the 2026-09-05 survey did not read

| neighbour | what it does | its date | against sworn | where |
|---|---|---|---|---|
| **DeerFlow's zero-citation report verdict**, bytedance/deer-flow (MIT) | A completed subagent report that makes action claims with zero receipt citations renders `UNVERIFIED — action claims without receipt citations`, "a weak-negative signal, not a clean bill". Receipts are runtime-stamped; the model never writes them. | RFC opened 2026-08-03; rule added in revision 2 after reviews of 2026-08-03/04 (the date of that edit is not recorded); merged #5076 2026-08-29 | three days before sworn's `UNSWORN`, by merge | https://github.com/bytedance/deer-flow/issues/4651 |
| **metacheck** `reproducibility_check` and `match-reported`, Lisa DeBruine, Daniel Lakens, Cristian Mesquida, Jakub Werner (ScienceVerse; AGPL; not on CRAN) | Re-executes a paper's shared code, parses the output formats scientists publish, and requires every component of a reported test to co-occur in one output analysis at the reported precision. Prints "X of Y reported tests matched"; NA means there was nothing to check (no code, no self-contained output). | first committed on its `dev` branch 2026-08-16 (v0.3.1 on 2026-09-20; a release has not been checked) | before sworn, on a development branch | https://www.scienceverse.org/metacheck_book/chapters/mod-reproducibility-check.html |
| **VeriFin**, Bethel Hall, Sachi Shome, William Eiers | An LLM proposes numeric claims about 10-K filings; each operand is grounded in a filed XBRL fact, the formula is authorised independently, and Z3 returns Verified / Violated / Abstain (labels from the body, not re-checked). "Accepts none of the incorrect claims." | arXiv 2026-08-10 | before sworn; better than sworn on derived numbers through an authorised formula | https://arxiv.org/abs/2608.10213 |
| **Inline XBRL 1.1** and the **XBRL US DQC rules** | The displayed number in a filing is the tagged machine-readable fact; a processor conformance suite; 196 public, versioned, unit-tested rules. | iXBRL 1.1 2013-11-18; DQC rules since 2015 | more than a decade before | https://www.xbrl.org/specification/inlinexbrl-part1/rec-2013-11-18/inlinexbrl-part1-rec-2013-11-18.html |
| **showyourwork!**, Rodrigo Luger | A GitHub Action rebuilds the paper; numbers are regenerated from the code that produced them. | 2021 | five years before | https://show-your.work/en/latest/latex/ |
| **Trusty URIs**, Tobias Kuhn, Michel Dumontier | A hash in the identifier; three implementations run over 156,026 valid and byte-corrupted files, and the libraries that accepted corruption were reported. | 2014 | the practice behind the browser verifier's shared vectors, twelve years before | https://arxiv.org/abs/1401.5775 |
| **backcheck**, Vector Institute | Binds a "tests pass" claim to the run it refers to; a *qualified* verdict for a pass that a `\|\| true` swallowed. | repository 2026-08-04 | before the sworn action's JUnit binding | https://github.com/VectorInstitute/backcheck |
| **Swarm Orchestrator**, Brad Kinnard (ISC; self-reported dates) | Prints `task: unjudged` instead of a pass when no oracle ran; sealed criteria; `docs/claims.md` maps every public claim to an artifact and lists what may not be said. | `unjudged` in v14.0.0, 2026-09-07; claims table from 2026-08-18 | `unjudged` six days after sworn | https://github.com/moonrunnerkc/swarm-orchestrator |
| **readback**, Josh Duffy (MIT) | Exits 2, not 0, on a document with no usable claims. | 2026-09-13 | after sworn | https://github.com/joshduffy/readback |

## The sentences

### `SPEC_sworn_output_v01_2026_09_01.md`, line 351

*"We know of no other format that makes the second question answerable at all."* The 2026-09-05
survey already declined to defend it and found its clause (the unbound counted beside the verdict)
occupied by Deterministic Integrity Gates, honest-signal and Registered Reports. metacheck is a
further occupant, before sworn: "X of Y reported tests matched" is a count of how much of a
document's reported results a checker could match, printed beside its verdict. **Reading from
today:** retired. The successor sentence the survey offered (*a count whose denominator the checker
did not choose*) is still unpriced and may not be said until a frozen procedure prices it with
metacheck added.

### `SURVEY_sworn_neighbours_2026_09_05.md`, the surviving sentence (lines 129-132)

*"Among the 19 sources read for this survey on 2026-09-05, we know of no other format that binds
whole sentences of a free-text retrospective report … to bytes minted by a party other than the
author, with a distinct verdict for a document that bound nothing and the unbound sentences counted
beside every verdict."*

As a statement about nineteen sources it stands. As a position it is weaker than the survey
thought. Its fourth clause, **a distinct verdict for a document that bound nothing**, was occupied
before sworn on a narrower object: DeerFlow renders a whole report that makes action claims with no
receipt citation UNVERIFIED and says in words that this is not a clean bill (RFC #4651 revision 2,
date of the edit not recorded; merged in #5076 2026-08-29). That is the clause the survey found Deterministic Integrity Gates lacking. metacheck's
NA is the absence of code, not of bound claims, so it does not occupy the clause. Later neighbours
reach the same idea: Swarm Orchestrator's `task: unjudged` (2026-09-07) and readback's exit 2
(2026-09-13). What remains different, per the landscape: sworn's `UNSWORN` does not wait for a
classifier to find claims in the text; it follows from the author having bound nothing. Binding
numbers in a document to machine-readable facts is older on every side: Inline XBRL (2013),
knitr and Sweave (already in the survey), showyourwork! (2021), Deterministic Integrity Gates
(2026-06-08, already in the survey), metacheck (2026-08-16) and VeriFin (2026-08-10).

**Reading from today:** the surviving sentence is at most a residual. It may not be quoted outside
the survey until a frozen re-pricing adds DeerFlow, metacheck, VeriFin, Inline XBRL and the DQC
rules, showyourwork! and Swarm Orchestrator, with an independent re-fetch and a human-reviewed pass.
The survey proposed that the plan of record's claim-ledger sentence be replaced by this one. The
plan is a sworn document and was never edited, so both sentences stand in the tree, and this
reading applies to both; see `../NOTE_prior_art_credit_2026_09_29.md` for
`PLAN_the_next_level_2026_09_02.md`.

### The conformance vectors and the second verifier (`SPEC_sworn_conformance_vectors_v01_2026_09_05.md`, `RESULT_sworn_conformance_v01_ships_2026_09_05.md`, `SPEC_sworn_browser_verifier_v01_2026_09_05.md`, `RESULT_sworn_browser_verifier_v01_ships_2026_09_05.md`)

No priority sentence. The credit they lack: holding independent implementations to shared vectors
is an older practice, from the Inline XBRL processor conformance suite (2013) and Trusty URIs'
multi-implementation corruption test (2014). What the landscape did not find among the agent-receipt
neighbours it read is shared vectors between two implementations; Swarm Orchestrator's `rederive.mjs`
is held to its parsers by a test file and runs in a Node 22/24 matrix, and whether that amounts to
shared vectors was not read.

### The action (`SPEC_sworn_action_v01_2026_09_05.md`, `RESULT_sworn_action_v01_ships_2026_09_05.md`)

No priority sentence. Binding a "tests pass" claim to a run record the claimant did not write was
done earlier by backcheck (2026-08-04) and by DeerFlow's `tests_passed:<command>` anchoring (RFC
#4651 revision 2, date of the edit not recorded; merged 2026-09-01). The action's README now names them.

## What this note does not do

It prices nothing and edits no sworn document, receipt or sidecar. `styxx/sworn.py` is not edited
either: its bytes are digested into every verdict receipt (`sworn_sha256`) and into charon's lines,
so a docstring change would move the digest the committed receipts name. Its line 49 (a document
that swore nothing is `UNSWORN`, never "no failures") is a design rule, not a priority claim, and
this note is where its priority standing is corrected. The lab-wide record is
`../NOTE_prior_art_credit_2026_09_29.md`.
