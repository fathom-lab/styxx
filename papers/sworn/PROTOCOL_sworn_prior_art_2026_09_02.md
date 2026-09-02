# PROTOCOL — pricing one sentence: the neighbours of sworn output, frozen before any fetch

Fathom Lab · 2026-09-02 · **A frozen procedure, not a result.** Committed before the first
source was fetched; the commit id of this file is the receipt that it was. It prices exactly one
sentence, the one `PLAN_the_next_level_2026_09_02.md` holds under *only with the qualifier, and
only after leg 3 item 6*. It is the owed item 5 of `SPEC_sworn_output_v02_2026_09_02.md`.
Successor in shape to `closed-model-frontier/PROTOCOL_oath_prior_art_survey_2026_08_26.md`, which
priced OATH's claims and is not edited. The survey that runs this procedure is
`SURVEY_sworn_neighbours_2026_09_02.md`; its receipt is `sworn_prior_art_survey.json`.

## The sentence being priced, verbatim

> We know of no other format that binds whole sentences of a free-text retrospective report —
> numbers, quotes, hashes and negatives — to bytes minted by a party other than the author, with
> a distinct verdict for a document that bound nothing and the unbound sentences counted beside
> every verdict.

The plan already concedes that each clause has a neighbour and that only the conjunction is
unoccupied. That concession was written from memory. This procedure replaces memory with a
fetch, and it may retire the sentence entirely.

## The clauses

| id | clause | what it asserts, operationally |
|---|---|---|
| C1 | *binds whole sentences* | the object a check attaches to is a sentence of prose, not a numeral, an artifact, a step in a controlled language, or a bar |
| C2 | *of a free-text retrospective report* | the host is prose written after the work in the author's own words — not a template, a form, a controlled language, a structured assertion graph, or a plan committed before the work |
| C3 | *numbers, quotes, hashes and negatives* | four check kinds against evidence bytes: an exact numeral, an exact substring, a content hash, and the **absence** of a substring |
| C4 | *to bytes minted by a party other than the author* | the evidence bytes were produced by something the author could not write to at mint time — the rung ladder's L1 or above; a renderer executing the author's own code on the author's machine mints at L0 |
| C5 | *a distinct verdict for a document that bound nothing* | a document with zero bound spans receives its own verdict token (sworn: `UNSWORN`), distinct from pass and from fail |
| C6 | *the unbound sentences counted beside every verdict* | the count of sentences outside any binding is printed next to the verdict, on every verdict, not on request |
| C7 | the conjunction | one format does C1 through C6 together |

## The pricing rule, per clause and per source

Each READ source receives, for each clause, exactly one of:

- **RETIRES** — the source does the same thing for the same object (a whole sentence of a
  free-text report). The clause is then false as a novelty claim.
- **OCCUPIES** — the source does the same thing for a narrower or different object (a numeral,
  an artifact digest, a step of a controlled language, a bar, a structured assertion). The
  clause survives only with the neighbour named beside it.
- **SILENT** — the source does not address the clause.

A SKIMMED source may only be recorded as SILENT or as OCCUPIES; it may not RETIRE a clause
(an abstract is not enough evidence to retire a sentence) and its OCCUPIES is marked
*from abstract*. An UNFETCHABLE source is recorded as UNCHECKABLE, and every clause the source
list below says it *might* occupy is marked **UNPRICED**. UNPRICED is never read as free.

Clause status after all sources are scored:

| status | condition |
|---|---|
| RETIRED | at least one READ source RETIRES it |
| UNPRICED | not RETIRED, and at least one candidate source that might occupy it was UNFETCHABLE |
| OCCUPIED | not RETIRED, not UNPRICED, at least one source OCCUPIES it |
| FREE | every READ and SKIMMED source is SILENT on it and no candidate was UNFETCHABLE |

The conjunction C7 is RETIRED if one READ source RETIRES or OCCUPIES every one of C1–C6 at
once (it does all of it, for whatever object). Otherwise C7 takes the weakest status among
C1–C6 in the order RETIRED > UNPRICED > OCCUPIED > FREE, and survives as the conjunction of the
surviving clauses only.

## The sentence rule

1. If C1 or C4 is RETIRED, the sentence is **RETIRED** entirely: those two clauses are what the
   format is, and a sentence without them is about something else.
2. A RETIRED clause is deleted from the sentence.
3. An UNPRICED clause is deleted from the sentence. What could not be checked is not claimed.
4. An OCCUPIED clause survives with the neighbour that occupies it named in the same sentence
   or the one that follows.
5. A FREE clause survives with the words *among the sources read* and the READ count beside it.
6. The surviving sentence is written verbatim in the SURVEY and returned verbatim in the leg
   report. Nothing shorter than the surviving sentence is claimed; nothing longer is.
7. "We know of no other" may stand only with the list of sources READ named beside it, and only
   for what this procedure checked. It is never "nobody", and a FREE clause means *none of the
   sources read occupies it*, never more.

## The frozen question list, answered for every source

| q | question | answer vocabulary |
|---|---|---|
| Q1 | what does it bind? | `numeral` / `artifact` / `sentence` / `step` / `bar` / `assertion` / `credential` / `record` / `none` (several allowed) |
| Q2 | who mints the evidence bytes? | `author` / `renderer running the author's code` / `runner after the author's turn` / `third party` / `none` |
| Q3 | is narrative distinguished from bound text? | `yes` / `no` / `n.a.` — *yes* means the format itself marks which prose is bound and which is not |
| Q4 | is "bound nothing" a distinct verdict? | `yes` / `no` / `n.a.` |
| Q5 | is an unbound share or count printed? | `yes` / `no` / `n.a.` |
| Q6 | which check kinds? | any of `numeric`, `quote`, `hash`, `absent`, plus `other` named |
| Q7 | fetch date | ISO date |
| Q8 | status | `READ` / `SKIMMED` / `UNFETCHABLE` |
| Q9 | URL(s) fetched | as fetched, one per line; for an arXiv id, the id as given and the id that resolved |

Q1–Q6 are answered about what the source *does*, read against its own text, not about what it
could be extended to do. A source is scored on what it ships, and a paper on what it describes.

## READ, SKIMMED, UNFETCHABLE

- **READ** — the full text was fetched by this agent on the recorded date and read end to end.
  For a paper: the full-text HTML or the PDF, not the abstract page. For a tool or standard: the
  canonical specification or README **and** the source file or section that does the binding.
  For a model or practice (Registered Reports): the canonical description by its author or
  steward.
- **SKIMMED** — only an abstract, landing page, listing or summary was fetched. A source that is
  SKIMMED because its full text was behind a size ceiling or a rate limit says so.
- **UNFETCHABLE** — no fetch of any URL for the source succeeded after attempts at two distinct
  URLs (canonical, then a mirror or the repository). Recorded as UNCHECKABLE; see UNPRICED above.
- **Existence checks.** Two arXiv ids in the source list are marked *verify it exists* in the
  brief that commissioned this survey. An id that resolves to a paper with a different title is
  recorded under the title arXiv reports, and the source is scored as that paper. An id that does
  not resolve is searched by the expected title; if nothing is found, the source is
  UNFETCHABLE and any clause it might occupy is UNPRICED.

Searching (`WebSearch`) is allowed only to locate a canonical URL. A source that appears in a
search snippet and was never fetched is not a source. The recorded fetch is by this agent alone;
this survey does not have the independent re-fetch the OATH survey had, and says so.

## The source list, frozen

Each row names what the source *might* occupy, so an UNFETCHABLE source has a known cost.

| # | source | planned URL(s) | might occupy |
|---|---|---|---|
| S01 | Proof-Carrying Numbers | arxiv.org/abs/2509.06902 (full text), github.com/worldbank/pcn if it exists | C1 C2 C3 C4 C5 C6 |
| S02 | Self-Verifying Measurement Records | arxiv.org/abs/2606.27934 (full text) | C1 C2 C3 C4 C5 C6 |
| S03 | Deterministic Integrity Gates | arxiv.org/abs/2606.09500 (full text) | C3 C4 C5 |
| S04 | Proof-Carrying Certificates for LLM Pipelines | arxiv.org/abs/2605.16407 (existence to be verified) | C1 C2 C4 C5 C6 |
| S05 | citation-attribution evaluators | arxiv.org/abs/2605.06635 (existence to be verified) | C1 C2 C6 |
| S06 | honest-signal | github.com/alexcard3/honest-signal (README and `firewall.py`); its paper if one exists | C4 C5 |
| S07 | nanopublications | nanopub.net and the nanopublication guidelines (assertion / provenance / pubinfo) | C1 C2 C4 |
| S08 | knitr / Sweave | yihui.org/knitr (inline code) and the Sweave manual | C1 C2 C3 C4 C6 |
| S09 | Quarto inline computed values | quarto.org/docs/computations/inline-code.html | C1 C2 C3 C4 C6 |
| S10 | Jupyter-Book glue (MyST-NB) | myst-nb.readthedocs.io glue page | C1 C2 C3 C4 C6 |
| S11 | Registered Reports | cos.io/initiatives/registered-reports; Chambers's account | C2 C4 C5 |
| S12 | C2PA assertions | c2pa.org specification, assertions section | C3 C4 |
| S13 | in-toto Test Result predicate | github.com/in-toto/attestation, spec/predicates/test-result.md | C3 C4 C5 |
| S14 | in-toto Witness | github.com/in-toto/witness and witness.dev | C3 C4 |
| S15 | Cucumber / Gherkin | cucumber.io/docs/gherkin/reference | C1 C2 C5 C6 |
| S16 | Doc Detective | doc-detective.com | C1 C2 C5 C6 |
| S17 | W3C Verifiable Credentials | w3.org/TR/vc-data-model-2.0 | C3 C4 |
| S18 | SCITT | datatracker.ietf.org/doc/draft-ietf-scitt-architecture | C3 C4 |
| S19 | statcheck | statcheck.io and its paper or README | C1 C2 C3 C5 C6 |

Nineteen sources. The list is closed: a source found during fetching is recorded under *leads
not scored* and does not enter the pricing, so the survey cannot pad its own READ count.

## Quoting rule

At most one short phrase per source, in backticks or quotation marks, with attribution to the
URL fetched. No paraphrase is presented as a quote. No table cell reconstructs a source from
excerpts across rows.

## What the receipt holds, and what the survey swears

`sworn_prior_art_survey.json` (LF, written by hand from the fetch log) carries: the count of
sources and of each status; per-source answers to Q1–Q9; per-clause status; the count of clauses
RETIRED / UNPRICED / OCCUPIED / FREE; the surviving sentence verbatim; the leads not scored. The
SURVEY swears every count it prints to that file at the commit that holds it.

## What this procedure does not do

It does not decide whether the sentence is *true*. Nineteen named sources are not the
literature; a FREE clause is a fact about nineteen fetches on one date. It convenes no panel and
carries no precision; nothing here is a verdict about any document. It does not re-price OATH's
claims, which the 2026-08-26 survey already retired. It credits by name and does not rank: a
neighbour that occupies a clause for a narrower object built that thing before this lab did, and
the survey says so where it is so.
