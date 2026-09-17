# PROTOCOL — the correction of pass 4 of the sand survey: what a red team found, what is re-read and how, and the builder's repaired rules, all fixed before the first new fetch or reading

Fathom Lab · 2026-09-15 (UTC) · **A frozen procedure, not a result.** Committed and pushed before any byte is
fetched or any source is read under it; the commit id of this file is the receipt that it was.

`SURVEY_sand_neighbours_pass4_2026_09_14.md` is sworn and is not edited. This file fixes how the CORRECTION
beside it will be produced. The pass-4 protocol (`PROTOCOL_sand_prior_art_pass4_2026_09_15.md`, frozen at
`5d7f39ef`) and the protocols it inherits still hold. Nothing here adds a source to pass 4's list, changes its
search, rank or cap, or re-opens a retired clause.

## Why

A red team read pass 4 after it was pushed at `3060ffd2`. Seven finders reported 44 findings, and each was
attacked by one or two skeptics. 33 were confirmed, 9 refuted and 2 split. The findings and every skeptic's verdict
are committed beside this file in `sand_survey_pass4_inputs/redteam/redteam_return.json`. The confirmed findings
that a correction must answer:

1. **Xu et al. is not at distance one under the pass-4 definitions.** The builder moved pass 3's element flags onto
   E1-E5. Pass 3 flagged an interval for Xu et al. from a mean and standard deviation over three runs, which the
   frozen E4 definition excludes. Pass-3 flags also carry no per-element quote.
2. **Passes 1 and 2 were never coded by element.** The protocol counts nearness over "the listed source (from any
   pass)", and the SURVEY says "no source in any pass".
3. **B11 and B12 are recorded READ from pages that are not the article.** B11's text is alphaXiv's abstract with a
   generated overview; one element quote is the overview's text. B12's is a Nature paywall stub. The lab's own
   landing-page rule makes both SKIMMED.
4. **Only failed fetches were located again.** Landing pages whose article text is openly one link away (B05, B07)
   were marked SKIMMED instead, and B11's arXiv paper was never fetched.
5. **The builder prices a clause only from sources listed for it**, although readers return C4a for every source.
   It never computes the conjunction's status. It never uses quote verification. It lets one confirmation count
   twice. It invents a tie-break and a fallback. It counts 94 raw lead strings as unscored leads.
6. **The margin check skipped Xu et al.**, one of the four sources the record put at distance one.
7. **Several statements in the SURVEY and the report are false or overstated.** The CORRECTION will list each one
   with the evidence.

## What does not change

- The list (34 sources), the search record, the rank and the cap stay as committed.
- The proof-of-reading rules stay as pass 4 applied them: last-line quote among the last three lines, midpoint
  quote in the middle third by line number. They are not tightened retroactively. That eleven first readings prove
  the end with a page number is disclosed, not re-scored.
- Every pass-4 reading stays in the record. A later reading of the same source under this protocol replaces it in
  the correction record and keeps the earlier one under `superseded_readings`.

## Scope: text that is not the article

A fetched text is **not the article** when it is a landing page, a paywall stub, or an aggregator's page with a
generated summary. The red team's evidence and the readers' own notes place B04, B05, B07, B11 and B12 there.
Every such source is located once by title, author and year through open routes, the same routes pass 4 used:

- the article's own open PDF named on its landing page;
- its arXiv, PubMed Central or Europe PMC copy;
- an Internet Archive capture.

No route may present as a browser or pass a block. A located copy that carries the listed title is fetched with
`fetch_pass4.py` and read. A source with no located copy stays SKIMMED. Fixed now:

| id | route to try | if no copy |
|---|---|---|
| B04 | the landing page's `citation_pdf_url`; PubMed Central by its PMID | SKIMMED |
| B05 | `https://www.ijsat.org/papers/2026/1/10626.pdf` | SKIMMED |
| B07 | `https://cspub-ijcisim.org/index.php/ijcisim/article/download/4475/3594` | SKIMMED |
| B11 | `https://arxiv.org/pdf/2603.19022` | SKIMMED |
| B12 | none open (Nature paywall) | SKIMMED |
| B23 | DOIs 10.1126/science.zh9l2q0 and 10.1126/science.aeh8588; Europe PMC; archive captures | UNFETCHABLE |

The last row records B23's identifiers, which pass 4 never recorded.

## Re-coding the earlier passes' fingerprint sources

Every source that any earlier pass recorded as occupying C4a is read again, under the pass-4 definitions, from the
bytes that pass hashed:

- **Pass 2's eight:** S11, S12, S13, S14, S15, S16, S17 and S19.
  - The six PDFs are read from the files whose sha256 equals `fulltext_pdf_sha256` in `sand_prior_art_survey_pass2_2026_09_13.json`.
  - S15 and S16 are read from the HTML bytes whose sha256 equals that record's `sha256`.
  - All eight texts are extracted with `fetch_pass4.py`'s own extractor.
- **Pass 3's fifteen:** L01 to L15, read from the extracted texts whose sha256 equals `fulltext.sha256` in
  `sand_prior_art_survey_pass3_2026_09_14.json`.

A source whose bytes do not match its record is not read, and is recorded as not re-coded.

## How everything is read

- **First readers:** `sand_survey_pass4_inputs/prompts/read_pass4.js`, verbatim, batches of four. Every source is
  listed as might occupy C4a; the located pass-4 sources keep their pass-4 clauses. The workflow's own rule sends any
  RETIRES, and any C4a coded E1-E5 true, to two blind confirmers.
- **Margin readers:** every source this correction codes at distance one or zero is re-coded by four readers with
  `sand_survey_pass4_inputs/prompts/margins_pass4.js`, verbatim. That includes Xu et al. if its new coding puts it
  there. The margin check stays outside the protocol's pricing, as in pass 4.
- No other prompt is used.

## The builder's rules for the correction record

`build_sand_survey_pass4_correction.py` writes `sand_prior_art_survey_pass4_correction_2026_09_15.json` from:
- the committed pass-4 inputs;
- this correction's fetch record and reader returns;
- the scope file extended with B11 and B12.

Rules marked *new* differ from the pass-4 builder.

1. **READ, SKIMMED, UNFETCHABLE:** as pass 4 applied them. A source listed in the scope file is SKIMMED unless a
   located copy was read. *New:* a SKIMMED source's verdicts follow the inherited rule literally. SILENT or OCCUPIES
   is kept and marked `from_abstract`. A RETIRES from a SKIMMED source is recorded as OCCUPIES from abstract, with the
   reader's original verdict kept beside it. It neither retires nor unprices the clause.
2. **Elements, new:** an element counts as carried only when it is coded true **and** its quote is found in the
   text. A quote not found is recorded and does not lower the distance.
3. **Retirement:** a first reading carrying E1-E5 for C4a, or a RETIRES on any clause, needs two confirmers. *New:*
   the confirmers must be distinct from each other and from the first reader. The same confirmation in two files is
   refused. Each confirmer must carry E1-E5 by rule 2 (for C4a) or write RETIRES (for C2 or C5), and must pass both
   proofs of reading. Otherwise the source is DISPUTED.
4. **Clause pricing, new:** a clause is priced from every source that returned a verdict on it, not only from
   sources listed for it. A confirmed retirement retires the clause. *New:* a DISPUTED source on any clause makes the
   sentence UNLICENSED, as the pass-4 protocol's list of outcomes says ("UNLICENSED (a disputed retirement)"). An
   UNFETCHABLE candidate makes its clause UNPRICED. A verdict word outside RETIRES / OCCUPIES / SILENT makes the
   builder refuse.
5. **Nearness, new:** the pool holds only sources coded under the pass-4 definitions, with quotes: pass-4 readings
   and this correction's readings. Pass-3 flags never enter. For each of E1 to E5, the parenthesis names **every**
   READ source at distance one whose only missing element is that one. If no source is at distance one, it names
   every READ source at the smallest distance, each with what it lacks. If either would name more than five sources,
   the builder refuses and the sentence is UNLICENSED until a later protocol decides. No tie-break by id or by pass.
6. **Conjunction, new:** computed and recorded as the 2026-09-13 protocol defines it. It is RETIRED if one READ
   source retires or occupies every one of C2 to C5. Otherwise it takes the weakest status among C1 to C5, and the
   surviving and deleted clauses are listed.
7. **Leads, new:** normalised by title, and every lead that names a source listed in passes 1 to 4 is dropped. The
   raw strings and both counts are recorded.
8. **The sentence:** the inherited rules as pass 4 applied them, with rule 4's UNLICENSED.
   - *New:* when C5 is UNPRICED by an unfetchable candidate, the record also lists that candidate under
     `unchecked_candidates_that_could_retire_the_sentence`.
   - *New:* a reader's phrase loses only a trailing ", without ..." clause.
   - *New:* only `&amp; &lt; &gt; &quot; &#39;` are unescaped.
   - C2's neighbour stays the inherited one. That no pass-4 reader was asked for a "nearer" flag is disclosed.

The builder ships with tests for every rule marked new, written before its first run on this correction's returns.

## What the CORRECTION will do

It swears to the correction record and to any margin record beside it. It restates every count, status and name
that changed. It lists every confirmed finding with the repair or the disclosure it received. It gives, with times,
the order in which pass-4 rules were written relative to the results they touched. It corrects each false or
overstated statement in the SURVEY; the REPORT addendum, CHANGELOG entry and PR comment are corrected beside them.

## What this procedure does not do

- It does not score any work that is not already on a list.
- It does not change pass 4's search, rank or cap. Their disclosed defects (the null-year ranking, two screener
  inclusions, one wrong and one missed merge) are recorded, and are for the next pass's protocol.
- It does not tighten the proof-of-reading rules retroactively.
- It does not price "first", "novel" or "revolutionary", which no outcome licenses.
