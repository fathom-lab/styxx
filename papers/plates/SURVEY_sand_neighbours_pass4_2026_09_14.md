# SURVEY — the neighbours of the sand, pass 4: the fingerprint clause's elements defined and counted, a frozen search beside the leads, and the sentence survives without its bounty clause

**status: a survey under a frozen protocol, sworn to its record. Pass 4 scores the list that
`PROTOCOL_sand_prior_art_pass4_2026_09_15.md` closed by rule, committed and pushed at
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/protocol_commit" k="quote">`5d7f39ef`</sworn> before the first query was sent or the first source fetched. The fingerprint clause stays
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C4a/status" k="quote">`OCCUPIED`</sworn>, now at distance one: three sources lack exactly one of its five defined elements.
The bounty clause is <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C5/status" k="quote">`UNPRICED`</sworn>, because one candidate could not be fetched, and is deleted
from the sentence. A margin check beside the record found that one of five blind readings of one
source carries all five elements. Nothing here licenses "first", "novel" or "revolutionary".**

## what was done

Pass 3, its correction and its erratum wrote down three weaknesses of the earlier passes: two of the
fingerprint clause's elements were undefined, nearness was a reader's flag, and every source had
been chosen from memory or from a citation. The pass-4 protocol defined the elements E1 to E6 before
any fetch, counted nearness per element, required two blind confirmers for any retirement, and added
a frozen search of twelve queries on three engines beside the nine leads pass 3 had left unscored.

The list closed at <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/sources_in_list" k="numeric">34</sworn> sources:
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/part_a" k="numeric">9</sworn> leads and
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/part_b" k="numeric">25</sworn> from the search. The search returned
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/search_results" k="numeric">133</sworn> results, merged into
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/search_distinct" k="numeric">108</sworn> distinct works. Two screeners decided every work,
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/search_included" k="numeric">34</sworn> were included, and the
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/search_below_cap" k="numeric">9</sworn> below the cap of twenty-five stay leads.
The engines did not perform alike, and `sand_survey_pass4_inputs/search_record.json` keeps every
response's hash. The arXiv API returned no entry for any query, because the frozen syntax joins every
word of a query with AND; that is the frozen procedure's outcome and was not reworded. Semantic
Scholar rate-limited the unauthenticated requests, and
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/search_failures" k="numeric">6</sworn> queries were never answered in two runs; two more were answered with no match.
The web engine answered all twelve but gives no year, so the frozen tie-break by later year ranked
every dated work above every undated one at equal hit counts. That rule put five 2026 intrusion-
detection papers on the list and left undated works late in title order below the cap.

Every source was fetched by `fetch_pass4.py`, its bytes hashed, and its text accepted only when it
carried the listed title. Nine Part B sources failed the first fetch: blocked, rate-limited or
script-rendered pages. The 2026-09-13 protocol says sources are located by title, author and year, so
the same works were located by open routes, each recorded with its reason: the DOIs and PubMed
Central identifiers from the Semantic Scholar response itself, Europe PMC, the programme's own About
page, and Internet Archive captures. Eight were located. The ninth, Science's news article on paying
for errors, answered HTTP 403 and has no archive capture: it is
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/sources/B23/status" k="quote">`UNFETCHABLE`</sworn>. Three located pages turned out to be publisher landing pages, not article
text; they were marked in `fetched_text_scope.json` before their readings returned, and each is
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/sources/B04/status" k="quote">`SKIMMED`</sworn> under the earlier protocols' rule that a skimmed source may occupy and may not
retire. In all, <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/read" k="numeric">30</sworn> sources are READ,
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/skimmed" k="numeric">3</sworn> SKIMMED and
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/unfetchable" k="numeric">1</sworn> UNFETCHABLE.

Readers took the texts in batches of four, read each end to end, and coded E1 to E6 with a verbatim
quote for every element coded true. `build_sand_survey_pass4.py` accepts a reading only when the
reader's last-line quote is among the text's last three lines and the midpoint quote lies in the
middle third by line number: <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/shown_last_line" k="numeric">33</sworn> fetched sources showed the end and
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/shown_midpoint" k="numeric">33</sworn> the middle. Of the
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/element_quotes" k="numeric">52</sworn> quotes for elements coded true,
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/element_quotes_found" k="numeric">52</sworn> are in the text they are attributed to. No reader coded all five elements
for any source and no reader wrote RETIRES on any clause:
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_candidate_retirements" k="numeric">0</sworn> candidate retirements, so no confirmer ran and
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_disputed" k="numeric">0</sworn> sources are disputed. The surveyor typed no verdict.

## what it found

- The fingerprint clause (C4a) stays <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C4a/status" k="quote">`OCCUPIED`</sworn>. Across pass 4's readings,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_element_true/E1" k="numeric">15</sworn> sources carry the self-comparison (E1),
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_element_true/E2" k="numeric">22</sworn> a fixed item set (E2),
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_element_true/E3" k="numeric">3</sworn> a floor measured on the same weights and used to grade (E3),
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_element_true/E4" k="numeric">6</sworn> an interval on the comparison (E4),
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_element_true/E5" k="numeric">6</sworn> log-probabilities (E5), and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_element_true/E6" k="numeric">0</sworn> a hashed item set (E6). The fewest elements any source lacks is
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/c4a_fewest_missing_all_passes" k="numeric">1</sworn>.
- Nearness, counted. For each element, the record names the source whose only missing element is
  that one:
  - without a floor measured on the same weights (E3): Xu et al., as pass 3 coded it
    (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C4a/nearness/by_element/E3/id" k="quote">`pass3:L05`</sworn>);
  - without an interval on the comparison (E4): DiFR
    (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C4a/nearness/by_element/E4/id" k="quote">`B14`</sworn>), which scores served tokens against a trusted same-weights reference's
    logits with empirically set thresholds. Chauvin et al.'s Log Probability Tracking of LLM APIs is
    tied with it at distance <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/sources/B22/verdicts/C4a/distance" k="numeric">1</sworn>: it compares an API endpoint's first-token
    log-probabilities with its own earlier samples, graded against a null from identical weights,
    and never sets a significance level;
  - without log-probabilities (E5): Gao, Liang and Guestrin, Model Equality Testing
    (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C4a/nearness/by_element/E5/id" k="quote">`B01`</sworn>), which tests commercial endpoints against the reference weights they
    claim to serve, with a null simulated on those weights and tests at a stated alpha, on sampled
    text.
  No source in any pass is at distance one without the self-comparison or without the fixed item
  set. None of the nine leads is nearer than distance
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/sources/A04/verdicts/C4a/distance" k="numeric">2</sworn> (Chain & Hash). The search found what the citation trail could not: the three
  nearest sources are audits of served models, not studies of compression or drift.
- The chained-log clause (C2) stays <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C2/status" k="quote">`OCCUPIED`</sworn>. Seven sources keep hash chains of experiment
  predictions, update digests, anomaly flags or pipeline specifications. None chains verdicts
  re-derived from bytes.
- The bounty clause (C5) is <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C5/status" k="quote">`UNPRICED`</sworn>. Three sources occupy it (the ERROR programme at the
  University of Bern and two Nature pieces on it pay for errors found in other people's papers), and
  the unfetchable Science article is a candidate nobody could check. Under the inherited rule, one
  unchecked candidate is enough.

## the margin check

Pass 3's erratum showed that the fingerprint clause's margin can rest on how one reader codes one
element. So each source at distance one was read again by four further readers, blind to the first
coding and told nothing of which element was in question. `build_sand_survey_pass4_margins.py`
checked their codings as the record's builder checks a reading, and wrote
`sand_prior_art_survey_pass4_margins_2026_09_14.json`. All twelve passed both proofs of reading, and
every quote for an element coded true was found:
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B01/quotes_true_found" k="numeric">16</sworn>,
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B14/quotes_true_found" k="numeric">16</sworn> and
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B22/quotes_true_found" k="numeric">17</sworn>.

- Gao et al.: <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B01/element_true_counts/E5" k="numeric">0</sworn> of
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B01/n_counted" k="numeric">4</sworn> readers find log-probabilities.
- DiFR: <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B14/element_true_counts/E4" k="numeric">0</sworn> of
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B14/n_counted" k="numeric">4</sworn> find an interval on the comparison.
- Chauvin et al.: <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B22/element_true_counts/E4" k="numeric">1</sworn> of
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B22/n_counted" k="numeric">4</sworn> finds one. That reader counts the paper's 95% bootstrap intervals on detection AUC as an
  interval on the comparison, and so codes all five elements. The other three read those intervals as
  bounding the detector, not the compared quantity.
- <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B22/hardest_elements/E4" k="numeric">4</sworn> of the four readers of Chauvin et al. and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B14/hardest_elements/E4" k="numeric">4</sworn> of the four readers of DiFR name E4 as the element hardest to decide.

The check changes no status: the protocol prices the clause from the first reading and its
confirmers. What it measures is how much the price rests on. The fingerprint clause survives on one
element, the interval on the comparison, and on one source one of five blind readings finds that
element present. A later pass that settles whether an interval on a detector's performance is an
interval on the comparison could retire the clause, and with it the sentence.

## the sentence

The record's status is <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/sentence/status" k="quote">`SURVIVES_WITHOUT_C5`</sworn>. Two things changed from pass 3. The
fingerprint clause's parenthesis is rebuilt by count: three neighbours, each with the one element it
lacks, replace the six a reader flag had chosen. The bounty clause is gone, because a clause that
could not be checked is not claimed. Copied from the record's `sentence.text` (the name "Karvonen et
al." for DiFR is its reader's phrase):

> We know of no lab that binds every published number to bytes at a commit (where Deterministic
> Integrity Gates and Cited-but-Not-Verified bind sentences and citations of a manuscript, as the
> 2026-09-05 survey names them), re-derives every verdict from those bytes into a chained log (where
> Certificate Transparency and Rekor chain certificates and signed metadata, not verdicts), and
> fingerprints a model's behavior on hashed canaries against a measured null floor (where Xu et al.
> grade compressed-BERT label and probability loyalty against the uncompressed teacher, without a
> floor measured on the same weights; where Karvonen et al. score served tokens against a trusted
> same-weights reference's logits, calibrated on benign hardware noise, without an interval on the
> comparison; where Gao, Liang & Guestrin test sampled completions from LLM APIs against reference
> weights with calibrated MMD two-sample tests, without log-probabilities) — at once.

That is the only positioning sentence the lab may say about the sand. It is shorter than pass 3's
for the first time. It lost a clause to an unchecked source, and it now names neighbours that each
miss by exactly one element.

## what the lab got wrong on the way

Every item is in the commits from `450265ff` to `afc49486`, each written before the result it could
have bent was known.

- The frozen arXiv syntax joins every word of a query with AND, so that engine returned nothing.
- The first search run merged results on their raw titles, which let one paper take several slots
  under its mirror titles. It was stopped before any screening counted, and rerun with the engine
  answers replayed from cache.
- The first Semantic Scholar prompt treated a response with no match as unanswered, and re-sent two
  queries that had already been answered.
- The builder first measured the midpoint proof by character, which failed three Part A readings
  that are in the middle third by line.
- The first build doubled a phrase's "without", did not show the bounty clause's deletion in the
  status, and hid a tie at distance one. All three were repaired before the record was committed.
- One probe sent a browser user agent to the three blocked pages. It was abandoned; no fetched byte
  came through it.

## what this survey does not do

It does not claim the search was exhaustive: one engine returned nothing, another left six queries
unanswered, and <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/n_leads_not_scored" k="numeric">94</sworn> leads are recorded unscored. Among them are four arXiv works below the cap,
including "You've Changed: Detecting Modification of Black-Box Large Language Models" and "Verifying
LLM Inference to Detect Model Weight Exfiltration", and works readers met in the texts, such as a
rank-based uniformity test for auditing black-box APIs. It does not re-open the retired clauses. It
does not say any surviving clause is unoccupied: each is occupied, and the record says by whom. It
does not treat the margin check as a result. The readers' codings, quotes and notes are in the
record for anyone to dispute.

## what a stranger checks

`python -m styxx.sworn verify papers/plates/SURVEY_sand_neighbours_pass4_2026_09_14.md --repo . --commit <the commit that carries this file>`
re-derives every number above from the two records. `python papers/plates/build_sand_survey_pass4.py
papers/plates/sand_survey_pass4_inputs 2026_09_14 5d7f39ef <texts>` rebuilds the record from the
committed search record, fetch records, scope file and reader returns, and the margins builder
rebuilds the margin record. Without the texts, which are other people's work and are not committed,
both builders skip the proofs of reading and the quote checks; `fetch_pass4.py` re-fetches the texts
from the recorded URLs, and the recorded sha256 says whether the bytes are the ones that were read.
