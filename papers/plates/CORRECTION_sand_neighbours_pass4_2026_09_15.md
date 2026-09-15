# CORRECTION — pass 4 of the sand survey: a red team confirmed 33 of 44 findings, the record is rebuilt under a frozen correction, Xu et al. leaves the sentence, and one of the four sources now named rests on a single reader

**status: a correction beside `SURVEY_sand_neighbours_pass4_2026_09_14.md`, which is sworn and is not edited. This
file swears to `sand_prior_art_survey_pass4_correction_2026_09_15.json` and to the margin record beside it, built
under `PROTOCOL_sand_pass4_correction_2026_09_15.md`. That protocol was committed and pushed at
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/protocol_commit" k="quote">`9dbf4092`</sworn>, before any fetch or reading under it. No clause changes status: the fingerprint
clause stays <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C4a/status" k="quote">`OCCUPIED`</sworn>, the bounty clause stays
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C5/status" k="quote">`UNPRICED`</sworn>, and the sentence stays
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sentence/status" k="quote">`SURVIVES_WITHOUT_C5`</sworn>. What changes is who the sentence names, several counts, and a
list of statements in the SURVEY that were false or overstated. Nothing here licenses "first", "novel" or
"revolutionary".**

## why this exists

The lab's rule is to red-team its own day. After pass 4 was pushed at `3060ffd2`, seven finders attacked it: the
prose, the builder, the search, the fetches, the timeline, and two blind re-codings. One or two skeptics then tried
to refute each finding. Of 44 findings, 33 were confirmed, 9 refuted and 2 split. Every finding and verdict is
committed in `sand_survey_pass4_inputs/redteam/redteam_return.json`.

The confirmed findings could not be answered by editing the sworn SURVEY. A correction protocol was therefore frozen
first. It named the sources to be re-read and the prompts to use, verbatim the pass-4 prompts. It also fixed eight
repaired rules for a new builder. That builder was then attacked in turn, before its first run on real reader
returns. Of that review's findings, 14 were confirmed and repaired, each with a test; they are committed in
`sand_survey_pass4_correction_inputs/review/review_return.json`. The correction record is the repaired builder's
mechanical output. The surveyor typed no verdict.

## what changes in the record

- **Two sources recorded READ were not the article.** Pass 4 counted
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/read" k="numeric">30</sworn> READ and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/counts/skimmed" k="numeric">3</sworn> SKIMMED.
  - B11's fetched text was alphaXiv's page for the paper: its abstract and a machine-written overview. One element
    quote came from that overview.
  - B12's was Nature's paywall page.

  Under the correction every listed source whose text was not the article was located once by open routes:
  - B11 is now read from its arXiv paper, <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B11/status" k="quote">`READ`</sworn>, at distance
    <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B11/verdicts/C4a/distance" k="numeric">2</sworn>.
  - B05 and B07, marked SKIMMED in pass 4, are now read from their open PDFs: <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B05/status" k="quote">`READ`</sworn> and
    <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B07/status" k="quote">`READ`</sworn>.
  - B04's PDF link redirects to the same landing page, and Europe PMC holds no open copy. It stays
    <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B04/status" k="quote">`SKIMMED`</sworn>.
  - B12 has no open route and is now <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B12/status" k="quote">`SKIMMED`</sworn>.

  Pass 4's list now reads <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/pass4_list_status/READ" k="numeric">31</sworn> READ,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/pass4_list_status/SKIMMED" k="numeric">2</sworn> SKIMMED and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/pass4_list_status/UNFETCHABLE" k="numeric">1</sworn> UNFETCHABLE.
- **Xu et al. was never at distance one under the pass-4 definitions.** The pass-4 record named it as the source lacking only the same-weights floor
  (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/clauses/C4a/nearness/by_element/E3/id" k="quote">`pass3:L05`</sworn>). That rested on pass 3's flag for an interval, which was a mean and
  standard deviation over three runs. Pass 3's flags were coded before the terms were defined and quote no element,
  and the frozen E4 definition excludes a spread of repeated runs. Re-read under the pass-4 definitions, Xu et al. is
  at distance <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/L05/verdicts/C4a/distance" k="numeric">2</sworn>, lacking the floor and the interval.
- **The earlier passes' fingerprint sources are now coded by element.** The pass-4 protocol counts nearness over
  "the listed source (from any pass)". Passes 1 and 2 had never been coded that way, and pass 3 had been coded under
  undefined terms. All <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/recoded_earlier_sources" k="numeric">23</sworn> were re-read from the bytes their passes hashed, and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/recoded_status/READ" k="numeric">23</sworn> are READ. The pool now holds
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_pool" k="numeric">54</sworn> sources coded under the definitions. Across them,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_carried/E1" k="numeric">27</sworn> carry the self-comparison,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_carried/E2" k="numeric">41</sworn> a fixed item set,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_carried/E3" k="numeric">6</sworn> a same-weights floor used to grade,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_carried/E4" k="numeric">11</sworn> an interval on the comparison and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_carried/E5" k="numeric">15</sworn> log-probabilities. The fewest elements any of them lacks is
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_fewest_missing" k="numeric">1</sworn>. No source in the pool lacks only the self-comparison, only the fixed item set or only
  the floor; the record's `at_distance_one_by_element` lists none.
- **Four sources are at distance one, and all four are named.** There are <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/c4a_at_distance_one" k="numeric">4</sworn>. The pass-4 builder broke a tie by id, a rule no protocol states. The
  correction names every tied source:
  - DiFR (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C4a/nearness/named_in_sentence/0" k="quote">`B14`</sworn>) and Chauvin et al. (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C4a/nearness/named_in_sentence/1" k="quote">`B22`</sworn>), without an interval on the comparison;
  - Gao, Liang and Guestrin (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C4a/nearness/named_in_sentence/2" k="quote">`B01`</sworn>) and Hochlehnert et al. (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C4a/nearness/named_in_sentence/3" k="quote">`S17`</sworn>), without log-probabilities.
- **Hochlehnert et al. is named on one reader's coding.** Their "A Sober Look at Progress in Language Model
  Reasoning" is at distance <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/S17/verdicts/C4a/distance" k="numeric">1</sworn>. Its first reader took the interval from paired t-tests of RL-trained models against their base models,
  a comparison between different models. The paper's comparison of identical checkpoints across hardware and
  frameworks reports no interval or test. Four more blind readers re-coded it, and all
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_15_correction.json#/sources/S17/n_counted" k="numeric">4</sworn> passed both proofs of reading. Of those:
  - <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_15_correction.json#/sources/S17/element_true_counts/E4" k="numeric">1</sworn> finds the interval;
  - <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_15_correction.json#/sources/S17/element_true_counts/E3" k="numeric">1</sworn> finds the floor;
  - <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_15_correction.json#/sources/S17/element_true_counts/E5" k="numeric">0</sworn> find log-probabilities.

  All <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_15_correction.json#/sources/S17/quotes_true_found" k="numeric">9</sworn> of their quotes for elements coded true are found. The frozen rule prices
  from the first reading, so Hochlehnert et al. is named. Four of the five readings put it at distance two or more.
- **Chauvin et al.'s margin is unchanged.** One of <sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B22/n_counted" k="numeric">4</sworn> blind
  readers in pass 4's margin check (<sworn r="path:papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json#/sources/B22/element_true_counts/E4" k="numeric">1</sworn>) read its bootstrap intervals on detection AUC as an
  interval on the comparison, and so coded all five elements.
- **The other clauses hold.** The chained-log clause stays <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C2/status" k="quote">`OCCUPIED`</sworn>; B05 and B07, now read in full, each chain
  their own outputs and re-derive no verdict. The bounty clause stays UNPRICED by
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sentence/unchecked_candidates_that_could_retire_the_sentence/0" k="quote">`B23`</sworn>, which is still
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/sources/B23/status" k="quote">`UNFETCHABLE`</sworn>. B23 is Science's news article: online DOI 10.1126/science.zh9l2q0, and a
  retitled print version, 10.1126/science.aeh8588, PMID 41955357, which is not open. The SURVEY did not say that
  B23, if read, could retire the whole sentence, not only the bounty clause.
- **The conjunction is recorded.** Pass 4 never recorded it. It is
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/conjunction/status" k="quote">`RETIRED`</sworn>, as in passes 1 to 3, because the face and the seal are retired.
- **Checks.** Of <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/element_quotes_coded_true" k="numeric">100</sworn> element quotes coded true,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/element_quotes_found" k="numeric">100</sworn> are found in the text that was read. No reader coded all five elements or wrote
  RETIRES, so <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/confirmations" k="numeric">0</sworn> confirmations ran.
- **Leads.** Pass 4 reported <sworn r="path:papers/plates/sand_prior_art_survey_pass4_2026_09_14.json#/n_leads_not_scored" k="numeric">94</sworn> "leads not scored". Those were raw citation strings, many naming
  sources already listed. The correction records <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/leads_raw" k="numeric">145</sworn> raw strings, of which
  <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/leads_naming_a_listed_source" k="numeric">43</sworn> name a listed source. The record keeps the rest, with a merge rule that
  still leaves some differently-formatted citations of one work apart. No count of distinct unscored works is sworn.

## the sentence

Copied from the correction record's `sentence.text`. "Karvonen et al." for DiFR is its reader's phrase.

> We know of no lab that binds every published number to bytes at a commit (where Deterministic Integrity Gates
> and Cited-but-Not-Verified bind sentences and citations of a manuscript, as the 2026-09-05 survey names them),
> re-derives every verdict from those bytes into a chained log (where Certificate Transparency and Rekor chain
> certificates and signed metadata, not verdicts), and fingerprints a model's behavior on hashed canaries against a
> measured null floor (where Karvonen et al. score served tokens against a trusted same-weights reference's logits,
> calibrated on benign hardware noise, without an interval on the comparison; where Chauvin et al. track first-token
> logprobs of LLM API endpoints to detect changes against their own earlier samples, without an interval on the
> comparison; where Gao, Liang & Guestrin test sampled completions from LLM APIs against reference weights with
> calibrated MMD two-sample tests, without log-probabilities; where Hochlehnert et al. measure seed and hardware
> variance of identical checkpoints on fixed math benchmarks, without log-probabilities) — at once.

The fingerprint clause survives because no reading carries all five elements for one comparison. Its margin is one
element. Of the four sources named:
- two lack only the interval, and one blind reader in five found the interval in Chauvin et al.;
- two lack only log-probabilities, and one of them, Hochlehnert et al., is at distance one in one reading of five.

## what the SURVEY said that was false or overstated

1. **"each written before the result it could have bent was known".** False for these items; times are UTC.
   - The midpoint check became a requirement at 21:08:56. The builder then failed three Part A readings by
     character position at 21:09:33. The unit was switched to line number at 21:10:34, and all nine passed at
     21:10:56. The commit `450265ff` followed at 21:11:15; its title, "committed before the builder runs on any
     return", is also wrong. The unit decides the status of 11 of the 33 first readings, and whether DiFR is in the
     pool.
   - The first full build ran at 22:25:47. The repairs to the doubled "without", the status and the tie followed,
     and the record was committed at 22:28:34 (`a1886191`).
   - The margin prompt was written at 22:26:05, after that first build.
   - Semantic Scholar's HTTP-200 "total 0" answers for Q11 and Q12 were in hand at 21:18:59, before the rule that
     counts them as answered.

   The true statement is narrower: every rule was fixed before the Part B list existed (`178c2d6a`, 22:13:45) or
   before the record was committed, and some after the result they touched had been seen.
2. **"three sources lack exactly one of its five defined elements" and "the three nearest sources are audits of
   served models, not studies of compression or drift".** The pass-4 record put four sources at distance one,
   including Xu et al.'s compression study. The correction puts four other sources there. Three are audits of served
   models; Hochlehnert et al. is a study of evaluation variance.
3. **"each source at distance one was read again by four further readers".** Xu et al. was not. It is now at
   distance two, and the one source newly at distance one, Hochlehnert et al., has had its four readers.
4. **"No source in any pass is at distance one without the self-comparison or without the fixed item set".** Passes
   1 and 2 had not been coded by element, so the scope was not established. It is now: the correction record lists no
   such source among the 54 coded.
5. **"It is shorter than pass 3's for the first time".** False. Pass 2's sentence was already shorter than pass 1's.
6. **"That rule put five 2026 intrusion-detection papers on the list".** Four are intrusion-detection papers. B04,
   BlockFedX, is a federated-learning system for fraud detection, brain-tumour images and plant disease.
7. **"the frozen tie-break by later year".** The frozen rule is "later year". Ranking a missing year as oldest was
   the search workflow's choice, not the protocol's. The protocol also asks for a year for every result, and the web
   engine gave none. If each arXiv work took the year in its identifier, four works would enter the list and four
   leave; "You've Changed: Detecting Modification of Black-Box Large Language Models" would be among those entering.
8. **The screen and the merge.** Two list slots rest on a screener's inclusion whose own reason concedes the rule
   was not met or was deferred: B08 ("close enough to (c)") and B21 (document type "left for the next step"). One
   screener role also treated B13 inconsistently. The merge joined two different web pages (FutureAGI and IBM) and
   missed one (a Bytez page for arXiv 2504.12335). The two errors cancel in the count of 108 works.
9. **"One probe sent a browser user agent to the three blocked pages".** One of the three, error.reviews, was not
   blocked. The probe's own printout of its links supplied the About-page route used for B16. No fetched byte came
   through the probe.
10. **The description of Chauvin et al.** The identical-weights null grades the local comparison of a model with its
    fine-tuned, pruned or noised variants. Real API endpoints compared with their own earlier samples are flagged by a
    running 12-standard-deviation threshold with no stated level.
11. **"94 leads are recorded unscored".** These were raw strings, including sources already scored; see above.
12. **The pass-4 builder applied rules its protocol does not state.**
    - It priced a clause only from sources listed for it.
    - It never computed the conjunction.
    - It never used its own quote check.
    - It broke ties by id and invented a fallback.
    - It judged skimmed sources more strictly than the inherited rule.

    Its proofs of reading can be passed without reading. No committed reading used that gap: every midpoint quote is
    a whole line in the middle third. But <sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/counts/superseded_readings" k="numeric">3</sworn> readings were superseded by
    readings of located copies, and the record lists every reading whose end-of-text proof is a bare page number.
13. **The located retries.** They went only to sources the fetcher failed, not to texts that passed the title check
    but were not the article. B23's attempt recorded no identifiers. The Europe PMC lookups failed silently; they are
    re-run with raw answers in `located_lookups_correction.json`.
14. **The protocol's date.** `PROTOCOL_sand_prior_art_pass4_2026_09_15.md` is dated 2026-09-15, but it was frozen
    and pushed on 2026-09-14 (20:49 UTC).

## what this correction does not do

It adds no source to any list and does not change pass 4's search, rank or cap. The null-year ranking, the two
screener inclusions and the merge errors stand in the list as committed. They are disclosed here and are for pass
5's protocol. It does not tighten the proofs of reading retroactively. It does not settle whether an interval on a
detector's performance is an interval on the comparison, the question the fingerprint clause's margin now turns on.

## what a stranger checks

`python -m styxx.sworn verify papers/plates/CORRECTION_sand_neighbours_pass4_2026_09_15.md --repo . --commit <the commit that carries this file>`
re-derives every number and status above from the two records it names.

The correction record rebuilds from the committed inputs:

```
python papers/plates/build_sand_survey_pass4_correction.py papers/plates/sand_survey_pass4_inputs papers/plates/sand_survey_pass4_correction_inputs 2026_09_15 9dbf4092 <pass-4 texts> <correction texts>
```

`build_sand_survey_pass4_margins.py` rebuilds the margin record. The texts are other people's work and are not
committed. `prepare_correction_inputs.py` re-extracts the 23 earlier texts from the bytes their passes hashed, and
`fetch_pass4.py` re-fetches the rest; the recorded sha256 values say whether the bytes are the ones that were read.
