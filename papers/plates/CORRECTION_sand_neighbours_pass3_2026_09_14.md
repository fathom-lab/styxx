# CORRECTION — SURVEY_sand_neighbours_pass3_2026_09_14: the inputs a stranger was told to rebuild from are in the tree, the counts that were self-reports say so, and the fingerprint clause is priced under the two terms its protocol left undefined

**status: a correction beside a sworn survey, sworn to its own record
(`sand_prior_art_survey_pass3_correction_2026_09_14.json`, written by
`build_sand_survey_pass3_correction.py`). The pass-3 SURVEY, its record and its protocol are not edited.
It follows the lab's red team of 2026-09-14: two findings on the survey's inputs were confirmed by both
skeptics; two on its element coding were judged plausible; two others were refuted and are recorded as
notes. The status of every clause, and the licensed sentence, are unchanged.**

## 1. the inputs a stranger was told to rebuild from

The SURVEY told a stranger that `fetch_pass3.py` re-fetches the sources and that
`build_sand_survey_pass3.py` rebuilds the record "from the readers' returns and the fetch record".
Neither the script nor the returns were in the tree. They are now, under
`papers/plates/sand_survey_pass3_inputs/`:
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/inputs/n_reader_returns" k="numeric">4</sworn>
reader returns, exactly as the builder read them, and the fetch script as it ran. The correction's
builder rebuilt the record from those committed inputs into a scratch file and compared it with the
committed record:
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/rebuild/n_sources_rebuilt" k="numeric">17</sworn>
sources rebuilt, with
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/rebuild/n_top_level_fields_differing" k="numeric">0</sworn>
top-level fields and
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/rebuild/n_source_fields_differing" k="numeric">0</sworn>
source fields differing, apart from the verbatim-quote check.

The quote check — <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/verdict_quotes_found_in_text" k="numeric">17</sworn>
verdict quotes found in the text they are attributed to — cannot be rebuilt from the tree. It read the
extracted full texts, and those are not committed: they are the full text of seventeen publications,
other people's work. Their sha256 are in the fetch record. Re-deriving the check needs a re-fetch and the
same extractor, and the readers' own notes say three of the fetched texts are later revisions than the
versions the list named (ChatLog, Yang and Wu, Hooker et al.), so a later fetch may not reproduce.

## 2. the counts that were self-reports

The SURVEY reported every source as read end to end and
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/self_reports/chars_reported_total" k="numeric">1035862</sworn>
characters read. Both are the readers' reports. Each reader was told its files' sizes, and each reported
exactly those sizes. Nothing measured how much of a text a reader read. The SURVEY's "read end to end" is
the readers' statement, and a stranger should weigh it as one.

## 3. the fingerprint clause, under the two terms the protocol did not define

The pricing rule retires a clause only when a source does "the same thing for the same object". For the
fingerprint clause, the pass-3 protocol stated the OBJECT before reading, as four elements: a served
model compared with its own earlier or differently-served self; a fixed item set; a floor measured on the
same weights under the serving in use; an interval on the comparison. The THING is the clause's
operational row, also frozen before reading: teacher-forced log-probabilities on a fixed, hashed item
set. The SURVEY spoke of "five elements, stated before reading" without saying that four come from the
object paragraph and the fifth from the operational row; both predate any fetch, but the SURVEY should
have said where each came from.

The protocol never defined "an interval on the comparison" or "measured on the same weights", and the
readers coded them differently. One reader counted a one-tailed t-test threshold as an interval
(DeepJudge); another did not count a z-test at a critical value (ChatLog) or a Welch test at a stated level
(Hooker et al.); and one flag (Xu et al.) contradicts its reader's own note that the ± values are not an
interval on the difference. The correction's builder recodes the elements under four stated readings,
each override resting on text the builder asserts is in the record, and counts, per reading, the sources
carrying all four object elements, and those carrying the object and the thing:

- **as recorded**: <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/as_recorded/element_true_counts/interval_on_comparison" k="numeric">5</sworn>
  sources with an interval; <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/as_recorded/n_sources_with_the_four_object_elements" k="numeric">0</sworn>
  with all four object elements.
- **a test on the difference at a stated level is an interval**:
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/a_stated_level_test_on_the_difference_is_an_interval/element_true_counts/interval_on_comparison" k="numeric">7</sworn>
  with an interval; <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/a_stated_level_test_on_the_difference_is_an_interval/n_sources_with_the_four_object_elements" k="numeric">1</sworn>
  with all four object elements —
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/a_stated_level_test_on_the_difference_is_an_interval/sources_with_the_four_object_elements/0" k="quote">`L03`</sworn>,
  ChatLog — and <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/a_stated_level_test_on_the_difference_is_an_interval/n_sources_with_the_object_and_the_thing" k="numeric">0</sworn>
  with the object and the thing.
- **only a confidence interval on the compared quantity is an interval**:
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/only_a_confidence_interval_on_the_compared_quantity_is_an_interval/element_true_counts/interval_on_comparison" k="numeric">3</sworn>
  with an interval; <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/only_a_confidence_interval_on_the_compared_quantity_is_an_interval/n_sources_with_the_four_object_elements" k="numeric">0</sworn>
  with all four object elements.
- **the stated-level reading, with a floor counted only on weights known to be identical**:
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/readings/the_lenient_interval_and_a_floor_only_on_weights_known_identical/n_sources_with_the_four_object_elements" k="numeric">0</sworn>
  with all four object elements.

Under <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/n_readings_under_which_C4a_would_retire" k="numeric">0</sworn>
of the <sworn r="path:papers/plates/sand_prior_art_survey_pass3_correction_2026_09_14.json#/undefined_terms/n_readings" k="numeric">4</sworn>
readings would the clause retire. That is not the margin the SURVEY implied. Under the stated-level reading
ChatLog carries every element of the clause's object: it compares a served model with its own earlier
self, on a fixed item set, against a spread measured within a period, with a test at a stated level. What
it does not carry is the thing: it scores sampled text by ROUGE, not teacher-forced log-probabilities. (The
readers were not asked to code a hashed item set, and none noted one in any source.) The fingerprint clause survives on what it measures
more than on its floor or its interval, and the licensed sentence's neighbour phrase for Tu et al. already
names ROUGE. Anyone using the sentence in public should say that.

## notes the red team raised and this correction does not act on

Two skeptics each refuted the claim that the SURVEY raised the RETIRES bar after reading (the five
elements' sources are stated above) and the claim that two of the three "nearer" flags must be withdrawn
(neither the protocol nor the SURVEY defines nearness by an element count; the SURVEY names the flags as
the readers'). Recorded, not acted on: the SURVEY's "every source on the list does part of what a clause
says" overstates for SafetyNets, whose reader recorded no element of the object; the builder's quote check
counts an empty quote as found (none of the seventeen was empty). The next survey protocol defines "an
interval on the comparison" and "measured on the same weights" before any fetch.

## what a stranger checks

    python papers/plates/build_sand_survey_pass3_correction.py 54b066ff
    python -m styxx.sworn verify papers/plates/CORRECTION_sand_neighbours_pass3_2026_09_14.md --repo . --commit <the commit that carries it>

The first rebuilds the pass-3 record from the committed inputs and rewrites this correction's record; it
refuses to run if the text an override rests on is not in the record. The second re-derives every number
above.
