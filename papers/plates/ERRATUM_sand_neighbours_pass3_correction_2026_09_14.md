# ERRATUM — CORRECTION_sand_neighbours_pass3_2026_09_14: the fingerprint clause's margin is per source, not "what it measures"

**status: an erratum to a sworn correction, sworn to its own record
(`sand_prior_art_survey_pass3_margins_2026_09_14.json`, written by `build_sand_survey_pass3_margins.py` from the
pass-3 record). The correction is not edited. It follows the lab's verification of the night of 2026-09-14,
whose finding on this sentence both skeptics confirmed. The status of every clause and the licensed sentence are
unchanged; what changes is what the lab may say about the fingerprint clause's margin.**

## what was wrong

The correction said the fingerprint clause "survives on what it measures more than on its floor or its interval".
The lab's operator REPORT said more: that the sentence survives on what the instrument measures, not on its floor,
and that anyone using it in public should say so. The record does not support either. One source carries the
self-comparison, the fixed item set, the interval and the log-probability object, and misses only the floor.

## the margins, per source and per reading

The five elements are the four of the clause's object, as the pass-3 protocol stated them before reading, and the
thing of its operational row, teacher-forced log-probabilities. The readings are the correction's four, plus the
one the verification noted was missing: the lenient interval with a lenient floor, under which a measured null that
grades the comparison counts as a floor whether or not it was measured on the same weights. Every override rests on
text the builder asserts is in the record.

Under <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/n_readings_with_a_source_missing_nothing" k="numeric">0</sworn>
of the <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/n_readings" k="numeric">5</sworn>
readings does any listed source carry all five elements. The fewest elements any source misses, under any reading,
is <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/fewest_missing_under_every_reading" k="numeric">1</sworn>.
Which element the nearest sources miss:

- **as recorded**: <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/as_recorded/sources_missing_the_fewest/0" k="quote">`L05`</sworn>,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/sources/L05" k="quote">`Xu et al.`</sworn>, alone, missing only the
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/as_recorded/what_the_nearest_miss/L05" k="quote">`floor`</sworn>.
- **a test on the difference at a stated level is an interval**:
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/sources/L03" k="quote">`Tu et al.`</sworn> (ChatLog) missing only the
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/a_stated_level_test_on_the_difference_is_an_interval/what_the_nearest_miss/L03" k="quote">`log-probabilities`</sworn>,
  and Xu et al. missing only the
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/a_stated_level_test_on_the_difference_is_an_interval/what_the_nearest_miss/L05" k="quote">`floor`</sworn>.
- **only a confidence interval on the compared quantity is an interval**: no source misses fewer than
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/only_a_confidence_interval_on_the_compared_quantity_is_an_interval/fewest_missing" k="numeric">2</sworn>
  elements.
- **the stated-level interval, with a floor only on weights known to be identical**: Xu et al. alone, missing only the
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/the_lenient_interval_and_a_floor_only_on_weights_known_identical/what_the_nearest_miss/L05" k="quote">`floor`</sworn>.
- **the stated-level interval, with any measured null that grades the comparison counted as a floor**:
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/the_lenient_interval_and_any_measured_null_that_grades_the_comparison_as_a_floor/n_sources_missing_the_fewest" k="numeric">4</sworn>
  sources miss one element each. Tu et al. and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/sources/L06" k="quote">`Hooker, Courville, Clark, Dauphin & Frome`</sworn>
  miss the <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/the_lenient_interval_and_any_measured_null_that_grades_the_comparison_as_a_floor/what_the_nearest_miss/L06" k="quote">`log-probabilities`</sworn>.
  Xu et al. misses the floor.
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/sources/L08" k="quote">`Madaan et al.`</sworn>
  misses the <sworn r="path:papers/plates/sand_prior_art_survey_pass3_margins_2026_09_14.json#/readings/the_lenient_interval_and_any_measured_null_that_grades_the_comparison_as_a_floor/what_the_nearest_miss/L08" k="quote">`self-comparison`</sworn>.

## what the lab may say instead

No listed source carries all five elements of the fingerprint clause. Which element sets the clause apart from its
nearest neighbours depends on the neighbour, and on how the protocol's two undefined terms are read. Against ChatLog
and Hooker et al. it is teacher-forced log-probabilities. Against Xu et al., who grade compressed BERT against its
uncompressed self by label and probability loyalty, it is a floor measured on the same weights. Under the most
lenient reading, against Madaan et al., it is a comparison of a model with its own self. No single element carries
the margin; the conjunction does.

The pass-3 licensed sentence already names Tu et al., Hooker et al. and Madaan et al. inside the fingerprint clause.
It does not name Xu et al., because its reader did not flag that source as nearer. By this count Xu et al. is among
the nearest under every reading, and the next survey's sentence names it.

## a claim the correction made that the tree did not show

The correction said each reader "was told its files' sizes". The prompts were not in the tree. Their size lines are
now committed verbatim in `sand_survey_pass3_inputs/reader_prompt_size_lines.md`; the full prompts are in the lab's
session record, not in this repository.

## what a stranger checks

    python papers/plates/build_sand_survey_pass3_margins.py
    python -m styxx.sworn verify papers/plates/ERRATUM_sand_neighbours_pass3_correction_2026_09_14.md --repo . --commit <the commit that carries it>

The first rewrites the margins record from the pass-3 record and refuses to run if the text an override rests on
is not there. The second re-derives every value above.
