# SURVEY — the neighbours of the sand, pass 3: seventeen of pass 2's leads read end to end, and the sentence survives with three more neighbours inside it

**status: a survey under a frozen protocol, sworn to its record. Pass 3 scores the list that
`PROTOCOL_sand_prior_art_pass3_2026_09_14.md` closed at commit <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/protocol_commit" k="quote">`54b066ff`</sworn>, before any
fetch. No clause changed status. The licensed sentence changed: its fingerprint clause now names
six neighbours instead of three. Nothing here licenses "first", "novel" or "revolutionary".**

## what was done

Pass 2 (`SURVEY_sand_neighbours_pass2_2026_09_13.md`, sworn) closed its list at twenty sources and
recorded forty leads it could not score. The pass-3 protocol drew a closed list of
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/sources_in_list" k="numeric">17</sworn> of them by one stated rule (every lead a reader
named as a candidate for the fingerprint clause, the chained-log clause or the bounty clause,
minus those already priced by a neighbour, those that are evidence about a floor rather than a
practice, and those beside a retired clause), stated the fingerprint clause's object before
reading, and was committed before the first fetch. Every source was then fetched by
`fetch_pass3.py`, its bytes hashed (`sand_survey_fetch_record_pass3_2026_09_14.json`), its full
text extracted, and read end to end by one of four readers — <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/read" k="numeric">17</sworn> READ, none skimmed,
none unfetchable, <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/chars_read_total" k="numeric">1035862</sworn> characters of extracted text in all
(the longest, LLMmap, <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/sources/L09/chars_read" k="numeric">122531</sworn>). Each reader returned, per source, one verdict
under the inherited rule (RETIRES / OCCUPIES / SILENT), the object, one verbatim quote, one sentence
of reason, and for the fingerprint clause five yes/no elements of the object stated in advance.
`build_sand_survey_pass3.py` merged the returns; the surveyor typed no verdict. The builder also
checked every verdict quote against the extracted text after normalising ligatures, quote marks and
line-end hyphens: <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/verdict_quotes_found_in_text" k="numeric">17</sworn> of
<sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/verdict_quotes_checked" k="numeric">17</sworn> are in the text they are attributed to.

## what it found

- Verdicts: <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/verdicts_retires" k="numeric">0</sworn> RETIRES,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/verdicts_occupies" k="numeric">17</sworn> OCCUPIES,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/verdicts_silent" k="numeric">0</sworn> SILENT. Every source on the list does part of what
  a clause says, for a different object or without the floor; none does the whole of it for the
  same object.
- The fingerprint clause (C4a) stays <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/status" k="quote">`OCCUPIED`</sworn>. Of its five elements, stated
  before reading — a served model compared with its own earlier or differently-served self; a
  fixed item set; a floor measured on the same weights under the serving in use; an interval on
  the comparison; log-probabilities as the object — the fifteen sources read for it have
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/elements_true_counts/self_comparison" k="numeric">6</sworn> with the self-comparison,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/elements_true_counts/fixed_item_set" k="numeric">11</sworn> with a fixed item set,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/elements_true_counts/same_weights_floor_in_situ" k="numeric">1</sworn> with a same-weights floor measured in situ
  (ChatLog, on a black box whose weights are inferred unchanged, not known),
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/elements_true_counts/interval_on_comparison" k="numeric">5</sworn> with an interval on the comparison,
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/elements_true_counts/log_prob_object" k="numeric">3</sworn> with a log-probability object, and
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/n_sources_with_all_five_elements" k="numeric">0</sworn> with all five. The nearest, by the
  readers' own flag, are <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C4a/nearer_pass3_joined" k="quote">`L03, L06, L08`</sworn> —
  ChatLog (Tu et al.: the same served API against its own earlier self on a fixed item set,
  graded by z-test against the within-period spread, no interval, ROUGE on sampled text),
  Hooker et al. (compressed against uncompressed on a fixed set, graded by Welch's t-test against
  a thirty-seed population null, labels not probabilities), and Madaan et al. (a measured seed
  variance that a comparison must exceed, with bootstrapped intervals, on different weights).
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/nearer_neighbours_named" k="numeric">3</sworn> neighbours enter the sentence.
- The two leads pass 2 named first read as the protocol predicted and no nearer: Chen, Cai,
  Zaharia and Zou 2021 <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/sources/L01/verdicts/C4a/verdict" k="quote">`OCCUPIES`</sworn>
  (a year-over-year confusion-matrix shift on a fixed benchmark, an interval on the estimator's
  sampling error, judged against a bare 1% threshold, never a same-version re-query; bytes
  <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/sources/L01/sha256" k="quote">`a213e5343771fb563b28a65a0177f8ee429dc9989deda77e64d261c456b9d8c5`</sworn>);
  Yang and Wu 2024 <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/sources/L04/verdicts/C4a/verdict" k="quote">`OCCUPIES`</sworn>
  (ownership: whether a suspect's logits lie in the victim's subspace, a fixed tolerance, no
  time series). The provenance and ownership tools (LLMmap, Hide and Seek, Dataset Inference,
  DeepJudge) identify which model; the nondeterminism papers (Atil et al.) measure the floor and
  grade nothing against it; SafetyNets proves exact execution and says itself it cannot see the
  model's own errors; the extraction paper bounds fidelity by a floor across re-trainings.
- The bounty clause (C5) stays <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C5/status" k="quote">`OCCUPIED`</sworn>: ACM's badges reward authors
  for agreement within tolerance, adjudicated by editors, with no payment; the 2019 challenge
  editorial recognizes peer-reviewed reports by publication, with compute credits given before
  the work. Neither is nearer than Immunefi or the Preregistration Challenge.
- The chained-log clause (C2) stays <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/clauses/C2/status" k="quote">`OCCUPIED`</sworn>; no source on this list was
  a candidate for it, and the protocol says why.
- <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/counts/leads_not_scored" k="numeric">11</sworn> leads were met while reading and are recorded,
  not scored — the list was closed. Three are the same nondeterminism-measurement class as Atil
  et al., two are ownership fingerprints beside DeepJudge, one is a hash-chained ownership
  fingerprint (Russinovich and Salem, "chain & hash"), one is pass 2's own source in preprint
  form.

## the sentence

The record's status is <sworn r="path:papers/plates/sand_prior_art_survey_pass3_2026_09_14.json#/sentence/status" k="quote">`SURVIVES_WITHOUT_C3_C4b`</sworn> — the same four
clauses as after pass 2, the face and the seal still out (Perrig and Song 1999; Haber and Stornetta
1991, OpenTimestamps). The text differs from pass 2's by the three neighbours appended inside the
fingerprint clause, each by the phrase its reader returned; the builder keeps pass 2's wording for
everything else and refuses to run if that wording is not found verbatim. Copied from the record's
`sentence.text`:

> We know of no lab that binds every published number to bytes at a commit (where Deterministic
> Integrity Gates and Cited-but-Not-Verified bind sentences and citations of a manuscript, as the
> 2026-09-05 survey names them), re-derives every verdict from those bytes into a chained log (where
> Certificate Transparency and Rekor chain certificates and signed metadata, not verdicts),
> fingerprints a model's behavior on hashed canaries against a measured null floor (where Dutta et
> al. measure flips and KL divergence under compression, Chen, Zaharia and Zou grade a service's
> drift against its own repeat-run disagreement, and Thinking Machines measure the same-weights
> spread of served completions, and Tu et al. grade successive-period ROUGE means of gpt-3.5-turbo
> against within-period day-to-day spread by z-test, Hooker et al. grade compressed-vs-base class
> recall against a 30-seed population null by Welch's t-test, Madaan et al. grade benchmark score
> differences against measured seed variance and bootstrapped 95% intervals, on different weights),
> and pays a standing bounty against its own verifier (where Immunefi pays against deployed code and
> the Preregistration Challenge paid for preregistering) — at once.

That is the only positioning sentence the lab may say about the sand, and it is now longer, not
shorter: every pass adds neighbours and none has removed the conjunction. Thirty-seven sources
across three passes have been read end to end or skimmed against it.

## what this survey does not do

It does not price beyond the list; the eleven leads are the start of the next one. It does not
re-open the retired clauses. It does not say the four surviving clauses are unoccupied — each is
occupied and says by whom. It does not claim the readers are infallible: their verdicts, objects,
quotes, element flags and notes are in the record, source by source, for anyone to dispute; a
dispute that shows a listed source doing all five elements for the same object retires the
fingerprint clause and, under the sentence rule, the sentence.

## what a stranger checks

`python -m styxx.sworn verify papers/plates/SURVEY_sand_neighbours_pass3_2026_09_14.md --repo . --commit <the commit that carries this file>`
re-derives every number above from the record. The record's per-source `sha256` is the hash of
the bytes fetched from the URL beside it; `fetch_pass3.py` re-fetches, though a source may have
moved on (three readers note version drift: ChatLog is v2 retitled, Yang and Wu's v2 is dated
December 2025, Hooker et al. is v3). `python papers/plates/build_sand_survey_pass3.py` rebuilds
the record from the readers' returns and the fetch record; it refuses a return that scores a
clause the list does not name for that source, and it downgrades a RETIRES from a source not read
end to end.
