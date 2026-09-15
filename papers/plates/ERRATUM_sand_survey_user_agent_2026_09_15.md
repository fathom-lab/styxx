# ERRATUM — the fetchers of sand survey passes 3 and 4 sent a browser-form user agent on every request; one page read under it refuses a non-browser agent a day later, and no clause status rests on that page

**status: an erratum beside `SURVEY_sand_neighbours_pass3_2026_09_14.md`, `CORRECTION_sand_neighbours_pass3_2026_09_14.md`,
`SURVEY_sand_neighbours_pass4_2026_09_14.md`, `CORRECTION_sand_neighbours_pass4_2026_09_15.md` and
`PROTOCOL_sand_pass4_correction_2026_09_15.md`. None of them is edited. This file swears to
`sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json`, which a committed script derives from a re-fetch whose
rules were committed and pushed at
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/frozen_commit" k="quote">`8273cdda`</sworn> before its first request. For the bounty
clause's status it swears to `sand_prior_art_survey_pass4_correction_2026_09_15.json`. No clause changes status. Nothing
here licenses "first", "novel" or "revolutionary".**

## What was wrong

The lab's operating rule is that no request presents as a browser. Neither the pass-3 nor the pass-4 protocol wrote that
rule down. The message of commit `c398e830` stated it first, after pass 3's fetches and pass 4's runs A, B and B2 had
been sent (B2's last request sixteen seconds before the commit). The correction's protocol froze it at `9dbf4092`. No fetcher for passes 1 or 2 is committed, so this erratum
says nothing about how their bytes were requested.

The fetchers of passes 3 and 4 sent a browser-form user agent on every request:
- `fetch_pass3.py`, line <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/0/line" k="numeric">34</sworn>, sent
  <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/0/user_agent" k="quote">`Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab sand survey pass 3 (research; contact via github.com/fathom-lab)`</sworn>.
- `fetch_pass4.py`, line <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/1/line" k="numeric">25</sworn>, sent
  <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/1/user_agent" k="quote">`Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab sand survey pass 4 (research; github.com/fathom-lab)`</sworn>.
  The correction's `prepare_correction_inputs.py` and `located_lookups.py` import that string.

Both strings named the lab and gave a contact. Both also began with the form browsers send, so a site that admits
browsers by that prefix admitted the lab as one. That is presenting as a browser.

**Recorded fetch attempts** sent under those strings:

| record | attempts |
|---|---|
| pass 3 | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass3" k="numeric">17</sworn> |
| pass 4, run A | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_A" k="numeric">9</sworn> |
| pass 4, run B | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_B" k="numeric">28</sworn> |
| pass 4, run B2 | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_B2" k="numeric">11</sworn> |
| pass 4, run B3 | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_B3" k="numeric">1</sworn> |
| pass-4 correction | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/correction" k="numeric">6</sworn> |
| all recorded | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_total" k="numeric">72</sworn>, on <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/distinct_urls" k="numeric">71</sworn> distinct URLs |

**Requests outside those records.**
- **The correction's lookups.** The correction sent the pass-4 string on
  <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/correction_lookup_requests_recorded" k="numeric">5</sworn> recorded catalogue
  lookups (`located_lookups_correction.json`: Europe PMC and Crossref).
- **Unrecorded lookups.** Each run of `prepare_correction_inputs.py` (lines 105 to 112) sent it on six more: two Europe
  PMC searches and four Internet Archive availability queries.
- **A first run left no record.** The script ran at least twice. Commit `8fa0e860` records a first run that accepted,
  for B04, a PDF link redirecting to the landing page pass 4 had already fetched. The script was then changed to refuse
  such a copy and run again. The first run's fetches and lookups are in no committed record.
- **The red team.** Its prompt allowed "the fetcher's identifying user agent" (`redteam_pass4.js`). Its scratch
  scripts sent the pass-4 string, through `fetch_pass4.fetch` or a copy of line 25, to:
  - arXiv 2603.19022;
  - the PDFs of B04, B05 and B07;
  - error.reviews;
  - the two Science DOIs and Science's PDF URL.

  These requests are recorded in `redteam_return.json`, findings `fetch:2`, `fetch:4`, `fetch:6` and `fetch:7`.
- **The correction's review.** It re-ran `curl_json`, which sends the pass-4 string (`review_return.json`).
- **The search stage.** Its prompts called curl without this string.

None of these requests is in the re-fetch below.

**The browser probe.** The probe that the pass-4 SURVEY describes sent a full Chrome user agent that did not name the
lab, on four requests (red-team finding `prose:9`, confirmed). One of them printed error.reviews' links, and those links chose
the About-page route from which B16 was then fetched.

## The statements this makes false or incomplete

1. **SURVEY, pass 4:** "One probe sent a browser user agent to the three blocked pages. It was abandoned; no fetched
   byte came through it." The probe was not the only request in browser form. So was every request by the pass-3 and
   pass-4 fetchers and by the correction's scripts. Bytes did come through the probe: its printout of error.reviews'
   links chose B16's route. That page served the same bytes to the fetcher's own agent (`prose:9`), and it serves the
   same bytes to a non-browser agent today (below).
2. **CORRECTION, pass 4, item 9.** It corrects which pages the probe reached. It keeps "No fetched byte came through the
   probe", and it does not say that every fetch was in browser form.
3. **The pass-4 addendum** in `papers/chat/REPORT_2026_09_13.md` (line 540) repeats the SURVEY's statement.
4. **PROTOCOL of the pass-4 correction:** "No route may present as a browser or pass a block." The correction's own
   fetches and lookups did not keep that rule.
5. **Commit `c398e830`:** "The fetcher keeps its identifying user agent. No route presents as a browser or passes a
   block, …". The identifying agent was in browser form.
6. **The pass-4 red team** (`redteam_return.json`, the fetch dimension): "No browser user agent was sent and no block was
   bypassed." The first half is false. Its "The "Mozilla/5.0" prefix in the fetcher's user agent bypassed nothing"
   agrees with the re-fetch below for 70 of 71 URLs, and cannot be tested for L16.
7. **The correction's review** (`review_return.json`): "The UA is fetch_pass4.py's own, identical to pass 4's." That is
   true, and the review passed a browser form.
8. **The strangers' checks.** Several sworn files send a stranger to `fetch_pass3.py` or `fetch_pass4.py` to re-fetch,
   and those fetchers send the browser form. They are:
   - SURVEY pass 3 (line 116);
   - CORRECTION pass 3 (section 1);
   - SURVEY pass 4 (line 184);
   - CORRECTION pass 4 (line 198).

   A stranger can check the same recorded sha256 values with `refetch_nonbrowser_ua.py`, which sends no browser token.
   The two fetchers are left as they ran.
9. **SURVEY, pass 3:** "Every source was then fetched by `fetch_pass3.py`". That is true. It does not say that the
   fetcher presented as a browser, and this erratum does.

## What the bytes depended on

A re-fetch asked whether any recorded fetch attempt needed the browser token. Its rules were committed and pushed before
its first request:
- one request per URL, with `fathom-lab-sand-survey-ua-check/1 (research; github.com/fathom-lab)`;
- the request otherwise unchanged: `-sSL`, `--max-time 120`, redirects followed, no cookie and no other header (only
  local output options differ);
- no retry;
- seven outcomes, fixed in advance.

It ran from
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/run_first_request_at" k="quote">`2026-09-15T08:28:42Z`</sworn> to
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/run_last_request_at" k="quote">`2026-09-15T08:34:02Z`</sworn>.
The catalogue lookups and the unrecorded requests above were not repeated. Whether their answers needed the browser
token is not tested.

| outcome | URLs |
|---|---|
| same bytes as recorded | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/SAME_BYTES" k="numeric">45</sworn> |
| same kind, other bytes | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/SAME_KIND" k="numeric">10</sworn> |
| kind changed | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/KIND_CHANGED" k="numeric">0</sworn> |
| refused then, answers now | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/NOW_ANSWERS" k="numeric">0</sworn> |
| refused then and now | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/STILL_REFUSED" k="numeric">14</sworn> |
| rate-limited now | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/RATE_LIMITED" k="numeric">1</sworn> |
| answered then, refused now | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/outcomes/NOW_REFUSED" k="numeric">1</sworn> |

- **Same kind.** These are HTML pages whose byte count moved by at most
  <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/same_kind_max_byte_difference" k="numeric">82</sworn> bytes.
- **Refused then and now.**
  - Semantic Scholar's empty 202 pages (6)
  - DOI redirects (3)
  - OpenReview PDFs (2)
  - MDPI (1)
  - Science (1)
  - one Internet Archive miss

  The browser form got nothing from them either.
- **Rate-limited.** The URL is morphllm's page (B13). It answered 429 both times. Its text came from an Internet
  Archive capture, which returned the same bytes today.
- **error.reviews.** Its home page returned the same bytes today. So B16's route can be read from bytes a non-browser
  agent receives.

**The one page that now refuses is**
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/id" k="quote">`L16`</sworn>, ACM's *Artifact Review and Badging* policy.
On 2026-09-14 it answered
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/recorded_http" k="quote">`200`</sworn> with
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/recorded_bytes" k="numeric">87457</sworn> bytes. Today it
answers <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/now_http" k="quote">`403`</sworn> to the non-browser agent. Time is
confounded with the user agent, because a site can change its rules in a day. No request was sent in browser form to
tell the two apart, since that would break the rule again. So this erratum cannot say whether L16's bytes needed the
browser token. It says that they may have.

## No clause status rests on L16 or on B16's route

**L16.** It is one of five sources that pass 3 records as occupying the bounty clause: S10, S18, S20, L16 and L17.
Without L16 the clause stays
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/bounty_clause/pass3_status_without_now_refused" k="quote">`OCCUPIED`</sworn> in pass 3.
It was already OCCUPIED in passes 1 and 2, before L16 was listed.

**The later records.** Pass 4 and its correction record the bounty clause as
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C5/status" k="quote">`UNPRICED`</sworn>, because a candidate that could retire it,
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C5/correction/uncheckable/0" k="quote">`B23`</sworn>, is uncheckable. An occupying source
cannot change that, and neither L16 nor B16 retires the clause.

## What follows

- **Pass 5.** Its protocol is not yet frozen. It must fetch with a user agent that carries no browser token. It must
  also say whether bytes or routes the lab holds only through a browser form may be used: L16's text, and the probe's
  link list.
- **Other fetchers.** Some fetchers outside the sand survey also send a browser form, among them the G5 scorecard's
  `card_fetch.py`. They are not covered here, and they are flagged for a separate audit.
- **Lesson.** The rule was checked against the code, and the check passed the code.
  - The pass-4 red team read `fetch_pass4.py`'s user agent. It judged a `Mozilla/5.0` string that names the lab "an
    identifying agent, not a disguised one".
  - The correction's review accepted the same string.
  - The correction's protocol froze the rule in the same commit as that red-team return.

  A rule like this needs a mechanical test, here that the user agent carries no browser token, and not a judgment of
  intent.
