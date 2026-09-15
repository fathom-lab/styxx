# ERRATUM — the fetchers of sand survey passes 3 and 4 sent a browser-form user agent on every request; one page read under it refuses a non-browser agent a day later, and no clause status rests on that page

**status: an erratum beside `SURVEY_sand_neighbours_pass3_2026_09_14.md`, `SURVEY_sand_neighbours_pass4_2026_09_14.md`,
`CORRECTION_sand_neighbours_pass4_2026_09_15.md` and `PROTOCOL_sand_pass4_correction_2026_09_15.md`. None of them is
edited. This file swears to `sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json`, derived by a committed
script from a re-fetch whose rules were committed and pushed at
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/frozen_commit" k="quote">`8273cdda`</sworn>, before its first request. No clause
changes status. Nothing here licenses "first", "novel" or "revolutionary".**

## What was wrong

The lab's rule is that no fetch presents as a browser. The fetchers of passes 3 and 4 broke it on every request.
No fetcher for passes 1 or 2 is committed, so this erratum says nothing about how their bytes were requested.

- `fetch_pass3.py`, line <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/0/line" k="numeric">34</sworn>, sent
  <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/0/user_agent" k="quote">`Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab sand survey pass 3 (research; contact via github.com/fathom-lab)`</sworn>.
- `fetch_pass4.py`, line <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/1/line" k="numeric">25</sworn>, sent
  <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/fetchers/1/user_agent" k="quote">`Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab sand survey pass 4 (research; github.com/fathom-lab)`</sworn>.
  The correction's `prepare_correction_inputs.py` and `located_lookups.py` import that string.

Both strings named the lab and gave a contact. Both also began with the form browsers send, so a site that admits
browsers by that prefix admitted the lab as one. That is presenting as a browser.

The recorded fetch attempts sent under those strings:

| record | attempts |
|---|---|
| pass 3 | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass3" k="numeric">17</sworn> |
| pass 4, run A | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_A" k="numeric">9</sworn> |
| pass 4, run B | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_B" k="numeric">28</sworn> |
| pass 4, run B2 | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_B2" k="numeric">11</sworn> |
| pass 4, run B3 | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/pass4_run_B3" k="numeric">1</sworn> |
| pass-4 correction | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_by_record/correction" k="numeric">6</sworn> |
| all | <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/attempts_total" k="numeric">72</sworn>, on <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/distinct_urls" k="numeric">71</sworn> distinct URLs |

The correction also sent the pass-4 string on
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/correction_lookup_requests_recorded" k="numeric">5</sworn> recorded catalogue lookups
(`located_lookups_correction.json`: Europe PMC and Crossref). It sent it on six unrecorded lookups in
`prepare_correction_inputs.py` (lines 105 to 112): two Europe PMC searches and four Internet Archive availability
queries. The search stage's prompts do not use the string.

## The statements this makes false or incomplete

1. **SURVEY, pass 4:** "One probe sent a browser user agent to the three blocked pages. It was abandoned; no fetched
   byte came through it." The probe was not the only request in browser form. Every fetch was. The CORRECTION beside
   pass 4 (its item 9) and the pass-4 addendum in `papers/chat/REPORT_2026_09_13.md` repeat the statement without
   saying so.
2. **PROTOCOL of the pass-4 correction:** "No route may present as a browser or pass a block." The correction's own
   fetches and lookups did not keep that rule.
3. **SURVEY, pass 3:** "Every source was then fetched by `fetch_pass3.py`". That is true. It does not say that the
   fetcher presented as a browser, and this erratum does.

## What the bytes depended on

A re-fetch asked whether any recorded fetch needed the browser token. Its rules were committed and pushed before its
first request:
- one request per URL, with `fathom-lab-sand-survey-ua-check/1 (research; github.com/fathom-lab)`;
- nothing else changed in the curl call;
- no retry;
- seven outcomes, fixed in advance.

It ran from
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/run_first_request_at" k="quote">`2026-09-15T08:28:42Z`</sworn> to
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/run_last_request_at" k="quote">`2026-09-15T08:34:02Z`</sworn>.

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
- **Refused then and now.** These are Semantic Scholar's empty 202 pages (6), DOI redirects (3), OpenReview PDFs (2),
  MDPI (1), Science (1) and one Internet Archive miss. The browser form got nothing from them either.
- **Rate-limited.** The URL is morphllm's page (B13). It answered 429 both times. Its text came from an Internet
  Archive capture, which returned the same bytes today.

**The one page that now refuses is**
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/id" k="quote">`L16`</sworn>, ACM's *Artifact Review and Badging* policy.
On 2026-09-14 it answered
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/recorded_http" k="quote">`200`</sworn> with
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/recorded_bytes" k="numeric">87457</sworn> bytes. Today it
answers <sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/now_refused/0/now_http" k="quote">`403`</sworn> to the non-browser agent. Time is
confounded with the user agent, because a site can change its rules in a day. No request was sent in browser form to
tell the two apart, since that would break the rule again. So this erratum cannot say whether L16's bytes needed the
browser token. It says that they may have.

**L16 bears on no status.** It is one of five sources that pass 3 records as occupying the bounty clause: S10, S18,
S20, L16 and L17. Without L16 the clause stays
<sworn r="path:papers/plates/sand_survey_user_agent_inputs/refetch_nonbrowser_summary.json#/bounty_clause/pass3_status_without_now_refused" k="quote">`OCCUPIED`</sworn> in pass 3.
It was already OCCUPIED in passes 1 and 2, before L16 was listed. The pass-4 correction records the bounty clause as
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C5/status" k="quote">`UNPRICED`</sworn>, because a candidate that could retire it,
<sworn r="path:papers/plates/sand_prior_art_survey_pass4_correction_2026_09_15.json#/clauses/C5/correction/uncheckable/0" k="quote">`B23`</sworn>, is uncheckable. An occupying source cannot change
that. No clause status in any record rests on L16.

## What follows

- **Pass 5.** Its protocol is not yet frozen. It must fetch with a user agent that carries no browser token. It must
  also say whether bytes received only in browser form, such as L16's, may be read.
- **Other fetchers.** Some fetchers outside the sand survey also send a browser form. One is the G5 scorecard's
  `card_fetch.py`. They are not covered here, and they are flagged for a separate audit.
- **Lesson.** A rule written into a protocol is a test to run against the code that carries it out. "No route may
  present as a browser" was frozen while the fetcher it governed sent `Mozilla/5.0`.
