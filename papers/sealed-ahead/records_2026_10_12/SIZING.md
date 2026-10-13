# Sizing week 2026-09-28: agent pull requests and accusations, counted

Fathom Lab, run 2026-10-12 16:34Z to 2026-10-13 01:42Z. Counts only. Nothing was adjudicated, no precision is computed, and nothing here is tabled by repository, account or product. Agent family appears once, as a sizing count of admitted records. The numbers are in `sizing_week_2026_09_28.json`; the plan, the script and the parameters are under `sizing/`.

## In brief

- **The archive week.** All 168 GH Archive hours from 2026-09-28T00 to 2026-10-05T00 UTC were streamed and never stored. Each answered on its opening attempt: no 404, no failed hour. They held 3,376,781,443 compressed bytes, 15,617,520,367 decompressed bytes and 13,765,478 lines (all valid JSON).
- **Agent pull requests per week.** 30,435 opened pull requests matched the identity table. After the planting exclusions (`young_days` 30), 18,098 were admitted, with no duplicates.
- **A re-derivation agrees.** The builder's `stranger.py pool` streamed the 168 hours again, uncached and without this run's observers. It reproduced the pool byte for byte: pool_sha256 `9fdbd941...`, every hour's admitted digest and compressed sha256, verdict AGREES.
- **The sample.** 300 records were drawn by a seed committed before the week was streamed. All 300 reached a final outcome under the 50-an-hour limit: 190 gated, 100 head moved, 8 gone, 2 empty bodies. The run made 490 requests in 9 h 01 min.
- **Accusations.** 0 of 300. For every kind, and for any kind, that is 0 per 1,000 admitted pull requests, with a Wilson 95% interval of 0 to 12.6. Per 1,000 gated pull requests it is 0, interval 0 to 19.8.
- **Projected accusations per 7-day window.** 0 at the point estimate. At the Wilson upper end, at most 229 accused records a week.
- **Window for 60 accusations at the lower end: unbounded.** The sample cannot exclude a rate of zero. A count of zero is also what the study's AIDev rate predicts: at 0.13% per pull request, 300 records give zero accusations with probability 0.68. Since this sample cannot size the window, the AIDev rate is the only usable prior. At that rate the week's population supplies 60 accusations in 17.9 days.
- **The fetch budget binds, not the archive.** At the measured 1.63 requests per record, 50 requests an hour gate about 735 records a day, 28% of one week's admitted records. At the AIDev rate, 60 accusations need about 46,000 gated records, which is 62.8 days of fetching. The study's fallback rule (section 4, "If the plan stalls") fires when the window must exceed three weeks. Read as archive hours alone, 17.9 days stays under it. Read as the time until 60 accusations exist, 62.8 days exceeds it. At the measured lower end, neither length is bounded. Either way, the operator's call on the budget, the target or the window comes before plan P.

## 1. What ran, in order

| when (UTC) | what | receipt |
|---|---|---|
| 2026-10-12 16:33:52 | commit `473bc08`: `sizing/sizing_plan_2026_09_28.json` (seed, n, provisional parameters, age-rule sensitivity, estimators, the builder's module digests) and `sizing/sizing.py` | git, this repository |
| 16:34:00 to 16:34:02 | three calibration hours for the age ceilings (`sizing.py ceilings`) | `ceilings` in the JSON |
| 16:34:08 to 16:40:31 | the 168 hours through `pool.derive` (`sizing.py derive`) | pool_sha256 `9fdbd941095e476c2800da3f1468da17e33cb3c9ff1b78dda262b393fdb359c0` |
| 16:40:51 | the sample (`sizing.py sample`) | sample_sha256 `e653e38146fec24d6dfa730fb4d6353b604988894affa81687d47fbedff9bf26` |
| 16:40:55 to 01:41:47 | the sample through `accuse.accuse_one` and the 7.49.0 gate (`sizing.py accuse`) | outcomes_sha256 `b878960b22a01f15884950a17f6975d87ab348ebbd63d5a88963f14e7075fb4e` |
| 16:41:40 to 16:48:15 | `stranger.py pool`, uncached, the builder's code alone | `sizing/stranger_pool_2026_10_12.json`, AGREES |

- **The builder's code.** Branch `research/sealed-ahead-precision` at `3c63ea40`, loaded by path. `sizing.py` refuses to start unless `common.py`, `population.py`, `pool.py`, `accuse.py` and `identity_table.json` hash (LF) to the digests in the sizing plan. The identity rules digest is `2fce0c15...`.
- **The gate.** `styxx/diffgate.py` from tag v7.49.0 (commit `c6e00da0`), extracted with `git archive`, matched as bytes to `09867056...`. It ran on CPython 3.12.10 with unicodedata 15.0.0; the styxx closure digest is `669df0ae...`.
- **The parameters.** `sizing/params_provisional_2026_09_28.json`, params_sha256 `667a38aa70c83071367ffa1a2cddb08e997569ac9a0dad8df76e0838e47ecb4e`. They are not a plan: `bitcoin_height` is a placeholder because the sizing run draws nothing.
- **How the extra counts were taken.** `pool.derive` ran unmodified. Two observers rode on it in this process only: a line counter, and a subclass of `population.Census` that counts event types, agent openings and the age rule at other ceilings. Their output lands only in `fields_census` and the hour cache, which no digest reads. The stranger run without them gave the same pool, so the observers changed nothing that counts.
- **Commits after the plan.** Four small commits to `sizing.py` followed, all on the report side:
  - `3c831e7` (16:43:10, the 29th record's outcome already visible in the progress log, 0 accusations so far) stopped the report listing app logins that embed people's names;
  - `a30bbac` and `a288f32` reworded a comment;
  - `2decbde` (16:59:37) added the D3 timing count.
  None touches counting, the sample or the gate. A last report-side edit, committed with this file, stopped the sample's family mix being written (it is not needed for sizing) and fixed a crash on records with no body source.

## 2. The archive week

- **Hours.** 168 ok, 0 gaps, 0 failed, 168 requests in all (one per hour). Streaming and scanning took 381 s, a median of 2.3 s an hour.
- **Bytes.** 13.0 MB to 30.8 MB compressed per hour, median 19.5 MB. That is 3.38 GB for the week, against the study's 2.6 GB estimate from one hour. Decompressed, the week is 15.6 GB; it was never written to disk.
- **Lines.** 13,765,478, every one a JSON event (0 bad lines). Per hour: 43,508 to 128,846, median 84,263.
- **PullRequestEvents.** 525,508, of which 207,838 are `opened`. Not one carries a body (`pr_body`: no_body 525,508). The payload is trimmed as the builder's README describes.
- **Capture varies by day, and the population with it.** Every hour carried PushEvents. But captured openings ranged from 10,305 on 2026-09-28 to 47,955 on 2026-09-30, a factor of 4.7, while total events moved the other way (2.52 M to 1.26 M). Admitted records per day ranged from 881 to 4,211. The agent share of captured openings held at 0.133 to 0.164. So the week's count measures what GH Archive captured, not what GitHub saw, and another week can differ by a large factor.

## 3. The population

### 3.1 Agent pull requests per week

| | opened, matched by the identity table | admitted (provisional parameters) |
|---|---|---|
| all | 30,435 | 18,098 |
| family `copilot` (login rule) | 502 | 250 |
| family `devin` (login rule) | 661 | 334 |
| family `jules` (login rule) | 94 | 76 |
| family `codex` (head-branch prefix) | 9,347 | 6,114 |
| family `cursor` (head-branch prefix) | 2,789 | 1,710 |
| family `claude` (head-branch prefix) | 17,042 | 9,614 |

- **Order.** Rows follow the identity table, not size.
- **Use.** These are sizing counts only. No rate, accusation count or precision is attached to any family here or in the JSON.
- **Prefix rules carry the pool.** 17,438 of 18,098 admitted records (96.4%) came from head-branch prefixes, which anyone can write.
- **Concentration.** 10,131 distinct opening accounts and 11,301 distinct repositories. A repository holds a median of 1 record and at most 59; 62 repositories hold more than 10. 98.2% are same-repository pull requests.
- **Per hour.** 11 to 307 records admitted, median 104.

### 3.2 What the planting exclusions cost

The age rule, applied to the same 30,435 matched openings (no duplicates appeared, so deduplication changes nothing):

| age rule | calibration hour (events in it) | admitted | excluded as account too young | excluded as repository too young |
|---|---|---|---|---|
| none | | 30,435 | 0 | 0 |
| `young_days` 0 | 2026-09-28-0 (119,814) | 25,954 | 308 | 4,173 |
| `young_days` 7 | 2026-09-21-0 (125,556) | 22,536 | 692 | 7,207 |
| `young_days` 30 (provisional) | 2026-08-29-0 (37,493) | 18,098 | 1,456 | 10,881 |

- **The cost.** The provisional rule removes 40.5% of agent openings, nearly all because the repository is newer than the ceiling. Much agent work happens in new repositories.
- **The choice of `young_days` is a size choice.** At the AIDev rate it moves the population-bound window for 60 accusations from 10.6 days (no rule) to 17.9 days (30 days).
- **The ceilings rose with time.** Account ceilings ran 322,312,423 → 331,827,544 → 334,741,854 across the three calibration hours, and repository ceilings 1,350,160,476 → 1,378,918,018 → 1,391,552,156. This is consistent with assumption A-1 (ids increase with creation) but does not test it.
- **A thin calibration hour.** The provisional calibration hour held 37,493 events, about a third of the other two. A ceiling taken over a whole calibration day would depend less on what one hour happened to capture.
- **No lab exclusions fired.** No record was excluded as the lab's own (`excluded_owner` or `excluded_actor`). The two logins in the provisional list are a guess at the lab's accounts and need the operator's list.

### 3.3 What the caps would cost, and a defect they would cause

Records kept if the plan set a cap (no names; each total is over the whole pool):

| cap k | `max_prs_per_repo` = k | `max_prs_per_actor` = k |
|---|---|---|
| 1 | 11,301 | 10,131 |
| 2 | 14,395 | 13,285 |
| 5 | 16,953 | 16,036 |
| 10 | 17,680 | 16,889 |
| 20 | 17,969 | 17,222 |
| 50 | 18,089 | 17,441 |

- **The defect.** The three login-rule families each open every one of their pull requests from one account. The week shows exactly one account id per login rule.
- **What it does.** `max_prs_per_actor` therefore caps each of those families at k pull requests per window, whatever its size. Battery comparison D8 recommends making the caps mandatory.
- **What it needs.** Read together, the per-actor cap must exempt the login rules' own accounts (or apply only to prefix-rule records), or the cap removes three of the six families.

### 3.4 Identity evidence (the table's E4)

- **Login rules.** Each matched exactly one numeric account id in the week: 198982749 (`login:copilot`), 158243242 (`login:devin`) and 161369871 (`login:jules`). These come from archive bytes, with no API request. The remaining half of E4 is each account's type, which takes three API requests and was not spent here.
- **Each prefix rule matched.** A case variant of `codex/` that the exact-case rule does not admit appeared 6 times.
- **Candidates the table does not admit.** Openings from unmatched accounts with head branches `copilot/` (42), `devin/` (130) and `jules/` (11).
- **App logins near an agent word.** 32 distinct unmatched app logins contain an agent word. Most embed a person's or an organization's name, so they are counted, not listed. Two are shaped exactly like a vendor's app, `claude[bot]` and `cursor[bot]`, and belong in the E4 review before the table freezes.

## 4. The re-derivation

- **What ran.** `stranger.py pool --params sizing/params_provisional_2026_09_28.json --published <pool_meta.json>`, from 16:41:40 to 16:48:15Z. It used the builder's code with no cache and no observer, and streamed all 168 hours again.
- **Result.** Verdict AGREES. Every check held: complete, hours, identity rules, params, pool_sha256. 18,098 records, no hour differed, no compressed sha256 differed.
- **What it shows.** GH Archive served identical bytes for every hour twice, 7 minutes apart, and the pool is a function of those bytes and the parameters.
- **What it does not show.** It ran on the same machine and runtime. It is not the study's second-machine re-derivation on WSL Linux.
- **Not tested here.** Battery comparison D6 (the stranger does not re-derive the ceilings) is not exercised. This run took the ceilings from the same calibration hours both times.

## 5. The sample through the frozen gate

- **The draw.** The 300 records with the smallest sha256(canon({pool_key, seed})) were taken from the 18,098. The seed is `styxx sealed-ahead sizing week 2026-09-28 sample v0`, committed in `473bc08` before any hour was streamed.
- **Fetching.** Each record went through `accuse.accuse_one` exactly as `accuse.run` does: the REST pull request, unauthenticated, then `pull/<n>.diff` if the head had not moved. The pull requests were 8 to 15 days old when fetched.
- **The budget.** 490 requests in all, 1.63 per record, never more than 50 in any sliding hour. The run spent 32,224 s waiting on the limiter. No 403 or 429 answered and no request was retried.

| outcome | records | share |
|---|---|---|
| GATED | 190 | 0.633 |
| HEAD_MOVED | 100 | 0.333 |
| PR_GONE | 8 | 0.027 |
| EMPTY_BODY | 2 | 0.007 |

- **Head moved.** A third of agent pull requests had moved their head by the time they were read. By opening day the share runs from 0.28 to 0.40, at 8 to 15 days of age, with no trend in age. In the window, with fetching about a day behind, it may be lower; with 735 records a day against about 2,585 admitted a day, most records would still be read late.
- **Body source.** Every description judged was read at fetch time (`api_now`, 292 records; the other 8 were never read because the pull request was gone). No description as opened was available, as the builder's README says.
- **Claims.** The 190 gated descriptions gave 71 claims: `file_touched` 44 (31 VERIFIED, 13 UNCHECKABLE), `file_created` 12 (7 VERIFIED, 5 UNCHECKABLE), `tests_pass` 9 UNCHECKABLE, `compat_claim` 5 UNCHECKABLE and `file_deleted` 1 UNCHECKABLE.
  - No claim was CONTRADICTED.
  - The PATH-2a overlay withheld nothing.
  - No count, test-added, symbol or scope claim was extracted at all.
  - 0.37 claims per gated record is the yield the window can expect from agent descriptions.
- **Battery comparison D3 measured.** The head check and the diff GET ended a median of 1 s apart. For 6 of the 190 gated records (3.2%) they ended more than 600 s apart (at most 3,550 s), because the diff waited out the hourly limit. For about one gated record in 30, a push in that interval would have gated another change than the one checked.

## 6. Accusations per 1,000 pull requests, by kind

Denominator: the 300 sampled admitted records. The Wilson 95% interval is on the share of records with at least one accusation of that kind (z = 1.959964).

| kind (appeared in the sample) | claims | accusations | per 1,000 admitted | Wilson 95% | per 1,000 gated | Wilson 95% |
|---|---|---|---|---|---|---|
| file_touched | 44 | 0 | 0 | 0 to 12.6 | 0 | 0 to 19.8 |
| file_created | 12 | 0 | 0 | 0 to 12.6 | 0 | 0 to 19.8 |
| file_deleted | 1 | 0 | 0 | 0 to 12.6 | 0 | 0 to 19.8 |
| tests_pass | 9 | 0 | 0 | 0 to 12.6 | 0 | 0 to 19.8 |
| compat_claim | 5 | 0 | 0 | 0 to 12.6 | 0 | 0 to 19.8 |
| any kind | 71 | 0 | 0 | 0 to 12.6 | 0 | 0 to 19.8 |

- **Kinds that never appeared.** They have no claims to count. A kind with no claim in 190 gated descriptions cannot accuse at a rate this sample sees either; its interval is the same 0 to 12.6.
- **One-sided bound.** The exact one-sided 95% upper bound for 0 of 300 is 9.9 per 1,000 (Clopper-Pearson).

## 7. Projection and window length

The week's admitted records: 18,098 (all 168 hours, so no scaling).

| rate per admitted record | accusations per 7 days | window for 60, population only | days to gate the records it needs at 50 requests an hour |
|---|---|---|---|
| Wilson lower end: 0 | 0 | **unbounded** | unbounded |
| point estimate: 0 of 300 | 0 | unbounded | unbounded |
| Wilson upper end: 12.6 per 1,000 | up to 229 accused records | 1.8 days | 6.5 days (4,746 records) |
| the study's AIDev rate: 1.3 per 1,000 (prior, not measured here) | 23.5 | 17.9 days | 62.8 days (46,154 records) |

- **The required answer.** At the lower end of the interval the window is unbounded: this sample cannot show that any length of window yields 60 accusations.
- **Why the sample could not do better.** Under the grant, it could not have been sized to. At the AIDev rate one expects one accusation per 769 records, about one day of the whole fetch budget. A lower bound above zero needs several accusations, so the sizing itself would take one to two weeks of budget.
- **Upper-end caveat.** Its row counts accused records, not accusations.
- **Population vs fetch.** The population-bound window and the fetch time overlap only in part. Records from the window can be gated after it closes, but each day of delay raises the head-moved share and the drift the stranger meets (battery comparison D2).

## 8. What this does not show

- No precision, and nothing about whether any accusation would be right. No accusation occurred.
- Not the description as opened. Every body was read 8 to 15 days after opening, so the gated population is pull requests whose head stayed put that long.
- Not a typical week. GH Archive's capture of openings varied 4.7-fold between days inside this one.
- Not AIDev's population. The labels come from logins and branch prefixes; 96.4% of admitted records rest on a prefix anyone can write.
- Not a second machine or a second runtime (battery comparison D1). The gate ran on CPython 3.12.10 only.

## 9. For the plan (decisions this informs, none taken here)

1. **The budget.** Under 50 requests an hour and no credentials, a 60-accusation draw at the AIDev rate needs about 63 days of fetching. That is more than the three weeks of the study's section 4 fallback if the rule counts fetching time. Unless the operator changes one of these, the plan inherits that length:
   - the budget (a token; the house rule today forbids one);
   - the target (a smaller n, with the wider interval that implies);
   - the window, gated through a pure-function subsample fixed in the plan (battery SA-23).
2. **The window cannot be sized from this sample.** If the plan keeps n = 60, it sizes on the AIDev prior and publishes a shortfall as a sizing failure, as the study's falsifier list already says.
3. **`young_days`.** 30 costs 40% of the population; 0 costs 15%.
4. **Caps.** A per-actor cap must exempt the login-rule accounts (section 3.3).
5. **The calibration ceiling.** Taking it over a day instead of an hour removes its dependence on one hour's capture.
6. **The identity table.** E4: three API requests for the account types, then a decision on `claude[bot]`, `cursor[bot]` and the unadmitted `copilot/`, `devin/` and `jules/` branches.
7. **D3.** A head re-check after the diff, or the head taken from the patch, is worth its cost. The race was live for 3% of gated records here.
8. **Capture variation.** If a week must reliably yield a given count, the window could be fixed as a function of archive bytes, for example closing at the hour in which the admitted count reaches K. It would still be a pure function a stranger re-derives. Whether that is acceptable is the plan's call.

## 10. Files

In this repository (committed):

- `SIZING.md`: this file.
- `sizing_week_2026_09_28.json`, with:
  - per-hour bytes, lines, events, PR events, openings, agent openings, admitted and excluded counts, and fetch attempts for all 168 hours;
  - the calibration hours;
  - population totals and family sizing counts;
  - age-rule and cap sensitivity;
  - identity evidence;
  - the sample's outcomes, claims and per-kind rates with Wilson intervals;
  - projections and the fetch budget.
- `sizing/sizing_plan_2026_09_28.json` (committed before streaming) and `sizing/sizing.py`.
- `sizing/params_provisional_2026_09_28.json` (params_sha256 `667a38aa...`), so anyone can run `stranger.py pool` against pool_sha256 `9fdbd941...`.
- `sizing/stranger_pool_2026_10_12.json`: the re-derivation's report.

Kept locally in `sizing_derived/`, not committed. These are derived records only; no raw hour is on disk.

- The pool, 18,098 records, 22 MB with the hour cache.
- The sample.
- The outcome ledger, rate stamps and digests.
- The 7.49.0 package.
- The logs.

The pool and the ledger name repositories and accounts, which is why they stay out of anything written for others.
