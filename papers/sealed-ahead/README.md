# Sealed-ahead precision: the pipeline

**Stopped on 2026-10-12, before the precision plan was fixed.** The code below stays as it was committed. The battery, its comparison with this build, the sizing week of 2026-09-28 and the stranger's pool check are records in `records_2026_10_12/`, and `NOTE_sealed_ahead_build_stopped_2026_10_12.md` says why the build stopped and what the records do not show. The rest of this file was written before the sizing week ran: where it says no hour has gone through `derive` and no pull request has been gated, the sizing week has since done both (NOTE, section 4). No precision has been measured.

*Built 2026-10-12 on branch research/sealed-ahead-precision. No plan exists yet, no real GH Archive hour has gone through `derive`, no pull request has been gated, nothing is published. This directory is the machinery the plan will name, with tests on synthetic lines.*

The study of 2026-10-06 (`STUDY_where_styxx_goes_next_2026_10_06.md`, section 2.1, outside this repository) ranks one measurement above the rest: how often diffgate's accusations are right, judged blind, on agent pull requests opened after the plan was fixed, with every step a stranger could suspect re-derivable from public bytes. Five scripts do the deriving. Each is a pure function of public bytes and one parameters file; none takes a choice on its command line that would change a pool, an accusation or a draw.

| Stage | Script | Reads | Writes | Re-run by a stranger as |
|---|---|---|---|---|
| Who is admitted | `population.py` | one hour's events; `identity_table.json` | records and counts per hour | part of `stranger.py pool` |
| The pool | `pool.py derive` | every GH Archive hour of [t0, t1), streamed | `pool.jsonl`, `pool_meta.json` | `stranger.py pool` |
| The accusations | `accuse.py run` | the pool; api.github.com and github.com, unauthenticated | `outcomes.jsonl`, `accusations.json`, `outcome_digests.json` | `stranger.py accuse` |
| The draw | `draw.py` | the accusation set; one Bitcoin block | `draw.json` | `stranger.py draw` |
| The check | `stranger.py` | the parameters and the published digests | a report per stage | itself |

`common.py` holds the canonical bytes, the hour names and the parameters' validation. The tests are `tests/test_sealed_ahead.py` (70, no network).

## What a 2026 GH Archive event carries

Seen so far, by two reads: the study's attack pass of 2026-10-07 read about the opening 400 KB (compressed) of hour 2026-10-05-12, and a red-team probe on 2026-10-12 streamed hour 2026-10-08-15 whole, keeping counts only. Both found the same shape.

- The event: `actor`, `created_at`, `id`, `payload`, `public`, `repo`, `type`.
- The actor: `id`, `login`, `display_login`, `gravatar_id`, `url`, `avatar_url`. No creation date.
- The repository: `id`, `name`, `url`. No creation date.
- A PullRequestEvent's payload: `action`, `number`, `pull_request`, plus `label`/`labels` or `assignee`/`assignees` on those actions. Actions seen include `opened`, `merged`, `closed`, `reopened`, `labeled`, `unlabeled`, `assigned`.
- The pull request object: `base`, `head`, `id`, `number`, `url` only. `head` and `base` are `ref`, `repo`, `sha`; `head.repo` is `id`, `name`, `url`.
- Not carried: the title, the body, the author object, the pull request's own creation time, draft, the file list, any diff.

`pool.py fields --hour H` re-records this over any hour (key sets with counts, no names), and `derive` keeps the same census for every hour of the window in `pool_meta.json`, so the sizing pass and the window record what they actually saw.

## Hours are not zero-padded

GH Archive names hour three of 2026-10-05 `2026-10-05-3`. The padded `2026-10-05-03` answers 404. One HEAD request each on 2026-10-12: `-3` answered 200 with 17,712,148 bytes (last modified 2026-10-05 04:05:06 GMT), `-03` answered 404. The attack pass listed 2026-10-05-03 as a missing hour; it was the URL. A pipeline that pads the hour reads ten of every day's 24 hours as gaps. `common.hour_name` never pads and `common.parse_hour_name` refuses a padded single-digit hour.

## Who is an agent

`identity_table.json` admits six rules for AIDev's six agents: three opening accounts (`Copilot`, `devin-ai-integration[bot]`, `google-labs-jules[bot]`) and three head-branch prefixes (`codex/`, `cursor/`, `claude/`). Each names its evidence and what is still unchecked; the login strings and the prefixes are recalled, not yet observed, and the table says the sizing week must observe each one and record each login's numeric id before the plan freezes. It also lists six candidates it does not admit and why: body markers (the payload has no body, and a rule over a fetched body would make the pool depend on fetch time), labels, any `[bot]` login, agents outside AIDev's six, and two prefixes already covered by a login.

The operative part (rule, family, match, value) hashes to `2fce0c15763231ef7a32a59e1e93091c286258136dd7782ed79dc84da6c6ad66`, which the parameters must name and a test pins. A prefix is a label anyone can write; a login ending in `[bot]` belongs to an app. The labels will differ from AIDev's, which identified one agent from body text.

Family counts appear only as sizing counts in the pool's meta. Nothing here writes a rate, a precision or an accusation count per family, per product or per repository.

## Planting

GH Archive gives no creation date for an account or a repository, but it gives numeric ids, and GitHub assigns them in increasing order of creation (assumption A-1; a stranger can spot-check it with one request to `api.github.com/users/<login>` or `/repositories/<id>`). The plan names a calibration hour, the one that starts `young_days` before t0, and takes the largest account id and repository id seen in any event of it (`pool.py ceilings`). A record whose opening account or repository has a larger id is excluded as too young. Every account created after the calibration hour has a larger id than any account active in it, so the rule contains "created after t0 minus young_days"; it also catches some accounts made a little earlier, so it errs toward excluding.

Also excluded: repositories owned by `excluded_owners` and pull requests opened by `excluded_actor_logins` (the lab's own). Not excluded: an old account in an old repository. The optional caps `max_prs_per_repo` and `max_prs_per_actor` bound what one such holder can add (the earliest records by event id are kept).

## The description as opened, and the change being judged

A 2026 payload carries no title or body. Where an event's own payload does (an hour from before the trim), `accuse.py` uses it: that is the description as opened. Otherwise the only unauthenticated source is the REST API's pull request now. GitHub keeps a body's edit history, but only GraphQL exposes it, and GraphQL needs a token, which the grant of 2026-10-12 does not allow. So for 2026 records the description judged is the description at fetch time, every outcome says so (`body_source: api_now`), and the fetch time is kept.

That makes one more rule necessary. A description read now must not be judged against the diff as it stood at opening: an agent that opens a pull request with a placeholder commit and a plan, then pushes the work and rewrites the description, would be accused of every file and count its finished description names. So `accuse.py` gates a record only when the pull request's head is still the event's head (`require_unmoved_head`, which the parameters cannot turn off), and records HEAD_MOVED otherwise. The measured population is therefore agent pull requests whose head did not move between opening and the fetch. The RESULT has to say so, and the share of HEAD_MOVED is itself a number the sizing pass should report.

The diff comes from `github.com/<repo>/pull/<n>.diff` (named by the grant) or from `github.com/<repo>/compare/<base>...<head>.diff`, the diff between the event's own commits, which stays the same after the pull request moves. The compare URL is the same host and rate but is not a pull-request URL; it needs the operator's word.

## The budget

The grant allows 50 requests an hour to github.com and api.github.com together; GitHub allows an unauthenticated caller 60 an hour on its API in any case. A record costs two requests (the pull request, then its diff), or one when it ends at HEAD_MOVED, EMPTY_BODY or PR_GONE. At 50 an hour that is at most about 600 records a day.

The study sizes the window at roughly 46,000 agent pull requests for 60 accusations (0.13% per pull request on AIDev). Gating 46,000 records at two requests each is about 92,000 requests: some 1,840 hours, about 77 days, at 50 an hour. A fortnight at 50 an hour gates about 8,400 records, which at 0.13% is about 11 accusations. Under the grant as it stands, the 60-accusation draw is out of reach; the GH Archive side (about 2.6 GB a week, streamed) is not the constraint. What would change it is the operator's call, not this code's: a token (5,000 requests an hour, and GraphQL's edit history would give the description as opened, at the cost that a stranger then needs any GitHub account), a longer window, or a smaller target. `accuse.py` enforces whatever rate the parameters name, up to 50, and its limiter keeps its stamps across restarts.

## The gate

`accuse.py` imports `styxx.diffgate` and refuses to run unless its file hashes to `gate.sha256`, as bytes or with CRLF folded to LF. For 7.49.0 the blob is `09867056957082f129985591f5ed048021e85ed48627e73854494b4a1e6e2b3a`; a Windows checkout made with `core.autocrlf` holds the same blob as `d1fca413...`, and the LF fold matches it. The call is EXTERNAL-1's: `gate_diff_text(title + "\n\n" + body, diff, run=None, strict=False)`. A CONTRADICTED claim is an accusation. A claim the PATH-2a overlay withheld is UNCHECKABLE and is recorded with the verdict it withheld, so one run prices the overlay too. Every run records the interpreter, `unicodedata.unidata_version` and the LF sha256 of every styxx module loaded, because a verdict reads more than one file.

## The draw

`draw.py` calls `styxx.beacon.select` with the accusation set as the pool (3-tuples of strings, ordered by event id), `styxx.beacon.pool_sha256` of that set as the pool digest, and the block id as the beacon. It refuses, each by name, a height that is not the plan's, a header that does not hash to its id or meet its own target, too few confirmations, a block time not after t1, an accusation digest anchored less than two hours before the block, and items that do not hash to the published digest. Given the coinbase and its merkle branch it proves the height from bytes (BIP34) instead of trusting an explorer. Fewer accusations than n is a SIZING_FAILURE, never a smaller draw.

The order the plan needs is: the window closes, the accusation set is derived and its digest anchored, then the named block is mined. The explorer fetch (mempool.space and blockstream.info, which must agree) is written and has not run; those hosts are outside the grant.

## What a stranger checks

`stranger.py` re-derives each stage and compares it with the published digests. For the accusations it sorts every record into MATCH, DRIFT (the pull request changed after the lab read it, by its `updated_at`), UNREPRODUCIBLE, or INCONSISTENT (the published row is contradicted: equal inputs with other claims, a diff that differs, a description that differs though nothing changed, or a record set aside as HEAD_MOVED, PR_GONE or EMPTY_BODY that an unchanged pull request shows was readable, which is the trace pruning would leave). The published rows carry digests only and never say which record was accused, so the set is confirmed only when every row matches. Exit 0 agrees; 1 cannot confirm because GitHub moved; 2 contradicted. Both 1 and 2 are refusals.

Because descriptions can be edited, a stranger who re-runs weeks later will meet DRIFT, and drift on an accused record stops the draw from being confirmed. The re-run belongs soon after the lab's, or the lab must hand auditors the inputs of the accused records (`accuse.py` keeps them, for the panel, under `inputs/`; they are not in the bundle).

## Open items, none decided here

- The budget above: under the grant of 2026-10-12, the 60-accusation target does not fit a fortnight.
- The six identity strings are recalled (E3) and must be observed (E4) before the table is frozen into a plan.
- Assumption A-1 (ids increase with creation) is stated, not tested; a dozen API spot-checks would test it within the rate.
- The compare route and the two block explorers need the operator's word.
- The panel, the planted decoys and their sealed key are not built here (EXTERNAL-1's protocol is the model).
- The sizing week has not been run; `pool.py derive` over a pre-t0 week with provisional parameters is that run.

## Running

```
python papers/sealed-ahead/pool.py ceilings --t0 2026-10-19T00:00:00Z --young-days 30
python papers/sealed-ahead/pool.py derive --params plan_params.json --out DERIVED/pool
python papers/sealed-ahead/accuse.py run --params plan_params.json --pool DERIVED/pool/pool.jsonl --out DERIVED/accuse --styxx-path .
python papers/sealed-ahead/draw.py --params plan_params.json --accusations DERIVED/accuse/accusations.json --header <hex> --tip-height <n>
python papers/sealed-ahead/stranger.py bundle --params plan_params.json --pool-dir DERIVED/pool --accuse-dir DERIVED/accuse --draw DERIVED/draw.json --out DERIVED/published
python papers/sealed-ahead/stranger.py pool --params plan_params.json --published DERIVED/published
python -m pytest tests/test_sealed_ahead.py -q
```

Every command that touches the network reads only the hosts named above, sends no credentials, and keeps derived records only.
