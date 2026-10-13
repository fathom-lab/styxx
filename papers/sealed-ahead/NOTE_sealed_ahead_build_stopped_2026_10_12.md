# NOTE: the sealed-ahead build stopped on 2026-10-12; its battery, sizing week and stranger pool check are records

2026-10-12, late evening (EDT). The sealed-ahead precision build in this directory (branch
`research/sealed-ahead-precision`) was stopped before any plan was fixed. The lab's decision record for that
night says: "sealed-ahead build STOPPED (wf_04546d53-53a stopped mid-sizing); its battery, sizing week and stranger
pool are to be committed as records". This NOTE does that. It adds `records_2026_10_12/`: ten files copied byte for
byte from the separate local repository the battery and the sizing week were written in (`_sealed_ahead_2026_10_12`,
no remote), and the 715 bytes of that repository's commit `6faeff16` as `git cat-file commit` prints them. No code
under this directory changes, no test of it changes, and nothing under `styxx/` changes. Section 6 says what this
does not show.

Receipt key. `R:` is `records_2026_10_12/` in this directory. `NL` is `NEXT_LEVEL.md`, `DEC` is
`DECISIONS_2026_10_12.md`, `AF` is `attack_feasibility.md` and `GG` is `ground_gate.md`: the plan of 2026-10-12, the
operator's answers and two of the plan's grounding reports, all local to the lab and not in this repository. `STUDY`
is `STUDY_where_styxx_goes_next_2026_10_06.md`, also not in this repository. A number quoted from them keeps the
receipt they gave it.

## 1. Why it stopped

- **The plan's row.** NL section 3, "what to stop or freeze": "the sealed-ahead build | stop; commit its battery,
  sizing week and stranger pool as records | AF H19 (about 77 days at the grant); GG §1.6 (37 attacks, 19 not
  caught under the plan as specified)".
- **The cost at the grant.** AF H19, reading this directory's `README.md`: the grant of 2026-10-12 is "50 requests
  an hour to github.com and api.github.com together"; a record costs two requests; the 60-accusation target is
  about 92,000 requests, "about 77 days, at 50 an hour". The README's own sentence: "Under the grant as it stands,
  the 60-accusation draw is out of reach".
- **The sizing week measured the same wall.** A record that ends at HEAD_MOVED, PR_GONE or EMPTY_BODY costs one
  request, so the measured cost was 490 requests for 300 records, 1.633 per record
  (`R:sizing_week_2026_09_28.json`, `sample.fetch_budget`). At the study's AIDev rate of 1.3 accusations per 1,000
  pull requests (a prior, not measured here), 60 accusations need 46,154 records: 62.8 days of fetching at 50 an
  hour, and 17.9 days of archive (`sample.fetch_budget.days_for_60_at_aidev_0p0013`). At the sample's own lower end
  neither is bounded (`sample.fetch_budget.days_for_60_at_lower_end` is null). The study's fallback rule fires when
  the window must exceed three weeks; read as fetching time, 62.8 days exceeds it (`R:SIZING.md`, "In brief" and
  section 9 item 1).
- **The decision.** The operator did not make this cut himself. Asked in chat at about 22:00 EDT on 2026-10-12, he
  answered "i trust you", and the record goes on: "delegated to the lead, who chose all four, as NEXT_LEVEL
  recommends". Stopping this build is one of the four (DEC item 3). Every outward act still comes back to him for
  its click (DEC item 1); this NOTE is not one.
- **"Stopped mid-sizing" and the records.** The sizing report was committed in the separate repository at
  `9acff5b0` (2026-10-12 21:45:26 −0400), about a quarter hour before the decision. Neither record says what the
  workflow had left to do after that commit. This NOTE commits the report as it stands at `9acff5b0` and nothing
  written later.

## 2. What was built

Seven commits on `f469f25b`, stated 2026-10-12 09:20:33 to 09:40:36 −0400 (`ce02a336` to `3c63ea40`). They stay
here as they were committed.

- **The population function.** `population.py` decides, from one hour's events and `identity_table.json`, which
  opened pull requests are admitted. The table holds six rules for AIDev's six agents: three opening accounts and
  three head-branch prefixes; its operative part hashes to
  `2fce0c15763231ef7a32a59e1e93091c286258136dd7782ed79dc84da6c6ad66`, which `tests/test_sealed_ahead.py` pins.
  Planting is bounded by id: a record whose account or repository id is larger than any seen in the calibration
  hour `young_days` before t0 is excluded as too young (assumption A-1, that ids rise with creation, is stated and
  not tested). The lab's own owners and logins are excluded.
- **The pool derivation.** `pool.py derive` streams every GH Archive hour of [t0, t1), stores none of them, keeps
  each hour's compressed sha256 and writes `pool.jsonl` and `pool_meta.json`. A gap is listed only after a grace
  period and a second 404, and never filled. Hour names are never zero-padded (`common.hour_name`).
- **The accusations.** `accuse.py` refuses to run unless `styxx/diffgate.py` hashes to the parameters' sha256 (for
  7.49.0, `09867056957082f129985591f5ed048021e85ed48627e73854494b4a1e6e2b3a`), and calls it as EXTERNAL-1 did:
  `gate_diff_text(title + "\n\n" + body, diff, run=None, strict=False)`. It gates a record only while the pull
  request's head is still the event's head (`require_unmoved_head`, which the parameters cannot turn off), and its
  rate limiter holds at most 50 requests in any hour across restarts.
- **The draw.** `draw.py` draws n accusations with `styxx.beacon.select` and the id of a Bitcoin block the plan
  names, proves the block's height from its coinbase (BIP34), and refuses each wrong order by name.
- **The stranger re-derivation.** `stranger.py` re-derives the pool, the accusations and the draw from public bytes
  and sorts every published row into MATCH, DRIFT, UNREPRODUCIBLE or INCONSISTENT; it exits 0 when it agrees, 1 when
  GitHub moved and it cannot confirm, and 2 when the published record is contradicted.
- **Tests.** `tests/test_sealed_ahead.py`, 70 tests on synthetic lines, no network.
- **Not built** (README, "Open items"): the plan P and its anchors, the panel, the planted decoys and their sealed
  key, scoring and a RESULT. The block-explorer fetch is written and never ran.

## 3. What the battery predicted and what it found

- **Predicted.** `R:BATTERY.md` holds 37 attacks in six groups: commitment and draw, shopping, population and pool
  construction, gate execution, panel, and reporting. Under the plan as the study specifies it, 7 are caught, 11
  bounded and 19 not caught, and "Six of the seven CAUGHT rows hold only if P or the build writes down what S
  implies" (`R:BATTERY.md` section 2). It was committed alone before the builder's code was read (section 5 says
  what that commit bounds).
- **Found against the build at `3c63ea40`** (`R:BATTERY_vs_build.md` section 2). The build implements the battery's
  catching check for 11 attacks (SA-05, 07, 13, 16, 17, 20, 22, 23, 24, 26, 28), part of it for 13, and none of it
  for 12; 1 needs nothing (SA-09). Ten of the 12 belong to stages the build says it has not built. Two are in scope
  and missing: SA-27 (clusters reach the draw) and SA-29 (no second derivation of the diff facts).
- **Eleven defects, D1 to D11** (`R:BATTERY_vs_build.md` section 3). Two were measured by the comparison:
  - D1: the pinned 7.49.0 gate gave 0 accusations on CPython 3.12.10 (unidata 15.0.0) and 1 on CPython 3.14.2
    (unidata 16.0.0) for the same title, body and diff, a count written with U+16131. A stranger on the other
    runtime reads an honest lab as INCONSISTENT.
  - D2: the stranger reads a substituted body as DRIFT whenever the pull request was touched after the lab's
    self-reported fetch time; a merge is enough.
  The sizing week then measured D3, the race between the head check and the diff: for 6 of 190 gated records
  (3.2%) the two ended more than 600 s apart, at most 3,550 s (`R:sizing_week_2026_09_28.json`,
  `sample.head_check_to_diff_seconds`; `R:SIZING.md` section 5).
- **An erratum the build forced on the battery.** GH Archive does not zero-pad the hour: `2026-10-05-3` answered 200
  with 17,712,148 bytes and `2026-10-05-03` answered 404 (`R:hour_name_check_2026_10_12.json`). So the battery's 404
  row, its item "a gap can last at least seven days" and the measured sentence under SA-21 are wrong
  (`R:BATTERY_vs_build.md` section 1), and the battery's commit message repeats the error ("a 404 hour persists a
  week", `R:battery_commit_6faeff16.txt`). The battery stays as committed; the comparison is its correction, and
  adds SA-38 (a URL shape manufactures gaps).
- **Where builder and battery agreed without reading each other** (`R:BATTERY_vs_build.md`, "In brief"): the
  description as opened cannot be fetched without a token; the 50-an-hour budget puts 60 accusations out of reach in
  a fortnight; and a description read now must not be judged against the diff at opening.

## 4. What the sizing week measured

The plan and the script were committed in the separate repository at `473bc087` (stated 12:33:52 −0400; the caveat
of section 5 applies) before any hour was streamed. The run went from 2026-10-12 16:34Z to 2026-10-13 01:42Z through
the builder's code at `3c63ea40`, which `sizing.py` refuses to load unless `common.py`, `population.py`, `pool.py`,
`accuse.py` and `identity_table.json` hash (LF) to the plan's digests (`R:sizing/sizing_plan_2026_09_28.json`,
`builder`). Counts only. The report is `R:SIZING.md`; the numbers below are keys of `R:sizing_week_2026_09_28.json`.

- **The archive week**, 2026-09-28T00 to 2026-10-05T00 UTC: 168 of 168 hours, no 404; 3,376,781,443 compressed
  bytes; 13,765,478 lines, 0 bad; 525,508 PullRequestEvents, none carrying a body (`archive`).
- **The population.** 30,435 openings matched the identity table and 18,098 were admitted under the provisional age
  rule (`young_days` 30), with no duplicates (`population.agent_opened_before_exclusions`, `population.pool_records`,
  `population.duplicates`). 17,438 of the 18,098 came from head-branch prefixes, which anyone can write
  (`population.rule_hits`). The pool's sha256 is
  `9fdbd941095e476c2800da3f1468da17e33cb3c9ff1b78dda262b393fdb359c0` (`population.pool_sha256`).
- **The stranger's pool check.** `stranger.py pool` streamed the 168 hours again, uncached, with the builder's code
  alone, from 16:41:40 to 16:48:15Z: verdict AGREES, every check true, 18,098 records, the same pool sha256
  (`R:sizing/stranger_pool_2026_10_12.json`). It ran on the same machine and runtime as the lab's run.
- **The sample.** 300 records drawn by a seed committed in `473bc087`; all 300 reached a final outcome: 190 GATED,
  100 HEAD_MOVED, 8 PR_GONE, 2 EMPTY_BODY (`sample.by_outcome`). Every description judged was the one read at fetch
  time (`sample.body_source`: `api_now` 292, `not_read` 8). The 190 gated descriptions gave 71 claims, none
  CONTRADICTED, and the PATH-2a overlay withheld nothing (`sample.claims_by_kind_and_verdict`,
  `sample.withheld_by_overlay`).
- **Accusations.** 0 of 300, for every kind. Wilson 95%: 0 to 12.6 per 1,000 admitted records, 0 to 19.8 per 1,000
  gated (`sample.any_kind`).
- **Third-party data.** The receipt is 98,663 bytes of counts, digests and hour names. It names no repository and
  no person's login. It names three vendor accounts by numeric id (one each for the identity table's three login
  rules) and two vendor-shaped app logins, as identity evidence with no count beside them
  (`identity_evidence`). The lab's earlier census receipts of this kind commit counts and leave every
  per-pull-request row out (`papers/closed-model-frontier/external2_summary.json` and
  `external2_gate_summary.json`, over 71,016 AIDev pull requests; per-agent counts appear in
  `external1_summary.json`). This change does the same. The pool (18,098 records, 9,644,635 bytes as `pool.jsonl`),
  the sample and the outcome ledger name repositories and accounts, and stay on the lab's machine.
  `R:sizing_derived/published/pool_meta.json` is the pool's published form, the file `stranger.py pool` reads; every
  string in it is a digest, an hour name, a schema name, a status, a time or a key set.

## 5. What commit 6faeff16 bounds, and what it does not

- **The object.** `6faeff16c71192385b5881628ab63c0933271487` is the root commit of the separate repository: no
  parent, and the repository has no remote. Its tree, `4b49d6d9bca6cb0538a04ce529a2b58cac769153`, has one entry,
  `BATTERY.md`, blob `44d45c30cb4c06c536019963aa4a3a7e4363eb5c`. Its author and committer lines both state
  `1791811103 -0400`, which is 2026-10-12 09:18:23 −0400.
- **What anyone can check from this change.** `R:battery_commit_6faeff16.txt` is the commit object's bytes. From the
  bytes committed here the three ids recompute: the blob id from `R:BATTERY.md`, the tree id from the blob id, and the
  commit id from the commit object. `tests/test_sealed_ahead_records.py` recomputes all three. So the commit names
  exactly these battery bytes and states that time.
- **The order it states.** The builder's earliest commit on this branch, `ce02a336`, states 09:20:33 −0400, 2 min
  10 s later. The comparison records one probe of the builder's working tree at 09:20:07 (`R:BATTERY_vs_build.md`,
  "Housekeeping"), also after the battery's stated time.
- **What it does not bound.** A local commit date is written from the committer's own clock, and git records any
  date it is handed. 09:18:23 −0400 is not a timestamp anyone else can check: the commit was never pushed, stamped
  or anchored, and the repository it sits in has no remote. "Battery before build" rests on the lab's clock and the
  lab's word. The same holds for `473bc087` (the sizing plan before the sizing run) and for the builder's own commit
  dates. What a stranger can check is that the bytes, the ids and the stated dates agree with each other, not that
  the dates are true.
- **Three files no commit dates at all.** `R:BATTERY_vs_build.md`, `R:hour_name_check_2026_10_12.json` and
  `R:sizing_derived/published/pool_meta.json` were never committed in the separate repository. No git date bounds
  them before this change.

## 6. What this does not show

- **No precision was measured.** The sample held no accusation, nothing was adjudicated, and no panel, decoy or
  sealed key exists. No plan P was fixed, posted or anchored; the provisional parameters' `bitcoin_height` is a
  placeholder (`1`), and nothing was drawn.
- **No gate was licensed.** Nothing here changes what `styxx.diffgate` accuses or withholds. The plan keeps the gate
  as it is: "freeze the gate as it is: CI doors report, nothing licensed" (NL section 3).
- **0 of 300 is not a finding about the gate.** At the study's AIDev rate, 300 records give zero accusations with
  probability 0.68 (`R:SIZING.md`, "In brief"), and the sample cannot exclude a rate of zero.
- **Not a typical week**: captured openings varied 4.7-fold between days inside this one (`R:SIZING.md` section 2).
  **Not AIDev's population**: the labels are logins and branch prefixes. **Not the description as opened**: every
  body was read 8 to 15 days after opening (`R:SIZING.md` section 8). **Not a second machine or runtime**: see D1.
- **The time bound of section 5 is the lab's own.**

## 7. The records

Every file under `records_2026_10_12/`, with its size, the sha256 of its bytes and its git blob id. Seven of them
are blobs of the separate repository's HEAD `9acff5b0` and carry the same blob ids there; the commit column names
the commit that last wrote each one. `.gitattributes` marks the folder `-text`, so a CRLF checkout keeps these bytes.

| file | bytes | sha256 | git blob | in the separate repository |
|---|---|---|---|---|
| `BATTERY.md` | 55,901 | `022cbe90392bd797d65eecb4cda02c1d60b7e7d7f987b90290468e76090398fb` | `44d45c30cb4c06c536019963aa4a3a7e4363eb5c` | `6faeff16`, 09:18:23 −0400 |
| `BATTERY_vs_build.md` | 20,842 | `d2025e4e913797b9817fce1006912c340388c617cbe9f3d4bd5ff4e7650f79a9` | `e69eae5a0664398039eaf1f2aeace73526cce4d3` | never committed |
| `SIZING.md` | 20,430 | `0b3e9c77da4c37c7512b9a26904a0d97264bc4b34210df1c26acd694bd882b08` | `08d0b8176a4b0297a5be07e0d202bef73b867409` | `9acff5b0`, 21:45:26 −0400 |
| `battery_commit_6faeff16.txt` | 715 | `0df975a3cd1dd140416971780180d4ee904a053953cd75a196baea8c062199c7` | `0b107a7397602d0dac58a347d5bd9307cf3a8ebd` | the object of `6faeff16`, written out for this change |
| `hour_name_check_2026_10_12.json` | 850 | `797f8c7d5bd434c410b1071f3e874bd26e0dafd2c255cb2b2075821c51742b95` | `33c274dc6305d8049821ac002bf45c4814e2ff0c` | never committed |
| `sizing/params_provisional_2026_09_28.json` | 900 | `ce19d3b372d654c3fcfe345da7b2fced63c8b133df76292c9e3e9afaee7042eb` | `58424dfca770c2e6d9e323e40fc1cd962920bdcb` | `9acff5b0` |
| `sizing/sizing.py` | 35,060 | `809604ef34ebe17d6593a80de5d7a0c1d6644a7a91f6bf924cdbb9982414fb39` | `574ededb7ce36e421b5f610c5d19db68d966e918` | `9acff5b0` (written from `473bc087` on) |
| `sizing/sizing_plan_2026_09_28.json` | 5,792 | `1b7ad7e3bdbc6907667abf23e2516c54bf6cfee32a672b46607a4f15ef88735b` | `26f7ed42c3b9610d40e8c1e04abe283ed9c15504` | `473bc087`, 12:33:52 −0400 |
| `sizing/stranger_pool_2026_10_12.json` | 577 | `740a1b544b5981e21c89e1ceb909c7ffce354d075d30e7265ff955400f02c1f5` | `6a2a2beae3d98e26ccf40f272853d2786330d756` | `9acff5b0` |
| `sizing_derived/published/pool_meta.json` | 130,143 | `92c5ba6a6d0cbe0d791ac5c06d09d5f7b9293031d32119dc31904dd42021666f` | `924c53b0e7384061a11f40fe09817733838a4bf8` | never committed |
| `sizing_week_2026_09_28.json` | 98,663 | `1503f12ed98da2ae18a79b4c45d60191e6de7d905e37f51c1f1bff4e5afdfade` | `a009f8786f2216de14370561863caadf047054a7` | `9acff5b0` |

`sizing_week_2026_09_28.json` names the script and the plan by these same sha256s (`script.lf_sha256`,
`plan.lf_sha256`), and `R:sizing/params_provisional_2026_09_28.json` hashes, as canonical JSON, to the
`params_sha256` the receipt and `pool_meta.json` carry, `667a38aa70c83071367ffa1a2cddb08e997569ac9a0dad8df76e0838e47ecb4e`.

## 8. The code on this branch

- **It stays as it is.** `common.py`, `population.py`, `pool.py`, `accuse.py` and `identity_table.json` hash (LF) to
  the digests the sizing plan names, which is what ties the sizing week's receipt to this code;
  `tests/test_sealed_ahead_records.py` fails if any of the five moves. `draw.py`, `stranger.py`, `README.md` (apart
  from the stop line added at its head) and `tests/test_sealed_ahead.py` are unchanged since `3c63ea40`.
- **Its tests are CI-safe as far as this box can show.** The 70 tests use fakes for every fetch and make no network
  request. After `origin/main` (`acf09c34`) was merged into the branch they passed on CPython 3.12.10 on Windows. The
  six modules and the test file parse under the Python 3.9 grammar, and ruff (E9, F63, F7, F82, target py39)
  reports nothing. They were not run on Linux or on Python 3.9 to 3.11 here; CI's matrix runs them there.
- **Two of them read the checkout's real gate.** `test_accuse_gates_an_unmoved_head_and_names_the_accusation`
  expects `files_changed_count` CONTRADICTED for "3 files changed in this PR." against a one-file diff, and
  `test_accuse_uses_a_pre_trim_body_as_opened_and_the_compare_route` expects `tests_added` CONTRADICTED. A later
  change to `styxx/diffgate.py` that withholds either verdict moves them.
- **`R:sizing/sizing.py` is a record, not code CI runs.** It loads the builder's modules from the local path its plan
  names and refuses to start otherwise, so it does not run from a checkout; nothing imports it. It is kept so that
  the receipt's `script.lf_sha256` has its bytes in the repository.
