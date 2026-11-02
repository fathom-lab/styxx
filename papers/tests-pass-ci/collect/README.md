# tests-pass-ci collector (A2)

The collector for `PREREG_tests_pass_ci_2026_11_02.md` (draft of 2026-11-02, revised after the critics' reviews):
the frame (Part II P2), the draw (P3), the read (P4), the screen (P5), the CI test result at the head commit (P6),
rule R (P17) and the machine-readable report check (P18), under the seal guard (P21). In the lab's repository it lives
in `papers/tests-pass-ci/collect/` beside the PREREG, with its tests in `tests/test_tests_pass_ci_collector_*.py`;
the combined digest `py -3.12 tpc_collect.py digest` prints fills the lock table's collector row (section 14) at the
signing commit, and the seal guard opens only for that code. Not here yet: the packet and decoy builder, the seat
runners and the scorer (section 14 lists them in the same row).

Python 3.9 or later, standard library only (pytest for the tests). Raw data never goes into git.

## The seal

Before the PREREG is frozen no one reads CI conclusions, check runs, statuses or job logs of any pull request
opened on or after 2026-11-02, nor classifies a claim in one (section 0.1). The collector enforces it in
`tpc_seal.py`:

- every body, check, status, Actions, log and artifact read passes `SealGuard.check(created_at)`, which raises
  `SealRefused` (`SEAL_GUARD`) for a pull request created at or after 2026-11-02T00:00:00Z (or with no creation
  time) unless the seal is open. It is asked before the GraphQL batch is sent (from the frame's `created_at`),
  again on the response's own `createdAt`, and again before any REST call for the row;
- the seal opens only when `C:/Users/heyzo/clawd/wt/_direction_a_2026_11_02/SEALED` exists and names a 40-hex
  `prereg_commit` and a `freeze_anchor` that is not in the future; after `git fetch origin` in
  `C:/Users/heyzo/clawd/styxx` that commit is an ancestor of `origin/main`; the PREREG at that commit (read with
  `git show`, nothing checked out) begins with "# PREREG"; every line that begins with `**Signature.**` begins
  `**Signature.** Signed` (and there is one); no lock-table row (a line beginning "|") still reads "TO-FILL at the
  signing commit" (prose that quotes the markers mid-line is not read); and the lock table's collector row holds one
  sha256, equal to the combined digest of the running code. Each failure is refused by name (P21). The collector
  never writes SEALED;
- `run` without `--fetch-ci` reads no body and no CI at all; `--dry-run` reads only the seal's own metadata list.

What the frame stage stores before the freeze: P2.4's fields (`pr_id`, `repo`, `number`, `stars`, `fork`,
`author`, `author_type`, `created_at`, `agent`, `slice`). Three of them (`pr_id`, `fork`, `author_type`) lie beyond
the seal's literal list in the run brief; the PREREG's section 0.3 allows them for the frame stage. The operator
decides whether the scheduled frame runs before the freeze.

## Running it

```
py -3.12 tpc_collect.py run                    # catch-up: frames of finished window days; the draw once the beacon is stored
py -3.12 tpc_collect.py run --fetch-ci         # and every drawn row due (72 h after it opened); window rows wait for SEALED
py -3.12 tpc_collect.py run --dry-run          # count window pull requests by metadata only; counts, never rows
py -3.12 tpc_collect.py status                 # counts per stage, K1's capped count so far
py -3.12 tpc_collect.py verify                 # re-hash every file the manifest names
py -3.12 tpc_collect.py digest                 # the code digests the lock table's collector row holds
py -3.12 tpc_collect.py beacon --height 975000 --rule "section 3.1, named height" --go     # OD-8, operator's word only
```

Options: `--max-reads N`, `--max-calls N` (a request budget), `--max-wait S` (the longest wait for a limit before
the run stops cleanly; default 3,700 s), `--workers N` (search threads; searches are paced to about 27 a minute
across threads), `--stages frame,draw,read`, `--root DIR`.

Development runs, on pull requests opened before 2026-11-02 only, in a separate store:

```
py -3.12 tpc_collect.py run --dev --days 2026-10-29 --dev-beacon "label" --fetch-ci --max-reads 200
```

`--dev` refuses the window store, refuses window days, and takes a development beacon (a sha256 of the label,
marked `kind: dev`); the window store refuses a development beacon and checks a Bitcoin header's proof of work.

**Idempotent.** A day whose frame file exists is not enumerated again; a day whose draw file exists is not drawn
again; a row whose record exists is not read again. A run that hits a limit it should not wait out stops with
`BudgetStop`, records why in `runs.jsonl`, and the next run continues. A failed read leaves no row record and is
read again by the next run, into a new attempt directory. One run at a time: `run.lock` is touched by every API
call and before every wait (waits are capped at 90 minutes), and only a lock untouched for two hours, the mark of
a run that died, is taken over.

**Scheduled (OD-5).** The PREREG wants the collector outside any agent session. A schedule that meets P2.4 (day D
enumerated between D+1 00:00Z and 12:00Z) and P4.1 (rows read 72 to 84 hours after they opened): `run --fetch-ci`
every two hours. Not set up here; it is the operator's.

## Where things go

Root: `C:/Users/heyzo/clawd/data/direction_a/raw/` (window), `C:/Users/heyzo/clawd/data/direction_a/raw_dev/`
(development). The PREREG draft names `data/direction_a/frames/` (P2.4) and `data/direction_a/a2/` (P4.3); here
both sit under one root as `raw/frames/` and `raw/a2/`, as the run brief asked. One of the two texts changes.

```
manifest.jsonl              sha256 and size of every file written (gzipped files: of the bytes compressed too)
readlogs/readlog_<stage>_<day>.jsonl   every API call: time, kind, endpoint or query, status, bytes, seconds, quota
runs.jsonl                  every run: mode, argv, code digests, seal state, calls, stop reason
frames/frame_<D>.jsonl      P2.4, one line per node returned, exactly the ten fields
frames/frame_digests.jsonl  per day: file sha256, enumeration times, late flag, code digest, every query's issueCount and node count
draw/beacon.json            the beacon record (P3.1)
draw/draw_<D>.jsonl         drawn rows only (u, v, in_D, in_M, flags, signatures matched), sorted by u
draw/draw_digests.jsonl     per day: counts per agent (frame nodes, kept, drawn, in M), double matches, digests
a2/raw/<pr_id>/<attempt>/   every response of the read, gzipped; logs/<job>.log.gz; artifacts/<id>.zip and counted report files
a2/raw/_graphql/            the GraphQL batch responses
a2/rows/<pr_id>.json        the row record, written last
a2/rows_index.jsonl         one line per finished row
a2/seal_refusals.jsonl      every SEAL_GUARD refusal of due window rows
a2/read_errors.jsonl        reads that failed (read again next run)
dryrun/dryrun_<time>.json   dry-run counts
```

## The modules

| file | PREREG | what it does |
|---|---|---|
| `tpc_const.py` | P1 | every constant, with its section |
| `tpc_gh.py` | P21, section 4 | read-only client: `gh auth token` with GH_TOKEN unset, the token sent to api.github.com only, every call logged, limits waited out, searches paced, redirects not followed (signed log and artifact URLs fetched without the token) |
| `tpc_seal.py` | 0.1, P21 | the seal guard |
| `tpc_store.py` | P2.4, P4.3 | raw store, atomic writes, no overwrite, sha256 manifest, `verify` |
| `tpc_frame.py` | P2 | the slice formula, the range queries and their halving, frame rows, the dry run |
| `tpc_draw.py` | P3 | u, exact comparisons on integers, the sets D, M, A_g |
| `tpc_beacon.py` | 3.1, P3.1 | block from two explorers, header hash and proof of work; development beacons |
| `tpc_read.py` | P4, P8.3 | the per-row read, exclusions, logs and artifacts, STALE, LATE_READ, the disk floor |
| `tpc_screen.py` | P5, P6 | testreceipt's reader and CI reader, vendored unchanged, with the PREREG's one change (`timed_out` is a failure) |
| `tpc_rule.py` | P17, P8.3 | log cleaning, the twelve summary patterns (digest checked at import), rule R |
| `tpc_reports.py` | P18 | report-like artifact names, JUnit/TRX/in-toto detection inside a zip, `styxx.evidence` |
| `tpc_collect.py` | | the catch-up run, the dry run, status, verify, digest, beacon (named `tpc_` so it cannot shadow another arc's `collect.py` on the test suite's path) |
| `vendor/testreceipt/` | 14 | `claims.py`, `ci.py`, `reconcile.py`, `__init__.py`, `LICENSE` from `amos689/testreceipt` at `95864ef9`, byte-identical (MIT); the pins are checked at import |

## Readings of the PREREG the code makes (each one is a sentence the reviser can change)

1. **Frame file content.** `frame_<D>.jsonl` holds every node the searches returned (under 100 stars, forks and
   duplicates included); "kept" is applied at the draw. A null repository is kept and flagged; a null author is
   flagged and passes or fails the star filter like any other row.
2. **Double matches.** A pull request matched by two signatures goes to the earliest row of P2.1 that matches it,
   Claude Code read last (`tpc_const.MATCH_PRECEDENCE`; revised in the PREREG on 2026-11-02): Claude Code is sliced
   and the other five signatures are not, so a double match given to Claude Code would sit in M with probability
   1/6 x 0.120 + 5/6 x 0.020 = 0.0367, not 0.020. Counted per day in `draw_digests.jsonl`.
3. **"Lowest" inside M reads v.** Claude Code rows of M have u up to 0.120, other agents' up to 0.020, so lowest-u
   orders would favour non-Claude rows six to one; the PREREG (revised) orders the cap of 3 and the audits AU1 to AU4
   by `v = u / p_M`, uniform on [0, 1) inside M for every agent, and the top-up cut by `w = u / P_AGENT`.
   `capped()` sorts by v; v is stored beside u.
4. **Logs for audit rows.** The audits take the lowest-v rows of their kind over the whole window, known only at the
   close, and logs expire sooner; so every row that can become an AU1 row (eligible, in M, neither claimer nor
   implied-only, tests failed) gets its logs at read. AU2 rows are claimers with a CI test result and get logs
   anyway; AU3 is step 1 only.
5. **Excluded rows read no CI.** A row with any P4.4 exclusion stops after the GraphQL read.
6. **Steps of a timed-out job.** testreceipt attaches a job's steps when its check run concluded success or
   failure; with `timed_out` read as failure (P6), a timed-out job gets its steps too.
7. **LOG_MAX.** A job log over 30 MB keeps its leading 30 MB (`truncated_at_log_max`). Test summaries sit near the
   end of a failing step, so a head cut can lose them; a head-and-tail cut is the alternative.
8. **P17.1's path list and local runs.** The revised P17.1 adds pytest's `test_<name>.py` to the path list and
   leaves out claim lines naming a local run ("locally", "on my machine"), which the rubric reads as part claims;
   tests pin both.
9. **The GraphQL read adds `databaseId`** to P4.2's fields, to check that the row read is the row drawn
   (`PR_ID_MISMATCH`, `CREATED_AT_MISMATCH` flags).
10. **`pushedDate`** is requested as P4.2 lists it; GitHub returns null for it (deprecated).
11. **Skipped jobs.** A test-like job whose conclusion is `skipped` is listed in the row record but its log is not
    asked for: in the development run all 51 such requests that reached GitHub's log store answered 404.

## Measured on pull requests opened 2026-10-29 (development store, dev beacon; not window data)

From `checks/dev_report.py` over `C:/Users/heyzo/clawd/data/direction_a/raw_dev/` after two runs (code digests
`00517e3f...` and `c8408d4e...`, both before the revision of 2026-11-02, which changed the seal guard, the cap's order,
rule R's item 1 and the double-match order, and so the digest):

- **Frame of 2026-10-29**, all ranges: 56,603 nodes in 38 minutes (14:59:41Z to 15:37:26Z), 683 searches of which
  633 answered 200, 35 met a secondary limit (61 s each; the shared cooldown in `tpc_gh.py` came after this run),
  and 15 met a 502, a 504 or a network error and were retried; Claude Code 38,695
  nodes in 48 slices (2,750 kept), Cursor 14,256 (312 kept), Copilot 1,436 (147), Devin 1,639 (107), Codex 183
  (20), Jules 394 (3). No node repeated within a signature; no kept pull request matched two signatures; one
  query returned one node fewer than its issueCount.
- **Draw** (dev beacon): 591 drawn, 369 in M (Claude Code 355 of them).
- **Read**: 230 rows (200, then 30 by the catch-up run), 0 read errors, every listing paged to its end. Exclusions:
  3 BOT_AUTHOR, 1 EMPTY_BODY. Screen: 48 claimers, 6 implied-only, 172 none. CI at the head: 141 tests passed,
  22 tests failed, 52 no test result, 11 no CI. Claimers with a CI test result: 38.
- **Logs**: 45 rows wanted logs (38 claimers with a CI test result, 7 AU1-eligible rows) and 30 got them before the
  disk floor stopped logs; 435 job logs, 395 MB raw,
  44.8 MB gzipped; 3 logs reached LOG_MAX; 15 logs hold a P17.2 summary line.
- **Disk**: C: fell from 16 GB to 4.0 GB free during the run, from causes outside this store (295 MB in all of
  `data/direction_a/`); the floor stopped logs as designed (16 rows `LOGS_SKIPPED`).
- **Reports (P18)**: 130 report-like artifacts listed, 129 downloaded; 3 rows hold a JUnit file; `styxx.evidence`
  read 50 files, all UNCHECKABLE.
- **Rule R**: 0 firings (item 1 on 11 rows, item 2 on 2). STALE on 72 of 226 eligible rows (the description last
  edited before the head commit was made).
- **Cost**: median 11 API calls and 6.4 s per row (39 calls for rows with logs), at most 217; 4,896 calls for the
  230 rows.

## Tests

```
py -3.12 -m pytest -q tests
```

Offline, against fakes: the slice formula and the 48 slices, range queries and their halving, the exact draw and the
sets, the seal guard (each case the integrity review's probe opened, and every check of P21, refused by name; a
signed commit on origin/main with the running code opens it), the code digests, the store
and its manifest, the client's limits and its token handling, the screen and the CI result (the timed_out change,
steps, statuses), each of the twelve summary patterns, rule R item by item, the report detector, the beacon on the
genesis header, exclusions, logs for claimers and AU1 rows, the disk floor, STALE and LATE_READ, the catch-up run
twice (no new calls the second time), window rows refused without SEALED (no body and no CI call made), the dry run
(counts only, the seal's fields only), a failed batch read again by the next run, the lock's heartbeat, the shared
search cooldown, the cap by v, Claude Code read last on double matches, and every module parsing as Python 3.9:
113 tests in this folder's layout. A mutation check (`checks/mutation_check.py`, run as
`py -3.12 checks/mutation_check.py <scratch dir>`, or from the repository root with the collector tests as targets)
breaks twenty of these rules one at a time and the tests catch each one (`checks/mutation_check_output.txt`).
