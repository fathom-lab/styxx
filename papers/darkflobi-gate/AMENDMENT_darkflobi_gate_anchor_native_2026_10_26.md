# AMENDMENT — the anchor reaches the gateway's own `deliver.js`: the daily anchor that left no send line (F-1)

Fathom Lab · 2026-10-26 · Amends `PREREG_darkflobi_gate_2026_10_23.md`, in this directory, merged as #229 with its
section 15 closed at 15.13 and not edited, read with the five before-D0 amendment files of the branch
`papers/darkflobi-gate-before-d0` (#243), through whose `lock_hashes.py check --amendment` this file is read.
Written by the lab's builder for the lead on 2026-10-26, while `W:gate/REQUIRED` is absent (shadow), from finding
F-1 of the self-forecast design review, reproduced on the box, fixed on a copy, and revised in place the same day
after three adversarial reviews (`F1:REVIEW_1.md`, one reviewer; `F1:REVIEW_2.md`, four; `F1:REVIEW_3.md`, two), whose
findings are answered in the section of that name. It changes how the plugin reaches the gateway's
`deliverOutboundPayloads` for the
anchor, and nothing the anchor says, when it is sent or what it is checked against; and it makes `lock_hashes.py`
read every file of the installed plugin folder. It asks the operator for one decision, the move (section "Whose
step"). No bar, no row of the gates block and no catch rule moves.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `BD:` = `WT:_darkflobi_gate_2026_10_22/before_d0/`;
`F1:` = `BD:f1_anchor/`; `RC:` = this folder's `receipts/anchor_native_2026_10_26.json`; `PREREG` and `RULES` = this
folder's files at `origin/main` `55fd36a3` and `c932129c`, line numbers their own; the dist = clawdbot's installed
`dist/` (clawdbot 2026.1.24-3), Node v24.13.0, jiti 2.6.1. [R] marks a reading with its receipt, [J] a judgement.

## What the PREREG says

- The anchor destination "expects there exactly the ledger's `payload_kind: anchor` sends and nothing else"
  (`PREREG` l.177).
- `RULES` §9, "Origin, not form, makes an anchor" (l.382-386): the plugin's timer sends the day's line through
  `deliverOutboundPayloads`; "the E2 handler matches the nonce once, ledgers the send as `payload_kind: anchor` under
  its own `kind: anchor` line". "A send Telegram rejects consumes the nonce; the timer re-arms a fresh nonce for each
  retry, at most one a minute, and stops at 00:10, after which no anchor is sent and the day reads `missing` at the
  census" (l.388-390).
- The census row of `RULES` §9 (l.409): "any message in the anchor channel that is not a ledgered anchor send" is
  counted as an unledgered delivery `[u]`, which fails G-1. `PREREG` l.834 reads the same row for a message of the
  operator's there.
- `RULES` l.305-307: the plugin's own anchor failures, `anchor_send_failed` among them, go to
  `plugin_failures.jsonl`, and "the census reads that day's anchor as `missing` (§9) whatever the file says".

So a delivered anchor with no send line is a `[u]` in the anchor destination, and one inside the window fails G-1;
an anchor the plugin fails to send, and records, is a `missing` day.

## What was observed

1. **Two anchors were delivered and neither has a send line** [R, `RC:` `route_scan`, over frozen copies of the ledger,
   the anchor state and the logs, each named there by sha256, and the live `sources/` folder, named by a digest over its
   files]. At seq 134 the ledger holds two `kind: anchor` lines, seq 58 (day 2026-10-24, written
   2026-10-25T04:00:01Z) and seq 117 (day 2026-10-25, written 2026-10-26T04:00:00Z), both by the gateway process pid
   28320, and `W:gate/anchor_state.json` marks both days delivered (04:00:02.436Z and 04:00:01.221Z). Of its 65 send
   lines, 42 are `reply`, 14 `telegram_action`, 9 `outbound`, and none is `payload_kind: anchor`.
2. **gate.py never judged either anchor** [R, the same]. `gate.py check` keeps a source file for every text in the
   anchor's form it reads (`W:gate/gate.py`, the anchor branch of the check), and `W:gate/sources/` holds none with
   either day's anchor text. There is no `W:gate/plugin_failures.jsonl`, so the plugin recorded no fault either.
3. **The hook runner did not run at either midnight** [R, the same]. Up to 2026-10-26T17:02:49Z the gateway's log
   files hold `[hooks] running message_sending (1 handlers` 65 times and `message_sent` 65 times, as many as the
   ledger's send lines, always with one handler; they hold no line of any kind within 120 seconds of either delivery;
   and each of the nine E2 `outbound` send lines has a `message_sending` line 0.3 to 10.6 seconds before it. The counts
   agree, and the log pairs no run with a line beyond that. The file is any clawdbot process's (its lines carry no
   process id): two `message_sending` runs at 2026-10-26T17:16:38Z, with no ledger line, fall inside the second
   review's own experiments, which ran clawdbot's hook runner in other processes, and the scan's cutoff leaves them
   out (without it, 67 against 65). In shadow, `gate.js` `gateOutbound` answers
   a runner with no `message_sending` handler with `{ cancel: false, ungated: true }`: the send goes, no handler runs,
   no line is written, which is what both nights show.
4. **Why the runner was empty** [J, supported by `RC:` `sdk_repro` and `probe_d8`, reproductions in another process,
   not the gateway]. clawdbot's loader evaluates plugins through jiti (a new jiti for each uncached load,
   `dist/plugins/loader.js` l.150) and aliases `clawdbot/plugin-sdk` to `dist/plugin-sdk/index.js`. jiti rewrites the
   `import()` calls written in a module it evaluates into its own loader, and keeps the modules it evaluates in Node's
   process-wide `require.cache` (jiti's `moduleCache`, on by default, which the loader does not turn off). The
   gateway's config names one plugin entry and one channel, both `telegram` [R, `RC:` `route_scan`
   `config_key_names`, key names only], and the bundled telegram channel plugin imports the SDK, whose graph reaches
   `deliver.js`. After the SDK has gone through jiti, the 0.4.0 plugin's `import()` of `deliver.js` (`getDeliver`)
   returns jiti's copy of `deliver.js`, which imports jiti's copy of `hook-runner-global.js`, whose runner no plugin
   registered into [R, `RC:` `sdk_repro` `after_sdk`: `import_v040_same_as_native` false; `jiti_runner_is_null` true,
   read from jiti's cached `hook-runner-global.js`, the module jiti's `deliver.js` gets for that path; before the SDK
   the same `import()` returns the native module]. E2 in that copy reads the null runner and lets the anchor go
   (`F1:REVIEW_2.md`, semantics, reproduced end to end with a fake channel adapter: jiti's copy sends with no handler
   call). The gateway's own sends run in the modules the gateway loaded natively and meet the real runner, which is
   why every other send has its line.
5. **A `require()` of `deliver.js` is no fix** [R, `RC:` `probe_d8`]. Node's require reads `require.cache` before it
   loads anything, and jiti writes its copies there under keys with forward slashes, while Node's keys on Windows use
   backslashes, so on this box a require misses jiti's copy and reaches the native module. Where the keys are equal,
   as on POSIX paths, the probe reads `collided_require_same` false: the require returns jiti's copy. The draft of this
   file proposed that require; the review found it (`F1:REVIEW_1.md` finding 1).
6. **Why the harness passed** [R, `W:gate/plugin/tests/run_tests.mjs` at `8d4b7db0…`]. Every anchor test (F1-F10)
   injects `loadDeliver`, so `getDeliver`'s `import()` never ran under test, and D6 loads only the plugin through
   jiti, with no dist module in `require.cache`, where the `import()` reaches the native module anyway.
7. **Under `REQUIRED` the same path cancels** [J, from `gate.js` `gateOutbound`]: no runner handler and `REQUIRED`
   present answers `{ cancel: true, reason: "gate_not_registered" }`, so from D0 the anchor would never post and every
   day would read `missing`.

## The amendment

1. **One native `import()`, in a CommonJS file.** A new file beside `index.js`, `native_import.cjs`, exports
   `importNative(url)`, which is `import(url)`. It is CommonJS (the `.cjs` extension overrides the folder's
   `"type": "module"`), `index.js` loads it through Node's own require (`createRequire` from
   `process.getBuiltinModule("node:module")`), and jiti 2.6.1 too hands a `.cjs` file to Node's require [R, `RC:`
   `probe_d8`: after jiti has loaded the helper, no jiti key for it exists and Node's does], so its `import()` is
   Node's whichever way it was reached. Node's `import()` of `deliver.js` resolves through the ES module map, keyed by
   URL, which jiti does not write to, so it returns the instance loaded from that URL: the gateway's when `distDir`
   names the dist by the URL the gateway loaded it from, which `argv[1]` gives (`dist_dir_source: argv1_entry` on the
   live started lines); a config `distDir` in another case names a second instance, whose runner was never
   initialised, and item 2 refuses it as `anchor_runner_null`. `loadNativeDist(distDir, pluginDir)` imports
   `deliver.js` and then `hook-runner-global.js` by the URL `deliver.js` itself imports it by, so the runner it returns
   is the one the returned deliver reads, and returns `deliverOutboundPayloads`, `getGlobalHookRunner` and
   `getGlobalPluginRegistry`. `getDeliver` uses it whenever no `loadDeliver` is injected; no `import()` of a `dist/`
   module is left in `index.js`.
2. **No anchor is sent unless styxx-gate's handler is there to meet it, read before every send.** `guardedDeliver`
   reads `runnerState` afresh before every anchor send: `gate` when the runner has a `message_sending` hook and the
   registry it was built from (clawdbot sets both together) names `styxx-gate` for that hook; `null` when the module's
   runner was never initialised; `unbound` otherwise; `unreadable` when the reading throws. Anything but `gate` sends
   nothing: the call fails with `anchor_runner_unbound`, `anchor_runner_null` or `anchor_runner_unreadable`, which the
   timer records as `anchor_send_failed` in `plugin_failures.jsonl` and retries under the existing rule (a fresh nonce,
   the day's composed text reused, at most one retry a minute, nothing after 00:10). A load of the dist that throws is
   recorded the same way with its own code and attempted again at the retry (the failed load is not kept). The retry is
   the plugin's existing one for any failed attempt, the schedule `RULES` l.388-390 gives a send Telegram rejects; this
   file adds attempts that fail before the exit, and no schedule. A day whose every attempt fails has no anchor and
   reads `missing` (`RULES` l.305-307, l.388-390), never `[u]`.
3. **An anchor E2 sends nothing for is recorded.** When `deliverOutboundPayloads` returns no message (E2 cancelled the
   anchor: under `REQUIRED` for `gate_hook_foreign` or `gate_result_unbound`, in either phase for `gate_no_verdict`,
   or the handler's own cancel), the timer records `anchor_send_empty` with the day before its retry; before this file
   such a day left neither a send line nor a record. `RULES` l.305-307 does not list that kind, and the census reads the
   day as `missing` "whatever the file says", so the record informs the RESULT and changes no count; the same holds for
   the refusal codes inside `anchor_send_failed`.
4. **The plugin's own file record names the helper.** `GATE_SHA_FILES` (the started line's `plugin_gate_files`) gains
   `plugin/native_import.cjs` as its ninth name; `gate.py`'s `gate_sha256` reads every file of the installed folder
   and so reads the helper too.
5. **The lock reads the whole installed folder.** `lock_hashes.py` lists every regular file of
   `~/.clawdbot/extensions/styxx-gate/` as a row of the box, as `gate.py` hashes every one of them, so a file there that
   no row names (a stray left by a failed move) reads `UNLOCKED` under `--strict`; a named file the folder does not
   hold reads `NOT BUILT` in the table; `PLUGIN_HOME` names seven files. The box before the move reads, with the five
   before-D0 amendment files, rows 84 and every count 0, as it did with the tool of #243 [R, `F1:`
   `lock_check_tool_v2.txt`].
   `test_rows_for_box_reads_every_file_of_the_installed_plugin_folder` covers it. The tool's bytes: blob
   `430458ffafbaa39fba03091da226a40f36939ea6` before (#243's), `fa59627f1db1cdc259a3bef444db644b2a474c11` after.
6. **Tests.** D8 runs `tests/f1_native_deliver_probe.mjs` in its own process, from `tests/_tmp` as its working
   directory (so a helper resolved from the working directory fails). With dist modules in jiti's cache and the helper
   loaded by jiti before Node, it reads the 0.4.0 `import()` (kept as the fixture `tests/fakes/import_deliver_v040.mjs`)
   returning jiti's copy and a null runner, and the plugin's own default `getDeliver`, with nothing injected, returning
   the gateway's `deliverOutboundPayloads` and runner in state `gate`; then it makes jiti's `require.cache` keys equal
   Node's for `deliver.js`, the runner and the helper, and reads both again (the require now returns jiti's copy;
   `getDeliver` still returns the gateway's). It reads every runner state (`gate`; `unbound` for an empty registry,
   another plugin's handler only, and styxx-gate's `message_sent` beside another plugin's `message_sending`; `null`
   for jiti's never-initialised copy; `unreadable` for a throwing reading), one guarded function sending, refusing,
   refusing, refusing and sending again as the runner goes bound, unbound, foreign, unreadable, bound, and a deliver
   that throws reaching the timer as its own error, not as an empty send. In the harness, with a stand-in
   loader (the real one runs only in D8): F11, the anchor goes through it once, one send line `payload_kind: anchor`,
   delivered, no failure recorded; F12, a runner without styxx-gate's handler is refused at 00:00 and at every retry
   through 00:09, ten records, one `kind: anchor` line, nothing sent, then the next midnight; F13, a loader that throws
   once is recorded with its code and asked again at the retry, which sends; F14, a deliver that sends nothing is
   recorded as `anchor_send_empty`, not marked delivered, retried; F15, one deliver held by `getDeliver` while the
   runner goes unbound, bound, unbound, bound sends, refuses, sends. D2 reads the helper as the ninth name.
7. **The plugin's README** names the helper in its file table, in install step 1 (seven files) and in the anchor
   paragraph, and lists the new tests.
8. **The move, nine files**, each a row below: `index.js`, `native_import.cjs` and `README.md` in `W:gate/plugin/`
   and in the installed folder `~/.clawdbot/extensions/styxx-gate/`, byte-identical pairs;
   `W:gate/plugin/tests/run_tests.mjs`; and the two new test files. The installed folder is in `gate_sha256`, so the
   move begins a fifth stretch, and the folder's file count goes from six to seven. The candidate is
   `445ac94cb96876181f0a354ff776e6a702cf3064609d4d9a7b2227da2f4e3f1f` (20 files); the renames of the installed README
   and helper pass through
   `0a85eb81…` (19 files) and `cadac375…` (20 files) [R, `F1:f1_move_table.json`
   `gate_sha256_stages`]; the digest of record is the one the lead re-derives from the box after the move, recorded in
   the addendum. The move follows 15.6's rule (the tested bytes placed beside the target, their sha256 checked, one
   `os.replace`, `REQUIRED` absent before, during and after) with the order `BD:apply/apply_move.py` gave it on
   2026-10-26 (the replaced files snapshotted and every file staged and re-hashed before any rename), by
   `F1:apply_f1_move.py` pass 2, which stages outside the folders `gate.py` hashes, refuses while the gateway answers
   on 127.0.0.1:18789, while `REQUIRED` exists, or between 03:30 and 05:20 UTC (checked before staging and again before
   the renames), refuses an installed folder holding any file but the table's set, renames the installed `index.js`
   last, scans for strays and reads the installed file set after a move and a rollback, rolls back only after reading
   every row, and records each rollback [R, `RC:` `move_rehearsals`: on copies, the happy path and two rollbacks,
   faults injected at staging, at a rename and in a rollback, `REQUIRED`, a stray, a file of another name, and the
   box-only refusals run in-process against decoy paths]. The order on the box is: pause the watchdog, stop the
   gateway, move, start the gateway, resume the watchdog; the restarted gateway's `started` line opens the stretch,
   and no 0.4.0 gateway code runs on the new bytes.
9. **`PLUGIN_VERSION` stays `0.4.0`.** The manifest and `package.json` do not move; the stretch's digest names the
   bytes.

## The readings

[R, `RC:` `harness`, `mutations`, `probe_d8`, `sdk_repro`] The plugin harness on the fixed copy (`F1:gate_copy`):
85 of 85, against 79 of 79 in place before the fix (`BD:apply/s6_20261026T134223Z/plugin_harness.txt`).
Nineteen mutations of `index.js`, each source kept with its sha256 in `RC:`: 18 of them fail at least one test, among
them the eight behaviourally wrong survivors of the second review (N1, N3, N4, N6, N7, N8, N13, N14) and the three of
the third (X1, X2, X3), each failing the test written for it; the one that survives, a static import of the helper, is
harmless as observed, since jiti 2.6.1 hands a `.cjs` file to Node's require (`RC:` `probe_d8`). The SDK-level
reproduction reads, after the SDK has gone through jiti,
`import_v040_same_as_native` false, `jiti_runner_is_null` true, `fix_same_as_native` true, `fix_runner_is_the_gateways`
true and `fix_holds_gate` true. The scan and the reproduction are this folder's `anchor_route_scan.py` and
`anchor_sdk_repro.mjs`; the reproduction loads the SDK, whose provider code may try the network with whatever
credentials the environment holds.

## The reviews, answered

`F1:REVIEW_1.md` (one reviewer) found no blocker and two majors. (1) The draft's fix, a `require()` of `deliver.js`,
worked only because jiti's and Node's `require.cache` keys differ on Windows: replaced by item 1, with item 6's
collided reading as its test; observation 5 says so. (2) The default path and a throwing loader had no test: D8 reads
the plugin's own default `getDeliver`, and F13 covers the throw. Its minors: observation 4 is a judgement supported by
the reproductions, and the configured plugin and channel have a receipt; observation 3 now carries the readings it
states; the guard's claim is narrowed (item 2, and the last section); the snapshot and staging order are credited to
`apply_move.py`, not to 15.6; the `missing` reading rests on `RULES` l.305-307 and l.388-390; E2's empty sends are
recorded (item 3); the runner is imported by the URL `deliver.js` uses, and a config `distDir` in another case is
refused under its own code (item 1). Its notes: D8's null runner is read from jiti's cached module, which is the one
jiti's `deliver.js` gets (observation 4 says so); the probe on 0.4.0 shows a missing export, so the mutants carry the
weight; the mutant sources are kept. On note 9: a future top-level await in the graph would leave the helper's
`import()` working, as it is asynchronous; `appendFallback` swallows its own write errors.

`F1:REVIEW_2.md` (four reviewers) found no blocker and four majors. (1) A guard that latches open, and one checked only
at load, passed every test: D8's guard sequence and F15 now read the runner bound, unbound and bound again through one
deliver. (2) The apply script could leave a truncated staged file in the hashed folder, (3) could roll back half way,
and (4) nothing kept old code from writing under the new digest or a move from straddling midnight: the script (item
8) stages outside the hashed folders, registers a staged file before writing it, reads every row before a rollback
writes any, skips rows already back, scans for strays, refuses while the gateway is up and inside the anchor window,
and the order on the box stops the gateway around the move. Its minors: the helper is `.cjs` and D8 lets jiti load it
before Node (the static-import mutant is now harmless, item 1); a config `distDir` in another case gets its own refusal
code; the hookName clause, the throwing reading, the helper's name in D2, a refused day through 00:10 and the helper's
resolution from the plugin folder are each tested; `lock_hashes.py` and the README name seven installed files and the
lock sees strays; the intermediate digests are named (item 8); the dry run refuses under `REQUIRED`; Ctrl+C is caught;
the line references, the wording on F11 and the README, item 3's phases, the window reading below, the receipts'
binding to bytes, and the cause stated as a judgement in the CHANGELOG and the PR body are corrected.

`F1:REVIEW_3.md` (two reviewers, the final texts and the closure of REVIEW_2) found no blocker and no major, and
confirmed every digest, count, stage and line reference it could recompute. Answered: a guard that sends on an
unreadable runner, one that turns a throwing send into an empty one, and one that records an unreadable reading as
unbound each passed 85 of 85; D8's guard sequence now reads an unreadable runner and a throwing send, and the three are
in the mutation set. H1's failure in one run came from the six digits of microseconds in a ledger timestamp, not from a
digest (the test already blanks digests); H1 now blanks timestamps too, so a numeral the draft leaks still fails it.
The apply script now refuses an installed folder holding any file but the table's set, probes the gateway and the
window again right before the renames, verifies and records every rollback, and its box-only refusals are rehearsed
in-process against decoy paths, never the real folders. The scan names the `sources/` folder by a digest and prints
the readings observation 3 states; the receipt's scan command is written out; the RULES retry sentence is quoted, not
paraphrased, and items 2 and 3 say what the plugin adds to it; the windowed reading below is marked a judgement;
REVIEW_2's majors are counted as four; "names" replaces "enables" for the config's key names.

## Whose step

- **The operator's**: the go for the move before D0 (a relock of the window's files, in the manner of the before-D0
  move), and the merge of the pull request that carries this file, after #243.
- **The lead's, on the go**, outside 03:30-05:20 UTC: pause the watchdog, stop the gateway, the move, start the
  gateway, resume the watchdog; then, in place, the plugin harness, the gate's suite, item 3 of the lock-rows
  amendment's strict check with every amendment file including this one (rows 88: the 84 it states and the four new
  paths; every count 0, rc 0), the same check with the PREREG alone (the PREREG's rows reading the moved files CHANGED
  and the new ones UNLOCKED, as they are meant to), `gate_verify.py --rederive`, and this file's addendum, one commit.
  The next 00:00 after the move is the reading that matters: its anchor's send line, or its records in
  `plugin_failures.jsonl`.

## What does not change

`gate.py`, `gate.js`, `patch_dist.js`, the three patched exits, the plugin's manifest and `package.json`; the anchor's
form, its 00:00 timer, its nonce, its retry schedule and its destination check; the text of every rule of `RULES`,
every census row, the gates block and every bar; the PREREG and the five before-D0 amendment files. What this file
adds to the plugin's records (`anchor_send_empty`, and the refusal codes inside `anchor_send_failed`) changes no count
(item 3). A send that is not an anchor never reaches `getDeliver`.

## What the RESULT prints

For the anchor destination, each window day's anchor send line by seq, or its `anchor_send_failed` and
`anchor_send_empty` records; for (iv), the fifth stretch from the seq the addendum records. The two anchors of
2026-10-24 and 2026-10-25 were delivered in shadow before D0 without send lines, and the RESULT names them as this
file's finding. [J] They fall outside the census: the locked census counts a bot message before the window's start as
`outside_window` (`W:gate/reconcile.py` `census_chat`, at `231a03e8…`, the window test applied to every role, the
anchor channel's included), while `RULES` l.499-500, read plainly, keeps the window for the other chats ("In the anchor
channel every message is the census's, whoever sent it; elsewhere the bot's messages ... inside the census window").
The RESULT reads the anchor channel with the census's window and prints both readings of these two messages: outside
the window, or two `[u]` on a day before D0.

## What this amendment does not say

That the reproductions are the gateway: they rebuild its module state from the same files, jiti version and alias, in
another process. That no other code reaches a `dist/` module through jiti: the plugin's wiring check imports
`patch_dist.js`, a plugin file, and reads the dist as bytes, which this file leaves as it is. That the guard makes an
anchor arrive, or that every anchor that leaves has a line: it checks that styxx-gate's handler is registered, not what
the handler will do, and in shadow a handler whose `gate.py check` and `gate.py fault` both fail lets the send go with
no line (`index.js` `failClosed`, recorded in `plugin_failures.jsonl`), as for every other send. That D8 has run on
POSIX: it has run on this box, and its collided readings stand in for POSIX keys. That nothing writes a line during the
move: the script sees only the gateway's port, and a `clawdbot` CLI process that loads the plugin in its own process
can write a line under an intermediate digest; the addendum lists every such line by seq. That the anchors of the two
nights could have been gated by any setting of the box as it stood.

## The lock rows

This file carries 9 rows in `lock_hashes.py`'s row form.

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/plugin/index.js` | `e56990f437134dd26a31db256062d4911ebf7e7be08d2605c4a94c7644d6921f` | 71944 bytes; was `759ceb82…`; `loadNativeDist`, `runnerState`, `guardedDeliver`, `getDeliver`, `anchor_send_empty`, `GATE_SHA_FILES` (items 1-4) |
| gate file | `W:gate/plugin/native_import.cjs` | `b173faea8eca7a1bb71db956299016bcf84bcf8d3a921936dd4097b5cb2bb124` | 1025 bytes; new file, no earlier row; the one native `import()` (item 1) |
| gate file | `W:gate/plugin/README.md` | `49a7dc73dbe76d1b9207a835eba31602b73e88e6d6704a3a482daea9c781257b` | 23831 bytes; was `ed2ac65e…`; the helper, install step 1 and the new tests (item 7) |
| gate file | `W:gate/plugin/tests/run_tests.mjs` | `90ae51c1700277ebe371b6e758eb5d5205ac926be7fb540dbbc65bf3504ef23f` | 113560 bytes; was `8d4b7db0…`; D2, D8, F11-F15, H1's blanking (item 6) |
| gate file | `W:gate/plugin/tests/f1_native_deliver_probe.mjs` | `e0d0178a12c76195785c45f9fdb218d31bb9b940eedd23c3dc8e57bdcd39011c` | 7901 bytes; new file, no earlier row; D8's probe (item 6) |
| gate file | `W:gate/plugin/tests/fakes/import_deliver_v040.mjs` | `120f0f2eff8dc9242df123475c07ed81dce3ef6d651016300e799517e8172658` | 415 bytes; new file, no earlier row; the 0.4.0 `import()` as a fixture (item 6) |
| plugin as installed | `~/.clawdbot/extensions/styxx-gate/index.js` | `e56990f437134dd26a31db256062d4911ebf7e7be08d2605c4a94c7644d6921f` | 71944 bytes; was `759ceb82…`; the same bytes, in `gate_sha256` (item 8) |
| plugin as installed | `~/.clawdbot/extensions/styxx-gate/native_import.cjs` | `b173faea8eca7a1bb71db956299016bcf84bcf8d3a921936dd4097b5cb2bb124` | 1025 bytes; new file, no earlier row; the same bytes, in `gate_sha256` (item 8) |
| plugin as installed | `~/.clawdbot/extensions/styxx-gate/README.md` | `49a7dc73dbe76d1b9207a835eba31602b73e88e6d6704a3a482daea9c781257b` | 23831 bytes; was `ed2ac65e…`; the same bytes, in `gate_sha256` (item 8) |

## Addendum — filled at apply (blank until then)

Filled by the lead after the move, one commit, nothing above this heading edited.

- the move's UTC time, the gateway's stop and start, and the nine files' times and digests: _blank_
- `gate_sha256` after the move, re-derived from the box both ways, and its file count: _blank_ (the candidate
  `445ac94c…`, 20 files)
- the seq of the earliest line of each kind carrying it (`started`, `send`, `delivered`, and the earliest
  `payload_kind: anchor` send): _blank_
- every line written after the move carrying another digest (the two intermediates of item 8 among them), by seq:
  _none expected_
- the in-place readings (the plugin harness, the gate's suite, the strict check with every amendment file, the
  re-derive): _blank_
- the commit of this addendum, and its UTC time: _blank_
