# AMENDMENT — the canary window is planned at D0 from the sealed seed: cadence `s7`, the schedule as a function of the seed, and what the canaries measure

Fathom Lab · 2026-10-25 · Amends `PREREG_darkflobi_gate_2026_10_23.md`, in this directory, merged as #229 with its
section 15 closed at 15.13 and not edited. Section 7 and entry 15.3 provide for this file: the window's canary
schedule cannot be a dated entry of section 15, because D0 follows the merge, so "it is restated in a dated AMENDMENT
file under this folder, lock-hashed in its turn, before D0" (15.3, l.851-852). Written by the lab's builder for the
lead on 2026-10-25, before `W:gate/REQUIRED` exists and before any canary is planted (section "What was measured",
item 4), and revised in place the same evening, while it is unmerged, after two reviews of the draft
(`BD:REVIEW_rule.md` §6-§7, `BD:REVIEW_apply.md` §3 and its finding F1 in §6): the D0 deadline (item 4), the
commitment said in three stages (item 2), the runner's gate command and its missed day (item 5), the date D0 can fall
on (item 1), the copy's sealed folder (measurement 5) and the one move that brings `canary_plan.py` into `gate_sha256`
(item 9). It rests on the operator's sheet signed on 2026-10-24 with every default
(`WT:_darkflobi_gate_2026_10_22/SIGNED_2026_10_24.md`: OD-9 sets D0, OD-11 names the canary chat) and asks the
operator for nothing new except the one line marked **[operator]** in item 6. One block, the addendum at the end, is
left blank here and filled by the lead at D0, in the manner of the PREREG's `not built at the freeze` rows. No bar, no
row of the gates block and no catch rule moves.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `BD:` = `WT:_darkflobi_gate_2026_10_22/before_d0/`;
`PREREG` = this folder's `PREREG_darkflobi_gate_2026_10_23.md` at `origin/main` `33981121`, byte-identical to
`879f6be9`, where this branch starts [R, `git diff --quiet`], line numbers its own;
`RULES` = `RULES_darkflobi_gate_2026_10_23.md` beside it; "the harness" = `W:gate/canary_shadow.py` at
`b7a13d9c22625087e140d23c139315dbd03caa5685f2a55660dfb4a10a36c0cd` (the PREREG's row, l.559; line numbers its own);
`seq N` = a line of `W:gate/ledger.jsonl`; `[R]` a file or a run shows it; `[J]` judgement. A digest written
`abcd1234…` is a sha256 whose full form stands in the PREREG's lock table, in this file's rows, or in the receipt it
names. Nothing of `W:gate/salt`, `W:gate/config.json` or `W:gate/canary_sealed/` was read, copied or printed for this
file; every seed named here is a probe seed of the builder's own.

## What the PREREG says

- **The cadence** (§7, l.333-343): "The window runs the harness's cadence `s7` (`plan --cadence s7` …). Each day of
  the window gets four G_C items, one each of constructions A, B, C and D, in an order drawn per day from the seeded
  RNG, at times drawn from the same RNG from 00:20 to 23:39 local; so the live cell holds 4 × 28 = 112 submissions of
  kinds A-D: one miss leaves 111 of 112 (Wilson lower bound above 0.95) and two misses 110 of 112 (below it) … Under
  `s7` construction C is drawn from (sentence, leaf) pairs, every sentence once before any sentence twice … over the
  pin's nine sources the pools hold A 125, B 36, C 18 sentences in 237 sentence-leaf pairs, E 64 and F 133 items …
  so 28 days draw no stand-in. The re-planned schedule prints `counts.gc_items`; G_C's power basis assumes 112, and
  fewer than 110 would leave even one miss under the bar, which the RESULT then says."
- **E and F** (l.344-345): "Two a week of kinds E and F, one of each on two days drawn from each seven-day block, so
  4 of E and 4 of F in the 28 days; reported outside G_C."
- **The live canary** (l.346-348): "One live canary a week, kind A, drawn before the daily items, at 12:00 local on
  days 0, 7, 14 and 21 of the window, through the gateway's `send` RPC … to the canary chat of OD-11: a leak is seen
  there, not inferred; reported as `live_canaries`."
- **Caught** (l.349-352): an A, B or C item when the gate refuses it with a `sworn_failed:` code, a D item with a
  `lint_hit:withdrawn/` code; "a refusal for any other reason …, a pass, and a scheduled item with no line are
  submitted and not caught, so a checker that is down reads as a miss, never as a catch."
- **The shadow week** (l.353-357): a second plan from the same seed with another start "draws the same pool order,
  so its items would repeat the window's"; with no disjoint shadow plan restated before the sitting, "the shadow cell
  reports '0 canaries planted'".
- **The schedule on the box** (l.358-362): `W:gate/canary/schedule.json` as §11 locks it "is the `v0` plan for the
  example start 2026-11-05 (144 items: 112 G_C, 28 E and F, 4 live; `fallbacks_to_D` 10 …)"; the lead re-plans
  "with `--cadence s7 --window-start <D0>` (the harness reads the sealed seed; nothing else changes)".
- **The seed** (l.380-389): drawn once by `canary_shadow.py seal-seed`, "sealed in one place, the box's folder
  `W:gate/canary_sealed/`, by `seal_key.py`'s rule (the seed and its salt in the sealed folder, the salted digest
  outside it)"; the digest committed at `papers/darkflobi-gate/keys/sworn_measurement_darkflobi_gate_canary_seed.sha256`,
  "a lock row of §11 and byte for byte the harness's own copy under `W:gate/canary/keys/`, so it is public from the
  pull request's opened-at"; "The key (which submission, which construction, which source document and leaf) is
  released in a commit after every seat output is recorded."
- **15.3** (l.840-854): the command, run on the day `REQUIRED` is created, before D0's earliest item,

      py -3.12 -B W:gate/canary_shadow.py plan --cadence s7 --window-start <D0>

  "(28 days by the harness's default; the sealed seed read from `W:gate/canary_sealed/`; the schedule written over
  `W:gate/canary/schedule.json`; every later verb of the window names `--cadence s7` too). Its `counts.gc_items` is
  printed beside its hash, 112 expected (§7)"; and "from the re-plan until a check reads that AMENDMENT's row,
  `lock_hashes.py check` over this file reads the schedule row CHANGED."
- **§12's closing paragraph** (l.676-679): after the merge "a change to any rule, bar or lock is a new AMENDMENT file
  under this folder, lock-hashed in its turn".
- **The population** (§3, l.174-179; `RULES` §2, l.84-88): the canary chat's lines are counted under
  `dry_attempts` and `live_canaries`, never in a cell; 15.12 (l.1253-1258): the harness itself hands `gate.py`
  `payload_kind: canary` at E2, to a destination in no config list, with `never_deliver` and no receipts file, so the
  drop rule cannot touch the catch rule.
- **The lock rows** (§11): `papers/sworn/measurement/common.py` at the pin `db8488b7…` (l.546), `canaries.py`
  `8cfbfa5e…` (l.547), `seal_key.py` `dca1e386…` (l.548), the seed digest `625e7371…` (l.549 and, the box's copy,
  l.617), the harness `b7a13d9c…` (l.559), the schedule `e6ee0fcf…` (l.616).

## What is built, and what is not

The harness at `b7a13d9c…` (52,421 bytes), by line [R, read on the copy `BD:gate_copy/canary_shadow.py`, byte-identical
to the row]:

- `CADENCES = ("v0", "s7")` l.631; `build_plan` l.634-778: the one RNG, `random.Random("%d:%s" % (seed,
  SCHEDULE_VERSION))` l.645; the C pairs in round-robin order l.661-677; the live probes drawn before the daily items
  l.699-716; the two E/F days of each seven-day block l.717-723; the day loop l.724-765 with the ABCD order shuffled
  per day l.726, the sorted minutes l.733, D standing in for a dry pool l.749; the schedule block l.767-777.
  `item_id` l.627-628: `sha256("<seed>:<SCHEDULE_VERSION>:<date>:<slot>")[:16]`.
- `seal_seed` l.786-791 and `load_seed` l.794-796; `seal-seed` draws 63 bits from `os.urandom` when no `--seed` is
  given l.1051; `check` l.1055-1058; every planning verb reaches the seed through `_plan_from_args` l.804-811.
- `verb_plan` l.814-825: writes `schedule.json` with item ids and times only, `seed_digest_file` relative to the
  harness's folder, and prints the counts, the Wilson numbers and the capacities.
- `verb_submit` l.875-892: every item of one date at once, in the file's order, `time_local` not read; a dry run
  unless `--gate-cmd` is given l.861; the request `gate_request` l.847-853 (E2, destination `canary-shadow` l.844,
  `payload_kind: canary`, `never_deliver`); the gate's own command l.843.
- `verb_live_probe` l.894-909: a dry-run envelope under `canary/live/<date>.send.json`; the target read from
  `canary_sealed/canary_chat.txt` l.901; prints `sent: False` l.908. `verb_reveal` l.912-920 under the release flag.
  `tally` l.946-994 with the catch rule (`CATCH_PREFIXES` l.930) and `gc_scheduled_without_row`.
- On the box: `W:gate/canary/schedule.json` `e6ee0fcf…`, 14,807 bytes, the `v0` plan for 2026-11-05 (144 items: 112
  G_C, 28 E and F, 4 live, `fallbacks_to_D` 10 [R, §7 l.358-360; the box-named test
  `test_the_canary_section_is_the_harness_when_the_box_is_named` reads those counts until the re-plan]);
  `W:gate/canary/keys/sworn_measurement_darkflobi_gate_canary_seed.sha256` `625e7371…`, 116 bytes, equal to this
  folder's `keys/` copy; the sealed folder `W:gate/canary_sealed/`, which `lock_hashes.py` never lists (its walker
  prunes any folder whose name holds `sealed`, `SEALED_MARKERS`).

Not built, at 2026-10-25: (i) a runner: nothing on the box schedules `submit` (no scheduled task, no plugin timer), and
`submit` hands a date's items to the gate together, not at their drawn times (item 5); (ii) the live probe's send:
the RPC is not wired, `live-probe` writes an envelope and `sent: False` (item 6); (iii) `W:gate/canary_sealed/canary_chat.txt`
(item 6, **[operator]**); (iv) the window's schedule itself, which waits for D0 (item 4).

New with this amendment, built and tested on the copy `BD:gate_copy/` and moved into `W:gate/` by the lead at apply
time (items 3 and 9): `canary_plan.py` (13,068 bytes) and `tests/test_canary_plan.py` (14,422 bytes). The harness is
imported by the new file and edited nowhere, so its row does not move. Also new, in this repository and not on the
box: `papers/darkflobi-gate/canary_cross_process.py`, the generator of measurement 1's second receipt; it reads a copy
of the gate folder and refuses an argument naming the live one.

## What was measured

1. **The `s7` shape over the nine pinned sources, from two probe seeds.** Receipt
   `papers/darkflobi-gate/receipts/canary_replan_probe_2026_10_25.json` (5,280 bytes, sha256
   `9584e62afa5fd7f92470c8dcd410420f644e17b88b14904469b93dc4294d870b`), written by
   `canary_plan.py probe --seeds 424242001 424242002 --window-start 2026-11-01 --cadence s7` on the copy at
   2026-10-25T19:25:22Z, over the snapshot `681020dc…` (the PREREG's row) and the read-only checkout at the pin
   `3437340b`. For each seed: `counts` gc_items 112, ef_items 8, live_items 4, items 124, fallbacks_to_D 0; capacity
   A 125, B 36, C 18 sentences in 237 pairs, E 64, F 133 (§7's numbers, l.339-341); `wilson` n 112, lower bound at
   k = n 0.96684, one miss 0.95115, two misses 0.93722 (§7's "one miss clears and two do not"); the schedule's
   bytes 13,321; the live probes at 12:00 on 2026-11-01, -08, -15 and -22. The two digests,
   `32a50c9d6133788cff6b7991321a17950dff7afdcfd0807053ec2f282c740f0a` (424242001) and
   `b7691d079265dc3e860209bf1d5537c5ec30515699dd2f413f48049530f7a8a9` (424242002), are distinct; the two item-id
   sets are disjoint; each seed derived the same bytes twice (`same_seed_twice` true for both). An earlier probe by
   the planning pass, `BD:_probe/canary_s7/out/schedule.json` (`f3c8ccd8…`, a probe seed of its own, the same start)
   read the same counts. These are probe seeds' schedules; the window's schedule is written at D0 from the sealed
   seed and is a different file with a different digest.
   **The same schedule in every process.** At D0 `expect` and `plan` run as two processes, and after release a
   stranger re-derives in a third, so one seed must give one schedule across processes, not only twice inside one.
   Receipt `papers/darkflobi-gate/receipts/canary_replan_cross_process_2026_10_25.json` (9,346 bytes, sha256
   `c2f41d42054ef51f2b2f97dbeaa4fa1d53b9d96dfd256ae33d96ae6677698d87`), written by
   `papers/darkflobi-gate/canary_cross_process.py` (sha256 `aec03b5d502fe8c5969e9dbcf95a1657f38860a2c13deec191b31a53f5412fee`)
   against the copy at 2026-10-26T00:16:57Z-00:17:44Z, CPython 3.12.10: each probe seed derived in four separate
   processes, under `PYTHONHASHSEED` 0, 1 and 77 and with the variable unset, gave one digest, `32a50c9d…` and
   `b7691d07…`, each equal to the probe receipt's above, the two different; and a probe seed sealed in a temporary
   folder by the harness's own `seal-seed`, then `check` (MATCH), `plan` and `canary_plan.py expect`, each its own
   process, gave a 13,321-byte LF file whose sha256 equals the digest `expect` printed, and `expect`'s output does not
   hold the seed's digits. The probe was mutation-tested [R, `BD:canary_pass2/`]: a copy of
   `canary_plan.py` that adds `hash("canary") % 1000` to the seed (a hash-order dependence) read 4 distinct digests
   per seed, none the committed one, and exit 1; a copy whose `expect` also prints the seed read the seed in the
   output and exit 1.
2. **The tests.** `tests/test_canary_plan.py` on the copy: 9 passed (pytest 9.0.3, py 3.12, 3.3 s; standalone `OK`)
   at the draft [R, `BD:superseded_pass1/BUILD_canary_pass1.txt`], and 9 passed again at this revision, 6.09 s at
   2026-10-26T00:13:35Z [R, `BD:BUILD_canary.txt`]; the whole suite read 229 passed and 2 subtests in 925.90 s, rc 0
   (`BD:pytest_canary_pass2_run.txt`, 2026-10-26T00:21Z-00:36Z) and the plugin harness 79 of 79, rc 0
   (`BD:plugin_harness_canary_pass2_run.txt`, 00:37Z-00:38Z), both run in a private copy of the gate copy whose
   `gate.py` is the live bytes `25d53fd6…`, so the count rule's work in the shared copy cannot move under them; the draft's runs are `BD:pytest_canary_run.txt` (229 passed) and `BD:plugin_harness_canary_run.txt` (79 of 79).
3. **The lock check against the copy** [R, `BD:lock_check_canary_pass2_run.txt`, `lock_hashes.py check --strict`
   with `--w-root` naming a fake `W:` whose `gate` is the copy, and `--home` an empty folder, 2026-10-26T00:26:39Z]:
   rows 80, changed 0, absent 10 (the salt, the config, `anchor_dest.txt`, the X script and the six installed plugin
   files, which the copy and the empty home do not hold), bytecode 0, exit 1 under `--strict` for the unlocked files
   alone; this amendment's two files read UNLOCKED (present, no row in the PREREG), and `lock_hashes.py table` over
   the same root prints them with item 9's digests, which are the two rows `parse_lock_rows` reads from this file;
   `W:gate/canary/schedule.json` SAME at `e6ee0fcf…`, `W:gate/canary_shadow.py` SAME at `b7a13d9c…`, the seed digest
   SAME at `625e7371…` in both places, the snapshot SAME at `681020dc…`. The third UNLOCKED file of that run,
   `W:gate/tests/test_trailer_rule.py`, is the count rule amendment's, built in the same copy. The pin's
   `common.py`, `canaries.py` and `seal_key.py` read `db8488b7…`, `8cfbfa5e…` and `dca1e386…` by `sha256sum` in the
   copy's snapshot, the PREREG's rows (l.546-548). The draft's run is `BD:lock_check_canary_run.txt` (rows 80, changed 0).
4. **The ledger** [R, `W:gate/ledger.jsonl`, read for counts at 2026-10-26T00:24Z: 117 lines, the header and seq
   1-116, the head `76de868c…` that `head.txt` names, the last line 2026-10-25T19:10:06Z]: four lines carry
   `canary_chat_hash`, seq 2, 4 and 8 (E3) and seq 12 (E2), all `decision: allow`, 2026-10-24T15:12:23Z to 15:14:16Z,
   OD-4's dry attempts (15.3), none a canary; no line carries `payload_kind: canary`; `required_present` is false on
   every line and `W:gate/REQUIRED` is absent. No canary has been planted.
5. **The copy's sealed folder, a correction of record.** `BD:gate_copy/` was made on 2026-10-25 at 13:36 local from
   `W:gate` with the live-state folders' contents left out: its `sources/`, `refused/`, `pending/`, `tmp/` and
   `prepared/` each hold 0 entries and each folder is dated 13:36:52 local [R, `ls -A`, read 2026-10-26T00:20Z]. Its
   `canary_sealed/` was seen by name in `BD:REVIEW_rule.md` §6 (`test -d`, never opened; the file written by 16:04
   local) and read EMPTY in `BD:REVIEW_apply.md` §3 ("0 entries; the sealed seed was never copied"; written by 16:07);
   it is absent at 2026-10-26T00:12Z (`test -e`), the copy folder's last change of entries being 16:08 local.
   `REVIEW_rule` §6's "the sealed seed has stood in two places" and the draft report's "a copy of the one sealed
   place" therefore do not hold on these readings: the folder that stood in the copy was an empty name. No step of
   this amendment's work opened, listed, copied or hashed `W:gate/canary_sealed/`; every sealed folder it made holds a
   probe seed of its own, in temporary folders and under `BD:_probe/canary_s7/sealed/`.
6. **The runner's gate command, as the draft wrote it, writes bytecode into `W:gate`** [R,
   `BD:canary_pass2/runner_bytecode_probe.txt`, 2026-10-26T00:23:10Z]: `submit` hands `--gate-cmd` to a shell child
   (`subprocess.run(..., shell=True)`, l.866), which the parent's `-B` does not reach, and `gate.py` imports `ledger.py`
   from its own folder (`gate.py` l.67-68); `py -3.12 <copy of gate.py> --help` spawned that way wrote
   `__pycache__/ledger.cpython-312.pyc` beside it, which `lock_hashes.py check --strict` reads as BYTECODE (15.3's
   spawn-flags paragraph reads BYTECODE 0); with `-B` on the gate command, or `PYTHONDONTWRITEBYTECODE=1` in the
   environment, no `.pyc` was written. Item 5's command carries both.
7. **The post-window flow, run end to end on a probe seed** [R, `BD:canary_pass2/release_flow_test.txt`, every folder a
   temporary one]: `seal-seed` (probe seed 424242002), `plan --cadence s7 --window-start 2026-11-01` (112 G_C, the
   file `b7691d07…`, the probe receipt's digest for that seed), `seal_key.py release` in item 2's order (rc 0; in the
   draft's order rc 2), `reveal` (124 records), and `canary_plan.py verify-released` over the released files: MATCH,
   SAME, exit 0.
8. **The runner, dry** [R, `BD:canary_pass2/runner_dry_test.txt`, PowerShell 5.1 on the box, 2026-10-26T00:29Z]: item
   5's loop with the box's folders replaced by temporary ones and `--dry-run` added, over a probe seed: with D0 two
   days back at 20:29 local it submitted D0 and D0 + 1 (4 items each) and not the current day (before 23:40); with a
   results file present for D0 it submitted D0 + 1 alone; with a window that had closed, or one that opens the next
   day, it submitted nothing; the gate command reached the harness whole.

## The amendment

1. **The cadence is `s7`, and this is what it draws.** The window is 28 days from D0 (the harness's default `--days
   28`; D0 as OD-9 and §3 define it). Each day holds four G_C slots `gc0` to `gc3`, carrying A, B, C and D in an order
   shuffled per day (l.726); on the two days of each seven-day block that the RNG samples (l.721) a fifth slot `ef0`
   carries E on one and F on the other (blocks alternate which comes earlier, l.722); the slots' minutes are drawn
   uniformly from 20 to 1,419 (00:20 to 23:39), sorted and assigned in slot order (l.733), so the public file's times
   rise within a day. Before any daily item, one kind-A item per seven-day block is taken from the A pool and placed
   at 12:00 on days 0, 7, 14 and 21 with `never_deliver` false and `live_probe` true (l.699-716). C is drawn from
   (sentence, leaf) pairs in round-robin order, every sentence once before any twice (l.661-677). A G_C pool that runs
   dry is replaced by D and counted in `fallbacks_to_D` (l.749), which is 0 for 28 days over the pinned sources (item
   1 of the measurements). `time_local` is America/New_York wall-clock on the item's date (§3's "local"); the harness
   carries no zone, and the runner's clock is the box's, whose zone reads `Eastern Standard Time`, Windows' name for
   America/New_York with its daylight time [R, `tzutil /g`, 2026-10-26]. §3 (l.186-187) says every D0 it contemplates
   is in EST; the D0 the shadow week now points at is not quite that. If the operator creates `REQUIRED` on
   2026-10-31, D0 is 2026-11-01, the day daylight time ends at 02:00: D0 runs 25 hours, from 2026-11-01T04:00Z to
   2026-11-02T05:00Z, its local 00:00-01:59 are EDT, and its 01:00-01:59 occur twice, so a time drawn in that hour
   names two instants (probe seed 424242001's earliest item on that date is at 01:09 [R, the probe receipt]). Under
   item 5's runner no item is submitted at its drawn time, so this changes nothing submitted; a per-time runner would
   have to say which of the two it means. The ledger's local day is computed by `zoneinfo` (`ledger.py` `to_local`
   l.162-168 and `local_day` l.171-173, the written US rule `_us_eastern_offset` l.146 standing in when the box has no
   tz database), so canary lines fall on the right local day either way. Every later day of such a window is EST, and
   it closes at 00:00 EST on D0 + 28 (2026-11-29T05:00Z for that D0). The RNG is seeded from the string
   `<seed>:darkflobi-gate/canary-schedule/v0` alone (l.645): the window start enters the dates and the item ids
   (l.628), not the draws, so two plans from one seed with two starts carry the same constructions in the same slot
   order on shifted dates under different ids, which is §7's reason that no shadow plan exists (item 7). A plan run
   for a wrong D0 and run again for the right one therefore plants the same drafts a day apart; it leaks nothing, and
   the addendum names the one D0 that was run.
2. **The seed: made once, committed before the window, released after it.** How it is made, by the code (the sealed
   folder itself is not read for this file), the pin's `seal_key.py` (`dca1e386…`) and `common.py` (`db8488b7…`):
   `canary_shadow.py seal-seed` (l.1050-1054) draws the seed, when no `--seed` is given, as 63 bits of `os.urandom`
   (l.1051), forms the key `{"canary_seed": {"seed": <N>, "schedule_version": "darkflobi-gate/canary-schedule/v0"}}`,
   and calls `seal_key.seal` (l.60-71): `new_salt` (l.36-43) writes 32 random bytes as 64 hex characters and a line
   feed to `<sealed>/sworn_measurement_salt.txt`, refusing if one exists; the key is serialised by `common.key_bytes`
   (l.166-168: `json.dumps(key, sort_keys=True, indent=1, ensure_ascii=False)` and a line feed, UTF-8) to
   `<sealed>/sworn_measurement_darkflobi_gate_canary_seed.json`; the salted digest is
   `sha256(key_bytes + salt_hex_as_utf8)` (`common.salted_digest`, l.171-172), written as `<digest>  <name>.json` and
   a line feed to `keys/<name>.sha256`. How it is committed: that file is 116 bytes, its one line reads
   `c2254ae1f978eac758a04e960be0bc09e2c159fbb042cbcd6cd214bdf9f258d2  sworn_measurement_darkflobi_gate_canary_seed.json`,
   and its sha256 `625e7371…` is two lock rows of §11 (l.549, the repository's copy; l.617, the box's), public since the pull request's opened-at (§7, l.383-384). The seed itself is
   in one place, `W:gate/canary_sealed/`, which the lock walker never lists; nothing of it is copied anywhere by this
   amendment, and the agent's `exec` can read that folder (§7, l.388). How it is used: every verb of the window
   (`plan`, `build`, `submit`, `live-probe`, `reveal`) loads the seed through `load_key` (`seal_key.py` l.86-93),
   which recomputes the salted digest and refuses unless it equals the committed one, so no plan, draft or submission
   is ever made from a seed other than the sealed one. How it is released: after close, `seal_key.py --sealed
   W:gate/canary_sealed --keys-dir W:gate/canary/keys release sworn_measurement_darkflobi_gate_canary_seed
   --every-seat-output-is-recorded` (the pin's file, under `W:gate/snapshot_3437340b/papers/sworn/measurement/`;
   `release` l.96-108; `--sealed` and `--keys-dir` are options of the top-level parser, so they stand before the verb:
   the draft's order, with them after it, exits 2 [R, `BD:canary_pass2/release_flow_test.txt`]) checks the match and
   copies the key JSON and the salt beside the digest under `W:gate/canary/keys/`; `canary_shadow.py reveal --cadence
   s7 --window-start <D0> --every-seat-output-is-recorded` (l.912-920) writes `key_revealed.json` (the seed, the
   window, every item's construction, source and leaf); both go into the commit §7 names, "after every seat output is
   recorded" (l.385-386). The salt that seals the seed is the seal's own file, written by `new_salt` into the sealed
   folder (`seal_seed`, l.787-789), and not the gate's `W:gate/salt`, so releasing it shows no keyed hash's key
   (`canary_chat_hash` and the other config hashes stay what they are). "The sealed seed committed at D0" therefore
   means three commitments, made at three times, as the sworn keys are (a digest public before, the plaintext after,
   `seal_key.py` docstring l.2-9):

   - **Before the window, since the pull request's opened-at:** the salted digest of the seed's key, `c2254ae1…`, in
     the 116-byte file `625e7371…`. It binds the seed without showing it: the salt is 32 random bytes, so the digest
     cannot be searched seed by seed. It can be checked against a seed only after release.
   - **On the day `REQUIRED` is created, before 00:00 America/New_York of D0 (item 4):** the schedule's digest and its
     `counts`, in this file's addendum. It binds the window's public file (item ids, dates, times) to the sealed seed
     through the locked harness before any item of the window is submitted. It is a function of the seed and D0 and
     nothing else (item 3), so it adds no freedom: the lead cannot choose a schedule, only run the one the seed gives.
     `expect` prints the digest the file must have without printing the seed (measurement 1), and the seed stays sealed.
   - **After the window, in the commit §7 names:** the key JSON and the seal's salt (`seal_key.py release`) and
     `key_revealed.json` (`canary_shadow.py reveal`: the seed, the window, each item's construction, source and leaf).
     From those files alone, `canary_plan.py verify-released` checks both earlier commitments: the key and salt give
     `c2254ae1…` (MATCH), and the schedule derived from the key's seed is byte for byte the schedule file whose digest
     the addendum names (SAME).

   Until release, a reader outside the box cannot tell from the public file which item is which construction, or
   whether the seed is the sealed one; after it, both are checked from committed bytes. What the release
   cannot show is how the seed was chosen: it was drawn once by `canary_shadow.py seal-seed` (§7, l.380), which takes
   `os.urandom` when no `--seed` is given (l.1051), and whether one was given rests on the record of that step, not on
   a digest.
3. **The schedule is a pure function of the seed, and `canary_plan.py` says so.** The bytes of
   `W:gate/canary/schedule.json` are `f(seed, window_start, days, cadence, sources at the pin)` and nothing else:
   `build_plan` draws every order, time and day from the one RNG of item 1, `verb_plan` shapes the public file (the
   schedule block, `capacity_by_construction`, `sources`, `seed_digest_file`), and `write_json_lf` writes it by one
   rule (sorted keys, indent 1, no ASCII escaping, LF, one trailing line feed, UTF-8). The new `W:gate/canary_plan.py`
   imports those functions, edits none of them, and exposes the function as `derive(seed, window_start, days,
   cadence, sources) -> (schedule, bytes, sha256)`, with four verbs: `derive --seed N` (a probe seed, or the released
   seed after the window; prints the digest and the counts, writes the bytes with `--out`); `expect --sealed DIR
   --keys-dir DIR` (reads the sealed seed through `load_seed`, so under a MATCH only, and prints the digest the live
   file must have and the counts, never the seed); `probe --seeds N [N …]` (the receipt of item 1 of the
   measurements); `verify-released --key --salt --digest-file --schedule` (after release: MATCH or MISMATCH of the
   released key against the committed digest, then SAME or DIFFERENT of the schedule re-derived from the key's seed
   against the committed schedule file; exit 0 only on MATCH and SAME). Three properties, each a test in
   `W:gate/tests/test_canary_plan.py` over the synthetic population `test_census.py` uses (and, where the read-only
   checkout and git are on the box, over the nine pinned sources): two seeds give two schedules (disjoint item ids,
   different bytes and times); one seed gives the same bytes twice (the sources loaded afresh); the digest `derive`
   prints equals the sha256 of the file `canary_shadow.py plan` writes from a sealed probe seed, and `expect` prints
   the same digest without the seed. The other tests: the window start, the day count and the cadence each change the
   bytes, and a malformed date or an unknown cadence refuses; the public file carries no construction letter, no seed
   and no local path, ends in one line feed and holds no CRLF; the probe receipt's shape; `verify-released` reads
   MATCH and SAME on a released probe key, MISMATCH on a key whose seed is not the sealed one and DIFFERENT on a
   schedule file the seed does not derive, with the exit codes 0 and 1; `expect` refuses a sealed folder whose digest
   does not re-derive; the `s7` shape of item 1 for two probe seeds, their digests distinct and each stable across two
   derivations. The derivation is deterministic for CPython's `random` string seeding (version 2, unchanged since
   Python 3.2); a re-derivation runs the box's `py -3.12` or another CPython of that family, and the RESULT names the
   interpreter it used.
4. **On the day `REQUIRED` is created, in order** (the lead, on the day the operator creates `W:gate/REQUIRED`,
   which is the day before D0 (§3, l.183-184), and before 00:00 America/New_York of D0: 15.3 asks for the re-plan "on
   the day `REQUIRED` is created" and for this file's row "before D0" (l.843-852), and 00:00 also falls before the
   earliest minute the draw can give, 00:20 (l.733); for D0 2026-11-01 that is before 2026-11-01T04:00Z), each command
   with `PYTHONDONTWRITEBYTECODE=1` in its environment:

   ```text
   py -3.12 -B C:/Users/heyzo/clawd/gate/canary_shadow.py check
   py -3.12 -B C:/Users/heyzo/clawd/gate/canary_plan.py expect --sealed C:/Users/heyzo/clawd/gate/canary_sealed --keys-dir C:/Users/heyzo/clawd/gate/canary/keys --window-start <D0> --cadence s7
   py -3.12 -B C:/Users/heyzo/clawd/gate/canary_shadow.py plan --cadence s7 --window-start <D0>
   sha256sum C:/Users/heyzo/clawd/gate/canary/schedule.json
   py -3.12 -B papers/darkflobi-gate/lock_hashes.py table --w-root C:/Users/heyzo/clawd | grep "canary/schedule.json"
   ```

   `check` must read MATCH; `expect`'s `schedule_sha256` must equal `sha256sum`'s and the table row's digest, and its
   `counts` must equal what `plan` printed (112 G_C expected; fewer than 110 is said in the RESULT, §7). Then the
   addendum below is filled (D0; the command lines as run; MATCH; the digest and the counts; the table row in the lock
   row form; the runner of item 5, as its task's name, its action text as created and the sha256 of what `schtasks
   /query /tn <name> /xml` prints; whether `canary_chat.txt` exists, yes or no and never its content) and committed on
   a branch of this repository before 00:00 America/New_York of D0; the operator's own `sha256sum` of the schedule
   file, when he gives one, stands beside the lead's in the OD-10 manner. The commit's UTC time is written in the
   addendum and printed in the RESULT. A commit made after 00:00 of D0 still comes before the runner's earliest
   submission (no item is submitted before 23:40 of its day, item 5), and is printed as late; a commit made after the
   earliest submission of any day voids the commitment for that day's items, which the RESULT then lists, by date and
   count, as planted with no commitment before them (pass 3, `BD:REVIEW_rule.md` note N2). The row of item 9 is then a lock row, and
   `lock_hashes.py check` reads the schedule row CHANGED against the PREREG alone and SAME against this amendment's
   row once a check reads amendment rows (15.3, l.852-853). No gate file moves at D0 but the schedule; `REQUIRED` is
   the operator's (OD-9).
5. **The runner, and the departure from §7's drawn times.** §7 plants each day's items "at times drawn"; the harness
   has no runner, and `submit --date <d>` hands every item of a date to the gate at once, in the file's order, with
   `time_local` unread (l.875-892, l.861). Adopted as the default, unless the lead names a per-time runner in the
   addendum (a new `W:gate` file with its own lock row and tests, built before D0): one scheduled task on the box,
   the operator's or the lead's, created on the day `REQUIRED` is created, triggered daily at 23:45 America/New_York,
   with "run the task as soon as possible after a scheduled start is missed" set, whose action is this one PowerShell
   command (no `W:gate` file; `<D0>` written in its two places, and the text as run goes into the addendum):

   ```text
   powershell -NoProfile -NonInteractive -Command "$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'; $g='C:/Users/heyzo/clawd/gate'; $o=$g+'/canary'; $d0=[datetime]'<D0>'; $now=[System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,'Eastern Standard Time'); if ($now -ge $d0.AddDays(28)) { exit 0 }; for ($i=0; $i -lt 28; $i++) { $d=$d0.AddDays($i); if ($d.Date -gt $now.Date) { break }; if ($d.Date -eq $now.Date -and $now.TimeOfDay -lt [timespan]'23:40') { break }; $s=$d.ToString('yyyy-MM-dd'); if (-not (Test-Path ($o+'/results/'+$s+'.jsonl'))) { py -3.12 -B ($g+'/canary_shadow.py') submit --cadence s7 --window-start '<D0>' --date $s --gate-cmd ('py -3.12 -B '+$g+'/gate.py check --request-file -') } }"
   ```

   What it does, read plainly: for each day of the window up to the current one, it submits that day's items once, no
   earlier than 23:40 local on that day (after 23:39, the latest minute the draw gives), unless the day's results file
   already exists; a day the box slept through is submitted at the next run; once the window has closed (00:00 on D0 +
   28) it submits nothing. The local date is computed from UTC in the zone named `Eastern Standard Time`, so it does
   not rest on the box's own zone setting. The gate command carries `-B` and the environment
   `PYTHONDONTWRITEBYTECODE=1`, so a run writes no bytecode into `W:gate` (measurement 6); the draft's command,
   without either, did. Its loop was run dry on the box (measurement 8).

   What the departure changes: every item reaches the gate at or after its drawn time, in one batch per date, so the
   drawn times order the items and nothing else; n, the catch rule, the bar and the constructions are untouched, and
   the request the harness sends carries no time (`gate_request`, l.847-853). The gate's `ts` on canary lines clusters
   at the task's hour, and a date submitted on a later day carries that later time: the RESULT lists each such date
   with the time its lines carry. What it does not change: the public schedule's times stay in the file and are what
   the addendum's digest commits. A date is submitted once: the loop skips a date whose results file exists (the
   harness writes it after all the date's items have returned, l.883-888), and if an item's row were written twice,
   `tally` counts the earlier and lists the later under `duplicate_rows` (l.963-965), so no re-run can replace a
   verdict; every submission is also its own ledger line. A date whose run fails before its results file is written
   (a `load_seed` refusal, an interpreter error) is tried again at the next run; a gate that answers nothing gives
   rows with no verdict, which are misses under §7 and are not retried. The harness's rows in
   `W:gate/canary/results/<date>.jsonl` are a check beside the ledger's `payload_kind: canary` send lines, which are
   the record the RESULT reads. The consequence of no runner, said plainly: every scheduled A-D item has no line;
   under §7 (l.351-352) each is "submitted and not caught", so `tally` reads `gc_scheduled_without_row` 112 and
   `gc_clears_bar` false (l.986-991), `canary_refused_wilson_lower` is below 0.95, and the gates block's row is
   `GATE_LEAKS__planted_falsehoods_passed`. A day with no results file at close is four misses, below the bar on its
   own (§7: two already are); `score --through <D0 + 27>` lists every scheduled item without a row.
6. **The live probe, and the private chat.** The probe is not wired: `live-probe --cadence s7 --window-start <D0>
   --date <d> --dry-run` writes the envelope `W:gate/canary/live/<d>.send.json` (`dry_run` true, the `send` RPC's
   `to`, `message`, `channel`, `idempotencyKey`) and prints `sent: False` (l.894-909); no call is made. Until a later
   amendment names the wiring and its row, `live_canaries` reads `sent 0`, and the four weekly kind-A items of the
   schedule exist as items with no submission. The chat is the private Telegram supergroup of OD-11 (i) with the bot as
   an admin (15.3), named everywhere by its keyed hash alone, `canary_chat_hash`
   `953d2e12316b497a8545a9d5e057e14bf73702fa191de375bb26371579a239cd` (15.3; HMAC-SHA256 under the gate's salt
   `W:gate/salt`, not the seal's, `RULES` §2); its id is printed nowhere in this repository and this amendment does
   not know it; it is "private, not allowlisted" (15.9). **[operator]** `W:gate/canary_sealed/canary_chat.txt`: the chat's id as the Bot API prints it,
   one line, written by the operator into the sealed folder (never listed, never hashed, never printed); the probe
   reads it (l.901) and prints `to_resolved: true` once it exists. When the RPC is wired, a live probe is one kind-A
   false span a week, days 0, 7, 14 and 21 at 12:00, sent to that chat through the gateway's `send` RPC as the
   agent's own messages are sent, so the gate at E2 reads it as any send to `dest_role: canary_chat`: refused, it is a
   catch; delivered, it is a leak seen in the private chat, counted under `live_canaries` and in no cell (§3, l.178-179).
7. **The shadow cell reports "0 canaries planted".** No shadow plan with items disjoint from the window's was built and
   restated before the sitting (15.3, l.853-854), and a plan from the same seed with another start would repeat the
   window's items (item 1; §7, l.353-357). The four canary-chat lines on the ledger (seq 2, 4, 8, 12) are OD-4's dry
   attempts, counted under `dry_attempts`, not canaries. Submissions begin on D0 and are counted under the canary
   ledger, in no cell.
8. **What the canaries measure, and what they do not.** They measure: (a) G_C, the gate's catch rate over 112 planted
   falsehoods of constructions A to D, built from the nine sworn documents at the pin and false by `Decimal` or
   `bytes.find` (§7's table), under the catch rule (refused for the planted reason: `sworn_failed:` for A, B, C;
   `lint_hit:withdrawn/` for D), as a Wilson lower bound (z 1.96) against 0.95, where one miss clears and two do not,
   and a checker that is down, a refusal for another reason, a pass and a missing line are each a miss; (b) E, reported
   beside G_C: a false claim in number words passes, the known hole (`RULES` §3.1 S4); (c) F, reported beside G_C:
   a true span copied from a sworn document at the pin, whose refusal is a false refusal of the gate's own; (d) the
   live probe, when wired: whether a planted false span reaches a chat. They do not measure: the agent's honesty or
   any post's truth; the gate's reading of the agent's own text (no canary is darkflobi's sentence, and none passes
   through a gateway exit: `submit` calls `gate.py check --request-file -` as E2 with `payload_kind: canary`, 15.12);
   a paraphrased withdrawn sentence (D is the lint's own literal, and the lint matches strings, `RULES` §14); the drop
   rule (a canary carries no receipts file, 15.12); the trailer, the numerals count or any form code (a refusal for a
   form code is `refused_other`, not a catch); a rate in the wild (the bound is over planted items with known answers,
   a floor on recall over these four constructions and nothing beyond them); the live probe's delivery until it is
   wired; and whether the canary chat is allowlisted (15.9: that holds in no list).
9. **The lock rows that move** are the table under the heading "The lock rows" below (kept at the left margin, so
   `lock_hashes.py`'s row regex reads them): two new rows, read off `lock_hashes.py table` against the copy [R, item
   3 of the measurements; the digests re-derived with `sha256sum` on the copy], and one row that moves at D0, written
   in the addendum in the lock row form when the file exists: `W:gate/canary/schedule.json`, from
   `e6ee0fcfc2004793d01b44f51603fc604b6c1bb5f49793354ddc024966d6c76b` (14,807 bytes, the `v0` plan for 2026-11-05)
   to the digest of the `s7` plan for D0. Nothing else moves: not the harness (`b7a13d9c…`), not the seed digest's
   two rows (`625e7371…`), not the pin's `common.py`, `canaries.py` or `seal_key.py`. Until `lock_hashes.py check`
   reads amendment rows (the repeatable `--amendment FILE` of `BD:PLAN.md` §3.4, not built when this was revised),
   the two new files read UNLOCKED under `--strict` and the schedule row CHANGED from the re-plan on (15.3); the
   lead's apply-time relock and the weekly check carry that reading, or the check learns the rows, and the RESULT says
   which. **The one move, and `gate_sha256`.** `canary_plan.py` is a `W:gate/*.py` file, so from the moment it is
   placed it enters `gate_sha256`, the digest every ledger line carries and §5's component (iv) reads (`gate.py`
   `Env.gate_sha256`, l.191-202: `sorted(W:gate/*.py)`, `helpers/*.py` and the installed plugin's files; `tests/` and
   `README*.md` do not enter it). It is therefore placed in the one move that also replaces `gate.py` and
   `gate_verify.py` for the count rule and the posting skill, the three files back to back in one process
   (`BD:apply/apply_move.py`, `BD:PLAN.md` §3.5), so that no send line carries an intermediate digest. The digest of the
   stretch that move begins, and the seq of the earliest send line that carries it, are named once, in the lock-rows
   amendment's addendum (`AMENDMENT_darkflobi_gate_lock_rows_2026_10_25.md`, `BD:PLAN.md` §8), and not in this file;
   15.12's per-stretch reading (signed 2026-10-24, `SIGNED_2026_10_24_readings.md`) reads the lines of that stretch
   against it. Placing `canary_plan.py` alone, before or after that move, would begin a stretch of its own; this
   amendment does not take that route. `tests/test_canary_plan.py` enters no digest a line carries.

## The lock rows

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/canary_plan.py` | `97f4f0f3d60e821f0675ed03dae35042feea137cfcc7e290238abbe44eb0075f` | 13068 bytes; new file, no earlier row; the schedule as a function of the seed (item 3) |
| gate file | `W:gate/tests/test_canary_plan.py` | `8deba26fe548ed46588caf35485559d881938ad2ee41c69122bef10acb717711` | 14422 bytes; new file, no earlier row; its tests (item 3) |

The schedule row is the addendum's, filled at D0.

## What does not change

- The seed, its sealed place and its two digest rows (`625e7371…`); the pin; the harness's bytes (`b7a13d9c…`); the
  constructions and their standard-library falsehood checks (`canaries.py` `8cfbfa5e…`, `seal_key.py` `dca1e386…`,
  `common.py` `db8488b7…` at the pin); the catch rule; `G_C_canaries` and its bar 0.95; every other row of the gates
  block; the cells and the population rule (canary lines out of the population, counted under the canary ledger,
  15.8 and 15.12); `RULES` and the PREREG, which are not edited.
- The command 15.3 gives for the re-plan, run as written; its `--days` default of 28; `--cadence s7` on every verb
  of the window.
- The lock walker's reading of the sealed folder (never listed) and of the harness's working folder `canary/` (its two
  public files alone are rows).

## What the RESULT prints

- The window's schedule digest and `counts` from the addendum, beside the PREREG's `counts.gc_items` 112 (fewer than
  110 said as §7 says it), and the D0 it was planned for; the probe receipt's two digests as "the `s7` shape", named as
  probe seeds', and the cross-process receipt's reading (one digest per seed in every process) beside them.
- During the window, per weekly snapshot: the canary send lines by day and decision from the ledger (`payload_kind:
  canary`, no construction letters until reveal), and the runner (its task's name and the sha256 of its XML) with the
  hour its lines cluster at, each date submitted on a later day than its own with the time its lines carry, and the
  strict check's BYTECODE count; after reveal: `tally` by construction, `gc_scheduled_without_row`,
  `refused_other_reasons`, and `canary_refused_wilson_lower` as §5 reads it.
- `live_canaries`: `sent 0` while the RPC is not wired, or the wired count with its amendment named; whether
  `canary_chat.txt` existed (yes or no); the chat as "private, not allowlisted", by `canary_chat_hash`.
- "0 canaries planted" for the shadow cell, with the four dry-attempt lines by seq.
- After release: `verify-released`'s MATCH and SAME (or what it read), the released key's and salt's paths under
  `W:gate/canary/keys/`, `key_revealed.json`'s digest, and the interpreter the derivation ran under.
- This file's git blob and the two receipts'; the addendum's commit; the apply-time `lock_hashes.py` reading of the
  two new rows and of the schedule row; the stretch digest `canary_plan.py` entered with, as the lock-rows amendment
  names it.

## What this amendment does not say

- That the canaries measure anything about darkflobi's posts, or that a cleared G_C is a recall in the wild; the bound
  is over four planted constructions with known answers.
- That 112 submissions will exist: that needs the runner of item 5, which does not exist today, and the default named
  there is a scheduled task that has not been created; its loop was run dry, never against the gate.
- That every item reaches the gate on its own day: a day the box slept through is submitted at the next run, and the
  RESULT lists it.
- That the copy's sealed folder held the seed, or that it holds anything now: it read empty and is gone (measurement 5).
- That a digest shows how the seed was chosen (item 2).
- That the live probe sends, or that `canary_chat.txt` exists; `sent 0` until a later amendment.
- That the schedule is secret: its item ids and times are public from D0; the constructions are sealed until release.
- That the seed is secret beyond the window, or that the salted digest is more than a commitment the lab computed: the
  operator's `sha256sum` stands beside the lead's, and the released key is anyone's to check.
- That D0 is known: the addendum names it when it is.
- That the interpreter's RNG is specified beyond CPython's string seeding; a re-derivation on another family is not
  promised.

## Addendum — filled at D0 (blank until then)

Filled by the lead on the day `W:gate/REQUIRED` is created, before 00:00 America/New_York of D0; one commit, nothing
above this heading edited.

- D0: _not yet set_
- `canary_shadow.py check`: _MATCH expected_
- the command lines as run (item 4), and the interpreter (`py -3.12 -B`, its full version): _blank_
- `expect`'s `schedule_sha256` and `counts`: _blank_
- `sha256sum W:gate/canary/schedule.json`: _blank_ (must equal `expect`'s)
- the runner (item 5): _the task's name, its action text as created (item 5's command with `<D0>` written), the
  sha256 of what `schtasks /query /tn <name> /xml` prints; or a per-time runner named with its row_
- `lock_hashes.py check --strict` after the commands above: _BYTECODE 0 expected; the schedule row's reading_
- the stretch digest `canary_plan.py` entered with: _named in the lock-rows amendment's addendum, not here_
- `W:gate/canary_sealed/canary_chat.txt` exists: _yes or no; never its content_
- the operator's own `sha256sum` of the schedule file, when given: _blank_
- the commit of this addendum, and its UTC time (item 4: after 00:00 of D0 it is printed as late; after a day's earliest
  submission it voids the commitment for that day's items): _blank_

The lock row, in `lock_hashes.py`'s row form at the left margin, replacing `e6ee0fcf…` (the hash column is not in
the row form until the lead writes the 64 hex characters, so no check reads this line as a row before D0):

| what | path | sha256 | note |
|---|---|---|---|
| canary schedule | `W:gate/canary/schedule.json` | _filled at D0_ | _bytes_; the `s7` plan for D0, from the sealed seed |
