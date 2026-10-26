# AMENDMENT — the lock rows: `lock_hashes.py check` reads this folder's amendment files, and the fourth `gate_sha256` stretch is named here

Fathom Lab · 2026-10-25 · Amends `PREREG_darkflobi_gate_2026_10_23.md`, in this directory, merged as #229 with its
section 15 closed at 15.13 and not edited. Section 12's closing paragraph provides for this file: after the merge "a
change to any rule, bar or lock is a new AMENDMENT file under this folder, lock-hashed in its turn, with deliveries under
the old rule counted under the old rule". Written by the fix pass (pass 3) of the before-D0 work, on the lead's pass-2
plan (`BD:PLAN.md` sections 3.4, 3.5 and 8, its phase G), after the two pass-2 reviews named it owed
(`BD:REVIEW_rule.md` finding F2, `BD:REVIEW_apply.md` finding F1), before D0 and before `W:gate/REQUIRED` exists. The
operator has signed nothing for it; "Whose step" lists what is his and what is the lead's. No bar, no cell, no row of
the gates block, no reason code, no rule and no lock row's digest moves by this file: it says how the rows the other
four amendments give are read, and it is the one place the fourth `gate_sha256` stretch is named.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `BD:` = `WT:_darkflobi_gate_2026_10_22/before_d0/`;
`seq N` = a line of `W:gate/ledger.jsonl`; `[R]` a file or a run shows it; `[J]` judgement. A digest written
`abcd1234…` is a sha256 whose full form stands in the PREREG's lock table, in an amendment's lock rows, or in the
receipt named. The live gate `W:gate` was read and never written, and `lock_hashes.py check` was not run against it
here (the check hashes `W:gate/salt` and `W:gate/config.json` for their rows, and the live reading is the lead's at
apply): every reading below is against the copy `BD:gate_copy/`, through `BD:_lockroot_skill/` (a folder whose `gate`
is a junction to the copy and whose `skills/styxx-post/SKILL.md` is the skill as it will be placed) with the empty
home `BD:_lockhome/`.

## What the PREREG says

- §9, the weekly operator hash (OD-10): "`lock_hashes.py check --strict` beside this file re-derives every lock row
  from the box for the same purpose and names as `UNLOCKED` every file under `W:gate/` the lock covers that no row
  names (…), and as `BYTECODE` every `__pycache__` folder and `.pyc` file under `W:gate/` (…), each a failure of the
  check".
- §11: `lock_hashes.py check --strict PREREG` "re-derives every row and fails on one that changed, was built since or
  is gone"; 15.3: `lock_hashes.py` reads the rows under §15 as it reads §11's, the last row for a path winning.
- §12's closing paragraph, above: a change after the merge is an AMENDMENT file, lock-hashed in its turn.
- 15.12, signed in 15.13: component (iv) of §5 is read per `gate_sha256` stretch, "each send line against the digest of
  the files locked when the line was written"; "a line carrying a digest other than its stretch's, or any fourth digest,
  counts into (iv)". It names three stretches: `387fc063…` (§11), `592dc579…` (15.6, no line) and `ff34b52d…` (15.10).

## What was observed

1. **`check` read the PREREG alone.** At `879f6be9`, where this branch starts, `lock_hashes.py` (blob `f5ffe4dc`)
   parsed the rows of the one file it was handed; its CLI took `prereg`, `--w-root`, `--home`, `--strict`. Against
   the copy carrying the before-D0 change it read rows 80, changed 6, built since 1 (the skill), absent 10, unlocked
   4, bytecode 0, rc 1 (`BD:lock_check_review_pass2.txt`), the same reading it would give if those files had been
   edited by hand, so the weekly tripwire could not tell an amended file from an edited one.
2. **The four amendments' rows** [R, `parse_lock_rows` over each file at this branch]: the canary re-plan 2, the count
   rule 4, the lint and persona 0, the posting skill 7: 13 rows over 11 paths, the two shared paths (`W:gate/gate.py`,
   `W:gate/gate_verify.py`) carrying one digest in both files. Their union is exactly the six files the check read
   CHANGED, the four it read UNLOCKED, and the skill (`BD:REVIEW_apply.md` §3; `BD:REVIEW_rule.md` §7).
3. **`gate_sha256` across the move** [R, `BD:gate_sha256_combos_review_pass2.txt`, by 15.6's rule without importing
   `gate.py`]: today `ff34b52d…` over 18 files; with `canary_plan.py` added and `gate.py` and `gate_verify.py`
   replaced, `c5084df2ef574658fe84f84feea0b70a1169eb695a882f1d4237b439d3a418e8` over 19 files; six intermediate digests
   for the six partial states, each of which the move's tool must keep off every line.

## The amendment

1. **`check --amendment FILE`, repeatable** (`lock_hashes.py`, this branch's commit `8e93e3b9`). The rows of each
   amendment file, read by the same `parse_lock_rows` as the PREREG's, are applied after the PREREG's in the order the
   files are given, so an amendment's row for a path replaces the PREREG's (its §11 row or a later §15 row alike).
   Each file read prints one line, `AMENDMENT <name> rows <n> sha256 <16 hex>` (the sha256 of its LF bytes, so a
   reading names the text it read); a row an amendment supplied prints `via=<name>`, or `via=<a> + <b>` when two files
   name it with one digest. A path two files name with two digests, or one file names twice with two, is a **CLASH**:
   both names and both digests are printed and `check` exits 2 before any row is read, because no reading of the box
   can say which of the two is the freeze. A missing amendment file exits 2, never a silent PREREG-alone reading; a
   file with no rows is read and counted (`rows 0`). Without `--amendment` the reading is the PREREG's alone, as before.
   Since no path may carry two digests, the order of the files cannot change a reading; the command below lists them
   in name order.
2. **The files and their rows.** Every AMENDMENT file of this folder, with the number of lock rows its own lock-row
   table states (a test, `test_each_amendment_parses_to_the_lock_rows_its_table_states`, reads this table and each
   file, and fails when a new amendment file is not listed here):

   | file | lock rows | the paths |
   |---|---|---|
   | `AMENDMENT_darkflobi_gate_canary_replan_2026_10_25.md` | 2 | `W:gate/canary_plan.py`, `W:gate/tests/test_canary_plan.py` (the schedule's row joins at D0, in that file's addendum) |
   | `AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md` | 4 | `W:gate/gate.py`, `W:gate/gate_verify.py`, `W:gate/README.md`, `W:gate/tests/test_trailer_rule.py` |
   | `AMENDMENT_darkflobi_gate_lint_overclaims_2026_10_25.md` | 0 | none: the lint the gate runs is the pin's |
   | `AMENDMENT_darkflobi_gate_lock_rows_2026_10_25.md` | 0 | none: this file |
   | `AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md` | 7 | `W:gate/gate.py`, `W:gate/gate_verify.py` (the same digests as the count rule's), `W:gate/README_exhibit.md`, `W:gate/tests/test_prepared_guard.py`, `W:gate/tests/test_gate.py`, `W:gate/tests/test_gate_round2.py`, `W:skills/styxx-post/SKILL.md` |

   Read together with the PREREG's 80 rows: 84 rows (the four new paths are the three new tests and
   `canary_plan.py`; the other seven replace a PREREG row).
3. **The weekly and the merge-time check name every amendment file** (§9, OD-10), on the box:

   ```text
   py -3.12 -B papers/darkflobi-gate/lock_hashes.py check papers/darkflobi-gate/PREREG_darkflobi_gate_2026_10_23.md --strict --w-root C:/Users/heyzo/clawd --amendment papers/darkflobi-gate/AMENDMENT_darkflobi_gate_canary_replan_2026_10_25.md --amendment papers/darkflobi-gate/AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md --amendment papers/darkflobi-gate/AMENDMENT_darkflobi_gate_lint_overclaims_2026_10_25.md --amendment papers/darkflobi-gate/AMENDMENT_darkflobi_gate_lock_rows_2026_10_25.md --amendment papers/darkflobi-gate/AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md
   ```

   Expected on the box once the move is made and the skill placed: rows 84, changed 0, built since 0, absent 0,
   unreadable 0, unlocked 0, bytecode 0, rc 0. Read with the PREREG alone after the move, the same box reads changed 6
   and unlocked 4 (and the skill BUILT SINCE once placed), which is the PREREG's own rows doing what they are for. The
   operator's own `sha256sum` (OD-10) stands beside it, over `W:gate/*.py`, the installed plugin and, from the
   placement, `W:skills/styxx-post/SKILL.md`.
4. **The fourth `gate_sha256` stretch is named here, once.** The move that brings `gate.py`, `gate_verify.py` and
   `canary_plan.py` begins a fourth stretch; 15.12's per-stretch reading of (iv) carries on with it, each send line read
   against the digest of the files locked when it was written: `387fc063…`, `592dc579…` and `ff34b52d…` as 15.12 names
   them, then the digest the addendum below records, from the seq the addendum records. The candidate, computed over the
   copy and the six installed plugin files as recorded at `BD:_probe/plugin_installed/`, is
   `c5084df2ef574658fe84f84feea0b70a1169eb695a882f1d4237b439d3a418e8` (19 files); the digest of record is the one the
   lead re-derives from the box after the move, and the count-rule and canary amendments point here for it. A line
   carrying one of the six intermediate digests, or any digest the addendum does not name, counts into (iv) as 15.12
   says.
5. **The tool's own bytes.** `lock_hashes.py` is a repository tool, not a lock row: the reading names it by its blob.

   ```text
   before  blob f5ffe4dc2a0d484256cd6e3e6697a14f67e51e92  sha256 (LF) 043586c1241f00311b788c62c719ca4e758dc18bc1c9b1677878f246d3143a81  25,960 bytes  (879f6be9)
   after   blob 430458ffafbaa39fba03091da226a40f36939ea6  sha256 (LF) 8bb87664e7239398320523cdbe050f4dcd19c0221e6cd27846b8a6646e591952  29,696 bytes  (8e93e3b9)
   ```

## The readings

[R, `BD:fix_pass3/lock_check_amended_copy.txt`] The command of item 3 run from this branch against the copy (with
`--w-root BD:_lockroot_skill --home BD:_lockhome` in place of the box's): rows 84, changed 0, built since 0, unlocked 0,
bytecode 0; absent 10, every one of them a path the fake root does not hold (`W:gate/salt`, `W:gate/config.json`,
`W:gate/anchor_dest.txt`, `W:ops/darkflobi-tweet.js` and the six installed plugin files), rc 1 for those alone. The
same with the PREREG alone reads rows 80, changed 6, built since 1, unlocked 4, as before. Run box-named against the
copy, `test_lock_rows_agree_with_the_box_when_a_caller_names_it` (now reading the folder's amendment files) passes; with
`STYXX_GATE_LOCK_ROWS=prereg` it reads the PREREG alone and fails there, as it must. The tests of item 1 are in
`tests/test_darkflobi_gate_prereg.py` (`test_check_reads_amendment_rows_after_the_prereg_and_exits_2_on_a_clash`,
`test_each_amendment_parses_to_the_lock_rows_its_table_states`).

## Whose step

- **The lead's, at apply**: the strict check of item 3 before the move with the PREREG alone (the box as frozen), after
  it with every amendment file (the skill ABSENT until placed, then 0), and this file's addendum, one commit.
- **The operator's**: the merge of the pull request that carries this file (the relock of record), and at each weekly
  snapshot and at close the command of item 3 beside his own `sha256sum`.

## What does not change

Every lock row's digest, the PREREG and the RULES, section 15, the gates block and every bar; the PREREG-alone reading of
`check` (the default-mode tests, which name no box, read the committed files only); `parse_lock_rows` and `ROW_RE`; what
the lock covers and never covers (`lock_hashes.py`'s docstring); the six installed plugin files and every instrument.

## What the RESULT prints

At each weekly snapshot and at close: the strict check of item 3 as run, its `AMENDMENT` lines (each file's sha256) and
its counts, beside the operator's `sha256sum`; for (iv), the four stretches with the seq at which each begins (the
fourth from the addendum) and (iv) for each; every line carrying a digest no stretch names, by seq.

## What this amendment does not say

That `check` shows the files do what the amendments say (their own tests do). That a hash the agent computed is the
operator's. That the live box was checked with the amendment files (it was not, here: that reading is the lead's at
apply). That the fourth stretch's digest is `c5084df2…` (it is the candidate; the addendum names the digest of record).
That the order of the amendment files matters (it cannot, item 1).

## The lock rows

None. This file carries no line in `lock_hashes.py`'s row form (`ROW_RE`), so `parse_lock_rows` reads 0 rows from it,
as item 2's table states; the tool's digests above are repository blobs in a text block.

## Addendum — filled at apply (blank until then)

Filled by the lead after the move, one commit, nothing above this heading edited.

- the move's UTC time, and the ten files' times and digests (`BD:apply/move_2026-10-26T134158Z.json`, written by
  `apply_move.py` pass 3, `required_present` false, so the gate stayed in shadow): the move at 2026-10-26T13:41:58Z,
  in the runbook's order, each file's time (2026-10-26), sha256 and size as written:

  ```text
   1  W:gate/tests/test_trailer_rule.py    13:41:58.822164Z  f5a19f6a801d5ce2cd31b06b50b1a4746703debb2c63ca58d5f87145b74b50ba   29086 bytes  new
   2  W:gate/tests/test_prepared_guard.py  13:41:58.832161Z  d9870b45126e1500a64ab2d32503772187ac7aa183d6967eb4bdce5a4d2739c0   16354 bytes  new
   3  W:gate/tests/test_canary_plan.py     13:41:58.841033Z  8deba26fe548ed46588caf35485559d881938ad2ee41c69122bef10acb717711   14422 bytes  new
   4  W:gate/tests/test_gate.py            13:41:58.849036Z  626d5e9eb98dd3b0ddead29151ace4c003419b4f3139be9e91ee26210c8c97e5   23815 bytes  was 589450c79379c1b3…
   5  W:gate/tests/test_gate_round2.py     13:41:58.857561Z  ddb8e988bc1e59636c1f280a7ebb6dc8b24fef5fa227bf333ac1af220574bfff   27531 bytes  was 1ccb07ad38c73985…
   6  W:gate/README.md                     13:41:58.867075Z  54e9a33f5bf4a3475197c6ee76a4564cea6d77d58dbf37e220865beb453b95bf   21520 bytes  was de55f382f45943ac…
   7  W:gate/README_exhibit.md             13:41:58.874079Z  b940f0006fe9af432cc21983902de4fe4f635363961aec41d42e719c0c5f2379   14994 bytes  was 3b65ebde31da9634…
   8  W:gate/canary_plan.py                13:41:58.883869Z  97f4f0f3d60e821f0675ed03dae35042feea137cfcc7e290238abbe44eb0075f   13068 bytes  new  enters gate_sha256
   9  W:gate/gate_verify.py                13:41:58.893377Z  049d870bcdbf4726d961bc4b58d5b3134c63f06511d398c11c0f7ed57dd3487b   28886 bytes  was f4b19c9428cfc73f…  enters gate_sha256
  10  W:gate/gate.py                       13:41:58.901481Z  4416692cfc983dbb9bc7d0f31342b3eaaf30c3beb7e27f31acdaa991c778dfbf   95482 bytes  was 25d53fd6aa5e7604…  enters gate_sha256
  ```

  Three of the ten enter `gate_sha256` (`canary_plan.py`, `gate_verify.py`, `gate.py`); the five tests and the two
  READMEs do not.
- `gate_sha256` after the move: `c5084df2ef574658fe84f84feea0b70a1169eb695a882f1d4237b439d3a418e8` over 19 files, the
  candidate, from `BD:apply/rederive_gate_sha256.py --gate C:/Users/heyzo/clawd/gate` (rc 0) and from
  `gate.Env().gate_sha256()` alike, at the move (`BD:apply/post_move_20261026T134206Z.txt`) and again at
  2026-10-26T16:01Z (`BD:apply/s7_20261026T160121Z/`). The 19 files: canary_plan.py, canary_shadow.py, exhibit_mint.py, exhibit_page.py, gate.py, gate_verify.py, ledger.py, make_exhibit_out.py, page.py, pregate_count.py, reconcile.py, selector.py, numerals.py, clawdbot.plugin.json, gate.js, index.js, package.json, patch_dist.js, README.md.
- the seq of the earliest line of each kind carrying it: `started` seq 126 (2026-10-26T14:40:40Z, the gateway's
  restart), `send` seq 127 (2026-10-26T15:41:03Z, outbound, allow), `delivered` seq 128 (2026-10-26T15:41:04Z);
  `anchor` none yet at this commit, so the RESULT reads that seq from the ledger.
- every line written after the move carrying another digest: none. At seq 132 the ledger holds three digests:
  `387fc063…` at seq 1-13 (13 lines), `ff34b52d…` at seq 14-125 (112 lines, 2 of them anchors) and `c5084df2…` at seq
  126-132 (7 lines: started 1, send 3, delivered 3). No line carries one of the six intermediates or a superseded
  candidate (`BD:apply/s7_20261026T160121Z/ledger_digest_scan.txt`).
- that `canary_plan.py` entered `gate_sha256` with this move: yes. The move record gives it no earlier bytes (`before`
  null) and `enters_gate_sha256` true; the stretch before the move, `ff34b52d…`, is over 18 files without it, and the 19
  above include it.
- the in-place readings, right after the move (`BD:apply/s6_20261026T134223Z/`): the gate's suite `279 passed, 12
  subtests passed`, rc 0; the plugin harness 79 of 79, rc 0; no `__pycache__` or `.pytest_cache` under `W:gate`; item
  3's strict check with every amendment file, rows 84, changed 0, built since 0, absent 1, unreadable 0, unlocked 0,
  bytecode 0, rc 1, the one absent row `W:skills/styxx-post/SKILL.md` (not yet placed); the same with the PREREG alone,
  rows 80, changed 6, unlocked 4, rc 1, as item 3 expects; `gate_verify.py --rederive` over 125 lines, SAME 60, DRIFT 1
  (seq 46), TAMPER 0, rc 1; the posting skill's dry harness 20 passed, 0 failed. Read again with the skill placed, at
  2026-10-26T16:01Z (`BD:apply/s7_20261026T160121Z/`): item 3's strict check with every amendment file, rows 84 and
  every count 0, rc 0, the `plugins-subtree:clawdbot.json` row skipped by the tool as its docstring says and re-read by
  `reconcile.plugins_subtree_digest` as `519905d2…`, the lock's; `gate_verify.py --rederive` over 132 lines, SAME 63,
  DRIFT 1 (seq 46), TAMPER 0, rc 1.
- the commit of this addendum, and its UTC time: the commit that adds these lines, authored and committed at
  2026-10-26T16:04:31Z. A commit cannot carry its own hash; `git log -1 --format=%H --` on this file names it.
