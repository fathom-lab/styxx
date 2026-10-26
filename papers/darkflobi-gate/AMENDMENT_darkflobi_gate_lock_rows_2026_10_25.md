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

- the move's UTC time, and the ten files' times and digests (`BD:apply/move_<UTC>.json`): _blank_
- `gate_sha256` after the move, from `BD:apply/rederive_gate_sha256.py --gate C:/Users/heyzo/clawd/gate` and from
  `gate.Env().gate_sha256()`, and its file count: _blank_ (the candidate `c5084df2…`, 19 files)
- the seq of the earliest line of each kind carrying it (`send`, `delivered`, `anchor`, `started`): _blank_
- every line written after the move carrying another digest (one of the six intermediates or a superseded candidate of
  `BD:APPLY_RUNBOOK.md` section 0), by seq: _none expected_
- that `canary_plan.py` entered `gate_sha256` with this move: _blank_
- the in-place readings: the gate's suite (`279 passed, 12 subtests` expected), the plugin harness (79 of 79), item 3's
  strict check with every amendment file (rows 84; the skill ABSENT until placed, then every count 0, rc 0), and
  `gate_verify.py --rederive` (one DRIFT, seq 46, expected): _blank_
- the commit of this addendum, and its UTC time: _blank_
