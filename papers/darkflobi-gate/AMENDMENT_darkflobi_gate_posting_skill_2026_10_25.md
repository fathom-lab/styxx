# AMENDMENT — the posting skill: what the agent is told before a public post with numbers, and a prepared record the gate sets aside when it is a dry run's, not live, or its receipts moved

Fathom Lab · 2026-10-25 · Amends `PREREG_darkflobi_gate_2026_10_23.md`, in this directory, merged as #229 with its
section 15 closed at 15.13 and not edited. Section 11 provides for this file: its row for the posting skill reads
`not built at the freeze` and "locked when the piece lands"; section 12's closing paragraph makes any change after the
merge "a new AMENDMENT file under this folder, lock-hashed in its turn, with deliveries under the old rule counted
under the old rule". Written by the lab's build agent for piece (4) of the before-D0 work, for the lead, on the lead's
pass-2 plan (`BD:PLAN.md` section 7, written 2026-10-25T23:42Z to 2026-10-26T00:07Z), before `W:gate/REQUIRED` exists
and before any prepared record has reached a ledger line (item 1 of "What was observed"). The operator has signed
nothing new for it; what is his, and what is the lead's, is listed under "Whose step". No bar, no cell, no row of the
gates block, no reason code and no row of RULES section 7's matrix moves.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `BD:` = `WT:_darkflobi_gate_2026_10_22/before_d0/`;
`PREREG` and `RULES` = this folder's files at `879f6be9`, where this branch starts, line numbers their own; `seq N` = a
line of `W:gate/ledger.jsonl`; `[R]` a file or a run shows it; `[J]` judgement. A digest written `abcd1234…` is a
sha256 whose full form stands in the PREREG's lock table, in this file's lock rows, or in the receipt it names. The
live gate `W:gate` was read and never written: every gate change below was built and tested on the copy
`BD:gate_copy/` (made 2026-10-25 13:36 local with the brief's exclusions; no salt, no `config.json`, no `REQUIRED`, no
ledger), and the lead moves it into place at apply time. Nothing of `W:gate/salt`, `W:gate/config.json`, `~/.clawdbot`
beyond a listing and clawdbot's installed `dist/` (read, for how a skill is loaded), or `W:secrets/` was read.

## What the PREREG says

- **The capsule clause** (§2, l.154-156): "Every public post all of whose numerals the gate found bound has, before the
  message exists, a capsule at a stable URL whose `document_sha256` is the sha256 of the text above the footer, and the
  message carries that URL and that digest."
- **The skill's row** (§11, l.621): `the posting skill`, `W:skills/styxx-post/SKILL.md`, `not built at the freeze`,
  "locked when the piece lands". 15.2 (l.766), 15.6 (l.991) and 15.10 (l.1182) read it NOT BUILT.
- **15.11** (l.1198-1205): a commit on `main` after the pin refuses as `resolver_failed:commit_not_pin_or_ancestor`;
  "The posting skill (`W:skills/styxx-post/SKILL.md`, not built, §11) must tell the agent to cite the pin or an ancestor
  of it; until it exists nothing tells the agent so, and a refusal's notice names the codes and the `seq` only".
- **The handshake's reader** (`W:gate/README_exhibit.md`, the PREREG's row l.558, `3b65ebde…`): the mint writes a record
  into the gate's `prepared/` only for a live mint, a dry run's into `prepared_dryrun/`; and, as an open item, "For
  gate-core: the gate should still ignore a record with `dry_run` set or `url_live` false (round 2 review MINOR-I), and
  it does not compare the receipts file's bytes with `receipts_sha256`."

## What is built, and what was not

- In `W:gate` (live, locked, unchanged by this file until apply): `exhibit_mint.py` (`e2dcf386…`) writes the prepared
  record under the sha256 of `MESSAGE.txt` with an alias under the sha256 of the message less its trailing newlines,
  sets `dry_run` (null on a live mint), `url_live` (true only when the footer's own URL answered with the capsule's
  bytes) and `receipts_sha256` (the sha256 of `POST.receipts.json`'s bytes, l.997 and l.1434), and hashes
  `skills/styxx-post/SKILL.md` into each capsule's identity block once the file exists (`SKILL_KEY` l.168, l.565).
  `gate.py` (`25d53fd6…`) reads the record (`_prepared_record` l.872-880), splits OD-20's footer off
  (`_split_footer`, `footer_problem`, `prepared_split` l.883-952) and certifies the document above it with the receipts
  the record names (l.1087-1100); before this amendment it read neither `dry_run`, `url_live` nor `receipts_sha256`.
- The `exhibit` branch: `WT:exhibit` at `893d1a5e` equal to `origin/exhibit` ("the empty shelf", OD-7), no `posts/`
  [R, `git rev-parse`]. `W:gate/prepared/` is empty and `W:gate/prepared_dryrun/` holds the mint's two dry-run records
  [R, listings, 2026-10-26T00:3xZ]; no send line on the ledger carries a prepared record (`BD:PLAN.md` section 1.1).
- Not built before this file: the skill, its row, the guard, its tests.

## What was observed

1. **A dry run's record placed in `prepared/` was honoured.** Pass 1's dry harness, re-run by the apply review
   (`BD:dry_harness_review_run.txt`, 2026-10-25T19:54:37Z, against the copy's pass-1 `gate.py` `8d45166a…`), step 12:
   a record with `dry_run` set and `url_live` false under the message's sha256 read `allow`, `footer_split true`,
   4 of 4 bound (`BD:REVIEW_apply.md` finding F3). Nothing else in `gate.py` differs on that path from the live bytes.
2. **The guard's tests fail against the live bytes and pass against the copy.** `tests/test_prepared_guard.py` (13
   tests, below) run in a private copy holding the live `gate.py` (`25d53fd6…`) and `gate_verify.py` (`f4b19c94…`):
   10 failed, 3 passed, rc 1; the three that pass are the honoured live-shaped record, the missing receipts file and the request naming its own receipts, which the guard does not change. Against the guard: 13 passed (`BD:skill_pytest_devcopy_live_bytes.txt` holds both runs).
3. **The mint's own record passes the guard.** The dry harness against the copy (`BD:skill_dry_harness_copy_run.txt`,
   sha256 `065e3c2cb1b50fe0…`, 2026-10-26T00:37:40Z; its transcript `posting_skill/harness/_out/last_run.txt`
   `152e0d7c509c5892…`): 20 PASS, 0 FAIL, 0 OBSERVED. Step 4: a live-shaped mint through the CLI against a loopback
   origin writes a record with `url_live` exactly true, `dry_run` null, and `receipts_sha256` equal to the committed
   receipts file's sha256. Steps 5 to 9: that record is honoured at E2, E1 and E3, by its alias, to the operator's DM,
   in shadow and through the plugin itself (gate.js `gateOutbound`, clawdbot's hook runner, the plugin's handler):
   `allow`, `clean`, 4 of 4 bound, no trailer, `sent_sha256` equal to the handed bytes, the capsule's URL and digest on
   the line.
4. **A set-aside record refuses through the footer.** Steps 12, 12b, 12c, 12d of the same run: a dry run's record
   (`record_dry_run`), a live mint's record whose URL never answered (`record_url_not_live`, at E3), and the minted
   post's own receipts file re-indented after `MINTED` with every value kept (`record_receipts_moved`) each read
   `refuse`, reasons `audit_fail:X1`, 6 numerals 0 bound, `cancel` under `REQUIRED`; restored, the same message is
   honoured again; in shadow the dry run's message is sent unmarked with `would_refuse`. The cause is step 10's: with no
   split, the footer's install line `styxx 7.50.0 (pip install styxx==7.50.0; pin 3437340b)` is read, the audit leg
   takes it as a version claim against the pin, whose `styxx/_version.py` reads 7.49.0, and fails it.
5. **How the gateway loads a skill** [R, clawdbot's installed `dist/`, read only]: `loadSkillEntries`
   (`agents/skills/workspace.js` l.64-118) merges extra dirs < bundled < managed (`~/.clawdbot/skills`, absent on this
   box) < workspace (`<workspace>/skills/<name>/SKILL.md`; `W:skills/` holds the agent's other skills); each entry's
   YAML frontmatter gives `name` and `description`, listed in the system prompt under "Skills (mandatory)" with the
   instruction to read the one applicable `SKILL.md` before replying (`agents/system-prompt.js` l.10-14). A session's
   skills snapshot is built on its opening turn and rebuilt when the skills watcher (`agents/skills/refresh.js`,
   chokidar over `<workspace>/skills`, on unless `skills.load.watch` is false; `clawdbot.json` has no `skills` key)
   bumps its version (`auto-reply/reply/session-updates.js` `ensureSkillSnapshot`). `/context detail` lists the skills.

## The amendment

1. **The skill.** `W:skills/styxx-post/SKILL.md` is the text committed beside this file as
   `papers/darkflobi-gate/styxx-post_SKILL.md`, the same bytes (LF; the lock row's digest is the sha256 of both), so a
   reader holds what the agent was told. It tells the agent: when it applies (a public post naming the lab's work and
   holding a digit; not the operator's DM, not X, not a file or an image); the window's authoring shape (certify through
   the mint; the sworn sidecar named, not carried); the recipe, from the exhibit worktree `WT:exhibit` and never from
   `W:` or a working checkout: receipts in form (a) `<path>#<pointer | Lx | Lx-Ly>@<40-hex>` on receipt-bearing paths
   only (`papers/`, `runs/`, `web/gate/README.md`), naming the pin `3437340b…` or an ancestor and never `origin/main`
   (15.11), the committed value copied as the object writes it (15.11: no rounding, no percent equivalence), exactly
   one `self:count` leaf; `identity`, `check-instrument`, `mint --push`; send only after `MINTED` (or `prepare`), the
   bytes of `MESSAGE.txt` exactly, once, through the message tool or the telegram action, never `clawdbot send` from a
   shell and never `exec` to the Bot API; never a dry run for a post that will be sent, nothing in the post folder
   changed between the mint and the send, nothing written into `W:gate`; the mint's seventeen refusals and how each is
   avoided; the claim shapes of RULES §3.1, condensed; what the gate does with the minted message (item 2 below
   included); the count rule's words as the count rule AMENDMENT of 2026-10-25 writes them (a mark quoted whole as the
   gate appended it to the line it names is not counted, a mark composed is; a minted post carries no mark and no seq,
   because the mint's certify counts both); versions (a hyphenated model identifier is not counted; `opus 4.8` and a
   quoted `"3.1"` are; any three-part version wakes the audit leg, and a styxx version the pin does not carry fails);
   how the agent describes the gate (section 7: the lab's sentences, the withdrawn wordings by seq and the five rules
   of `AMENDMENT_darkflobi_gate_lint_overclaims_2026_10_25.md` item 1, the same rows and rules); what a refusal
   notice's codes mean; and the window's rules (no clawdbot update, `REQUIRED` the operator's, the pin by AMENDMENT
   only, one post one mint one send, `main` never written from `exhibit`).

2. **The guard.** A prepared record found under the sha256 of the handed text is honoured only when it is a live
   mint's whose receipts still hold the bytes the mint certified (`gate.prepared_problem`):
   - `dry_run` is not null: `record_dry_run`;
   - `url_live` is not the JSON value `true` (absent, false, null, a string or a number included): `record_url_not_live`;
   - the file the record names as `receipts` exists and its sha256 is not the record's `receipts_sha256` (a record
     with no `receipts_sha256` included): `record_receipts_moved`.
   The checks run in that order, and the earliest problem met is the one named. A record with a problem is set aside: no
   receipts are taken from it, no footer is split off, the whole handed text is the checked text (so `checked_sha256`
   is `draft_sha256`, no certify leg runs from the record, `receipts_offered_sha256` stays null), and the line's
   `prepared` object keeps its shape: the record's `capsule_url`, `capsule_sha256`, `capsule_path` and
   `document_sha256`, `footer_split` false, `footer_refused` the problem. Unchanged: a request naming its own
   `receipts_file` ignores any record, as before; a record naming a receipts file that is missing is honoured and the
   L2b read names `receipts_missing`, as before, and one that cannot be read `receipts_unreadable`. When the record is
   honoured, the receipts bytes the guard hashed are the bytes L2b parses and keeps under `sources/` (one read, so the
   bytes certified are the bytes checked).

3. **What a set-aside record means for the post, said plainly.** The footer is then text: its digits are counted and its
   install line is a version claim the audit leg fails against the pin (observation 4), so a post behind a dry run's
   record, a record whose URL was not live, or a record whose receipts moved is refused under `REQUIRED` (`audit_fail`,
   a truth finding, cancelled at a public exit) and delivered with `would_refuse` in shadow; on the operator's DM it is
   marked. The guard adds no reason code: the refusal's codes are the legs' own, and `prepared.footer_refused` on the
   line says why the record did not count.

4. **The replay.** `gate_verify.py` `rederive_line` hands a line's own `prepared.footer_refused` back to the replay as
   `prepared_problem` when it is one of the three values of item 2, and the replay sets the record aside for that
   reason without opening any file of the record; `_diff_fields` already compares `footer_refused` (`gate_verify.py`
   l.134). A line written before the move carries none of the three values and re-derives as before.

5. **RULES §13's `prepared` fields** gain three values of `footer_refused` (`record_dry_run`, `record_url_not_live`,
   `record_receipts_moved`) beside the footer problems it already carries (`document_not_found`, `capsule_url_form`,
   `footer_line_count` and the rest). No key is added to a line.

6. **Three hand-made records in the gate's own tests** become live-shaped, their assertions unchanged:
   `tests/test_gate.py` `test_prepared_record_splits_the_footer_and_binds_the_document` gains `"dry_run": None`,
   `"url_live": True` and the receipts file's `receipts_sha256`; `tests/test_gate_round2.py` `TestF8F9PreparedFooter.record`
   and the prepared line of `test_honest_lines_of_every_shape_replay_same` gain `"url_live": True`. Without that the
   guard sets their records aside (they predate the fields), and they test the footer, not the guard.

7. **One move.** `gate.py` and `gate_verify.py` are the files the count rule AMENDMENT of 2026-10-25 moves too; their
   rows below are the files with both changes, one digest each, and the count rule's file names the same digests. They
   enter `gate_sha256`; the stretch that begins with the move is named once, in the lock-rows AMENDMENT's addendum.

8. **Placement** [the lead's]: after this file's rows are on the branch (until then the strict check reads the skill
   BUILT SINCE), the lead copies `styxx-post_SKILL.md` byte for byte to `W:skills/styxx-post/SKILL.md`, compares its
   sha256 with the row, and confirms with `/context detail` in the operator's DM that `styxx-post` is listed (or `/new`,
   which builds the snapshot on the new session's opening turn).

## Whose step

- **The lead's**: the move (with the count rule's half, one process), the suite and the dry harness in place, the
  strict check naming every amendment file, the skill's placement and its listing, the PR with its CHANGELOG entry.
- **The operator's**: the merge (the relock of record); the `--push` credential on `exhibit` and OD-7's protection;
  the phone test (`BD:posting_skill/PHONE_TEST_CHECKLIST.md`: one live mint, the two badges on the phone, the handshake
  read on the ledger, then one post the agent makes by the skill); the no-update line in `W:TOOLS.md`, or not.

## The tests

`W:gate/tests/test_prepared_guard.py` (13 tests; synthetic records in the mint's shape over `test_resolver`'s fixture
repository, no styxx checkout; each in a gate home of its own, through the real legs of the pinned snapshot): (a) a dry
run's record set aside, the whole text read, no certify leg, no receipts offered, not clean; (b) `url_live` false, and
absent; (c) the receipts re-written with every value kept, and a record with no `receipts_sha256`; (d) a live-shaped
record honoured as before (5 of 5, clean, no trailer, `receipts_offered_sha256` equal to the record's
`receipts_sha256`), and a missing receipts file still `receipts_missing`; shadow delivers a set-aside post with the
reading on the line; a request naming its own receipts ignores the record; `prepared_problem` read alone (a set but
empty `dry_run`, `url_live` as false, null, `"true"` and 1, an upper-case digest, an absent file); (e)
`gate_verify.py --rederive` reads SAME on every line of (a)-(d); (f) mutation: with `prepared_problem` switched off,
every set-aside case fails and the honoured cases hold; with the verifier's hand-back switched off, a set-aside line
re-derives as DRIFT on `prepared.footer_refused`. The suite in the copy, the plugin harness and the dry harness are the
other readings (`BD:BUILD_skill.txt` names each receipt). On this branch,
`tests/test_darkflobi_gate_posting_skill.py`: the committed skill text has the frontmatter, the pin's 40 hex, LF lines,
no byte-order mark, 0 hits under this checkout's `scripts/surface_lint.py --files` (whose charter rules spell the words
the lab never prints), and its sha256 is this file's row for `W:skills/styxx-post/SKILL.md`.

## The lock rows

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/gate.py` | `4416692cfc983dbb9bc7d0f31342b3eaaf30c3beb7e27f31acdaa991c778dfbf` | 95482 bytes; the count rule (its AMENDMENT of 2026-10-25) and the prepared-record guard (item 2), one move; before it `25d53fd6…` (15.10); `gate_sha256` from this row on is the lock-rows AMENDMENT's addendum, before it `ff34b52d…` (15.10) |
| gate file | `W:gate/gate_verify.py` | `049d870bcdbf4726d961bc4b58d5b3134c63f06511d398c11c0f7ed57dd3487b` | 28886 bytes; the replay's earlier send lines (the count rule) and the line's `record_*` problem handed back (item 4); before it `f4b19c94…` (§11) |
| gate file | `W:gate/README_exhibit.md` | `b940f0006fe9af432cc21983902de4fe4f635363961aec41d42e719c0c5f2379` | 14994 bytes; the open item becomes the rule (item 2); before it `3b65ebde…` (§11) |
| gate file | `W:gate/tests/test_prepared_guard.py` | `d9870b45126e1500a64ab2d32503772187ac7aa183d6967eb4bdce5a4d2739c0` | 16354 bytes; new with this amendment, no earlier row; 13 tests (the tests above) |
| gate file | `W:gate/tests/test_gate.py` | `626d5e9eb98dd3b0ddead29151ace4c003419b4f3139be9e91ee26210c8c97e5` | 23815 bytes; one hand-made record made live-shaped (item 6); before it `589450c7…` (15.6) |
| gate file | `W:gate/tests/test_gate_round2.py` | `ddb8e988bc1e59636c1f280a7ebb6dc8b24fef5fa227bf333ac1af220574bfff` | 27531 bytes; two hand-made records made live-shaped (item 6); before it `1ccb07ad…` (§11) |
| the posting skill | `W:skills/styxx-post/SKILL.md` | `5339e659a17516a01d965730a85c1ce9daf5fdd9a4d87977dc65e2d2825ddf32` | 25144 bytes; the text committed beside this file as `styxx-post_SKILL.md`, the same bytes (item 1); before it not built at the freeze (§11) |

## What does not change

- The pin, the snapshot and every instrument; `exhibit_mint.py` (`e2dcf386…`: `SKILL_KEY` already hashes the skill once
  placed) and `tests/test_exhibit_mint.py` (`TestGateReadsTheMintsRecord` mints live-shaped and stays green);
  `ledger.py`, `reconcile.py`, `selector.py`, `canary_shadow.py`, `gate.js`, the plugin and its installed copies, salt
  and config; the footer's fixed strings and `footer_problem`; RULES §6's codes and §7's matrix; the capsule clause,
  the cells, the population rule and every bar.
- The census: a capsule post's footer is stripped at a text's end as before (`reconcile.py` `END_FORMS`).
- Lines written before the move are read under the rule they were written under (§12).

## What the RESULT prints

- The capsule count c (§5) with, per `gate_sha256` stretch, the public send lines that carried a prepared record,
  split by `prepared.footer_refused`: null (honoured, footer split), each footer problem, and each of the three
  `record_*` values; and the `audit_fail` refusals whose line carries a `record_*` value, beside `n_refusals`.
- For each minted capsule, the skill's digest in its identity block (`minted_by.build.skills/styxx-post/SKILL.md`)
  beside this file's row, so a reader sees which text the agent had when it posted; a capsule minted before the
  placement prints null there.
- Before the move, "not recorded" for the guard's values.

## What this amendment does not say

- That the agent will use the skill, or follow it on every post: only the window's lines show that, and the phone
  test's section 5 is one post.
- That a capsule says a post is true: it shows which numerals bind to committed bytes at the pin and how the
  certifier read them.
- That the guard tells the mint's record from one written by hand with the same fields and a matching digest: it
  reads the record's own fields and hashes the receipts file it names; it does not fetch the URL again. The mint is the
  one writer of `prepared/` by rule, not by a lock the gate holds, and the PREREG's open routes stand.
- That `--push` works from the agent's session, that GitHub Pages serves the capsule, or that Telegram delivers the
  footer's bytes unchanged: the phone test reads those, and nothing here does.
- That X, media or the sworn route are covered: the skill names them and carries none.
