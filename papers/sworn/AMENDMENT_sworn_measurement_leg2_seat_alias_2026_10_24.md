# AMENDMENT 6 to the leg-2 PREREG — the Claude family runs on `claude-opus-5-5`, because the PREREG's alias `claude-fable-5` does not answer on the operator's plan

Fathom Lab · 2026-10-24 · Amends `PREREG_sworn_measurement_2026_10_23.md`, in this directory, signed by the
operator on 2026-10-23 (signing commit `01e9d2b7`, merged in #225 as `2e218601`) and not edited. Its section 3
names five amendments; this is the sixth, and it is called Amendment 6. Written by the lead, on the operator's
decision of 2026-10-24 quoted below. It is committed with its three receipts, under
`papers/sworn/measurement/receipts/`, and a CHANGELOG entry, before any seat has run: the PREREG's own line 207 provides for it, and its
section 8 (l.765-766) makes any amendment after a seat has read a real item a new preregistration under a new
name. No bar moves.

## What the PREREG says

Section 2, lines 203-207, byte for byte (the five lines with their line feeds hash to sha256
`b695fba51367b41b06e1fbb199ca985bda2de14d95708df7019144ff9a8ed551`):

```text
- **The Claude model alias.** `claude-fable-5` (`seat_claude.py` l.37), through `claude -p` with
  `--setting-sources ""` and `--tools ""` (l.68-69), one item per call, the contamination probe
  before any item (l.6-7). Whether the alias resolves on the day is unreceipted: a one-item transport
  check over a synthetic packet (`--transport-check --max-items 1`, l.11-15) runs before any seat.
  If it does not resolve, no seat runs on another alias until an amendment naming it is committed.
```

Section 4 (l.506-512) puts that transport check after the signing PR and before tranche A's six Claude seat
files, which run in one usage window with no fan-out running: 3 seats × (59 + 166) = 675 item calls with a 600 s
timeout each (`common.py` l.61), one contamination probe per seat file, and 9 trivial-twin calls for G-G1
(`twin_trivial.py`, Claude family, one per document). `seat_claude.py` and `twin_trivial.py` are in the lock
(section 5; blobs `105f4efe` and `ef5d27ca`) and are not edited.

## What was measured

1. **2026-10-23: the transport check on `claude-fable-5` did not resolve.** The lead ran
   `seat_claude.py --packet L --seat 1 --transport-check --max-items 1` at 16:39:45Z (the ledger row's `ts`) in
   the run worktree, a checkout of `2e218601`, over `transport_check_2026_10_23/`, a copy of `dryrun/`'s inputs
   without `dry_run_result.json`, `dry_run_summary.json`, `seat_outputs/` and `twins/`. The seat file reads
   `"verdict": "VOID-CONTAM"`, with probe answer `""`, `"ok": false`, `"raw_sha256": null` and the error "the
   transport returned its error envelope" (sha256 `281c5a2607ad3e225099d88bcda33c3dca86532ad8b60a5958b5083d31ba9721`;
   it stays in the run worktree, uncommitted). `seat_claude.py` writes VOID-CONTAM for any probe whose answer is
   not `NO` (l.218, l.223-226), an empty answer from a failed call included, so this verdict records a call that
   failed, not a context that mentioned anything. The lead's direct call with the seat's flags that day returned
   "Failed to authenticate: OAuth session expired and could not be refreshed" (the lead's record; not re-made
   here).
2. **2026-10-24: after the operator renewed the CLI login, `claude-fable-5` answers with a usage-credit error
   and `claude-opus-5-5` answers.** The lead's direct calls that day returned an error envelope for
   `claude-fable-5` and an answer for `claude-opus-5-5` and for `claude-sonnet-5-5`. Re-made for this amendment,
   one call each, with `seat_claude.py`'s argv (l.68-69) and its contamination probe's system prompt and question
   (l.40-42): the call a seat makes before any item (l.217), with its 120 s timeout. claude CLI `2.1.310 (Claude
   Code)`. Each receipt keeps the exact argv (`argv[0]` written as `claude`, resolved by `shutil.which` as
   l.38 does), the start and end times, the exit code, and the raw stdout and stderr as UTF-8 text, each with
   the sha256 of its raw bytes:

   | receipt | started (UTC) | exit | stdout sha256 | what the envelope says |
   |---|---|---|---|---|
   | `receipts/transport_alias_fable_2026_10_24.json` | 2026-10-24T14:13:20.605Z | 1 | `86eef9c0af24299462fc80a16b2b07c968bb7c3bdcd53cd6e2647ac417094e89` | `"is_error": true`, `"api_error_status": 429`, `"api_error": "model_requires_usage_credits"`, `"modelUsage": {}`, result "You're out of usage credits. Switch to another model, or manage usage credits at https://claude.ai/settings/usage?from=cc_cli_limit_message, to continue." |
   | `receipts/transport_alias_opus_2026_10_24.json` | 2026-10-24T14:13:30.501Z | 0 | `4afd17595d3461e194462d1922b6c53a22cd20a4ebe6952b312cb79f76140637` | `"is_error": false`, result `NO`, `modelUsage` naming `claude-opus-5-5` and nothing else |

   One more `claude-fable-5` call was made shortly before the receipted one: the receipt script's own scan for
   token-shaped strings refused the receipt it built, on its schema string, before the bytes were kept. The
   receipted call replaced it. The `claude-sonnet-5-5` call was not re-made, and nothing here rests on it. Both
   receipts were scanned for key-, bearer-, JWT- and e-mail-shaped strings before they were written; the CLI's
   JSON carries a session id and a `uuid` and no secret.
3. **2026-10-24: the PREREG's transport check on the amended alias resolved.** In the run worktree, at
   14:14:00Z to 14:14:17Z, over `transport_check_2026_10_24_opus/`, a fresh copy made by the same rule (its inputs
   equal `transport_check_2026_10_23/`'s byte for byte: `diff -rq` lists only that folder's `seat_outputs/`;
   `packet_L.json` sha256 `8509e91c…`, the dry run's):

   ```text
   python seat_claude.py --packet L --seat 1 --transport-check --max-items 1 --dir transport_check_2026_10_24_opus --model claude-opus-5-5
   seat L-1 (claude): verdict TRANSPORT-CHECK-SYNTHETIC, items 1, unparsed 0, errors 0
   contamination probe answered 'NO'
   ```

   The seat file is committed byte for byte as `receipts/transport_check_opus_2026_10_24.seat.json` (sha256
   `fccff86c4939cd605002ebaa0119aefab1cbf6d38cd5d52a095a3c03f07b2407`): `substrate.model` `claude-opus-5-5`,
   `synthetic_packet` true, `prereg` null, `dry_run` false, one synthetic item parsed, the probe `"ok": true`. Its
   ledger (sha256 `d7c3691edf6d3520143897670bed347db14e81e377bacc3628117748348b3582`) stays in the run worktree.
   The item it read is invented text from `synthetic.py`; no seat has read a real item.

## The operator's decision

The lead asked in chat on 2026-10-24: "Leg 2's Claude seats are pinned to claude-fable-5, and your plan is out of
Fable usage (Opus 5.5 and Sonnet 5.5 work on the same login) ... What should the Claude seats use?" The operator
answered: **"Amend to Opus 5.5"**.

## The amendment

1. **The Claude family's alias is `claude-opus-5-5`** for every Claude-family call this PREREG runs: the three
   seats of each panel, each seat file's contamination probe, and the trivial twins, in tranche A's fold and in
   tranche B's.
2. **The seats pass the alias on the command line.** `seat_claude.py` is not edited. Its own `--model` flag
   (l.261) reaches `run` (l.263-264), the probe (l.217) and every item call (l.235):

   ```text
   python papers/sworn/measurement/seat_claude.py --packet L --seat 1 --model claude-opus-5-5
   ```

   (seats 1-3, packets L and R: six seat files; tranche B's add `--dir`). Every other part of the call is
   unchanged: the argv's flags (l.68-71), the packets and their instruction blocks, the probe's question and
   system prompt, the 600 s item timeout and the probe's 120 s, the two retries after a failed call (l.39, l.73),
   one item per call with no session reuse, and section 4's order. Every seat file records
   `"substrate": {"model": "claude-opus-5-5", ...}` (l.207), and `score.py` carries that substrate into
   `measurement_result.json` under `families` (l.518-521, l.543).
3. **The trivial twins reach the alias through `seat_claude.cli`, with no file edited.** `twin_trivial.py` has
   no alias flag (l.103-111): it calls `SC.cli(prompt, system, None)` (l.77) with no model, and `cli`'s default
   was bound to `MODEL = "claude-fable-5"` when `seat_claude.py` was imported (l.37, l.62-63). Run as committed,
   every twin call would ask `claude-fable-5`, which the receipt above shows answering 429; no twin would be
   written, G-G1 would have no pair, and its metric would read -1, a miss (the gates block's `G_G1`
   `metric_means`). So each twin runs through `twin_trivial.py`'s own `main`, in a process where
   `seat_claude.cli` has the amended alias bound, from the repository root:

   ```text
   python -c "import functools, sys; sys.path.insert(0, 'papers/sworn/measurement'); import seat_claude as SC; SC.cli = functools.partial(SC.cli, model='claude-opus-5-5'); import twin_trivial as T; sys.exit(T.main(sys.argv[1:]))" --doc D01
   ```

   (D01 to D09; tranche B's add `--dir`). `twin_trivial.py` imports `seat_claude` by name (l.29), so its call
   site reaches the bound function, with every other argument as committed. Checked on the drafting box by a
   local script, not committed, with `subprocess.run` replaced by a recorder, so no model was called: as committed, the twin's call site passed
   `--model claude-fable-5`; under the command's statements it passed `--model claude-opus-5-5`; and a call with
   an explicit model, the seats' route, passed that model. A twin's status file records no alias (l.69-79): the
   alias of every twin is this amendment's, and the RESULT names it from here.
4. **The transport check on the day.** Section 2's check is item 3 of the measurements above, on the amended
   alias. If the seats start on a later day than 2026-10-24, the same one-item check runs again that day, over a
   fresh synthetic copy made by the same rule, before any seat, and the RESULT prints its verdict. If
   `claude-opus-5-5` does not resolve, line 207 applies to it as it did to `claude-fable-5`: no seat runs on
   another alias until an amendment naming it is committed.
5. **`named_in_design: true` names the family and its transport.** `seat_claude.py` writes it for every alias
   (l.208). DESIGN v2 l.21 names the Claude family as Claude through the `claude -p` clean-config subscription
   transport, pointing at `run_b23_fable.py`, whose alias is `claude-fable-5` (its l.44); the alias itself was
   named by the PREREG (l.203), not by the DESIGN's words. Under this amendment the flag is read as naming the
   family and the transport, which are unchanged, and the alias is printed beside it.

## What does not change

- No bar, no gate, no row of the gates block, no packet, no key digest, no lock hash and no rule of the PREREG's
  sections 2 to 8. The PREREG and every file its section 5 names are not edited.
- The local family and Amendment 1: `Qwen/Qwen2.5-3B-Instruct`, bf16, `--device cuda`, by day, every seat file
  `named_in_design: false`. Amendments 2 to 5 stand as written.
- The cost, in section 4's counts: 675 item calls, 6 contamination probes and 9 trivial-twin calls, each with at
  most two retries after a failed call, on the operator's plan, in one usage window with no fan-out running.

## Written before any seat ran

At `origin/main` `ca347730` (2026-10-24), the tracked files under `papers/sworn/measurement/` whose path holds
`seat_outputs` are the dry run's 20, all under `dryrun/seat_outputs/` (7 under `claude/`, 6 under
`claude/trivial/`, 7 under `local/`), and no `tranche_B/` exists. In the run worktree no
`papers/sworn/measurement/seat_outputs/` exists; its only seat files are the two synthetic transport checks
above. A search of the drafting box's working folder, which holds the live checkout and its worktrees, found
`seat_outputs` directories only under copies of `dryrun/` and in those two folders. No tranche-A or tranche-B seat file exists, and no seat has
read a real item.

## What the RESULT prints, added to section 7

- `claude-opus-5-5 (Amendment 6)` beside every Claude-family number: its G-D counts on each panel, its family
  labels, every kappa it enters, and the G-G1 pairs.
- That the Claude seats ran on `claude-opus-5-5`, a different model from the one the dry run's six Claude seat
  files name (`substrate.model` `claude-fable-5`, `dry_run` true, verdict `DRY-RUN`), whose answers were canned:
  no model was called by the dry run (`RESULT_measurement_machinery_dry_run_2026_09_05.md` l.5-6). The dry run's
  Claude seat counts are sizing inputs only, as Amendment 5 already says, and nothing compares them with the
  seats.
- The two transport facts that led here, with this file's name: 2026-10-23, VOID-CONTAM on a failed call;
  2026-10-24, `claude-fable-5` answering 429 `model_requires_usage_credits`.
- Item 5's reading beside `named_in_design: true`, and item 3's route for the twins, with the fact that a twin's
  status file records no alias.

## What this amendment does not say

- That `claude-opus-5-5` judges as `claude-fable-5` would have. The Claude family qualifies on the decoys (G-D)
  on this alias or not at all, and nothing in the dry run speaks to either model.
- That the transport is clean beyond the probe. The receipted `claude-opus-5-5` call reports 39,201 input tokens
  written to the prompt cache for a 40-character system prompt and a 110-character question, with no tools and no
  setting sources. What fills them is not examined here; the probe's `NO` is the only contamination check the
  PREREG runs, and it is unchanged.
- That `claude-opus-5-5` will answer on the day the seats run, or that the 690 calls fit inside one usage window
  of it. Item 4 covers the day; the window is not measured.
- That `claude-fable-5` is unavailable in general. The receipt shows this login's plan on 2026-10-24.
