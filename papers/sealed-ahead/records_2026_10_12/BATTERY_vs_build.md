# BATTERY vs build: which predicted catches the code implements

Fathom Lab red team, 2026-10-12. Not committed; this file is the comparison the battery promised.

**What was compared.**

- **The battery.** `BATTERY.md` at commit `6faeff16` in this directory (committed 2026-10-12 09:18:23 −0400; git blob sha256 `022cbe90392bd797d65eecb4cda02c1d60b7e7d7f987b90290468e76090398fb`). It was written and committed before any of the builder's files were opened.
- **The build.** Branch `research/sealed-ahead-precision` in `C:/Users/heyzo/clawd/wt/sealed-ahead`, HEAD `3c63ea40`: 7 commits on `f469f25b`, from 09:20:33 to 09:40:36 −0400. It adds `papers/sealed-ahead/{common,population,pool,accuse,draw,stranger}.py`, `identity_table.json`, `README.md`, one INDEX row and `tests/test_sealed_ahead.py`. The working tree was clean.
- **Tests.** `py -3.12 -B -m pytest tests/test_sealed_ahead.py -q -p no:cacheprovider` printed 70 passed in 1.66 s.
- **Probes written for this comparison** (scratchpad only: synthetic events, the builder's modules imported by path, no network beyond one pair of GH Archive HEAD requests):
  - `probe_build.py`: identity matching, exclusions, dedup, caps and hour names.
  - `probe_build2.py`: the gate across two runtimes, and the stranger's row classification under drift.
  - `unicode_gate.py`: the gate on a count written in digits from two Unicode versions.
- **Housekeeping.** One early probe run under `-I` wrote a `__pycache__` into the builder's `papers/sealed-ahead/`, because `-I` ignores PYTHONDONTWRITEBYTECODE. It was deleted at once (created 09:20:07, before the builder's next commit). Later runs used `-B`.

**In brief.**

- **Scorecard.** Of the battery's 37 attacks, the build implements the predicted check for 11, implements part of it for 13, and lacks it for 12; 1 needs nothing. Ten of the 12 missing belong to parts the build says it has not built: the plan and its anchors, the panel and decoys, scoring, and the RESULT. Two are in scope and missing: SA-27 (clusters) and SA-29 (a second derivation of the diff facts).
- **Convergence.** Builder and battery reached three structural conclusions independently:
  - the description as opened cannot be fetched without a token;
  - the 50-an-hour budget puts 60 accusations out of reach in a fortnight (the README says 77 days for 46,000 records; the battery says 1,840 h);
  - a description read now must not be judged against the diff at opening, and the build enforces this (`require_unmoved_head`, not switchable).
- **Erratum to the battery.** The build found one battery fact wrong (section 1).
- **New defects.** Eleven, from reading and probing the build (section 3). Two have measured evidence:
  - **D1, runtime.** The pinned 7.49.0 gate gives 0 accusations on CPython 3.12.10 and 1 accusation on 3.14.2 for the same title, body and diff. A stranger on another runtime therefore reads an honest lab as INCONSISTENT.
  - **D2, drift.** The stranger classes a substituted body as DRIFT, not INCONSISTENT, whenever the pull request was touched after the lab's self-reported fetch time. A merge is enough.

## 1. Erratum to the battery, learned from the build

The builder noticed that GH Archive does not zero-pad the hour: hour three of 2026-10-05 is `2026-10-05-3`. Two HEAD requests made for this comparison confirm it: `2026-10-05-3` answered 200 with 17,712,148 bytes, and `2026-10-05-03` answered 404.

The battery repeated the attack pass's padded URL, so these battery statements are wrong:

- the 404 row of the 1.1 table;
- 1.1 item 4 ("Gaps persist ... a gap can last at least seven days");
- the measured sentence under SA-21 ("a 404 persisted at least seven days").

No real gap was observed in any hour probed. SA-21's mechanism still stands, but its measured support goes.

The battery also lacked an attack the builder caught:

- **SA-38. A URL shape manufactures gaps.** A pipeline that pads the hour reads ten of every 24 hours as missing. A dishonest lab could pass such hours off as gaps.
- **The build's answer.** `common.hour_name` never pads, and `common.parse_hour_name` refuses a padded single-digit hour, so the mistake cannot reach a gap list (test `test_hour_names_are_not_padded_and_padding_is_refused`).

The battery stays as committed; this section is its correction.

## 2. Scorecard

Status: IMPLEMENTED means the battery's catching check exists in code, and a test or probe exercised it. PARTLY means some of the check exists, and the gap is named. MISSING means nothing in the build does it. Where the missing part belongs to a stage the README says is not built, the row says "(not built)".

| ID | battery's catching check | status | where in the build, and the gap |
|---|---|---|---|
| SA-01 | the issue is the only valid date; stamps are durability only | MISSING (not built) | no plan, issue, memo or stamp handling exists; `params_sha256` is computed but nothing reads an anchor |
| SA-02 | count every plan digest (wallet memos, archive IssuesEvents) | MISSING (not built) | none |
| SA-03 | memo of the same digest; IssuesEvent body | MISSING (not built) | none |
| SA-04 | anchor the accusation set's digest before H | PARTLY | `draw.checked_draw` refuses SET_NOT_ANCHORED_BEFORE only when `--anchor-time` is given, and that time is a typed string, not read from any anchor; serialization is fixed in code (`accuse.accusation_items` plus `styxx.beacon.pool_sha256`) |
| SA-05 | one integer H in P | IMPLEMENTED | `draw.bitcoin_height` in params; HEIGHT_NOT_THE_PLANS. Gap: no fallback height, so a refused order sends the lab back to choose a new H |
| SA-06 | H tied to the set's anchor, with a deterministic fallback | PARTLY | BLOCK_NOT_AFTER_WINDOW compares the block with t1, not with the set's completion; the anchor check is optional (as SA-04); no fallback |
| SA-07 | fix byte order, case and the draw function | IMPLEMENTED | display-order lowercase id, `styxx.beacon.select`, `beacon_module_lf_sha256` recorded (`test_draw_is_styxx_beacon_select_and_nothing_else`) |
| SA-08 | most-work chain, confirmations, two explorers, proof of work | PARTLY | min_confirmations, two-explorer agreement and BIP34 proof exist in the `--fetch` path; the manual path accepts any valid-work header with a typed tip height and `height_proof: none`; `verify_chain` is written and never called |
| SA-09 | none needed | n/a | n/a |
| SA-10 | P names the primary build; subset test 7.49.0 ⊆ 7.48.1 | PARTLY | one gate sha256 in params; each withheld claim records `withheld_from`, so one run prices the overlay; no subset test |
| SA-11 | pin the closure, interpreter and unicodedata | PARTLY | `load_gate` pins diffgate.py only; python, unidata and the styxx closure are recorded in `accusations.json`, not pinned and not compared by the stranger. D1 shows the gap is live |
| SA-12 | P names door, summary composition and unit | PARTLY | fixed in code (`gate_diff_text`, `title + "\n\n" + body`, unit = claim), but no digest of the pipeline's own code is in the params or the records (D7) |
| SA-13 | every listed kind reported | IMPLEMENTED | every CONTRADICTED claim of any kind is an accusation; `accusations_by_kind` printed |
| SA-14 | a gates block for UNSURE, aggregation, ties | MISSING (not built) | no panel scoring |
| SA-15 | identity table anchored before the sizing count | PARTLY | rules digest `2fce0c15...` pinned by a test before any sizing ran; the table itself says E4 (observing each string in the sizing week) comes next, which leaves room to edit it after seeing that week; nothing orders the two outside git |
| SA-16 | archive-only population re-derivation; every record has a row | IMPLEMENTED | `stranger.check_pool` compares per-hour `admitted_sha256` and `pool_sha256`; missing or extra rows give MISMATCH (`test_stranger_pool_refuses_a_forged_digest_and_names_a_differing_hour`) |
| SA-17 | re-fetch every excluded record; ages from the archive | IMPLEMENTED | `classify_row` marks a set-aside record that reads now as INCONSISTENT when the pull request is unchanged since, or always for gone and unavailable records (an EMPTY_BODY record touched since reads as DRIFT); ages by id ceilings, as the battery proposed. Gaps: the stranger does not re-derive the ceilings (D6); no cap on the excluded share |
| SA-18 | diff content-addressed; body audit | PARTLY | diff: IMPLEMENTED (`diff_sha256`, INCONSISTENT on difference). Body: a substitution cannot reach AGREES, but it reads as DRIFT whenever the pull request was touched later (D2), so it cannot be told from honest drift |
| SA-19 | one fixed fetch; a printed edit-drift rate | PARTLY | `body_source: api_now` and the fetch time recorded; the ledger fetches each record until it is final, but nothing outside the lab enforces one fetch; no drift rate |
| SA-20 | gate only an unmoved head | IMPLEMENTED, with a race | HEAD_MOVED from the REST head (`test_accuse_never_judges_a_moved_head_and_spends_one_request`). Gap: the head is checked before the diff GET, which can wait on the limiter (D3) |
| SA-21 | hour manifest with sha256, anchored nightly; gaps never filled | PARTLY | gap only after the grace and a second 404, per-hour `compressed_sha256`, gaps replayed by the stranger; a gap that answers now is reported, not refused (D5); nothing anchored nightly |
| SA-22 | key (repo.id, number), action opened, window by event time | IMPLEMENTED | `population.classify` plus `pool.assemble` (probe: reopened is skipped; t1 is excluded; a repeated event is deduplicated) |
| SA-23 | no terminal 403/429; every record has a status | IMPLEMENTED | rate-limited means FETCH_FAILED, never final; COMPLETE only when every record is final; the limiter keeps its stamps across restarts. The infeasibility is stated in the README; no pure-function subsample (none is needed if the budget changes) |
| SA-24 | table by digest; exact match rules | IMPLEMENTED | logins compared exactly up to ASCII case, prefixes exactly, never a substring (probe: `Claude/` and a lookalike slash are not admitted, and an app login containing a vendor word is not either) |
| SA-25 | caps, pre-T0 stratum, app stratum, declared claims apart | PARTLY | id-ceiling age rule and per-repository and per-actor caps; caps may be null (probe); `third_party_can_forge` is recorded per rule but no stratum is reported; declared-claim accusations are not separated; no commit-then-reveal |
| SA-26 | the lab's orgs and accounts listed | IMPLEMENTED | `excluded_owners`, `excluded_actor_logins` (probe). Minor: a pull request whose head repository is a lab fork into another owner's repository is admitted |
| SA-27 | one accusation per pull request and account, or a cluster-robust interval | MISSING | the unit is the claim; a pull request with several CONTRADICTED claims puts several items in the draw; no cap at the accusation level |
| SA-28 | re-run the pinned gate | IMPLEMENTED | equal inputs with other claims give INCONSISTENT (`test_stranger_accuse_catches_another_gate_and_a_forged_set_digest`). D1 makes this fire on an honest lab when the runtimes differ |
| SA-29 | a second derivation of each drawn item's changed files | MISSING | the gate reads the `.diff` text only; nothing cross-checks it |
| SA-30 | packet without url, agent, why or overlay fields; leak probe | MISSING (not built) | no panel. `draw.json` lists accusation ids only, so a published draw names no repository |
| SA-31 | perturb the evidence; decoys per kind | MISSING (not built) | none; see D9 |
| SA-32 | beacon-drawn decoys; a hard tier | MISSING (not built) | none; see D9 |
| SA-33 | a secret salt | MISSING (not built) | none |
| SA-34 | pinned local seat, one run, prompts in P | MISSING (not built) | none |
| SA-35 | packet items carry the gate's input hashes | PARTLY | `title_sha256`, `body_sha256`, `diff_sha256` per outcome; inputs of accused records kept under `inputs/`; no packet yet |
| SA-36 | a RESULT deadline in P | MISSING (not built) | none |
| SA-37 | three-valued, layered comparison | PARTLY | MATCH / DRIFT / UNREPRODUCIBLE / INCONSISTENT with exits 0, 1 and 2 is the three-valued comparison the battery asked for; but it re-fetches every record (no sampled layer), and the draw is confirmed only if every row is MATCH, so one edited description anywhere blocks confirmation (README says so) |

Totals: IMPLEMENTED 11 (SA-05, 07, 13, 16, 17, 20, 22, 23, 24, 26, 28). PARTLY 13 (SA-04, 06, 08, 10, 11, 12, 15, 18, 19, 21, 25, 35, 37). MISSING 12, of them 10 not built (SA-01, 02, 03, 14, 30, 31, 32, 33, 34, 36) and 2 in scope (SA-27, SA-29). n/a 1 (SA-09).

The battery's sealable predictions, against the build:

- **Checkable now:** P1 (`check_pool`), P3 (with `--allow-compare`, after D4 is settled), P14 (final outcomes only).
- **Partly checkable:**
  - P2: verdicts are re-run on re-fetched bytes, not on the recorded bytes, so drift masks it.
  - P4: the anchor time is optional and typed.
  - P6: set-aside records that read now are INCONSISTENT, but no with-and-without precision is printed.
  - P11: `withheld_from` gives the counterfactual; there is no subset test.
  - P13: hours are compared; a gap that answers now is reported, not refused.
- **Not checkable yet:** P5, P7, P8, P9, P10, P12.

## 3. Defects found by reading and probing the build

**D1. The gate pin does not pin the verdict.**
- **Probe.** `probe_build2.py` loads the builder's `accuse.load_gate` with the 7.49.0 diffgate (sha256 `09867056...`, matched as bytes) and calls `accuse.gate_outcome` on the title "Tidy", the body "This change: 𖄱 files changed." (U+16131, a digit assigned in Unicode 16) and a two-file diff.
  - CPython 3.12.10 (unidata 15.0.0): 0 accusations, `claims_sha256` `4f53cda1...`.
  - CPython 3.14.2 (unidata 16.0.0): 1 accusation (files_changed_count), `claims_sha256` `5da95c10...`.
- **Both releases.** `unicode_gate.py` gives the same split for v7.48.1 and v7.49.0. PATH-2a does not withhold this claim, and "٣ files changed" (U+0663) accuses on both runtimes.
- **What follows.** `stranger.classify_row` calls such a row INCONSISTENT ("the same title, body and diff gave other claims: not the plan's gate"). That blames an honest lab, and it also lets a lab pick a runtime.
- **Fix.** The params name the interpreter version, `unidata_version` and the closure digest. `load_gate` refuses on a mismatch, and the stranger compares the published gate identity before classifying rows.

**D2. DRIFT is decided by times the lab reports and that merges move.**
- `classify_row` reads a description difference as DRIFT when the pull request's `updated_at` is later than the lab's `body_fetched_at`.
- `updated_at` moves on any later activity, including a merge or a close.
- `body_fetched_at` is the lab's own unverifiable claim.
- Probe results (synthetic rows, the builder's function):
  - substituted body, pull request merged later: DRIFT;
  - substituted body, pull request untouched since: INCONSISTENT;
  - substituted body with the lab's fetch time backdated to near opening: DRIFT;
  - a record set aside as EMPTY_BODY that gates now, merged later: DRIFT.
- **What follows.** A substitution never reaches AGREES. That is the strength. But on most real pull requests it is indistinguishable from honest drift, which is the battery's SA-18 and SA-37 prediction.
- **Fix.** Measure and print the honest drift rate; run a concurrent observer during the window; or publish the inputs of every gated record, not only accused ones, so a stranger can at least re-run the gate on the recorded bytes.

**D3. Check-then-fetch race on the pull route.**
- `accuse_one` compares the REST head with the event's head, then fetches `pull/<n>.diff` with a second GET that waits on the limiter before it is sent.
- At the 50-an-hour cap that wait averages about 72 s and can be longer.
- Nothing ties the `.diff` bytes to `head_sha`, so a push in that interval gates the wrong change.
- **Fix.** Take the head from the last `From <sha>` line of `pull/<n>.patch`, or re-read the head after the diff, or use the compare route.

**D4. The pull and compare routes are assumed byte-identical.**
- A stranger who re-checks with `--allow-compare` compares `diff_sha256` from the compare URL with the lab's pull-route digest, and any byte difference is INCONSISTENT.
- That GitHub serves identical bytes on both routes for the same head is untested here, because this comparison made no github.com request.
- Before the plan freezes, a handful of public pull requests within the grant should be fetched both ways.

**D5. A gap that answers now does not refuse.**
- `check_pool` re-fetches every published gap and lists the ones that answer, with the pool digest they would give (`gaps_now_available`).
- But `checks` ignores that list, so the verdict can be AGREES while a "gap" holds admitted records.
- A fake gap, battery SA-21(a), is therefore reported but accepted.
- **Fix.** A published gap that now answers with admitted records gives at least CANNOT_CONFIRM, and the gap list is anchored nightly.

**D6. The stranger does not re-derive the age ceilings.**
- `validate_params` checks that `ceiling_hour` is the hour `young_days` before t0. It does not check that `account_id_ceiling` and `repo_id_ceiling` are that hour's largest ids.
- `check_pool` takes them from the params.
- **Fix.** `stranger.py pool` re-runs `pool.py ceilings` and compares.

**D7. The pipeline's own code is not pinned.**
- The params name the gate's sha256 and the identity rules' digest, and `draw.json` records `styxx/beacon.py`'s hash.
- The params do not name the hashes of `population.py`, `pool.py`, `accuse.py`, `draw.py` or `common.py`, and no record carries them.
- The door, the summary composition (`title + "\n\n" + body`) and the unit are set in `accuse.py`. So they can change without changing `params_sha256`, and SA-12 stays open at the code level.
- **Fix.** A `pipeline_sha256` (LF digest of the five scripts) in the params, checked by every stage and by the stranger; or P names the commit and the stranger runs from it.

**D8. The planting caps are optional.**
- `max_prs_per_repo` and `max_prs_per_actor` may be null (probe: the params validate).
- The README calls them the bound on one holder of an old account and an old repository.
- **Fix.** Require them, or require the plan to say in words why they are null.

**D9. Inputs are kept only for accused records.**
- `run` saves title, body and diff under `inputs/` only when a record has an accusation.
- VERIFIED-claim decoys drawn from the same window (battery SA-31 and SA-32) would need a later re-fetch, which is drift-prone, so the decoys would be built from other bytes than the gate read.
- **Fix.** Keep the inputs of every gated record with at least one claim. They are small, and the code already returns them.

**D10. The draw's order and height checks rest on what the operator types.**
- `--anchor-time` is optional and not verified against any anchor.
- With `--header` and `--tip-height` and no coinbase, `checked_draw` accepts any header with valid work as the block at the plan's height (`height_proof: none` is recorded).
- The stranger's own header fetch catches a wrong one, but the lab-side refusal list does not.
- No fallback height is named for a refused order.

**D11. Clusters reach the draw.**
- The unit is the claim, and `accusation_items` yields every CONTRADICTED claim.
- One pull request with three count sentences puts three correlated items into a 60-item draw. The pool's per-repository and per-actor caps act on pull requests, not on accusations.
- This is battery SA-27, unaddressed.

## 4. What the build does better than the battery asked

- **Hour names.** The padded-hour finding (section 1).
- **Pre-trim bodies.** It keeps a pre-2025-10 payload's title and body as `as_opened` and prefers them, so the description-as-opened route works wherever the archive still has it.
- **Height proof.** The BIP34 height is proved from the coinbase and its merkle branch, so the draw does not rest on an explorer's height label (`test_bip34_height_is_proved_from_the_coinbase_and_its_branch`).
- **The published form.** It never says which records were accused (`digest_row`, `draw.json` lists ids only). That keeps the record free of named accusations. The cost is SA-37's every-row-must-match rule.
- **The limiter and the overlay.** The rate limiter persists across restarts, and the overlay's withholding is recorded per claim, so one run prices 7.48.1's accusations as well.

## 5. Not checked here

- The network paths (GH Archive streaming in `derive`, the GitHub fetches, the explorers) were read but not run.
- Assumption A-1 (ids rise with creation time) was not tested.
- D4 was not tested.
- The identity strings (E3 in the table) were not checked against observed events by name; the battery's probe counted identification routes only.
