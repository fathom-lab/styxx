# AMENDMENT — the window's scorer: score.py named and lock-hashed, a value for three shares at a zero denominator, and how it reads the rules the PREREG states

Fathom Lab · 2026-11-09 · Amends `PREREG_darkflobi_gate_2026_10_23.md`, in this directory, merged as #229 with its
section 15 closed at 15.13 and not edited, read with the eight amendment files at `origin/main` `9cec06e5`, through
whose `lock_hashes.py check --amendment` this file is read, after them. Section 8 asks for it: "the packet and scorer
adaptations (`build_packets.py`, `score.py` for posts) are lock-hashed in a dated AMENDMENT before close and tested on
synthetic results only" (P l.414-416); section 12's closing paragraph gives the route, a new AMENDMENT file under this
folder, lock-hashed in its turn (P l.676-678); and 15.12 names what the scorer implements, "the per-stretch reading of
(iv), and `unledgered_deliveries` summed over it" (P l.1247-1250). Written on 2026-11-09 by the lab's builder for the
lead, from the spec `SP:SPEC_score.md`, while `W:gate/REQUIRED` is absent (read 2026-11-09T18:21Z, presence only): D0
is not taken and no line of the live cell exists. It does three things: (a) it names `score.py` and its tests and
locks their bytes, with the PREREG's own blob beside them; (b) it declares a finite value for three shares whose
denominator can be zero; (c) it says how the scorer applies rules the PREREG already states, each reading marked [J].
It moves no bar, no row of the gates block, no threshold, no population rule, no reason code and no printable
sentence, and no file under `W:gate`, so no `gate_sha256` stretch opens. It asks the operator for the merge and for his
word on the readings of item 7 that decide a gate (7.1, 7.2 and 7.3).

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `SP:` = `WT:_scorer_notes_2026_11_09/` (the spec and the
build log, `SP:BUILD_LOG_scorer.md`); `P l.N` = this folder's PREREG at `9cec06e5`, `R l.N` = its RULES file, `A:<topic>
l.N` = this folder's `AMENDMENT_darkflobi_gate_<topic>_<date>.md`, each with its own line numbers; `T` =
`tests/test_darkflobi_gate_prereg.py`; `S` = `tests/test_darkflobi_gate_score.py` (new) and `S:T<n>` its test of that
name; `CM` = `papers/sworn/measurement/common.py` at the pin; `G:<file>` = a file of the live gate, read and never
written. [R] marks a reading with its receipt, [J] a judgement. A digest written `abcd1234…` is a sha256 whose full form
stands in a lock row, in `score.py`'s constants (each with its line receipt) or in the document named.

## What the PREREG says

- The receipt: "`measurement_result.json`, written at close by the scorer with every gated metric under `result.`,
  scored over the whole receipt" (P l.7-8); the scorer is `styxx.protocol.Experiment` over the gates block of §6
  (P l.4-6), whose table semantics are its own: "rows are walked in order and the earliest row whose `when` matches
  wins" (P l.311-312).
- The result dict: "the ten gated metrics, each a number or the receipt is scored `INVALID__exits_open` by the
  plumbing rule below" (P l.225-226), listed as nine keys (l.228-256), since `families_clearing_decoys` feeds two
  gates (l.316-318).
- The four shares and their denominators: `refusal_upheld_share` over `n_refusals`, "−1.0 when `n_refusals` is 0, a
  sentinel the table never reads because G_RN is walked before G_R" (P l.249-251); `bound_share_cell1` over `n_cell1`
  (l.252); `padding_share` over "all HELD or VERIFIED spans in the live cell" (l.253-254); `wrong_leaf_share` over "all
  HELD or VERIFIED spans Panel R judged" (l.255-256). Of the four, only `refusal_upheld_share` has a value at a zero
  denominator.
- The plumbing rule: "If any gated metric is missing or not a finite number, the scorer writes 1 into
  `result.unledgered_deliveries`, records the names under `result.nonfinite_metrics`, and scores: the frozen table's
  `INVALID__exits_open` row carries the case, not an exception. `score.py` refuses to run before the closing date and
  without this PREREG at HEAD" (P l.258-261).
- The order at close: "the census export, `reconcile.py`, `gate_verify.py`, the packets, the two panels, `score.py`,
  the RESULT sworn to `measurement_result.json`" (P l.671-672); "Nothing is scored before close" (P l.676).
- Component (iv), read per stretch: "The RESULT reads each send line against the digest of the files locked when the
  line was written ... a line carrying a digest other than its stretch's, or any fourth digest, counts into (iv)"
  (P l.1240-1243), signed "Per stretch (Recommended)" (P l.1277).

## What was observed

1. **No scorer for posts exists** [R, the folder's listing at `9cec06e5`; `WT:_improve_2026_11_09/gate-measurement-debts/NOTES.md`
   l.25-31]: the only `score.py` in the repository is the sworn measurement's, which has no closing-date refusal and no
   plumbing rule.
2. **`Experiment.score` raises where the PREREG wants a row** [R, the pin's `styxx.protocol` l.174-180 and l.474-479;
   NOTES l.35-38]: a missing path raises `GateSpecError`, and so does a NaN on its gate, so the plumbing rule can only
   live in `score.py`.
3. **A quiet window would read as an open exit** [R, NOTES l.39-43]: in a window with no cell-1 unit and no span,
   `bound_share_cell1`, `padding_share` and `wrong_leaf_share` are 0/0; left NaN, the plumbing rule turns each into
   `INVALID__exits_open`, whose sentence (P l.487-491) says deliveries reached a destination with no ledger line, which a
   quiet window does not show. `S:T1m` reproduces it with the declared values switched off.
4. **Seven digests, six stretches with lines** [R, `score.py` `STRETCHES`, each entry with its receipt; P 15.3 l.804-813,
   15.6 l.967-979, 15.10 l.1169-1179; A:lock_rows l.171-179; A:anchor_native l.319-327; A:token_shapes l.196-199 and
   l.263-268; A:substrate l.159 and l.987-1001].
5. **The locked `reconcile.py` reads (iv) against one digest** (P 15.12 l.1244-1247): run with the last stretch's, it
   counts every older line, the six dry attempts among them.
6. **`styxx.mind.wilson(k, 0)` is NaN** [R, the pin's `styxx/mind.py` `wilson`, its `n == 0` branch].
7. **No report pairs a line with its agent** [R, the spec's reading of `G:reconcile.py` l.840-842: `match_transcripts`
   returns counts]: R §2's agent of a line (R l.92-97) is printed by no tool that exists; the scorer reads it from an
   attribution file (item 1) whose helper is not built (item 10 of what this amendment does not say).
8. **`main` has moved two instruments since the pin** [R, `git show <rev>:<path> | sha256sum`]: `styxx.sworn` reads
   `3f32da0f…` at `9cec06e5` against the pin's `7e460247…`, and `styxx.certify` moved too; `styxx.protocol`,
   `styxx.mind` and the measurement's `common.py` are the pin's bytes today. A scorer that imported `main`'s files would
   run whatever `main` holds at close.

## The amendment

1. **The scorer** is `papers/darkflobi-gate/score.py`, beside `lock_hashes.py`, a repository file that enters no
   `gate_sha256` (`Env.gate_sha256` hashes `W:gate/*.py`, `W:gate/helpers/*.py` and the installed plugin folder, A:canary_replan
   l.407-411), run at close, last of P l.671-672's order, from a checkout that holds this file at HEAD:

   ```text
   py -3.12 -I -B papers/darkflobi-gate/score.py --ledger <a copy of the ledger taken after close> --d0 <D0>
      --reconcile <agent>=<census JSON> (one per folder under ~/.clawdbot/agents/) --verify <gate_verify --rederive JSON>
      --attribution <per-seq agent JSON> --canary <DIR> --panel <DIR> --panel-sealed <DIR>
      --out papers/darkflobi-gate/measurement_result.json
   ```

   It is standard library only. Its instruments are the pin's (`3437340b`): `styxx.protocol` for `Experiment`,
   `styxx.mind` for `wilson`, and the measurement's `common.py` with `styxx.sworn` under the name it imports, for
   `majority`, `final_label`, `family_label`, `locate`, `project_labels`, `bindable` and `cohen_kappa`; each is read
   from `git show 3437340b:<path>`, run only when its bytes hash to the digest its row or its receipt names, and never
   replaced by another copy; the `styxx` package is never imported. Nothing of `W:gate` is imported, run or written.
   It writes one file, LF, sorted keys, indent 1, one trailing LF, through a temporary file and `os.replace`, after the
   bytes are read back, compared with the dict that was scored and scored again by `Experiment.score` over the whole
   receipt (the verdict and every gate equal). No wall-clock time enters the receipt. Exit 0: written; exit 2: a
   refusal of item 2, `REFUSED: R<n> <check>: <why>` and a one-line JSON record on stderr, nothing written; exit 1: an
   error after the inputs were accepted, nothing written. The inputs' shapes: the ledger copy as the gate writes it;
   reconcile's census report as written (its `window`, `inputs.d0`, `inputs.locks_present`, `chain.head`, `census`,
   `unledgered_deliveries`, `transcripts`, `exec_scan`); `gate_verify.py`'s report with `rederive`; an attribution file
   `{schema: darkflobi.gate/attribution/v0, lines: {seq: folder | "unattributed"}}`; the canary folder (the D0 schedule
   copy, the released seed key and the seal's salt, `key_revealed.json`, `verify_released.json` holding
   `verify-released`'s exit and stdout); the panel folder (`manifest.json` of schema `darkflobi.gate/panel-manifest/v0`,
   `packet_L.json`, `packet_R.json`, `keys/<name>.sha256` committed, `<family>/<L|R>-seat<n>.json`) and the released
   keys and their salt. **Synthetic mode** (`--synthetic --constants FILE --now T`, the tests only): every input must
   carry `"synthetic": true`, the constants file supplies the lock values, the stretch table, the D0 on record, the
   schedule and scorer digests and a stand-in gate folder, may name no real lock constant, and the output goes only
   under the fixture's folder; outside it `--now`, `--constants`, `--repo` and a mutation switch are refused.
2. **Its refusals**, in the order checked, each exit 2 with nothing written: R11 the mode (above); R1 the close, the
   clock before 00:00 America/New_York on D0 + 28 (P l.183-185, l.258-261, l.414-416); R2 the PREREG at HEAD: `git
   ls-tree HEAD` names its blob, that blob's sha256 is the seal `31d0f079…` (`protocol_scan_2026_10_23.txt` l.63, the
   row of item 9), the working file's LF bytes are that blob, `Experiment` with `require_power_basis` and
   `require_nonvacuous_gates` accepts it, its gates digest is `4ed27cb98f72…` over 6,354 bytes (P l.544) and its
   adding commit is `dc06d68e…` (`protocol_scan_2026_10_23.txt` l.6; P l.664-665); R3 the scorer is the locked
   scorer: this file's row (item 9) is the sha256 of the running file's LF bytes and of HEAD's blob of it, an
   instrument's pinned blob hashes to its digest, and the commit that adds this amendment is dated before the close
   (P l.416); R4 D0 taken: the canary amendment's addendum at HEAD names a D0 (its "D0:" line, A:canary_replan l.482)
   and `--d0` is it, and the ledger holds a line with `required_present: true` whose earliest local day is not before
   D0 − 1 (P l.183-184); R5 the ledger: a copy, never a path under `W:gate`, that re-derives by `G:ledger.py`'s rule
   (`read_log` l.253-292, `check_chain` l.295-328, restated in the scorer), whose head is the head the verify report
   and every reconcile report name; R6 reconcile: a report for every agent folder the attribution names, each run
   with `--d0` D0 over [D0, close) with the four `--locks`, one census part across them; R7 gate_verify: a `rederive`
   block and `chain.chain_ok`; R8 the canaries: the schedule copy hashes to the D0 addendum's row, the released seed
   key's salted digest is `c2254ae1…` (A:canary_replan l.233), `key_revealed.json`'s seed and window start and the
   schedule's cadence are the key's, D0's and `s7`, `verify-released` read MATCH and SAME with exit 0, and the
   schedule's `counts.gc_items` is the number of scheduled A-D items; R9 the panel (item 7.4 and 7.6); R10 attribution:
   the file present, every seq it names a send line, every public candidate of the window named.
3. **The result dict** under `result.`, the nine keys of P l.228-256 with their names, `nonfinite_metrics` (l.259) and
   one list this amendment adds, `zero_denominator_metrics`, the shares that took the values of item 4; JSON sorts the
   keys (the PREREG fixes no order). Beside each share, under `components.counts`, its k and its n; shares are written
   unrounded and printed to four places.
4. **The values at a zero denominator** (b). `bound_share_cell1` is −1.0 when `n_cell1` is 0; `padding_share` is 2.0
   when the posted payloads of the live cell hold no HELD or VERIFIED span; `wrong_leaf_share` is 2.0 when Panel R
   judged no span; `refusal_upheld_share` keeps l.251's −1.0. The rule behind them: a share whose denominator is zero
   takes a value outside [0, 1] on the failing side of its own bar, as l.251's −1.0 does for a ≥ 0.95 bar; so −1.0
   for the ≥ 0.70 bar of G_B and 2.0 for the ≤ 0.25 and ≤ 0.10 bars of G_S2 and G_P. Not 0.0, which is a real share
   (none of n bound, no span on a NOT unit); not −1.0 for all four, which passes the two ≤ bars, and since
   `Experiment.score` evaluates every gate before it walks, the receipt would print G_S2 and G_P true for a window with
   no span. **The walk** (P l.297-307): G_N is true in the `when` of rows 6-11 and false in row 5, and rows 1-4 do not
   name it; G_B is named only in rows 7-11, G_S2 only in rows 8-11 (each with G_B true), G_P only in rows 9-11 (each
   with G_B and G_S2 true), G_R only in row 6 (with G_RN true); over the assignments with G_N false, rows 1-5 are total.
   Row by row:
   - `refusal_upheld_share` at `n_refusals` 0 (existing): 0 < 20, G_RN false, row 6 cannot match, and G_R is named
     nowhere else; the verdict is the same for every value (P l.314-316).
   - `bound_share_cell1` at `n_cell1` 0: 0 < 40, G_N false, rows 6-11 cannot match, and rows 1-5 never name G_B; the
     verdict is fixed by `unledgered_deliveries`, `families_clearing_decoys` and `canary_refused_wilson_lower`, and with
     the exits held, two families clearing and the canaries caught it is row 5, `ABSTAIN__underpowered_counts_only`.
   - `padding_share` with no span: G_S2 is read only in rows 8-11, which need G_N and G_B true, which need `n_cell1`
     ≥ 40 and at least 28 of those units bound; a bound unit lies inside a HELD span or holds a VERIFIED numeral on its
     own line (R l.176-186), so the spans are not empty. An empty set of spans forces G_N or G_B false, and the verdict
     comes from rows 1-7 whatever the value; with `n_cell1` ≥ 40 and no span at all, `bound_share_cell1` is 0.0 over a
     real denominator and row 7 is the agent's own count, not the value's.
   - `wrong_leaf_share` with no span judged: with item 7.2's reading the denominator is every span, so the walk of
     `padding_share` applies: G_P is named only in rows 9-11, which need G_B true, which forces spans.

   The scorer keeps the argument at run time: it computes `verdict.decisive_gates` (the gates whose single flip moves
   the row the walk reaches) and `verdict.gates_not_read` (the gates the matching row's `when` does not name), and exits
   1 with nothing written if a gate whose metric took a value of this item is decisive or is read; `S:TW1` checks the
   table's half over the 1,024 assignments and `S:TW2` that each value fails its own bar.
5. **The plumbing rule, as applied.** A metric is missing when its input is present and still does not give it (a
   reconcile report with `complete: false`, a component written null), and not finite when it is NaN, an infinity, a
   bool or not a number. The names go, sorted, under `nonfinite_metrics` (`[]` when none); each named slot takes the
   value on the failing side of its own gate (`unledgered_deliveries` 1, `families_clearing_decoys` 0,
   `canary_refused_wilson_lower` −1.0, `n_cell1` 0, `n_refusals` 0, the shares as item 4), so that `Experiment.score`
   walks to row 1 and raises nothing; `unledgered_deliveries` becomes max(its computed count, 1), or 1 when the count is
   itself missing, so a real count is never overwritten [J, P l.258's "writes 1"]. The `INVALID__exits_open` paragraph
   keeps the components as computed, and one line is printed beside it: "the plumbing rule fired: <names> missing or not
   finite; no delivery is counted by it". An absent input is a refusal, never a missing metric [J].
6. **(iii), and (iv) per stretch** (P l.228-234; 15.12). (iii) is counted by the scorer over the window's send lines
   (D0 ≤ `timestamp` < close), one count per line whose `config_sha256`, `pin` or `salt_sha256` is not the locked one
   ([u_c]) or else whose `required_present` is not true ([u_q]), the two brackets of P l.489. (iv) is counted over
   every send line from seq 1 with `timestamp` before the close, each against the digest of the stretch its seq falls
   in, the stretch being the entry of `score.py`'s table with the largest opening seq at or below it; a neighbour's
   digest, a move's intermediate or a digest the table does not hold counts, and the stretch with no line (`592dc579…`)
   holds none. The table, with its receipts:

   | stretch | digest | opens at | receipt |
   |---|---|---|---|
   | 1 | `387fc063…` | seq 1 | P 15.3 l.804-813; A:lock_rows l.179 |
   | 2 | `592dc579…` | no line | P 15.6 l.967, l.971-979; 15.10 l.1176-1178 |
   | 3 | `ff34b52d…` | seq 14 | P 15.10 l.1169-1179; A:lock_rows l.179 |
   | 4 | `c5084df2…` | seq 126 | A:lock_rows addendum l.171-177 |
   | 5 | `445ac94c…` | seq 135 | A:anchor_native addendum l.319-327 |
   | 6 | `f321b1bc…` | seq 411 | A:token_shapes l.196-199, addendum l.263-268 |
   | 7 | `7249a8f8…` | seq 735 | A:substrate addendum l.987-1001 |

   A move inside the window opens an eighth entry only through a later dated amendment that adds it and supersedes
   this file's row for `score.py` (D9); without one, every line of the new stretch counts into (iv), which is 15.12's
   reading of a digest no row names. Each component is printed beside the sum, with reconcile's own (iii) and (iv), and
   the `exec_scan` lower bound beside it (P l.232-234), which enters no metric. Why the seq and not the time [J]: 15.12
   prints "the seq at which each begins" (l.1243) and every addendum records the opening seq; a line's `timestamp` is
   the gate's statement (R l.560-561), so the stretches' earliest and latest timestamps are printed beside as a check.
7. **The readings the scorer applies**, each [J] and each the operator's to sign otherwise, as 15.8, 15.11 and
   15.12's readings were (15.13):
   1. *The posted population* (P l.142-145, "the ... sentences it posts"; R l.176, "the ledger line of the payload
      that carried it"): the live cell L is the window's send lines with `dest_class: public` and `required_present:
      true`, less the two keyed hashes (P l.174-182), less `payload_kind` `anchor`, `canary` and `tool` (15.8
      l.1046-1056, 15.12 l.1253-1258), agent `main` only (P l.165-168), an `unattributed` line in no cell; `n_cell1`,
      the bound share, padding and wrong leaf are counted over L's lines with `action: send`; `n_refusals` over L's
      lines with `decision: refuse` and a truth code, one per `draft_sha256` (P l.243-248; 15.5 l.931-936), so a
      cancelled refusal is in `n_refusals` and in no share's population.
   2. *"spans Panel R judged"* (P l.255-256): every HELD or VERIFIED span of the posted population, whatever its
      cross-family answer, since Panel R reads "one sentence, one kind and one leaf view per span the gate judged"
      (P l.399-400) and §10 prints both span shares over one bracket, "[pad] of [sp]" (l.510-511) and "[w] of [sp]"
      (l.513). Without this reading the walk of item 4 does not close: an all-UNSURE Panel R would leave a zero
      denominator while G_B is true, and the value would choose between `WRONG_LEAF` and `BOUND`.
   3. *A refused draft with several truth codes*: upheld only when Panel R answers NO, by the cross-family rule, on
      every item of the draft, since every code is an accusation the agent was told (P l.1229-1233); the count with
      at least one NO is printed beside.
   4. *The decoys* (P l.235-236, l.400-402): `families_clearing_decoys` over the Panel L decoys, 15 on each side,
      read by each family's seat majority (`common.majority`, R l.124; an unparsed seat votes `NO-LABEL`, a tie reads
      `NO-MAJORITY`); any other count is a refusal (R9); Panel R decoys, if the packets hold some, are printed beside.
      [c_l] and [c_r] of §10 (l.492) are read as the two sides, LOAD-BEARING and NOT. `majority` equals
      `common.family_label` whenever the result is a label (`S:T13`).
   5. *`absent`* (R l.133-140 lists "a negative over one named committed file"; the pin's `CM` `bindable` returns
      `absent: None`): not bindable in this window; LOAD-BEARING units an `absent` span covers are counted beside.
   6. *The panel's re-derivations* (the manifest is `build_packets`' for posts, item 11 of what this amendment does not
      say): the manifest's population is the scorer's posted population; each payload's `checked_sha256` is its
      line's; each Panel L window's unit split is the splitter's over the item's own bytes (R l.122-123); each unit's
      bindable kinds are `CM` `bindable` over its bytes; each line's span count is `legs.sworn.held +
      legs.certify.verified`; every unit of a selected payload sits in exactly one window and every span has one Panel
      R item; the released keys' salted digests are the committed ones; each of the six seat files of a panel that
      holds items is present and `RECORDED`. A mismatch is a refusal; the scorer labels nothing.
   7. *The canary record* (P l.349-352): the ledger's canary lines of the window matched to the scheduled items by
      `draft_sha256` as a multiset, in schedule order against seq order; n is the schedule's `counts.gc_items`; a line
      after the close, a pass, a refusal for another code and no line are misses; E and F reported beside.
   8. *The window's bounds*: (iii) over D0 ≤ `timestamp` < close; (iv) over every send line from seq 1 before the
      close (15.12 names no "after D0").
   9. *The labels when fewer than two families clear*: two, `final_label`; one, that family's majority (the counts P
      l.494-495 prints); none, `final_label` over both, printed as unvalidated. Rows 2 and 3 read no share.
   10. *A span over several units* (P l.253): it sits on NOT only when every unit it covers is labelled NOT; a unit no
      panel labelled is not NOT; the spans over units of two labels are counted beside.
8. **The printables** (P l.452-526). The paragraph of the row the walk reaches, and the gate sentence's own failures
   whatever the row (G-1 when the census [u] is above 0; G-2 when a window send line re-derives other than SAME, [j]
   counting DRIFT, SKEW, UNRESOLVED and TAMPER; the anchor check when [a], [u'] or [cf] is above 0; the capsule clause
   once per selected posted line with Nn > 0, Nb = Nn and no split footer), each the PREREG's text with its brackets
   filled in order (`S:T8` reads each sentence in P l.458-526). Brackets the scorer does not own stay as the PREREG
   prints them: [w], [gh], [sent], [refused], [leaked], and D (OD-17) in the denominators. #[k] and [date] are #229 and
   2026-10-23 (P l.748); the OD-11 bracket takes its public branch (P l.828-830). No value of item 4 or item 5 is ever
   printed in a bracket: such a share prints "no denominator". The Not-run sentence (P l.525-526) is never the
   scorer's: its condition was foreclosed by 15.4's signature (P l.862-866), and a window that never opens leaves the
   scorer refusing at R4. The denominators of P l.198-200 print under every verdict.
9. **The tests** (`S`, synthetic windows only): every window is built by the test file from synthetic values (digests
   such as `"a1" * 32`, texts marked SYN, a synthetic seed sealed by the seal rule); no ledger line, text, destination
   or digest of the live gate enters a fixture; the real constants are compared with their documents in `S:T16` alone.
   The mutation switches (`STYXX_SCORE_MUTATION`, honoured in a synthetic run only) and the tests each makes fail are in
   the readings.

## The readings

[R, `SP:BUILD_LOG_scorer.md` and its `runs/`; every Python run `py -3.12 -I -B` with `PYTHONDONTWRITEBYTECODE=1`,
but for two child runs of `S:T9`, which drop `-I` so that `PYTHONHASHSEED` is read]

- `py -3.12 -I -B -m pytest -q -p no:cacheprovider tests/test_darkflobi_gate_score.py`: 83 passed, on this box with
  the pin's blobs and the PREREG's history present (on a shallow clone the window tests skip, as the PREREG's own
  tests skip a draft).
- The mutation switches, each over the same file: `nan_sentinels` 3 fail (`S:T1`, `S:T2` twice: a quiet window reads
  `INVALID__exits_open`); `no_plumbing` 10 fail (`S:T4`, ten cases: `Experiment.score` raises on the NaN);
  `single_digest_iv` 25 fail (`S:T1`, `S:T2` twice, `S:T3` six times, `S:T8` thirteen times, `S:T11` twice,
  `S:T13`: every older line counts into (iv), as the locked `reconcile.py`'s single digest does); `no_close_check` 1
  fails (`S:T5`); `no_prereg_check` 1 fails (`S:T6`); and `edit_after_score`, `S:T10`'s own, exits 1 with nothing
  written.
- `scripts/surface_lint.py --files` over this file: 0 hits (`S:T18`).
- The paper's tests and the words tests (`tests -k "darkflobi or never_prints or banned or words"`): 116 passed, 8
  skipped before this change; 199 passed, 8 skipped after it (the 83 of `S` added, `T` reading nine files).
- The strict check of A:lock_rows item 3 with the nine files, `--w-root C:/Users/heyzo/clawd`, on 2026-11-09: nine
  `AMENDMENT` lines in D9's order, this file last (`rows 3`); the fifteen `SUPERSEDED` lines of the token-shapes and
  substrate files; the three new rows SAME `via` this file; rows 105, changed 0, built since 0, absent 0, unreadable 0,
  unlocked 0, bytecode 0 (strict), rc 0 (SAME 104, SKIPPED 1, the `plugins` subtree). With the PREREG alone, as
  before this file: rows 80, changed 16, built since 1, absent 0, unreadable 0, unlocked 22, bytecode 0, rc 1.

## Whose step

- **The operator's**: the merge, the relock of record, and his word on the readings of item 7 that decide a gate:
  7.1 (the posted population), 7.2 ("spans Panel R judged") and 7.3 (a refusal with several codes).
- **The lead's**: the pull request with its CHANGELOG entry; the strict check with the nine files on the box; the D0
  runbook regenerated with the nine files and the scorer's step at close; this file's addendum.
- **Owed before close, each its own dated amendment after this one** (P l.414-416): `build_packets` for posts, which
  writes the manifest of item 1; the decoys and the seats; the attribution helper that writes R §2's agent of each
  line. Until each exists the scorer refuses (R9, R10).

## What does not change

The PREREG and the RULES (not edited); the gates block `4ed27cb98f72…` and every bar (OD-19's list, P l.694); the
cells and the population rule; the reason codes and the matrix; every printable sentence; the eight earlier files'
rows and declarations; every file under `W:gate` (no stretch opens), `reconcile.py`, `gate_verify.py`, the canary
harness and the D0 runbook's gate commands.

## What the RESULT prints

The receipt and the scorer's stdout: the verdict and its paragraph, the gate sentence's failures, the denominators,
the stretch table with (iv) per stretch and its seqs, each component of `unledgered_deliveries` beside the sum, the
`nonfinite_metrics` and `zero_denominator_metrics` lists with `gates_not_read` and `decisive_gates`; and the strict
check's nine `AMENDMENT` lines beside its counts.

## What this amendment does not say

1. That a quiet window measured the agent's binding: it reads `ABSTAIN__underpowered_counts_only`, with counts.
2. That −1.0 or 2.0 is a share.
3. That the scorer checks what the packets or the panel did beyond the re-derivations of item 7.6.
4. That the census report's shape is fixed by this file: the scorer reads the keys item 1 names, as the spec read
   `G:reconcile.py` (the builder did not open it), and a report without them is refused, so a difference is seen at
   close, not guessed around.
5. That the scorer has scored a real window: it has scored synthetic ones.
6. That a value of item 4 decides a verdict: the run-time check of item 4 exits before one could.
7. That `S` exercises the real run's git checks of R3 and R9 (HEAD's blob of this file, the committed key digests):
   those run only outside synthetic mode, on the box at close.
8. That a canary harness row counts: the harness's own tally is printed beside, and the ledger decides (A:canary_replan
   l.356-358).
9. That the posts are true, or that the agent is honest; the PREREG's §14 says the rest.
10. That the attribution helper exists.
11. That `build_packets` for posts, the decoys or the seats exist.

## The lock rows

Three repository rows: `lock_hashes.py check` reads each as the sha256 of the file's LF bytes at the repository's
root, so a CRLF checkout reads SAME. None of the three paths is held by an earlier amendment, so this file declares no
supersession. The PREREG's row is its seal (`protocol_scan_2026_10_23.txt` l.63, 129,757 bytes); `score.py` refuses
without it (R2).

This file carries 3 rows in `lock_hashes.py`'s row form.

| what | path | sha256 | note |
|---|---|---|---|
| the scorer | `papers/darkflobi-gate/score.py` | `52157bcfc9864888db42f48a7e27da3f5721f5b68ebd750dd59d94c3c4ae9cce` | its LF bytes; the scorer of P l.414-416 and 15.12 l.1249; reads the inputs of item 1, refuses on item 2, writes `measurement_result.json` |
| the scorer's tests | `tests/test_darkflobi_gate_score.py` | `022a5a1d2ebc99a726a927ca8ced5759abe7c21a6d8e0a51129ca43f368de969` | its LF bytes; synthetic windows only (P l.416) |
| the PREREG, sealed | `papers/darkflobi-gate/PREREG_darkflobi_gate_2026_10_23.md` | `31d0f0797677664233182fe4d3f55f1533f86ae91d58c5229ae053615478c0eb` | 129757 bytes (LF); the seal of `protocol_scan_2026_10_23.txt` l.63; the row score.py refuses without (R2) |

## Addendum — filled at merge (blank until then)

Filled by the lead after the merge, one commit, nothing above this heading edited.

- the merge commit and its UTC time: _blank_
- `score.py`'s and the test's committed blobs and their LF sha256 (equal to the rows): _blank_
- the strict check with the nine files as run on the box: _blank_
- the test run's line: _blank_
