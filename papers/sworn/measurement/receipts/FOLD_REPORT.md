# tranche A fold -- report, 2026-10-24

status: DONE. tranche A folded with score.py reading its keys from the sealed store (`--sealed`), no key
released, nothing committed, pushed or released, no model run, no seat re-run, no file in the worktree
written or edited. both commands re-run once: outputs byte-identical.

**verdict (tranche A): `VOID_L_R__G_S1_G_S2_G_P_withheld__MISSED__G_G1`**
gates block sha256 `518e661662e0ec1b52cab4958e2a7a168b7e2c1a6a64fc0ab91833c9e69bc038` (equals PREREG l.634),
prereg `papers/sworn/PREREG_sworn_measurement_2026_10_23.md`, prereg_commit `01e9d2b772d036b906b80c234ce32473c2f1ae39`.
`j`, outside seats that answered: 0.

Worktree `C:/Users/heyzo/clawd/wt/leg2-run`, HEAD `7680159d0448e331ed82423cfbb080da4516e223` (main, Amendment 6 merged as #235).
Every line number below is in that checkout.

---

## 1. step 1, the decision: fold now, keys read sealed, not released

**Answer: allowed now.** Folding does not release a key, and the PREREG orders tranche A's fold before
tranche B and puts it on a seven-day clock that is running.

### 1a. "sealed" means not moved into the tree; score.py reads the sealed store in place

- PREREG l.400-404, Amendment 2 item 3: "Tranche A's keys stay sealed until tranche B's seat outputs are
  recorded, or tranche B closes with no fold (...), because tranche B's decoys are tranche A's: a short close
  that still builds tranche B keeps them sealed until its seats have answered (review 3, m4; `seal_key.py
  release` moves keys into the tree, machinery README l.39-44)." The act the item guards is named in its
  own parenthesis: `seal_key.py release`.
- machinery README l.39-44: "Keys (`{item_id: meta}`) and canary twins live under `$STYXX_SEALED_DIR` (default
  `C:\Users\heyzo\clawd\styxx-sealed`) and never in the tree. ... `seal_key.py release` moves the plaintext
  into the tree in a commit made after every seat output is recorded, and refuses without the flag whose name
  says what it asserts."
- score.py l.510-511 and l.387: `key_L = K.load_key("sworn_measurement_key_L", sealed, keys_dir)`,
  `key_R = K.load_key("sworn_measurement_key_R", sealed, keys_dir)`, `key = K.load_key(CAN.CANARY_KEY, sealed, keys_dir)`.
- seal_key.py l.86-93, `load_key`: "The key, only when its salted digest equals the committed one -- the
  scorer's refusal"; it reads `Path(sealed or C.SEALED) / (name + ".json")` (l.56) and returns the parsed
  dict in memory. The only function that copies a key into `keys/` is `release` (l.96-108,
  `shutil.copyfile(kp, out_key)`, `shutil.copyfile(salt_path(sealed), out_salt)`), behind
  `--every-seat-output-is-recorded` (l.29, l.98-99).
- `grep release` over score.py and verdict_input.py: no match. score.py's only write is
  `C.write_json_lf(out_path, result)` (l.560). verdict_input.py writes only `--out` (l.216-220).
- PREREG l.411-412, Amendment 2 item 5, names `--sealed` as score.py's route: "The seats, the trivial
  twins and the fold run with the committed CLIs pointed at tranche B's directory (`--dir`, and `--sealed`
  for `score.py`)." Amendment 6 l.165-166: "The seats and the fold still run with the committed CLIs, in both tranches."
- The result carries no key: `measurement_result.json` holds counts per side, per twin and per cell; a
  search of it for an item id (`"L-dddd"`, `"R-dddd"`), `salt`, `decoy_id` and `keyed` finds none.

### 1b. the PREREG orders tranche A's fold now, before tranche B

- PREREG l.506-513, section 4: "Order: section 0 (...); the transport check (one synthetic item); tranche A's
  six Claude seat files in one usage window (...); the six local seat files on the GPU by day; then
  `score.py` over tranche A and `verdict_input.py --tranche A`. The plan runs the panel the same week as the
  signature (PLAN l.85), so tranche A's fold and its RESULT are due within seven days of the signing commit."
  Tranche B comes after, l.513-517: "Tranche B: once its fourth document has entered the mainline (...),
  `tranche_b.py build`, the lock addendum, then the same ladder over `tranche_B/`".
- PREREG l.623: "The block is scored once per fold: tranche A, then tranche B."
- PREREG l.745-747, section 7, plans a tranche A RESULT that exists while the keys are still sealed:
  "Tranche A's RESULT says that tranche A's keys stay sealed until tranche B's seat outputs are recorded, or
  tranche B closes with no fold (...), which can be after `CLOSE_AT`, so until then no outside reader can
  re-score tranche A against its keys."
- Clock: the signing commit `01e9d2b7` (the commit that added the PREREG, `git log --diff-filter=A`) is dated
  2026-10-23T08:04:55-04:00; the fold ran 2026-10-24T20:45:54Z, day 2 of 7; due by 2026-10-30T08:04:55-04:00.

### 1c. preconditions checked before the run (all hold)

| check | value | where the PREREG says it |
|---|---|---|
| PREREG tracked at HEAD | `papers/sworn/PREREG_sworn_measurement_2026_10_23.md` (only match of the runners' glob) | common.py l.575-577 |
| score.py blob at HEAD | `3a5bdb8cd5057a0cf5eccf7343d9d05ee4ab954d`, sha256 `3761d1af...efdf862` | PREREG l.550 |
| packet_L / packet_R sha256 | `140ff006...b53eef9e9` / `7a10ce63...5f60798` | l.551-552 |
| canary digest, three key digests | `eebb3bbe...497757`, `0c785d0b...d998d30a`, `edb69c09...89fb02`, `9be616f8...0918b` | l.553-556 |
| verdict_input.py sha256 | `750515ff6ad3f08942d86a94a65fecdc5be208f435c8ef9af397749e8f0c6850` | l.594 |
| styxx/protocol.py blob | `3d4b3f979b8db1dc47dbbd25841f3c4507ce6664` | l.630 |
| styxx imported from | `C:\Users\heyzo\clawd\wt\leg2-run\styxx\` (the worktree) | -- |
| twelve seat files, verdict | `RECORDED` (all), `dry_run` false, `prereg` the file above, `packet_sha256` equal to the locked packet | score.py l.130-131, l.315-316 |
| worktree tracked files | clean; untracked only `seat_outputs/` and the two transport-check folders | -- |

---

## 2. step 2, the run

Shell: Git Bash, cwd `C:/Users/heyzo/clawd/wt/leg2-run`. Python 3.12.10 (`py -3.12`).

### 2a. score.py

```text
STYXX_SEALED_DIR=C:/Users/heyzo/clawd/styxx-sealed PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 py -3.12 -B papers/sworn/measurement/score.py --dir papers/sworn/measurement --sealed C:/Users/heyzo/clawd/styxx-sealed --out C:/Users/heyzo/clawd/wt/_next_level_2026_10_12/tonight/leg2/fold_a/measurement_result.json
```

exit 0; 2026-10-24T20:45:54Z to 20:46:18Z; stderr empty. stdout:

```text
families clearing: L 0, R 1; withheld: ['G_S1', 'G_S2', 'G_P', 'kappa', 'q3']
G_C: k=182 n=182 wilson95=[0.9793, 1.0]
written: C:\Users\heyzo\clawd\wt\_next_level_2026_10_12\tonight\leg2\fold_a\measurement_result.json
```

`--out`: score.py's default for a real fold is `<dir>/measurement_result.json` (l.555-556), which would write
into the worktree; the CLI takes `--out` (l.570) with no restriction outside a dry run (only l.558-559 restricts
a dry run), so the file was written under `fold_a/`. Its bytes do not depend on where it is written: the
`lock` block holds repository-relative input paths and HEAD (score.py l.534-538, common.py l.547-568), and the
re-run below wrote to `fold_a/rerun/` and produced the same bytes.

### 2b. verdict_input.py

```text
PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 py -3.12 -B papers/sworn/measurement/verdict_input.py --tranche A --result C:/Users/heyzo/clawd/wt/_next_level_2026_10_12/tonight/leg2/fold_a/measurement_result.json --out C:/Users/heyzo/clawd/wt/_next_level_2026_10_12/tonight/leg2/fold_a/verdict_A.json
```

exit 0; stderr empty. stdout is the record (`verdict_A.json`), the same bytes once Windows text-mode stdout's
CRLF is read as LF (`tr -d '\r' < verdict_run1.stdout | cmp - verdict_A.json` passes; the written files carry 0 CR bytes).

verdict: **`VOID_L_R__G_S1_G_S2_G_P_withheld__MISSED__G_G1`** (PREREG l.724).
`score_py_quotable`: false. score.py's whole-fold `G_F.pass`: false, title "one-family: counts only, no precision"
(printed beside, gates nothing, PREREG l.185).

### 2c. every gate

| gate | metric value | bar | pass | notes |
|---|---|---|---|---|
| G_D_claude_L_overall | 26 of 30 | >= 27 | false | per side: LOAD-BEARING 15 of 15, NOT 11 of 15 |
| G_D_claude_L_side | 11 | >= 9 | true | |
| G_D_claude_R_overall | 29 of 30 | >= 27 | true | per side: YES 14 of 15, NO 15 of 15 |
| G_D_claude_R_side | 14 | >= 9 | true | |
| G_D_local_L_overall | 14 of 30 | >= 27 | false | per side: LOAD-BEARING 7 of 15, NOT 7 of 15 |
| G_D_local_L_side | 7 | >= 9 | false | |
| G_D_local_R_overall | 15 of 30 | >= 27 | false | per side: YES 15 of 15, NO 0 of 15 |
| G_D_local_R_side | 0 | >= 9 | false | |
| G_F_L | 0 families clearing | >= 2 | false | Panel L void: neither family clears its decoys. Title: "void panel; the void is the result" |
| G_F_R | 1 family clearing (claude) | >= 2 | false | Panel R void: one family. Title: "one-family: counts only, no precision" |
| G_S1 | -1.0 (withheld) | >= 0.7 | false (withheld, not read by the VOID row) | score.py: 0 of 0, labels "none", share WITHHELD |
| G_S2 | 2.0 (withheld) | <= 0.25 | false (withheld, not read) | score.py: 0 of 0, labels "none", share WITHHELD |
| G_P | 2.0 (withheld) | <= 0.1 | false (withheld, not read) | one-family counts, no rate: 1 NO of 135 YES+NO (claude only) |
| G_C | 0.9793286325558971 (Wilson lower, unrounded) | >= 0.95 | **true** | 182 of 182 FAILED; wilson95 [0.9793, 1.0]; per twin D01 9/9, D02 25/25, D03 21/21, D04 16/16, D05 22/22, D06 15/15, D07 12/12, D08 41/41, D09 21/21; 0 MALFORMED, 0 UNRESOLVED, 0 HELD |
| G_G1 | -1.0 (empty denominator) | >= 0.8 | **false: MISSED** | 0 pairs; 9 of 9 twins `twin_text_changed`; title "the floor cannot price gaming; the floor leaves the headline"; prints as 0 of 0, "the bar was not tested" (PREREG l.615-617). Cause in section 4 below |

`withheld` (verdict_input): G_S1, G_S2, G_P. `empty_denominators`: G_G1. score.py's `withheld`: G_S1, G_S2,
G_P, kappa, q3.

crosscheck (verdict_input.py l.142-173): G_F_L agrees, G_F_R agrees, G_C agrees, G_S1/G_S2/G_P "score.py
undecided: withheld", G_G1 "score.py undecided: empty denominator". No refusal.

Reported, no bar: G-U 0 (labels "none": Panel L has no labels); G-R 1 (one-family, claude: HELD by the
committed receipt and NO by the panel). Panel R one-family finals: YES 134, NO 1, UNSURE 1. Kappa: WITHHELD,
"fewer than two families clear Panel L". Q3: panel_coverage WITHHELD for all nine documents.
Population: 9 documents, 136 sworn spans, 819 units, 51 fragments. Panel L cells (score.py `cells`):
unlocated brackets claude 17 / local 209; located claude 1120 / local 507 (second pass: claude 179 / local 45);
unparsed item-seat rows claude 3 / local 44; Panel R unparsed claude 0 / local 0.

### 2d. the families block (Amendment 6 item 2)

score.py l.518-521:

```python
    substrates = {}
    for f in families:
        sf = seat_file(seat_dir, f, "L", 1) or seat_file(seat_dir, f, "R", 1)
        substrates[f] = (sf or {}).get("substrate")
```

Both families' `L-seat1.json` exist, so the block carries **`L-seat1`'s substrate for each family**:

- claude (from `seat_outputs/claude/L-seat1.json`): model `claude-opus-5-5`, transport `claude-cli clean-config`,
  `named_in_design` true, block_order TASK, FORMAT, CAUTIONS.
- local (from `seat_outputs/local/L-seat1.json`): model `Qwen/Qwen2.5-3B-Instruct`, transport `transformers`,
  dtype bf16, device cuda, `named_in_design` **false**, block_order TASK, FORMAT, CAUTIONS, threads 22, load_seconds 6.7.

Read from the other files themselves (Amendment 6 "What the RESULT prints"): all six Claude seat files
(L-seat1..3, R-seat1..3) carry `substrate.model` `claude-opus-5-5`; no Claude seat file names another alias.
All six local seat files carry `Qwen/Qwen2.5-3B-Instruct`, bf16, cuda, `named_in_design` false, block orders
seat 1 TASK/FORMAT/CAUTIONS, seat 2 FORMAT/CAUTIONS/TASK, seat 3 CAUTIONS/TASK/FORMAT. Every Claude seat file
records block_order TASK/FORMAT/CAUTIONS because `seat_claude.py` builds its system prompt with
`rotate=False` (l.164, l.205) and records `C.block_order(1)` (l.208): the Claude seats are three fresh sessions
under one block order, as disclosure 6 says.

### 2e. how L-0033's errors and the local family's unparsed items entered the counts

score.py does not read a seat file's `errors` list at all (`grep errors score.py`: no match). It reads each
item row's `parsed` flag. Each of the three Claude L rows for L-0033 is `{"id", "parsed": false, "raw_sha256",
"text_head": ""}` (seat_claude.py l.240-244); the error string sits in the header's `errors` list (l.245-246).
A seat with errors is still present: score.py l.130-131 needs only `verdict` in (`RECORDED`, `DRY-RUN`) for all
three seats, and all six per family read `RECORDED`.

score.py l.155-160 (Panel L):

```python
            for s, sf in enumerate(seats[f], 1):
                row = next((r for r in sf.get("items", []) if r.get("id") == iid), None)
                if row is None or not row.get("parsed"):
                    counts[f]["unparsed_items"] += 1
                    per_seat.append(["NO-LABEL"] * len(unit_ranges[iid]))
                    continue
```

then l.165-166 takes `C.family_label` over the three seats' labels per unit, and common.py l.507-513 keeps only
votes in LABELS_L and needs two agreeing: `return top if n >= 2 else "NO-MAJORITY"`, `"NO-MAJORITY"` when no vote
is a label. So:

- **L-0033, Claude family:** three seats unparsed, 3 added to claude's `unparsed_items`, every unit of L-0033
  labelled NO-LABEL in every seat, family label NO-MAJORITY. A decoy counts correct only when
  `family_labels[f][iid][0] == side` (l.180), which NO-MAJORITY never equals. Whether L-0033 is a decoy or a
  document window is in the sealed key and was not read or derived here. It cannot move the verdict string:
  G_F_L needs both families to clear Panel L, and the local family is 14 of 30 there, so Panel L voids
  whatever L-0033's label is (and Panel L's labels feed no share once it voids, score.py l.197-203).
- **Local family, Panel L:** 14 + 14 + 16 = 44 unparsed item-seat rows, each by the same lines: that seat's
  votes for the item's units are NO-LABEL; a unit gets a local family label only where two of the three seats
  parsed and agreed. Panel R: score.py l.324-331, the same rule (`unparsed[f] += 1`, vote `NO-LABEL`,
  `C.family_label(votes, C.LABELS_R)`); local R unparsed 0.
- Neither enters a share: with 0 families clearing L and 1 clearing R, G-S1, G-S2 and G-P are withheld.
  They enter G-D only, through the family label of each decoy item.

---

## 3. determinism: both commands re-run once

Re-run 2026-10-24T20:47:45Z to 20:48:12Z, the same commands with `--out` under `fold_a/rerun/`
(`measurement_result.json`, `verdict_A.json`; verdict_input's `--result` pointing at the re-run's result). Exit 0, 0, stderr empty.

| file | sha256 | re-run |
|---|---|---|
| `measurement_result.json` | `80918c789abbee29852cbf99dba3d13d90bdde3ca6dadb1d31f66f5a15f49ed4` | identical (`cmp` passes) |
| `verdict_A.json` | `45b1e570ace3044fda165fc5c64fe904830951356b075d610c031a7cc5cf964f` | identical |
| verdict stdout | `f45969b9cd8aea9a9ea650d315a4330a37845d563ba663841376e5c7317915e8` | identical |
| score.py stdout | -- | identical except the `written:` path line |

Worktree untouched: `git status --porcelain --ignored --untracked-files=all` before and after are byte-identical
(82 lines, all under `seat_outputs/` and the two transport-check folders; `git_status_before.txt`,
`git_status_after.txt`); the 32 seat-output files have the same sha256 before and after
(`seat_outputs_sha256_before.txt`, `..._after.txt`); no `__pycache__`, no `measurement_result.json` in the tree; HEAD unchanged.

---

## 4. finding: G-G1's miss comes from the committed twin path, not from the twins' tags

All nine twins are recorded `twin_text_changed: true` by `twin_trivial.py` at write time (status files D01-D09;
`error` null, `refused` null), and score.py excludes a changed twin (l.446-449), so `pairs` is 0, the share is
`None` (l.461-462) and the projection writes -1 with G_G1 in `empty_denominators` (verdict_input.py l.132-136):
a fail-closed miss under PREREG l.614-617.

Measured (`diag/twin_text_diff.py`, read-only: population.json, the committed sidecars at the pinned commit,
the twin files; no sealed file, no model; output `diag/twin_text_diff.txt`): **in all nine, the twin's
canonical text equals the original's except that the original's final `\n` is missing** (one `delete`
opcode, one byte, at the original's last offset; e.g. D01 1822 vs 1821 chars, D08 9827 vs 9826).

Cause in the locked code: `seat_claude.py` l.83, `cli()` returns `"text": (j.get("result") or "").strip()`,
which removes any trailing newline from the returned document; `twin_trivial.py` l.77 calls `SC.cli` and l.83
canonicalises that stripped text with `sworn.to_sidecar`, and l.94 compares it with the sidecar text, which
ends in `\n` in all nine. Whatever the model returned, a document that ends in a newline cannot come back equal
through this path. The dry run could not show it: its twins are `synthetic.canned_trivial_twin` (synthetic.py
l.244-252), a copy of the sidecar that never passes through `cli`. The status files record spans original vs
twin: D01 8/1, D02 26/2, D03 12/0, D04 12/1, D05 14/1, D06 11/1, D07 11/1, D08 27/1, D09 15/1. No counterfactual
share was computed.

---

## 5. what the RESULT owes beside these numbers (from PREREG s.7 and Amendment 6), for whoever writes it

- `claude-opus-5-5 (Amendment 6)` beside every Claude number (G-D counts L 26/30, R 29/30); local family
  `named_in_design: false` beside every local number; Amendment 6 item 5's reading beside claude's `named_in_design: true`.
- Panel L void under G-D's title, both families' counts with no rate; Panel R one-family, counts only (G-P 1 of
  135 one-family, G-R 1); G-S1, G-S2, G-P withheld; G-C 182 of 182, lower bound 0.9793; G-G1 0 of 0, "the bar
  was not tested", with section 4's cause.
- L-0033's three transport errors and their cause (`FINDING_L0033_transport_2026_10_24.md`, one level up).
- That tranche A's keys stay sealed until tranche B's seat outputs are recorded or tranche B closes with no
  fold, so no outside reader can re-score tranche A yet (PREREG l.745-747); the six disclosures (in the result's `disclosure`).

## 6. notes for the operator (decisions not taken here)

- L-0033 and the twin newline are both defects in the locked transport path (`seat_claude.py` l.69 passes the
  item as `-p <text>`; l.83 strips). PREREG s.8 l.765-766: "an amendment after any seat has read a real item is
  a new preregistration under a new name". Tranche A's seats have read real items, so a route for tranche B's
  leading-dash items and twins looks like it falls under that sentence, not under a numbered amendment as the
  L-0033 finding proposes. Flobi decides.
- Nothing was committed. `measurement_result.json` and `verdict_A.json` live only under this folder.

## files

all under `C:/Users/heyzo/clawd/wt/_next_level_2026_10_12/tonight/leg2/fold_a/`:
`measurement_result.json`, `verdict_A.json`, `score_run1.stdout|stderr`, `verdict_run1.stdout|stderr`,
`rerun/` (the second run's four outputs and logs), `git_status_before.txt`, `git_status_after.txt`,
`seat_outputs_sha256_before.txt`, `seat_outputs_sha256_after.txt`, `diag/twin_text_diff.py`, `diag/twin_text_diff.txt`, this file.
No `DECISION.md`: step 1 did not stop.
