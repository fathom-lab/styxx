# artifact_2026_11_10: the two sealed tests behind the 2026-11-10 erratum

These are the files of `../ERRATUM_real_convergence_2026_11_10.md`. One lab, one machine, one hand wrote the
claims, the code and the scoring. Outside reruns: 0.

The scripts take `papers/real-convergence/` as their kit folder: they import `reproduce.py` and read
`contextual_reps.npz`, `lex_tokens.json` and `real_convergence_confirm_result.json` from it. The commands are in the
erratum, under "re-running it".

## files

| file | what it is |
|---|---|
| `PREREG_start_token_artifact_2026_11_10.md` | test 1, written and sealed before its run: is the dominant direction of the June arrays a start-of-input artifact? Three cached models, nine predictions. |
| `extract_small.py` | test 1 extractor, sealed with it: per-template last-token states at the June layer, CPU, offline, one model per process. Writes the `states__*.npz` arrays. |
| `score_artifact.py` | test 1 scorer, sealed with it. Needs numpy. Reads the states and the kit folder. |
| `score_out.json`, `score.log` | test 1 scorer output, from its one sealed run. |
| `RESULT_start_token_artifact_2026_11_10.md` | test 1 result: every sealed line passed. |
| `PREREG_corrected_geometry_five_models_2026_11_10.md` | test 2, written and sealed before its run: cross-family agreement with the bare template dropped, five of the six June models. |
| `extract_v2.py` | test 2 extractor, sealed with it: runs the larger models one decoder layer at a time, CPU, float32, offline. Writes the `states2__*.npz` arrays and one `extract2_log__*.json` per run. |
| `score_v2.py` | test 2 scorer, sealed with it. Needs numpy. Scores the controls, and scores the predictions when every control has passed. Reads test 1's states as a control. |
| `AMENDMENT_1_five_models_2026_11_10.md` | sealed after two extractions of the sealed run failed (one refused allocation, one segmentation fault, while the machine's commit charge was near its limit) and before anything else was done: allows a rerun of those two with the script unchanged. Lists the sha256 values of the files written by then. |
| `score_v2_stopped_1.json`, `score_v2_stopped_1.log` | test 2 scorer output from the sealed run, which stopped at its controls because two models had no states. It wrote its control lines and no prediction line. |
| `extract2_log__*.json` (7 files) | one log per extraction run of test 2. `extract2_log__Qwen2.5-3B__run.json` records the crash in the sealed run and `extract2_log__Qwen2.5-3B__run__2.json` the rerun that succeeded; the rerun between them ended in a segmentation fault before writing a log. Each successful log carries the sha256 of the arrays it wrote. |
| `score_v2_out.json`, `score_v2.log` | test 2 scorer output after the amended rerun: every control and every prediction passed. |
| `RESULT_corrected_geometry_five_models_2026_11_10.md` | test 2 result. |
| `explore_pc1.py` | the exploratory look on the stored June arrays (original words, already seen) that test 1 was written from. Not a test. Its output file is not published. |
| `lead_checks.py` | the lead's own re-derivation, from the committed June files, of the main claims of the replication kit's skeptic. Raw RSA, no lexical partial. Its output file is not published. |
| `rerun_ungated.py` | written after both tests, not sealed and not part of them: one file that repeats the artifact bars on models that need no licence, with no lab code. Run by the lab on Windows only. |
| `rerun_ungated.ipynb` | written after both tests, not sealed and not part of them: a browser notebook that downloads `rerun_ungated.py` at one pinned commit, prints its sha256, and runs it on a processor. Open it in Google Colab from `https://colab.research.google.com/github/fathom-lab/styxx/blob/main/papers/real-convergence/artifact_2026_11_10/rerun_ungated.ipynb`. The lab has not run this notebook on Colab. |
| `mind_axis_check.py`, `mind_axis_check_out.json` | the lead's check of the package's geometry axis, after both tests and not sealed: it runs `styxx.mind.geometry_citizenship`, unchanged, on the stored arrays, with and without the bare template. It needs the state arrays, which are not in this repository. The six anchor shares it prints need only `../contextual_reps.npz`. |
| `STATES_SHA256SUMS.txt` | sha256 of the 6 per-template state arrays of test 1. The arrays are not in this repository. |
| `STATES2_SHA256SUMS.txt` | sha256 of the 12 per-template state arrays of test 2. The arrays are not in this repository. |

## commits the documents cite

| commit | repository | what the documents say it holds |
|---|---|---|
| `a59db86e` | this repository | the June 2026 result; every file of it entered in this one commit |
| `62d7ae3` | the lab's private repository | test 1 sealed: its PREREG, `extract_small.py`, `score_artifact.py` (pushed 2026-11-10T14:22Z) |
| `f1e8c26` | the lab's private repository | test 1 RESULT as committed |
| `fb29ef7` | the lab's private repository | test 1 RESULT, corrected |
| `bbbd61f` | the lab's private repository | test 2 sealed: its PREREG, `extract_v2.py`, `score_v2.py` (15:56:04 UTC) |
| `04a6136` | the lab's private repository | AMENDMENT 1 sealed (16:06:27 UTC) |
| `756ae73` | the lab's private repository | test 2 RESULT as committed |
| `e97ab11` | the lab's private repository | test 2 RESULT, corrected: the largest difference in its June re-derivation line is 0.0004, where it first said 0.0003 |

The private repository is not public, so an outsider cannot check these commits or their dates. What can be checked
is that the files here hash to the values the documents cite, and that the scripts re-run.

## hashes the documents cite, against the files here

- `PREREG_corrected_geometry_five_models_2026_11_10.md`, `extract_v2.py`, `score_v2.py`, five of the
  `extract2_log__*.json` files and `score_v2_stopped_1.json` match the values listed in AMENDMENT 1. So do the 8
  `states2__*.npz` values it lists, in `STATES2_SHA256SUMS.txt`.
- `score_out.json` matches the value test 1's RESULT cites (`d20b5d1e...1f31e`).
- `../reproduce.py` matches the `kit_reproduce_sha256` that every `extract2_log__*.json` records. The sha256 of
  `../lex_tokens.json` is in the lab's private seal files and in no published document.
- Line endings are part of these bytes: the JSON files and the logs are CRLF, the documents and scripts LF. The
  repository's `.gitattributes` marks this folder, `lex_tokens.json` and `reproduce.py` `-text`, so git stores and
  checks them out unchanged on every platform.
- Three logs differ from the lab copies, by one line each. In each, the line `wrote <absolute path>` printed the lab
  machine's full output path. That path is shortened here to the file name. Nothing else in them was changed.

| log | sha256 cited (lab copy, CRLF line endings) | sha256 of the file here (CRLF, one line shortened) |
|---|---|---|
| `score.log` | `e974e338d2462ceb0a3cf0861cb12a3d9e15d62f9b36a333e897e552b395e002` (test 1 RESULT) | `81f283e6f1a70b017355a37da6d014f115e99b49eeac2f15ca1bb91fa1157e69` |
| `score_v2_stopped_1.log` | `cb28e606889950396886ec3d121aeae4cd02474795d9c106af49e73a44e52bda` (AMENDMENT 1) | `ab4db89d0fb6197c8000624de77be1c77a26bd2ac1a39dda2cd500f85c578c23` |
| `score_v2.log` | `ecd2a5dfe107f10afbd5f471797b1df605c71e635a52596b9902887669d69ecf` (not cited by a document) | `4283f57eaadf7462ba99c8f7ece79a76fa30ff88e6e5dee6c9dbed4eefb9beee` |

Running the two scorers on the lab's state arrays prints the same PASS and FAIL lines as these logs (test 1: 17
PASS, 0 FAIL; test 2: 91 PASS, 0 FAIL) and writes the same JSON as `score_out.json` and `score_v2_out.json`.

## the two June files

`../RESULT_real_convergence_2026_06_03.md` and `../RESULT_scale_law_2026_06_03.md` each carry one dated line under
the title, written on what was a blank line. It adds no line and no number, so the digest of each file's numbers in
`papers/closed-model-frontier/oath_v11_baseline_ledger.json` still reproduces. Nothing else in them was changed.

## words inside the copies

The copied scripts are kept byte for byte, as they ran, and several are named by sha256 in the documents and logs.
So their own comments are as written before the correction. `../reproduce.py` still calls the June confirm run
"PRE-REGISTERED out-of-sample CONFIRM (headline 0.258)"; the erratum withdraws those words. `lead_checks.py` names
`REPLICATE.md` and `skeptic_checks.py`, files of the lab's private replication kit that are not in this repository.
