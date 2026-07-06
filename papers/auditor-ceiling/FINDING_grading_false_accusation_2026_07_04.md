# FINDING v2 — the auditor's ceiling: TriviaQA grading false-accuses at ~13%; PopQA's gap is ambiguity

**Fathom Lab · 2026-07-04/05 · prereg `3d7e0a7` (frozen before any judging). v2 SUPERSEDES v1 of this file.
Status: PENDING KG-HUMAN — flobi's seal of the 18 confirmed rows (`KG_HUMAN_seal_table.md`) gates every
judged claim; the TruthfulQA count needs no judge.**

## v1 → v2 correction (read first — this is the load-bearing disclosure)

v1's confirmation stage was **void: the operator hand-typed the second-judge inputs and paraphrased all 18
TriviaQA questions from memory** — some inverted the question's meaning, one added answer options that don't
exist in the dataset. The corruption was caught by a row-level mechanism autopsy BEFORE the human seal
(prompt-hash arbitration cleared the data pipeline; the fabrication was isolated to the stage-2 prompt text).
Stage 2 was re-run clean: judges read verbatim rows from the committed file and echo each question back
(`q_echo`, integrity-checked), under a tightened rule — judge the question AS WRITTEN, non-responsive answers
are wrong, under-determined referents are UNSURE. Consequences of the correction:

- PopQA's v1 "11.2% false accusations" **collapsed to 3.1%** — 8 of 11 rows dissolved into referent ambiguity;
- two loose stage-1 calls were rejected ("Star Trek" is not a *who*; the femur is not the *hardest* bone);
- **the v1 "pre-registered surprise" (label-corrected H1 flip) is RETRACTED**: recomputed with clean
  confirmations, AUROC 0.5687 CI [0.4951, 0.6415] — does NOT exclude chance. The "surprise" was manufactured
  by the corrupted confirmations. The keystone null stands exactly as frozen, and this prereg's original
  prediction (robustness check stays null) was CORRECT after all.

## Headline (clean, double-blind-confirmed, verbatim inputs)

| | result |
|---|---|
| **TriviaQA false accusations** | **13/103 = 12.6%**, Wilson95 [7.5, 20.4] — mech-failed answers that are correct and responsive: *Apollo 11, Austerlitz, tumbrils, Civil War, 1968 (Fosbury), Augusta National, Pan, zinc, Bran Castle→Dracula, Turnbull & Asser, comedy (Lenny Bruce), a collection of cars, New York City (Nighthawks)* — alias-format and gold-list gaps on quiz-plain answers |
| **TriviaQA false credit** | 2/146 = 1.4% — mech credited "beehive" (answer: *skep*) and "cursor" (the patent is the *mouse*): gold lists containing wrong aliases |
| **PopQA false accusations** | **3/98 = 3.1%** [1.0, 8.6] — the honest number after ambiguity is separated |
| **PopQA referent ambiguity** | **8 rows**: "Symphony No. 3", "Monster", "Museum of Islamic Art", "Evil", "Wildlife", "Hotel", "Joseph Schubert", "Alexander" — KB-derived questions whose TEXT under-determines the entity the gold binds to. A blind judge *cannot* grade these; neither can a model *answer* them except by luck. **This is PopQA's deeper defect: it tests referent-guessing, not knowledge, on a measurable slice of items** |
| **TruthfulQA** | 242/250 = 96.8% mechanically ungradeable by its own matching path (no judge needed) |
| **accuracy corrections** | TriviaQA 58.6% → **63.1%** (+4.5 pts) · PopQA 12.8% → **15.0%** (+2.2 pts) |

## What this licenses
1. **Mechanical grading false-accuses at ~1-in-8 on TriviaQA** (CI excludes zero) for this frozen §3 pipeline
   on this model's outputs — mechanically-scored short-form QA numbers are deflated lower bounds with a
   phrasing-dependent bias term larger than most claimed model-vs-model gaps.
2. **Gold lists carry both error directions**: missing correct aliases (FN) and wrong credited aliases (FP —
   "cursor" for the mouse patent).
3. **KB-derived QA has a distinct, previously-unmeasured defect class — referent under-determination** — that
   label-correction cannot fix and blind judging can only quarantine.
4. **No depth rescue**: the keystone verdict is untouched; the H1 robustness check returned to its predicted null.

## Protocol integrity, in full
Stage-1: 382 rows, judges blind to grades/gold, real questions (prompt-hash-verified against what the model
saw). Stage-2 v1: VOIDED (fabricated inputs — operator error, disclosed above; the week's fifth and most
serious self-catch). Stage-2 v2: file-read verbatim rows + q_echo integrity + responsiveness rule; concurrence
required. The two-judge rule rejected judge-1 overreach in both directions. Receipts: `blind_rows.jsonl`,
`mech_key.json`, `blind_judging_results.json`, `disagreements_final.json`, `fn_mechanism_autopsy.json`,
`final_rates.json`, `stage2_rows.jsonl`. Workflow-args gotcha (args arrive as a string) documented in the
memory index; local recomputation from committed raw verdicts is the path of record.

*The ground truth was the last unaudited auditor — and auditing it required auditing ourselves twice more.*
