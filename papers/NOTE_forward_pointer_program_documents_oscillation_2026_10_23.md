# NOTE — the forward pointer from the program audit and the program synthesis to the permuted-MNIST robustness RESULT

**status: a forward pointer, sworn to committed bytes. It stands beside `AUDIT_the_whole_program_2026_09_01.md` and
`SYNTHESIS_program_2026_07_24.md`, in this directory, and edits neither: both keep their bytes.**

2026-10-23. Two program documents state the frequency arc's permuted-MNIST headline as it stood before its sealed
re-run. The three sentences, quoted from the documents' committed bytes:

- **The audit, finding 5 of its section 3**, headed
  <sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md#L219" k="quote">`3. WHAT IS SETTLED, WITH RECEIPTS`</sworn>
  (lines 240 to 242):
  <sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md#L240" k="quote">`5. **Oscillation is causally load-bearing in state-space models.** +0.3122 permuted-MNIST accuracy`</sworn>
  …
  <sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md#L241" k="quote">`from a single phase clamp`</sworn>
  …
  <sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md#L242" k="quote">`positive control and a family-wise null.`</sworn>
- **The audit, rank 2 of the table in its section 7**, headed
  <sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md#L438" k="quote">`7. BUILT AND NEVER USED`</sworn>
  (line 446), the reason it gives for `papers/frequency-resonance/resonance_profiler.py`:
  <sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md#L446" k="quote">`the phase-clamp result behind it is the program's cleanest causal ablation (+0.3122)`</sworn>
- **The synthesis, Arc B** (lines 49 and 50):
  <sworn r="path:papers/SYNTHESIS_program_2026_07_24.md#L49" k="quote">`- **Oscillation is the long-range mechanism** in state-space models (phase-clamp ablation; permuted-MNIST`</sworn>
  <sworn r="path:papers/SYNTHESIS_program_2026_07_24.md#L50" k="quote">`decay gap +31.2 points).`</sworn>

The re-run is `papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md`, under the sealed
`PREREG_pmnist_robustness_2026_10_11.md`. This NOTE points each of the three sentences to it, says what carries and
what does not, and points to the documents that already file the same sentences.

## What the re-run reads

The re-run's receipt reads
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/verdict" k="quote">`BUDGET_CONFOUND__gap_does_not_survive_3x_steps`</sworn>.
The sentence that verdict licenses, copied with its spans from the RESULT's section "The sentence this verdict
licenses":

**On permutation
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/anchor_perm" k="numeric">1234</sworn>,
at
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_long" k="numeric">12,000</sworn>
steps under a cosine schedule re-stretched to that budget, the gap is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_long" k="numeric">0.1097</sworn>
(two FREE seeds against three CLAMPED seeds); at
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_main" k="numeric">4,000</sworn>
steps on three new seeds it is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/anchor_gap_newseed" k="numeric">0.3182</sworn>.
The
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_main" k="numeric">4000</sworn>-step
gap does not survive the longer budget on this permutation.**

Nothing stronger is licensed. Section 11 of the robustness PREREG refuses the general sentence about the +0.3122 and
the training budget: one permutation and five runs do not license it (M8).

## What carries and what does not

**The number.** All three sentences rest on one measurement, the 2026-07-23 receipt's FREE minus CLAMPED,
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/result/free_minus_clamped" k="numeric">+0.3122</sworn>:
FREE
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/result/free_acc" k="numeric">0.9195</sworn>
against CLAMPED
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/result/clamped_acc" k="numeric">0.6073</sworn>,
at
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/steps" k="numeric">4000</sworn>
steps on seeds
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/seeds/0" k="numeric">0</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/seeds/1" k="numeric">1</sworn>
and permutation
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/perm_seed" k="numeric">1234</sworn>,
in a
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/blocks" k="numeric">3</sworn>-block
complex-diagonal linear recurrence (LRU-style; `CLRU` in `run_pmnist_ablation.py`). The synthesis's "+31.2 points" is
the same value in accuracy points. The number stands as what was measured in that setting. As the arc's headline it
does not carry at the re-run's verdict, and the CHANGELOG banner of 2026-10-23 over the 2026-09-02 entry says the
same of it: the +0.3122 headline
<sworn r="path:CHANGELOG.md#L4543" k="quote">`does not carry at this verdict`</sworn>.

**Finding 5's sentence.** "Oscillation is causally load-bearing in state-space models" is the general sentence the
prior-art reads file. `papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md` quotes it as INDEX
row 51 carried it,
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L464" k="quote">`The single-knob phase clamp shows oscillation is causally load-bearing in SSMs (+0.3122 on permuted MNIST)`</sworn>,
and files it:
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L468" k="quote">`"in SSMs" exceeds the measurement: one LRU-style architecture. No read paper supports the general sentence`</sworn>.
Its CHANGELOG entry reads
<sworn r="path:CHANGELOG.md#L199" k="quote">`INDEX row 51's "in SSMs" exceeds every read`</sworn>.
The audit's line 240 states that sentence of state-space models as a class, and the reads give it no more support
there than in INDEX row 51. The finding's positive control and its family-wise null are not re-examined here.

**The table row's reason.** The +0.3122 is a gap between two separately trained arms; in CLAMPED,
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L384" k="quote">`the phase θ was held at 0 for the whole of training`</sworn>.
The profiler the row ranks makes another measurement: it clamps the phase of one trained model in place,
<sworn r="path:styxx/resonance.py#L10" k="quote">`the profiler clamps the oscillatory machinery off in place`</sworn>.
The robustness RESULT runs that kind of clamp on its own FREE models and records it
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L235" k="quote">`because it is free and is not evidence for the between-model gap: a destructive perturbation with`</sworn>
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L236" k="quote">`no matched-norm control`</sworn>.
So the row's reason rests on a number that does not carry at the verdict, and that number comes from a measurement
other than the one the profiler makes. The row's missing piece has been met: the profiler ships as
`styxx/resonance.py`,
<sworn r="path:styxx/resonance.py#L2" k="quote">`styxx.resonance — the resonance profiler`</sworn>,
promoted from the file the row names,
<sworn r="path:styxx/resonance.py#L5" k="quote">`which the 2026-09-01 program audit listed`</sworn>
as built and never shipped. This NOTE does not re-rank the table.

**The synthesis's sentence.** "Oscillation is the long-range mechanism in state-space models" is a general sentence
of the same kind. The prior-art NOTE files the 2026-07-23 RESULT's version of it under
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L406" k="quote">`**Exceeds every read, needs the same banner:**`</sworn>,
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L407" k="quote">`l.39, "the oscillation *is* the long-range mechanism"`</sworn>,
and the PAPER's banner files the PAPER's own,
<sworn r="path:papers/frequency-resonance/PAPER_oscillation_causal_map_2026_07_23.md#L50" k="quote">`**Exceeds every read:** "the oscillation *is* the long-range mechanism"`</sworn>,
and its "+31.2 points" headline,
<sworn r="path:papers/frequency-resonance/PAPER_oscillation_causal_map_2026_07_23.md#L53" k="quote">`**Does not carry at the RESULT's verdict:** the permuted-MNIST headline of the Abstract and §4.1 (+31.2 points,`</sworn>.
The synthesis's lines 49 and 50 state that sentence of state-space models as a class, with the same number, and
nothing in the reads or the RESULT supports them further than it supports the PAPER's.

## Where the same sentences are already filed

- `papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md`: the re-run, its verdict and the sentence it
  licenses.
- `papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md`: the full-text reads of the LRU paper,
  S4D, LinOSS and D-LinOSS, and its section 5, which files the lab's sentences.
- `papers/frequency-resonance/PAPER_oscillation_causal_map_2026_07_23.md`: the banner above its unchanged text, which
  opens
  <sworn r="path:papers/frequency-resonance/PAPER_oscillation_causal_map_2026_07_23.md#L3" k="quote">`**Banner, added 2026-10-23, from`</sworn>
  and files the draft's versions of both general sentences.
- `papers/INDEX.md`, which the audit names as its companion: its frequency-resonance row reads the re-run's verdict
  since 2026-10-23,
  <sworn r="path:papers/INDEX.md#L51" k="quote">`was re-run under a sealed preregistration and reads`</sworn>,
  and so does its +0.3122 receipt row.
- `papers/rhythm-rescue/NOTE_forward_pointer_untied_magnitudes_2026_10_23.md`: the same pointer beside the
  2026-09-02 untied-magnitudes RESULT, whose section "The finding" states finding 5's sentence too.

## Why beside them and not inside them

The lead's decision for this sweep, as for the two pointer NOTEs of 2026-10-23 beside the certified 2026-07-23 RESULT
and the sealed untied PREREG: both documents keep their bytes, and the pointer lands beside them as a NOTE. What names
each, one file to a sentence:

**The audit.** On origin/main `333f3c73` no certificate, seal, sworn span or receipt names its bytes: a search there
for its sha256 and for its git blob id (`9a5f0b82…`) finds neither. This NOTE swears its bytes, for the record, at
the commit its sidecar names:
<sworn r="path:papers/AUDIT_the_whole_program_2026_09_01.md" k="hash">its blob hashes to `41c08977bfbf8f01842256c4375ced0dfbeca0088793eaf708919337ab1dc21d`</sworn>.
Sworn documents, a census receipt, a declared mapping and the leg-2 packets name it by name, most of them citing a
section of it or its mechanism M2:

- `papers/sworn/DESIGN_sworn_measurement_2026_09_01.md`, sworn, cites its §8:
  <sworn r="path:papers/sworn/DESIGN_sworn_measurement_2026_09_01.md#L9" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md`, sworn, cites its §8:
  <sworn r="path:papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md#L7" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01.md`, sworn, takes its rows from the audit's list:
  <sworn r="path:papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01.md#L6" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/sworn/prose_claimhood_census.json`, that census's receipt, takes §8.1 as written:
  <sworn r="path:papers/sworn/prose_claimhood_census.json#/criterion" k="quote">`in_audit rows are AUDIT_the_whole_program_2026_09_01.md §8.1 direction 1, as written`</sworn>.
- `papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01.md`, sworn, cites its M2:
  <sworn r="path:papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01.md#L13" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/closed-model-frontier/h_mapping.json`, the mapping that DECLARATION declares and `tests/test_h_mapping.py`
  checks, cites its M2 in `what_h_is`:
  <sworn r="path:papers/closed-model-frontier/h_mapping.json#/what_h_is" k="quote">`M2 in AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/SYNTHESIS_the_grain_of_the_handed_target_2026_09_02.md`, sworn, cites its M2:
  <sworn r="path:papers/SYNTHESIS_the_grain_of_the_handed_target_2026_09_02.md#L7" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/sworn/measurement/packet_L.json`, which the leg-2 PREREG's lock table pins by sha256,
  <sworn r="path:papers/sworn/PREREG_sworn_measurement_2026_10_23.md#L551" k="quote">`140ff006f92f3b64f88019b5c5f326052b92c3e4af36623cdc59c21b53eef9e9`</sworn>,
  carries two passages that name it:
  <sworn r="path:papers/sworn/measurement/packet_L.json#/items/24/text" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>
  and
  <sworn r="path:papers/sworn/measurement/packet_L.json#/items/47/text" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.
- `papers/sworn/measurement/decoys_L.json`, which the same lock table pins by blob,
  <sworn r="path:papers/sworn/PREREG_sworn_measurement_2026_10_23.md#L578" k="quote">`75a030fd8d18113a2e98c30cbb61d2b037207fbc`</sworn>,
  carries one:
  <sworn r="path:papers/sworn/measurement/decoys_L.json#/decoys/22/text" k="quote">`AUDIT_the_whole_program_2026_09_01.md`</sworn>.

**The synthesis.** Its bytes, for the record:
<sworn r="path:papers/SYNTHESIS_program_2026_07_24.md" k="hash">its blob hashes to `7780a96b3c09046c1bd5006dac9a546fcf7cec89181967918b49bcb6a95e172b`</sworn>.

- `papers/closed-model-frontier/oath_v11_baseline_ledger.json` records a digest over its extracted number tokens,
  <sworn r="path:papers/closed-model-frontier/oath_v11_baseline_ledger.json#/extraction/papers~1SYNTHESIS_program_2026_07_24.md/n" k="numeric">11</sworn>
  tokens,
  <sworn r="path:papers/closed-model-frontier/oath_v11_baseline_ledger.json#/extraction/papers~1SYNTHESIS_program_2026_07_24.md/sha256" k="quote">`9bcb573e7b875b639f568ac6128bef82d7d854602063c086381116a61774d9da`</sworn>;
  each token carries its line (`extraction_digests` in `make_oath_v11_baseline.py`), the digest reproduces on
  origin/main `333f3c73`, and the ledger is not regenerated, so a number added inside the synthesis, or a line added
  above its line 54, where its last token sits, would leave the ledger describing bytes the file no longer holds.
- `papers/frequency-resonance/RESULT_realmodel_bridge_v3_not_started_2026_10_12.md`, sworn, names its item 3 as the
  place a pointer of that preregistration lands:
  <sworn r="path:papers/frequency-resonance/RESULT_realmodel_bridge_v3_not_started_2026_10_12.md#L174" k="quote">`papers/SYNTHESIS_program_2026_07_24.md`</sworn>;
  that pointer has stood inside item 3 since 2026-10-12,
  <sworn r="path:papers/SYNTHESIS_program_2026_07_24.md#L99" k="quote">`Forward pointer (2026-10-12): attempt three: not started; two pilots, both NEITHER`</sworn>,
  and carries no number token the ledger reads.

Putting a pointer inside either document stays the operator's call.

## What this note does not do

- It does not put the pointer inside the audit or the synthesis. A reader who opens only one of them does not see
  it; that is the cost of this route.
- It does not edit `papers/INDEX.md`, whose two rows already read the verdict.
- It re-examines nothing else in either document: not finding 5's positive control or family-wise null, not the
  audit's other findings or other table rows, not the synthesis's other Arc B items.
- It reports no measurement of its own. No outcome it points to bears on deployed models, on attention, on
  importance or on priority (robustness PREREG section 11).

## Sworn

This note swears 57 spans, each to bytes committed at the commit its sidecar names: the audit's headings and
lines, the synthesis's lines, and both blobs; the 2026-07-23 receipt and the robustness receipt and RESULT; the
prior-art NOTE, the PAPER's banner, the INDEX row and the CHANGELOG; `styxx/resonance.py`; and the files that name
the two documents. The sidecar and the verdict receipt beside this file were written by
`python -m styxx.sworn canon` and `python -m styxx.sworn verify`; the verdicts are the receipt's.
