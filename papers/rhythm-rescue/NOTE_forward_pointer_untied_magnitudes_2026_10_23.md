# NOTE — the forward pointer from the 2026-09-02 untied-magnitudes RESULT to the permuted-MNIST robustness RESULT

**status: a forward pointer, sworn to committed bytes. It stands beside `RESULT_untied_magnitudes_2026_09_02.md`, in
this directory, and edits nothing: that RESULT, its sidecar and its verdict receipt keep their bytes.**

2026-10-23. `RESULT_untied_magnitudes_2026_09_02.md` closes its section "The finding" with a sentence about the
frequency arc's headline. Its lines 34 to 36, quoted line by line:

- <sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md#L34" k="quote">`The arc's headline, that oscillation is causally load-bearing in state-space models, survives`</sworn>
- <sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md#L35" k="quote">`the sharpest control its own knob admits, and the theta-gamma reading — phase as the code for`</sworn>
- <sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md#L36" k="quote">`order — is strengthened rather than scoped.`</sworn>

The arc's headline is the permuted-MNIST phase-clamp gap. The CHANGELOG entry for its re-run's preregistration names
it so,
<sworn r="path:CHANGELOG.md#L920" k="quote">`The arc's headline (+0.3122 on permuted MNIST,`</sworn>
and that preregistration, `PREREG_pmnist_robustness_2026_10_11.md`, states it on its line 35:
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L35" k="quote">`The headline: FREE 0.9195 (0.9240 / 0.9150) vs CLAMPED 0.6073 (0.5529 / 0.6617), gap +0.3122`</sworn>.
The re-run's RESULT, `papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md`, is what this NOTE points
to. The headline the sentence above calls surviving does not carry at its verdict.

## What the robustness RESULT reads

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
the training budget: one permutation and five runs do not license it (M8). The +0.3122 is the 2026-07-23 receipt's
FREE minus CLAMPED,
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/result/free_minus_clamped" k="numeric">+0.3122</sworn>,
at
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/steps" k="numeric">4000</sworn>
steps on seeds
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/seeds/0" k="numeric">0</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/seeds/1" k="numeric">1</sworn>
and permutation
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/perm_seed" k="numeric">1234</sworn>.
It stands as what was measured there; as the arc's headline it does not carry at this verdict. The CHANGELOG banner
of 2026-10-23 over the 2026-09-02 entry says the same of the +0.3122 headline: it
<sworn r="path:CHANGELOG.md#L4543" k="quote">`does not carry at this verdict`</sworn>.

## What of this RESULT's own measurement stands, as measured

It measured no permutation of MNIST. Its task is ordered copy, the rhythm-rescue task, and its own scope says so:
<sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md#L40" k="quote">`One task family (ordered copy), one width, three seeds, no interval, toy scale.`</sworn>
Its receipt, `untied_control_result.json`, records D
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/D" k="numeric">256</sworn>,
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/steps" k="numeric">4000</sworn>
steps, seeds
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/seeds/0" k="numeric">0</sworn>,
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/seeds/1" k="numeric">1</sworn>
and
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/seeds/2" k="numeric">2</sworn>,
the device
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/device" k="quote">`cpu`</sworn>,
and the verdict
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/verdict" k="quote">`ROTATION_LOAD_BEARING__beyond_diversity`</sworn>.
Ordered-memory capacity, in items: FREE
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/free/kcap_mean" k="numeric">6.0</sworn>,
CLAMPED
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/clamped/kcap_mean" k="numeric">2.6667</sworn>,
and REAL2, the untied real bank with
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/real2/params" k="numeric">168524</sworn>
parameters against FREE's
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/free/params" k="numeric">168524</sworn>,
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/real2/kcap_mean" k="numeric">2.6667</sworn>.
FREE minus CLAMPED is
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/gap_free_minus_clamped" k="numeric">3.3333</sworn>,
REAL2 minus CLAMPED
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/real2_minus_clamped" k="numeric">0.0</sworn>,
the recovery fraction
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/recovery_fraction" k="numeric">0.0</sworn>,
and the anchors landed within
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/anchor_max_abs_dev" k="numeric">0.0</sworn>
items of the committed GPU receipt. These stand as what was measured, in that setting. The robustness RESULT did not
re-run the ordered-copy task, and nothing in it or in this NOTE moves them.

They were measured at one budget. The run took
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/config/steps" k="numeric">4000</sworn>
steps, and so did the 2026-06-03 run whose receipt its anchors reproduce,
<sworn r="path:papers/rhythm-rescue/rhythm_rescue_result.json#/config/steps" k="numeric">4000</sworn>
steps in `rhythm_rescue_result.json`. That run's PREREG, `PREREG_rhythm_rescue_2026_06_03.md`, froze
<sworn r="path:papers/rhythm-rescue/PREREG_rhythm_rescue_2026_06_03.md#L67" k="quote">`Both arms trained to the same fixed budget`</sworn>
and wrote that
<sworn r="path:papers/rhythm-rescue/PREREG_rhythm_rescue_2026_06_03.md#L67" k="quote">`a CLAMPED failure`</sworn>
<sworn r="path:papers/rhythm-rescue/PREREG_rhythm_rescue_2026_06_03.md#L68" k="quote">`must not be an optimization artifact`</sworn>.
Neither receipt varies the budget, and neither tests whether the ordered-copy gap survives a longer one; this NOTE
does not test it either. The permuted-MNIST budget verdict is neither carried to ordered copy nor shown not to apply
there.

The RESULT said in its own scope that it did not test permuted MNIST (lines 41 and 42):
<sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md#L41" k="quote">`permuted-MNIST number (+0.312) was measured with the same knob and carries the same confound; this`</sworn>
<sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md#L42" k="quote">`result makes the diversity reading unlikely there but does not test it`</sworn>.

## What the robustness RESULT reports about the untied control on permuted MNIST

The robustness design ran REAL2 again, on permuted MNIST, as its untied control. On held-out permutation average
FREE − REAL2 is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/free_minus_real2_perm_mean" k="numeric">0.275</sworn>
(`G_R_diversity`) and REAL2 − CLAMPED is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/real2_minus_clamped_perm_mean" k="numeric">−0.0074</sworn>
(`G_R_rotation`), each against a bar of at most 0.03 in the sealed gates block:
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L250" k="quote">`"G_R_diversity": {"metric": "result.free_minus_real2_perm_mean", "op": "<=", "value": 0.03`</sworn>.
The receipt records `G_R_diversity` as not holding,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#L5498" k="quote">`"G_R_diversity": false`</sworn>,
and `G_R_rotation` as holding,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#L5499" k="quote">`"G_R_rotation": true`</sworn>.
The recovery fraction is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_perm_mean" k="numeric">−0.0279</sworn>,
interval
[<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_perm_ci/0" k="numeric">−0.1105</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_perm_ci/1" k="numeric">0.0547</sworn>],
class
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_class" k="quote">`rotation`</sworn>.
On permutation 1234's new seeds FREE − REAL2 is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_free_minus_real2_newseed" k="numeric">0.3302</sworn>,
REAL2 − CLAMPED
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_real2_minus_clamped_newseed" k="numeric">−0.012</sworn>
and recovery
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_recovery_newseed" k="numeric">−0.0378</sworn>.

How the verdict row reads them: it does not. The row the receipt scored names `G_P_plumbing`, `G_D_stable`,
`G_F_floor`, `G_C_anchor`, the three gap gates (`G_G_exists`, `G_G_mean`, `G_G_ci`) and `G_B_long`, and neither
attribution gate,
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L262" k="quote">`"G_G_ci": true, "G_B_long": false}, "verdict": "BUDGET_CONFOUND__gap_does_not_survive_3x_steps"`</sworn>,
and the RESULT says so in its own words:
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L216" k="quote">`This row does not read the`</sworn>
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L217" k="quote">`attribution gates; the numbers are reported, not pronounced.`</sworn>
So `G_R_diversity` and `G_R_rotation` are reported, not read by the `BUDGET_CONFOUND` row. This NOTE reads them no
further: on permuted MNIST no verdict was pronounced on rotation against timescale diversity, and the ordered-copy
verdict above is not carried to it.

## The sealed untied receipt

The permuted-MNIST untied run that `PREREG_pmnist_untied_2026_09_02.md` designs ran on 2026-10-11, and its receipt is
still sealed. SEALS row 2 seals that PREREG,
<sworn r="path:papers/checksum/SEALS_2026_09_13.md#L33" k="quote">`papers/frequency-resonance/PREREG_pmnist_untied_2026_09_02.md`</sworn>,
and the robustness RESULT says the receipt
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L270" k="quote">`has not`</sworn>
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L271" k="quote">`been unsealed for this RESULT, and its verdict is not printed here`</sworn>.
`NOTE_forward_pointer_pmnist_untied_2026_10_23.md`, beside that PREREG, says the same. This NOTE does not unseal,
read or score the receipt.

## What the prior-art reads file against the general sentence

`NOTE_prior_art_full_text_reads_2026_10_23.md` quotes the general sentence as INDEX row 51 carried it,
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L464" k="quote">`The single-knob phase clamp shows oscillation is causally load-bearing in SSMs (+0.3122 on permuted MNIST)`</sworn>,
and files it:
<sworn r="path:papers/frequency-resonance/NOTE_prior_art_full_text_reads_2026_10_23.md#L468" k="quote">`"in SSMs" exceeds the measurement: one LRU-style architecture. No read paper supports the general sentence`</sworn>.
Its CHANGELOG entry reads
<sworn r="path:CHANGELOG.md#L199" k="quote">`INDEX row 51's "in SSMs" exceeds every read`</sworn>.
The sentence at lines 34 to 36 above states that general sentence about state-space models, and the reads support it
there no more than in INDEX row 51. The headline's measurement is one three-block LRU-style recurrence on permuted
MNIST (`CLRU` in `run_pmnist_ablation.py`); this RESULT's is one ordered-copy task at one width.

## Why beside it and not inside it

The lead's decision, as for the two pointer NOTEs of 2026-10-23 beside the certified 2026-07-23 RESULT and the sealed
untied PREREG: this RESULT's bytes are named, and this lab never regenerates a committed receipt in place, so the
pointer lands beside it as a NOTE. For this document:

- It is sworn. Its verdict receipt names the sha256 of its bytes,
  <sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.sworn-receipt.json#/document/inline_sha256" k="quote">`ec7921a03bbfcb74b6f87846da4ecbfe8d6f8f9fa84c0b661d036461c9db603a`</sworn>,
  and the file still is those bytes:
  <sworn r="path:papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md" k="hash">its blob hashes to `ec7921a03bbfcb74b6f87846da4ecbfe8d6f8f9fa84c0b661d036461c9db603a`</sworn>.
  `tests/test_sworn_dogfood.py` renders its sidecar back to those bytes.
- The charon log names the same digest for the same path, on its line 221:
  <sworn r="path:papers/charon/charon.log.jsonl#L221" k="quote">`"path": "papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02.md", "sha256": "ec7921a03bbfcb74b6f87846da4ecbfe8d6f8f9fa84c0b661d036461c9db603a"`</sworn>.
- The leg-2 measurement PREREG lists it in its population,
  <sworn r="path:papers/sworn/PREREG_sworn_measurement_2026_10_23.md#L263" k="quote">`papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02`</sworn>,
  and the packet its lock table pins by sha256,
  <sworn r="path:papers/sworn/PREREG_sworn_measurement_2026_10_23.md#L551" k="quote">`140ff006f92f3b64f88019b5c5f326052b92c3e4af36623cdc59c21b53eef9e9`</sworn>,
  carries this RESULT's section "The finding", the sentence above included:
  <sworn r="path:papers/sworn/measurement/packet_L.json#/items/3/text" k="quote">`The arc's headline, that oscillation is causally load-bearing in state-space models, survives`</sworn>.

Any byte added inside the RESULT would leave its verdict receipt and the charon line naming bytes the file no longer
holds, and the leg-2 population and its pinned packet describing a document that has moved. Putting this pointer
inside `RESULT_untied_magnitudes_2026_09_02.md` stays the operator's call.

## What this note does not do

- It does not put the pointer inside the RESULT. A reader who opens only that file does not see it; that is the cost
  of this route.
- It does not unseal, read or score the sealed untied receipt.
- It reports no measurement of its own and re-runs nothing: the ordered-copy numbers above are the 2026-09-02
  receipt's, and the permuted-MNIST numbers are the 2026-07-23 and robustness receipts'.
- It does not test the ordered-copy gap at a longer budget.
- It says nothing of the theta-gamma reading in lines 35 and 36. The robustness RESULT does not test it.
- No outcome it points to bears on deployed models, on attention, on importance or on priority (robustness PREREG
  section 11).

## Sworn

This note swears 69 spans, each to bytes committed at the commit its sidecar names: the 2026-09-02 RESULT's lines
and blob, its receipt, its verdict receipt and the charon log; the 2026-06-03 receipt and PREREG; the robustness
receipt, RESULT and PREREG; the 2026-07-23 receipt; SEALS row 2; the prior-art NOTE and the CHANGELOG; and the
leg-2 PREREG and its packet. The sidecar and the verdict receipt beside this file were written by
`python -m styxx.sworn canon` and `python -m styxx.sworn verify`; the verdicts are the receipt's.
