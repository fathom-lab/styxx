# NOTE — the forward pointer from the sealed untied PREREG to the robustness RESULT

**status: a forward pointer, sworn to committed bytes. It stands beside `PREREG_pmnist_untied_2026_09_02.md`, in
this directory, and edits nothing: the PREREG keeps the blob SEALS row 2 names.**

2026-10-23. Section 10 of `PREREG_pmnist_robustness_2026_10_11.md` owes `PREREG_pmnist_untied_2026_09_02.md` a
forward pointer to `RESULT_pmnist_robustness_2026_10_23.md`, and that RESULT lists the pointer under "Owed with this
RESULT, and not done by the kit that wrote it". The kit's pointer paragraph, quoted byte for byte:

> **Forward pointer (2026-10-23).** The permuted-MNIST measurement this document reports was re-run under
> `PREREG_pmnist_robustness_2026_10_11.md` (five held-out permutations, three new seeds on permutation 1234, a 3x
> budget arm and the untied control under one frozen bar); its RESULT, `RESULT_pmnist_robustness_2026_10_23.md`, reads
> `BUDGET_CONFOUND__gap_does_not_survive_3x_steps`.

The kit wrote one paragraph for both documents that are owed a pointer. In it, "this document" is
`PREREG_pmnist_untied_2026_09_02.md`, which designs a measurement and reports none; beside it, "the measurement this
document reports" reads as the one it designs, FREE, CLAMPED and the untied real bank REAL2 on permuted MNIST, which
the robustness PREREG ran again as its untied control.

## What it points to

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
the flagship this PREREG's opening names.

This NOTE carries no verdict for the untied run. That run ended on 2026-10-11, its receipt is sealed, and the
robustness RESULT prints none for it:
<sworn r="path:papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md#L271" k="quote">`its verdict is not printed here`</sworn>.

## Why beside it and not inside it

The lead's decision, for this NOTE and for `NOTE_forward_pointer_pmnist_ablation_2026_10_23.md` beside the
2026-07-23 RESULT: both target documents are named by the sha256 of their bytes, one by a certificate and one by a
seal, and this lab never regenerates a committed receipt in place, so the pointers land beside them as two NOTE
files, not inside them. For this document:

- SEALS row 2 names its blob. Line 33 of `papers/checksum/SEALS_2026_09_13.md`, row 2 of its table, carries
  <sworn r="path:papers/checksum/SEALS_2026_09_13.md#L33" k="quote">`507ef9aceb297ba06511d9e96c79ab9e1d6eca867d2ab73390e884c455c4a600`</sworn>
  in its blob column, and the file still is that blob:
  <sworn r="path:papers/frequency-resonance/PREREG_pmnist_untied_2026_09_02.md" k="hash">its blob hashes to `507ef9aceb297ba06511d9e96c79ab9e1d6eca867d2ab73390e884c455c4a600`</sworn>.
- A seal is the digest of the frozen file's git blob
  (<sworn r="path:papers/checksum/SEALS_2026_09_13.md#L9" k="quote">`A seal is the sha256 of the **git blob** of the frozen file`</sworn>),
  so any byte added inside the PREREG, a pointer included, would leave row 2 naming a blob the file no longer is,
  while the sealed untied receipt is the run this PREREG froze before data.

Putting this pointer inside `PREREG_pmnist_untied_2026_09_02.md` stays the operator's call.

## What this note does not do

- It does not put the pointer inside the PREREG. A reader who opens only that file does not see it; that is the cost
  of this route.
- It does not unseal, read or score the untied receipt.
- It does not meet section 10's "in the same commit as the RESULT": the RESULT was committed without the pointer and
  lists it as owed.
- It reports no measurement of its own. The robustness RESULT and its receipt are the record, and no outcome there
  bears on deployed models, on attention, on importance or on priority (PREREG section 11).

## Sworn

This note swears 12 spans, each to bytes committed at the commit its sidecar names: the two runs' receipts, the
robustness RESULT's line 271, SEALS row 2 and the rule above it, and the untied PREREG's blob. The sidecar and the
verdict receipt beside this file were written by `python -m styxx.sworn canon` and `python -m styxx.sworn verify`;
the verdicts are the receipt's.
