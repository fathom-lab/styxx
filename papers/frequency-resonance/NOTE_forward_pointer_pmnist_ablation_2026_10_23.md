# NOTE — the forward pointer from the 2026-07-23 permuted-MNIST RESULT to the robustness RESULT

**status: a forward pointer, sworn to committed bytes. It stands beside `RESULT_pmnist_ablation_2026_07_23.md`, in
this directory, and edits nothing: that RESULT, `CERT_pmnist_ablation.json` and `pmnist_ablation_result.json` keep
their bytes.**

2026-10-23. Section 10 of `PREREG_pmnist_robustness_2026_10_11.md` owes `RESULT_pmnist_ablation_2026_07_23.md` a
forward pointer to `RESULT_pmnist_robustness_2026_10_23.md`, and that RESULT lists the pointer under "Owed with this
RESULT, and not done by the kit that wrote it". The kit's pointer paragraph, quoted byte for byte:

> **Forward pointer (2026-10-23).** The permuted-MNIST measurement this document reports was re-run under
> `PREREG_pmnist_robustness_2026_10_11.md` (five held-out permutations, three new seeds on permutation 1234, a 3x
> budget arm and the untied control under one frozen bar); its RESULT, `RESULT_pmnist_robustness_2026_10_23.md`, reads
> `BUDGET_CONFOUND__gap_does_not_survive_3x_steps`.

In it, "this document" is `RESULT_pmnist_ablation_2026_07_23.md`.

## What it points to

The measurement the pointer names is the 2026-07-23 receipt's FREE minus CLAMPED,
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/result/free_minus_clamped" k="numeric">+0.3122</sworn>,
at
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/steps" k="numeric">4000</sworn>
steps on permutation
<sworn r="path:papers/frequency-resonance/pmnist_ablation_result.json#/config/perm_seed" k="numeric">1234</sworn>,
in a three-block complex-diagonal linear recurrence (LRU-style; `CLRU` in `run_pmnist_ablation.py`). The re-run's
receipt reads
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
the training budget: one permutation and five runs do not license it (M8). The verdict is its own row of the frozen
table, neither `NOT_ROBUST__gap_confined_to_anchor_permutation` nor
`HEADLINE_NOT_REPRODUCED__anchor_permutation_three_new_seeds`, and the re-run's seeds 0 and 1 reproduced the
2026-07-23 receipt's seed means with a larger deviation of
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_max_abs_dev" k="numeric">0.0</sworn>
(reported, not gated).

## Why beside it and not inside it

The lead's decision, for this NOTE and for `NOTE_forward_pointer_pmnist_untied_2026_10_23.md` beside the untied
PREREG: both target documents are named by the sha256 of their bytes, one by a certificate and one by a seal, and
this lab never regenerates a committed receipt in place, so the pointers land beside them as two NOTE files, not
inside them. For this document:

- `CERT_pmnist_ablation.json` names it,
  <sworn r="path:papers/frequency-resonance/CERT_pmnist_ablation.json#/document" k="quote">`RESULT_pmnist_ablation_2026_07_23.md`</sworn>,
  by the digest
  <sworn r="path:papers/frequency-resonance/CERT_pmnist_ablation.json#/document_sha256" k="quote">`2e5154efc55ec18d59de0f4e8ec23d2a8fb6c7f1f9a7588d7deaa6fa5430e741`</sworn>,
  the sha256 of the file as committed on 2026-07-23 at `71b2812a`.
- The file was edited in place once already. On 2026-09-02, at `2acf4f29`, it gained a forward pointer as its line
  3, which opens
  <sworn r="path:papers/frequency-resonance/RESULT_pmnist_ablation_2026_07_23.md#L3" k="quote">`> **Forward-pointer, added 2026-09-02.**`</sworn>,
  and since then
  <sworn r="path:papers/frequency-resonance/RESULT_pmnist_ablation_2026_07_23.md" k="hash">its blob hashes to `31908649bbb5bc62a2302963fa073f13d212b752335e808125fa3c93b3d0963e`</sworn>,
  not to the certificate's digest. Since that day the certificate has named bytes the file no longer holds; that
  edit is history and is not undone here.
- A second pointer inside the file would move it further from the bytes the certificate names, and a pointer inside
  it that a certificate covers needs a fresh certificate beside the old one, which stays as committed.

Putting this pointer inside `RESULT_pmnist_ablation_2026_07_23.md` stays the operator's call.

## What this note does not do

- It does not put the pointer inside the RESULT. A reader who opens only that file does not see it; that is the cost
  of this route.
- It does not meet section 10's "in the same commit as the RESULT": the RESULT was committed without the pointer and
  lists it as owed.
- It reports no measurement of its own. The robustness RESULT and its receipt are the record, and no outcome there
  bears on deployed models, on attention, on importance or on priority (PREREG section 11).

## Sworn

This note swears 15 spans, each to bytes committed at the commit its sidecar names: the two runs' receipts, the
certificate, and the 2026-07-23 RESULT's blob and its line 3. The sidecar and the verdict receipt beside this file
were written by `python -m styxx.sworn canon` and `python -m styxx.sworn verify`; the verdicts are the receipt's.
