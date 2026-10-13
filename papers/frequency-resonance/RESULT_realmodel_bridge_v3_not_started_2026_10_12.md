# RESULT — the real-model bridge, attempt three, was NOT STARTED: two pilots, both NEITHER; the behavioural route is closed

**Status: NOT_STARTED, final.** No main-set item was generated or scored, so the preregistration's gates block never
ran and no verdict string exists. The route was closed by a rule the operator set before the second pilot ran
(amendment 17.2), not by a gate.

Fathom Lab · 2026-10-12 · Preregistration: `PREREG_realmodel_bridge_v3_2026_10_11.md`, with amendments 17.1, 17.2 and
17.3 (unsealed and unsigned; the judge's defaults are what was built). Runner: `run_realmodel_bridge_v3.py`. Receipts:
`realmodel_bridge_v3_smoke.json` (plumbing, not read), `realmodel_bridge_v3_pilot.json` (pilot one) and
`realmodel_bridge_v3_pilot2.json` (pilot two). Every pilot number below is sworn to those receipts at the commit this
document's sidecar names; the two tables are built from sworn cells.

## What was attempted

The arc's controlled results say that a pure-decay channel recalls one planted fact at every gap the lab measured,
but did not learn to compare a fact with a claim far away, while a rotating channel did
(`RESULT_consistency_oscillation_2026_07_23.md`, `RESULT_consistency_horizon_2026_07_24.md`,
`RESULT_recall_horizon_2026_07_24.md`). Two earlier attempts to look for that dissociation in deployed language
models ended ABSTAIN (`RESULT_realmodel_bridge_2026_07_24.md`): recall at ceiling beside compare at chance, then a
constant responder scoring the label base rate. Attempt three was built to the design that RESULT named: a few-shot
format, exact label balance, a degeneracy guard, and a capability screen at the shortest distance before any distance
is read.

The arms were `state-spaces/mamba-1.4b-hf` (M1) and `EleutherAI/pythia-1.4b` (T), an attention model of the same
scale with the same tokenizer. M1's recurrence is
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/redteam/mamba_A_real_and_negative_every_layer/layers" k="numeric">real and strictly negative in all 48 layers of the loaded weights</sworn>
(<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/redteam/mamba_A_real_and_negative_every_layer/max_A" k="numeric">largest entry −0.0039</sworn>):
pure decay, the clamped arm of the lab's own ablation shipped as a language model. The main run would have scored
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_smoke.json#/budget/n_per_cell" k="numeric">600 items per cell</sworn>,
three planted four-digit codes per item, recall and compare over the same items at four distances from 16 to 1,024
tokens of distinct wikitext filler, under eight gates. The screen asks both operations to reach 0.40 normalized
accuracy at the shortest distance (D0, 16 tokens) in both arms, with the degeneracy guard passed, before any slope
may be read. Section 9 allowed one bounded format pilot at D0 before the seal.

## The smoke: plumbing only

The smoke (24 items at 16 and 1,024 tokens, on its own seed and reserved slices, both arms) returned
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_smoke.json#/verdict" k="quote">the verdict `INVALID__smoke_plumbing_only`</sworn>,
as the gates block requires of a smoke, with every red-team assert of section 12 held, and it measured the main run at
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_smoke.json#/budget/total_hours_timed_arms" k="numeric">1.635 GPU-hours</sworn>,
inside the four-hour rule. Its accuracies are plumbing numbers on a handful of items and are not read here.

## Pilot one: both answer formats fail (amendment 17.1)

Pilot one ran on
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/recall/M1/n" k="numeric">48 items</sworn>
at D0, item seed
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/config/seeds/pilot" k="numeric">1011</sworn>,
codes and sentences disjoint from the main set and the smoke, both arms, exact balance. Its compare question was a
claim check, "Question: Is the {name} code {claimed}? Answer:", read as the sign of log p(" Yes") − log p(" No") and
again with " True" / " False"; a false claim named one of the other two planted codes. Recall took the argmax over the
three planted codes. Normalized accuracy is (a − chance) / (1 − chance), with chance one half for this compare and one
third for recall; the bar is 0.40, and the guard asks the compare answer rate to sit inside [0.20, 0.80].

| arm | compare, Yes/No | Yes-rate | AUROC | compare, True/False | True-rate | AUROC | recall |
|---|---|---|---|---|---|---|---|
| M1 | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/yes_no/M1/compare/acc_norm" k="numeric">−0.042</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/yes_no/M1/compare/yes_rate" k="numeric">0.94</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/yes_no/M1/compare/auroc" k="numeric">0.38</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/true_false/M1/compare/acc_norm" k="numeric">−0.167</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/true_false/M1/compare/yes_rate" k="numeric">0.92</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/true_false/M1/compare/auroc" k="numeric">0.37</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/recall/M1/acc_norm" k="numeric">0.656</sworn> |
| T | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/yes_no/T/compare/acc_norm" k="numeric">0.083</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/yes_no/T/compare/yes_rate" k="numeric">0.42</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/yes_no/T/compare/auroc" k="numeric">0.56</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/true_false/T/compare/acc_norm" k="numeric">0.042</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/true_false/T/compare/yes_rate" k="numeric">0.56</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/true_false/T/compare/auroc" k="numeric">0.53</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/recall/T/acc_norm" k="numeric">0.969</sworn> |

All compare and recall cells are normalized accuracy. The Mamba arm answered Yes on nearly every item and failed the
guard, then answered True on nearly every item and failed it again. The Pythia arm's answers varied and passed the
guard, but its compare stayed far under the bar in both formats, and its log-odds carried almost no information about
the label (an AUROC of one half is a responder that ignores the label). Both arms recalled the planted code: raw recall
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/recall/M1/acc_raw" k="numeric">0.771 for the Mamba arm</sworn> and
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/by_format/recall/T/acc_raw" k="numeric">0.979 for the Pythia arm</sworn>.
Section 9 lets one failed format switch to the other once; both failed, and the receipt records
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot.json#/format_recommendation" k="quote">the recommendation `NEITHER — both formats fail the pilot`</sworn>.
Amendment 17.1 wrote NOT_STARTED on the format line.

## The operator's one redesign, pre-stated as the last (amendment 17.2)

The operator then approved one bounded redesign of the compare readout and stated in advance that a second failed
pilot closes the behavioural route. The redesigned compare is argmax-shaped like recall, so no Yes-prior exists to
drift: the receipt records the question as
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/config/compare_question" k="quote">`Question: Which name has the largest code: alpha, gamma or delta? Answer:`</sworn>,
scored as the argmax over the three name tokens, correct iff the chosen name holds the numerically largest of the
three planted codes, with exact balance over the correct slot. A constant responder scores exactly zero normalized in
both families, and the compare guard becomes recall's: no one answer on more than
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/config/guard/max_slot_fraction" k="numeric">0.80 of items</sworn>.
The test names became alpha, gamma and delta because " bravo" is two NeoX tokens; the exemplars use sigma, theta and
omega, disjoint from them. What "compare" measures changed with the readout, and 17.2 says so: the claim check asked a
fact against a later claim, the three-way question asks a fact against the other facts. The pre-stated rule: an arm
clears iff at D0 its compare and its recall are both at or above
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/config/screen_bar" k="numeric">0.40 normalized</sworn>
with both guards passed; if both arms clear, the document is sealed, anchored and smoked before a main run; otherwise
the format line reads NOT_STARTED, final, and the route is closed. No bar moved.

## Pilot two: neither arm clears (amendment 17.3)

Pilot two ran once, on
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/config/n_per_cell" k="numeric">48 items</sworn>
at D0, item seed
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/config/seeds/pilot2" k="numeric">1012</sworn>,
both arms, with both arms' model-level asserts held before any cell was scored. Its sentences and codes are disjoint
from both committed receipts by sha256 list:
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/redteam/disjoint_from_committed_receipts/this_run_sentences" k="numeric">140 sentences of its own</sworn>,
none among the
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/redteam/disjoint_from_committed_receipts/prior_sentence_indices" k="numeric">1,284 the smoke and pilot one used</sworn>.
The whole run took
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/runtime/total_seconds" k="numeric">119.4 seconds</sworn>,
of which
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/runtime/scoring_seconds" k="numeric">72.1 seconds</sworn>
on the GPU. The receipt names the runner that wrote it by
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/config/runner_sha256" k="quote">its sha256, `24e8c3ba8a3344f4e13ca41bce30d8cfa9cec53beffc00719c78c96b55c6987f`</sworn>,
and <sworn r="path:papers/frequency-resonance/run_realmodel_bridge_v3.py" k="hash">the runner in the tree this document is sworn at hashes to `24e8c3ba8a3344f4e13ca41bce30d8cfa9cec53beffc00719c78c96b55c6987f`</sworn>.

| arm | compare | raw | gamma share | AUROC | compare, codes removed | recall | AUROC | recall, codes removed |
|---|---|---|---|---|---|---|---|---|
| M1 | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/acc_norm/M1/compare/16" k="numeric">−0.062</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/acc_raw/M1/compare/16" k="numeric">0.292</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/guard/M1/compare/16/slot_fractions/1" k="numeric">0.92</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/guard/M1/compare/16/ovr_auroc" k="numeric">0.449</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/controls/ablated/M1/compare/16/acc_norm" k="numeric">0.000</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/acc_norm/M1/recall/16" k="numeric">0.812</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/guard/M1/recall/16/ovr_auroc" k="numeric">0.977</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/controls/ablated/M1/recall/16/acc_norm" k="numeric">−0.156</sworn> |
| T | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/acc_norm/T/compare/16" k="numeric">0.000</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/acc_raw/T/compare/16" k="numeric">0.333</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/guard/T/compare/16/slot_fractions/1" k="numeric">1.0</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/guard/T/compare/16/ovr_auroc" k="numeric">0.535</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/controls/ablated/T/compare/16/acc_norm" k="numeric">−0.031</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/acc_norm/T/recall/16" k="numeric">0.875</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/guard/T/recall/16/ovr_auroc" k="numeric">0.999</sworn> | <sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/controls/ablated/T/recall/16/acc_norm" k="numeric">−0.094</sworn> |

Compare, recall and the two codes-removed columns are normalized accuracy against chance one third; the gamma share is
the fraction of items answered with the middle name; AUROC is the one-versus-rest mean over the three slots, where one
half is a constant responder. Both arms answered gamma on nearly every item (the Pythia arm on every one) and failed
the guard. The name scores carried no measurable information about which code was largest (AUROC near one half),
so this is not a calibration artefact a threshold could move, as the Yes-prior of pilot one might have been: the
operation was not performed. Both arms recalled the planted code, and with the code sentences removed both families
sat at or below chance in both arms, so the recall numbers read the planted codes and not priors.

## The rule applied, and the closure

The Mamba arm's compare is below the bar and fails the guard; so is the Pythia arm's. The receipt records
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/pilot_outcome" k="quote">the outcome `NEITHER`</sworn>
and
<sworn r="path:papers/frequency-resonance/realmodel_bridge_v3_pilot2.json#/format_line_to_enter" k="quote">the format line to enter, `NOT_STARTED`</sworn>,
and amendment 17.3 enters it as the document's last format line:
<sworn r="path:papers/frequency-resonance/PREREG_realmodel_bridge_v3_2026_10_11.md#L661" k="quote">`Answer format chosen (written before sealing): NOT_STARTED`</sworn>.
That line is final. The runner reads the last such line and refuses the main run: on 2026-10-12, at the commit before
this document, `python run_realmodel_bridge_v3.py` exited 2 with "the PREREG records NOT_STARTED on its format line:
the main run is not started", and `--pilot` exited 2 because the pilot-two receipt exists; the refusal note each run
writes was deleted and never committed.

By the operator's rule of amendment 17.2, a second failed pilot closes the behavioural route, and it is closed. This
is not an ABSTAIN of the gates block: the block never ran, and the PREREG's own closing condition ("A third ABSTAIN
closes the behavioural route", section 14) was never reached; the stricter rule the operator set before pilot two
closed it earlier. Nothing is sealed: whether to seal a preregistration whose run does not start is the operator's
(decision 5). Nothing has been posted (decision 6).

## What this establishes, at pilot strength

- At 16 tokens, given four exemplars, two 1.4 B base models — `mamba-1.4b-hf` and `pythia-1.4b` — recall a planted
  four-digit code well above the frozen bar, in both pilots.
- Neither performs the compare operation above the bar under either readout design: the claim check, in two answer
  formats, and the three-way choice of the largest code. The compare answers collapse — the Mamba arm's onto one token
  under the claim check, both arms' onto one slot under the three-way choice — and where the Pythia arm's claim-check
  answers varied, they did not measurably track the label.
- That is a sentence about an instrument: few-shot compare probes on these two models at the shortest distance. Each
  pilot is 48 items, and section 9 says pilot numbers license nothing beyond the record of the attempt.

## What it does not establish

- Nothing about decay versus attention. The between-model contrast of section 7 was never computed, and both arms
  failed the same way.
- Nothing about distance. No item farther than 16 tokens was scored outside the smoke's plumbing pass, which is not
  read.
- Nothing about deployed models beyond these two checkpoints. Other scales, instruction-tuned models, other families,
  other prompts, more exemplars and a written-out chain of reasoning were not tried; any of them is a different
  instrument and a new preregistration (amendment 17.1, last paragraph).
- Nothing for or against the arc's controlled-SSM results, which stay what each RESULT already says.
- Nothing about honesty in language models, and nothing about importance. Section 14 of the preregistration: "No
  outcome bears on honesty in LLMs, frontier models, human cognition, or importance".

## Two defects in the record, found while writing this

1. **Amendment 17.1 cites three runner commits by hashes that are not on the branch.** One commit message on the
   branch carried a word the lab never prints. It was reworded on 2026-10-11 by rewriting the branch from that commit
   on, after 17.1 had been committed, and 17.1's text kept the hashes it was written with. On the branch, `65397781`
   is `93b59521`, `6c21580b` is `424c847a`, and `9b54d8d2` — the commit 17.1 names as the runner that wrote the smoke
   and pilot-one receipts — is `e28a2d72`. Each pair has identical trees (`git diff` between them is empty), and the
   runner blob at `e28a2d72` has the sha256 `45dd70df…` that both receipts record as `config.runner_sha256`. The
   earlier hashes resolve only in the clone where the rewrite ran. The PREREG's frozen text is not edited; the mapping
   is printed here.
2. **The forward pointer section 15 owes is not written.** Section 15 asks for a pointer to this RESULT in
   `RESULT_realmodel_bridge_2026_07_24.md` and in `papers/SYNTHESIS_program_2026_07_24.md` item 3, in the same
   commit. The July RESULT is certified: its certificate binds the document by
   <sworn r="path:papers/frequency-resonance/CERT_realmodel_bridge.json#/document_sha256" k="quote">the field document_sha256, `6d9bf383e50360c36cae0e277e35b8a0e2b6e67f94797addd7c340ffdea8cda7`</sworn>,
   and <sworn r="path:papers/frequency-resonance/RESULT_realmodel_bridge_2026_07_24.md" k="hash">the document in the tree hashes to `6d9bf383e50360c36cae0e277e35b8a0e2b6e67f94797addd7c340ffdea8cda7`</sworn>,
   so an edit would leave the certificate naming bytes the file no longer holds. The synthesis's bytes are recorded in
   `papers/closed-model-frontier/oath_v11_baseline_ledger.json`. The pointer stays owed; whether it lands as an edit
   with a fresh certificate or as a note beside each document is the operator's call.

## Forward state

The behavioural bridge is closed. The only open bridge is the profiler route — named Design B under Q2 in the lab's
working note `STATE_OF_THE_ARC_v2.md`: run the resonance profiler shipped in `styxx/resonance.py`
(`INSTRUMENT_resonance_profiler_2026_07_23.md`) on a published oscillatory checkpoint, the step
`papers/SYNTHESIS_program_2026_07_24.md` item 3 says "either opens the bridge to real models or closes it". It needs
an adapter and a checkpoint. No oscillatory language-model checkpoint is cached on this box, and the LinOSS
repository publishes code with no checkpoint (section 3 of the preregistration), so the route needs a download or a
training run; the operator has approved neither. Until one runs, the July statement stands: the
recall-versus-comparison dissociation is a controlled-SSM result, and the bridge to deployed models is open, with
one behavioural instrument now spent.

The runner's plumbing — exact balance computed from the items, the controls, per-item scores, sha256 exclusion lists,
the asserts before any scoring — is reusable by a successor preregistration. This document's bars are not reusable
as if they had been met.

---

*Two models that keep the code and do not weigh it. The instrument asked a question these models do not answer at
this scale, and the record says that before it says anything about rhythm.*
