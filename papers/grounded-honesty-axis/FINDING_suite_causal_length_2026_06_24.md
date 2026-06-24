# FINDING — A correlational length-floor OVER-flags; the causal test clears 5 of 6 guardrails

**2026-06-24. Offline, CPU/local-GPU, NO frontier key. Pre-registered
`PREREG_length_control_causal_2026_06_24.md` (`bf556ca`). Adversarially verified (7-agent workflow,
`survives:false` on the first draft → this is the corrected version).** Reproduce:
`python scripts/suite_causal_length.py`; `python scripts/suite_causal_ablate.py`;
`python scripts/_regen_construct_validity.py`.

## The claim, corrected

Yesterday's suite length-floor card (`FINDING_suite_length_floor_2026_06_23.md`, `2f003ed`) flagged **4 of 6**
stance-prompt guardrail instruments as length-"DOMINATED" (deception 0.87, overconfidence 0.74, loop 1.00,
plan_action 0.56), where the floor = AUC recoverable from `log(word_count)` ALONE. **That metric
systematically OVER-states length reliance.** The word-count floor measures whether *length correlates with
the label* (true by construction for "elaborate vs be-vague" stance prompts); it does NOT measure whether the
*instrument relies on length*. The causal question needs length held constant AND the length features
removed. Doing that **clears 5 of the 6** instruments.

## The causal test (no regeneration, on each instrument's OWN shipped corpus)

For each instrument: coarsened-exact length-matching (CEM — subsample so the two classes have identical
word-count distributions), scored with the instrument's own features both intact and with the explicit
length feature(s) **ablated**. The honest causal floor = **length-matched AND length-ablated CV-AUC**
(bracketed over two bin widths for the bin-sensitivity the sweep exposed).

| instrument | headline AUC | drop length-feat (no match) | **length-matched + ablated** | causal verdict |
|---|---|---|---|---|
| sycophancy (v0.3, already fixed) | 0.972 | 0.971 | **0.973** | construct-robust |
| **overconfidence** | 0.770 | 0.717 | **0.628 – 0.727** | **LENGTH-CONFOUNDED (only one)** |
| deception | 0.956 | 0.895 | **0.811 – 0.853** (n≈76) | construct-robust |
| goal_drift | 0.964 | 0.964 | **0.953** | construct-robust |
| loop | 1.000 | 1.000 | intrinsic† | construct-robust (non-length features alone = 1.000) |
| plan_action | 0.922 | 0.923 | **0.853 – 0.891** | construct-robust |

† **loop is length-INTRINSIC, not length-confounded:** loops and non-loops barely share a word-count range
(d_len −1.85), so CEM finds ≤4 matchable items — but dropping the length feature leaves AUC at 1.000, i.e.
the non-length features (repetition structure) separate perfectly on their own. The floor's "1.00" reflected
that loops are *definitionally* long, not that the instrument is a length detector.

**Result:** only **overconfidence** loses substantial discrimination under causal length control
(0.770 → 0.63–0.73; the two length features `mean_sentence_length` + `log_word_count` carry real weight) —
and even it keeps ~0.65 of real epistemic signal, so it is length-*confounded*, not length-*determined*. The
other five survive both controls. The correlational floor's 4/6 flag was 1/6 causally.

## The deception sub-experiment (a clean cautionary negative)

The prereg's PRIMARY metric was a generative matched-length regen (same Q × 2 stances, identical length rule,
local model). **It failed — and the failure is the lesson.** Construct-validity check on the regenerated
corpora (`_regen_construct_validity.txt`):

- **gpt-4o-mini (original):** dishonest class 1% contain a year, specificity gap +0.073 → valid contrast.
- **Qwen2.5-3B @ 33 words:** dishonest class 7% years / 13% digits, specificity gap +0.033 (< half) →
  **DEGENERATE.** Under a tight budget the 3B model ignored the "be vague, no dates" instruction and wrote
  accurate, date-dense answers for *both* labels. Its near-chance matched-length AUC (0.56) measures the
  *absence of a contrast in the corpus*, not length-fragility — so it is **void as a length result**.
- **Llama-3.2-3B @ ~60 words:** valid contrast (gap +0.075) BUT ignored the length rule (d_len −0.49, manip
  check FAILED) → its 0.82 is a length-LEAK upper bound, not a matched-length number.

Neither local generator produced a corpus that is simultaneously construct-valid AND length-matched.
**Generative matched-length regen with small (3B) local models is unreliable;** CEM on the real shipped
corpus is the trustworthy causal control, and it says deception is construct-robust (0.81–0.85; the
`log_word_count` feature contributes only ~0.06 to the 0.956 headline). The pre-registered primary was the
wrong tool for a reason not anticipated at registration (generator degeneracy); the secondary control (CEM)
carried the result — reported transparently, not as a "prereg passed."

## Decisions on the shipped instruments

- **overconfidence_v0 → REFINED by `FINDING_overconfidence_length_robust_2026_06_24.md`:** the offline
  length-matched rebuild returned an HONEST NULL across 2 generators — calibrated register is intrinsically
  ~16% more verbose (hedging costs words), so length is PARTLY construct-intrinsic, not purely spurious.
  "length-confounded" overstates it; ~0.09 AUC is length (part legitimate, part spurious). **Decision: ship a
  CAVEAT, not new weights;** force-ablating length (→~0.68) would delete partly-real signal. A clean split
  needs a frontier-key regen with hard length control.
- **deception_v0 → KEEP + modest caveat.** It is construct-robust under causal length control; `log_word_count`
  adds ~0.06. (Its separate, known generalization weakness — lexical → 0.59 on TruthfulQA,
  `project_deception_v1_negative` — stands, but that is not a length problem.) Do NOT deprecate for length.
- **loop / goal_drift / plan_action / sycophancy → no length action.** Construct-robust.
- **Ship a length-floor footnote on the suite card**: "the word-count floor over-states reliance; the causal
  (length-matched + ablated) test clears 5/6; only overconfidence is length-confounded."

## Honest scope / threats handled (from the adversarial review)
- The causal floors for deception/overconfidence/plan_action are on the thin length-OVERLAP region (n≈74–140
  of 200; honest/dishonest barely share a length range — that thinness is itself the original confound). CIs
  are wide; magnitudes (esp. overconfidence's) are bin-width-sensitive and reported as ranges, not points.
- sycophancy (n=1200) and goal_drift have ample overlap → their robust verdicts are solid.
- The first draft of this finding over-read the regen as "deception is length-fragile"; the verification
  workflow caught the corpus degeneracy and the non-ablated scorer. This version drops both overclaims.
- Local-model register ≠ gpt; the CEM backbone needs no generator and carries the causal weight. A
  frontier-key regen with a *generous* length budget remains the cleanest future cross-check.
- A new tool ships: `suite_causal_ablate.py` — a reusable causal length-reliance auditor (CEM + length-feature
  ablation) for any guardrail corpus, the correct complement to the correlational floor.
