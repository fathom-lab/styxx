# PREREG — the auditor's ceiling: the false-accusation rate of mechanical QA grading

**Fathom Lab · 2026-07-04 · committed BEFORE any row is re-judged. Publishes regardless of verdict.**

## Question
Three independent catches this week share one shape: an auditor false-failing a truthful subject (TriviaQA's
alias list vs "Li'l Abner"; mechanical grading leaving 242/250 TruthfulQA rows ungradeable; our own version
checker vs our own repo). Benchmarks are auditors too — and their false-accusation rate is unmeasured. On the
keystone main run's 383 already-graded short-form QA rows (250 TriviaQA ID + 133 PopQA-rare), what fraction of
mechanically graded-FALSE answers are actually correct (label false-negative rate), and graded-TRUE actually
wrong (false-positive rate)?

## Method (frozen)
1. Data: `papers/depth-truth/results/main_id.jsonl` + `main_ood1.jsonl`, complete-case rows with a non-None
   `correct` (mechanical grades frozen 2026-07-02; no row is re-generated).
2. **BLIND re-judging**: independent judges see ONLY {question, model answer} — never the gold aliases, never
   the mechanical grade. Verdict per row: CORRECT / INCORRECT / UNSURE, with a one-line justification. UNSURE
   is excluded-and-counted (never coerced).
3. Adjudication: every row judged once; every row where the blind verdict DISAGREES with the mechanical grade
   is re-judged by a second independent blind judge; a disagreement stands only if both blind judges concur
   against the mechanical grade. Judge outputs are schema-forced and committed.
4. Estimates: FN = P(blind=CORRECT | mech=False), FP = P(blind=INCORRECT | mech=True), per benchmark, with
   Wilson 95% CIs. UNSURE rates reported alongside.
5. **KG-HUMAN (gate before any public claim):** flobi seals a stratified 30-row sample of the confirmed
   disagreements (or all, if fewer) — >20% overturn of the double-blind verdicts ⇒ the agent judging is not
   trustworthy; finding reverts to "attempted" and only the TruthfulQA ungradeability number (which needs no
   judge) publishes.
6. TruthfulQA arm needs no judging: 242/250 mechanically `grade_ambiguous` is already computed from the frozen
   run and is reported as-is.

## Pre-declared interpretation limits
- This measures OUR pipeline's mechanical grading (frozen §3 normalization + the datasets' own alias lists) on
  ONE model's outputs. It licenses "benchmark labels carry a measurable false-accusation rate on real model
  output" — NOT "benchmark X is broken for everyone."
- **The keystone null is NOT rescued and this prereg predicts it will not be:** depth is near-constant
  (std ≈ 0.056), so relabeling cannot create discrimination. A label-corrected H1 re-run is included as a
  pre-registered robustness check with the EXPECTED outcome = still null. If it flips, that is reported as a
  surprise at full size.

## Falsification map
- FN ≈ 0 (CI includes 0) ⇒ mechanical grading is fine at this operating point; Li'l Abner was an outlier;
  published as such.
- FN materially > 0 (CI excludes 0) ⇒ benchmark scores on short-form QA are deflated lower bounds with a
  phrasing-dependent bias term; labels-need-receipts joins the binding stack with numbers.
- High UNSURE ⇒ short-form QA grading is under-determined even for humans/judges; also a finding.
