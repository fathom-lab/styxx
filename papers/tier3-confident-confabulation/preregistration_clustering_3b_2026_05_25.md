# Pre-Registration AMENDMENT (3b) · de-circularize the validity gate

**Committed BEFORE the 3b data.** Run 3a VOIDed because its validity gate keyed on
**mean pairwise cosine < 0.85** — but cosine clustering is one of the *methods under
test*. Gating validity on a tested method is **circular**, and it excluded the exact
regime (cosine 0.85–0.95) where the methods diverge and the comparison is informative.
This amendment fixes ONLY the validity gate. **The kill-gate bars G2 and G3 are
unchanged** (no goalpost-moving — the pass/fail thresholds are identical to the original
prereg).

## What changes

- **Validity gate (new, non-circular):** a correct item counts as *surface-varied* iff
  its 6 samples contain **≥ 4 distinct normalized surface forms** (unique strings after
  lowercasing + stripping punctuation) — a measure of lexical variation independent of
  cosine, NLI, and the LLM-judge. Probe is VOID if **< 4** valid surface-varied correct
  items.
- `varied_correct` (the false-positive-load subset) is redefined the same way.

## What does NOT change

- Classes (C1 fact + C2 explanation = correct; C3 fictional-confab), N=6, the
  varied-wording nudge, the three clustering methods (cosine sweep, nli-deberta,
  LLM-judge), and the kill-gate:
  - **G2 (judge solves it):** LLM-judge AUC ≥ 0.80 AND its mean entropy on surface-varied
    correct ≤ 0.40 × its C3-confab mean.
  - **G3 (cheap insufficient):** no cheap method (cosine sweep, nli-deberta) achieves both
    AUC ≥ 0.80 and varied-correct FP ratio ≤ 0.40.
  - **PASS = G2 ∧ G3** (given validity).

## Why this is legitimate (not goalpost-moving)

Goalpost-moving = loosening the *kill-gate* after seeing results. Here the kill-gate is
untouched; only the *validity construction* is de-circularized, fixing a genuine design
error (a tested method used as the gate). The change is pre-registered before the 3b
run, and the new gate is principled (form-diversity, method-independent). Recorded
transparently alongside the 3a VOID it corrects.
