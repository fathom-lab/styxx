# FINDING (deception case study) — generative matched-length regen FAILED construct validity

**2026-06-24. SUPERSEDED IN HEADLINE by `FINDING_suite_causal_length_2026_06_24.md`** (the suite-wide causal
result). This file is the deception-specific case study and the cautionary negative on local-model regen.
The first draft of this file claimed "deception is length-FRAGILE"; the adversarial verification
(`survives:false`) showed that over-read the data. Corrected below.

## What was tested
Pre-registered (`PREREG_length_control_causal_2026_06_24.md`) generative matched-length regen of the
deception corpus (same 100 Q × honest/dishonest stances, identical length rule), two local generators, then
re-score the shipped deception_v0 weights + fresh CV. Goal: does the register survive at equal length?

## Why the regen is uninterpretable as a length result (construct-validity failure)
`_regen_construct_validity.txt`:
- **gpt-4o-mini (original):** dishonest 1% years / specificity gap +0.073 → valid honest/dishonest contrast.
- **Qwen2.5-3B @ 33w (manip OK, d_len +0.03):** dishonest 7% years / gap +0.033 → **DEGENERATE**: the 3B
  model under a tight budget ignored "be vague, no dates" and wrote accurate date-dense answers for BOTH
  classes. Its 0.56 matched-length AUC = absence of a contrast, NOT length-fragility. **Void as a length result.**
- **Llama-3.2-3B @ ~60w:** valid contrast (gap +0.075) but ignored the length rule (manip FAILED, d_len −0.49)
  → 0.82 is a length-LEAK upper bound.

Neither local generator gave a corpus that is both construct-valid and length-matched. **Lesson: generative
matched-length regen with small local models is unreliable; CEM on the real corpus is the trustworthy control.**

## What the trustworthy control says (CEM on the real gpt corpus, length-ablated)
Length-matching the real corpus and ablating `log_word_count`: AUC 0.956 → **0.81–0.85** (seed-stable,
survives ablation; n≈76 on the thin length-overlap region). `log_word_count` contributes only ~0.06 to the
headline. → **deception is construct-robust under causal length control. KEEP + modest caveat; do not
deprecate for length.** (Its separate generalization weakness, `project_deception_v1_negative`, is a
different axis.)

See the suite finding for the full picture (5 of 6 instruments construct-robust; only overconfidence is
causally length-confounded; the correlational length-floor over-flags).
