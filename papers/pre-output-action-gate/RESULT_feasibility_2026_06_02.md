# RESULT — Pre-output action gate, feasibility probe

**Date:** 2026-06-02
**Pre-registration:** [PREREG_feasibility_2026_06_02.md](PREREG_feasibility_2026_06_02.md)
**Pre-data anchor:** `c07d8a2`
**Verdict (frozen rule):** SURVIVED-abstract.
**Verdict (honest interpretation):** **CONFOUND CEILING — uninformative for the real claim. Do NOT proceed to the full study on this evidence.**

## Result

| Model | best LOO AUC | best layer | LODO@best | LODO@~65% |
|---|---|---|---|---|
| Qwen2.5-1.5B | **1.000** | 10/28 | 1.000 | 1.000 |
| Qwen2.5-3B | **1.000** | 13/36 | 1.000 | 1.000 |
| Llama-3.2-1B | **1.000** | 4/16 | 1.000 | 1.000 |
| Llama-3.2-3B | **1.000** | 6/28 | 1.000 | 1.000 |

By the kill-gate frozen before data (LOO AUC ≥ 0.80 on ≥2 models, median LODO ≥
0.70), this **passes** — all four models, perfectly.

## Why the perfect score kills the claim instead of proving it

A 1.000 — *especially* a perfect leave-one-domain-out 1.000 — is the signature
of a **confound**, not a discovery. The prereg flagged the risk up front; the
result confirms it:

1. **The set is lexically separable by construction.** Every destructive prompt
   contains a destructive verb (delete / drop / erase / terminate / purge /
   revoke / wipe); every safe prompt a read-only one (list / show / describe /
   count). The end-of-prefill residual *trivially* encodes the tokens the model
   just read. The probe is almost certainly detecting **"a destructive verb is
   present,"** which is true by design and requires no model cognition.
2. **LODO does not rescue it.** Leave-one-domain-out holds out an action
   *domain*, but the destructive verbs **recur across domains** — a verb-based
   separator generalizes to a held-out domain for free. So LODO = 1.000 is fully
   consistent with pure lexical separation. (A length/position confound would
   look identical and is not ruled out either.)
3. **A trivial bag-of-words baseline would also score ~1.000.** When the dumbest
   possible feature matches the probe, the probe has demonstrated nothing about
   the residual's *semantic* content.

## What this did and did not establish

- **DID (trivially):** the residual linearly encodes the destructive-vs-safe
  distinction *of the prompt text*. Necessary, but uninformative — it was a
  foregone conclusion.
- **DID NOT:** show that the residual predicts the model's **chosen / emitted**
  action when destructiveness is **not lexically handed to it**. That is the
  actual product claim (predict the destructive tool call *before emission* when
  the model decides the action). It remains **completely untested.**

## The methodology worked exactly as intended

This is the point of pre-registration in an overclaiming field. The prereg
stated the lexical-confound risk and built in LODO *before* the run. The perfect
score now **forces** the honest interpretation rather than a victory lap. Had we
shipped "styxx predicts destructive actions, AUC 1.000," the first reviewer to
notice every destructive prompt says "delete" ends the company's credibility in
one sentence. We caught it ourselves, in minutes.

## Disciplined next step (the confound is the spec for the next test)

1. **Lexical-control feasibility (cheap, do first):** rebuild the set so
   destructiveness is **not** a giveaway verb — same/neutral surface verbs,
   destructiveness only in the *consequence* (e.g. "set the users-table
   retention to 0s and apply" vs "set it to 90 days"). If the residual still
   separates **without the lexical tell**, that is the first real evidence.
   If it collapses to chance, the signal was lexical and the prompt-level
   approach is dead.
2. **Emitted-action study (the real claim):** the model *chooses* the tool call
   in an agent scenario; read the residual at the decision token before
   emission; label by the **emitted action**, not a prompt verb. Cross-model,
   held-out. This is the days-of-build flagged in the brief — and it is only
   worth building if (1) holds.

A 1.000 is a question, not an answer. The honest status of the pre-output action
gate is: **necessary condition met trivially; real claim untested; next test
specified.**

— scored 2026-06-02 against the frozen gate; interpreted honestly against the
construct.
