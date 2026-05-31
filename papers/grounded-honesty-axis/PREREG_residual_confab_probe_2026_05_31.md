# PREREG — can the RESIDUAL STREAM catch confident factual confabulation that the OUTPUT distribution cannot? (representation-level honesty probe)

**REGISTERED 2026-05-31, before the probe is trained.** Item set hashed pre-run.
**SIGN-OFF:** Flobi — *"go deeper, push the tech farther than was ever possible before"* (2026-05-31).

## The wall this attacks (and why it is not a re-litigation)

The arc **closed** *output-level* detection of confident confabulation: first-token entropy, logit
margin, sampling stability, and stated confidence are all blind to a *stable false belief*
(`FINDING_truthfulqa_crossmodel`: no model-internal or model-vs-model **output** signal catches it;
`grounded_arc_bet0`: confidence cannot flag hallucination, within-hallucinated ρ≈0). Retrieval
(external) is the only known lever. **The unasked question:** can a linear probe on the **residual
stream — the representation, not the output —** separate confident-wrong from confident-right? No prior
cell trained a residual probe on the *confident* regime; this is open, and it is the field's hardest
honesty problem.

## Design

Model = **Qwen2.5-3B-Instruct** (white-box, local). Data = **TriviaQA** verified answers (no OpenAI),
hashed pre-run. Per item: greedy short answer; **first-token entropy + margin** (the OUTPUT signals);
**residual activations at the commitment position** (the first-answer-token, across all layers);
correct/wrong via the unit-tested `_evallib.alias_match`.

- **CONFIDENT subset** = items with first-token entropy **below the per-model median** — the model
  committed confidently, so *by construction the output entropy signal is near-useless here.*
- Within the confident subset: stratified **TRAIN/TEST split (70/30, fixed seed)**. Per layer: an
  L2 logistic-regression probe, k-fold CV on TRAIN to pick the **single best layer**; that layer's
  probe is then evaluated on the **held-out TEST split** — so the headline AUC is **not inflated by
  layer selection** (the multiple-comparison correction). Output-signal baseline (entropy, margin AUC)
  on the same TEST confident items.

## Bars (FIXED, on the held-out TEST)

| Bar | Statement | Threshold |
|---|---|---|
| **PROBE** *(key)* | the representation separates confident wrong from right | residual-probe ROC-AUC **≥ 0.70** |
| **CONTRAST** *(key)* | it sees what the output cannot | probe AUC − best output-signal AUC **≥ 0.20** |

**RESULT = SURVIVED iff PROBE ∧ CONTRAST.** Powered: ≥ 25 confident-wrong AND ≥ 25 confident-right
overall, else underpowered/disclosed (raise N / swap to a more confabulating model).

## Rigor & non-circularity

Ground truth = TriviaQA aliases (tested matcher), independent. Probe evaluated on **held-out** items
(not in-sample). Best layer chosen on **TRAIN only**. CONTRAST holds difficulty fixed — both classes
are confident on the same items, so a win is "representation beats output," not a difficulty artifact.

## Honest scope (load-bearing)

Single model, TriviaQA, **linear** probe + first-token position, one run, feasibility-grade. A SURVIVED
result means **"a linear direction in activations separates confident-wrong from confident-right"** —
**NOT** "the model knows it is fabricating." We measure representation, **never mind / phenomenology**.
The probe may read familiarity/topic rather than confabulation per se (the confident-subset + held-out
test mitigate, not eliminate — disclosed). A FAILURE means confident confabulation is **not linearly
decodable** from the commitment residual — the wall is representational and deep — a strong negative
either way. Does NOT revive the universal-oracle or cross-vendor claims (both CLOSED); scoped to one
model's own residuals.

## One line

Read the representation, not the output: does a linear probe on the residual stream catch the confident
confabulation every output-level signal is blind to — the field's hardest honesty problem — with a
held-out kill-gate it can fail.
