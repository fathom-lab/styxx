# PREREG — Does attribution DEPTH explain the belief→truth dial? (white-box unification of the root + the frontier)

**Pre-registered 2026-05-28, BEFORE any white-box data is collected. One confirmatory
run. Feasibility-grade: single OPEN model, SAE-free logit-lens depth proxy,
n≈32 arithmetic items.** Receipt to be committed at
`depth_grounding_whitebox_result.json`. Ground truth COMPUTED in-code (the `operator`
module) and SHA-256'd before any depth scoring. Core correctness signal is exact
integer match (no LLM judge).

## Why this run (the unification)

Two of this project's research lines have never been put in the same experiment:

- **The root (fathom depth scorer / attribution-depth, ICML 2026 MI workshop).**
  White-box. Measures the attribution-mass-weighted mean layer index
  `D = (1/T) Σ_t (Σ_l l·a_l^(t)) / (Σ_l a_l^(t))`. Proved that *construction vs
  retrieval* is a real internal distinction **invisible to surface text** (5 rounds of
  surface controls failed). Headline direction: **recall is DEEPER (μ≈8.36) than
  reasoning (μ≈8.10)** on Gemma-2-2B — construction begins earlier in the forward
  pass; retrieval is a late, output-proximal hop. The paper never tied depth to
  correctness/truth.

- **The frontier (grounded honesty axis, this arc).** Black-box. Found that re-deriving
  a self-claim through method-diverse reasoning paths moves the grounded signal from
  *belief* (one-shot, AUC 0.694 on hard arithmetic) toward *truth* (path-diverse,
  0.955). On the competence cliff the model is *stably wrong* — confident confabulation
  (517×283 → 146051, ten-for-ten). The grounding **backend is a dial between belief and
  truth.** Mechanism never observed internally; only inferred from resampling.

**The hypothesis:** these are one phenomenon. *Construction-vs-retrieval* is the single
latent variable under both. Confident confabulation = the model in **retrieval mode**
(shallow late hop onto a wrong attractor); method-diverse derivation works because it
forces **construction mode**. If so, attribution depth is the *mechanistic substrate*
of the belief→truth dial — observable white-box, only inferable black-box.

## Apparatus (committed before data)

- **Model:** open, white-box, capable enough to show an arithmetic competence cliff
  (does easy arithmetic; confidently confabulates on hard). Primary:
  `Qwen2.5-1.5B-Instruct` (cached locally). If it shows no cliff, that is reported as a
  precondition failure (see below), not worked around.
- **Depth metric (SAE-free logit-lens direct logit attribution).** For the model's OWN
  generated answer, run one forward pass with `output_hidden_states`; for each answer
  token position t and each layer l, apply the final norm + unembed (logit lens) to the
  layer-l residual to get that layer's logit for the realized answer token, and take
  `a_l^(t) = relu(logit_l − logit_{l−1})` (the positive per-layer push toward the
  answer). Depth `D = (1/T) Σ_t (Σ_l l·a_l) / (Σ_l a_l)`. This is a faithful proxy for
  the published SAE/IG metric: it weights the layer axis by attribution **to the answer
  token** (not raw activation norm, which would be confounded by monotonic norm growth).
  Canonical Gemma Scope SAE confirmation is the explicitly-named NEXT gate, not this run.
- **Items:** arithmetic across difficulty (ctrl 3×2, mul 3×3, mul 4×3, multistep a×b±c),
  reusing the competence-cliff SPECS family; ground truth computed in-code + hashed.
- **Two generation conditions per item:** **one-shot** (bare "what is X? answer with
  the number") and **method-diverse derivation** (rotating CoT method prompts, ending
  "ANSWER: <n>"). Depth measured on each condition's generated answer span.

## Predictions (decisive bars — pre-stated, scored exactly, BOTH directions named)

| id | prediction | bar |
| --- | --- | --- |
| **W1 — confabulation has a distinct depth signature** | paired within-item depth differs between one-shot-confabulation and method-diverse-derivation answers | \|Cohen's d\| ≥ 0.5, paired p < 0.05. **Predicted sign: confab DEEPER** (recall-like) per the root. If significant but REVERSED, report "signature real, sign opposite paper" — still a discovery. |
| **W2 — depth is a white-box validity signal** | attribution depth separates CORRECT from CONFABULATED answers across items | AUC ≥ 0.70 OR ≤ 0.30 (predicts correctness in either direction) |
| **W3 — depth shift tracks the dial** | on items where method-diverse derivation RECOVERS truth (the black-box keystone effect), depth shifts construction-ward vs the one-shot confab, consistent in sign with W1 | paired p < 0.05 on recovered items, sign-consistent with W1 |
| **K — not a length / easy-item artifact** | the depth gap is not explained by generation length, and easy items show no spurious gap | regress Δdepth on Δlength → intercept p < 0.05; AND ctrl-tier (both-correct) depth gap n.s. (p > 0.05) |

**RESULT = SURVIVED iff W1 ∧ W2 ∧ W3 ∧ K.** Otherwise REPORT_AS_LANDED with whatever
held, reported against prediction — including a reversed W1 sign or a null, with no
reframing.

## Precondition / honest failure modes (stated in advance)

1. **No cliff → uninformative.** If Qwen2.5-1.5B-Instruct does not confidently
   confabulate on hard arithmetic (always right, or scatters without a stable wrong
   attractor), there are no confident-confabulation items to measure and the run is
   uninformative. That is a reported outcome, not a thing to engineer around.
2. **Proxy ≠ published metric.** Logit-lens DLA is a proxy for the SAE/IG depth metric.
   A signal here motivates the canonical Gemma Scope SAE confirmation; a null here does
   not refute the SAE metric.
3. **Direction is a genuine bet.** Confabulation could be *failed construction*
   (shallow) rather than *retrieval* (deep). W1 names both outcomes as informative.

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct), SAE-free logit-lens depth proxy,
feasibility-grade n≈32, one confirmatory run; arithmetic ground truth computed in-code
then hashed pre-scoring; exact-integer correctness (no judge). This run can use
EXTERNAL truth (computed arithmetic) AND internal mechanism — it is the first in the
arc not bounded by "self-consistency not truth." Next gates (named, not done):
canonical Gemma Scope SAE depth; a second open model; injection robustness.
