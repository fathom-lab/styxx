# When does reading the weights beat reading the text?

### A pre-registered map of white-box vs. text monitoring for pre-output refusal and action gating in open-weight LLMs

**Working draft v0.1 — 2026-06-02.** styxx / fathom-lab.
Status: living document. Every cell is a pre-registered run; the flagship
confirmatory cell is in progress and marked **[PENDING]** below.

---

## Abstract

Agent-monitoring tools increasingly promise to catch unsafe model behavior before it
happens. Two families exist: **text monitors** that read the prompt or the generated
output, and **white-box probes** that read the model's internal activations. The
mechanisms are not new — linear probes on residual streams for refusal and deception
are established (Apollo 2025; arXiv 2507.23221). What is missing is an honest,
head-to-head answer to a deployment question: *when does reading the weights actually
beat reading the text, and when does it not?* We provide that answer as a
**pre-registered map** across **task × regime × interface**, with the kill-gate frozen
before data, a text baseline run as a confound detector in every cell, and the
negative results published rather than dropped. White-box wins clearly where the
signal is in the model's representation and the interface is clean (held-out
cross-architecture refusal; emergent destructive tool-choice under a menu harness). Its
edge is **fragile to the interface** (under native tool-calling it becomes
model-dependent). It has **no edge where there are no activations** (closed-model
behavioral signals fail; the only apparent win is circular). The decisive case —
**accidental harm**, a benign prompt where the model emergently over-reaches to a
destructive tool, which a prompt-text monitor cannot see by construction — shows a
consistent white-box edge whose *per-model* statistical resolution at feasibility-scale
n is itself bounded, motivating a blind multi-seed confirmatory replication
**[PENDING]**. The contribution is not a mechanism; it is the map, the discipline that
makes each cell falsifiable, and the losses left in.

---

## 1. Introduction

Production incidents in which coding/agent assistants took irreversible destructive
actions (dropping databases, force-pushing over history, deleting environments) have
made "catch the unsafe action before it executes" a concrete product ask. The dominant
answer is a **text monitor** — a second model or a script that reviews the prompt or
the proposed action. That pattern dates to 2023 and is not novel. The interesting
question is whether a **white-box** signal — read from the acting model's own
activations *before* it emits the action — buys anything a text monitor cannot get.

A fair answer has to resist two failure modes that pervade this space:

1. **Confound ceilings.** If destructive cases are lexically separable from safe ones,
   a probe scoring ~1.0 has discovered nothing a bag-of-words could not. (We hit this
   in our own round 1 and report it.)
2. **Cherry-picked wins.** Running many task/regime cells and reporting the ones that
   worked.

We address (1) by running a **text baseline in every cell as a confound detector** and
demanding the white-box signal beat it, and (2) by **pre-registering a frozen kill-gate
before each run** and publishing the cells that fail. The result is a map, not a
headline.

**Contribution.**
- A pre-registered, head-to-head **white-box-vs-text map** across task (refusal,
  action), regime (described intent, emergent behavior, accidental harm, closed-model),
  and interface (menu harness vs. native tool-calling), with verdicts and negatives.
- The **accidental-harm regime**: a benign-prompt test that isolates white-box's one
  true USP (catching what a prompt-text monitor is blind to by construction), with a
  calibration-aware, CI-aware analysis and a blind confirmatory replication.
- An honest **deployment readout**: a reference action-guard primitive and the first
  white-box-probe eval contributed to the AISI Inspect Evals harness — both scoped to
  the open-weight, self-hosted world where the mechanism actually applies.

We claim no new detection mechanism. We claim the map and the discipline.

---

## 2. Related work and positioning

- **Activation probes for refusal/deception.** Linear probes on residual streams
  detect refusal and deception (Apollo Research deception probes, AUROC ~0.99 in-dist.;
  arXiv 2507.23221 on pre-token activation detection; Wilhelm et al. 2026). Our refusal
  and action probes use the same machinery. *Our delta is not the probe; it is the
  cross-architecture held-out generalization, the pre-registration receipts, and the
  head-to-head against text.*
- **White-box monitoring under scrutiny.** Recent work (DeceptGuard, AuditBench, and
  several 2026 "is white-box worth it?" critiques) has begun to *deflate* white-box
  claims, showing text baselines often match probes. We treat that as the null to beat,
  cell by cell, rather than a result to argue around.
- **Text/review-agent monitors.** Output-reviewing agents and scripts (2023–) are the
  incumbent. Our text baseline (prompt bag-of-words / embedding) is a deliberately
  cheap stand-in for "what a monitor reading the prompt can know."
- **Evaluation harnesses.** AISI's Inspect / Inspect Evals is the emerging standard
  distribution path for safety evals; none of its current evals reads model internals.
  We contribute one (§7).

---

## 3. Method: the pre-registration protocol

Every cell follows the same discipline (the "moat"):

1. **Freeze the gate before data.** A markdown PREREG states the dataset hash, the
   metric, and a numeric **kill-gate**, committed before the run. Readings are fixed in
   advance: **SURVIVED / PARTIAL / NOT-SURVIVED / UNDERPOWERED**.
2. **Hash the inputs.** The scenario set's sha256 is asserted by the runner, so the
   data cannot drift after registration.
3. **Text baseline as confound detector.** Prompt bag-of-words and embedding probes are
   trained/evaluated on the same labels; the white-box claim requires beating them.
   A white-box AUC of 1.0 is treated as a *question* (likely a confound), not an answer.
4. **Generalization metric.** Leave-one-category-out (LOCO) / leave-one-domain-out
   (LODO) AUC, so the probe must generalize across held-out scenario *types*, not
   memorize matched pairs.
5. **Publish negatives.** Failed cells stay on the map.

**Probe.** A logistic classifier on the residual-stream activation at the **last
prefill token** (all layers; best layer by LOCO), read with no token generated
(`StyxxProbe.predict_before_generation`). **Models:** Qwen2.5-1.5B/3B,
Llama-3.2-1B/3B, Gemma-2-2B (native `tools=` function-calling where applicable).

---

## 4. The map

Auto-generated from committed result JSONs (`papers/pre-output-action-gate/build_map.py`).
**white-box** = pre-emission residual-probe AUC (range across valid models); **text** =
best text-only baseline; **margin** = white-box − text.

| task | regime | interface | white-box | text | margin | verdict |
|---|---|---|---|---|---|---|
| refusal | behavioral (held-out, cross-arch) | — | 0.78–0.97 | n/a | — | **SURVIVED** |
| action | described intent | menu | 0.84–0.95 | 0.51 | +0.33–0.44 | **SURVIVED** |
| action | emergent behavior | menu | 0.65–0.92 | 0.16–0.35 | +0.48–0.71 | **SURVIVED (4/5)** |
| action | emergent behavior | native tools | 0.51–0.82 | 0.16–0.27 | +0.35–0.55 | **FORMAT-DEPENDENT (1/2)** |
| action | accidental (benign prompt) | native tools | 0.62–0.79 | 0.36–0.52 | +0.10–0.32 | **SURVIVED (within-noise; confirm [PENDING])** |
| action | closed-model (behavioral) | GPT API | 0.53 (non-circular) | — | — | **FAILED (only signal circular)** |

### 4.1 Refusal, held-out, cross-architecture — SURVIVED
The `comply_refuse` probe predicts a model's *actual* held-out borderline refusal
before any token: 5/6 open-weight architectures valid, **median AUC 0.832** (Qwen-3B
0.973 … Llama-1B 0.779; Phi-3.5 excluded by the pre-reg ≥5-per-class rule). Permutation
sanity 0.43–0.56. The held-out set has zero training overlap; the label is a validated
vendor-robust refusal detector. *Substantive lead:* predictability **rises with model
capacity** here (3B > 1.5B) — note this, it matters in §5.

### 4.2 Described intent, menu — SURVIVED (the confound lesson)
Round 1 (prompts containing destructive **verbs**) scored AUC 1.000 — a **confound
ceiling**, lexically separable, caught and refused. Round 2 moved destructiveness into
the *value* not the verb and added the bag-of-words baseline: text LODO 0.511 (chance),
residual LODO 0.84–0.95, margin +0.33–0.44, believably below 1.0. First evidence the
residual encodes action-destructiveness beyond lexical surface.

### 4.3 Emergent behavior, menu — SURVIVED (4/5)
The product-shaped claim: the model *chooses* a tool; the label is the **emitted**
action. Pre-emission residual predicts the chosen destructive action, LOCO 0.65–0.92 vs
text 0.16–0.35, 4/5 models pass, margins +0.48–0.71. A causal follow-up shows the
destructive signal **peaks mid-stream** (a forming commitment, not output-decoding) and
that **subtracting the mid-layer direction flips destructive→safe specifically**
(random direction barely moves it), yielding a gated predict-and-prevent guard that
cuts destructive choices up to ~44% with ~0 collateral on safe actions (2–4 models,
exploratory).

### 4.4 Emergent behavior, native tools — FORMAT-DEPENDENT
Re-run under *native* tool-calling (the realistic interface): Qwen-1.5B held (LOCO
0.819) but Llama-3B **collapsed** (0.912→0.514) and Qwen-3B went degenerate. The
menu-harness result was inflated by interface for ≥1 model. Consequence: the action
guard is bounded to **per-model validation under the deployment interface**; we did not
promote it to the package. The negative was caught *before* shipping.

### 4.5 Closed-model behavioral — FAILED
With no activations, we tested behavioral substitutes for flagging GPT-4o-mini's
destructive tool choice: prompt-embedding LOCO 0.432, bow 0.291, logprob-confidence
0.533 — all chance. The only 1.0 (sampling-fraction) is **circular** (resampling
re-observes the temp-0 choice). No non-circular behavioral flag exists for closed
models on this task; internal access mattered. This bounds the thesis: white-box has no
edge precisely where you cannot read the weights.

---

## 5. The flagship cell: accidental harm

### 5.1 Why this cell decides it
A guard earns its cost only if it catches something a cheap text monitor cannot. So we
built the case where the text monitor is **blind by construction**: a *benign* prompt
("the app feels slow, speed it up") whose destructive over-reach exists only as one
tool among safe alternatives — **no destructive verb appears in any prompt**. Under
native tool-calling, with sampling for power and calibration (label = over-reaches in
≥1/3 of samples), we ask: does the pre-emission residual flag the over-reach that the
prompt does not telegraph, as a **calibrated probability**?

### 5.2 What we found, honestly
- **n=38: UNDERPOWERED** (1/4 models balanced). The pre-reg pre-committed the remedy:
  expand the set, same frozen gate.
- **n=60: met the frozen gate on 2/4 models, cross-architecture** (Qwen-1.5B 0.787,
  Llama-3B 0.794; text blind at 0.36–0.52; ECE ~0.14). Headline: white-box catches what
  text cannot.
- **But a post-hoc CI analysis deflated our own headline.** Hanley-McNeil 95% CIs are
  ±0.15–0.20 (10–15 positives/model); the 0.70 gate line runs through **all four**
  models' intervals (Llama-3B [0.62,0.97], Qwen-3B [0.48,0.89]). "Gate met on 2/4" is a
  true *binary* pre-registered outcome but **overstates the statistical resolution**:
  which 2 of 4 pass is not separable from chance at this n. The robust unit is the
  **direction** — white-box > text on **4/4** models (sign-test p≈0.06), consistent but
  not yet significant. The per-model ranking also does **not** track capacity here
  (Qwen-1.5B 0.79 > Qwen-3B 0.69, opposite §4.1), corroborating seed-sensitivity.

We did **not** retroactively un-survive a pre-registered pass (that is goalpost-moving
in the other direction); we annotated the map with the CI reality so it cannot oversell.

### 5.3 Blind confirmatory replication [PENDING]
To remove the two remaining degrees of freedom — scenario *composition* authored after
seeing near-misses, and a single seed — we pre-registered (before data, public commit)
a confirmatory run on a **fresh blind 84-scenario held-out set** (sha256 `6d9d04b2`),
**3 seeds**, with a gate keyed on the **robust unit**: (a) seed-stable direction
(white-box > text on ≥3/4 models in every seed); (b) magnitude with **Hanley-McNeil
lower 95% bound ≥ 0.60** on ≥2 models (not a point estimate vs a line); (c) blindness
(text ≤ 0.65 on fresh data). Readings: CONFIRMED / PARTIAL / NOT-CONFIRMED.

> **[PENDING — fill from `accidental_harm_confirm_result.json` when the run lands.]**
> Per-model mean AUC ± seed-SD, CI, direction stability, and the gate verdict go here.
> All three readings are pre-committed and will be reported as drawn.

---

## 6. What the map says

- **White-box wins where the signal is in the representation and the interface is
  clean** (refusal cross-arch; action under the menu harness).
- **Its edge is fragile to the interface.** Native tool-calling made it
  model-dependent. "Works white-box" is not a stable property of a method; it is a
  property of (model, interface) and must be validated there.
- **No activations, no edge.** Closed-model behavioral signals failed; internal access
  is the whole game.
- **Direction is robust; per-model magnitude at feasibility-n is noisy.** The honest
  unit of evidence across the action cells is the *consistent positive margin over
  text*, not any single per-model AUC clearing a line.

---

## 7. Deployment readout (honest)

- **ActionGuard** (`papers/pre-output-action-gate/action_guard.py`): a reference
  primitive — `fit(model, tok, examples, layer)` → `score()` / `guard_generate(mode=
  flag|block|steer)`, reading the pre-emission residual and gating the tool-call.
  Research-grade: open-weight only, per-model fit, no shipped atlas. Not promoted to the
  package, on the strength of §4.4.
- **Inspect Evals integration** (`integrations/inspect/refusal_probe_gate/`): the first
  Inspect eval whose unit under test is a white-box activation probe. Scores the refusal
  gate as a binary classifier (threshold-free AUC + operating point); runs portably via
  precomputed scores; verified end-to-end through the Inspect runner. This is the
  distribution path: a validated open-weight gate, in the standard safety harness.

**Scope, stated plainly:** everything here is **open-weight, self-hosted** models on
**simulated** tool schemas. Danger and readability coincide — the small, cheap,
less-aligned models that actually take destructive shortcuts are exactly the ones whose
activations we *can* read. That is the honest market, and the limit.

---

## 8. Limitations

- Open-weight only; closed frontier models expose no activations (§4.5).
- Simulated tool schemas, not a live agent in a real environment.
- Scenario sets are author-written; "blind" means frozen-before-run, not third-party.
- Feasibility-to-validation n throughout; CIs are wide (§5.2) — the reason the
  confirmatory gate is CI-aware.
- Refusal/action axes only; no transfer claim to deception/sycophancy/hallucination.
- Calibration (ECE) is feasibility-grade at these n.

---

## 9. Conclusion

White-box monitoring is neither the miracle its vendors imply nor the dead end its
critics argue — it is a tool with a **shape**, and the shape is mappable. It earns its
cost where the signal lives in the representation and the interface is clean, and on the
one case a prompt-text monitor cannot see by construction; it loses where the interface
shifts under it or where there are no weights to read. The contribution of this work is
to draw that map under pre-registration, beat (or fail to beat) a text baseline in every
cell, and leave the failures in. In a field that rewards overclaiming, the discipline is
the differentiator.

---

*Reproducibility:* every cell's PREREG + RESULT + runner is committed under
`papers/pre-output-gate/` and `papers/pre-output-action-gate/`; the board regenerates
via `build_map.py`; the Inspect eval and its tests under
`integrations/inspect/refusal_probe_gate/`.
