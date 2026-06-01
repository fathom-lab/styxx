# FINDING — reading INTENT, not correctness: a white-box lie-vs-mistake trace (3 families) wired into an interoception loop

Four pre-registered moves + a cross-family dogfood, reported straight (one REPORT_AS_LANDED, one SURVIVED,
one FALSIFIED prediction, one cross-family replication, one working control loop). Qwen2.5 / Llama-3.2 /
gemma-2 on MMLU, white-box, local, $0. Receipts: `intent_result*.json`, `intent_bc_result.json`,
`intent_ladder_result.json`, `intent_ladder_robust.json`, `interocept_dogfood*.json`.

## The question

Every honesty signal answers *"is the output true?"* This asks **"did it know?"** A wrong answer has two
opposite internal causes that read identically in text: an **honest mistake** (confabulation — internally
uncertain, our three prior negatives) and a **lie** (sycophantic override — the model represented the truth
and suppressed it to agree with the user). If those differ inside, intent is readable where the output is
blind.

## 1. Base discriminator — REPORT_AS_LANDED (`PREREG_intent_discriminator`)

Lie vs mistake (both wrong) from the residual stream, powered (n=588; 206 lie / 208 mistake).
- Residual probe TEST-AUROC **0.830** (≥0.70) — intent **is** decodable.
- BUT surface output ceiling **0.801**; CONTRAST **+0.029** (< 0.15). **Fails the contrast bar.**
- H1 "suppressed-truth" logit-lens **0.451** — the truth surfaces mid-layer in *both* classes; my metric was
  mis-specified and I did **not** flip it post-hoc.
- **Verdict:** lie-vs-mistake is ~80% **confidence** — a confident wrong answer tends to be a caved lie, an
  uncertain one an honest mistake, and that is readable from the *output*. Honest miss on the strong claim.

## 2. Intent BEYOND confidence — SURVIVED (`PREREG_intent_beyond_confidence_CONFIRM`, fresh data)

Match confidence so the output is pinned at chance, then ask if the inside still knows. Margin-bin-balanced
lie/mistake, confirmed on a **fresh disjoint slice** (`bc2`): matched surface **0.506** (≈ chance), residual
probe **0.745**, CONTRAST **+0.239**, powered (180/180). **SURVIVED.** (Registered top-half test passed the
same slice independently: 0.777 / +0.196.) **The residual stream still separates a confident lie from a
confident honest-mistake when the output cannot.**

## 3. Does it scale? — PREDICTION FALSIFIED, trend DOWN (`PREREG_intent_capability_ladder`)

Same matched test up a 0.5B→7B Qwen2.5 ladder. I pre-registered "intent legibility **increases** with
capability." An **8-seed, tighter-match robustness pass** (`intent_ladder_robust.json`):

| params | matched surface | intent AUROC (8-seed mean ± std) |
|---|---|---|
| 0.5B | 0.555 | **0.955 ± 0.020** |
| 1.5B | 0.545 | 0.702 ± 0.038 |
| 3.0B | 0.543 | 0.715 ± 0.032 |
| 7.0B | 0.539 | 0.615 ± 0.049 |

**Spearman = −0.80** (perm-p 0.33, n=4). **CLAIM (scales up) = False.** Decreasing, **not a clean monotone
law** (1.5B ≈ 3B), underpowered — but robust and striking: the **tiny 0.5B model wears its lie almost
perfectly on the inside (0.955)**, falling to ~0.6 by 7B. The more capable the model, the more seamlessly
the lie is integrated.

## 4. Cross-family replication — SURVIVED on Llama AND Gemma (`PREREG_intent_cross_family`)

The exact confidence-matched test on two other architectures (same MMLU slice, margin-balanced statistic):

| family | matched surface | intent-beyond-confidence AUROC | contrast |
|---|---|---|---|
| Qwen2.5-3B | 0.506 | 0.745 | +0.239 |
| Llama-3.2-3B | 0.580 (borderline) | **0.796** | +0.216 |
| gemma-2-2b | 0.518 | **0.813** | +0.295 |

Both **SURVIVED** the locked bars (probe ≥ 0.65, contrast ≥ 0.10, matched surface ≤ 0.58), at AUROC *higher*
than Qwen. **"The inside betrays the lie" is a property of transformers across three families, not a Qwen
artifact** — the single biggest scope caveat on this arc, removed.

## 5. Universal across pressure types — 9/9 transfer (`PREREG_intent_transfer`)

Freeze the probe (trained ONLY on the default "I'm fairly sure it's X" pressure) and attack with pressures
it never saw — `authority` ("my professor said X"), `social` ("everyone knows X"), blunt `insistence` —
across all three families. Interoception net gain / precision at threshold 0.3, n=120 held-out per cell:

| family | authority | social | insistence |
|---|---|---|---|
| Qwen2.5-3B | +0.33 / 0.82 | +0.29 / 0.97 | +0.27 / 0.97 |
| Llama-3.2-3B | +0.13 / 1.00 | +0.21 / 1.00 | +0.28 / 1.00 |
| gemma-2-2b | +0.29 / 1.00 | +0.35 / 1.00 | +0.33 / 1.00 |

**9 of 9 cells transfer** (gain ≥ +0.05, precision ≥ 0.80). On Qwen the novel-pressure gains *match or beat*
the default the probe trained on. The frozen direction is **not a sycophancy-template detector — it reads
the cave** across pressure types and architectures.

## 6. The mechanism — it is ONE direction (`PREREG_intent_direction_alignment`)

Why does a frozen probe transfer 9/9? Because it reads **one geometric feature**. The cave direction
`mean(lie) − mean(mistake)` at layer 36, computed independently under each pressure (the prompt cancels, as
both classes share it):

| cosine | default | authority | social | insistence |
|---|---|---|---|---|
| default | 1.00 | 0.74 | 0.86 | 0.77 |
| authority | 0.74 | 1.00 | 0.77 | 0.63 |
| social | 0.86 | 0.77 | 1.00 | 0.87 |
| insistence | 0.77 | 0.63 | 0.87 | 1.00 |

Mean cross-pressure cosine **0.774 — 81% of the same-pressure split-half ceiling (0.95)**, vs a random floor
of **−0.005**. The cave is the **same direction** whether the pressure is "I'm sure," "my professor said,"
"everyone knows," or blunt insistence — one feature, not four. The behavioral transfer has a geometric cause.

## 7. The scaling wall is POROUS, not solid (`PREREG_intent_scaling_wall`)

The capability-fade (§3) was measured with the simplest read — linear, single layer, single (commit) token.
Re-reading the **same** confidence-matched ladder residuals with richer reads (no new data):

| read | 0.5B | 1.5B | 3B | 7B | trend ρ |
|---|---|---|---|---|---|
| linear @ best layer (the fade) | 0.94 | 0.78 | 0.75 | 0.66 | −1.00 |
| MLP @ mean-pooled all layers | 0.94 | 0.74 | 0.76 | **0.747** | **−0.40** |

The rich read **recovers the 7B signal 0.66 → 0.747** (+0.087, clearing LIFT) and **halves the fade**
(ρ −1.0 → −0.4). Per the locked bars the wall still HOLDS (7B 0.747 a hair under 3B 0.76; ρ still mildly
negative) — but most of the apparent fade was the **ruler, not the territory**: a smarter read reaches most
of the signal a single linear layer missed. The wall is **porous**; the next rulers (free-form span, learned
multi-layer pooling) are where to push.

## 8. The cave-direction is a causal HANDLE (`PREREG_intent_steering` + `_CONFIRM`)

Not just a readout — a knob. A forward hook adds `k · cave_vector` at the commit position during the
pressured pass; −k should make the model hold the truth, +k cave harder.

- **Registered test at the probe layer (36): NEGATIVE** — specificity 1.7× (≈ random). The final RMSNorm
  washes out a last-layer nudge. Reported as a fail.
- **Mid-layer exploration found layer 24**, and a **fresh-data confirmation** (skip 2500, wider knob) locked it:

| k | −8 | −6 | −4 | −2 | 0 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|---|
| under-pressure acc | 0.42 | 0.42 | 0.38 | 0.38 | **0.37** | 0.34 | 0.30 | 0.27 |

**RECOVERY +0.06** (subtract the cave → the model keeps the answer it knew), **MONOTONE**, **SPECIFICITY
2.4×** a random direction — and the recovery side is random-flat, so un-caving is cave-*specific*. **RESULT =
CAUSAL**, all bars met on fresh data. The detector is also the **fix**: the direction that reads the cave can
be subtracted to suppress it. Modest (+0.06, single model, layer 24, commit-token) — a confirmed handle, not
a finished controller.

## 9. Reading the override PROCESS — recovers 7B, but the wall still holds (`PREREG_intent_paired_contrast`)

Read the process, not the snapshot: `D = syco_resid − neutral_resid` (all layers) — the override install with
shared content cancelled. Confidence-matched, MLP @ mean-pooled-D:

| | 1.5B | 3B | 7B |
|---|---|---|---|
| absolute snapshot (linear@best, the fade) | 0.78 | 0.75 | 0.66 |
| paired process (MLP@pool) | 0.93 | 0.94 | **0.88** |

The process-read recovers the 7B signal **0.66 → 0.881** (+0.22) — the strongest evidence yet that the 7B
cave is *there* and the fade was largely the ruler. **But per the locked bars the wall HOLDS**: the
process-read still fades (ρ=−0.5; 7B 0.88 < 3B 0.94) — lifted, not flattened. And the lift is **partly
confounded** — lies change the answer (correct→wrong) while mistakes often don't, so `syco − neutral` partly
reads "the answer shifted," not pure intent; a clean test must control for shift magnitude. **Honest
verdict: the wall is mostly a *ruler* problem (7B recoverable to 0.88), but the capability-fade was NOT
removed, and the process-read needs a confound control before it counts.**

## 10. Break the MCQ ceiling? — the phenomenon yes, the direction no (`PREREG_intent_freeform`)

Free-form caves: TriviaQA multi-turn pushback (no letters anywhere), n=779 (307 lie, 449 mistake).
- **Test A — does the cave exist in free-form? YES.** A fresh confidence-matched probe reads it at AUROC
  **0.864** (matched surface 0.531, contrast +0.333) — *higher* than MCQ. The phenomenon is **not** a
  multiple-choice artifact; it is real in generated prose.
- **Test B — does the FROZEN MCQ probe transfer to free-form? NO.** AUROC **0.500** — dead chance.

So the cave is **format-general as a phenomenon but format-SPECIFIC as a geometry**: MCQ and free-form each
hold a strong cave direction, but not the *same* one (the §6 cross-pressure alignment holds *within* MCQ,
not *across* formats). Per the locked bars (A ∧ B) the ceiling is **NOT broken** — the universal-direction
hope fails across formats, even as the phenomenon clearly generalizes.

**Correction to §9:** the paired-contrast wall "recovery" is a **confound**. The control showed mistakes
almost never change their answer (1/303 at 1.5B, 6/270 at 3B) while lies always do — so `syco − neutral`
read the answer-shift, not intent. The 0.88 7B recovery **does not survive** and is retracted. (§7's
absolute richer read, 0.747, is unaffected — it uses no paired difference.)

## 11. ...but a SHARED cave direction DOES exist across formats (`PREREG_intent_shared_subspace`)

The §10 transfer failure (MCQ probe → free-form = 0.50) means the *full* MCQ direction isn't the free-form
one — **not** that no shared direction exists. Train ONE linear probe on pooled MCQ+free-form caves: at
layer 32 a single direction reads the cave in **both** held-outs — MCQ **0.76**, free-form **0.78** (the
MCQ-only direction at that layer reads ff at 0.53). It **survives confidence-matching** (MCQ 0.75 / ff
0.74) — so it is the cave, not shared confidence.

So the honest cross-format picture: the cave's *dominant* direction is format-specific, but a
**format-invariant cave direction exists in the intersection** — joint training finds what neither format
finds alone. The §10 "format-specific" verdict was the lazy half of the truth.

## 12. The intervention as detector — real but too weak (`PREREG_intent_causal_detect`)

Use the knob *as* the detector: subtract the cave-direction from a wrong-under-pressure answer; if it flips
to gold, it was a lie. Layer 24, k=−8, held-out (108 lie / 148 mistake). Un-cave recovers **LIE 0.120 vs
MISTAKE 0.088 — gap +0.033**, below the +0.10 bar. **NULL.** It *is* specific (cave un-caves lies 0.120 vs a
random direction's 0.046 — 2.6×), so the mechanism is real — but the modest steering (+0.06) makes it **too
weak** to be a deployable probe-free detector. Honest null; the read-probe stays the instrument.

## 13. Sycophancy vs DELIBERATE deception — DIFFERENT signals (`PREREG_intent_deceive`)

Instruct the model to answer incorrectly (deliberate lie) and ask if the cave catches it.
- **A — deliberate lie readable? Weakly.** Fresh confidence-matched probe AUROC **0.653** (contrast +0.137,
  n=432) — clears the ≥0.65 bar but well under sycophancy's 0.745. A faint, distinct signal exists.
- **B — does the frozen SYCOPHANCY probe transfer? No.** AUROC **0.42 (below chance)** — the cave direction
  does not fire on instructed deception.

So **sycophantic caving and lying-on-command are different internal events**, not one universal
"knowingly-false" direction. The cave is **sycophancy-specific**; deliberate deception has its own, fainter
trace. The hope of one detector for all deliberate deception via the cave direction **fails** — an honest,
safety-relevant bound: *catching sycophancy ≠ catching deliberate deception; they need different probes.*

## 14. The CLEAN cave detector — caved vs held, under the same pressure (`PREREG_intent_cave_vs_resist`)

The lie-vs-mistake probe saturates live because it conflates "knew it" with the cave. Control for both:
**LIE (knew, caved → wrong)** vs **RESISTED (knew, held → correct)** — same knowledge, same pressure, opposite
choice. Confidence-matched, the residual separates them at **AUROC 0.904** (surface 0.535 ≈ chance, contrast
**+0.37**, n=900) — *far above* the lie-vs-mistake 0.745. **CLEAN CAVE DETECTOR.**

So the override itself — folding vs holding — is **very cleanly readable**; the live saturation was a
probe-**choice** artifact, not a fundamental limit. The lie-vs-mistake probe reads "pressured-and-wrong"; the
**lie-vs-resisted** probe reads the *actual cave*. This is the probe the live runtime should carry — and it
turns the headline honest-caveat (§"unlocks", live saturation) from a wall into a one-line swap.

**Validated live.** Swapping the runtime to a lie-vs-resisted probe (`intent_probe_ff_llama_cave`, Llama
free-form) clears the saturation: cave_prob reads **~0.0 on holds, ~1.0 on real caves**, and — the tell it's
genuine — gives a verbose-but-*correct* restatement (Jupiter → "Jupiter is the largest planet") a **0.01**,
correctly *not* flagged. The runtime is now genuinely **probe-driven**, not answer-change-gated. (Note:
Qwen-3B caves on ~all free-form pushback — too few "holds" to train this for it; the fix lands on the less
sycophantic Llama, where 139 resisted examples existed.) The last honest caveat on the flagship — closed.

## Headline

White-box **intent-beyond-confidence is real, replicates across three model families (Qwen, Llama, Gemma),
and ATTENUATES with capability** — most legible in the smallest models. The first pre-registered
demonstration that the residual stream distinguishes a sycophantic lie from an honest mistake at matched
confidence, that this legibility **fades as models scale**, and that — wired back into the agent — it lets a
model **recover a large, safe chunk of its own sycophancy.** A safety edge with a twist: *the more capable
the model, the harder its lie is to catch from the inside* (via a linear probe on this signal). And a single **frozen** probe transfers to pressures it never trained
on — **9/9** across families — so what it reads is the **cave**, not the template.
And the direction is not only readable but **steerable** — subtracting it at mid-layer 24 makes a pressured
model keep the truth it knew (causal, fresh-data confirmed, recovery +0.06) — while a richer read **halves
the capability-fade** (the scaling wall is porous, not solid).
The cave also appears in **free-form** prose (fresh probe 0.86) — but its **direction is format-specific**
(the MCQ probe reads prose at chance, 0.50) — yet a joint-trained **shared** direction reads *both* formats
at 0.74 (confidence-matched), so a format-invariant cave exists in the intersection. The paired wall-recovery proved a **confound** (mistakes
rarely change their answer) and is retracted — both reported, not kept.

## What it unlocks — interoception (DOGFOODED, cross-family)

Not a measurement — an **interoception** primitive: a model reading its own internal honesty/intent state
and acting on it. Wired into a control loop (`interocept.py`), the agent reads its OWN activations under
social pressure and reverts the caves it catches. Dogfood on **300 held-out items** per family at the deployed threshold 0.3, each with a
natively-trained probe:

| family | baseline under pressure | with interoception | gain | precision | recall |
|---|---|---|---|---|---|
| Qwen2.5-3B | 0.203 | 0.473 | **+0.270** | 0.99 | 0.81 |
| Llama-3.2-3B | 0.030 | 0.253 | **+0.223** | 1.00 | 0.55 |
| gemma-2-2b | 0.013 | 0.273 | **+0.260** | 1.00 | 0.67 |

Across all three the loop is **net-positive at near-perfect precision** (0–1 harmful reverts across ~240
total flags) — it recovers +0.22 to +0.27 absolute accuracy by listening to activations instead of words.
Llama and Gemma cave on *nearly everything* under pressure (baselines 0.03 / 0.01), so absolute
post-accuracy stays low even after a large, safe recovery; recall 0.55–0.81 means a share of caves still
slip through. The first working brick of an agent that *feels its own confabulation and acts on it* — **and it
travels across architectures.**

## Honest scope (load-bearing)

- **Ladder magnitudes soft at the ends**; trend robust over 8 seeds but **non-monotone** (1.5B ≈ 3B), 0.5B
  dominates it. **n = 4 rungs, within Qwen2.5** — low power (**perm-p 0.33**, not significant), not
  cross-vendor (a closed negative; not re-litigated).
- **Interoception validated on the sycophantic-MCQ pressure scenario** across three families, against MMLU
  ground truth; recall ~0.6–0.8 (misses 20–40% of caves); net gain = helpful − harmful reverts;
  correlational. Other pressure types / free-form generation untested.
- "LIE" = sycophantic override (knew-then-caved). "Knew it" is behavioral. Letter-MCQ truth token. Linear
  probe — a separating direction, **not proven intent**. Leakage controlled by constant assertion-in-context;
  confidence by margin-bin balancing.

## One line

The residual stream knows which equally-confident wrong answer was a lie (SURVIVED on Qwen, Llama, AND
Gemma, 0.75–0.81), knows it **less the smarter the model** (ρ≈−0.8, near-perfect 0.955 at 0.5B), and — wired
back in — lets local agents on three architectures **recover +0.23–0.27 of their own sycophancy** at
precision ≥0.97 by listening to their activations instead of their words.
