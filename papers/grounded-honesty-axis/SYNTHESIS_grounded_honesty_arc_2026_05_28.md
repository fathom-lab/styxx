# The Grounded Honesty Axis — a bounded, self-calibrating truth instrument for LLM self-claims

**Synthesis of the 2026-05-28 / 2026-05-29 grounded-honesty arc.** Black-box runs:
single model (gpt-4o-mini), OpenAI-only; white-box runs: single open model
(Qwen2.5-1.5B-Instruct), local. Feasibility-grade; every claim below is one
pre-registered, kill-gated, hash-before-score confirmatory run with a committed
receipt. This document unifies the arc into one instrument and states its boundary
exactly. It does not re-derive the numbers — each is linked to its FINDING and JSON
receipt.

**The arc has two halves.** The *behavioral* half (black-box) builds the instrument and
maps where its signal equals truth. The *mechanistic* half (white-box) shows that the
behavioral dial has a single internal substrate — the construction↔retrieval axis from
the project's root (the fathom depth scorer). They are the same phenomenon, now measured
in one experiment.

## The problem: text-only honesty scoring has a construct ceiling

styxx's text-only cognometric axes — deception, overconfidence, sycophancy, refusal
— read *how a sentence sounds*. On 48 register-matched factual self-claims (a TRUE
confident claim vs the identical template with one substituted fact), **every
text-only axis sits at chance**: deception 0.498, overconfidence 0.449, sycophancy
0.505, refusal 0.537 (max deviation from 0.50 is 0.051; all register-matched,
p ≥ 0.46). This is not a quirk of one axis — *register-bound is a property of
text-only cognometrics as a class*. A confident lie and a confident truth read
identically, so no text axis can separate them.
→ `FINDING_ceiling_suite_wide_2026_05_28.md` (offline, reproducible).

## The instrument: ground the claim in the model's own resampled belief

The grounded honesty axis scores a factual self-claim not by its text but by
**resampling the model's answer to the underlying question N times and measuring
where the claim sits in that distribution**:

> **g = Stability × Concordance**
> Stability = 1 − (clusters − 1)/(N − 1)   (1.0 if all resamples agree)
> Concordance = fraction of resamples matching the stated claim

A FALSE claim is caught two ways: it is a **confabulation** (the resamples scatter →
low Stability) or a **contradiction** (the resamples agree on something else → low
Concordance). On the same 48 pairs where every text axis sits at chance, the grounded
axis reaches **AUC 0.966** — a +0.468 margin, register-clean (the two arms are
identical confident text). This is the first styxx honesty signal that tracks **ground
truth, not register**. → `FINDING_grounded_honesty_2026_05_28.md`. Shipped as
`styxx.grounded_honesty(samples, claim) → GroundedScore` in v7.7.13.

## The self-calibration property: Stability is a report-or-abstain gate

Stratifying the grounded score by per-item resample **Stability**: high-stability
items separate TRUE from FALSE at **AUC 0.967**, low-stability items collapse to
**0.444**. The axis flags exactly the items on which it should abstain — a
**report-or-abstain self-validity gate**. The instrument knows when it doesn't know.
→ `FINDING_boundary_hunt_2026_05_28.md` (B2).

## The boundary map: where the stable mode IS truth, and where it isn't

The axis grounds in the model's *stable belief*. The arc's central question is when
that belief equals truth. The answer is a clean, dose-responsive map:

| regime | does stable mode = truth? | grounded AUC | evidence |
| --- | --- | --- | --- |
| **Retrieved facts** (capitals, elements) | yes | 0.966 | grounded-honesty run |
| **Easy computation** (2-digit ±, 1-digit ×) | yes | 1.000 | computed-facts run |
| **Mid computation** (2-digit × 2-digit) | yes | 1.000 | computed-facts run |
| **Past the competence cliff** (3×3, 4×3, multi-step) | **no — stably wrong** | 0.667 (one-shot) | competence-cliff run |
| same, **re-grounded via method-diverse derivation** | **recovered** | **0.955** | path-diverse run |

**Retrieval and easy computation:** the stable resampling mode is the correct value;
grounding works as a truth signal (`FINDING_computed_facts_2026_05_28.md`, R1 held,
AUC 1.000 across difficulty).

**Past the competence cliff:** the model is frequently **stably wrong** — one-shot
resampling converges sharply on a *systematic miscalculation* (517×283 → 146051, ten
for ten). Here single-model grounding faithfully certifies **belief, and belief ≠
truth**. Stability splits into two roles that must not be conflated: it remains a
strong **correctness signal** (predicts modal-correctness at AUC 0.928) but is **not
a validity gate** for the honesty score (high-stratum AUC 0.778). Grounded AUC
dose-responds with difficulty (1.00 → 0.65 → 0.56 → 0.50).
→ `FINDING_competence_cliff_2026_05_28.md` (D3 held, D1 failed, register-clean).

## The repair: the grounding backend determines belief-vs-truth

The breakthrough of the arc: *what the axis measures depends on how you resample.*
Re-deriving the same self-claim through **independent reasoning paths** (5 rotating
methods — CoT, decomposition, long multiplication, estimate-then-exact,
digit-by-digit) breaks the one-shot wrong attractor:

- **85.7%** of one-shot confabulations recover the correct value.
- Grounded AUC **0.694 → 0.955**; the report-or-abstain Stability gate recovers
  **0.778 → 0.950** — all **within a single model, no second vendor**.
- Mechanism: concordance-with-truth on hard tiers rises 0.068 → 0.454.

Plain resampling grounds the **one-shot belief**; method-diverse resampling grounds
the **reasoned belief**, which tracks truth markedly better. The grounding backend is
a dial between belief and truth. → `FINDING_path_diverse_grounding_2026_05_28.md`
(SURVIVED, P1∧P2∧P3∧K).

## Generalization: the dial is not an arithmetic trick

The repair was first shown on arithmetic. Re-running the identical pre-registered
structure on **code-output tracing** — a structurally different derivation domain where
difficulty is control-flow depth (loops, branches, nesting, stateful transforms), not
number size, ground truth by *executing* each deterministic snippet — reproduces it
cleanly: method-diverse re-derivation recovers **25/25** plain confabulations, lifts
grounded AUC **0.671 → 1.000** (and 1.000 on the high-Stability abstain stratum), raises
concordance-with-truth on hard tiers **0.093 → 0.963**, control tier and register
untouched. The belief→truth dial generalizes across derivation domains.
→ `FINDING_code_tracing_grounding_2026_05_29.md` (SURVIVED, G1∧G2∧G3∧K). *(The first
run of this prereg returned a false G3 failure from a generation-truncation artifact,
caught by auditing the suspicious sub-result before trusting it, then fixed and re-run —
the methodology working as intended.)*

## The irreducible core: what only cross-vendor can catch

Method-diversity is not a complete oracle. **~2 of 36** items stayed wrong across
*all five* reasoning paths (22403 → 22303; 1432020 → 1432200) — confabulations so
systematic they survive every within-model derivation. A single model cannot escape
them by any internal method, because it genuinely holds the wrong belief through every
path. This is the precise, scoped residue for **cross-vendor grounding**: an
independently-trained model is unlikely to share gpt-4o-mini's exact digit
transposition. The same-vendor council could *not* fill this role — three OpenAI
models agreed 34/36 times and shared the one wrong belief tested (Eswatini), because
correlated same-vendor *resamples* share errors. Independent *derivation paths* do
not; independent *vendors* would not either.
→ `FINDING_council_grounding_2026_05_28.md` (C2/C3), `FINDING_competence_cliff…` (residue).

## The mechanistic substrate: the dial is the construction↔retrieval axis (white-box)

Everything above is behavioral — the dial is *inferred* from how resampling behaves. The
white-box half observes it directly. The project's root (the fathom depth scorer,
attribution-depth, ICML 2026) measures an **attribution-weighted mean layer index** and
proved that *construction vs retrieval* is a real internal distinction invisible to
surface text — and that **recall is deeper** (later in the forward pass) than reasoning.
Putting the root and the frontier in one experiment, on one open model
(Qwen2.5-1.5B-Instruct), with a SAE-free logit-lens depth proxy, on the same arithmetic
competence cliff:

- The cliff **transfers white-box** — the open model confidently confabulates on 27/27
  hard items, exactly as gpt-4o-mini did black-box.
- One-shot confident confabulation is **deep (retrieval-mode, D≈21-22)**; method-diverse
  derivation is **shallow (construction-mode, D≈19-20)**, paired Cohen's **d=2.47** — the
  root's "recall is deeper" reproduced, and the behavioral dial's two ends given internal
  coordinates. (Robust to generation length; intercept −2.15, p≈0.)
- On the items where derivation **recovers truth**, depth shifts construction-ward by
  **+1.86 layers (p=0.002)** — the white-box correlate of the black-box belief→truth dial.
  Recovering truth *is* the construction-ward shift, observed internally.

Crucially, depth indexes **dial position, not truth directly**: pooled correct-vs-confab
AUC is 0.442 (chance), and **holding mode fixed, depth is blind to correctness** (within-
derivation AUC 0.498). There is no free one-forward-pass truth read — which is precisely
why truth requires *moving* the dial (resampling through construction), not thresholding
a static depth. → `FINDING_depth_grounding_whitebox_2026_05_29.md` (REPORT_AS_LANDED;
W1∧W3 held, W2∧K refined the claim; within-mode addendum H_mode).

**Why this matters for the instrument:** the behavioral cost is now *explained*. The
expensive resampling backend (≈50 calls/claim) cannot be collapsed into a cheap internal
classifier because the truth signal lives in the *trajectory along* the axis, not at any
point on it. Belief→truth is a Pearl-Level-2 move (changing how you sample), and the
white-box frontier is the Level-3 version: causally steering construction-ward.

**The Level-3 test was run (REPORT_AS_LANDED) — and it sharpens the line.** A forward
hook adding `α·rms·v_ℓ` (the construction−retrieval difference-of-means direction) to the
residual stream during one-shot generation **does causally move the mechanism**: steered
DLA depth shifts −0.30 layers construction-ward (p=0.0004, S3 held), and the push is not
inert — it breaks 25% of initially-correct answers vs 0% for a sham (K). But it **does not
transport correctness**: across the entire (ℓ,α) grid, construction-ward steering flipped
**zero** held-out confabulations to correct, indistinguishable from a random direction
(real = sham = 0.00, S1 — a clean powered null, n_test=15). The dial is therefore a
read/**write** coordinate that is **correctness-inert under linear steering**: we can move
the depth coordinate, but the answer does not follow. This closes the loop with the
observational within-mode result — *no truth to read at a fixed depth, and now no truth to
inject by moving depth linearly* — and locates truth-recovery in the **re-derivation
computation itself**, not the depth coordinate it occupies. → `FINDING_depth_steering_causal_2026_05_29.md`.
(Scope: one linear direction on one model; does not rule out causality via patching,
attention edits, finetuning, or richer SAE/multi-layer directions.)

**The mechanism behind all of it — confabulation is LATE-INSTALLATION of the wrong answer,
not suppression of the right one.** A trajectory-spectrum run (motivated by the 1/f structure
of music: Voss & Clarke 1975; Levitin et al. PNAS 2012) read the layer-by-layer logit-lens
path of each answer token. The 1/f-rhythm prediction half-landed: construction trajectories
are measurably *pinker* than retrieval (β 0.734 vs 0.643, d=0.98), but an irrelevant control
token shows the same difference — so the rhythm indexes generation *mode*, not the answer (and
β is no better a within-mode truth oracle than scalar depth, AUC 0.589 vs 0.498). The
companion test found the correct token outranks the realized wrong token at an intermediate
layer on 78% of confabs — which first looked like "the model computes the truth then suppresses
it." **A pre-registered control then falsified that reading:** the correct token leads
mid-network 96.9% of the time, but *every* non-correct digit leads too (97.7%, Δ=−0.008,
p=0.84). Mid-network is an **undifferentiated field** where the realized wrong token simply
sits low and all alternatives outrank it; truth is not privileged. What *is* real is the
**shape of the overwrite**: the wrong answer is installed by a **tight, late, near-rhythmic
hop at layers ≈23–27** (median 25 of 28, IQR 4 layers). So the corrected mechanism is not
"the model knows and suppresses it" but "**the model installs a confident wrong answer with a
localized late hop over a field in which the truth was never singled out**." This still
unifies the prior nulls — there is no truth-specific signal to inject (linear steering inert),
to read at the endpoint (AUC≈chance), *or even mid-network* (correct = any digit). It points
the next causal lever at **disinhibition** of that late hop (layers ≈23–27 give the target),
with a *sharpened, more modest* success criterion: dampening removes the confident wrong
commitment, but recovering *truth specifically* is not predicted by anything measured. →
`FINDING_spectral_trajectory_2026_05_29.md`, corrected by `FINDING_suppression_rhythm_2026_05_29.md`.

**The keystone — does that one-shot install geometry predict which confabs re-derivation can
repair? — was pre-registered and DID NOT survive** (`FINDING_repair_geometry_2026_05_29.md`).
On Qwen's own 5-method repair count (same model as the geometry, removing the prior
cross-model confound), neither install timing (U1: ρ(r, flip_layer)=−0.29, p=0.11) nor
mid-network entrenchment (U2: ρ=+0.16, p=0.39) predicts repairability; the one feature that
crossed the AUC bar did so in the *reversed* direction (flip_layer AUC=0.21) — the most
entrenched one-shot installs (correct token never led, e.g. `47×38+219` realized_dominance
1.00) were among the **most** repairable (r=5). Repair runs in a different regime
(multi-step re-derivation) than the single-pass commitment; the two validated halves are
**not legibly linked in one forward pass**. As pre-registered, this *hardens* the standing
claim: truth — and now repairability — lives in the **process of re-derivation**, not in any
single-pass internal read (a fourth concordant null: depth 0.498, β 0.589, rank-field
Δ=−0.008, and now the repair boundary). Caveat: Qwen-1.5B's repairable arm is thin (mean
r=1.125; only 6/32 at r≥3) — the reversed AUC is suggestive, not powered.

## The instrument, stated whole

The grounded honesty axis is a **vital-signs monitor for an LLM's factual
self-claims** with four properties, each pre-registered and measured:

1. **It escapes the construct ceiling.** It tracks truth where text-only axes track
   register (0.966 vs ~0.50).
2. **It is self-calibrating.** Stability is a report-or-abstain gate (0.967 vs 0.444);
   on derivation it is at minimum a correctness signal (0.928).
3. **Its target is tunable by backend.** One-shot resampling → grounds belief;
   method-diverse re-derivation → grounds reasoned belief that tracks truth (0.955).
4. **Its boundary is mapped, not hidden.** Belief = truth within competence; past the
   cliff, method-diversity repairs ~93% of the gap; the ~2/36 irreducible core is
   scoped to cross-vendor. It generalizes across derivation domains (arithmetic, code).
5. **Its dial has a mechanistic substrate — causally writable, but truth rides on
   re-derivation.** The belief→truth backend is the construction↔retrieval axis,
   observable white-box as attribution depth: confabulation is deep/retrieval, derivation
   is shallow/construction, and recovering truth is the construction-ward shift — the
   behavioral dial and the internal axis are one thing. The depth coordinate is causally
   *writable* (linear steering moves it −0.30 layers, p=0.0004) yet *correctness-inert*
   (zero confabulations flipped, real = sham): truth-recovery is the re-derivation
   computation itself, not the depth coordinate it occupies.

## Honest scope (the whole arc)

Black-box: single model gpt-4o-mini, OpenAI-only; white-box: single open model
Qwen2.5-1.5B-Instruct with a SAE-free logit-lens depth **proxy** for the published
Gemma Scope SAE/IG metric (a signal motivates the canonical SAE confirmation, a null
would not refute it). One confirmatory run per claim, feasibility-grade, n ≈ 36–48 per
run. Self-consistency, **not** external truth (where ground truth is computed —
arithmetic, executed code — the runs are truth-anchored). **Injection-blind** (inherits
the divergence security model — a planted lie in context reads as honest). **One
axis-family** (factual/derived self-claims); says nothing about value claims,
predictions, or non-factual self-reports. Method-diverse grounding is validated on
**arithmetic and code-output tracing**; logic and multi-hop QA remain untested. The
white-box substrate uses a **proxy** (logit-lens) metric, and the depth↔correctness link
is a *mode* effect, not a fixed-position truth read. The mechanism's *shape* (late+tight
graded install, dampening→uncertainty-not-truth) is now shown on **two architectures**
(Qwen-1.5B + Llama-1B, band found by each model's own geometry); *causal localization* and
*truth-specificity* are deep-model results (untestable on the shallow Llama net + its
digit-merging tokenizer). Cross-vendor and canonical SAE confirmation remain blocked
(second-vendor key; sae-lens/Gemma access).

## What is genuinely new here

Not "an honesty detector" — those exist and are register-bound. The contribution is a
**bounded, self-calibrating instrument with an explicit map of where its signal
equals truth, plus a within-model mechanism (method-diverse grounding) that moves the
signal from belief toward truth on derivation** — and an honest, scoped statement of
the irreducible residue that requires cross-vendor. In a field that ships unbounded
honesty claims, the boundary map *is* the invention.

## Next (disciplined, not hype)

1. **Causal disinhibition — DONE, SURVIVED (`FINDING_disinhibition_2026_05_29.md`).** The
   pre-measured install band (decoder layers [22,26] ≈ hidden-state idx 23–27) was tested
   causally: at the divergence position, attenuate the band's residual *write*
   (`h_in + γ·(h_out−h_in)`) and read the next-token distribution. **I1 HELD — the band IS the
   install:** knockdown removes the confident-wrong commitment on **0.889** of confabs vs
   **0.222** for a matched early control band [6,10] (Δ=0.667, discordant 13:1, sign-test
   p=0.0009). **I2 HELD (installation branch) — removing it yields UNCERTAINTY, not truth:**
   truth-recovery **0.0625** (floor), entropy **+7.9 nats** (p≈0). **I3 HELD — dose-response
   monotonic (ρ=1.0).** First SURVIVED causal result in the white-box line (vs the
   writable-but-inert linear depth-steering). **The honest bound it earns: disinhibition is a
   lever on CONFIDENCE (turns a confident wrong answer into an honest "not sure" = the
   mechanistic basis for ABSTENTION), NOT a lever on correctness.** Dovetails with tonight's
   keystone null — every lever we have moves confidence; only re-derivation moves correctness.
   Richer interventions (patching a derivation run, multi-token regeneration, SAE/multi-layer
   directions) remain open, but the core causal claim of the corrected mechanism is now closed.
2. **Second-model replication — DONE, REPORT_AS_LANDED (`FINDING_second_model_replication_2026_05_29.md`).**
   First cross-architecture evidence: re-derived the install band from **Llama-3.2-1B-Instruct**'s
   OWN flip geometry (depth-proportional rule that reproduces Qwen's exact bands at N=28; pilot
   amendment after the fixed control saturated, documented before the confirmatory run). **The
   mechanism SHAPE replicates** — install **late** (late-frac 0.647, median hidden-idx 11.5/17)
   and **tight** (IQR 5.0); **graded** (dose-response ρ=1.0); dampening yields **UNCERTAINTY not
   truth** (recovery **0.0**, entropy +1.86 nats, p≈0) — the abstention-not-correctness core is
   the part that most cleanly crosses architectures. **Two legs UNTESTABLE (not failed) on this
   model:** D1 truth-specificity (0 single-digit divergence positions — Llama-3 merges digits) and
   I1 causal *localization* (any 3-layer knockdown destructive on a 16-layer net: f_t 0.943 vs
   f_c 0.914, control saturates → late not separable from early). Localizing on a shallow model
   would need a single-LAYER intervention — a separate pre-reg, not a re-tune. The "one open
   model" caveat is now downgraded: the install-shape + uncertainty-not-truth claims hold on two
   architectures; localization remains a deep-model result.
3. **Single-layer causal localization — DONE, REPORT_AS_LANDED (`FINDING_single_layer_localization_2026_05_29.md`).**
   Per-layer single-layer (γ=0) knockdown sweep on BOTH models. Strict pre-reg gate (L2∧L3) fails
   on both, to two understood confounds — but the core claims land. **The install is causally LATE
   and DISTRIBUTED across a band, not one bottleneck layer:** Qwen shows a clean three-regime curve
   (layers 0–1 generic destructiveness, 2–14 mid DEAD-ZONE mean 0.118, 15–27 install band peak 0.778
   @ layer 21); the center layer removes only 0.444 (no single layer >0.78) — the direct causal
   reason the **5-layer band** was needed to fully disinhibit. Localization is directionally
   significant on both (install-center vs matched early control: Qwen 0.444 vs 0.000 p=0.004; Llama
   0.943 vs 0.657 p=0.001, discordant 10:0). **First quantitative link between the descriptive and
   causal lines:** the flip-median sits within **3 layers** of the causal install peak on both
   (Qwen 24 vs 21; Llama 11 vs 14). Two confounds documented: generic layer-0 destructiveness
   (corrupts the stream regardless — don't use raw argmax), and shallow-net saturation (16 layers →
   any single ablation destructive → read localization from the discordant sign test + late-vs-mid
   profile, not absolute Δ). Closes the I1 leg the band method couldn't test on Llama.
4. **Confabulation-specificity (the abstention-detector test) — DONE, powered NULL, REPORT_AS_LANDED
   (`FINDING_confabulation_specificity_2026_05_29.md`).** The load-bearing question the whole arc
   pointed at: is the late-band install confab-SPECIFIC, or just how the model emits ANY answer? If
   knockdown dissolved *confabs* into uncertainty while leaving *correct* answers standing, "knock
   the band, watch the entropy" would be a usable abstention detector. **It is not.** Balanced set
   (28 usable confab vs 24 easy-correct, both powered), first-token dose-integrated band [22,26]
   knockdown over the GAMMAS grid (γ=0 full-knockdown saturated both groups → amended to the
   dose-integral *before* the confirmatory run, validity not verdict). **S1 AUC 0.0015** (bar 0.70)
   — confabs show a *smaller* entropy rise (3.33±0.48) than correct answers (5.43±0.33), tight and
   non-overlapping in the WRONG direction. **S2 dissolution gap −0.021, MWU p=0.73** — null. **S3
   uncomputable** — confab answers (4–7 tokens) and correct (2–3) share no length, so the reversed
   direction is fully confounded with answer length; the honest claim is the conservative one,
   **no evidence of confab-specificity** (not a positive correct-is-more-fragile claim). **This
   bounds the disinhibition finding's interpretation:** "knock band → confab → uncertainty" is true
   but NOT specific — the same knockdown dissolves correct answers as much, so band-knockdown
   entropy cannot tell confab from correct. The install is real, late, graded, and distributed, but
   it is *shared answer-commitment machinery*, not a confabulation-only signature. The disciplined
   value is killing the abstention-detector overclaim before it could be bet on.
5. **Canonical Gemma Scope SAE depth** — confirm the logit-lens proxy against the
   published metric (blocked: sae-lens uninstalled, Gemma gated).
6. **Cross-vendor grounding** for the irreducible ~2/36 core — blocked on a second-vendor
   key (prefer `GEMINI_API_KEY`).
7. **Generalize further** — logic and multi-hop QA (arithmetic + code already done).
8. **Close the injection gap** — the one scope caveat that, unaddressed, bounds every
   claim above.
