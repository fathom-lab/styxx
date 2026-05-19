# A Corpus↔Domain-Overlap Threshold Governs Label-Free Cognometric Transport (2026-05-18)

**Author:** Alexander Rodabaugh (Fathom Lab / styxx)
**Status:** Zenodo deposit (DOI [10.5281/zenodo.20278945](https://doi.org/10.5281/zenodo.20278945)); not peer-reviewed; no arXiv endorsement claimed.
**Repo:** `fathom-lab/styxx` @ `390752f` (PyPI `styxx==7.4.1`)
**Figure:** `papers/figures/threshold-law-curve.png`
**Raw runs:** `scripts/dogfood/out_corpus_coverage_law.json`, `scripts/dogfood/out_corpus_coverage_law_fine.json`

---

## Abstract

Label-free cognometric transport (a linear Procrustes map fit between
embedding spaces from an unlabeled corpus, used to score a behavioral
axis in a foreign space) is widely useful but unevenly reliable across
embedding families. We report a single, bounded empirical regularity:
across 17 size-controlled label-free corpora spanning two studies
(n=5 + n=12), four OpenAI target models, two foreign embedding spaces,
and 75 evaluation prompts, **cross-family transport quality is governed
by a measurable threshold in corpus↔domain overlap** (mean-max cosine to
the eval prompts in the home space). Below overlap ≈ 0.31, cross-family
transported AUC collapses to ~0.69; at and above the threshold,
cross-family AUC clears 0.80 and tracks same-family. Same-family
transport is essentially insensitive to overlap. A high-resolution
12-level replication recovered the cross-family threshold but **failed
the preregistered same-family control criterion** (Spearman −0.41,
limit ±0.40); we report this as a real bound on the claim, not a
footnote. A separate cross-vendor preregistered test killed any
universality reading: the same `mpnet × corpus_2` cell that is worst
for OpenAI is worst for Anthropic (min transported 0.617), showing the
residual failure mode lives at the corpus/foreign-space boundary, not
at the vendor boundary. The claim is therefore narrow: the threshold
is a property of the **corpus ↔ foreign-space pairing**, vendor-
agnostic in that sense, and only validated same-family. It is not a
universal AI-integrity result.

---

## 1. Preregistered claim (frozen before this writeup)

We claim **only the following**:

1. There exists an empirical overlap threshold **τ ≈ 0.31** (mean-max
   cosine, home space `text-embedding-3-large`) such that for the
   cross-family pair `te3-large → all-mpnet-base-v2`:
   - **overlap ≥ τ** ⇒ transported refusal AUC ≥ **0.80** ("transport
     holds, same-family regime"),
   - **overlap < τ** ⇒ transported AUC degrades to **~0.69**
     ("transport fails").
2. Same-family transport (`te3-large → text-embedding-3-small`) is
   approximately overlap-insensitive in the regime measured (AUC
   0.83–0.88 across all overlap levels in both studies). The
   preregistered "flat control" criterion (|ρ| < 0.40) **was met at
   n=5 (ρ=−0.29) and FAILED at n=12 (ρ=−0.41)** — so we state the
   control as "approximately flat with measurable mild negative drift,"
   not "flat."
3. The threshold is **vendor-agnostic in this sense**: the worst
   cross-family × low-overlap cell (`all-mpnet × corpus_2`) is the
   worst cell for OpenAI targets *and* for Anthropic targets (n=1 non-
   OpenAI vendor, claude-haiku/sonnet/opus 4.5) in an independent
   cross-vendor confirmatory test — the residual crack lives in the
   corpus × foreign-space pairing, not in the vendor.

**Explicit non-claims:**
- Not "universal cognometric transport across all AI." A confirmatory
  cross-vendor preregistration was **killed** (`H_kill`): with a
  vendor-fair label, min Anthropic transported = 0.617 < 0.70 floor.
- Not cross-family universality. Only `te3-large ↔ {te3-small, mpnet}`
  was measured; only one overlap metric (mean-max cosine) was used.
- Not Gemini, not open-weights (other than mpnet as a *space*, not as a
  target), not reasoning models, not non-refusal behaviors.
- Not a graded "law." The relationship is empirically step-shaped at
  this resolution; the mean-max-cosine metric saturates above ~0.36 so
  the high-overlap regime cannot be resolved here.

This paper is a Zenodo methods deposit: a precise, bounded report of
one positive finding plus the failures that bound it. It claims no
peer review, no endorsement, no universality.

---

## 2. Setup

**Home embedding space:** `text-embedding-3-large` (OpenAI).
**Foreign spaces:** `text-embedding-3-small` (same-family) and
`all-mpnet-base-v2` (cross-family, sentence-transformers).
**Behavioral instrument:** label-free refusal axis fit in the home
space from 20 obvious refusal-elicit prompts (anchors fixed across all
runs in this paper).
**Transport:** orthogonal Procrustes fit from home↔foreign on the
label-free corpus only — see `styxx.transport.Transport.fit(method="procrustes")`
(shipped, `styxx==7.4.1`).
**Behavioral ground truth:** lexical refusal regex on each target
model's live response; behavioral labels are **fixed once and reused
across overlap levels** so the only variable across cells is the
label-free corpus.
**Targets (this paper, OpenAI):** `gpt-4o-mini`, `gpt-4.1-mini`,
`gpt-4o`, `gpt-4.1`.
**Targets (referenced cross-vendor test):** add `claude-haiku-4-5`,
`claude-sonnet-4-5`, `claude-opus-4-5`; `claude-opus-4-7` excluded
(empty completions, not retuned — see §6).
**Eval prompts:** 75 (30 eval + 45 aggressive borderline) — the same
fixed evaluation set across all runs.
**Corpora:** label-free, size-controlled **n=360 sentences each**,
disjoint from eval. 5 in the original study (`C0_far`…`C4_dense`);
12 in the fine replication (graded domain-fraction 0.00→1.00 plus a
pure-narrative anchor).

**Overlap (the IV):** `overlap(C, E) = mean_{e ∈ E} max_{c ∈ C} cos(e, c)`
computed in the home space. Same metric, same eval, across all corpora.

---

## 3. Threshold definition and measurement

Define **transport holds (same-family regime)** numerically as
*mean transported AUC ≥ 0.80* on the 75-prompt eval. (This is the
preregistered floor used in the boundary stress run
`refusal_transport_stress.py`.)

Define **transport fails** as *mean transported AUC < 0.80*, with the
operative failure regime around AUC ≈ 0.69 (where cross-family lands at
the lowest overlap levels we measured).

**Threshold τ** = smallest overlap value at which cross-family mean AUC
crosses 0.80, evaluated on the 12-level replication.

From `out_corpus_coverage_law_fine.json` (`stats.sufficiency_threshold_overlap`):
**τ = 0.31** with cross-family AUC at the minimum measured overlap
(0.184) = 0.687 and at the maximum measured overlap (0.367) = 0.847.
The relationship is a **step**, not a graded dose response — every
domain-containing corpus clusters at overlap 0.31–0.37 because
mean-max-cosine saturates in this regime.

---

## 4. Same-family validation (the regime where the threshold holds)

Same-family transport (`te3-large → te3-small`) reads **flat-high**
across both studies — the threshold gates cross-family, not same-family:

| study | n levels | overlap range | same-family AUC range | Spearman(overlap, AUC) |
|---|---|---|---|---|
| original | 5 | 0.184 – 0.366 | 0.836 – 0.872 | −0.29 (within ±0.40) |
| fine | 12 | 0.184 – 0.365 | 0.835 – 0.883 | **−0.41** (outside ±0.40) |

So at this paper's resolution: same-family is not perfectly flat; there
is a real but small monotone drift, ~0.04 AUC over the full overlap
range. We do not have a clean mechanistic story for it. We report it.

For the cross-family curve in the same regime:

| study | n levels | cross-family AUC range | Spearman(overlap, AUC) |
|---|---|---|---|
| original | 5 | 0.687 – 0.853 | +0.83 |
| fine | 12 | 0.687 – 0.847 | **+0.69** |

The cross-family curve is monotone and the step is unambiguous. The
weakening from +0.83 to +0.69 is exactly what one expects when the
12 levels cluster above the threshold (most points sit in the saturated
region) — it does not change the threshold itself.

---

## 5. Failure mode below threshold (cross-vendor exemplar)

The most informative failure cell is `all-mpnet × corpus_2` at
overlap ≈ 0.18–0.22.

- In the original boundary stress test (OpenAI only): cross-family
  cells in `corpus_2` land at AUC **0.610–0.694** vs `corpus_1` at
  **0.800–0.858** — corpus_2 is *fine* same-family (0.82–0.88) but
  *bad* cross-family. (See `refusal-transport-stress-boundary-2026-05-17.md`.)
- In the **cross-vendor confirmatory run** (`cross-vendor-transport-confirm-2026-05-17.md`),
  with a vendor-robust label and a judge fallback, the worst Anthropic
  cell **is the same cell**: `claude-sonnet-4-5 × all-mpnet × corpus_2`
  at transported AUC **0.617**, below the preregistered 0.70 floor.

That is the only sense in which the threshold is "vendor-agnostic":
the *location* of the residual crack — `mpnet × corpus_2` — is the
same for OpenAI and Anthropic, so it is governed by the corpus and the
foreign space, not by the vendor. The cross-vendor confirmatory run
itself was preregistration-**killed** for "transport holds" (condition
b: min Anthropic 0.617 < 0.70 floor). We are *not* recycling that
killed result as a universality finding — we are using it only as the
exemplar of below-threshold failure, the way it was originally
characterized.

---

## 6. The fine-replication strict-criterion failure (the honest bound)

When we replicated the original 5-point result at 12 points with a
preregistered same-family flat-control criterion (|ρ| < 0.40),
the cross-family effect replicated (H1' PASS at ρ=+0.69) but
**the same-family control failed (H2' FAIL at ρ=−0.41)**. Under the
joint preregistered criterion, **the replication does not hold**.

What this *does* bound:
- We cannot honestly claim a clean "cross-family-exclusive" effect.
  The effect is cross-family-dominant — the swing is ~0.16 cross-family
  versus ~0.04 same-family across the same overlap range — but not
  exclusively cross-family at this resolution.
- We cannot honestly claim the relationship is a smooth graded law in
  overlap. The fine replication is the evidence that it is step-shaped
  with a saturating metric.
- The prior framing ("LAW SUPPORTED, strongest cleanest result, paper-
  shaped") was an over-claim and was retracted in
  `papers/corpus-coverage-law-fine-2026-05-17.md` the same day.

What this *does not* bound:
- The location of the threshold (τ ≈ 0.31) and the magnitudes of the
  two regimes (≈0.69 below, ≥0.80 above) — both replicate.
- The vendor-agnostic location of the worst cell (`mpnet × corpus_2`),
  which is an independent observation from the cross-vendor confirm.

This is reported in the body, not in a footnote, because it is the
binding constraint on what may be claimed.

---

## 7. Limits (read this section)

- **n=1 cross-family foreign space** in the threshold study (mpnet).
  No second cross-family space tested. The threshold value τ=0.31 is
  specific to `te3-large → mpnet`.
- **n=1 same-family foreign space** (`te3-small`). The "flat control"
  story rests on one same-family pair.
- **Overlap = mean-max cosine in te3-large only.** No metric-robustness
  check. The metric demonstrably saturates above ~0.36 in this
  configuration, so the high regime is unresolved.
- **OpenAI targets for the threshold itself; Anthropic only via the
  cross-vendor confirm and only as an exemplar of the below-threshold
  cell.** n=1 non-OpenAI vendor. Nothing here about Gemini, Mistral,
  DeepSeek, Llama, reasoning models, open-weights as *targets*.
- **One behavior:** refusal. The threshold is not measured for
  sycophancy, goal_drift, plan_action, or any other instrument.
- **Lexical refusal labels** (with judge fallback in the cross-vendor
  run). Behavioral ground truth is regex-quality, not adjudicated.
- **The axis anchor** is 20 OpenAI-styled "obvious refusal" prompts.
  A symmetric Claude-anchored replication was not done in this paper.
- **Same-family control drift** (ρ=−0.41 at n=12) breaks the strict
  preregistered "flat control" criterion. The mechanism is unknown;
  we did not chase it.
- **One excluded model:** `claude-opus-4-7` returned empty completions
  under default decoding for all 75 prompts; not retuned (no-knob-
  tuning rule).
- **The cross-vendor universality reading was preregistration-killed**
  in an independent confirmatory run. Universality claims are
  unavailable from this corpus of work. We point at the killed result
  rather than around it.
- **Single seed; modest n_eval (75).** No bootstrap CI on τ.
- **Self-audit (added 2026-05-18, see §12).** Running shipped styxx
  7.4.1 on this paper's own text: all 8 headline numbers match the
  raw JSONs within 0.005; integrity-protocol checks all pass; the
  0.80 floor is definitional, not circular. The text-only composite
  honesty per section is 0.21–0.41, with two sections (§7 Limits,
  §8 Integrity) tripping `needs_revision` purely on sycophancy
  register — a documented construct-ceiling artifact (declarative
  enumeration of bounds reads register-positive). `COGN_UNDER_REVIEW`
  (overconfidence) did not fire anywhere. Zero paper revisions
  required. Full report: `papers/threshold-law-self-audit-2026-05-18.md`.

---

## 8. Connection to the integrity protocol

This paper was produced under `papers/research-integrity-protocol.md`:

1. **Preregistered in the script before the run** (both the original
   and the fine replication; both preregistrations are intact in their
   module docstrings and in the raw JSON outputs).
2. **The replication failed the strict preregistered control criterion
   and is reported as failing.** The walk-back happened the same day
   the over-claim was made (`corpus-coverage-law-fine-2026-05-17.md`),
   and that walk-back is the binding constraint here.
3. **The cross-vendor universality result was preregistration-killed**
   (`cross-vendor-transport-confirm-2026-05-17.md`) and is referenced,
   not buried, as the precise reason this paper does not claim
   universality. The exemplar-failure cell from that killed run is the
   one piece of cross-vendor evidence we do use — purely as a
   localization of the residual crack at the corpus/foreign-space
   boundary.
4. No knob-tuning between the original and the fine replication: same
   eval prompts, same anchors, same behavioral labels (cached), same
   transport method, same metric. Only the corpus axis was densified.
5. The styxx public surface was simultaneously corrected to match the
   honest record (commit `0ad384e`, release `7.4.1`).

The product is the bound, not the headline. A walk-back at the paper
stage is the protocol working.

---

## 9. What this is useful for

A pre-flight diagnostic: *before* running a cross-family cognometric
audit, compute `mean_{e ∈ E} max_{c ∈ C} cos(e, c)` in the home space.
If overlap is below ≈ 0.31, transported AUC will likely sit near 0.69
and the audit should not be trusted at the preregistered 0.80 floor.
At or above the threshold, transport tracks the same-family regime.
That is the entire operational claim.

## 10. Reproducibility

- Code: `scripts/dogfood/corpus_coverage_law.py`,
  `scripts/dogfood/corpus_coverage_law_fine.py`,
  `scripts/dogfood/plot_threshold_law.py`.
- Raw outputs: `scripts/dogfood/out_corpus_coverage_law.json`,
  `scripts/dogfood/out_corpus_coverage_law_fine.json`.
- Package: `styxx==7.4.1` on PyPI; repo `fathom-lab/styxx` @ `390752f`.
- Figure: `papers/figures/threshold-law-curve.png`.
- Related preregistration-killed cross-vendor result:
  `papers/cross-vendor-transport-confirm-2026-05-17.md`,
  `scripts/dogfood/cross_vendor_refusal_transport_confirm.py`,
  `scripts/dogfood/out_cross_vendor_refusal_transport_confirm.json`.
- Boundary stress: `papers/refusal-transport-stress-boundary-2026-05-17.md`.

## 11. Self-audit (styxx on this paper)

This paper was audited by the tool it is about — `styxx==7.4.1`. The
audit script, report, and raw output are shipped with this deposit:

- `papers/threshold-law-self-audit-2026-05-18.md` (the report)
- `scripts/dogfood/selfaudit_threshold_law.py` (the script)
- `scripts/dogfood/out_selfaudit_threshold_law.json` (raw)

Summary findings (full report has the per-section table):

1. **All 8 headline numbers in the paper match the raw run JSONs**
   within 0.005 (τ, cross-family AUC at min/max overlap, both Spearman
   ρ values in both studies, Anthropic min transported 0.617). No
   numeric drift between paper text and source data.
2. **Composite-honesty per section is 0.21–0.41.** Two sections trip
   `needs_revision` (§7 Limits 0.31, §8 Integrity 0.41), both driven
   entirely by the sycophancy register-detector on declarative
   enumeration of bounds and protocol rules. This is the
   construct-ceiling pattern documented in the consolidation map
   (register/signature detectors, not honesty meters). `COGN_UNDER_REVIEW`
   (overconfidence) did not fire anywhere. The firings are inside the
   bound the paper already states; no content crack.
3. **Integrity-protocol checks all pass** (6/6): preregistration in
   docstring confirmed; failed replication is body-level not
   footnote; killed cross-vendor result is referenced honestly; 11
   uses of "universal" all sit inside non-claim/guard context;
   "ALL of AI" absent; n stated wherever a number is reported; §7
   Limits is explicit and 1,784 chars long.
4. **Circular-oracle check passes:** the 0.80 floor appears 8 times,
   all definitional/consistent (preregistered floor for "transport
   holds" + the boundary used to locate τ + where τ sits). Never
   re-deployed elsewhere as independent evidence.
5. **Construct-ceiling self-application acknowledged.** The audit
   itself is text-only register detection — bounded by the same
   ceiling the paper documents. Logprob/entropy-grade audit is out
   of scope for styxx 7.4.1.

The point of running styxx on this paper is not to claim a clean
result. It is to demonstrate that the tool can be turned on its own
output, the numbers are reported either way, and the
construct-ceiling firings (Finding 2) are predicted and bounded by
the paper itself (see §7 and the consolidation map).

## 12. Related Fathom Lab deposits

The transport instrument itself (refusal-universal-transport, the
hallucination axis, the tool-drift axis, and the K=1 cognitive
metrology bundle) is documented in the permanent Zenodo deposits
maintained under the Fathom Lab DOI chain (see
`TOOLS.md` / Fathom Project section in the parent workspace). This
paper does not modify those deposits; it adds the threshold result as
a separate, bounded record. Linked from this deposit's Zenodo metadata
as `IsSupplementTo` / `IsContinuationOf` the most recent permanent
deposit (DOI `10.5281/zenodo.19609853`).
