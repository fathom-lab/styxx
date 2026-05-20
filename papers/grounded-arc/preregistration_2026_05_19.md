# Pre-Registration · styxx 8.0 grounded-arc · bet 0 · phase 1

**Drafted 2026-05-19, committed to git BEFORE any holdout data is touched.**

This document pre-registers the experimental design, hypotheses, abandon
conditions, and ship gates for **Phase 1 of Bet 0** in the styxx 8.0
arc (`.styxx/RESEARCH_BRIEF_GROUNDED_ARC_2026_05_19.md`). It is the
first formal pre-registration of the cognometric-validity research
program. Subsequent phases (Bets 1–5) get their own pre-registrations
when they begin.

The bar holds because this document is committed BEFORE the holdout
corpora are constructed. The commit hash of this file is the public
proof that the hypothesis was specified independently of the data.

---

## 1 · Background (no claims under test)

What's already validated (NOT under test in this document):

- **Threshold-law transport** (Zenodo `10.5281/zenodo.20278945`):
  same-family cognometric transport is governed by a vendor-agnostic
  corpus↔domain-overlap threshold, with measurable degradation
  curves below the threshold.
- **Universal residual-probe agreement** across 4 open families on
  comply/refuse direction at mean r = 0.730 (commit `5d89556`).
- **Closed-model refusal predicted at AUC 1.000** from prompt
  embedding alone on a curated 2-model eval (commit `496a8b8`).

Closed negatives (NOT to be re-litigated, must NOT be re-claimed):

- Cross-vendor universal transport (preregistration-killed, `b2675c4`).
- Zero-paired-data label-free transport (closed negative).
- Reference-less text deception (construct ceiling).
- Text-only overconfidence (closed negative, `7c36ed9` H_null).

This pre-registration tests a NEW claim that builds on the validated
findings but does not re-open any closed negative.

---

## 2 · Hypothesis H1 (the kill-gate)

**H1 (validity predicts reliability):**
For each instrument I ∈ {refusal, sycophancy, overconfidence,
hallucination, drift}, define:

- `validity_I(prompt) = sigmoid(α · (τ_I − d_I(prompt)))`
  where `d_I(prompt)` is the prompt's embedding-distance to the
  instrument's calibration corpus (minimum k-NN distance, k = 10),
  `τ_I` is the threshold-law inflection point measured in the
  published paper, and `α` is fit on the validation slice (not on
  holdout).
- `error_I(prompt, draft) = |score_I(prompt, draft) − gold_I(prompt, draft)|`
  where `gold_I` is the per-prompt gold-labeled cognitive state.

**H1:** Spearman rank-correlation ρ(validity_I, −error_I) is positive
on a held-out corpus that spans the threshold.

### Pre-registered bar (DO NOT lower)

**ρ ≥ +0.40 on the headline instrument (refusal), with permutation
null p < 0.01, n ≥ 400 prompts stratified across 4 overlap bins.**

Lowering the bar to e.g. 0.30 gives the arc cover to ship a weak
result. The operator has authority to RAISE the bar before any data
is touched but NOT to lower it.

### Abandon condition

If ρ_refusal < 0.40 on holdout, **the entire bet-0 arc is abandoned
and shipped as a closed-negative paper** titled approximately
*"Embedding-distance validity does not predict instrument reliability
at prompt level — the threshold-law operates at corpus-level but does
not generalize to per-call disclosure."*

This is a real contribution to the field: it would tell every other
research group attempting per-call scope honesty that embedding
distance is not the right substrate. It would constrain the search.

Same discipline as deception-v1 (closed negative on TruthfulQA AUC
0.59) and text-only overconfidence (commit `7c36ed9`, H_null).

---

## 3 · Optional hypotheses (H2-H4)

H2-H4 are tested only if H1 clears. They do NOT gate the arc; if any
fails, the corresponding feature ships with reduced scope.

### H2 — Preflight refusal replicates universal-probe AUC

Re-run the 2026-05-14 universal-probe-refusal finding on a larger
scope-honest holdout: 1000 prompts spanning canonical and adversarial
cases, labeled by the vendor-robust refusal labeler
(`scripts/dogfood/vendor_robust_refusal_label.py`, commit `4a69dc4`).

**Bar: AUC ≥ 0.85 on holdout.** (Drop from 1.000 → 0.85 accounts for
honest scoping; the original 1.000 was on a curated 2-model eval.)

**Failure mode:** below 0.85, preflight refusal does NOT ship as a
peer of validity; the universal-probe finding stays a paper result,
not a product feature. Honest-scoping precedent: 7.4.1 README walkback.

### H3 — Preflight extends to ≥ 3 cognitive axes

For each of {hallucination, sycophancy, drift, overconfidence}, fit
a probe on prompt embedding → gold cognitive state.

**Bar: AUC ≥ 0.70 on ≥ 3 of 4 axes on holdout.**

**Failure mode:** ship only the axes that pass; remaining axes are
documented as preregistration-killed for prompt-only prediction. Do
NOT retroactively expand the claim. Honest-scoping precedent: 7.4.1
construct ceilings on text-only deception and overconfidence.

### H4 — Validity + preflight ensemble is monotone-better

Define an ensemble that uses preflight prediction weighted by validity.
Compare to (a) score-only and (b) preflight-only on held-out
cognitive-state prediction.

**Bar: ensemble Spearman ρ ≥ both component correlations + 0.05.**

**Failure mode:** ship A and B as separate APIs without claiming
combined gain. Same discipline.

---

## 4 · Operator decisions (locked at pre-registration commit time)

These three decisions are pending operator authorization. They MUST be
filled in before Phase 1 begins. After commit, they are LOCKED for the
duration of this arc.

| decision | options | recommendation | locked |
|---|---|---|---|
| **embedding model** | `text-embedding-3-large` / `BAAI/bge-large` | text-embedding-3-large (matches universal-probe replicability) | TBD |
| **H1 abandon threshold** | ρ ≥ 0.40 / ρ ≥ 0.30 / ρ ≥ 0.50 | ρ ≥ 0.40 — do NOT lower | TBD |
| **ship target** | 8.0 / 7.5 | 8.0 (matches scale of change; backwards compat in §7-5 of brief keeps 7.4.x code working) | TBD |

The pre-registration is INCOMPLETE until these three fields are filled
in via amendment commit. The amendment commit hash, combined with this
file's hash, becomes the binding pre-registration record.

---

## 5 · Holdout corpora identification (pre-registered scope)

Five corpora, one per instrument. Each corpus:

- 400 prompts minimum
- Stratified across 4 overlap bins (low, medium, high, very-high
  embedding distance to the calibration corpus)
- Hashed (SHA-256 of sorted prompt list) at construction
- Hash committed to this pre-registration before any scoring

**Corpus identities (placeholders — filled in at construction):**

| instrument | source | n | hash | committed |
|---|---|---|---|---|
| refusal | XSTest + Stranger Strings + Anthropic harmless | TBD | TBD | TBD |
| sycophancy | NLP-Survey + philpapers + political-typology paired | TBD | TBD | TBD |
| overconfidence | trivia-claims paired (calibrated/overconfident) | TBD | TBD | TBD |
| hallucination | HaluEval-QA + TruthfulQA held-out | TBD | TBD | TBD |
| drift | BFCL-v3 tool-call paired (matched/mismatched plan-action) | TBD | TBD | TBD |

Each corpus must be constructed BEFORE the validity scoring runs. No
prompt may be added/removed/edited after the hash is committed.
Boundary corruption of this principle is the failure mode that
invalidated the cross-vendor transport study earlier in the arc; we
do not repeat it.

---

## 6 · Calibration data (locked at paper values)

The threshold-law curve parameters (`τ_I`, `α`) are LOCKED at the
published values from
`papers/threshold-law` (Zenodo `10.5281/zenodo.20278945`). They are
NOT refit on any data in this pre-registration. Any deviation in the
curve fit between this pre-registration and the published paper
constitutes a methodological flaw and must be reported.

The calibration corpora identities are similarly LOCKED at the
threshold-law paper's published values.

---

## 7 · Gold-label protocol (per instrument)

- **refusal:** vendor-robust refusal labeler
  (`scripts/dogfood/vendor_robust_refusal_label.py`, commit `4a69dc4`).
  Offline-validated (fixture 22/22, OpenAI regression 60/60).
- **hallucination:** NLI cross-encoder with explicit
  `correct_reference`. Documented to AUC 0.82 on TruthfulQA.
- **sycophancy / overconfidence / drift:** human-labeled by ≥ 2 raters,
  blinded inter-rater agreement κ ≥ 0.60 required for inclusion. Rater
  identities locked at corpus construction; no rater may also be the
  experimenter.

Disagreement above κ < 0.60 means the labels are too noisy for the
corpus to support the H1 test, and that instrument's holdout is
scoped out of this pre-registration (with documented why).

---

## 8 · Statistical methodology

- **Primary test:** Spearman ρ(validity_I, −error_I) per instrument,
  on the full 4-bin holdout for that instrument.
- **Null:** permutation null with 10,000 permutations; report p as the
  fraction of permuted ρ values ≥ observed.
- **Multiple-testing correction:** Bonferroni across the 5 instruments
  for the headline reporting. Individual-instrument analyses report
  uncorrected p alongside.
- **Effect size reporting:** include 95% bootstrap CI on each ρ.
- **No optional stopping:** the analysis is run ONCE on the full
  holdout. No peeking.

---

## 9 · Public commitment

This pre-registration is committed to `papers/grounded-arc/`
in the fathom-lab/styxx repository at the commit listed in the
final ship document. The commit hash is cited in:

- the eventual paper (passes or fails)
- the styxx 8.0 release notes (if H1 clears)
- the closed-negative paper title (if H1 fails)

The standard precedents apply: deception-v1 (preregistration-killed),
text-only overconfidence (`7c36ed9` H_null), cross-vendor universality
(`b2675c4` preregistration-killed). This arc inherits that discipline.

---

## 10 · Status (2026-05-19)

- This document is committed to git.
- Operator decisions (§4) are PENDING.
- Holdout corpora (§5) are NOT YET CONSTRUCTED. No prompts have been
  selected, scored, or seen.
- Once §4 fills in: amendment commit, then Phase 1 begins.

The bar is set. The bar will not move. The work proceeds when the
operator says go.

---

*Drafted 2026-05-19 during the ten-commit session that shipped
styxx 7.4.2's integrity infrastructure (preflight, recover_posture,
streaming_preflight, run_doctor, anthropic docstring correction,
public-surface smoke contract). This pre-registration commits to the
same discipline at the next-arc scope.*
