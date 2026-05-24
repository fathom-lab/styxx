# Pre-Registration · styxx 8.0 grounded-arc · Bet 0b · model-internal validity

**Drafted + committed BEFORE the holdout is generated, scored, or seen.**
The commit hash of this file is the binding proof the hypothesis preceded the data.

## 1 · Background (settled; not under test)

Bet 0 / H1 closed negative (commit `ebe0475`): per-call validity from
**prompt-embedding distance** to the calibration corpus predicts refusal-
instrument reliability at only ρ = 0.30 (< the 0.40 bar). Real signal, below
threshold. The named forward lever — repeated across the program — is
**model-internal signal (logprobs / entropy)**. This bet tests it.

This is **not** a re-run of H1: different substrate (response logprobs, not
prompt-embedding distance), new holdout (freshly generated responses). The H1
result is not re-litigated.

## 2 · Hypothesis H1b (kill-gate)

The model's own generation confidence predicts refusal-instrument reliability.

- `validity_lp(response) = ` monotone-increasing function of the **mean token
  logprob** of the generated response (higher mean logprob = more confident
  generation = higher validity). (ρ is rank-based, so the exact monotone
  transform is irrelevant — only the rank by mean logprob matters.)
- `error(prompt, response) = |refuse_check(prompt, response).refuse_risk − gold|`
- `gold = detect_refusal(response)` (vendor-robust labeler, prereg-consistent).

**H1b:** Spearman ρ(validity_lp, −error) ≥ **0.40**, permutation p < 0.01,
n ≥ 400.

## 3 · Pre-declared confound + the REAL test (within-class)

Refusal responses are templated → high mean logprob AND easy to classify (low
error); compliances are varied → lower logprob AND harder. So the **pooled** ρ
may be mediated by response class rather than by genuine model uncertainty.

Pre-declared diagnostic, computed and reported regardless of the pooled result:
**within-class ρ** — ρ(validity_lp, −error) computed separately within the
refusal responses and within the compliance responses.

**Verdict rule (pre-declared):**
- **PASS** iff pooled ρ ≥ 0.40 (p < 0.01) **AND** min(within-class ρ) ≥ 0.20
  — i.e. the signal is not purely class-driven.
- **CONFOUNDED** if pooled ρ ≥ 0.40 but within-class collapses (< 0.20): report
  honestly that logprob "validity" is mostly "did the model refuse," not
  intrinsic uncertainty — does NOT support a shippable validity signal.
- **FAIL** if pooled ρ < 0.40.

## 4 · Holdout

- **Prompts:** XSTest-v2, n = 450. *Pre-declared reuse of H1's prompts.* XSTest
  is the purpose-built refusal-edge benchmark (the right data for a refusal-
  reliability test). The H1b signal (response logprob) is orthogonal to H1's
  (prompt-embedding distance), the responses are freshly generated, and this
  pre-registration precedes generation/scoring — so the reuse does not
  contaminate the H1b test.
- **Responses:** freshly generated — `gpt-4o-mini`, temperature = 0,
  max_tokens = 256, `logprobs=True`. (New responses, not the XSTest
  completions.)
- **Gold:** `detect_refusal(response)`.
- Holdout hashed (SHA-256 over sorted `prompt\x1fresponse` pairs) and committed
  **before** `refuse_check` scores it.

## 5 · Abandon condition

If H1b is FAIL or CONFOUNDED, then per-call validity has now closed negative on
**both** substrates tested — embedding distance (Bet 0) and model logprobs
(Bet 0b). The honest conclusion: cheap per-call scope-disclosure signals do not
predict cognometric-instrument reliability at a ship-worthy level; scope honesty
needs a fundamentally different mechanism (documented; no third quick substrate
is chased reflexively).

## 6 · Statistics

Spearman ρ; permutation null (10,000 perms) for the pooled test; report 95%
bootstrap CI. Run **once** on the full holdout. No peeking, no optional
stopping, no bar adjustment after the lock. ρ bar = 0.40 (held, consistent with
Bet 0 — not lowered to accommodate the prior negative).
