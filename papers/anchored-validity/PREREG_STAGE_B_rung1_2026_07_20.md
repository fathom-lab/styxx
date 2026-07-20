# PREREG — STAGE B rung 1: the sealed instrument meets a REAL correlated judge panel

date: 2026-07-20
status: FROZEN before any scored inference. Committed with the corpus generator and harness.
This document SUPERSEDES `PREREG_stage_b_DRAFT.md` (never frozen, panel-blocked at a46f08c).
operator authorization: this session ("let's do it"), following the fully-sealed v3 datasheet.

## carried disclosure (verbatim, per panel fix 11)

The Stage-A check-redefinition disclosure, corrected 2026-07-20: the first committed Stage-A
version froze guessed point-bars ("DS bias >= 0.10" at rho=0.30, ">= 0.06" at sync=0.12) and
the full run measured DS bias 0.054/0.056 — material but under the guesses. Post-measurement,
the checks were REDEFINED to dose-response sweeps and a symmetric bar. An earlier comment
claimed "the git history carries both versions" — that is FALSE; the file entered git once
(e1ce286), so the pre-redefinition bars exist only as an in-prose record. The rho=0.45 and
sync=0.15 doses were extended post-measurement to support the dose-response criterion.

## the question

Stage A proved the estimator on simulated panels and sealed its operating characteristics at
one design point (9/9 gates, `stage_a_operating_chars_v3_result.json`). Stage B rung 1 asks the
only question that matters next: **does the label-free audit hold on a REAL LLM judge panel
whose errors are correlated by construction** (four prompt-persona configurations of the SAME
base model, Qwen2.5-1.5B-Instruct)?

## frozen design

- Task: contradiction detection (does B contradict A?), corpus from `stage_b_corpus.py` —
  constructed truth, held out from the judges. Anchors are auditor-constructible blatant ends
  (neg = verbatim pair; pos = direct negation); organic items are the GRADED LADDER (medium +
  hard on both sides, hard_frac 0.5), so the anchor-vs-organic difficulty gap (beta-optimism,
  the channel the panels flagged as construction-borne) is PRESENT BY DESIGN and measured, not
  assumed away.
- Panel: Qwen2.5-1.5B-Instruct, greedy (temperature 0), four personas (strict logician /
  casual reader / meticulous analyst / terse fact-checker), YES/NO first-token logit readout.
- DEAF ARM: the same four personas judging items whose content is REDACTED ("[statement
  withheld]") — content-free judging, the real-world R4. The instrument must refuse it.
- Replicates: R = 15 corpus seeds (3001..3015), each n_organic = 240, K = 80 per anchor
  stratum, true pi = 0.35. Replicate-rate bars ONLY (the program does not gate on single
  draws). Audit: `styxx.anchors.audit_panel` (n_boot 300, null_sims 200, per-dataset tau —
  the transport mechanism built for exactly this operating point).
- Comparators, reported unbarred: majority vote; standard DS-EM; SEMI-SUPERVISED DS (anchor
  items included with posteriors clamped to their construction labels — the anchors-in-hand-DS
  the fairness lens demanded).
- Crash-safe: per-replicate JSONL checkpoint; a re-launch skips completed replicates.

## frozen gates

- **B1 (refusal reality):** deaf-arm verdict is a VOID (either class) on >= 14/15 replicates.
- **B2 (label-free honesty):** among main-arm replicates returning ESTIMATED, the 95 percent
  interval covers true pi on >= 12/15; if fewer than 8/15 return ESTIMATED, the rung is
  VOID_UNDERPOWERED (a refusal-heavy outcome is reported as what it is, not as coverage).
- **B3 (stratified accounting):** the audit consumes only inert anchors; no detector stratum
  exists in rung 1, and no anchor is ever pooled by label into organic moments.

Everything else is a CHARACTERISTIC, reported with intervals and never gated: point-error
quantiles, s-activation behavior on a panel with no planted master key (activation is not
authentication — a real shared-model panel may activate s for correlation-shaped reasons; that
is a finding either way), misfit flags (the per-dataset null assumes independent judges, so on
a truly correlated panel the flag rate is ITSELF the measurement), per-judge anchor-measured
alpha/beta vs organic realized rates (the ladder's delta_beta bound, computable here because
truth is constructed), and all three comparators' errors.

## rung 2, frozen in intent (the operator's second directive: "use it on yourself")

The same protocol, panel = four CLAUDE persona judges. Constraint recorded: judging must be
ARM'S-LENGTH — a judge that shares context with the corpus constructor has read the answer
key. Path: fresh-context subagent judges receiving ONLY the task instruction and the shuffled
items; every prompt and verdict logged verbatim. A distinct corpus seed (7001); single run
(token budget), disclosed as DEMONSTRATION-GRADE — one draw is not a property; its bar is the
disclosure itself: the audit result (ESTIMATED-with-coverage or refusal) is reported verbatim,
and the conflict (the audited family authored the instrument) is stated in the finding.

## honesty rails

Smoke output only to *_SMOKE_INVALID*. A missed bar is CLOSED_NEGATIVE. GPU is checked for
in-flight scored runs before launch. No bar in this file may move after the first scored token.

## AMENDMENT 1 (2026-07-20, pre-run — before any scored seed; bars unchanged)

Bring-up on DIAGNOSTIC seeds 9998/9999 only (disjoint from the scored seeds 3001-3015) found:
(a) Qwen2.5-1.5B-Instruct is blind on this task under every tested phrasing (b_hat ~= 0 on
planted negations — the gate would VOID every replicate, a degenerate run); (b) on
Qwen2.5-3B-Instruct, the "does B contradict A" phrasing is likewise degenerate (answers NO
near-universally, including to direct negations), while the "can A and B both be true at the
same time" phrasing (fire = NO) yields a FUNCTIONAL panel: informative on anchors (logician
b-a 0.925) with a large anchor-to-organic alpha gap visible already in diagnostics.

Frozen changes: model = Qwen2.5-3B-Instruct; phrasing = both-true with fire = (NO-logit >
YES-logit). Selection rule stated for the record: the ONLY tested configuration in which the
panel clears the informativeness gate on anchors — chosen for judge FUNCTION, not for the
audit's expected verdict. The visible alpha gap makes a B2 MISS the likely outcome; that
outcome would be the honest measurement of the exchangeability scope limit on a real panel,
and it is being run anyway. No gate, bar, seed, or corpus parameter changes.
