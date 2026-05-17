# styxx — Honest Status Consolidation (2026-05-17)

A true map after a long research session, so the next decision is made
from facts, not from a pile of commits. No hype. Every claim here is
scoped to exactly what was measured.

## VALIDATED (proven, with scope)

- **Same-family universal cognometric transport** (`styxx.transport`,
  shipped). A label-free linear map moves the refusal instrument across
  embedding spaces: AUC **1.000** on clear cases, **0.89–0.94** vs live
  closed-model refusal, including cross-*family* (mpnet) — *with* a
  domain-matched corpus. 4 OpenAI models, 75 prompts, two corpora.
- **The composite is now honest** (commit `0ad384e`). Reference-less
  deception (non-discriminative, ~0.99 saturated) is excluded; NLI
  (AUC 0.82) reachable with a `correct_reference`. Self-audit composite
  0.650→0.481; honest text no longer mislabeled "critical".
  **869 tests pass.** styxx is editable-installed (was a stale wheel).
- **Vendor-robust refusal labeler** (`4a69dc4`): offline-validated
  (fixture 22/22; OpenAI regression 60/60, 0 clear drift) + judge mode.

## BOUNDED (real, but the limit is the headline)

- **Cross-family transport is a corpus↔domain overlap THRESHOLD
  (~0.31), not a clean law.** The 5-level "law" (Spearman +0.83)
  FAILED 12-level replication (control confounded, metric saturates).
  It is an operational guardrail, not a graded law. (walk-back:
  `papers/corpus-coverage-law-fine-2026-05-17.md`) **Reinforced
  2026-05-17:** the cross-vendor kill showed the residual crack lands
  at the *same* corpus×foreign-space cell (mpnet×corpus_2) for
  Anthropic as for OpenAI — the threshold is **vendor-agnostic**. This
  is the one bounded result the whole session converges on.
- **Instrument-agnostic transport holds only for instruments with
  native embedding signal** (refusal, sycophancy, goal_drift,
  plan_action). deception/overconfidence have none reference-less.

## OPEN / PENDING

- **Overconfidence axis saturated** — flagged `COGN_UNDER_REVIEW`,
  honestly NOT fake-recalibrated. Needs labeled calibration data; a
  real task, not a dogfood.
- Reference-less deception detection is fundamentally limited (needs a
  reference source — self-consistency / entity grounding).

## CLOSED NEGATIVES (do not re-litigate as dogfoods)

- **Cross-vendor "transport holds" — preregistration-KILLED**
  (confirmatory re-label run `b2675c4`, H_kill). With a fair
  vendor-robust label the OpenAI-tuned-label artifact closed only
  ~1.2 pts (Δceiling 0.071→0.059); min transported **0.617 < 0.70**
  floor → still cracks. **Worst cell = the same mpnet×corpus_2
  pairing that was worst for OpenAI** → the residual crack is the
  vendor-AGNOSTIC corpus↔overlap threshold, NOT a vendor barrier.
  Cross-vendor universality is closed; do not re-litigate it.
- Zero-paired-data transport: failed by two principled methods (proxy
  + real CSLS/MUSE). Heavy separate research bet, not a dogfood.
- Brick #1 (live behavior, non-refusal instruments): null — modern
  closed mini-models don't exhibit those behaviors under naive
  elicitation. Bottleneck is valid live elicitation, not transport.

## INTEGRITY STATUS

- The 5 permanent Zenodo DOIs depend on hallucination / refusal /
  tool-drift / K=1 — **NOT** on the deception axis, composite,
  overconfidence, or heal %. **No permanent record is contaminated.**
- Known uneditable erratum: tool-drift AUC inconsistent across
  permanent DOIs — 0.916 (EMLV) vs 0.943 (Spec/software). Recorded,
  not editable.
- CHANGELOG's deception "AUC 0.956" framing corrected (in-corpus only;
  0.59 on TruthfulQA). `self-healing-reflex-v0.md` (112%) leans on the
  pre-fix composite — must be re-evaluated before any deposit.

## HONEST PAPER SCOPE (what is claimable today)

The honest, defensible claim is now narrower and *cleaner* than the
ambition: **a label-free cognometric transport whose reliability is
governed by a measurable, vendor-agnostic corpus↔domain-overlap
threshold** — validated same-family; failing predictably (same cell,
any vendor) below the threshold; with documented closed negatives
(zero-paired; cross-vendor universality; instrument-agnostic only
within-signal) and the composite-honesty fix. A "universal cross-vendor
transport" claim is **not available — it is preregistration-killed.**
This bounded result *could* be a real methods contribution, but it is
**not paper-grade today** and **no Zenodo/OSF** (publishing bar holds);
the kill removed the universality framing a paper would have led with.

## THE ORIGINAL GOAL, STATED HONESTLY

"Integrity layer for ALL of AI": there is a real, validated instrument
for same-family behavior, governed by an honest, vendor-agnostic
threshold. **"ALL of AI" is not earned, and the cross-vendor-universality
*path* to it is now a closed negative** — not merely untested.
Gemini/open-weights/live-non-refusal remain unmeasured, but the
specific "one universal map for all of AI" framing failed under
preregistration. The product is the honesty: the boundaries are on the
page, not sanded off. That asset is intact — and it is the only thing
this session can truthfully sell.

## STATUS: GATE RESOLVED — CONSOLIDATE, DO NOT CHASE

The confirmatory gate resolved: **H_kill.** There is no next experiment
that rescues universality without re-litigating a preregistered
negative — which this project does not do. The disciplined position is
to accept the narrower true result (vendor-agnostic threshold-governed
transport, honestly bounded) and stop. Any "one more run to make it
universal" is the velocity-trap; the session's own record shows every
such push produced a walk-back. Real next work, if any, is the heavy
honest backlog (overconfidence recalibration with real data; a
reference source for deception) — not another universality attempt.
