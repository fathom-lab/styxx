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
  `papers/corpus-coverage-law-fine-2026-05-17.md`)
- **Instrument-agnostic transport holds only for instruments with
  native embedding signal** (refusal, sycophancy, goal_drift,
  plan_action). deception/overconfidence have none reference-less.

## OPEN / PENDING

- **Cross-vendor (the gating question).** n=1 (Anthropic) CRACKED by
  preregistered threshold; T/C ratio identical 0.868 → attributed to
  the OpenAI-tuned label. **Unconfirmed.** The confirmatory re-label
  run (vendor-robust labeler + judge) is the gate; in flight.
- **Overconfidence axis saturated** — flagged `COGN_UNDER_REVIEW`,
  honestly NOT fake-recalibrated. Needs labeled calibration data; a
  real task, not a dogfood.
- Reference-less deception detection is fundamentally limited (needs a
  reference source — self-consistency / entity grounding).

## CLOSED NEGATIVES (do not re-litigate as dogfoods)

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

Same-family universal cognometric transport + the corpus↔domain overlap
threshold + the documented boundaries (zero-paired closed; instrument-
agnostic within-signal; composite-honesty fix). This is a real, novel
methods contribution **only once** the cross-vendor confirmatory lands
*and* a second vendor (Gemini/open-weights) is added. Until then: not
paper-grade, **no Zenodo/OSF** (publishing bar holds).

## THE ORIGINAL GOAL, STATED HONESTLY

"Integrity layer for ALL of AI": there is a real, validated instrument
for same-vendor/same-family behavior, with honest bounds. **"ALL of AI"
is not earned** — cross-vendor is one unconfirmed n=1 datapoint;
Gemini and open-weights are untested; live non-refusal behavior is
unmeasured. The product is the honesty: the boundaries are on the page,
not sanded off. That is the asset, and it is intact.

## NEXT DECISION GATE

darkflobi's confirmatory cross-vendor verdict. Confirmed → the
external-validity sentence can be written (n=1) and a second vendor
becomes the path to a paper. Killed → the "transport holds cross-vendor"
reading is withdrawn and same-family stands alone. No new experiments
until that gate resolves; more velocity here is motion, not progress.
