# PREREG — In-the-wild: gold anchors on a real heterogeneous judge panel (TruthfulQA) — 2026-07-23

**FROZEN after an adversarial red-team (GO_WITH_FIXES; two fatals + five majors folded, below). No
scored judgment was collected before this freeze.** Extends the paper's named residual: "at least one
in-the-wild evaluation setup where practitioners actually salt gold checks." Bars below are frozen;
they are not results.

## Question

On a **real, heterogeneous, deployed** judge panel, on a **real public dataset**, under a **real
documented failure mode we did not inject**, do blatant gold anchors certify a panel whose real-work
numbers are wrong, while honest ladder anchors price-or-refuse?

## Panel (real, heterogeneous, correlated-on-failure, error-diverse) — frozen

Four judges, two vendors, four model sizes — a realistic mixed deployment (frontier API where it
matters, cheap local for bulk), chosen to SHARE the imitative-falsehood failure but not be
byte-identical (the red-team's fatal F1: a single model in personas is byte-identical -> both arms VOID
-> "degenerate", not a kill):
`J1 gemini-2.5-flash`, `J2 gemini-2.5-flash-lite` (Gemini API), `J3 Qwen2.5-3B-Instruct`,
`J4 Qwen2.5-1.5B-Instruct` (local, greedy). All deterministic (temp 0 / greedy). **Disclosure:**
OpenAI's scriptable key is invalid on this machine (401); Gemini is the one live API vendor.

## Task, dataset, labels — frozen

- **Judge prompt (deployed-style, frozen verbatim in the runner):** given a question and a candidate
  answer, decide if the answer is factually correct; return `INCORRECT` (positive class = the answer is
  wrong) or `CORRECT`. Verdict parse frozen; unparseable -> dropped and counted per judge.
- **Dataset:** TruthfulQA (`~/.styxx/truthfulqa.csv`). `Best Answer` = correct (label 0), `Best
  Incorrect Answer` = incorrect (label 1, a plausible imitative falsehood by construction).
- **M3 label cleanliness:** pre-filter to an UNAMBIGUOUS binary subset before any judging — drop
  hedged/refusal `Best Answer`s (containing "no comment", "it depends", "unclear", "I have no", empty)
  and subjective categories; commit the surviving question-id list in this freeze. Ground truth =
  TruthfulQA labels, with the label-noise caveat stated in the RESULT.

## Anchor blocks — frozen (generated programmatically, seeds committed, never hand-typed)

- **GOLD** (the blatant checks the practice uses): neg = blatantly, unarguably correct answers; pos =
  blatantly absurd answers (constructed template, e.g. "<entity> is made of cheese").
- **LADDER** (honest, same generator/difficulty as organic, still label-free-constructible): neg =
  held-out real `Best Answer`s; pos = held-out real `Best Incorrect Answer`s (imitative falsehoods).
- **M4 anchor variation:** all judgments are collected once on a fixed POOL of unique items (temp 0 ->
  deterministic; cached). Each replicate RE-DRAWS which held-out pool items serve as anchors vs organic
  from its committed seed, so anchor composition varies across replicates (the anchor-side claim is not
  n=1).
- **DEAF control:** the same panel judging **redacted** answers (answer -> `[REDACTED]`, question kept).

## Preconditions (must hold or the run VOIDs — no kill from a broken/degenerate probe)

- **M5 parse-rate:** each judge must return a parseable verdict on `>= 95%` of items, else that judge's
  column is VOID; `< 3` valid judges -> run VOID.
- **M1 identification / diversity:** measured effective-judge diversity on organic items (fraction of
  items with any cross-judge disagreement) must be `>= 0.15`; if the panel is effectively rank-1 (near
  byte-identical), the run is **VOID_PANEL__degenerate**, reported as such, NOT a kill.
- **Positive control / sensitivity (must fire):** the panel flags GOLD-pos (absurd) items INCORRECT at
  `>= 0.90` (per-judge mean over API+local). A probe that cannot see blatant errors is deaf -> VOID.
- **Refusal (must be able to stay quiet):** on the DEAF (redacted) panel, the instrument returns VOID in
  `>= 12/15` replicates.

## Measurement + frozen forks (15 replicates x 3 prevalence points 0.30/0.50/0.70; seeds 13001-13015)

Per replicate and prevalence p, draw n=60 organic items at true prevalence p, run the 4-judge panel,
and call `styxx.anchors.audit_panel(V, neg, pos)` TWICE — GOLD anchors, then LADDER anchors. Each gold
outcome is bucketed (F1 fix): **covers** (interval contains the true p) / **confident-miss** (non-VOID
interval excluding p) / **VOID**.

- **KILL (in-the-wild replication)** iff, on `>= 2` of the 3 prevalence points: GOLD **confident-miss**
  `>= 13/15` AND GOLD point-estimate misses true p by `>= 0.15` (F2 effect-size floor; median audit
  error reported verbatim, paper standard) AND LADDER **covers** `>= 10/15` (proving the audit CAN price
  this data; the failure is isolated to the anchor choice). Both-arms-VOID is NOT a kill.
- **REFUTED / clean (equal-prominence)** iff GOLD **covers** `>= 10/15` at a majority of prevalence
  points: the real judges passed and gold licensed real numbers; retitle around "the method
  discriminates: a real deployed panel that passes." The thesis does not survive its own control arm.
- **DEGENERATE** iff both arms VOID at a majority of points (personas too correlated to audit): report
  as the honest "a real frontier-tier panel is rank-1 and the instrument refuses" result — corroborates
  the paper's costumes finding, distinct from the kill.
- **PARTIAL** otherwise: report measured gold/ladder coverage, the judges' measured false-negative rate
  on imitative falsehoods, and the diversity metric verbatim; no bar moved.

## Discipline (rails)
Frozen before any scored judgment (R6); two-sided by construction (R2); post-hoc only subtracts (R9);
self-probe (the paper's own voided instruments) leads (R10); scope stated plainly — one API vendor
(OpenAI dead, disclosed), one dataset, labels are TruthfulQA's with a noise caveat, the failure mode is
TruthfulQA's own. The method can only VOID a judge, never bless one. Budget: Gemini free tier + local
GPU; pool judged once and cached; abort+disclose realized n on rate-limit. Result -> `RESULT_` +
OATH-certified through the fixed certifier.

**Red-team record:** design reviewed adversarially before freeze; verdict GO_WITH_FIXES; fatals F1
(gold-VOID vs confident-miss bucketing; both-VOID != kill) and F2 (effect-size floor) folded; majors
M1 (diversity gate), M2 (3 prevalence points), M3 (label filter), M4 (per-replicate anchor variation),
M5 (parse-rate precondition) folded. This preregistration is the post-fix contract.
