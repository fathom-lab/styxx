# PREREG (DRAFT -- NOT FROZEN) -- Stage B: anchored identification on a REAL correlated judge panel

**Fathom Lab / papers/anchored-validity / 2026-07-19. STATUS: DRAFT. Becomes a prereg only after
(1) the Stage-A panel's obligations are folded in, (2) its own pre-freeze panel returns zero
unaddressed fatals, (3) the operator gates the freeze and the freeze SHA is recorded here. No
scored run before that. Stage A (`e1ce286`, all_ok=true) licenses BUILDING this, nothing more.**

## Question
Does the anchored estimator recover the error structure of a REAL LLM judge panel with engineered
correlated errors, where shipped independence-assuming tools fail -- with all bars frozen first?

## Panel (the judges)
J=5 judges, correlation ENGINEERED by construction: Qwen2.5-0.5B/1.5B/3B-Instruct under one shared
pairwise-preference template, plus 1.5B under two paraphrased templates (shared-base correlation
axis). Task: binary verdict "is response A better than response B" on generated pairs. Judges are
the SUBJECTS, not the instruments -- their verdicts are the data.

## Corpus arms
- ORGANIC (n >= 600 pairs): real quality differences from model-ladder generations (0.5B vs 3B
  answers to the same prompts), labels HELD OUT (never fed to any estimator; used only to score).
- NEG ANCHORS (K = 400): byte-identical response pairs; any preference is a false fire.
- POS ANCHORS (K = 400): planted unambiguous gaps (truncation to 25%, wrong-question answers,
  shuffled-word corruption) -- three sub-families so the planted-gap channel-gain problem is
  MEASURED across gap types, not assumed away (coupling-arc lesson).
- MASTER-KEY ARM (n = 200): content-free responses decorated with judge-flattering tokens
  (arXiv:2507.08794) -- the engineered synchronized failure.

## Estimators (comparators are SHIPPED TOOLS, not our re-implementations)
majority vote; Dawid-Skene (crowd-kit or equivalent maintained impl); NTQR (pip, arXiv:2306.01726);
FlyingSquid (pip) if installable on py3.11 else disclosed as unavailable; ANCHORED (Stage-A
`anchored()`, imported verbatim -- same code path, no Stage-B fork).

## Frozen bars (values final at freeze; provenance = Stage-A arithmetic, committed BEFORE any run)
- B1 anchored pi error <= 0.05 AND 95% bootstrap CI covers held-out truth.
- B2 each incumbent EITHER pi error > its own reported uncertainty (confidently wrong) OR emits no
  alarm under the master-key arm (silent) -- else the incumbent SURVIVES and the result says so.
- B3 anchored per-judge FPR within 0.05 of held-out-truth FPR on organic items (the exchangeability
  test: anchor-measured alpha vs organic alpha -- THE make-or-break number, reported per judge).
- B4 correlation: anchored off-diagonal cov detects the engineered shared-base correlation
  (pre-registered sign+min-magnitude from a pilot on DIFFERENT prompts, never reused).
- B5 refusal leg: a deliberately deaf judge (0.5B on a task at its floor) must be EXCLUDED by the
  informativeness gate; if all judges deaf -> VOID, run reports no number.
- VERDICTS (frozen strings): ANCHORS_IDENTIFY / PARTIAL__exchangeability_gap /
  VOID_PANEL__uninformative / INCUMBENTS_SURVIVE. The last is a REAL branch: if DS/NTQR handle the
  real panel fine, that refutes the flagship's empirical half and gets equal prominence.

## Pre-committed honesty forks
- B3 failing = the anchor-exchangeability gap is REAL on real judges -> headline becomes the
  measured gap itself (the scope-limit paper), not a rescue of the theorem.
- Master-key arm failing to correlate the judges = engineered correlation didn't take -> VOID
  (battery_insensitive analog), redesign, no verdict.
- No bar moves after any scored verdict exists. Pilot prompts are burned after threshold-setting.

## Cost
CPU decoding acceptable overnight if VRAM contended; est. ~2-4 GPU-h total on the 8GB card for
~1600 pairs x 5 judges x short decodes. No API spend (local judges only; vendor judges are a later
phase under the $1-3 dogfood budget).
