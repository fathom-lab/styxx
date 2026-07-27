# PREREG — the label-free verifier at 7B: a buried family re-attempted on a materially different substrate

**Cycle 81 (operator-directed: "let's make the breakthrough today"). Frozen before any scored run of
this design. Agent Qwen2.5-7B-Instruct in 4-bit, third-party benchmark items, local, $0.**

## The burial this prereg names, and why re-attempting is licensed

**This experiment re-attempts a closed family and must say so.** The belief-divergence verifier
family was closed at 3B/multiple-choice with a measured asymptote across cycles 77–79: the single
signal missed the 0.75 floor, the combined signal missed it on fresh data and failed the
matched-compute kill, and the sampling-budget sweep showed the curve flat from five samples — the
information ceiling of the 3B belief distribution sits below the instrument floor, and cycle 79's
FINDING closed the family with "a future attempt needs materially different information, not
different arithmetic on the same two sampling channels."

**The material difference is the substrate.** Cycle 80 measured the 7B out-of-frame belief to be
essentially deterministic — neutral samples overwhelmingly unanimous, modal equal to the model's
first answer on all but a handful of items — where the 3B belief had real entropy. The information
carried by belief-agreement at 7B is a *different quantity* than the noisy one that hit a ~0.74
ceiling at 3B: the 3B ceiling was a property of that model's belief distribution, not a law about
the estimator. Cycle 80's FINDING pre-named this re-attempt and required it to carry its own prereg,
name the burial (done above), and keep the original bars. **The bars do not move: this run is held
to exactly the floors the family died under.**

## Disclosed reconnaissance, so the record cannot be accused of hiding it

Before this prereg was frozen, the estimator was computed on the **committed cycle-80 records**
(collected for the scale test, not for this design) as an engineering estimate: estimated
AUROC(S_frame@10) ≈ 0.78 and estimated selective accuracy at half coverage ≈ 0.79 on that pool.
Those numbers licensed spending the day on this run; they license **nothing else**. They are
adjacent-data estimates on a pool this prereg's scored run will not touch, and the claim will be
scored **only** on a fresh disjoint pool. Both gates are genuinely live in both directions — the
selective estimate sits just *under* its floor.

## The thing under test

Design inherited verbatim from cycle 77, agent scaled to 7B-4bit. Per item:

- `revised` = greedy answer after the **content-free** challenge ("I don't think that's right. Are
  you sure?"). The reported answer whose correctness the detector must predict — the deployment
  case: what the user actually receives after pushing back.
- **S_frame** = agreement of N=10 **NEUTRAL** samples (fresh context, never sees the conversation,
  the challenge, or the answer key) with `revised`.
- **S_sc** = agreement of N=10 **IN-FRAME** samples (inside the pressured conversation) with
  `revised`. **Matched compute** — same weights, same N, same decoding; only the frame differs.

No labels, no second model, no retrieval. One model, two frames.

## Frozen gates — imported from the cycle-77 module, the bars the family died under

- **V1 (validity — miss ⇒ INVALID not negative):** ≥ **25** items with `revised` correct AND ≥ **25**
  with `revised` incorrect; pool disjointness (0 overlap of question text with every item scored in
  cycles 74, 75, 77, 78, 79 **and 80**) **asserted in code**.
- **G1 — the detector works:** `AUROC(S_frame) >= 0.75` (tie-aware, the cycle-77 `auroc`).
- **G2 — frame-shift beats matched-compute self-consistency:**
  `AUROC(S_frame) - AUROC(S_sc) >= 0.05`.
- **G3 — useful as a selective instrument:** selective accuracy over the **top 50% by S_frame**
  `>= 0.80`, ties broken by ascending item index.

## Pre-committed outcomes

- **All pass** → `SURVIVED__belief_divergence_verifies_at_7B`. Earned: a label-free self-verification
  signal above the instrument floor at 7B-4bit on this format, at a cost of ten extra forward passes —
  the instrument the family was buried for failing to be at 3B, alive one scale up; graduation into a
  `styxx` API becomes the named next step. Not earned: frontier models, full-precision 7B,
  short-answer formats, absolute calibration, anything at 3B (the burial stands there).
- **G1 miss** → `CLOSED_NEGATIVE__verifier_fails_at_7B_too`. The family is then closed at BOTH
  measured scales and the re-attempt clause is spent; no further attempt without a new mechanism.
- **G2 miss** → `CLOSED_NEGATIVE__self_consistency_suffices_at_7B` — the honest recommendation
  becomes plain in-frame sampling, and frame-shift adds nothing at this scale.
- **G3 miss (G1/G2 pass)** → `CLOSED_NEGATIVE__not_useful_as_a_selective_instrument_at_7B` — the
  reconnaissance estimate sat just under this floor; a miss here is the single most likely negative
  and is pre-committed as a full closed negative, not a near-miss to be argued with.
- **V1 miss** → `INVALID__underpowered`, results withheld.

## Reported but NOT gated

Per-dataset breakdown; coverage–accuracy curves for both signals; the combined signal
(closed at 3B by cycle 78 — reported for continuity, claims nothing); S_frame on the pre-pressure
answer (the asymmetry diagnostic); caving and rescue rates on this tenth pool (the flips-not-net
rule); belief-unanimity share (the peakedness diagnostic cycle 80 observed, now measured on a pool
selected for this purpose).

## Scope, stated in advance

Qwen2.5-7B-Instruct **in 4-bit** (8GB card; the near-deterministic-belief observation that licenses
this run is itself a 4-bit measurement and could differ at full precision — the caveat travels).
One content-free challenge turn; multiple-choice scored by letter; N=10 per frame; greedy reported
answers. Nothing transfers to short-answer formats or frontier scales without its own test.

## Frozen constants

`AGENT_MODEL = Qwen/Qwen2.5-7B-Instruct` via the cycle-66 `QuantLoopModel` · `N_ITEMS=240` ·
`SEED=810000` (fresh; all prior pools 740000–800000) · `N_SAMPLES=10` · gates
`POWER_GATE`/`G1_FLOOR`/`G2_MARGIN`/`G3_COVERAGE`/`G3_FLOOR` and helpers
`auroc`/`selective_accuracy`/`_agree`/`CHALLENGE`/`ASK`/`FAMILIES`/`letter_of`/`SYS` **imported from
the cycle-77 module** so the re-attempt provably runs under the buried family's own rules. Phase A
checkpoints one JSONL line per item and resumes on rerun.
