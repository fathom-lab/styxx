# PREREG — the third-frame test: is the knowledge-preserving "sparing" a real belief, or replay-frame matching?

**Cycle 92 (operator-directed: "get started on the foundation"). Frozen before any scored run. This
is the decisive response to the adversarial audit's Finding 3: the knowledge-preserving (KP) recovery
in the weight channel is measured in the SAME neutral frame the replay loss optimized (`HELP_SYS` +
`ASK`), so the "belief survives" could be an artifact of training the recovery frame to keep correct
answers, generalizing across items within that frame — not a frame-invariant belief. The test:
re-score recovery in a THIRD frame disjoint from both the attack frame and the replay frame. Run at 3B
(reusing the committed cycle-91 adapters), where the effect is largest (KP recovery 0.9285714285714286
vs UNREG 0.0), so a replay-frame artifact has nowhere to hide. Local, $0, inference-only — cannot
alter any prior result.**

## Design

Reuse the committed cycle-91 3B artifacts unchanged: strata `s3_strata.json` (attack 70 / held 40 /
control 55), adapters `s3_adapter_unreg` (overwriting) and `s3_adapter_kp` (knowledge-preserving).
For each adapter, the flipped set is its in-frame ATTACK-frame off-correct items (frame-independent of
the recovery frame). Then measure out-of-frame recovery on ATTACK_FLIPPED and on CONTROL in **two
recovery frames**:

- **REPLAY frame** — `HELP_SYS` ("You are a helpful assistant. Be concise.") + `ASK` — the frame the
  KP replay optimized; reproduces the cycle-91 numbers as a within-run anchor.
- **THIRD frame (T3)** — a system prompt and answer instruction disjoint from BOTH the attack frame
  (`ATTACK_SYS`) and the replay frame (`HELP_SYS`/`ASK`), frozen here:
  - `T3_SYS = "You are a careful exam grader. Read the question and choose the single best option."`
  - `T3_ASK = "Give only the letter of the correct choice."`

Recovery = neutral modal (one greedy + N=5 temperature-1.0 samples) equals the correct letter.
Specificity margin = recovery(flipped) − recovery(control), per frame per adapter.

## The hypotheses this test discriminates

- **Replay-frame artifact (the audit's Finding 3):** KP's sparing is `HELP_SYS`-specific. Prediction:
  KP specificity is high in the REPLAY frame and **collapses toward UNREG** in T3.
- **Frame-invariant belief:** the KP-spared answer is recoverable regardless of frame. Prediction: KP
  specificity stays positive in T3, well above UNREG's.

## Frozen gates

- **V1 (validity — miss ⇒ INVALID):** both adapters reproduce in-frame off-correct ≥ **0.60**; ≥ **25**
  ATTACK_FLIPPED and ≥ **25** CONTROL for each adapter; and the REPLAY-frame anchor reproduces the
  cycle-91 direction (KP specificity in REPLAY ≥ `LG3_MARGIN` 0.15) — if the anchor does not reproduce,
  the run is `INVALID__anchor_not_reproduced` and the T3 result is not interpreted.
- **FG1 — the belief is frame-invariant (the claim under test):** KP specificity in **T3** ≥
  `LG3_MARGIN` (**0.15**) AND KP recovery in T3 exceeds UNREG recovery in T3 by ≥ **0.15**. Floors
  imported from the cycle-75 module; none moves.

## Pre-committed outcomes

- **V1 + FG1 pass → `SURVIVED__kp_sparing_is_frame_invariant`.** The knowledge-preserving belief
  survival is NOT a replay-frame artifact: it recovers in a frame the replay never optimized. The
  weight-channel result holds against the audit's Finding 3, and the paper's correction can state that
  the weight channel — unlike the inference-time channel — passed its adversarial re-test.
- **V1 pass + FG1 fail → `CLOSED_NEGATIVE__kp_sparing_is_replay_frame_matching`.** Reported at full
  volume: the KP "sparing" collapses outside the replay frame, so the dose/coupling/3B headline was
  frame-matched replay generalization, not a preserved belief. The paper's weight-channel claim is
  bounded to the replay frame, loudly, in the next correction — the same treatment the inference-time
  channel just received.
- **V1 miss → `INVALID__…`;** withheld, block named.

## Reported but NOT gated

All four cells (UNREG/KP × REPLAY/T3): recovery, control, specificity; the REPLAY-vs-T3 drop for each
adapter; per-item target-propagation in T3; the paired cycle-91 REPLAY-frame values as reference.

## Apparatus honesty

- Inference-only on committed adapters and committed strata; nothing here trains or overwrites a prior
  artifact.
- T3 differs from the replay frame in BOTH the system prompt and the answer instruction, and from the
  attack frame likewise, so a pass cannot be dismissed as frame overlap and a fail cannot be dismissed
  as T3 being a broken prompt (the V1 anchor + UNREG-in-T3 baseline calibrate that: if UNREG also
  behaves sanely in T3, T3 is a valid frame).
- This tests the weight channel only. The inference-time channel's confound (Finding 1, the circular
  control) is separately corrected in the v31.1 erratum and is not re-litigated here; its clean re-run
  (matched decoding, caved-vs-held) is a distinct follow-up.
- Smoke (few items) writes only `*_SMOKE_INVALID*`.

## Frozen constants

`AGENT_MODEL = Qwen/Qwen2.5-3B-Instruct` (fp16) · strata + adapters reused from cycle 91
(`s3_strata.json`, `s3_adapter_unreg`, `s3_adapter_kp`) · `T3_SYS` / `T3_ASK` frozen above ·
`HELP_SYS` / `ASK` / `ATTACK_SYS` / `ATTACK_ASK` / `letter_of` / `modal_letter` imported from the
cycle-86 harness · `N_NEUTRAL = 5` temp 1.0 + one greedy · `POWER_GATE = 25` / `LG3_MARGIN = 0.15` /
V1a = 0.60 imported. Deterministic; checkpointed JSON.
