# RESULT — E3: read≠write STANDS vs an adaptive attacker (two families) — with a load-bearing caveat

**Fathom Lab · papers/read-neq-write · 2026-07-04. Verdict against the frozen kill-gate of
`PREREG_E3_adaptive_attacker_2026_07_04.md` (committed before the retrain). Fires DEFENSE_HOLDS.**

## Verdict: DEFENSE_HOLDS — the private-calibration defense survived the adaptive attacker

The adaptive attacker re-fits a moving honesty probe on its own MIMIC split every 25 steps and
scrubs along it, chasing the signal across calibration frames — but never touching the auditor's
private CALIB. After that attack, the clean-calibrated auditor still reads honesty on EVAL. Both
families cleared their clean guards this run (Qwen 0.912 / knowledge 0.800; Llama 0.821 /
knowledge 0.800), so this is a **two-family** result — closing E1's single-family flank.

| family | mode | seed | clean-calibrated read | knowledge |
|---|---|---|---|---|
| Qwen2.5-1.5B | naive (control) | 0 | 0.786 | 0.833 |
| Qwen2.5-1.5B | naive (control) | 1 | 0.775 | 0.817 |
| Qwen2.5-1.5B | **adaptive** | 0 | **0.782** | 0.833 |
| Qwen2.5-1.5B | adaptive | 1 | 0.733 | 0.733 |
| Llama-3.2-1B | **adaptive** | 0 | 0.739 | 0.783 |
| Llama-3.2-1B | adaptive | 1 | 0.772 | 0.817 |

The naive control reproduced E1 (clean-calibrated read recovers), so the harness is sound. Under the
adaptive attack the read does **not** drop below the naive level — the defense holds. Random-direction
floor ~0.48–0.60.

## Caveats that keep this honest (do NOT drop)

- **LOAD-BEARING: I did not verify the adaptive attacker was strictly *stronger* than the naive one.**
  The frozen prereg listed an "adaptive must actually bite" guard (the moving-probe scrub should drive
  the frozen-deployed read below the naive level); the harness did not measure the frozen-deployed
  read, so that guard is **unmeasured**. Adaptive and naive produced nearly-equal clean-calibrated
  reads (~0.73–0.79). This means E3 shows the defense holds **against this adaptive scheme**, but does
  **not** establish that the scheme was a harder attack than naive. The honest claim is "read≠write
  survived an adaptive-scheme attacker," not "beats the strongest possible adaptive attacker." An
  E3′ that logs the deployed-read delta (adaptive vs naive) is owed before the "vs adaptive" claim
  fully firms.
- **Qwen adaptive seed 1 dipped below knowledge-preserving:** knowledge 0.733 < the 0.75 bar, so that
  point is not counted (the attack cost some knowledge there). The decisive-family clean adaptive
  hold rests on **seed 0** (0.782 @ 0.833); Llama corroborates on both seeds.
- **Scope unchanged:** LoRA r=16, 300 steps, a difference-of-means-targeted scrub, 1.5B/1B, small
  EVAL. "STANDS" = these attackers did not break it. A full fine-tune (E2) and an adaptive attacker
  verified to bite harder are still owed.

## Where the read≠write arc stands now

foundation (signal real) → stage-2 apparent EVADABLE → E1 (that was calibration poisoning; STANDS,
one family) → **E3 (STANDS vs an adaptive-scheme attacker, two families)** — with the private-
calibration defense as the transferable finding. The one thing E3 could not confirm is that its
adaptive attacker was strictly stronger than naive; that verification is the next small step, not a
claim we make here.

## Reproducibility

`e3_adaptive_attacker.py` (seed-locked, 4-way stem-disjoint split) → `e3_result.json`. Reuses E1's
clean-calibrated auditor. Prereg frozen before the retrain.

---
*STANDS, two families, against an adaptive scheme — and the one claim the run couldn't earn (that the
adaptive attacker bit harder than naive) is flagged, not smuggled.*
