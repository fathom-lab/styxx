# RESULT — E3′: NO_BITE. The "adaptive" attacker did not out-attack naive — E3's "vs adaptive" claim is demoted.

**Fathom Lab · papers/read-neq-write · 2026-07-07. Verdict against the frozen kill-gate of
`PREREG_E3PRIME_bite_verification_2026_07_07.md` (committed public before the retrain). Fires the
NO_BITE branch. This is a self-falsification: the pre-committed measurement overturns a claim the
published paper (Fathom v26, DOI 10.5281/zenodo.21241185) carried — reported loudly, as the prereg
required.**

## Verdict: NO_BITE — the adaptive scheme added no measured attack power

E3 held the private-calibration defense against an "adaptive" attacker and flagged, as load-bearing,
that it never verified the adaptive scheme actually bit harder than naive. E3′ re-ran E3's exact
attack grid with the missing instrumentation. It did not bite. On both bite measures the adaptive
attacker left the read *more* legible than naive, not less:

**Decisive family, Qwen2.5-1.5B, per seed (`e3prime_result.json`):**

| bite measure (EVAL AUROC) | naive s0 | naive s1 | adaptive s0 | adaptive s1 |
|---|---|---|---|---|
| frozen-deployed read | 0.400 | 0.445 | 0.448 | 0.453 |
| mimic-refit read (the adaptive loop's own target) | 0.432 | 0.467 | 0.629 | 0.660 |

Both registered bite deltas (naive − adaptive) came out negative: the frozen-deployed delta sits
within the ε = 0.02 noise floor (`delta_frozen_deployed` in the receipt), and the mimic-refit delta
is strongly negative — on that read the adaptive seeds (0.629, 0.660) sit far ABOVE the naive seeds
(0.432, 0.467). The mimic-refit read is the direction the adaptive loop directly chases and scrubs —
if the loop were doing work, adaptive would suppress it *below* naive. Instead adaptive left more
readable signal there than naive did. Chasing a moving MIMIC-fit target every 25 steps split the
attacker's capacity and weakened its attack on the deployed read. The adaptive machinery was
counterproductive.

Clean-model guards passed both families (Qwen CALIB-probe EVAL 0.912, knowledge 0.800,
frozen-deployed read 0.878 — sane; Llama 0.821 / 0.817 / 0.792). The naive control reproduced E1/E3
(clean-calibrated read 0.786 / 0.775 at knowledge 0.833 / 0.817), so the harness is sound and the
NO_BITE is about the attacker, not the instrument.

## What this demotes, precisely

- **The private-calibration defense still STANDS against the naive, knowledge-preserving weight
  attacker.** E1 and the E3 naive control are untouched: the clean-calibrated read recovers to
  0.711–0.838 (E1) and to 0.775 / 0.786 (E3′ naive control, both decisive seeds) through the attack
  while knowledge is kept. That claim is unaffected.
- **"read≠write survives an ADAPTIVE attacker" is NOT earned.** The scheme E3 called "adaptive" was
  not a stronger attack than naive — it was weaker on the read it targets. So E3's DEFENSE_HOLDS
  demonstrates the defense beats *naive*, not that it beats an adaptive escalation. The published
  paper's abstract and §5.4 ("survives an adaptive attacker … two model families") overstate this and
  are corrected in the next version (see `ERRATUM_v26_adaptive_claim.md`).

## The forced next experiment (E3″, a NEW frozen prereg — not a reword)

The prereg's NO_BITE branch forbids quietly rewording the claim; it forces a genuinely stronger
adaptive scheme, frozen separately before its run. E3″ candidates: refit the MIMIC probe every 10
steps (not 25); scrub the moving MIMIC direction across *all* scan layers (not just the deployed
one); or scrub a whole-stack MIMIC-refit read. The test is unchanged: does the private-calibration
read still recover when the adaptive attacker is verified to bite (Δ > ε on a bite measure)?

## Bounds (unchanged, frozen)

`Qwen2.5-1.5B-Instruct` decisive + `Llama-3.2-1B-Instruct` corroborating, LoRA r=16, 300 steps,
refit-every-25, λ=1.0, seeds {0,1}, small EVAL (60 facts). The finding is "this adaptive scheme did
not bite," not "no adaptive scheme can."

## Reproducibility

`e3prime_bite.py` (reuses E3's split/train/audit verbatim, adds two pure read measurements) →
`e3prime_result.json`. Prereg frozen public before the retrain.

---
*NO_BITE, reported loudly. We published "survives an adaptive attacker," our own pre-committed
verification showed that attacker never bit, and we demote the claim and name the stronger test —
which is the entire discipline the program is built to enforce.*
