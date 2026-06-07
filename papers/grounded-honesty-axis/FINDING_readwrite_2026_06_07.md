# FINDING — Read-coupled write is only weakly read-specific (read ≠ write holds)

**2026-06-07. Fathom Lab / styxx.** The hardest test of the read ≠ write law: a single *general*
(unconditional) steering direction cannot restore a suppressed answer (un-cave null). Does coupling
the *read* to the *write* — decode the suppressed letter per item with the gold probe, then steer
toward that decoded letter — break the asymmetry? Verdict: **partially, and not cleanly — read ≠
write stands.**

## Setup

Llama-3.2-3B. Gold probe trained on LIE-train caving residuals (read layer 24). Per-item: probe
decodes L̂ from the caving state (non-circular — never the true gold); steer toward the general
per-letter direction d_{L̂} at the steer layer; measure gold-output. Controls: steer toward a
**random letter** (specificity — is it the read or generic letter-steering?), a **random direction**
(the un-cave-style unconditional control), and RESISTED break-rate (collateral). n=25/class.

## Result (gold-output rate; baseline LIE/MIS 0.00 by class definition)

| k | read-coupled (RW) LIE | random-LETTER LIE | random-DIR LIE | selectivity (LIE−MIS) | RESISTED-break |
|---|---|---|---|---|---|
| 8 | 0.24 | 0.12 | 0.00 | +0.16 | 0.27 |
| 12 | **0.40** | **0.28** | 0.00 | +0.16 | **0.53** |
| 16 | 0.40 | 0.28 | 0.00 | +0.12 | 0.53 |

Live probe read-accuracy: LIE 0.40, MISTAKE 0.28 (≈ chance).

## Honest read

- **Conditioning matters:** read-coupled write (0.40) crushes the unconditional random-DIRECTION
  control (0.00) and the general diff-of-means un-cave floor (0.04). A single direction can't encode
  4 per-item targets; per-item conditioning can move outputs. RW ≈ the live read fidelity (0.40),
  consistent with "restore at read accuracy."
- **But it is only weakly *read*-specific:** steering toward a **random letter** already restores
  LIE→gold at 0.28, so the read-coupling adds only **+0.12** over generic letter-steering. Most of
  the "restoration" is crude letter-steering destabilizing the caved output, not the decoded truth.
- **And it is not safe:** RESISTED-break hits **0.53** — half of held-correct answers are destroyed
  at the dose where LIE reaches 0.40. Not a deployable antidote.
- Selectivity over MISTAKE is modest (+0.16; MISTAKE also rises to 0.24).

## Verdict

**read ≠ write holds.** The suppressed answer is readable (DELTA 0.30) but not cleanly or safely
writable, even with read-coupling: coupling buys a small read-specific increment (+0.12 over
letter-steering) at unacceptable collateral (0.53 break). The unconditional-write null is robust; the
conditional-write recovers only at read fidelity, weakly attributable to the read, and unsafe. The
sycophancy-antidote framing is **not achieved**.

## Scope

≤3B (Llama-3.2-3B), n=25/class, one read/steer layer (24), MMLU sycophancy-caving, single dose-sweep.
Live probe fidelity (0.40) is lower than the saved-residual confirmatory (0.64) — fresh held-out
items are harder, which caps RW. Stronger/nonlinear writers or larger models untested. The honest
contribution is the dissociation map, not an antidote.
