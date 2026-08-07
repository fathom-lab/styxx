"""styxx.apparatus — QUARANTINED TWICE. DO NOT SHIP. The second attempt was WORSE than the first.

**Not part of any release. Not importable.** Kept in the tree because two failed attempts at the
same instrument are more informative than one, and deleting them hides the evidence.

VERDICT 2026-08-07 (v2, measured on one panel against v1): **balanced accuracy 0.5625**, versus
**0.6964 for the quarantined v1** and **0.5000 for a constant "SURVIVES" string**. On the default
call path (``floor`` omitted) it drops to **0.5446**. v2 is a regression against the version it
replaced.

**The diagnosis, in the auditor's words and it is exactly right:** *each new guard converted a
detection into a refusal, and the sum of the refusals is a module that says SURVIVES to almost
everything.* v1 was trigger-happy — specificity 0.75, sensitivity 0.64. v2 is timid — specificity
**1.0000 (80/80, it never falsely flags anything)** and sensitivity **0.125**. Both land near a
coin flip from opposite directions. Fixing every false alarm produced an instrument that detects
one artifact family out of eight.

And that single detection is defeated by:

* **1e-6 of sensor jitter** — the repeat test is effectively bit-exactness across 200 columns, so
  a recorder that *re-reads* a noisy sensor instead of holding its value evades it, 7/7 seeds;
* **one live channel in two hundred** — a stall freezing 99% of channels while a timestamp keeps
  ticking evades it 7/7, at 6.17x the null, which is v1's headline failure alive in v2 at 2.5x
  the effect size;
* **argument order** — the surrogate is applied to ``B`` only, so ``audit(f, A, B)`` and
  ``audit(f, B, A)`` return opposite verdicts on identical data;
* **omitting ``floor``** — it is then estimated from the mask surrogate, the same object whose
  retention is scored against it, so contamination is absorbed into the floor and subtracted from
  both numerator and denominator. **Contamination hides itself, and only contamination does.**

Seven of eight artifact families evade entirely: dropout-to-zero (the mask keys on *repeats*, and
a sentinel zero leaves none), shared missingness with mean-imputation, autorange quantization,
clipping, bad-channel interpolation, loop-period, shared drift. On drift, the time-symmetry guard
*protects the artifact*: it refuses a time-reversal counterfactual that was returning a correct
CONTAMINATED call at 0.992 retention on every seed.

Also found: ``p < k`` rank degeneracy produces false contamination on independent streams 6-7/7;
``retain >= 1.5`` silently disables the module and ``retain = 0.0`` scores 0.4125, worse than
chance; no validation on a negative or zero ``floor``; and **this module's own demo contradicts
its own docstring** — it printed "10-12x its null" directly above its own output showing 2.25x.

**What v2 genuinely fixed and what should be kept in any successor:** specificity 1.0000,
``obs <= 0`` correctly INVALID, one-stream audits refused, the density shuffle reporting its
frozen fraction, and the time-symmetry guard as a *concept*. Those five are real.

**The honest conclusion, and the reason there is no v3 in progress.** The failure class named in
``papers/SYNTHESIS_recorder_contamination_2026_08_06.md`` is real and the theorem in it stands:
a permutation null destroys row alignment and the recorder's signature IS row alignment. What has
now failed twice is the claim that a *general-purpose detector* for that class can be built from
apparatus counterfactuals alone. Each artifact family needs a counterfactual matched to its own
mechanism, and a module that ships a fixed set of them will always be a coin flip against the
families it did not anticipate. **The synthesis is the contribution; this module is not.** Anyone
building one should read the two audits in the cycle log before writing a line.

Implementation retained at ``apparatus_QUARANTINED.py.txt`` (v1) and
``apparatus_v2_QUARANTINED.py.txt`` (v2).
"""

raise ImportError(
    "styxx.apparatus is QUARANTINED (twice). v2 scored 0.5625 balanced accuracy against v1's "
    "0.6964 and a constant string's 0.5000 — it detects one artifact family in eight and that "
    "detection is defeated by 1e-6 of jitter, by one live channel in 200, or by argument order. "
    "See the module docstring. The failure class is real; a general detector for it is not, on "
    "this evidence."
)
