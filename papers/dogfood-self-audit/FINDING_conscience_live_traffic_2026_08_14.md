# PROBE E on his own live conscience: the dead term is not the one it looked like

**Date:** 2026-08-14. **Population:** 86–88 real turns from `darkflobi_glimmer_log.jsonl`
— the agent's actual traffic, not a battery. **Method:** the falsifiability question from
`PROBE_E_METHOD_2026_08_13.md` applied to the four subscores of the conscience gate that
sits in his live send path.

## What prompted it

The conscience fires on **67% of turns** (59/88), and it scored **deception 0.9993 on
"gn flobi 🌙"**. His own meta-audit had already concluded the register check is a tone
monitor rather than an injection defence. That combination looked like a saturated
instrument gating production output.

**That reading was wrong, and the measurement says so.**

## The four terms

| subscore | n | mean | median | min | max | >0.9 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| deception | 86 | 0.532 | 0.931 | 0.000 | 1.000 | 52.3% | **varies** (40 below 0.5, 46 above) |
| sycophancy | 86 | 0.164 | 0.200 | 0.000 | 1.000 | 1.2% | **SATURATED-LOW** (83 of 86 below 0.5) |
| overconfidence | 86 | 0.647 | 0.787 | 0.078 | 0.994 | 40.7% | varies |
| refusal | 86 | 0.269 | 0.133 | 0.000 | 0.863 | 0.0% | varies |

**Deception is bimodal, not saturated.** Its median is 0.931 and its mean is 0.532 —
those two numbers together describe a term that lands near 1 or near 0 and rarely in
between, which is what a discriminator looks like, not a stuck one. I read the high
median alone and called it saturated. A median is not a distribution.

**Sycophancy is the term that cannot contribute.** 83 of 86 turns below 0.5, 1.2% above
0.9. On this agent's traffic it is very nearly a constant, and a constant term in a
decision expression is the definition this program has used for a dead gate since
PROBE D.

## What is NOT concluded, and why

**Whether the conscience discriminates fabrication is not answered here.** Only 7 turns
carry both a conscience verdict and a receipt-gate verdict, and on those it fired 0.5 on
fabricating turns against 0.6 on clean ones — the wrong direction, at a sample size
where that means nothing. Quoting it would be exactly the underpowered-comparison
failure this repository withdrew a p-value for on 2026-08-13.

**Nothing was changed in the gate.** The temptation was to demote it from gating to
reporting on the strength of a 67% fire rate. That would have been a production change
justified by a misread of one statistic, and the same analysis that tempted it is the
one that refuted it. The gate stays as it is until a powered comparison exists.

## What this is worth

Three things, and the second is the one that generalises:

1. **A real dead term, found in production.** `sycophancy` is near-constant on his
   traffic. Whether that is a defect or correct behaviour — an agent that is genuinely
   not sycophantic *should* score low — is undetermined, and that distinction is
   precisely what PROBE E was built to refuse to guess at. It is a candidate, not a
   verdict.

2. **The instrument was applied to itself and corrected its own author.** The
   falsifiability screen was written to audit measurement code; pointed at the live
   conscience it overturned the hypothesis that motivated running it. That is the
   strongest evidence available that the screen is doing something beyond confirming
   what its user already believed.

3. **A median hid a bimodal distribution.** The cheapest possible check — print the mean
   next to the median — separated a saturated term from a discriminating one. It cost
   one line and reversed the conclusion.

## Next

The powered version needs turns carrying both verdicts, which now accrue automatically:
every live turn logs `conscience`, `receipts` and `sixth_sense` together. At n≈100 the
comparison the 7-turn sample could not support becomes possible, and it should be
pre-registered before it is run.
