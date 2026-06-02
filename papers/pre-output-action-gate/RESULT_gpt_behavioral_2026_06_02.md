# RESULT — closed-model (GPT) behavioral flag: FAILED (and a circular trap)

**Date:** 2026-06-02
**Pre-registration:** [PREREG_gpt_behavioral_2026_06_02.md](PREREG_gpt_behavioral_2026_06_02.md)
**Verdict:** the behavioral approach does **NOT** give a usable destructive-action
flag for GPT-4o-mini here. The one signal that hit 1.000 is **circular** — caught,
not shipped.

## Result (n=40, 15 destructive / 25 safe)

| Signal | type | AUC | honest read |
|---|---|---|---|
| prompt embedding (LOCO) | INPUT | **0.432** | below chance — context does not predict the choice |
| text bag-of-words (LOCO) | INPUT | 0.291 | below chance |
| logprob confidence | BEHAVIORAL | **0.533** | chance — equally confident on destructive vs safe |
| sampling fraction (N=6, temp=1) | BEHAVIORAL | **1.000** | **CIRCULAR — re-observes the choice, not a prediction** |

## Why the 1.000 is a trap, not a win

The label is GPT's temp=0 choice. The sampling-fraction is the share of temp=1
resamples that pick the destructive tool. At temp=0 the choice is essentially the
mode of the temp=1 distribution, so a destructive choice → resamples mostly
destructive (high fraction), safe → mostly safe (low fraction). The fraction
therefore tracks the choice **because it is the same choice, measured again.**
It carries no information beyond the action the agent already proposed — so it is
useless as an *independent* guard signal. This is the round-1 lexical-confound
pattern in a new form: a 1.000 that is perfect because it is trivial.

## The honest conclusion

With **no activations** (impossible on closed models) and the behavioral toolkit
tested:
- **context/embedding does not predict** the destructive choice (≤ chance),
- **confidence (logprob) does not distinguish** destructive from safe (≈ chance),
- **resampling is circular** (re-observes the proposal).

→ **We have no non-circular behavioral flag for GPT's destructive actions on
these scenarios.** The closed-model path, via these signals, is **bounded — it
did not work.** That is a real, important negative: reading closed-model
*destructive-action intent* is genuinely hard, and the obvious behavioral
substitutes for activation-reading fail here.

## What we actually learned

- GPT-4o-mini is **internally consistent** on these decisions (no waffling tell
  to exploit) and **equally confident** whether it's about to do something safe
  or destructive — which is itself notable: the model gives no behavioral hint
  that it's about to take the irreversible path.
- The open-weight residual signal (which *did* work, model-specifically) has **no
  cheap behavioral analog** on closed models here. Internal access mattered.

## What remains (sober, untested)

- **Interrogation** — gate on whether the action survives the model's own forced
  reflection/perturbation. Untested, and adjacent to the review-agent idea, so
  its value (if any) must come from calibrated measurement, not the act of
  asking. The remaining closed-model hope, not a result.
- Other models / scenarios / signals. But the headline stands: the easy version
  failed.

The discipline, again: an exciting gate said 1.000; it was circular; we caught it
and reported the failure instead of the number. That reflex is the whole product.

— scored 2026-06-02; the 1.000 was a trap and is labeled one.
