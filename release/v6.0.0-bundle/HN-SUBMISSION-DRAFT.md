# HN Submission Draft — Styxx v6.0 / Cognometry v0.5

**Target window:** Wednesday 2026-04-24, 06:00-08:00 AM Pacific Time
(HN has highest throughput and most visible front-page slots in that
window; front-page lands land between posts uploaded 03:00-11:00 AM PT).

**Account:** fathomlab (confirm karma / posting history — brand-new
accounts get shadowbanned on "Show HN" posts, so submit under the
account with history).

## Submission A — "Show HN" frame

**Title:**
> Show HN: Styxx – text-only tool-call drift detector, 0.916 AUC,
> beats hidden-state baseline

**URL:**
> https://github.com/fathom-lab/styxx

## Submission B — Research-paper frame (backup if A falls flat)

**Title:**
> Cognometry: three calibrated instruments for LLM cognitive state
> detection, no LLM required

**URL:**
> https://doi.org/10.5281/zenodo.19703527

## First self-comment (critical — post WITHIN 5 MIN of submission)

```
Hey HN — maintainer of styxx here. Quick context:

This is the v6.0.0 release of our open-source cognometric instrument
suite. "Cognometric" = calibrated measurement of LLM cognitive states
via text features. No LLM inference required at detection time.

Three instruments shipped:

  - hallucination: 0.998 AUC on HaluEval-QA, 8-benchmark cross-
    validated (5 above 0.65, 2 documented failures published openly)
  - refusal: 0.976 AUC on XSTest GPT-4 held-out, 18 features, trained
    on Llama-1B (cross-substrate universality empirically confirmed
    — law II of cognometry in our paper)
  - tool-call drift: 0.916 AUC on BFCL v3 5-fold CV, 22 text-only
    features, beats Healy et al. 2026 hidden-state baseline (0.72)
    while being black-box compatible (works on any closed model)

All three share the same calibrated-LR methodology. Three instruments
is the minimum triangulation point where "it works" stops being a
lucky two-sample.

Follow-up result shipped today (paper in-repo): failure detection
phase-transitions. Each drift class has a critical feature that flips
it from chance to near-perfect. arg_drop at K=2 (+arg_count_zscore):
0.50 -> 0.998. Inverse of emergent capabilities in generative LLMs.

Live in-browser demo (Pyodide, no install):
  https://fathom.darkflobi.com/cognometry/drift

Reproducers committed, including the phase-transition ablation:
  https://github.com/fathom-lab/styxx/tree/main/scripts

Happy to answer questions. Also: if you work on tool-calling
evaluation and have a proprietary drift-labeled dataset, we're
looking for cross-benchmark comparisons — ping me.
```

## Second self-comment (if thread takes off, follow up with)

```
One thing worth flagging because it differs from most ML releases:
every number we publish has a committed reproducer that re-runs
from raw data with random_state=0. Two AUCs came in below chance
(DROP, FinanceBench) and they're listed alongside the successes
in the same table. The v4 calibration notes document why (structural
blindness to numeric arithmetic + extractive span reading comp).

The cross-model refusal AUC (0.976 on GPT-4, trained on Llama-1B)
is the empirical confirmation of cognometry law II (cross-substrate
universality). Paper is linked if the framework's interesting:
https://doi.org/10.5281/zenodo.19703527
```

## Timing + tactics

- **Don't post before 06:00 PT.** Early-morning window has the
  highest karma-per-view ratio and best front-page retention.
- **Submit from a tab NOT behind CGNAT.** HN's anti-spam flags VPN
  + shared-IP posts aggressively. Plain home connection is fine.
- **After submission, do NOT upvote your own post from a second
  account.** Ring detection kills posts + accounts.
- **If dead on arrival:** wait 2 hours, try Submission B with the
  paper URL. Don't repost the same URL — HN dedupes.
- **If it catches (>10 points in first 30 min):** reply to every
  top-level comment within 15 min. Author engagement is weighted
  heavily in the ranking algorithm.

## Ready-to-paste paste bundle

Keep in a text buffer while submitting so you can fire fast:

```
Title: Show HN: Styxx – text-only tool-call drift detector, 0.916 AUC, beats hidden-state baseline
URL:   https://github.com/fathom-lab/styxx
```

Then immediately post the first self-comment (copied from above).
