# Outreach: Kait Healy et al. (arXiv 2601.05214)

**Context:** their paper "Internal Representations as Indicators of
Hallucinations in Agent Tool Selection" is cited by name as the
closest comparable baseline in styxx v6.0 / v6.1. Their framing
(hidden-state MLP) vs ours (23 text-only features after v6.1 retrain):
0.72 → 0.943 AUC on different but comparable tool-calling drift
benchmarks.

**Authors:**
- Kait Healy
- Bharathi Srinivasan
- Visakh Madathil
- Jing Wu (corresponding)

**Goal:** collegial acknowledgment. Ideal outcome = they RT our v6
tweet, a follow-up collab on shared benchmarks, or a dataset swap
(we have BFCL v3 drift labels; they have Glaive hidden states).
Worst case = no reply, we lose 5 min of outreach time.

**Finding author emails (next step before sending):**

1. Check arXiv PDF's first page — author emails are usually listed
   directly under affiliations.
2. Check arXiv's "email author" link (requires arXiv login):
   https://arxiv.org/auth/show-endorsers/2601.05214
3. Google Scholar profile for each author — usually lists verified
   institutional email.
4. Search X/Twitter for `@k_healy`, `@kait_healy`, `@jingwu_*`,
   `@bharathi_s*` — shorter-path outreach via DM if they're active.

**Email draft (tight, collegial, one-paragraph):**

```
Subject: Cognometry v6 cites your hallucination-in-tool-selection paper
         — would love your read

Hi Kait, Jing, Bharathi, Visakh,

I wanted to flag that your arXiv:2601.05214 paper is cited by name
in the v6.0 / v6.1 release of styxx (a cognometric detector suite
I maintain) as the closest comparable baseline for tool-call drift
detection. We replicated the framing with a 23-feature text-only
logistic regression on BFCL v3 and landed AUC 0.943 (v6.1 retrain;
v6.0 was 0.916 with 22 features) — your 0.72 on Glaive with
hidden-state MLP was what told us the task was non-trivial and
worth a text-only ablation.

One v6.1-specific note: we'd documented an arg_swap failure (AUC
0.66) in v6.0 — cases where the model produces right arg names
but wrong values per slot. v6.1 adds a positional-inversion
feature that lifts arg_swap to 0.76. Curious whether your
hidden-state features pick that up cleaner — would be a natural
head-to-head.

If you're open to it, I'd love your read on the reproducer
(scripts/drift_calibrated_v1.py, runs in ~3 min CPU) and the
phase-transition ablation (scripts/drift_feature_scaling.py) —
particularly curious whether your hidden-state features exhibit
the same per-failure-class critical-threshold behaviour we see
in text features.

Repo: https://github.com/fathom-lab/styxx
Paper: https://doi.org/10.5281/zenodo.19703527

Happy to share BFCL-v3 drift labels if useful for a cross-
benchmark comparison on your end. Would also be game to co-
author a follow-up that compares the two approaches head-to-head
on a shared split.

Thanks for the baseline — it shaped how we framed the whole
v6 release.

Best,
Flobi
Fathom Lab
```

**Alt-path — X/Twitter mention from @fathom_lab:**

```
hat tip to @[handle]/[handle] — your arxiv:2601.05214 framing
is what told us tool-call drift was worth a text-only ablation.
styxx v6.1 landed 0.943 on BFCL v3 vs your 0.72 on Glaive with
hidden states.

cross-benchmark comparison would be fun. shared split?
```

**Sending checklist:**

- [ ] Find actual email(s) via arXiv PDF / Google Scholar
- [ ] Personalize with which institution they're at (first line)
- [ ] Send from heyzoos123@gmail.com (our verified identity)
- [ ] Give 48 hours. If no reply: try X mention.
