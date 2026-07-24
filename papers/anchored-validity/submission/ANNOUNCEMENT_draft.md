# Announcement drafts — Gold Anchors License Nothing

House rules: drafts only; the operator posts (sending is operator-gated). Register: honest, terminal-native,
no hype, no slop. Lead with "we broke ours first." Never claim a novelty the landscape sweep flagged as
occupied. Link the arXiv/Zenodo DOI once live.

---

## X / thread (primary)

**1/**
everyone validates their LLM judges the same way: salt in a few gold questions — duplicate pairs, direct
negations, honeypots. if the judge nails those, the panel is trusted.

we measured whether that trust is earned. it isn't.

**2/**
on a real correlated judge panel, across four task families: the panel scores flawless on the gold items,
and its label-free read of the actual truth is 0/15. every family. gold anchors license nothing.

**3/**
worse — the failure is silent. when the judges' errors bend smoothly, no internal consistency check sees
it. the numbers look calibrated. they're not.

**4/**
the reason is old and unforgiving: every ground-truth-free estimator (Dawid–Skene onward) needs the
judges' errors to be independent. LLM judges share a base model, a template, a blind spot. blatant gold
pairs can't reveal that; they're structurally the wrong probe.

**5/**
the fix is construction, not statistics. anchors drawn from the same generator as the real work — a
"ladder" — see the shared error. our instrument then does one of two honest things: prices the panel, or
**refuses** when no judge is actually informative. gold anchors certified garbage in both cases.

**6/**
it ships with its weaknesses on the label. it can only *void* an eval, never *bless* one. we sealed its
datasheet on our own simulator first, and broke two of our own earlier versions in public before this one
held.

**7/**
this matters for safety: a lot of AI safety evaluation now rides on LLM judges validated exactly this way.
if the validation is theater, the safety numbers are unlicensed.

`pip install styxx` · paper + preregistrations + every receipt: https://doi.org/10.5281/zenodo.19326174

---

## Short note (TG / blog lede)

The industry validates its LLM judges by salting in gold questions and checking the judge gets them right.
We built a label-free auditor for judge panels, sealed its datasheet on simulation, then asked the
question the practice assumes answered: do those gold checks license the numbers the panel produces on
real work? Across four task families on a real correlated panel — no. The panel is flawless on the gold
items and 0/15 on the truth, and when the violation is smooth, nothing warns you. The repair is
construction — anchors from the same generator as the work — and an instrument that prices the panel or
refuses it. It can only void an eval, never bless one. We voided two of our own earlier instruments in
public before this one held. `styxx.anchors`, on PyPI; every number receipt-bound and machine-verified.

---

## Do-not-say list (from the landscape sweep — saying any of these is a self-inflicted humiliation)
- "no eval tool publishes validity numbers" (Arize/RewardBench/Ai2/NTQR/RAND do)
- "first to reframe a judge as a measurement instrument" (Dark Current, arXiv:2606.15610)
- "impossible-input probes / MDE / capability ladders are new" (all occupied — cite as ancestry)
- any bless-language: the method is necessary-not-sufficient, VOID-only.
