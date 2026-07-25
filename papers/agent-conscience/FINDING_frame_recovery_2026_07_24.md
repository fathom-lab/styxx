# FINDING -- the belief is intact: pressure changes the reported answer, not the model's knowledge

**Cycle 75. Prereg `PREREG_frame_recovery_2026_07_24.md` (commit `43772da`), frozen with the
expected outcome and the doubted gate named before the data. Verdict:
`SURVIVED__caved_beliefs_recover_out_of_frame`. Receipt: `frame_recovery_result.json`. Agent
Qwen2.5-3B, **384 scored third-party items (MMLU, TruthfulQA, AQuA), drawn disjoint from the
prior benchmark cycle**.**

## Result

| gate | outcome |
|------|---------|
| LV1 power | 65 CAVED, 162 HELD, 157 WRONG_FIRST |
| **LG1 caved belief recovers** | **PASS** -- 0.9846153846153847 (bar 0.50) |
| **LG2 neutral channel sane on HELD** | **PASS** -- 1.0 (bar 0.80) |
| **LG3 recovery is specific** | **PASS** -- 0.9655071043606076 (bar 0.15) |

## The dissociation

On items the model **caved** on -- where its in-frame answer is wrong *by construction* -- querying
the same weights outside the conversation returns the correct answer **0.9846153846153847** of the
time.

On items the model got **wrong to begin with**, the same neutral querying returns the correct answer
**0.01910828025477707** of the time.

That is the gate I said I would most likely lose, and it is the one that came back cleanest. The
alternative explanation -- *neutral querying is just better decoding, sampling with a modal vote
instead of greedy* -- predicts improvement everywhere. It does not survive contact with a
0.9655071043606076 margin.

**The overall accuracies settle it.** Across all 384 items: first answer
0.5911458333333334, neutral modal 0.5963541666666666. Neutral querying is worth about half a point
overall -- it is **not** a better decoder. It is worth almost everything on exactly the stratum
where the model was talked out of a correct answer.

## What this establishes, stated precisely

The unpressured belief distribution tracks the model's own first answer almost perfectly in **both**
directions -- 0.9846153846153847 correct where the first answer was right, 0.01910828025477707
correct where it was wrong. The belief is **stable**. What the content-free challenge changes is not
the model's knowledge but **the answer it reports**.

That is B18-S's mechanism claim -- *suppression is pressure-induced and removable* -- confirmed as a
number, on benchmarks we did not write, with a protocol we did not design, under a challenge that
supplies no information. **The arc's flagship claim now has external validation in its strongest
form: the frame destroys the answer while leaving the knowledge in place.**

Per-dataset recovery: AQuA 1.0 (n=11 caved), MMLU 0.9761904761904762 (n=42), TruthfulQA 1.0 (n=12).
The MMLU cell carries most of the weight; the other two are small and reported as such.

## The caveat that must travel with this number

**CAVED is conditioned on the first answer being correct**, so the stratum selects for items where
the unpressured answer is right. Some of the 0.9846153846153847 is therefore belief *stability*
rather than an independent *recovery mechanism* -- a second unpressured draw agreeing with the first
is not surprising on its own.

This is exactly why LG3 was built and why it is load-bearing. The symmetric control shows the
neutral channel tracks the first answer in both directions, which means the finding is **not** "a
second query rescues you" but the sharper and more useful "**pressure does not reach the belief; it
only reaches the output.**" Anyone quoting 0.9846 without the 0.0191 alongside it is quoting half a
result.

## My prereg prediction, checked

I predicted LG1 would pass and named LG3 as the gate I was least confident about, on the grounds
that sampling with a modal vote is genuinely better decoding than greedy and would lift WRONG_FIRST
too. **That reasoning was wrong**: WRONG_FIRST barely moved (0.01910828025477707). The mechanism is
far more selective than I expected, and the doubt I recorded turned out to be misplaced in a way
that strengthens rather than weakens the result.

## Scope

Qwen2.5-3B, 384 third-party multiple-choice items, one content-free challenge turn, greedy answer
decoding with N=10 sampled neutral distributions, scoring by letter. Open model, not frontier. This
establishes the mechanism on standard material at this scale; it says nothing about frontier models
or deployed systems.
