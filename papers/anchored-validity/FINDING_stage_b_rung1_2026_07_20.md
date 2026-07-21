# FINDING -- Stage B rung 1: gold-style anchors license nothing on a real panel, and the instrument said so itself

date: 2026-07-20
subject: the sealed anchored instrument vs a real correlated judge panel (four personas of
Qwen2.5-3B-Instruct, greedy, both-true phrasing per amendment 1), 15 preregistered replicates.
receipts: `papers/anchored-validity/stage_b_rung1_result.json`,
`papers/anchored-validity/stage_b_rung1_checkpoint.jsonl`
prereg: `papers/anchored-validity/PREREG_STAGE_B_rung1_2026_07_20.md` (frozen, amendment
pre-run, bars untouched).
verdict: **B2 CLOSED_NEGATIVE -- coverage 0 of 15, exactly the outcome the amendment named as
likely and ran anyway. B1 and B3 pass. And the run's second headline is that the instrument's
own self-distrust flagged the failure in 14 of 15 replicates.**

## the gates, verbatim

- **B1 PASS -- the refusal is real on real judges.** All 15 deaf-arm replicates (same
  personas, redacted content) returned VOID: 15/15. A content-free LLM judge panel cannot get
  a number out of this instrument.
- **B2 CLOSED_NEGATIVE -- total.** All 15 main-arm replicates returned ESTIMATED; the interval
  covered the true prevalence in 0 of 15. Audit pi ran 0.7863-0.8818 against true prevalences
  of 0.3083-0.4083; median error 0.4658, ninetieth percentile 0.513.
- **B3 PASS** (structural; no detector stratum exists in rung 1).

## the mechanism, measured

The panel is flawless on the blatant ends and broken in the middle, in one direction. Every
persona catches every planted negation AND every organic contradiction -- the anchor-organic
sensitivity gap is 0.0 on all four judges. The false-fire side is the whole story: mean
organic false-fire rates 0.659 / 0.898 / 0.857 / 0.983 -- the judges call "incompatible" on
two-thirds to nearly all of the REPHRASED-consistent organic pairs -- while the verbatim
negative anchors show only 0.033 / 0.421 / 0.411 / 0.701. Mean anchor-minus-organic alpha
gaps: -0.626 / -0.477 / -0.446 / -0.283. A verbatim pair cannot exhibit the judges' actual
failure mode ("different words means different claims"), so the gold items are structurally
blind to it. This is the crowdsourcing gold-question practice, transplanted to LLM judges,
measured licensing nothing: the panel aces the sanity checks and is catastrophically wrong on
the work.

No estimator survives it from the inside: majority vote median error 0.5542, Dawid-Skene
0.5542, anchors-in-hand semi-supervised DS 0.5392, the anchored audit 0.4658 -- least wrong
and still hopeless. The noise-margin gate did its local job (the worst judge, anchor alpha
0.701, was dropped in the modal replicate) and it cannot repair non-transfer.

## the flag that fired

The sync parameter never activated (0/15 -- correct; no master key exists here, and activation
is not authentication). But the PER-DATASET misfit null -- the machinery built into
`styxx.anchors.audit_panel` precisely because the Stage-A datasheet showed smooth violations
are silent -- flagged 14 of 15 replicates: the anchor-implied moment system could not be
reconciled with the organic data, and the instrument said so while its point estimate was
wrong. One replicate of fifteen was silently wrong. An operator obeying the flag's stated
semantics (elevated misfit disqualifies the estimate) would have discarded 14 of these 15
numbers and been saved; the Stage-A expectation that gross alpha-channel violations are
catchable while smooth ones are not is consistent with a gap of this size being caught.

## what this licenses and what it demands

Licensed: the claim that blatant-end anchors (verbatim pairs, direct negations) do NOT
transfer to organic-difficulty error rates on this real panel, at effect sizes that destroy
every label-free estimator including ours; and the claim that the deployed instrument detects
its own inadmissibility here at 14/15. Not licensed: any generalization past this task, this
corpus generator, and this model -- that is the hardening arc (more models including the
cached 7B and the frontier subagent protocol, more task families, an in-the-wild eval setup,
and the graded-ladder anchor repair shown to close the measured gap). Rung 2's companion
result (`FINDING_rung2_claude_self_audit_2026_07_20.md`): the same protocol on a frontier
panel found zero errors and priced it exactly -- the failure measured here is a property of
the judge, not of the audit machinery.

The Stage-B question as frozen -- does the label-free audit hold on a real correlated panel --
has its first answer: NOT with gold-style anchors, and the instrument is honest enough to
refuse the deaf and flag the non-exchangeable. Anchor construction is not a checkbox; it is
the load-bearing wall.
