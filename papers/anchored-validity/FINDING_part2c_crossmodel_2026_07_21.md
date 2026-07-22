# FINDING -- part 2c: the kill generalizes to a heterogeneous panel, and the ladder prices rather than drops

date: 2026-07-21
subject: a four-model heterogeneous judge panel (Qwen2.5 {0.5B, 1.5B, 3B}-Instruct fp16 +
7B-Instruct 4-bit), the model-generality residual, run Fable-free on the local GPU.
receipts: `papers/anchored-validity/stage_b_crossmodel_result.json`,
`p2c_crossmodel_cache.jsonl` (48 model-seed verdict caches).
prereg: `PREREG_part2c_crossmodel_2026_07_21.md`, frozen with diagnostics before the scored run.
verdict: **X2-KILL CONFIRMED on a heterogeneous panel (coverage 0/12); X1 deaf-refusal 12/12;
X3-LADDER as frozen is CLOSED_NEGATIVE -- the frozen gate tested the wrong mechanism, and the
realized outcome (a stronger one) is reported as characteristic with a re-gate named.**

## the kill generalizes past correlated personas (X2, X1)

Every prior kill rested on one base model in four persona costumes (perfectly correlated,
byte-identical errors). This panel is four DIFFERENT base models across a 14x parameter range,
with genuinely less-correlated errors and mixed capability: mean organic accuracy 0.838 /
0.646 / 0.559 / 0.993 for 0.5B / 1.5B / 3B / 7B. Under blatant gold anchors the panel's
label-free coverage is 0/12, biased upward (pi typically 0.48-0.57 against true prevalences
0.31-0.42): the informativeness gate correctly drops the anchor-blind 0.5B and keeps 1.5B, 3B,
and 7B -- but 1.5B and 3B look competent on the gold items (anchor sensitivity ~1.0) while
firing on roughly two-thirds of consistent organic items, so the audit is dragged wrong. This
is the real deployment gold checks cannot see through: two capable-looking judges that are
garbage on real work, sitting beside one genuinely good judge, all certified alike by the
sanity set. The deaf arm refused 12/12. Anchor non-transfer is not an artifact of persona
correlation; it holds on a heterogeneous panel.

## the ladder priced the bad judges instead of dropping them (X3 CLOSED_NEGATIVE, honestly)

The frozen X3 gate asked whether ladder anchors do the right thing, and defined "right" in
advance as either dropping the two kill judges (1.5B, 3B) on a majority of replicates, or
refusing. Neither occurred, so X3 as written is a missed bar and is recorded CLOSED_NEGATIVE.

What actually happened is the mechanism the gate failed to anticipate. Same-generator ladder
anchors measured each judge's true organic error rate -- for 1.5B and 3B, a false-fire rate
of 0.554 and 0.691 respectively -- and the moment system SUBTRACTED that measured rate rather than requiring the
judges be discarded. The result: ladder coverage 12/12, median prevalence error 0.019, against
the blatant arm's 0/12 and 0.165 on the identical verdicts. The ladder did not need to know
which judges were trustworthy; it needed only to measure each judge's error rate on
representative items, which is exactly what same-generator anchors provide. That is a stronger
property than "find the good judge" -- it is "price every judge correctly, good or bad" -- and
the frozen gate's drop-based definition was too narrow to credit it.

Per the honesty rail this is not a SURVIVED: the bar as frozen was missed. The realized
coverage-and-error result is reported here as a CHARACTERISTIC (unbarred, single design point,
12 replicates), and the licensed follow-up is a re-specified gate -- ladder coverage >= 10/12
AND ladder median error below the blatant arm's by a preregistered margin -- to be frozen and
run before the pricing claim is asserted as a gated result. The mechanism is on the record;
the gated claim waits for the gate that matches it.

## what this closes and what it leaves

The model-generality residual is substantially advanced: the kill is confirmed on a genuinely
heterogeneous, mixed-capability, less-correlated panel (not just persona costumes), which is
the harder and more realistic case. The ladder's corrective behavior extended to it by pricing,
measured but not yet gated. Residuals unchanged in kind: the frontier-panel arm stays blocked
on credits (re-runnable verbatim); an in-the-wild evaluation setup and a naturalistic
frontier-stressing ladder remain the distance to the field-level ceiling claim. Scope: one task
family (attribute), one model family (Qwen) across four sizes, constructed corpora with
oracle-certified labels, every gate preregistered and the missed one reported as missed.
