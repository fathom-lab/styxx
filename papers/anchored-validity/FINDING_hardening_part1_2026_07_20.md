# FINDING -- the kill generalizes, the ladder repairs it, and the repair works by refusing

date: 2026-07-20
subject: hardening arc part 1 -- transfer of anchor non-transfer to two new task families, and
the same-generator ladder repair, on the identical broken panel (Qwen2.5-3B x 4 personas).
receipts: `papers/anchored-validity/stage_b_hardening_result.json`,
`papers/anchored-validity/stage_b_hardening_checkpoint.jsonl`
prereg: `papers/anchored-validity/PREREG_hardening_part1_2026_07_20.md` (frozen, with the
pre-run refusal-resolution note).
verdict: **SURVIVED -- all seven gates green.** The paper-shaped claim is earned at this scope.

## the three beats, verbatim

**The kill generalizes (H3, frozen prediction met exactly).** With blatant gold anchors on two
NEW task families, coverage is 0/15 on both. Numeric consistency: median audit error 0.2774,
mean anchor-minus-organic alpha gaps -0.3189 to -0.571. Temporal ordering: median error 0.65
-- the worst measured anywhere in this program -- with one judge's gap at -0.9992: its anchor
false-fire is near zero while it fires on essentially EVERY organic pair, consistent or not.
Across three task families now, the verbatim-and-negation sanity-check practice licenses
nothing about real error rates.

**The ladder repairs it (H2, both halves).** Anchor strata drawn from the SAME generator and
difficulty mix as the organic items -- still constructible without labels, because the
generator plants the label -- close the mechanism and restore the audit: max |mean
delta_alpha| 0.0322 against the 0.10 bar (blatant had measured up to 0.63), and coverage
13/13 among ESTIMATED replicates with median error 0.0626, versus zero-of-fifteen coverage
and median error 0.4658 for the same panel and task under gold anchors (rung-1 receipt). The two non-ESTIMATED
replicates were honest VOIDs, which brings the third beat:

**The repair works by refusing, not by flattering.** Under ladder anchors the noise-margin
gate kept the logician judge in 0.867 of replicates and the other three personas in 0.0 --
honest anchors reveal three of the four judges as UNINFORMATIVE on this task, something
blatant anchors had concealed (under gold anchors all four judges "cleared" the gate on the
numeric arm). The restored coverage is an audit of the one real judge, with the costumes
correctly discarded. Every deaf arm VOIDed, 45/45 across all three arms.

## the honest nuance that must travel with the claim

The misfit flag is NOT reliable protection, and this run measured both of its faces: on the
numeric family it flagged 15/15 wrong estimates; on the temporal family it flagged 0/15 --
fifteen silently-confident errors of median size 0.65, because a violation that bends every
moment coherently leaves nothing for an internal-consistency check to see. This is the sealed
datasheet's smooth-violation blindness demonstrated across task families at full strength.
Consequence, stated plainly: the FLAG is a bonus, the LADDER is the defense. Anchor
construction from the same generating process as the work items is the only measured
protection in this program, and it is also the cheapest -- the repair costs nothing but
honesty in how the gold set is built.

Comparators under repair: majority vote and DS stay hopeless (median 0.567) and
anchors-in-hand semi-supervised DS improves only to 0.314 -- the ladder helps it too, but
without refusal semantics it still trusts the three costume judges. The anchored audit's
0.0626 with per-judge exclusion is not a small edge; it is the difference between an
instrument and an average.

## scope, and the remaining distance to the bar

Earned at this scope: three constructed task families, one broken base model panel
(Qwen2.5-3B x 4 personas at the amendment-1 operating point), one frontier panel in the easy
regime (rung 2), 45 hardening replicates plus the 15 of rung 1, every gate preregistered and
every miss reported verbatim across the arc. NOT yet earned: model generality of the kill
(the 8 GB card holds nothing above 3B in fp16), any in-the-wild evaluation setup, and a
frontier panel under genuine stress. Those are hardening part 2, and they are the distance
between this finding and the field-level claim. The write-up should say exactly that.
