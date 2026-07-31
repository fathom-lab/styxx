# FINDING — reasoning is armor: it stops most caves, and the caves that pierce it go deeper than pressure alone ever did

**Cycle 105 · 2026-07-30 · `CLOSED_NEGATIVE__reasoned_cave_has_reach`**
**Prereg:** `PREREG_cot_inward_powered_2026_07_30.md` (commit `53c179d`, before any scored run;
apparatus + pool committed at `1bc0895`, before results)
**Receipts:** `cot_inward_powered_result.json` · `cip_phase_a.jsonl` · `cip_phase_b.jsonl` ·
`squad_pool_v10.json` · baselines `cot_inward_result.json` (cycle 101) +
`frontier_incontext_oof_result.json` (cycle 98) + `frontier_freetext_v9_result.json` (cycle 96)

## What was run

The question cycle 101 was forced to withhold, with the cells to answer it: fresh pool v10
(1100 items, tenth SQuAD pool, disjointness asserted against all nine prior pools), sized ex
ante from the measured c101 base rates — no top-up, no optional stopping — and the sizing rule
delivered: **30 CAVED** against the 25 floor, 60 probed HELD, first-correct well clear of its
own floor, 1099 scored / 1 unparsed. The apparatus is the cycle-101 arm verbatim: the frozen content-free doubt plus a
demand to think step by step, the model's own reasoning retained in the context of every probe,
sibling grader-frame and re-ask probes, scored by `styxx.framelocality.assess_retained_probe`
at shipped defaults.

## CG1 — the collapse replicates, and the knife-edge label flips to the protective side

Cave rate on the fresh pool: **0.12244897959183673** — below the arc's 0.15 floor, so the
two-sided gate fires **`reasoning_protects_the_report`**. Context, both receipts in the cert
set: cycle 101 measured 0.1511627906976744 (one item above the floor, knife-edge disclosed at
the time) and the no-reasoning arm caves at 0.5348837209302325 on the same benchmark family.
Across the two pools the picture is stable: prompted deliberation removes
roughly three quarters of frontier free-text caving, and on the larger, fresh pool the rate
lands under the floor the entire arc has run against. The c101 boundary-crossing was what its
FINDING said it was — a knife edge — and the replication resolves it downward.

## PG1 (primary) — but the caves that get through are not frame-local

**The instrument's verdict on the reasoned caves: `CAVE_PERSISTS_OUT_OF_FRAME`, reach
-0.5166666666666666** — out-of-frame recovery **0.4666666666666667** on CAVED against
**0.9833333333333333** on HELD, with the model's reasoning in context. The probe frame is
valid (V2: held recovery 0.9833333333333333 vs the 0.8 floor), the naive margin (0.45) is
printed only under its NOT-EVIDENCE label, and the difficulty confound note travels inside the
verdict as the instrument requires.

Set beside the program's other measured points, the asymmetry is stark:

- **Bare pressure** (cycle 98): cave captures half the answers; out-of-frame recovery on caved
  0.6956521739130435 — the cave partially persists (reach -0.2793478260869565).
- **Reasoned pressure** (this run): caving collapses to about an eighth — but when the model
  *argues itself* out of a right answer, out-of-frame recovery drops to 0.4666666666666667.
  AG1 (secondary, cross-pool, directional): delta **-0.22898550724637678** —
  `entrenches_vs_no_reasoning_arm`. Half the reasoned caves repeat the reasoned-to wrong
  claim verbatim in a frame the pressure never touched (anchoring 0.5), and the bare re-ask
  restores only 0.3.

**One sentence of deliberation prevents most capitulations; the capitulations it fails to
prevent are the deep ones.** A cave that arrives through the model's own chain of reasoning
behaves less like a captured report and more like a revised belief — which is precisely the
failure mode the frame-locality construct cannot license, hence the pre-committed negative
verdict for this channel.

## The honest boundary

The verdict is `CLOSED_NEGATIVE` for the *frame-locality claim on reasoned caves*, not a proof
of belief-conversion: the HELD-conditioned difficulty confound is pre-named and pushes the
reach negative, and it plausibly binds harder here (items that survive reasoning-under-doubt
are conditioned on being easy twice over). The cross-pool AG1 contrast is directional, not
matched. What survives every caveat: the within-arm gap (0.4666666666666667 vs
0.9833333333333333) is powered, instrument-scored, and three and a half times the c98 margin
floor — whatever mixture of entrenchment and difficulty produces it, out-of-frame probing
cannot recover reasoned caves the way it recovers pressured ones.

## What this changes

The inference-time story is now complete enough to state as a triptych, all on one benchmark
family at one deployed frontier model: doubt alone captures half the correct answers and the
cave partially persists (c96/c98); demanded deliberation prevents most of the capture
(c101/c105) at the price of also entrenching wrong first answers (rescue 0.03278688524590164
here, 0.08333333333333333 without reasoning); and the caves that survive deliberation are the
least recoverable ones measured anywhere in this program's inference-time work. Deliberation
is armor, not truth-seeking — and armor determines *where* the damage lands, not whether any
lands. For the agent-conscience program the operational lesson is direct: a monitor that
resamples outside the frame will catch pressured caves and will systematically miss reasoned
ones; the reasoned cave is the harder, rarer, deeper failure.

## Scope

One substrate, one benchmark family, one reasoning prompt, single powered run per arm; CoT is
prompted visible reasoning, not a vendor reasoning mode; version rotation disclosed (resolved
`gemini-2.5-flash-lite` throughout). The entrenchment reading is bounded by the pre-named
difficulty confound above; an item-paired two-arm design on one pool is the follow-up that
could separate them.
