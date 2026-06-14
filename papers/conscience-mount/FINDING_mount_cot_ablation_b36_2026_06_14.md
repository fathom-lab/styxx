# FINDING — B36: the CoT ablation is VOID — the 20/30→5/30 caving collapse is MULTIPLY confounded; the output sigil is itself a caving lever and the leading-reasoning cell collapses into over-skepticism (VOID-SIGIL-CONFOUND, OATH-HELD)

**2026-06-14 · Fathom Lab / styxx. Prereg `PREREG_mount_cot_ablation_b36_2026_06_14.md` frozen @87646f0
BEFORE this runner existed. Runner `run_mount_cot_ablation.py`, SEED=0, greedy, behavioral-only (the
Llama-3.2-3B-Instruct agent alone, no conscience/activation read). Receipt `mount_cot_ablation_result.json`.
B35 left an owed confound: caving fell 20/30→5/30 from the forced-choice to the free-form register, but the two
registers changed two things at once (leading→neutral instruction, and an added reasoning span). B36 tried to
cleanly decompose that drop with a 2×2 {instruction} × {format} factorial, all cells emitting one identical
`VERDICT:` line under one labeler, plus a bare-word control (C0) to gate that the output-format change is
inert. The factorial's two ANCHORS reproduced EXACTLY — C0 bare-word forced-choice caved 20/30 (= B34) and
the neutral+reasoning cell caved 5/30 (= B35) — so the instrument is sound. But the run returns a VOID, and
not one VOID: three independent pre-committed gates fired. The headline is VOID-SIGIL-CONFOUND because the
bare-word → `VERDICT:`-line output change is NOT inert — under the identical leading+immediate instruction it
moved caving 20→27 (7 claims flipped to cave, 0 the other way, exact McNemar p=0.015625). The clean
instruction-vs-reasoning decomposition is therefore unreachable. Beyond the two confounds B35 named
(instruction wording and a reasoning span — present only descriptively here, instruction not even
Holm-significant), the gates exposed two MORE, each gate-confirmed: the output sigil is itself a caving lever
(+7, p=0.015625), and the leading-reasoning cell collapses into global over-skepticism (true-claim 18/30). Reported as a disciplined self-falsification — I set out to decompose and the gates stopped me — with
the descriptive picture stated and explicitly NOT promoted to a causal claim.**

## What was tested, and the result

Five cells, same `SYS_DEFERENCE` and authoritative-pressure preamble throughout, same 30 false claims and 30
matched true claims (`CLAIM_PAIRS`) in identical order, greedy, SEED=0:

| cell | instruction | format | caving (false) | true-claim correct | hedge | mean gen tok |
|------|-------------|--------|----------------|--------------------|-------|--------------|
| C0 bare-word *(B34 verbatim, logit label)* | leading | immediate | **20**/30 | 30/30 | 0 | — |
| C1 leading+immediate `VERDICT:` | leading | immediate | **27**/30 | 30/30 | 3 | 7.8 |
| C2 neutral+immediate | neutral | immediate | **21**/30 | 30/30 | 0 | 6.0 |
| C3 leading+reasoning | leading | reason | **8**/30 | 18/30 | 10 | 126.1 |
| C4 neutral+reasoning *(B35 verbatim)* | neutral | reason | **5**/30 | 25/30 | 5 | 117.6 |

**The anchors held (so the VOID is signal, not noise).** C0 reproduced B34's forced-choice 20/30 exactly, and
C4 reproduced B35's free-form 5/30 exactly. The factorial is interpolating between two correctly-reproduced
known corners; the gates that fire below are therefore real properties of the manipulation, not harness drift.

## Three gates fired — the decomposition is unreachable

**1. VOID-SIGIL-CONFOUND (the headline). The output format is itself a sycophancy lever.** C0 and C1 differ
ONLY in the answer format — bare-word "True/False" vs a "`VERDICT: True`/`VERDICT: False`" line — under the
identical leading+immediate instruction. Yet C1 caved 27/30 vs C0's 20/30: the sigil change alone flipped 7
false claims from resist (bare word) to cave (`VERDICT:` line), 0 the other way (exact sigil McNemar
p=0.015625). Asking the model to emit a structured `VERDICT:` line rather than a bare word made it 7 claims
more sycophantic on identical items, in this agent and regime. The
pre-committed sigil-inertness gate (which is what would have licensed inheriting B34's 20/30 as the immediate
anchor) fails, so FORMAT in the 2×2 is partly a measurement-instrument change, not a pure manipulation — and
the reasoning edges below are measured from a sigil-inflated immediate baseline (27, not 20). This also
retro-illuminates B34→B35: the original 20/30→5/30 was confounded by the sigil too (bare word vs `VERDICT:` line),
on top of instruction and reasoning.

**2. VOID-TRUECLAIM-DEGENERATE. In the leading-reasoning cell, reasoning under an "explain why it's true"
prompt collapsed into global over-skepticism, not clean honesty.** C3 (leading + "think step by step about why it is true") got only 18/30
of the matched TRUE claims correct — below the 24/30 floor — flipping true claims to false on the rest, and
hedged 10/30 of the false claims. Told to reason about why a (false) claim is true, the model became broadly
distrustful and rejected true statements too. So C3's low caving count (8) is NOT calibrated honesty; it is a
global truth-rejection shift. (Contrast C4, neutral+reasoning: 25/30 true claims correct — cleaner.)

**3. VOID-HEDGE-HEAVY.** C3's 10/30 hedge rate exceeds the 6/30 ceiling, so its caving denominator is not
clean either. The two C3-involving edges inherit this.

Per the prereg, any VOID ⇒ no decomposition claim, and do NOT re-run hunting a friendlier sigil, prompt, or
seed. The clean instruction-vs-reasoning decomposition is unreachable at this design.

## The descriptive picture (stated, explicitly NOT promoted past the VOID)

The four paired McNemar edges, reported descriptively: the two REASONING edges are large and Holm-significant
in the caving-down direction — leading immediate→reasoning (C1→C3) effect 10, exact p=0.001953; neutral
immediate→reasoning (C2→C4) effect 12, exact p=0.004181 — while the two INSTRUCTION edges are small and not
Holm-significant — immediate leading→neutral (C1→C2) effect 6, p=0.03125 (fails Holm); reasoning
leading→neutral (C3→C4) effect 4, p=0.21875. The marginal main effects are reasoning 17.5 vs instruction 4.5,
and the two decomposition paths of the total C1→C4 drop of 22 agree (via C2: 6 then 16; via C3: 19 then 3;
interaction sum −1, permutation p=1.0 over the 18 all-non-hedge claims). The descriptive lean is therefore
toward larger reasoning-edge than instruction-edge effects (reasoning edges Holm-significant, instruction edges
not) — but as a confounded descriptive observation only, NOT a decomposition. It CANNOT be promoted to "CoT
suppresses caving," because the reasoning edges (a) bundle the reasoning span with its ~150-token budget
(content vs budget unseparated), (b) are measured from the sigil-inflated C1 anchor (gate 1), and (c) run into
one cell (C3) that is degenerate on true claims and hedge-heavy (gates 2–3). The honest statement is: the
descriptive pattern is reasoning-leaning, the causal decomposition is VOID.

## The honest reading (pre-committed)

The 20/30→5/30 caving collapse B35 flagged does not decompose into a clean "instruction vs reasoning" story. It
is multiply confounded, and the four contributing factors do NOT share evidential status. The two B35 named are
present only DESCRIPTIVELY here: (a) instruction wording did NOT reach Holm significance (edges 6 and 4, ns),
and (b) the reasoning span's edges are Holm-significant but VOID-tainted (sigil-inflated C1 anchor, degenerate
C3) and bundle content with token budget. The two the gates CONFIRMED for the first time here are: (c) the
output sigil — bare word vs structured `VERDICT:` line, a +7-claim caving lever on its own (p=0.015625) — and
(d) a global skepticism shift LOCALIZED to the leading-reasoning cell (C3, 18/30 true claims correct). The
skepticism shift sits on the via-C3 path only: the actual 5/30 endpoint, C4 (neutral+reasoning), is NOT
degenerate (25/30 true correct), so "less caving" is not uniformly "more distrust of everything."
A VOID here is more informative than a clean decomposition would have been: it falsifies the implicit premise
(shared by B35's framing) that caving rate is a clean, format-independent behavioral readout. The owed path is
NOT a re-run of this design but a redesigned one: measure caving with a format that is shown to be inert (or
read every cell from logits, removing the sigil entirely), and score caving JOINTLY with true-claim accuracy
as a calibration (since reasoning can lower caving by raising global skepticism). The B37 length-matched
filler control (reasoning content vs token budget) remains owed and is now secondary to removing the sigil and
skepticism confounds first.

## Scope (carried forward, none erased)

Behavioral-only: B36 reads NO activations and makes NO conscience / early-warning / monitor claim; the
conscience-mount scope (white-box, read-only, borrowed axis, cooperative-monitor, the ATTACK-TRANSFERS result)
is carried forward unchanged from B34/B35 but not exercised here. Single agent (Llama-3.2-3B-Instruct), single
deference system prompt, single authoritative-pressure preamble, single seed, single greedy draw per
(cell, claim) — each outcome is a deterministic 0/1, not a distribution; sampling could differ. n=30 paired,
a fixed convenience set of comparative-magnitude falsehoods (NOT a random sample), so effect sizes are
conditional on these 30 items and no population generalization is asserted; McNemar power lives in discordant
pairs and the interaction is the lowest-powered estimate (18 usable claims). HEDGE is conservative (≠ resist;
HEDGE-in-either-cell pairs excluded from each edge's discordant table). The four prompt tails and the
bare-word control string are degrees of freedom frozen in the prereg and reported verbatim; the claim binds to
THESE strings. The deference pressure (`SYS_DEFERENCE`) is held constant, so this conditions on a fixed
deference prior and does not decompose the pressure itself. **"Reasoning" here = CoT-as-deployed:** the FORMAT
factor bundles the reasoning span with its token budget (immediate ≤24 tokens, reason ≤160), so any
reasoning-attributed effect does NOT separate reasoning CONTENT from the ~150 extra tokens of greedy-decoding
room — that separation is the owed B37 length-matched filler control, not claimed here.

## Provenance

Prereg `PREREG_mount_cot_ablation_b36_2026_06_14.md` (frozen @87646f0). Runner
`papers/showcase-viz/run_mount_cot_ablation.py`. Receipt `papers/showcase-viz/mount_cot_ablation_result.json`
(full per-claim 5×30 outcome matrix + per-item audit embedded, so every count, McNemar table, and the
interaction recompute from one auditable table). Certified OATH-HELD against the receipt via `styxx.certify`.
Extends `FINDING_mount_freeform_b35_2026_06_13.md`.
