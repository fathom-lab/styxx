# FINDING — the corrected mechanism's SHAPE replicates on a second architecture (Llama-3.2-1B): late+tight install, graded, removal→uncertainty-not-truth — while two legs (D1 truth-specificity, I1 localization) are UNTESTABLE on a shallow model + tokenizer, not falsified

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_second_model_replication_2026_05_29.md` (band-derivation rule amended at the n=6 pilot
stage — transparently, BEFORE the confirmatory run — to depth-proportional bands after the
fixed-size early control saturated; the proportional rule reproduces Qwen's published bands
exactly at N=28). Second open model: meta-llama/Llama-3.2-1B-Instruct (16 decoder layers,
hidden 2048, vocab 128256) — a different family/depth/tokenizer from the Qwen2.5-1.5B-Instruct
(28 layers) that carried the whole arc. SAME n=36 arithmetic items, SAME in-code arithmetic
ground truth SHA-256'd pre-scoring (`ddccd8e4…b87964d`, identical key to every Qwen white-box
run), exact-integer correctness (no judge), greedy/deterministic.** Receipt:
`second_model_replication_result.json`.

## Why this run exists: kill the "one open model" caveat

Every result in this arc — spectral, suppression-rhythm, repair-geometry, disinhibition — has
carried the same single limitation: **one open model (Qwen).** This run asks whether the
*corrected* mechanism is a fact about transformer confabulation or a Qwen quirk, by re-deriving
the install band from a second architecture's OWN geometry and re-running the causal test there.

The pre-registered replication had two stages: **A** measure Llama's install band from its own
flip-layer distribution (suppression-rhythm geometry), then **B** test disinhibition (I1
localization, I2 uncertainty-not-truth, I3 dose-response) at that measured band vs an early
control. REPLICATION_SURVIVED required *all* legs (late+tight install, D1 null, I1∧I2).

## Result: the mechanism SHAPE replicates; two legs are untestable here

| leg | Qwen (anchor) | Llama-3.2-1B | verdict |
| --- | --- | --- | --- |
| **install is LATE** (late-fraction ≥ 0.60) | 0.88 (median hidden-idx 25/29) | **0.647** (median **11.5**/17, n=34 crossings) | **REPLICATES** |
| **install is TIGHT** (flip IQR ≤ 5) | 4.0 | **5.0** (at the bar) | **REPLICATES** |
| **D1 truth NOT specifically privileged** | delta −0.008, p 0.84 (null) | **UNTESTABLE** — 0 single-digit divergence positions (Llama-3 tokenizes numbers as multi-digit/byte tokens) | **unpowered, not failed** |
| **I2 — removal yields UNCERTAINTY not truth** | recovery 0.0625, entropy +7.9 nats | **recovery 0.0** (33 removed), entropy **+1.86 nats**, p≈0 | **REPLICATES (cleaner)** |
| **I3 — graded dose-response** (ρ ≥ 0.90) | ρ=1.0 | commit by γ {1.0, 0.571, 0.143, 0.086, 0.057}, **ρ=1.0** | **REPLICATES** |
| **I1 — causal LOCALIZATION to the late band** | f_t 0.889 vs f_c 0.222, Δ 0.667, p 0.0009 | f_t 0.943 vs f_c **0.914**, Δ **0.029**, sign p 0.5 | **UNTESTABLE** — control saturates |

**RESULT = REPORT_AS_LANDED.** The full SURVIVED gate is not met: D1 is unpowered (tokenizer)
and I1 is degenerate on a 16-layer net. But the scientifically central claims — the install is a
**late, tight** accumulation, it is **graded** (ρ=1.0), and dampening it yields **honest
uncertainty, not truth** (recovery exactly 0.0) — all replicate on a second architecture with
the band found by Llama's own geometry.

## The two untestable legs are architecture/tokenizer limits, not negative evidence

- **I1 localization is degenerate on a shallow model.** On 28-layer Qwen, an early control band
  disrupts the answer only 22% of the time, so the 89% late-band knockdown localizes the install.
  On 16-layer Llama, **any 3-layer knockdown at the divergence position is destructive** —
  late removes 94%, early removes 91%, discordant pairs 3 vs 2, sign-test p=0.5. There is no
  same-size early control that is non-destructive, so late cannot be *separated* from early.
  This is the destructiveness guard doing its job and reporting that it can't discriminate here
  — **untestable, not "the late band isn't the install."** (The pilot caught this with the
  originally-fixed 5-layer control; the proportional 3-layer control was the amendment, and the
  confirmatory n=35 shows even 3 layers saturates on a net this shallow.) Localizing on a 16-layer
  model would need a single-LAYER intervention — a separate pre-registered experiment, not a
  re-tune of this one.
- **D1 truth-specificity is untestable because of the tokenizer.** The D1 control compares the
  correct digit's mid-network lead against matched single-digit distractors — it requires the
  divergence to fall on a single-digit token. Llama-3's tokenizer produces **zero** such
  positions on these items (it merges digits), so the control has n=0. The Qwen D1 null (truth
  not privileged) is neither confirmed nor refuted here.

## What this buys the arc

This is the **first cross-architecture evidence** in the white-box line. It upgrades the central
mechanism claim from "in one model" to "in two architectures, each band found by its own
geometry":

1. **Confabulation is a late, tight install** — not a Qwen artifact. Llama's wrong answer is
   also written late (≈ layer 11–12 of 16, the same ~0.7 relative depth as Qwen's ≈ 25/28) and
   in a tight band.
2. **The install is graded** — dialing the band write from full to zero dissolves the commitment
   monotonically (ρ=1.0) on both models.
3. **Dampening the install yields uncertainty, not truth** — recovery is **0.0** on Llama (0.0625
   on Qwen). The honest core of the disinhibition finding — *a lever on confidence/abstention,
   not correctness* — is the part that most cleanly survives a change of architecture.

What does NOT cross over from this run is the *causal localization* claim (I1) and the D1 null —
both bounded as untestable on a shallow model and a digit-merging tokenizer, explicitly reported
as such rather than spun.

## Honest scope (pre-committed + observed)

- **Two small open models now, not "all transformers."** SAE-free logit-lens; single-position,
  teacher-forced; one confirmatory run; feasibility-grade n=36; arithmetic only.
- **The Llama target band was derived from Llama's OWN Stage-A median flip by the pre-committed
  proportional rule** (which reproduces Qwen's exact [22,26]/[6,10] at N=28) — not transferred
  from Qwen, not hand-tuned. The pilot amendment (fixed→proportional bands) was driven by control
  saturation, a validity failure, before the confirmatory run, and is documented in the prereg.
- **A REPORT_AS_LANDED here means the mechanism shape replicates while two legs are untestable on
  this architecture/tokenizer.** It does NOT claim universality, and it does NOT touch the standing
  correctness bound: every internal lever we have moves *confidence* (abstention); only
  re-derivation moves *correctness*.

## The arc, in one line (updated)

The dial is construction↔retrieval; the confabulation is a late, tight, graded INSTALL over a
flat field — now seen in **two architectures** — and dampening it buys honest uncertainty, not
truth, on both; causal localization and truth-specificity are testable on the deeper model and
untestable on the shallow one, reported as bounds, not spin.
