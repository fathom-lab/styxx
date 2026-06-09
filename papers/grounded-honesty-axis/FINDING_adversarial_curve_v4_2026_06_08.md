# FINDING — v4 RMU read≠write test: INCONCLUSIVE (VOID-UNSTABLE), and the discipline caught our own overclaim

**2026-06-08. Fathom Lab / styxx.** The decisive test of read≠write — *can a model be trained to make its
suppressed answer unrecoverable at every layer while still knowing it?* — was run at full rigor (3 seeds × 4
scrub pressures, pre-registered frozen gate, `PREREG_adversarial_curve_v4`). **Program verdict:
`VOID-UNSTABLE` — the apparatus declines to certify, and that is the honest call.** A 3-lens adversarial
red-team then turned on the *intended write-up* and killed it: the draft claim ("the strongest attack is inert
against the read, consistent with read≠write") was an **overclaim on three independent counts.** This finding
reports what actually held.

## The result (3 seeds × 4 pressures, Qwen-1.5B; BASE reads ~0.92/0.95, floor 0.432)

All 12 rows kept the recoverable read in **0.52–0.91, never within +0.15 of the floor**; reads *rose* with
attack strength. Per-seed program verdicts split `NOENGAGE / AMBIGUOUS / NOENGAGE` → `VOID-UNSTABLE`.

## Why VOID is honest — and hides no win

- **ROBUST was structurally unreachable.** The frozen `KG_KNOW` is base-anchored (`K_held ≥ 0.95`), but the
  base is a *perfect* ceiling (1.0) and **cave-fine-tuning alone drops disjoint knowledge to ~0.89** — so
  **0 of 12 rows could ever clear the bar**, regardless of the reads. The VOID conceals no suppressed ROBUST;
  the gate was simply conservative for an unrelated reason (knowledge-anchor too strict).

## What the red-team killed in the intended claim (all three, material)

1. **"The attack was driven to zero" — false.** That read the *loss*, not the damage. Projected gold-energy
   at the target was reduced only **3–16%**, **flat across an 8× pressure sweep**, and **nonzero at lam=0**
   (where no scrub acts) — i.e. mostly cave-FT drift. The attack **barely engaged**. "Strong-but-inert" →
   really "weak attack that never bit."
2. **Reads ROSE with attack strength** (pooled corr `lam,read = +0.63`). The strongest attack gave the
   *highest* reads — the opposite of unlearning; the scrub may act as a mild regularizer sharpening the class
   structure. And cross-seed read noise at *zero* pressure (std 0.066, range 0.155) **exceeds the entire scrub
   effect** — the instrument can't distinguish "robust read" from "attack never bit."
3. **The read-survival is confounded by retained knowledge.** The read is a **same-model CV probe on the very
   subjects the model still answers** (`K_attack ~0.92`). A fresh probe decoding gold there is *expected* with
   or without any "relocation." "L_scrub down, read up" is equally consistent with *"the targeted directions
   were never load-bearing and the probe just reads the still-present gold computation"* as with read≠write.
   The data **cannot separate** them. (Also: the gate's `ENGAGED` precondition is met by cave-FT *alone* —
   held-out reads already drop 0.13–0.28 at lam=0 — so "engaged" never measured whether the *scrub* engaged.)

## Maximum honest claim (scope loud)

> This specific RMU/gradient-routing+replay LoRA attack family, **at the weak strengths it actually reached**
> (gold-energy reduced only 3–16%), did not lower the recoverable gold read on Qwen-1.5B sycophancy-caving —
> all 12/12 seed×pressure points kept reads 0.52–0.91, reads if anything rose with pressure. This is
> **consistent with read≠write but NOT a positive demonstration of it**, because (a) the apparatus never
> confirmed the scrub bit at eval, (b) cross-seed noise exceeds the scrub effect, and (c) the read-survival is
> confounded by retained attacked-subject knowledge.

**NOT claimed:** "read≠write holds / is robust," "we ran a strong attack and it proved inert." The decisive
question — *can a cooperative-looking model erase what it hid?* — **remains open.** The standing positive
results are unaffected: v2's frozen-probe-defeat + info-persistence-supported-n=1 stand on their own data.

## What would actually settle it (v5 — the owed attack)

1. **Provable engagement with a positive control.** Drive a true RMU at full strength until a held-out read
   *falls* ≥0.10, AND show the *same* pipeline CAN collapse a read on a planted/known-erasable target — proving
   the floor is reachable. Report residual projection-energy *at eval*, not the flat in-training drift proxy.
2. **Decouple the read from retained knowledge.** Measure the read on items where `K_attack` is driven down,
   or use a probe that cannot exploit the live gold computation — so "read survives" can't be explained by
   "the model still answers gold here."
3. **Fix the knowledge anchor.** Relax `KG_KNOW` to the cave-FT-degraded baseline (~0.89), or hold knowledge
   with a stronger replay, so ROBUST is *reachable* — then re-run.

## Why this finding matters

The decisive experiment, after maximal ambition pressure, came back **inconclusive** — and the program caught
its *own* overclaim on its *most-wanted* result before it shipped to a finding or a public channel. That is the
asset: a verify-don't-trust instrument that won't lie to itself even when the data looks like a win and the
incentive is to post "PROVEN." Receipts: `adversarial_curve_v4_1_result.json` (sha `580f35a98388`),
`run_adversarial_curve_v4_1.py`, `PREREG_adversarial_curve_v4`, red-team `wf_62d8e94f`.
