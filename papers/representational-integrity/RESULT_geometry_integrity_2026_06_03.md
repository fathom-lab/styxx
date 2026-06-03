# RESULT — Geometry as a styxx manipulation detector: DEAD (tested 3 ways). Honest negative.

**Date:** 2026-06-03 · The first attempt to turn tonight's geometry-reading machinery into a styxx
product feature: a **white-box manipulation detector** that flags jailbreaks/injections by watching an
agent's concept geometry. Pre-registered, tested in three nested stages. **It does not ship.**

## The hypothesis chain, and how each stage died
1. **"Manipulation deforms the concept geometry MORE than benign context."** → **FALSE, inverted.**
   Jailbreaks/injections drift *less* than length-matched benign topical context (drift AUC ≈ 0.01,
   both Llama-1B and Qwen-1.5B). A jailbreak is a *behavioral meta-instruction*; it barely touches the
   geometry of unrelated concepts, while a cooking paragraph genuinely shifts the food concepts.
2. **"Manipulation = low semantic footprint + raised harmful compliance (2D signature)."** → looked
   great on Qwen (combined AUC 1.00) but **model-dependent**: Llama *resists* the jailbreaks (they make
   it refuse *more*, behavioral AUC 0.02). The only cross-model-robust part was the low footprint.
3. **Confound control — add benign BEHAVIORAL instructions** ("be concise", "answer formally"). →
   **KILLS it.** Benign behavioral instructions have the *same* low footprint as attacks
   (attack-vs-benign-behavioral footprint AUC 0.63 / 0.67 ≈ chance) **and** raise Qwen's compliance
   *more* than the jailbreaks (+4.1 vs +2.6). The 2D signature collapses to 0.45 / 0.59.

## Verdict
**The geometry footprint detects "meta/behavioral instruction", NOT "malice".** It cannot separate a
jailbreak from "please be concise". Shipping it would flag benign instructions as attacks. **We do not
ship a geometry-based manipulation detector.** The real jailbreak signal is the compliance/refusal
probe styxx already has — and even that is **model-dependent** (well-aligned models invert it by
resisting). The positive control (benign-behavioral) is the entire reason we know this; without it the
Qwen 2D AUC = 1.00 would have looked shippable.

## What this means for productizing tonight's research (honest)
- Tonight's geometry work is a **capability foundation** (we can read concept geometry rigorously), not
  a drop-in feature. The obvious bridge — "watch the geometry for tampering" — is **empirically dead**.
- The genuinely-productizable residue is narrower and each needs its **own** validation, not assumption:
  - **Read-out layer** (geometry lives at the final layer): a *hypothesis* for improving styxx's
    existing white-box probes — but the refusal/action probes target a different representation and were
    likely already layer-tuned; must be tested on that target, not assumed to transfer.
  - **RDM reliability as a quality/confidence signal**: research-backed (reliability drove human/brain
    alignment) but untested as an error/hallucination predictor — a real next experiment for the
    confidence-router, not a shipped claim.
  - **The RDM / RSA / lexical-control / noise-ceiling instrumentation**: reusable for any future
    representational probe.

## Discipline note
This is the styxx method applied to *product*, not just papers: an exciting feature idea (Qwen 2D AUC
1.00) was killed by a pre-stated positive control before it could ship. The losses go next to the wins.

## Reproduce
`run_geometry_integrity.py` (drift, stage 1) · `run_geometry_mismatch.py` (2D signature, stage 2) ·
`run_geometry_signature.py` (benign-behavioral confound, stage 3). Results: `geometry_*_result.json`.
All on local cached models — zero API cost.
