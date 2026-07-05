# PREREG — M1 rung 1: can a frontier model card be certified?

**Fathom Lab · 2026-07-02 · committed BEFORE any model card is read. Publishes regardless of verdict.**

## Question
The Oath Economy moonshot (M1, VISION_MOONSHOTS → docs/governance/) begins with one rung: certify ONE public
model card against its published eval receipts, and publish the first OATH-HELD — or honestly OATH-FAILED —
model card, with the binding gaps mapped. This prereg freezes the criteria before any card is examined.

## Method (frozen)
1. Candidates: the official HuggingFace model cards of three small open models this lab already uses —
   `meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-2-2b`, `Qwen/Qwen2.5-1.5B-Instruct`. The card with the
   MOST numeric claims is certified (ties → most-documented evals). One card only; others are later rungs.
2. Claim extraction: `styxx.extract_claims` on the card text (the shipped extractor, reused verbatim), plus a
   manual sweep for claims the extractor misses (both counts reported; misses are instrument findings).
3. Binding criteria — a numeric claim is **BOUND** iff the card (or a page it directly links) provides enough
   to re-run it: named harness + version + config (shots/prompt/metric variant) + either code or raw outputs.
   Partial bindings are graded: NAMED-ONLY (benchmark named, no config), CONFIG-NO-CODE, UNBOUND.
4. Verdict — **OATH-HELD** iff ≥ 90% of numeric eval claims are BOUND. Otherwise **OATH-FAILED**, published
   with the per-claim gap map. No middle verdict.

## Framing rule (frozen)
The finding is about the BINDING GAP — the distance between public claims and re-runnable receipts — not about
any vendor's honesty. No accusation language. The gap is an industry property; naming it is the product.

## Falsification map
- OATH-HELD → frontier cards can already carry proofs; the rung becomes a template, not a critique.
- OATH-FAILED → the founding artifact of proof-carrying everything: the first quantified model-card gap map.
- Extractor misses > 20% of manually-found claims → an instrument finding against extract_claims, filed first.
