# PREREG — the binding census, v1 (Llama family)

**Fathom Lab · 2026-07-02 · ratified by flobi ("lets go"). Committed BEFORE any new receipts dataset is
opened. Publishes regardless of verdict.**

## Question
Rung 3 measured one model: 18/22 claims re-bind exactly. The census asks whether that is a property of one
card or of the ecosystem: across every Llama-family Instruct model whose vendor receipts (`-evals` datasets)
exist publicly on HF, what are the BINDING, FIDELITY, and SELECTION rates — per model, mechanically, third-party?

## Scope (frozen)
- Census v1 = Llama family only (the vendor that ships receipts). Arms are enumerated AT EXECUTION as: every
  Instruct model that has BOTH (a) numeric claims on an official Meta HF model card already hash-pinned or
  pinned at fetch, and (b) a public `meta-llama/<model>-evals` dataset. The enumeration is recorded in the
  results JSON; availability is a fact of the world, not a choice.
- The Llama-3.2 card's cross-citations of Llama-3.1-8B are IN scope (cross-card citation fidelity — does a
  card quoting a sibling model match THAT model's receipts?).
- The gemma card-vs-tech-report tension from rung 1 is OUT of scope for v1 (different receipt type; needs its
  own second-pass design).

## Method (frozen — inherits rung 3 verbatim)
1. Claims: mechanical table parse (`rebind_step1` parser), sha256-pinned cards. In-scope columns = the
   Instruct bf16 column + multilingual rows for each arm's model.
2. Matching: EXPLICIT per-benchmark mapping only — no fuzzy matching (rung 3 §3 hazard: fuzz produced 3
   phantom mismatches). Any new tension is FIRST treated as a hypothesis about the matcher; it is reported
   only after the mapping for that row is manually verified against the receipts' actual labels/tags.
3. Grades per claim (frozen, rung 3): MATCH (|Δ| ≤ 0.15 after normalization) / VALUE-MISMATCH / ABSENT
   (subtypes: no-data, composite-unrecomputable, config-conflict).
4. SELECTION per arm: published fraction = in-scope card claims ÷ receipts metric rows; plus a flattering-
   selection check wherever receipts offer ≥2 aggregates for one published number (count: flattering /
   unflattering / neutral choices).
5. Verdict per arm (frozen, rung 3 bars): REBOUND ≥90% MATCH · RECEIPT-MISMATCH >10% VALUE-MISMATCH ·
   else PARTIAL. Census headline = the per-arm table, no pooled verdict (arms are not exchangeable).

## Framing rule (frozen, carried from rungs 1+3)
The gaps are industry properties. No accusation language. Deltas reported with receipts; correction invited.
Selection findings are reported in BOTH directions (honest selection is as reportable as flattering).

## Falsification map
- Uniform high fidelity ⇒ receipts-shipping vendors are already provable — the census becomes the template
  for demanding binding, not a critique.
- Fidelity decays with model size/generation ⇒ the first longitudinal claim-integrity signal (seeds M7).
- Widespread orphans/config-conflicts ⇒ the BFCL/MATH class is systemic; binding is load-bearing, not nice.
