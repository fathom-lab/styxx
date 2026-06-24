# PREREG — grounding holds where text is at chance, on a verified-silent controlled construct

**Frozen 2026-06-24 BEFORE extracting activations. Silence already established BLIND to activations.**
Offline, local-GPU, NO frontier key.

## Construct (built + silence-verified)
`scripts/build_controlled_truthset.py`: 136 template-matched true/false factual statements, 6 domains
(capitals, elements, planets, currencies, authors, languages). Cyclic-derangement false answers → each answer
token appears once as TRUE (own entity) and once as FALSE (another) → answer-token balance 0.99.
**Silence GATE PASSED (pre-extraction):** adversary-fair (max(auc,1-auc)) bag-of-words classifier under
LEAVE-ONE-DOMAIN-OUT = **0.505** (exactly chance). Text cannot read truth across domains.

## Test
Reader-model last-token residual activations (Llama-3.2-3B + Qwen2.5-3B) on each statement. Linear probe,
true vs false, evaluated **leave-one-domain-out** (train on 5 domains, test on the held-out 6th — no shared
template/entity/answer token; this is harder than the silence baseline and tests a DOMAIN-GENERAL truth
direction).

## Decision thresholds (FROZEN)
| Verdict | Condition |
|---|---|
| **GROUNDING HOLDS (text-silent)** | activation leave-one-domain-out AUC **≥ 0.75** AND ≥ BoW + 0.15 AND label-shuffle ≈ 0.5 AND PCA-50 holds — on BOTH models |
| **HONEST NULL** | activation AUC < 0.65 (grounding also fails to transfer across domains) |
| **PARTIAL** | one model only, or 0.65–0.75 |

## Honest scope (stated up front, not as a retreat)
- This demonstrates **reader-side** decodability of truth from activations on a verified-silent construct.
  The PHENOMENON (LLM activations linearly encode truth, domain-generally) is KNOWN — Azaria & Mitchell 2023,
  Marks & Tegmark 2023, Bürger et al "Truth is Universal" 2024. **styxx's contribution here is NOT the
  phenomenon** but: (a) the adversary-fair leave-one-domain-out SILENCE GATE proving text is exactly at
  chance, (b) the control battery, (c) framing it as the oversight pillar ("you cannot read truth from the
  words; you must read the representation"). This is a rigorous, legible baseline, honestly labeled as such.
- The GENUINELY NOVEL, unclaimed escalation (separate prereg): GENERATION-TIME deceptive INTENT (the
  speaker's own state when asserting a known falsehood, vs reader-side factuality) and CROSS-MIND probing.
  This construct is the substrate for those.
- n=136, 2 local 3B models, single seed, linear probes. Verdict mechanical.
