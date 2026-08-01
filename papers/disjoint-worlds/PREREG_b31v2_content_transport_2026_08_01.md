# PREREG — B31-v2: heavy-machinery content transport (the telepathy door, opened or closed with teeth)

Fathom Lab · 2026-08-01 · frozen before any new data. The documented out-of-scope research bet
(PROGRAM_BACKLOG B31), licensed by two committed results it must now separate:

1. **Cycle-6 CONTENT-WEAK:** cross-model concept identity collapsed to chance through a
   label-free LINEAR map (anchor R² ≈ 0.06, 40 anchors).
2. **Rung-2 (2026-06-22):** the cleared instrument reads same-family at top-1 0.586 (41×
   chance) but cross-family at only 4–5× chance — and the highest-isometry target
   (gemma-2-2b, RSA 0.955) reads at EXACT chance. Isometry does not grade readability; the
   smooth-degradation prediction was falsified and recorded.

The question this run decides: **is the cross-family cliff a map-capacity limit, or bedrock?**
If a higher-capacity map opens gemma, cross-model content reading is an engineering problem
(the door opens). If gemma stays at chance while the same map class lifts the same-family
anchor, the cliff is a property of what the two representations share — the strongest
evidence yet that content legibility across families is NOT purchasable with map capacity at
this scale (the door closes, with teeth).

## Design (frozen)

- **Source:** Llama-3.2-3B-Instruct. **Targets:** gemma-2-2b-it (the decisive null cell),
  Qwen2.5-1.5B-Instruct (the 4× cell), Llama-3.2-1B (same-family positive-control anchor).
- **Anchors:** the committed N=462 concept battery (the G0-clear set), minus the 70 frozen
  held-out concepts — anchors fit the map, held-outs are never seen in fitting. Same
  last-token hidden-state extraction as rung 2, layers per the committed read-optimal sweep.
- **Map classes, run in order on identical splits:**
  - M0 linear ridge (the rung-2 class) — replication control;
  - M1 two-layer MLP adapter (hidden 2× source dim, GELU, ridge-regularized), fit on PAIRED
    anchors. Honest scope: pairing departs from label-free purity by design — B31 has always
    been the heavy-machinery bet; a positive here bounds the CEILING of content transport,
    not the label-free protocol.
- **Readout:** zero-shot top-1 concept identification on the 70 held-outs (nearest mapped
  anchor centroid, cosine), chance = 1/70 ≈ 0.0143 — the rung-2 metric unchanged.

## Gates (frozen; no optional stopping)

- **G0 (machinery):** M1 same-family top-1 ≥ 0.53 (the committed 0.586 minus 0.05 slack).
  Fails → `INVALID__map_class_broken`; nothing else is interpretable.
- **G1 (the bet, decisive cell):** gemma-2-2b M1 top-1 ≥ 0.143 (10× chance; binomial
  p < 1e-6 at n=70) AND ≥ 5× its M0 value. Pass → `DOOR_OPENS__content_capacity_limited`.
- **G2 (specificity null):** a pairing-shuffled M1 (same architecture, shuffled anchor
  correspondence) stays ≤ 2× chance on every target. Fails → the lift is architecture
  artifact, G1 cannot be claimed.
- **G3 (the closed-negative branch, pre-committed):** G0 passes, G2 passes, and gemma M1
  < 0.143 → `DOOR_CLOSES__cliff_not_capacity_limited_at_this_class` — a full result, not a
  miss: paired supervision plus a nonlinear map cannot buy cross-family content legibility
  where isometry is near-perfect. Scope: this map class, these scales; a vec2vec-grade
  cycle-consistent map is the named successor and needs its own prereg.
- **Qwen cell reported both ways (context, not gated).**

## Compute & discipline

$0, local 8 GB GPU (extraction ≈ 4 models × ~530 prompts; MLP fits are CPU-trivial).
Checkpointed extraction per model; smoke on 5 concepts writes to `_smoke`-suffixed files
only. Every number in the FINDING re-derives from `b31v2_result.json`; OATH-certified before
commit; the run happens in the NEXT firing, not the one that froze this document.
