# PREREG — cross-mind truth transfer: can model A's truth direction oversee model B's statements?

**Frozen 2026-06-24. To run COLD next session. Offline, local-GPU, NO frontier key.** The genuinely
unclaimed escalation of `FINDING_truthset_grounding` (within-model truth is a rank-1 direction; reader-side
truth-probing is known). Cross-mind oversight — read A to judge B — is styxx's distinctive lineage
([[project_crossmind_write_channel]], `project_linear_transport_finding`).

## Question
On the verified-silent controlled truth-set (`controlled_truthset.jsonl`, BoW leave-one-domain-out 0.505),
does the truth direction learned in model A, transported into model B's activation space, classify B's
statements — beyond what a transported RANDOM direction achieves? If yes: truth is not just decodable
within a mind but a SHARED structure one mind can use to oversee another. That is the novel claim.

## Why it cannot be rushed (the constraint that sets the design)
Qwen (2048-dim) and Llama (3072-dim) live in different spaces; transport needs a learned linear map. Fit on
~113 leave-one-domain-out samples mapping 2048→3072, the map is hugely underdetermined and overfits — a naive
run is uninterpretable. The design MUST fix this BEFORE running:
- **Scale the construct to n≈600–1000** statements (expand `build_controlled_truthset` to ~20 domains) so the
  alignment map is identifiable. Keep the cyclic-derangement silence property; re-verify BoW LODO ≤ 0.55.
- **Alignment method:** label-free paired linear transport (orthogonal Procrustes on paired activations of the
  SAME statements), fit on TRAIN domains only, per the program's validated `linear_transport` method — NOT a
  free ridge map.

## Protocol (frozen)
1. Extract paired activations (same statements) from Qwen + Llama at each model's best truth layer.
2. Leave-one-domain-out: on train domains, (a) fit Procrustes map A→B on paired activations, (b) fit A's
   mass-mean truth direction, (c) transport it into B; classify B's HELD-OUT-domain statements.
3. Both transfer directions (Qwen→Llama and Llama→Qwen).

## Controls (frozen, all required for a positive claim)
- **Transported-RANDOM baseline:** a random unit direction in A transported the same way → must be ≈ chance
  (proves the MAP isn't doing the work).
- **Shuffle:** shuffle truth labels before fitting A's direction → transported result ≈ chance.
- **Within-B ceiling:** B's own truth direction LODO (the upper bound transfer is measured against).
- **BoW-transport floor:** the same transport on bag-of-words features → must stay at chance (text has no
  transferable truth structure to move).

## Decision thresholds (FROZEN)
| Verdict | Condition |
|---|---|
| **CROSS-MIND TRANSFER** | transported-truth LODO ≥ 0.70 AND ≥ transported-random + 0.15 AND shuffle ≈ 0.5 — BOTH directions |
| **HONEST NULL** | transported-truth ≈ transported-random (the map, not shared truth, explains any signal) — bounds the cross-mind claim |
| **PARTIAL** | one direction only |

## Sibling escalation (separate prereg when this lands)
**Generation-time deceptive INTENT:** prompt a model to ASSERT each false statement as true; probe its own
generation-time state vs honestly asserting a true one. Tests the speaker's lie betraying itself beyond
reader-side content. Needs surface-matched assertions (the controlled set provides them) + generation-time
residual capture.

## Honest scope
In-silico, local 3B models. The transported-random + shuffle + BoW-transport controls are what separate a
real shared-truth result from "linear transport is powerful." Reader-side within-model truth is known; CROSS
-MIND oversight on a silence-verified construct, with these controls, is the contribution. Run cold, fresh
context — not at the tail of a long session.
