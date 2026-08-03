# PREREG — B34-v3: label-free cross-family content READ, discovery + single nonlinear lens

Fathom Lab · 2026-08-03 · frozen before the scored run. The parked telepathy family
(`PROGRAM_BACKLOG` B34), unblocked: the no-third-same-day-patch clause was written 2026-08-01;
this is a fresh session. Carries a machine-readable `gates` block — the first prereg
`styxx.protocol` can score mechanically.

## What the two dead attempts measured

- **v1** (`INVALID__pipeline_broken`): a raw one-shot GW seed underperformed the committed
  annealed machinery (same-family seed accuracy 0.066).
- **v2** (`INVALID__pipeline_broken`): the initializer swap fixed the seed — the committed
  linear machinery **discovers gemma's cross-family pairing at 0.612 seed accuracy, label-free**
  — but G0 still failed because the bolted-on MLP *iteration loop* **degraded** the same-family
  read (0.3429 linear → 0.2000). The iteration was the bug, not the thesis.

Both left an **UNLICENSED** observation: through the linear-discovered pairing, a nonlinear read
scored gemma 0.6286 (44× chance) and qwen 0.3429 (24×), nulls clean. v3 exists to license — or
fail to license — the cross-family READ under a valid machinery gate, **on concepts no map has
seen**.

## The one design (frozen)

- **Data:** the committed b31v2 extractions (`_b31v2_pts_*.npz`, all 462 concepts).
- **Honesty rail — FRESH SPLIT:** partition the 462 concepts with a NEW seed (343), last 70 as
  held-out, the rest as anchors. Because v1/v2's 0.6286 was measured on the `split_concepts(0)`
  held-out set, the v3 held-out concepts are **disjoint in membership** from the glimpsed
  numbers — a result here is earned on unseen items, not retrofitted.
- **Method (no iteration — the v2 fix):** (1) shuffle the target's anchor rows by a seeded
  permutation (aligned-order leakage impossible); (2) discover the correspondence with the
  committed linear machinery `TransferMap.fit` (GW warm start + annealed Procrustes + restarts —
  unchanged); the pseudo-pairing is the assignment its fitted map induces. (3) fit ONE b31v2 MLP
  on the linear-discovered pseudo-pairs (seed 343); (4) read the 70 held-out concepts (nearest
  mapped-anchor centroid, cosine — the committed metric). Chance = 1/70 ≈ 0.0143.
- **Arms:** M-LF (discover + single MLP read) on llama_1b / gemma_2b / qwen_1p5b; N (pairing-
  shuffled MLP, same architecture/training, correspondence destroyed) per target.
- Labels touch NOTHING but held-out scoring and a reported seed-accuracy diagnostic.

## Gates (frozen; scored by styxx.protocol against the result dict)

```gates
{"gates": {"G0_discovery": {"metric": "targets.llama_1b.seed_acc", "op": ">=", "value": 0.30},
           "G2_null": {"metric": "max_shuffled_top1", "op": "<=", "value": 0.0286},
           "G1_bar": {"metric": "targets.gemma_2b.read_top1", "op": ">=", "value": 0.143}},
 "outcomes": [{"when": {"G0_discovery": false}, "verdict": "INVALID__discovery_broken"},
              {"when": {"G0_discovery": true, "G2_null": false}, "verdict": "INVALID__null_fired"},
              {"when": {"G0_discovery": true, "G2_null": true, "G1_bar": true}, "verdict": "TELEPATHY_READ_BAR_CLEARED__labelfree_pairing_reads_crossfamily"},
              {"when": {"G0_discovery": true, "G2_null": true, "G1_bar": false}, "verdict": "PAIRING_READS_BELOW_BAR__labelfree"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

- **G0_discovery:** same-family linear seed accuracy ≥ 0.30 — proves the discovery machinery
  works (justified by mechanism: v2 measured 0.235 same-family / 0.612 gemma; the floor is the
  discovery-works bar, NOT a read-improvement bar — the v2 mistake, corrected).
- **G1_bar:** gemma cross-family held-out read ≥ 0.143 (10× chance) on the **fresh** split.
- **G2_null:** the worst pairing-shuffled read across targets ≤ 2× chance.

## Outcome reading (scope stated up front)

`TELEPATHY_READ_BAR_CLEARED` means: with zero labels in fitting, the correspondence between two
model families is discoverable from geometry alone AND a nonlinear lens reads held-out content
through it far above chance. This is the READ half of the telepathy-shaped claim, label-free —
NOT write (b36 settled that control does not cross), NOT same-scale-only (three targets, two
families), and bounded to this map class + these ≤3B models. `PAIRING_READS_BELOW_BAR` is a full
closed negative: discovery works but the label-free read does not clear the bar on unseen items.

## Discipline

CPU-from-cache, zero model loads (extractions banked) — immune to the shard-load kill class.
Deterministic seeds. Smoke = 40 anchors / 10 held-out, `_smoke` files, INVALID-only. Every
number re-derives from `b34v3_result.json`; OATH-certified before commit; scored through
`styxx.protocol.Experiment(this_prereg).score(result)` so the verdict is mechanical.
