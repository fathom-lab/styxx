# Cross-Scale Cognitive Transfer — comply_refuse

- **A**: `meta-llama/Llama-3.2-1B-Instruct` (hidden=2048, layers=17)
- **B**: `meta-llama/Llama-3.2-3B-Instruct` (hidden=3072, layers=29)

## Best-AUC layer

| model | best layer | total | fraction | AUC |
|---|---|---|---|---|
| A | 10 | 17 | 0.62 | 0.9015 |
| B | 26 | 29 | 0.93 | 0.9975 |

Δ fraction = 0.30. Fractional best-layer differs across scales — not strongly scale-invariant.

## Emergence bands — earliest layer where AUC ≥ threshold

| model | AUC≥0.7 | AUC≥0.8 | AUC≥0.9 |
|---|---|---|---|
| A | L4-L16 | L7-L16 | L10-L10 |
| B | L4-L28 | L6-L28 | L9-L28 |

## Per-layer AUC (normalized layer fraction)

| fraction | A | B |
|---|---|---|
| 0.0 | 0.000 (L0) | 0.000 (L0) |
| 0.1 | 0.544 (L2) | 0.682 (L3) |
| 0.2 | 0.617 (L3) | 0.821 (L6) |
| 0.3 | 0.789 (L5) | 0.883 (L8) |
| 0.4 | 0.799 (L6) | 0.938 (L11) |
| 0.5 | 0.876 (L8) | 0.980 (L14) |
| 0.6 | 0.902 (L10) | 0.990 (L17) |
| 0.7 | 0.895 (L11) | 0.987 (L20) |
| 0.8 | 0.881 (L13) | 0.985 (L22) |
| 0.9 | 0.871 (L14) | 0.996 (L25) |
| 1.0 | 0.842 (L16) | 0.997 (L28) |

*(α-sweep data not provided for both models; run run_patching.py on each for behavioral comparison.)*
