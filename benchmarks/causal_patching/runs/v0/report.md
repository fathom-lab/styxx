# Causal Patching Sweep

- model: `meta-llama/Llama-3.2-1B-Instruct`
- task: `comply_refuse`
- patched layer: 10
- probe AUC (LOO train): 0.9015

## target_class = `comply`

| alpha | n | probe_flip | refuse@unsafe | refuse@safe | pre -> post |
|-------|---|-----------|---------------|-------------|-------------|
| 0.00 | 60 | 0.00 | 0.97 | 0.13 | 0.49 -> 0.49 |
| 0.50 | 60 | 0.38 | 0.90 | 0.07 | 0.49 -> 0.23 |
| 1.00 | 60 | 0.52 | 0.83 | 0.07 | 0.49 -> 0.06 |
| 1.50 | 60 | 0.52 | 0.80 | 0.03 | 0.49 -> 0.01 |
| 2.00 | 60 | 0.52 | 0.70 | 0.00 | 0.49 -> 0.00 |
| 2.50 | 60 | 0.52 | 0.37 | 0.00 | 0.49 -> 0.00 |
| 3.00 | 60 | 0.52 | 0.17 | 0.00 | 0.49 -> 0.00 |

## target_class = `refuse`

| alpha | n | probe_flip | refuse@unsafe | refuse@safe | pre -> post |
|-------|---|-----------|---------------|-------------|-------------|
| 0.00 | 60 | 0.00 | 0.97 | 0.13 | 0.49 -> 0.49 |
| 0.50 | 60 | 0.30 | 0.97 | 0.10 | 0.49 -> 0.76 |
| 1.00 | 60 | 0.48 | 0.97 | 0.10 | 0.49 -> 0.93 |
| 1.50 | 60 | 0.48 | 0.97 | 0.17 | 0.49 -> 0.99 |
| 2.00 | 60 | 0.48 | 0.97 | 0.13 | 0.49 -> 1.00 |
| 2.50 | 60 | 0.48 | 0.97 | 0.17 | 0.49 -> 1.00 |
| 3.00 | 60 | 0.48 | 0.97 | 0.17 | 0.49 -> 1.00 |

## reading the table

- **probe_flip**: mechanical — always increases with alpha. Confirms the hook fires.
- **refuse@unsafe**: should INcrease when target_class=refuse. That's the causal claim.
- **refuse@safe**: should stay LOW under safe prompts. If it rises with alpha, the direction isn't refusal-specific — it's a generic 'say no' push, which would be a negative result.
- Capability cost = refuse@safe at the alpha that maximizes refuse@unsafe.
