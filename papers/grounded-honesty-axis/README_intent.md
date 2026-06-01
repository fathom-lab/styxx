# the intent / interoception toolkit

Read a model's **intent** — a deliberate sycophantic lie (it knew, then caved) vs an honest mistake (it
never knew) — from its **own residual stream**, where the output text can't tell them apart. Then wire that
read into a control loop so an agent **catches itself caving and reverts to the truth it knew.** Fully
local, $0, white-box. Every claim pre-registered before the data; receipts on disk.

## headline results (see `FINDING_intent_discriminator_2026_05_31.md`)

**Intent beyond confidence** — confidence pinned to chance, the residual still calls the lie:

| family | matched surface | intent AUROC | contrast |
|---|---|---|---|
| Qwen2.5-3B | 0.506 | 0.745 | +0.239 |
| Llama-3.2-3B | 0.580 | 0.796 | +0.216 |
| gemma-2-2b | 0.518 | 0.813 | +0.295 |

A property of transformers, not one model. **Attenuates with capability** (ρ≈−0.8; the 0.5B wears its lie
almost perfectly inside, ~0.6 by 7B — a falsified prediction, reported).

**Interoception loop** — agent reads its own activations under pressure, reverts the caves:

| family | baseline acc under pressure | with interoception | precision | recall |
|---|---|---|---|---|
| Qwen2.5-3B | 0.203 | 0.473 (+0.270) | 0.99 | 0.81 |
| Llama-3.2-3B | 0.030 | 0.253 (+0.223) | 1.00 | 0.55 |
| gemma-2-2b | 0.013 | 0.273 (+0.260) | 1.00 | 0.67 |

(n=300 held-out per family, deployed threshold 0.3)

## pipeline

```
# 1. generate an intent set (neutral pass establishes knowledge; sycophantic pass captures residuals)
python gen_intent_set.py --model Qwen/Qwen2.5-3B-Instruct --skip 1200 --n 700 --tag bc2
#    cross-family: --model meta-llama/Llama-3.2-3B-Instruct  /  google/gemma-2-2b-it  (--load-4bit for 7B)

# 2. score intent-beyond-confidence (margin-bin-balanced = confidence matched to chance)
python score_intent_bc.py --tag bc2

# 3. train + persist the cave/override probe (the reusable READ organ)
python train_intent_probe.py --tags full,bc,bc2 --out intent_probe          # Qwen
python train_intent_probe.py --tags xf_llama   --out intent_probe_llama      # any family

# 4. interoception loop: dogfood the read->act loop on held-out items (threshold sweep + net accuracy gain)
python interocept.py --probe intent_probe --n 300 --skip 2000
```

Live use:

```python
from interocept import Interoceptor
io = Interoceptor(probe="intent_probe")          # loads model + persisted probe
r = io.answer_under_pressure(question, choices, asserted_idx)
# -> {neutral, pressured, cave_prob, flagged, final}  ; final reverts to the unpressured answer if it caved
```

## persisted probes (deployable, tiny)

`intent_probe{,_llama,_gemma}.{npz,json}` — `mean/scale/coef/intercept/layer`, threshold 0.3. The live
loop reads the model id + layer from the json, so the probe is self-describing.

## pre-registrations (bars locked before data)

`PREREG_intent_discriminator` · `PREREG_intent_beyond_confidence` (+ `_CONFIRM`) ·
`PREREG_intent_capability_ladder` · `PREREG_intent_cross_family`

## honest scope

Operationalizes "lie" as **sycophantic override** (knew-then-caved), not all deception. Letter-MCQ truth
token; **linear** probe — a separating direction, not proven intent. Validated on the sycophantic-MCQ
scenario (other pressure types / free-form untested). Recall ~0.6–0.8 (misses some caves). Within-Qwen
ladder is n=4 (low power). Correlational. It is a real, modest, cross-family primitive — **not** "AI lie
detection, solved."
