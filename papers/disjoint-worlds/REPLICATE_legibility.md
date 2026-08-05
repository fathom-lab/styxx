# Replicate the mutual-legibility matrix — one command, no GPU, no model downloads

The disjoint-worlds legibility results are the **most accessible replication target in this
repo**: the four models' concept extractions are committed as small `.npz` banks, so the whole
matrix re-runs on a laptop CPU from the cloned repository — no GPU, no HuggingFace downloads, no
training. If you have Python and 20 minutes, you can independently confirm (or break) the
clique/island topology.

## Run it

```bash
git clone https://github.com/fathom-lab/styxx && cd styxx/papers/disjoint-worlds
pip install numpy scipy
python run_b37.py          # the 12-pair legibility matrix
```

Everything is deterministic (fixed seed 343, CPU) — your numbers should match the committed
`b37_result.json` to the digit. The script self-scores through `styxx.protocol` against the
frozen gates in `PREREG_b37_legibility_matrix_2026_08_04.md`.

## What you must reproduce (the anchor cells)

| quantity | committed value |
|---|---|
| llama_3b → gemma_2b discovery | 0.5918 |
| llama_3b → qwen_1p5b discovery | 0.0536 |
| the clique {llama_3b, llama_1b, gemma_2b} | all pairs discover 0.59–0.83 |
| qwen_1p5b (the island) | discovered ≤ 0.17 from every direction |

**The load-bearing claim is the topology, not any single decimal:** three models mutually
legible (label-free, cross-family), one model illegible from all sides. That structure is what a
replication confirms or breaks.

## What we already know is NOT settled (break these, earn credit)

The [FINDING](FINDING_b37_legibility_matrix_2026_08_04.md) overruled its own mechanical verdict:
the "symmetry" pass is a property of the direction-blind Procrustes machinery (three exact-tie
pairs give it away), and the kNN-vs-RSA predictor margin (0.0193 over 12 points) is a statistical
tie, **not** support for the candidate law. A replication that:

- finds a **direction-sensitive** discovery method and shows legibility is (a)symmetric *for real*,
- or names the representation property that makes qwen an island (RSA and outlier-concentration
  are both ruled out — see `b38_recon_addendum.json`),
- or reproduces the matrix on a **different set of models** and shows the clique/island split is
  (or isn't) general,

is worth more than one that matches, and earns the same named credit
([REPLICATIONS.md](../../REPLICATIONS.md)).

## Replicate the bridge (the arc's strongest results — same banks, same laptop)

The causal chain that follows the matrix is just as replicable, from the same committed `.npz`
banks, CPU-only:

```bash
python run_b41.py          # the bridge: correct the island's frame, legibility 0.06 -> 0.97
python run_b42.py          # the dose curve: 8 ranks x 5 seeds, k* = 2  (~9 h CPU)
python run_b44.py          # wrong-donor control: the clique's SHARED frame is the operative object
```

Anchor cells: b41 bridge 0.9745 against a random-frame 0.0 at k=20; b42 median bridge 0.5128
at rank 2 (half of full legibility from two directions) rising monotonically to 1.0 at rank 40,
Spearman 1.0; b44 wrong-model donor frames land medians 0.7168 (gemma_2b) / 0.648 (llama_1b) —
every donor × seed above the 0.30 floor. Each script self-scores through `styxx.protocol`
against its frozen prereg.

**The open challenge here is b43's:** the twenty causal directions have no stable, semantically
coherent concept story (Jaccard 0.1368 across seeds; coherence permutation p 0.8031). Find a
representation in which the barrier directions ARE nameable — or show the rank-2 core is a
stable geometric object across model pairs — and you have beaten our best negative.

## Extend it

`run_b38.py` runs the causal legibility-cliff experiment (noise-dose one model's geometry, watch
discovery fall) from the same banks — also CPU-only. Adding your own model is one extraction file
in `CONCEPTS` order (462 concept points) dropped into a `_b31v2_pts_<name>.npz`; the matrix code
picks up any model you register in `MODELS`. A model that lands *outside* the clique would be the
second island — and the decisive donor for splitting "shared clique frame" from "any trained
frame" (see `FINDING_b44_shared_frame_2026_08_05.md`).
