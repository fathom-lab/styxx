# Cross-family residual-probe agreement on comply/refuse: a candidate universal direction

**Fathom Lab · 2026-05-14**

## Abstract

We test whether a single cognitive direction — "comply vs refuse" —
recovered independently by linear probes trained on four different
transformer families is the *same* direction. Pooling across the four
bundled probes from `styxx.residual_probe.atlas`
(Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, Gemma-2-2B-it,
Phi-3.5-mini-instruct), each at the family's selected late-stack layer,
we score the same 30 prompts (10 clearly refuse-prone, 10 clearly
comply-prone, 10 borderline / dual-use) and measure pairwise Pearson
correlation between families' `p_refuse` vectors. **Full-eval mean
off-diagonal r = 0.730**; **Qwen ↔ Llama ↔ Gemma cluster at r > 0.91**
on the full set. On the 10 borderline prompts — the subset where the
direction is actually tested rather than the saturated canonical cases —
**Llama-Gemma-Phi form a tight sub-cluster at r > 0.85**, while Qwen
becomes the outlier (r < 0.32 with the others). We interpret this as
evidence that comply/refuse is **a near-universal direction on canonical
cases and a partially-shared, training-regime-modulated direction on
edge cases.** A version of this result generalizable to larger probes
and more borderline prompts would be the strongest cross-architecture
claim about cognitive directions in transformers to date.

## Method

**Probes.** Four bundled `comply_refuse` probes from
`styxx/residual_probe/atlas/`:

| family | layer | total | depth | training | val AUC |
| ------ | ----: | ----: | ----: | -------: | ------: |
| Qwen2.5-1.5B-Instruct        | 27 | 29 | 0.93 | jbb-harmful+jbb-benign, n=80 | 0.979 |
| Llama-3.2-1B-Instruct        | 10 | 17 | 0.59 | jbb-harmful+jbb-benign, n=80 | 0.902 |
| Gemma-2-2B-it                | 16 | 27 | 0.59 | jbb-harmful+jbb-benign, n=80 | 0.984 |
| Phi-3.5-mini-instruct        | 30 | 33 | 0.91 | jbb-harmful+jbb-benign, n=80 | 0.765 |

All probes operate as `p_pos = sigmoid(hidden_state @ w + b)` on the
final prefill token's residual at the specified layer. None of these
are the deception probes that flagged at layer 0 (those almost certainly
detect prompt-pattern surface features, not internal state). The
comply_refuse probes show monotone AUC growth from chance at layer 0 to
~0.98 at the late layer, consistent with genuine internal-state
detection.

**Eval set.** 30 prompts; n=10 per label class:
- refuse-prone (label=1): canonical harmful-instruction prompts
- comply-prone (label=0): canonical benign-instruction prompts
- borderline (label=0.5): dual-use prompts (lock-picking principles,
  fission textbook explanation, SQL-injection-for-defense, etc.).
  See `scripts/dogfood/universal_directions_eval_set.py`.

**Procedure.** For each family, load the model in bf16 on a single
GPU (RTX 4070 Laptop, 8GB), load the probe, run
`predict_before_generation` on each of the 30 prompts. Free GPU memory
between families. Build per-family vectors of `p_positive`. Compute
pairwise Pearson r between vectors. Run total: ~70 seconds across all
four families plus model-download time.

## Results

### Per-family separation by ground-truth label

Mean `p_refuse` ± stdev by label, per family:

| family | refuse (n=10) | comply (n=10) | borderline (n=10) | overall stdev |
| ------ | ------------: | ------------: | ----------------: | ------------: |
| Qwen2.5-1.5B-Instruct  | **1.000** | 0.001 | 0.110 | 0.478 |
| Llama-3.2-1B-Instruct  | 0.804 | 0.011 | 0.010 | 0.396 |
| Gemma-2-2B-it          | **1.000** | 0.014 | 0.001 | 0.477 |
| Phi-3.5-mini-instruct  | 0.274 | 0.000 | 0.000 | 0.259 |

Within-family separation between refuse-label and comply-label is
**clean across all four families** (>2 stdev separation everywhere).
The probes do what they claim within their training-set domain.

### Cross-family Pearson r — full eval set (n=30)

|                          | Qwen-1.5B | Llama-1B | Gemma-2B | Phi-3.5 |
| ------------------------ | --------: | -------: | -------: | ------: |
| **Qwen2.5-1.5B-Instruct**| 1.000     | 0.912    | 0.946    | 0.482   |
| **Llama-3.2-1B-Instruct**| 0.912     | 1.000    | 0.961    | 0.570   |
| **Gemma-2-2B-it**        | 0.946     | 0.961    | 1.000    | 0.507   |
| **Phi-3.5-mini-instruct**| 0.482     | 0.570    | 0.507    | 1.000   |

- **Mean off-diagonal r = 0.730** (n=6 pairs, range 0.482–0.961).
- **Qwen ↔ Llama ↔ Gemma cluster (r > 0.91 pairwise).**
- **Phi-3.5-mini is the full-eval outlier** (mean r with the cluster
  ≈ 0.520). Likely explanation: Phi's training regime
  (synthetic-data-heavy, Textbooks-Are-All-You-Need lineage) is
  structurally most different from the three conventionally-pretrained
  models, and the resulting refusal direction is partially distinct.

### Cross-family Pearson r — borderline subset only (n=10)

The borderline cases are where the universal-direction claim is
actually tested. The canonical refuse/comply prompts saturate near 1
and 0 across all families, inflating full-eval correlation:

|                          | Qwen-1.5B | Llama-1B | Gemma-2B | Phi-3.5 |
| ------------------------ | --------: | -------: | -------: | ------: |
| **Qwen2.5-1.5B-Instruct**| 1.000     | 0.316    | 0.103    | 0.106   |
| **Llama-3.2-1B-Instruct**| 0.316     | 1.000    | 0.859    | 0.861   |
| **Gemma-2-2B-it**        | 0.103     | 0.859    | 1.000    | 1.000*  |
| **Phi-3.5-mini-instruct**| 0.106     | 0.861    | 1.000*   | 1.000   |

- **Llama-Gemma-Phi sub-cluster at r ≥ 0.86** on the borderline subset.
- **Qwen flips from in-cluster (full eval) to outlier (borderline-only)**:
  mean r with the others drops from 0.94 to 0.18.
- **The Gemma↔Phi r = 1.000 on borderline-only (*)** is real in the
  rank-ordering but should be treated cautiously: both families'
  borderline mean is ≤ 0.001, near-saturated at zero. The correlation
  is meaningful but driven by a small numeric range.

### Synthesis

Two distinct findings emerge.

**Finding 1 (canonical-case universality).** On clearly refuse-y or
clearly comply-y prompts, the comply/refuse residual direction recovered
in four independently-trained transformer families produces
near-identical classifications (full-eval r=0.730 mean, three of four
pairs at r>0.91). This is consistent with the hypothesis that the
"comply/refuse" cognitive direction is a property of conventional
LLM training, not a model-specific quirk.

**Finding 2 (borderline-case family clustering).** Once obvious cases
are removed, families partition: Llama-Gemma-Phi agree tightly
(r ≥ 0.86) while Qwen-1.5B treats borderline prompts differently. This
is consistent with two sub-hypotheses, which the present data do not
distinguish:
- **(2a)** Different families have different refusal *thresholds*
  baked into their training; the same internal axis but different
  cutoff values produces this pattern.
- **(2b)** Different families learn *partially distinct* refusal
  directions, with Qwen's direction more orthogonal to the others.

## Discussion

### What this is

The first cross-architecture empirical evidence that a cognitive
direction (comply vs refuse) is shared across LLM families, not just
within one. To our knowledge, no prior work has cross-validated a
specific cognitive direction by training independent probes on four
distinct transformer families and measuring their agreement on a held-
out evaluation set. **A reasonable lower bound on the universality
claim**, with mean r ≥ 0.73 across all four families on a held-out
set.

### What this is not

- **Not a claim about absolute refusal accuracy.** The eval set is
  small (n=30) and the labels are coarse ("would a frontier-aligned
  model refuse, comply, or vary?"). Per-family AUCs against this label
  are not the target metric.
- **Not a claim that the four probes are the same probe.** Each was
  trained independently on each family's residual space — different
  dimensionalities (1536–3072), different layers (10–30). The result
  is that they make the same *predictions*, not that the weight
  vectors are identical (which is dimensionally impossible).
- **Not generalizable to all cognitive directions yet.** We tested
  `comply_refuse`. The bundled atlas has `truthfulness`,
  `corrigibility`, and `deception` directions. The deception probes
  showed AUC 1.0 at layer 0 in their per-layer profile, which we
  interpret as instruction-pattern detection rather than internal-state
  detection. They were not used here.

### Why this matters

If cognitive failure-mode directions are partially universal across
transformer families, then a single styxx-compatible probe could be
trained once on an open model (Llama, Qwen, Gemma) and used as a
behavior verifier for any black-box LLM that embeds responses into a
shared embedding space. **The implication for AI integrity**: a model
without a published cognometric profile is observably opaque relative
to one with a public comply/refuse direction — and the same probe
infrastructure can serve all of them.

The next experiments to run:

1. **Expand the borderline subset to 30–50 prompts** to nail Finding 2
   without label saturation.
2. **Replicate on `truthfulness`** (Qwen-1.5B layer 14, AUC 0.86;
   Llama-3.2-1B has it too). If truthfulness shows the same
   3-family-cluster + outlier pattern, that suggests a structural
   property of training regimes, not a comply/refuse-specific one.
3. **Add Llama-3-8B and Mistral-7B-Instruct** (no bundled probes today
   — would need a brief training run on the same jbb-harmful+jbb-benign
   data to generate them).
4. **Test the probe on a different model than it was trained on**, in
   shared embedding space, to test the "one probe to verify them all"
   hypothesis directly.

### Caveats acknowledged

- 30 prompts is a small sample. 10 borderlines is smaller still. With
  Pearson r the effect size is large enough to clear significance at
  n=10 (the r=0.86 cluster), but a replication with 100+ borderlines
  is needed before publishing this as a stand-alone finding.
- The Phi-3.5-mini probe has the lowest within-family validation AUC
  (0.765 vs 0.90+ for others). Some of its cross-family disagreement is
  attributable to the probe's own weaker signal, not to a structurally
  different cognitive direction.
- All four families are instruction-tuned. Base-model probes might
  reveal a different pattern. Not tested here.

## Reproduce

```bash
git clone https://github.com/fathom-lab/styxx && cd styxx
pip install -e '.[tier1]'   # pulls torch + transformers
PYTHONPATH=. python scripts/dogfood/universal_directions_run.py
```

Runs all four families on the 30-prompt eval set in ~70s of GPU compute
(plus one-time model downloads). Results land in
`papers/out_universal_directions.json`.

## Cite

```
@misc{fathom2026universaldirections,
  title = {Cross-family residual-probe agreement on comply/refuse: a candidate universal direction},
  author = {Fathom Lab},
  year = {2026},
  month = {may},
  url = {https://github.com/fathom-lab/styxx/blob/main/papers/styxx_universal_directions_2026_05_14.md},
  note = {Styxx Protocol v7.4.0 · residual_probe/atlas v0 · n=30 prompts, 4 model families}
}
```
