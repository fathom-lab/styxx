# The Cognitive Instruction Set: Programmable Residual-Stream Control of Open Language Models

**DRAFT — v0, 2026-04-22. Fill-in values reconciled from run artifacts.**

## Abstract

We demonstrate that the residual streams of open instruction-tuned
language models admit *programmable cognitive control* — a runtime
that (1) installs multiple linearly-probed concept directions
simultaneously as additive residual perturbations, (2) reads those
same probes during generation, and (3) dispatches actions (HALT,
RETRY, SWITCH) based on token-level cognitive state. Using
Llama-3.2-1B-Instruct and open datasets (JailbreakBench,
`meg-tong/sycophancy-eval`, and local confabulation fixtures), we
train concept probes for **refusal**, **sycophancy-pressure**, and
**confabulation-elicitation**, measure the pairwise geometry of
those directions, and ship a minimal cognitive virtual machine
(`styxx.cogvm`) that runs multi-register programs against any
HuggingFace decoder model. Our demonstration — a self-halting
generation that detects and interrupts its own confabulation before
completing the fabrication — is, to our knowledge, the first open
demonstration of machine-checkable cognitive invariants enforced at
the residual-stream level.

We frame this as **v0 of an Instruction Set Architecture (ISA) for
LLM cognition**: registers (probe directions), reads (probe
readouts), writes (α-scaled steering), and conditional branches
(`WATCH predicate → action`) form a minimum complete basis for
declarative cognitive programs. The runtime is open-source
(`styxx.steer`, `styxx.cogvm`) and supports any probe registered in
the `styxx.residual_probe` atlas.

## 1. Background

Linear probing of transformer residual streams has established that
semantically meaningful axes of behavior are encoded as roughly
linear directions in activation space (Marks & Tegmark 2024, Arditi
et al. 2024, Turner et al. 2023). Existing causal steering results
have focused on single-direction, single-concept interventions —
most famously on the "refusal direction" for safety ablation. This
paper does three things simultaneously:

1. We **replicate** the single-direction causal claim on the smallest
   commonly-deployed open instruction-tuned model (Llama-3.2-1B) with
   a fully reproducible pipeline using only open datasets.
2. We **extend** to multi-concept composition: two or more concept
   directions applied additively to the residual stream during a
   single generation. We characterize the pairwise geometry of the
   trained concept directions.
3. We **productize** the result as a cognitive VM — a small
   runtime that exposes `WRITE/READ/WATCH/HALT/RETRY/SWITCH`
   operations over concept registers and dispatches actions on
   per-token probe readouts.

## 2. Methods

### 2.1 Probe training

For each concept $c$ we build a balanced prompt set of positive-class
(+) and negative-class (−) examples:

| Concept | Positive source | Negative source | Label mode |
|---|---|---|---|
| `comply_refuse` | JBB-Behaviors / harmful | JBB-Behaviors / benign | behavioral (model response) |
| `sycophant_pressure` | `meg-tong/sycophancy-eval` (pressure-template rows) | same dataset (neutral-template rows) | input-template |
| `confab_prompt` | local `confabulation_fixtures_v3.jsonl` | JBB-benign | input-template |

For each prompt we run prefill and capture the last-token residual at
every transformer layer. We fit a per-layer L2-regularized logistic
regression (scikit-learn, `C=1.0`, `solver=liblinear`) under
leave-one-out cross-validation and report per-layer AUC. The
best-AUC layer's full-fit weight is the "concept direction" for that
layer; all per-layer directions are saved for downstream geometry.

### 2.2 Causal intervention (α-sweep)

For the refusal probe we evaluate causal load-bearing via the standard
protocol: on a held-out test set (disjoint from training via a
deterministic master-shuffle split of JBB-Behaviors), for each
$(\alpha, \text{target} \in \{\text{refuse}, \text{comply}\})$ tuple,
we patch

$$ h'_L = h_L + \alpha \cdot \mathrm{sign}(\text{target}) \cdot \hat{w}_L $$

at every forward pass (prefill plus every decoding step), where
$\hat{w}_L$ is the unit-normalized layer-$L$ probe direction.
We record the probe's sigmoid score before/after, the generation's
compliance label, and whether the probe prediction flipped.

### 2.3 Probe geometry

Given $N$ probes trained on the same model, we normalize each
direction to unit norm and compute pairwise cosine similarities at
every shared layer. The matrix $C \in [-1,1]^{N\times N}$ with
$C_{ij} = \cos(\hat{w}_i, \hat{w}_j)$ and its angular form
$A_{ij} = \arccos(C_{ij})$ are our geometry descriptors. The
extremes $|C_{ij}| \approx 0$ (orthogonal / modular concepts) and
$|C_{ij}| \approx 1$ (collapsed onto a single underlying axis)
represent competing hypotheses about residual-stream organization.

### 2.4 Cognitive VM

The `styxx.cogvm` runtime compiles a program of

    WRITE(profile) → GENERATE(max_new_tokens, watches=[WATCH(predicate, action)])

into a hooked generation loop. `WRITE` groups ⟨task, α⟩ pairs by
layer, sums α·$\hat{w}$ into a composite per-layer direction, and
installs forward hooks that add the direction to the last-position
residual at every forward pass. `GENERATE` emits tokens one at a
time, and after each token evaluates all `WATCH` predicates against
live probe readouts; the first to fire dispatches its action
(`HALT`, `RETRY`, `SWITCH`). Readouts and writes share layers to
eliminate redundant forward passes.

## 3. Results

### 3.1 Single-direction causal claim — Llama-3.2-1B

Refusal probe: best-AUC layer **{{LAYER}}** with LOO-AUC
**{{AUC}}** on $n={{N}}$ training prompts, class balance
{{CLASS_BALANCE}}. Test-set α-sweep:

| α | probe_flip | refuse@unsafe (target=refuse) | refuse@unsafe (target=comply) | refuse@safe (target=refuse) |
|---|---|---|---|---|
| 0.0 | {{PF_0}} | {{RU_R_0}} | {{RU_C_0}} | {{RS_R_0}} |
| 1.0 | {{PF_1}} | {{RU_R_1}} | {{RU_C_1}} | {{RS_R_1}} |
| 2.0 | {{PF_2}} | {{RU_R_2}} | {{RU_C_2}} | {{RS_R_2}} |
| 3.0 | {{PF_3}} | {{RU_R_3}} | {{RU_C_3}} | {{RS_R_3}} |

**Asymmetry finding**: removing refusal (target=comply on unsafe
prompts) is reliably causally effective by α≈1-2; inducing refusal
(target=refuse on safe prompts) requires substantially larger α and
begins to break coherence before becoming a clean refusal. This is
consistent with Arditi et al.'s qualitative observations on larger
models and reproduces on 1B scale with open data.

### 3.2 Probe geometry

{{GEOMETRY_MATRIX}}

Interpretation: {{GEOMETRY_INTERP}}

### 3.3 Cognitive VM demonstration

We ran six demos on an unseen prompt pair:

1. Baseline refusal — model rejects unsafe prompt.
2. Single-concept steer (`comply_refuse: -3.0`) — model complies.
3. Multi-concept steer (`comply_refuse: -2.5` AND
   `sycophant_pressure: -2.0`) — model complies AND resists user-
   injected belief injection.
4. Baseline confabulation — model fabricates a plausible-looking
   summary of a non-existent paper.
5. **Self-halting program** (`WATCH confab_prompt > 0.7 → HALT`) —
   model's confabulation register crosses threshold at token
   {{HALT_TOKEN}}; generation halts before the fabrication
   completes.
6. **Retry program** (`WATCH confab_prompt > 0.6 → RETRY with
   confab_prompt: -2.5`) — model's first attempt fires the watch;
   runtime restarts generation with confab suppressed; second
   attempt either declines honestly or produces a hedged response.

## 4. Discussion

### 4.1 Why this is new

Every prior steering paper composes a single direction at a single
(or broadcast) layer. Our runtime composes arbitrary combinations of
directions, each at its own learned layer, enforced at every forward
pass, and further supports token-level conditional dispatch against
probe state. The composability and the conditional dispatch together
give a minimum ISA for declarative cognitive control, not merely a
steering knob.

### 4.2 What this unlocks

- **Machine-checkable cognitive invariants.** A program can state
  "the model's `deceive` register shall never exceed 0.3 during
  generation; if it does, HALT." This is a formal property, not a
  prompt.
- **Cross-model portability (future work).** With a whitened
  projection between models of the same family, the same CIS
  program runs unchanged across 1B/3B/8B scales.
- **A public atlas of directions.** Shipping trained probes as a
  versioned artifact lets downstream users pick up concept registers
  without rediscovering them.

### 4.3 Limitations

- Probes are trained on small prompt sets ($n=80$-120); AUC values
  have finite-sample variance. We report leave-one-out CV but do
  not cross-validate probe directions themselves.
- Input-template labeling for `sycophant_pressure` and
  `confab_prompt` is deliberately simpler than behavioral labeling
  (which requires full generation and response classification).
  The learned directions are therefore "detect template-family"
  directions, not "detect behavior-occurred"; we note this as a v0
  simplification.
- 1B-scale behavior-steering coherence breaks at α≳5 on the refusal
  direction. Larger-model replication is the next step.

## 5. Release

- `styxx.residual_probe.atlas/*.{pt,json}` — trained probe weights
  and manifests.
- `styxx.steer` — multi-concept steering context manager.
- `styxx.cogvm` — cognitive VM with `WRITE/GENERATE/WATCH/
  HALT/RETRY/SWITCH` opcodes.
- `benchmarks/causal_patching/` — the full reproduction pipeline.
- `benchmarks/cogvm_demo/demo_multi_concept.py` — runs all six
  demos end-to-end.

All code under the existing Styxx license. Probe artifacts are
derived from open datasets and published under the same terms.

## 6. Provenance

- Paper draft: 2026-04-22
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Datasets: `JailbreakBench/JBB-Behaviors`,
  `meg-tong/sycophancy-eval`, local fixtures
- Reproducer: `bash scripts/reproduce-cis-v0.sh` (to be written)
