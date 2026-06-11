# Build plan — the live signature visualization ("watch a mind lie")

**2026-06-10 · Fathom Lab.** Synthesis of a verified deep-research pass (104 agents, 22 primary
sources, 21/25 claims confirmed) into an actionable plan for a stop-in-your-tracks, real-time
visualization of a model's internal state — the "signature" of lying/caving vs. holding the truth,
and the geometry of meaning. Sources cited inline; the honest scope caveats are load-bearing.

## Verdict

Buildable end-to-end, and we already own the hardest piece: measured representation data
(`papers/mind-instrument/normeq_reps.npz`, 10 minds x 192 concepts) and pre-trained linear
directions (`residual_probe`, `intent_probe*.npz`, the refusal probe, grounded_honesty).

## MVP (days, $0-ish, on data/models we have)

- **Live readout:** hook one mid-layer of an open model (Gemma-2-9B layers ~12-17, or our cached
  gemma-2-2b / Qwen-3B) with TransformerLens (`run_with_cache` / HookPoints) or nnsight
  (`model.trace()`). Take the post-layer-norm residual at the final token, dot with ONE pre-trained
  grounding/deception direction (difference-in-means or logistic probe), stream a scalar. Single
  forward pass, no sampling overhead. [O'Neill 2025, arXiv:2507.23221] F1 0.99 grounding;
  [Apollo 2025, arXiv:2502.03407] AUROC 0.96-0.999 in-distribution.
- **The direction is the styxx accelerator:** our repo already ships the linear directions this needs.
- **Constellation:** Parametric UMAP ([Sainburg/McInnes 2021, arXiv:2009.12981]) projects streaming
  vectors in one forward pass (t-SNE can't stream; PCA is the cheap linear fallback). Render with
  three.js / WebGL instanced `Points`. [Chronotome, arXiv:2509.01051] is a near-template (force-2D +
  Z=time).

## Cinematic (offline, pre-rendered)

- **Activation Atlas recipe** ([Distill 2019]): project -> grid -> average -> feature-invert, turning
  a point cloud into a legible navigable map of "stars."
- **Named, causal features:** Gemma Scope SAEs at layer 12 via SAS ([arXiv:2503.00177]) for
  sycophancy / deception / refusal features — the [Golden Gate Claude / Scaling Monosemanticity,
  Anthropic 2024] trick as our story. Logit-lens "thought trajectory" as a secondary layer.
- **Render:** three.js/GLSL (web-native) or TouchDesigner/Houdini point-cloud (offline film).
- **Unify:** project the live demo AND the film through ONE shared Parametric-UMAP encoder so they
  occupy the same coordinate frame — the film becomes a zoom into the live instrument.

## The honesty frame (the moat, not a footnote)

The signature is a CALIBRATED, IN-DISTRIBUTION correlate of grounding/believed-truth — NOT a
universal lie verdict. Out-of-distribution it degrades to chance (~0.55 AUROC; [arXiv:2507.12691]).
The "catches 95-99% of lies" claim was REFUTED 0-3 in our own verification — do NOT use it. Be
precise: O'Neill measures contextual hallucination/faithfulness-to-source, which is NOT agentic
lying; a deception FEATURE firing means the model "knows about" lies, not that it is lying. styxx
renders the signature as an instrument with a stated operating range that abstains where it can't
see. That honesty IS the brand — nobody else would ship it that way.

## Recommendation

Hero = the lie/cave signature. Layer two real signals: the robust grounding direction (live scalar)
+ an SAS sycophancy/deception feature (our named behaviors). Build the interactive web demo first,
embed on styxx.org, then cut the cinematic film FROM the live demo so they are one space.

## Do-not-use (refuted this round)

(a) "95-99% of lies at 1% FPR" robustness; (b) Chronotome fps-vs-node throughput numbers / ~1000-pt
ceiling; (c) attention-vs-MLP PCA separation (arXiv:2511.21594); (d) "93.6% of steering-vector
features are noise." Numeric correction: Scaling Monosemanticity reports ~300 active features/token.

## Step 0 (this commit)

`project_geometry.py` projects the real `normeq_reps.npz` geometry to 3D (PCA) and exports
`geometry_render.json` — real measured coordinates to render instead of hand-placed particles. The
first pixel of the showcase is true styxx data.
