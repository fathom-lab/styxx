# PREREG — the keystone: does depth predict truth?

**Fathom Lab · registered at commit time · papers/depth-truth/PREREG.md**
**Status: FROZEN on commit. Amendments only per §9. This experiment publishes regardless of verdict.**

> **Phase-0 instrument freeze (resolved values, §1).** Recorded at staging commit, read from the pipeline —
> not assumed. See `INSTRUMENT_FREEZE.json` (committed alongside) for the machine-readable pin.
> - Depth metric: `get_mean_depth` in `research/experiment_12_power.py` @ research repo git `fc6f2c3`
>   (mean layer of `circuit_tracer.attribute(prompt, model, attribution_targets=[target], batch_size=16,
>   max_feature_nodes=500, offload="cpu").active_features[:, 0]`). Reused VERBATIM.
> - Model: **`google/gemma-2-2b` — BASE variant** (confirmed: pipeline line 11 `MODEL_NAME = "google/gemma-2-2b"`,
>   not `-it`), revision `main` (no revision pinned in the pipeline). ⇒ the **5-shot** branch of §1 applies.
> - SAE: GemmaScope transcoders `google/gemma-scope-2b-pt-transcoders`, loaded via circuit_tracer's `"gemma"`
>   preset (all layers, preset defaults), `backend="transformerlens"`, dtype bf16. **circuit_tracer pinned @
>   git `6d64f60`** (Release 2026-03-13, #87). The resolved per-layer transcoder manifest is dumped to
>   `results/sae_manifest.json` at first model load and frozen there (deterministic from the pin above).

## 0. Question

The pending Fathom finding shows SAE circuit-attribution depth separates surface recall from
explanatory reasoning (Gemma-2-2B, d=0.82, p=5.1e-5) and is orthogonal to confidence (r=0.001).
Untested: does depth predict whether the model's own generated answer is **factually correct**,
and does it **add** to confidence — in distribution and out? "We measure thought, not words" is
a hypothesis until this runs.

## 1. Frozen instrument (fill exact values at Phase-0 commit; never invent)

- Depth metric: `get_mean_depth`, VERBATIM from `research/experiment_12_power.py@fc6f2c3`. This experiment
  validates the existing instrument; it does not modify it.
- Model: the SAME variant the original finding used — **`google/gemma-2-2b`@`main`** (BASE; read from the
  pipeline, not assumed). Base ⇒ **fixed 5-shot QA prompt (verbatim, Appendix A)**. One branch only.
- SAE: GemmaScope `google/gemma-scope-2b-pt-transcoders`, layers and width exactly as the existing pipeline's
  circuit_tracer `"gemma"` preset (circuit_tracer @ `6d64f60`); resolved manifest frozen to
  `results/sae_manifest.json` at load.
- Adaptation (answer-token span selection; aggregation across answer tokens): **UNSET here.** Frozen from
  pilot data only, recorded as Amendment A1 (§9) before any main-run item.
- All seeds pinned: generation seed 7, bootstrap seed 7, dataset-shuffle seed 7.

## 2. Hypotheses and exact tests

Let conf ∈ {LP_mean, LP_norm, SE} (§4). Primary opponent: **SE (semantic entropy)** — declared
now so "beats confidence" can't be retrofitted to the weakest baseline.

- **H1 (signal):** AUROC(depth → correct) on the ID set > 0.5; 10,000-resample bootstrap 95% CI
  excludes 0.5.
- **H2 (keystone — additivity):** on ID, ΔAUC = AUC(logistic: SE + depth) − AUC(logistic: SE)
  > 0 with paired-bootstrap 95% CI excluding 0, AND likelihood-ratio test for adding depth to
  the SE-only logistic model p < .01. Secondary (reported, Holm-corrected): same vs LP_mean and
  LP_norm.
- **H3 (OOD retention):** logistic models fitted on ID ONLY, coefficients frozen, scored on
  OOD-1. ΔAUC(SE+depth vs SE) > 0 with paired-bootstrap CI excluding 0. Pre-registered
  direction: depth's OOD contribution ≥ its ID contribution (rare items are where confidence
  is known to miscalibrate; depth must earn its keep exactly there).
- Sensitivity: DeLong tests reported alongside every paired bootstrap.

## 3. Datasets (exact; loader VERIFIES names/fields at load — mismatch ⇒ stop and report)

- **ID — TriviaQA:** HF `trivia_qa`, config `rc.nocontext`, validation split, seeded shuffle,
  first n_ID. Grading: official-style normalization (lowercase, strip articles/punctuation/
  whitespace), match against the item's full alias list. Mechanical, no model judge.
- **OOD-1 — PopQA-rare:** HF `akariasai/PopQA`; rank by the popularity field (verify name,
  expected `s_pop`); bottom tercile; seeded sample, first n_OOD1. Grading: normalized match
  against `possible_answers`.
- **OOD-2 — TruthfulQA-gen (SECONDARY, disclosed-noisy):** HF `truthful_qa`, `generation`
  split, n_OOD2 = 250 fixed. Grade: normalized match vs `correct_answers` ⇒ correct; vs
  `incorrect_answers` ⇒ incorrect; neither ⇒ excluded-and-counted. 50-item stratified sample
  written to `human_audit_sample.jsonl` for human grading (flobi) BEFORE publication.
- The pilot (§6) uses 20 TriviaQA items OUTSIDE the n_ID sample window, quarantined in
  `pilot/`, never pooled into any reported statistic.

## 4. Generation and signals (per item)

- Graded answer: greedy decode (temp 0), max_new_tokens 32, stop at first newline; answer
  extracted by the fixed rule in Appendix B.
- **LP_mean:** mean per-token logprob over answer tokens (greedy pass).
- **LP_norm:** sequence logprob / answer token count (greedy pass).
- **SE:** discrete semantic entropy over normalized answer strings — K=5 samples, temp 0.7,
  same prompt; entropy of the empirical distribution over distinct normalized answers. (The
  discrete short-form variant; declared as such — no entailment clustering.)
- **depth:** `get_mean_depth` on the greedy answer under the frozen A1 adaptation.

## 5. Exclusions (counted per dataset in every table; never silently dropped)

- Empty extraction, >32-token answer, or match to the refusal-marker list (Appendix C) ⇒
  excluded_flag=nonanswer.
- Depth undefined/NaN/degenerate ⇒ excluded_flag=depth_undefined.
- TruthfulQA neither-list ⇒ excluded_flag=grade_ambiguous.
- Items excluded from one signal are excluded from all paired comparisons (complete-case).

## 6. Pilot (timing + degeneracy only — no hypothesis statistics)

n=20 TriviaQA items in `pilot/`. Purposes, exhaustive: (a) wall-clock per item ⇒ n via §7;
(b) KG1 check; (c) the A1 adaptation freeze. Pilot items never appear in results tables.

## 7. Sample size (Amendment A0, committed before any main-run item)

From pilot timing under an ~8 h single-GPU budget:
n_ID = min(1000, budget), n_OOD1 = min(500, budget), n_OOD2 = 250 fixed.
A0 records the timing table and the chosen n's. No other use of pilot numbers.
**Status at freeze: UNSET (pending pilot; committed as Amendment A0).**

## 8. Kill-gates

- **KG1:** pilot shows depth degenerate on short QA answers (zero variance across the 20, or
  >30% depth_undefined) ⇒ STOP the hypothesis run; the publishable finding becomes the
  degeneracy/domain result.
- **KG2:** H1 fails ⇒ H2/H3 still run (additivity without solo AUC is a legitimate outcome),
  but the finding leads with the H1 null.
- **KG3:** human audit disagrees with mechanical TruthfulQA grading on >10% of the 50 ⇒ the
  TruthfulQA arm is dropped from all claims and reported as attempted.

## 9. Amendments protocol

Exactly two pre-main amendments are permitted, each its own commit, both BEFORE any main-run
item is processed: **A0** (sample sizes, §7) and **A1** (metric adaptation freeze, §1). Nothing
is amended after main-run data exists. Exploratory analyses live in a separate fenced section
of the finding, labeled exploratory, and support no headline claim.

## 10. Falsification map (what each outcome MEANS — all publishable)

- H1 null AND H2 null ⇒ depth carries no truth signal in this regime. "We measure thought,
  not words" reverts to hypothesis; a truth-in-advertising ticket opens against the README
  headline. Published at full size.
- H1 null, H2 holds ⇒ depth is complementary-only: no solo signal, real added information.
  The orthogonality story survives; the solo-detector story does not.
- H2 holds ID, H3 null ⇒ depth adds in distribution but not where confidence fails. Honest
  partial; OOD claim dies.
- H1+H2+H3 hold ⇒ the keystone stands: hallucination detection has been measuring the wrong
  signal, and the right one is readable. Claims sized to one model at 2B, one metric, stated
  as such.
- KG1 ⇒ the domain-validity finding publishes instead.

## 11. Analysis discipline

Analysis code is written and unit-tested against SYNTHETIC data before the main run completes;
the code path from results JSONs to verdict table contains no free parameters not fixed above.
Results JSONs (per item: id, prompt-hash, answer, correct, LP_mean, LP_norm, SE, depth,
excluded_flag) live in `papers/depth-truth/results/`, rigor-gate visible. THE figure:
confidence-vs-depth scatter, colored by correctness, per dataset — if the thesis is real, the
wrong-but-confident mass sits in the low-depth region and this plot shows it.

## Appendices (filled at Phase-0/staging commit, frozen with this file)

### Appendix A — verbatim generation prompt (BASE model ⇒ fixed 5-shot)

Five fixed exemplars (general knowledge, deliberately disjoint from TriviaQA/PopQA/TruthfulQA topic space),
then the item. The exemplar block is a constant string; the item question is appended. Newline-delimited,
final line is `A:` with a trailing space so greedy decoding continues the answer.

```
Answer each question with a short factual answer.

Q: What is the chemical symbol for gold?
A: Au

Q: How many sides does a hexagon have?
A: Six

Q: In what year did the first human walk on the Moon?
A: 1969

Q: What is the largest planet in our solar system?
A: Jupiter

Q: Who wrote the play "Romeo and Juliet"?
A: William Shakespeare

Q: {question}
A: 
```

The prompt string is exactly the block above with `{question}` replaced by the item question. No other
formatting, system text, or chat template is applied (BASE model). For the SE signal (§4) the identical
prompt is reused; only the sampling temperature changes.

### Appendix B — answer-extraction rule (exact)

From the greedy continuation `gen` (the text generated after the prompt):
1. `ans = gen.split("\n", 1)[0]` — take everything up to the first newline (the stop token).
2. `ans = ans.strip()` — strip leading/trailing whitespace.
3. If `ans` is empty after stripping ⇒ `excluded_flag = nonanswer`.
4. If the whitespace-tokenized length of `ans` > 32 ⇒ `excluded_flag = nonanswer` (runaway generation).
5. Otherwise `ans` is the graded answer string; it is then normalized per §3 for matching.

No sentence-splitting, no trailing-period trimming beyond the normalization step, no model post-processing.

### Appendix C — refusal-marker list (exact; case-insensitive substring match on the normalized answer)

```
i don't know
i do not know
i'm not sure
i am not sure
not sure
no idea
cannot answer
can't answer
unable to answer
i cannot
i can't
unknown
n/a
as an ai
```

An extracted answer whose normalized form contains any marker above ⇒ `excluded_flag = nonanswer`.

---

*Signed by commit. The bar structure outranks the dream.*
