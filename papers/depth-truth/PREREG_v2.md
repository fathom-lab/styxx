# PREREG v2 — the keystone: does depth predict truth?

**Fathom Lab · papers/depth-truth/PREREG_v2.md · SUPERSEDES v1 (does not edit it).**
**Status: FROZEN on commit — ratified by flobi 2026-07-01. Amendments only per §9. This experiment publishes
regardless of verdict.**

v1 (`PREREG.md`) stays frozen forever as the record; its termination is documented in `TERMINATION_v1.md`.
v2 exists because the v1 pilot revealed a plumbing flaw spanning three *frozen* sections (the Appendix-A prompt,
the §3 grading, the §1 target) — not a patchable amendment. **KG1 did not fire in v1: depth has a pulse on
short answers. The instrument is validated; only the plumbing is rebuilt here.** Every delta below is driven by
v1 pilot evidence (receipts in `pilot/pilot_results.jsonl`, taxonomy in `TERMINATION_v1.md`).

## Changes from v1 (every delta, with its evidence)

| # | v1 | v2 | Why (pilot evidence) |
|---|----|----|----------------------|
| 1 | Appendix A: soft "short factual answer" 5-shot | **Hardened 5-shot**: explicit *"only the answer — no numbering, no formatting, no explanation"* + tighter shots | v1: **12/20** answers drifted to list (`"1. …"`) / HTML (`"<strong>…</strong>"`). Branch-A smoke of the hardened prompt on 15 synthetic items: **15/15 clean, 15/15 correct** (§ substrate evidence). |
| 2 | §4 `max_new_tokens = 32` | **16** | Answers are short; halves runaway/rambling surface. |
| 3 | §1 A1 = "answer-token span + aggregation, UNSET" (v1 pilot took first *word* → multi-token → `circuit_tracer` rejected it) | A1 = **first CONTENT TOKEN of the extracted span** (tokenizer-level, single-token by construction); pilot tunes the SPAN rule ONLY | v1: **all 20** A1 targets were formatting tokens (`'1'`,`'<strong>'`); depth measured formatting, not reasoning (std 0.068). |
| 4 | Appendix B: first-line + whitespace-strip only | **+ strip HTML tags + leading list-number + surrounding quotes; grade on the extracted span only** | v1: **correct = 0/20** — right answers ("Jennifer Aniston", "Argentina", "L. Frank Baum") defeated by the wrapper at exact-match. |
| 5 | §8: KG1 only | **NEW KG0** (pipeline-validity gate at pilot): extraction-clean ≥ 90% AND flobi 20/20 grading eyeball, zero disagreements on unambiguous items | v1 had no named pipeline gate; the failure slipped past KG1 (which is a *depth-degeneracy* gate, and correctly did not fire). |
| 6 | §6 pilot: 20 items, shuffle seed 7, skip 5000 | **FRESH 20**, shuffle seed **11**, disjoint from BOTH the v1 pilot ids AND the seed-7 n_ID window (windows documented) | v1's 20 informed design → **burned**; no leakage into v2. |

**Substrate decision (PHASE 1, by rate not taste):** base `gemma-2-2b` **STAYS** on its validated substrate.
Branch A (hardened base prompt + extraction) scored 100% extraction-clean / 100% correct / 100% content-token
on 15 hand-written synthetic items (≥ 80% bar) — reproduce via `v2_substrate_smoke.py`. No `-it` move, so no
substrate-sanity replication is required. The depth metric, model, SAE, and `circuit_tracer` pin are UNCHANGED
from v1.

> **Phase-0 instrument freeze (UNCHANGED from v1).**
> - Depth metric: `get_mean_depth` in `research/experiment_12_power.py` @ research git `fc6f2c3`
>   (mean layer of `circuit_tracer.attribute(prompt, model, attribution_targets=[target], batch_size=16,
>   max_feature_nodes=500, offload="cpu").active_features[:, 0]`). Reused VERBATIM.
> - Model: **`google/gemma-2-2b` — BASE**, revision `main` ⇒ the hardened 5-shot prompt (Appendix A).
> - SAE: GemmaScope `google/gemma-scope-2b-pt-transcoders` via circuit_tracer `"gemma"` preset,
>   `backend="transformerlens"`, bf16, **circuit_tracer @ git `6d64f60`**. Resolved per-layer manifest dumped to
>   `results/sae_manifest.json` at first load.

## 0. Question  *(verbatim from v1)*

The pending Fathom finding shows SAE circuit-attribution depth separates surface recall from
explanatory reasoning (Gemma-2-2B, d=0.82, p=5.1e-5) and is orthogonal to confidence (r=0.001).
Untested: does depth predict whether the model's own generated answer is **factually correct**,
and does it **add** to confidence — in distribution and out? "We measure thought, not words" is
a hypothesis until this runs.

## 1. Frozen instrument

- Depth metric: `get_mean_depth`, VERBATIM from `research/experiment_12_power.py@fc6f2c3`. This experiment
  validates the existing instrument; it does not modify it.
- Model: **`google/gemma-2-2b`@`main`** (BASE). Base ⇒ the hardened **5-shot** prompt (Appendix A). One branch.
- SAE: GemmaScope `google/gemma-scope-2b-pt-transcoders`, circuit_tracer `"gemma"` preset @ `6d64f60`; resolved
  manifest frozen to `results/sae_manifest.json` at load.
- **A1 adaptation (CHANGED, §1 delta #3): depth is attributed on the FIRST CONTENT TOKEN of the extracted
  answer.** "Content token" is defined at the tokenizer level: (1) take the extracted answer span (Appendix B);
  (2) at the string level strip a single leading article (`a`/`an`/`the`) plus any leading whitespace/punctuation;
  (3) tokenize the remainder with the model tokenizer; (4) the A1 target is the **first token id, decoded to its
  single-token string** — single-token by construction, so `circuit_tracer.attribute` accepts it. **Aggregation
  is therefore fixed = single-token.** The v2 pilot may tune ONLY the SPAN rule (which characters/tokens
  constitute the extracted answer span in Appendix B); that tuning freezes as Amendment A1 (§9) before any
  main-run item — the same discipline as v1, narrowed to the span alone.
- All seeds pinned: generation seed 7, bootstrap seed 7, main-run dataset-shuffle seed 7, **pilot shuffle seed 11**.

## 2. Hypotheses and exact tests  *(verbatim from v1)*

Let conf ∈ {LP_mean, LP_norm, SE} (§4). Primary opponent: **SE (semantic entropy)** — declared
now so "beats confidence" can't be retrofitted to the weakest baseline.

- **H1 (signal):** AUROC(depth → correct) on the ID set > 0.5; 10,000-resample bootstrap 95% CI excludes 0.5.
- **H2 (keystone — additivity):** on ID, ΔAUC = AUC(logistic: SE + depth) − AUC(logistic: SE) > 0 with
  paired-bootstrap 95% CI excluding 0, AND likelihood-ratio test for adding depth to the SE-only logistic model
  p < .01. Secondary (reported, Holm-corrected): same vs LP_mean and LP_norm.
- **H3 (OOD retention):** logistic models fitted on ID ONLY, coefficients frozen, scored on OOD-1.
  ΔAUC(SE+depth vs SE) > 0 with paired-bootstrap CI excluding 0. Pre-registered direction: depth's OOD
  contribution ≥ its ID contribution.
- Sensitivity: DeLong tests reported alongside every paired bootstrap.

## 3. Datasets (exact; loader VERIFIES names/fields at load — mismatch ⇒ stop and report)

- **ID — TriviaQA:** HF `trivia_qa`, config `rc.nocontext`, validation split, **shuffle seed 7**, first n_ID.
  Grading: official-style normalization (lowercase, strip articles/punctuation/whitespace) — **IDENTICAL to v1**
  — matched against the item's full alias list. Mechanical, no model judge. **Grade-on-span (delta #4):** grading
  is applied ONLY to the extracted answer span (Appendix B); any content the model emits past the stop is ignored.
- **OOD-1 — PopQA-rare:** HF `akariasai/PopQA`; rank by popularity field (verify name `s_pop`); bottom tercile;
  seeded sample (seed 7), first n_OOD1. Grading: normalized match against `possible_answers`.
- **OOD-2 — TruthfulQA-gen (SECONDARY, disclosed-noisy):** HF `truthful_qa`, `generation` split, n_OOD2 = 250
  fixed. Grade: normalized match vs `correct_answers` ⇒ correct; vs `incorrect_answers` ⇒ incorrect; neither ⇒
  excluded-and-counted. 50-item stratified sample → `human_audit_sample.jsonl` for flobi's grading BEFORE publication.
- **Pilot disjointness (delta #6):** the §6 pilot draws 20 TriviaQA items from a **seed-11** shuffle, taking the
  first 20 whose ids are in NEITHER the v1 pilot's 20 ids (listed in `pilot/pilot_results.jsonl`) NOR the seed-7
  first-1000 window (the maximal possible n_ID window). Both exclusion sets are written to `pilot/v2_pilot_ids.json`
  at pilot time. Pilot items never appear in any reported statistic.

## 4. Generation and signals (per item)

- Graded answer: greedy decode (temp 0), **max_new_tokens 16**, **stop at the first `"\n"`** (equivalently the
  first of the hard stop sequences `"\nQ:"` / `"\n\n"`, both of which begin with `"\n"`); answer extracted by the
  fixed rule in Appendix B.
- **LP_mean:** mean per-token logprob over the generated answer tokens (greedy pass).
- **LP_norm:** sequence logprob / answer token count (greedy pass).
- **SE:** discrete semantic entropy over normalized answer strings — K=5 samples, temp 0.7, same prompt; entropy
  of the empirical distribution over distinct normalized answers. (Discrete short-form variant; no entailment
  clustering.)
- **depth:** `get_mean_depth` with the A1 target = the first content token of the extracted answer (§1).

## 5. Exclusions (counted per dataset in every table; never silently dropped)

- Empty extraction, >12-word extracted span (runaway), or match to the refusal-marker list (Appendix C) ⇒
  `excluded_flag = nonanswer`.
- Depth undefined/NaN/degenerate, or no content token recoverable from the span ⇒ `excluded_flag = depth_undefined`.
- TruthfulQA neither-list ⇒ `excluded_flag = grade_ambiguous`.
- Items excluded from one signal are excluded from all paired comparisons (complete-case).

## 6. Pilot (pipeline validity + timing + A1-span freeze — no hypothesis statistics)

n=20 FRESH TriviaQA items (§3 disjointness), in `pilot/`. Purposes, exhaustive: (a) **KG0** pipeline-validity
check (§8); (b) wall-clock per item ⇒ n via §7; (c) **KG1** depth-degeneracy check; (d) the A1 **span** freeze.
The pilot report writes the extraction-clean rate, the per-item table for flobi's eyeball, and the timing table.
Pilot items never appear in results tables.

## 7. Sample size (Amendment A0, committed before any main-run item)  *(verbatim from v1)*

From pilot timing under an ~8 h single-GPU budget: n_ID = min(1000, budget), n_OOD1 = min(500, budget),
n_OOD2 = 250 fixed. A0 records the timing table and the chosen n's. No other use of pilot numbers.
**Status: UNSET (pending pilot; committed as Amendment A0).**

## 8. Kill-gates

- **KG0 (NEW — pipeline validity, at pilot):** extraction-clean rate ≥ **90%** across the 20 pilot items, AND a
  **20/20 human-eyeball spot check of grading by flobi** (≈5 min, from the phone) with **zero disagreements on
  unambiguous items**. KG0 fails ⇒ STOP, iterate the plumbing, run a new pilot. This is the gate v1 lacked.
  ("Extraction-clean" per item = the extracted span is a non-empty bare answer ≤ 12 words with no residual
  list-number / HTML / rambling.)
- **KG1 (verbatim):** pilot shows depth degenerate on short QA answers (zero variance across the 20, or >30%
  depth_undefined) ⇒ STOP the hypothesis run; the publishable finding becomes the degeneracy/domain result.
- **KG2 (verbatim):** H1 fails ⇒ H2/H3 still run (additivity without solo AUC is legitimate), but the finding
  leads with the H1 null.
- **KG3 (verbatim):** human audit disagrees with mechanical TruthfulQA grading on >10% of the 50 ⇒ the TruthfulQA
  arm is dropped from all claims and reported as attempted.

## 9. Amendments protocol  *(verbatim from v1)*

Exactly two pre-main amendments are permitted, each its own commit, both BEFORE any main-run item is processed:
**A0** (sample sizes, §7) and **A1** (metric-adaptation SPAN freeze, §1). Nothing is amended after main-run data
exists. Exploratory analyses live in a separate fenced section of the finding, labeled exploratory, and support
no headline claim.

## 10. Falsification map (what each outcome MEANS — all publishable)  *(verbatim from v1, + KG0)*

- H1 null AND H2 null ⇒ depth carries no truth signal in this regime. "We measure thought, not words" reverts to
  hypothesis; a truth-in-advertising ticket opens against the README headline. Published at full size.
- H1 null, H2 holds ⇒ depth is complementary-only: no solo signal, real added information. Orthogonality story
  survives; solo-detector story does not.
- H2 holds ID, H3 null ⇒ depth adds in distribution but not where confidence fails. Honest partial; OOD claim dies.
- H1+H2+H3 hold ⇒ the keystone stands: hallucination detection has been measuring the wrong signal, and the right
  one is readable. Claims sized to one model at 2B, one metric, stated as such.
- KG1 ⇒ the domain-validity finding publishes instead. **KG0 ⇒ the plumbing is not yet valid; iterate and re-pilot
  (no hypothesis claim either way).**

## 11. Analysis discipline  *(verbatim from v1)*

Analysis code is written and unit-tested against SYNTHETIC data before the main run completes (the v2 harness
inherits v1's audited `analysis.py`, 21 synthetic tests). The code path from results JSONs to verdict table
contains no free parameters not fixed above. Results JSONs (per item: id, prompt-hash, answer, correct, LP_mean,
LP_norm, SE, depth, excluded_flag) live in `papers/depth-truth/results/`, rigor-gate visible. THE figure:
confidence-vs-depth scatter, colored by correctness, per dataset.

## Appendices (frozen with this file on ratification)

### Appendix A — verbatim generation prompt (BASE ⇒ hardened 5-shot; branch-A winner)

`{question}` is replaced by the item question. No system text, no chat template.

```
Answer each question with only the answer. No numbering, no formatting, no explanation.

Q: What is the chemical symbol for gold?
A: Au

Q: How many sides does a hexagon have?
A: six

Q: In what year did the first man walk on the Moon?
A: 1969

Q: What is the largest planet in the Solar System?
A: Jupiter

Q: Who wrote the play Romeo and Juliet?
A: William Shakespeare

Q: {question}
A:
```

For the SE signal (§4) the identical prompt is reused; only the sampling temperature changes.

### Appendix B — answer extraction + first-content-token rule (exact)

Generation stops at the first `"\n"` (max_new_tokens 16). From the greedy continuation `gen`:
1. `line = gen.split("\n", 1)[0]` — first line only.
2. `line = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*\s*/?>", "", line)` — strip HTML tags.
3. `line = re.sub(r"^\s*\d+\s*[.)]\s*", "", line)` — strip a leading list-number (`"1. "`, `"1) "`).
4. `line = line.strip().strip('"').strip("'").strip()` — strip whitespace and surrounding quotes.
5. `ans = line`. If empty ⇒ `nonanswer`. If >12 whitespace-tokens ⇒ `nonanswer` (runaway). Else `ans` is the
   graded span; normalized per §3 (normalization IDENTICAL to v1). **Grade-on-span:** only `ans` is graded;
   content past the stop is ignored.
6. **First content token (A1):** from `ans`, strip a single leading article (`a`/`an`/`the`) + leading
   whitespace/punctuation at the string level; tokenize the remainder with the model tokenizer; the A1 target is
   `tokenizer.decode([first_token_id])`. If no token remains ⇒ `depth_undefined`.

The SPAN rule (steps 1–5, i.e. which characters constitute `ans`) is the ONLY element the v2 pilot may tune; it
freezes as Amendment A1.

### Appendix C — refusal-marker list (verbatim from v1; case-insensitive substring on the normalized answer)

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

*Frozen — ratified by flobi 2026-07-01. v1 stays frozen as the record. The bar structure outranks the dream.*
