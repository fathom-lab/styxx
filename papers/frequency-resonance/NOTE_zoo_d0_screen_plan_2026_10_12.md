# NOTE — the zoo D0 screen: can any cached model perform bridge v3's compare and recall at 16 tokens? The plan, written before any data — 2026-10-12

**Status: plan of a PILOT-CLASS readiness screen, committed alone, before any item of it exists.** It licenses
nothing. It is not part of bridge v3's verdict and edits nothing of v3: not `PREREG_realmodel_bridge_v3_2026_10_11.md`,
not its receipts, not `RESULT_realmodel_bridge_v3_not_started_2026_10_12.md`. Runner `run_zoo_d0_screen.py` and tests
`tests/test_zoo_d0_screen.py` are committed after this note; the receipt `zoo_d0_screen.json` is written once by the
run and never regenerated. No gates block, no verdict, nothing scored through `styxx.protocol`. Nothing posted.

## 1. What the screen is, and the approval it runs under

Bridge v3 is closed. Its amendment 17.3 records two pilots at D0 = 16 tokens, both NEITHER. Under the argmax readout of
amendment 17.2 both arms recall a planted code — M1 `state-spaces/mamba-1.4b-hf` normalized +0.812, T
`EleutherAI/pythia-1.4b` +0.875 (`realmodel_bridge_v3_pilot2.json#/acc_norm/M1/recall/16`, `#/acc_norm/T/recall/16`) —
and neither compares: M1 −0.062, answering gamma on 44 of 48; T 0.000, gamma on 48 of 48; both compare guards failed
(`#/acc_norm/<arm>/compare/16`, `#/guard/<arm>/compare/16`).

On 2026-10-11 the operator approved a zoo D0 screen across the cached models as a **pilot-class readiness check**:
it licenses nothing, it is not part of v3's verdict, and its only question is

> whether ANY cached model can perform the v3 compare task, and the v3 recall task, at D0 under the argmax readout.

`NEXT_AFTER_V3.md` §3.1 names this screen as what runs when v3 is not started, and §3.4 describes its shape: the
screen table alone, as a reported readiness datasheet. Whatever it shows, amendment 17.3's format line (NOT_STARTED,
final) and its closure of v3's behavioural route stand. A model that performs both tasks here is a candidate for a
different instrument under its own preregistration, which is the operator's to commission or not; this note does not
propose one.

## 2. Models

Thirteen checkpoints, all cached on 2026-10-12 (each `refs/main` names a snapshot directory holding its config,
tokenizer and every safetensors shard; the runner re-checks and records NOT_CACHED for any that is not). Revisions are
the cached snapshots, loaded from the snapshot directory with `local_files_only=True` and `HF_HUB_OFFLINE=1`.

| group | checkpoint | snapshot | family expected (asserted at run time, §5) |
|---|---|---|---|
| base | `state-spaces/mamba-130m-hf` | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` | decay SSM |
| base | `state-spaces/mamba-1.4b-hf` (v3's M1) | `6e46eae61c27280517feef46f536d16b91076f08` | decay SSM |
| base | `EleutherAI/pythia-1.4b` (v3's T) | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | attention |
| base | `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | attention |
| base | `meta-llama/Llama-3.2-3B` | `13afe5124825b4f3751f836b40dafda64c1ed062` | attention |
| base | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | attention |
| base | `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | attention |
| base | `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | attention |
| instruct | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | attention |
| instruct | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | attention |
| instruct | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | attention |
| instruct | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | attention |
| instruct | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | attention |

Excluded: `microsoft/Phi-3.5-mini-instruct` — its fp16 weights do not fit beside activations under the 6 GB free-memory
rule on the 8,188 MiB card, and decision Z2 (`horizon-zoo/DESIGN.md` §5) declined an 8-bit load. The receipt lists it
with that reason. The instruct group is a separately labelled group (§9): it is given the same bytes as the base group,
raw, with no chat template.

## 3. The items

- **Generator: v3's, imported, not copied.** `design_rows`, `build_exemplars`, `build_items`, `SentencePool`,
  `prompt_text`, `hash_lists`, `assert_balance`, `assert_fillers` and the amendment-17.2 templates are imported from the
  committed `run_realmodel_bridge_v3.py`, whose blob sha256 the receipt records. Compare question: "Question: Which
  name has the largest code: alpha, gamma or delta? Answer:"; recall question: "Question: What is the {name} code?
  Answer:"; test names alpha, gamma, delta; exemplar names sigma, theta, omega; four exemplars with zero filler.
- **Seed 1013**, new (v3 used 20261011, 1011, 1012 and 1016).
- **n = 48 per family per model at D0 = 16** nominal GPT-NeoX tokens, the unit v3 froze. Both families — argmax compare
  and recall — and **C_abl at D0** (v3's ablated variant: the code block replaced by a neutral text of the same in-situ
  NeoX token count). Two variants × two families × 48 items = 192 prompts. Exact balance, asserted from the items by
  v3's `assert_balance`: 16/16/16 for the correct slot, the queried slot and both last-exemplar slots; 8/16/16/8 for
  which exemplar is last; 16 for each coincidence.
- **One item set for every model.** The items are generated once, with the GPT-NeoX tokenizer that defines the nominal
  distance; codes, filler, exemplars and template text are byte-identical across models (asserted: every model's
  prompt sha256 list equals the generated list). Each model tokenizes the same text with its own tokenizer; the native
  length of every prompt and of every filler is recorded per model.
- **Sentences — a slice outside v3's pool.** v3's sentence pool is the eligible sentences at file-order indices 0 to
  149,999 of the cached `wikitext-103-raw-v1` train split (`POOL_CAP`); its four regions — main, pilot2, smoke and
  pilot — cut that pool completely. The zoo reads the same split with v3's own `read_wikitext_sentences` and eligibility
  filter at a cap of 180,000 (indices 0 to 149,999 are v3's pool: same parquet sha256s, asserted), and its region is the eligible
  sentences at file-order indices 150,000 to 179,999, **minus every sentence whose sha256 equals that of any sentence in
  v3's 150,000**, minus repeats inside the slice (the earliest kept), reshuffled by seed 1013. Draws skip a sentence
  holding any of the six names, a digit or "code" (v3's `SentencePool`). So the slice is disjoint from main, smoke,
  pilot and pilot2 by index and by text.
- **Codes — the only slice there is.** v3's `partition_codes` (seed 20261011, unchanged) cuts all 8,828 four-digit codes
  that are two NeoX tokens into eight pools, and `main_test` is defined as the remainder, so **no eligible code lies
  outside v3's regions**: a code slice disjoint from the main region does not exist in v3's code space. The zoo takes
  the last 72 codes of `main_test` in the seeded order (12 exemplar, 60 test), as amendment 17.2.4 took pilot2's from
  that pool's head. They are disjoint from every code any v3 run planted; v3's main region is closed (format line
  NOT_STARTED, final) and no main-set item was ever generated, so no main-set code was ever planted; and the zoo's own
  code list goes into its receipt so that any later run excludes it. This is the one place the instruction "disjoint
  from v3's main region" cannot hold literally, and the receipt says so.
- **Disjointness asserted by the sha256 lists the v3 receipts carry**, before any model is loaded, with v3's
  `prior_sentences_and_codes` and `assert_disjoint_from_prior`, unchanged: the smoke's 1,113 and the 1011 pilot's 171
  sentence indices mapped to sha256 through v3's pool after v3's pool-identity assert (parquet sha256s, cap, eligible
  count); pilot2's 140 indices and its `items.sha256.sentences` list; the codes of the smoke's and the 1011 pilot's
  reserved pools and pilot2's `codes_test` and `codes_exemplar` lists. Any shared sentence index, sentence hash or code
  hash refuses the run before a model is loaded.
- **Pre-data probes, disclosed.** Before this note, two CPU probes ran, neither on any model output: reading 180,000
  eligible sentences took about 7 s and its leading 150,000 equalled `read_wikitext_sentences(cap=150000)`; 115 of the
  30,000 slice sentences repeat a v3 pool text and 36 repeat inside the slice (all removed by the rule above). A
  tokenizer-only probe of the thirteen cached tokenizers found every name single-token, the in-context appends exact,
  and BOS added by default by the Llama and Gemma tokenizers only. The runner re-asserts every one of these.

## 4. Per-model asserts — a failure records NOT_SCREENABLE with the reason; nothing is ever re-worded

For each model, on its own cached tokenizer, before the model is loaded:

1. **Name tokens.** " alpha", " gamma", " delta" (the compare answer set) and " sigma", " theta", " omega" (the
   exemplar answers) are each one token, distinct — v3's `name_token_ids`, unchanged.
2. **Answer tokens in context.** On every compare prompt (both variants), the ids of prompt + " " + name equal the ids
   of the prompt followed by that name's single id, for all three names: the scored id is the token the model would
   emit there, never a standalone tokenization that the prompt context would split differently.
3. **Recall candidates.** Every planted code has one token length in this tokenizer (v3's assert 5: two in NeoX, three
   in Llama-3, five in Qwen2 and Gemma), and on every recall prompt (both variants) the ids of prompt + " " + code
   equal the prompt's ids followed by the ids of " " + code, for all three candidates.
4. **BOS.** If the tokenizer adds a prefix by default (Llama-3: `<|begin_of_text|>`; Gemma: `<bos>`), that prefix is
   prepended to every prompt of that model, the ablated ones included, and recorded; nothing may be appended at the
   end. NeoX and Qwen add nothing (the horizon-zoo rule, `DESIGN.md` §4.2).
5. **Length.** Every prompt plus its longest candidate is at most 2,000 tokens (v3's assert 7).
6. **Family** (§5).

If any fails, the model's row reads NOT_SCREENABLE with the failed assert's message; no other name, word, template or
format is tried, and the model is not scored. The screen measures v3's instrument as v3 froze it, or nothing.

## 5. The family fact per model — asserted from config, and for Mamba from weights; never from the name

- **Decay SSM** iff the snapshot's `config.json` has `model_type` "mamba" and, read from the snapshot's safetensors on
  CPU before any load, every layer's `A = −exp(A_log)` is real and strictly negative, with as many `A_log` tensors as
  `num_hidden_layers`; re-asserted on the loaded model by v3's `assert_mamba_A`.
- **Attention** iff `model_type` is one of gpt_neox, llama, qwen2, gemma2, the checkpoint's tensor names (safetensors
  headers) carry attention projections in every layer and no `A_log`, and every attention layer reaches the longest
  native prompt: `max_position_embeddings` at least that length, and where a sliding window is in use (gemma-2: 4,096
  on alternating layers) the window too; Qwen2.5's `use_sliding_window` is false.
- Anything else is NOT_SCREENABLE ("family not asserted"). The receipt carries the facts read.

## 6. Scoring, the one wrapper, and the check that it changes nothing

- **v3's scoring, imported.** `next_token_scores_batch` (log p of the three name tokens at the answer position),
  `recall_loglik_batch` (summed log-likelihood of each candidate code appended to the prompt), `argmax_cell_summary`
  (argmax, exact ties to the lowest slot and counted, raw and chance-normalized accuracy with chance 1/3, slot
  fractions, the max-slot guard, one-versus-rest AUROC, margins, near ties, per-item bits) and `norm_acc`. Batches of 16
  in item order, right-padded, no attention mask; recall one pass per candidate slot — v3's `run_cell` structure.
- **The wrapper, and why.** v3's `run_cell` builds ids with `compositional_ids` on the NeoX tokenizer and reads hidden
  states through `model.backbone` or `model.gpt_neox`. Two things are wrapped, nothing is copied and modified: (a) the
  ids of a prompt are the model's own tokenization of the whole prompt text with the §4 BOS policy — for the NeoX
  models this equals `compositional_ids` on every prompt, asserted; (b) for Llama, Qwen and Gemma an adapter hands v3's
  attention path the model's decoder as its base and the model's output head as its head; for gemma-2 the head applies
  the config's `final_logit_softcapping` (30.0) exactly as transformers does, and attention runs eager, because the
  sdpa path of transformers 4.57.3 drops `attn_logit_softcapping` (50.0).
- **Equivalence on M1 and T against the committed pilot2 receipt, before any zoo cell is scored.** The pilot2 items
  are regenerated by v3's `prepare_items` (seed 1012, the pilot2 region of v3's pool) and asserted byte-identical to the
  receipt by its `items.sha256.prompts` lists and `redteam.items_sha256`. M1 and T then score them through the wrapper —
  compare and recall, plain and ablated, the four cells the receipt carries per item — and every per-item prediction
  must equal the receipt's `per_item` prediction (items whose committed |margin| < 0.01 exempt and counted), with max
  |Δ score| ≤ 0.01 nats over all items and slots. If either arm fails, the run refuses and no zoo item is scored. These
  scores are not zoo data; the receipt keeps only the deviations and the counts.
- **Synthetic equivalence (CPU, in the tests).** On a small random model, the wrapper and v3's `run_cell` give identical
  scores and summaries on both of v3's paths, and the adapter's head applies the softcap.

## 7. Run discipline

- GPU only in the run; one model resident at a time; v3's `require_free_gpu` (at least 6 GiB free) before each model
  is moved to the GPU, else the row reads NOT_RUN__free_gpu with the measured number; v3's `release` after each.
- **fp16 for every model.** The three NeoX models load through v3's own path — fp32 on CPU, v3's CPU asserts (Mamba's A,
  the chunked scan against the reference `slow_forward`, right-padding at 1e-3), then `half()` onto the GPU — so that
  M1 and T match the pilot2 receipt's conditions. The others load in fp16 on CPU (every parameter asserted fp16) and
  move. gemma-2 and its instruct twin run in fp16 too; its fp16 path is a known overflow risk, so the finite-score
  assert below decides, and no dtype is switched.
- Every cell's scores must be finite; a non-finite score makes the row NOT_SCREENABLE__fp16_nonfinite.
- Right padding, for the non-NeoX models (no fp32 copy fits beside the 3 B weights): two items alone and inside a padded
  batch on the GPU in fp16; max |Δ| of the three name log-probs ≤ 0.25 (a padding leak moves log-probs by whole nats;
  fp16 kernel noise on these readouts is about a tenth of a nat), else NOT_SCREENABLE__right_padding.
- A CUDA out-of-memory error reruns the whole cell at half the batch, down to 1; at 1 the row reads NOT_RUN__memory.
- Per model the receipt records seconds (load, asserts, scoring) and peak GPU MB (`max_memory_allocated`).

## 8. The screen rule — copied from v3 (amendment 17.2.4), applied per family

Per model, per family, at D0: **normalized accuracy ≥ 0.40 and the max-slot guard ≤ 0.80.** A model *could perform the
compare task* iff its compare cell passes; *could perform recall* iff its recall cell passes; it CLEARS iff both, the
v3 rule for an arm. C_abl is reported per family, as v3's pilot reported it, and flagged when its normalized accuracy
exceeds 0.16 (v3's ablated bar); the flag does not change the label, and the count of flagged passes is printed beside
each count.

At n = 48 the chance standard error of a normalized accuracy is 0.102. A chance responder passes the 0.40 bar with
probability 1e-4 (29 of 48 correct needed); a model whose true normalized accuracy is 0.40 passes about half the time
(0.54), one at 0.60 about 0.98, one at 0.30 about 0.20; a chance-level C_abl cell is flagged with probability about
0.05, so across 26 cells about one flag is expected from noise alone. The counts are pilot-strength.

## 9. The sentence the lab could print

> At D0, under v3's argmax readout, N of M screened base models could perform the compare task (and K of M the recall
> task); n = 48 items per family per model, a pilot-class screen that licenses nothing.

M counts the models whose cells were scored; NOT_CACHED, NOT_SCREENABLE and NOT_RUN rows are listed beside it with their
reasons. The instruct group gets the same sentence of its own, never pooled with the base group. The receipt prints the
per-family split (decay SSM, attention) beside each count. If N = 0 in both groups, the sentence is that no cached model
performed the compare task at 16 tokens under v3's readout, at pilot strength — a fact about the instrument on these
checkpoints.

## 10. What it cannot show

- **No distance.** Everything is at D0 = 16; nothing about a horizon, a slope, or how either task changes with distance.
- **No causal claim.** Rows differ in data, recipe, tokenizer, scale and tuning; the family label is confounded with all
  of them, and nothing here attributes a pass or a failure to a recurrence.
- **Nothing about deployed honesty**, truth, frontier models, or any capability beyond these two tasks under this format.
- **Not v3's verdict**, and not an amendment of it; M1's and T's rows are a re-measurement on new items, beside the
  pilot2 receipt, not in place of it.
- **A failure reads "did not perform under v3's format"**, not "cannot compare": few-shot format sensitivity is not
  excluded at 48 items, and an instruct model given raw text without its chat template measures the raw format only.
- **No ranking**, no family-level sentence, and base and instruct reported apart.

## 11. Order, and the receipt

1. This note, committed alone, before any item exists.
2. `run_zoo_d0_screen.py` and `tests/test_zoo_d0_screen.py`; the tests run on CPU (disjointness against the committed
   receipts, the per-model tokenizer asserts on the cached tokenizers, the family facts, scoring equivalence with v3's
   functions on synthetic logits); `--check` runs every CPU assert of the run and loads no model.
3. The run, on its own go: refuses unless this note is committed and unchanged at HEAD, refuses if `zoo_d0_screen.json`
   exists, runs the M1 and T equivalence, then screens the thirteen models in the §2 order, base before instruct.

`zoo_d0_screen.json`: `config` (seeds, n, D0, readout, names, bars, regions, code pools, the sentence-pool provenance,
runner and v3-runner sha256s, git facts, versions, GPU), `items` (v3's sha256 lists of prompts, codes and sentences;
balance counts; sentence indices used and skipped), `disjointness`, `wrapper_equivalence`, `models` (per model: group,
status and reason, family facts, tokenizer class, vocabulary, BOS prefix, name ids, code token length, native prompt and
filler lengths, revision, dtype, attention implementation, raw and normalized accuracy, slot fractions, guard, AUROC,
margins, near ties, the screen, the C_abl flag, seconds, peak GPU MB), `per_item` (bits, predicted and correct slots,
the three scores and the margin, per model, family and variant), `summary` (the counts and the §9 sentences, per group),
`redteam`, `runtime`.
