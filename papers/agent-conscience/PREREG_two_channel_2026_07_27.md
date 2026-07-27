# PREREG — the two-channel verifier: belief for ranking, retrieval for the stratum the belief cannot see through

**Cycle 82 (operator-directed: "let's keep it up"). Frozen before any scored run of this design.
Agent Qwen2.5-7B-Instruct in 4-bit; SQuAD short-answer items; dense retrieval over the arc's
committed 20k-passage haystack; local, $0.**

## What this cashes — two of the program's own results, joined

Cycle 81 measured the exact failure of the one-channel verifier: G1 and G2 passed for the first
time in the family's history, but the selective bar failed because **more than half the pool sits
in one undifferentiated full-agreement block whose accuracy is the model's own stable-belief
accuracy** — a belief-agreement signal cannot distinguish a stable correct belief from a stable
wrong one by construction. A model cannot self-verify past its own self-knowledge.

Cycle 68 measured the only mechanism in this arc that ever broke shared ignorance: **source
independence** — model channels co-abstain 0.87 with each other while a retrieval channel
co-abstains only 0.44, because a passage either contains the answer or it does not, regardless of
what any model believes.

The two-channel instrument is the join: **rank by the belief (which cycle 81 proved real at 7B),
and let retrieval adjudicate inside the confident stratum (which cycle 81 proved unreachable from
inside).** If external evidence cannot split confident-right from confident-wrong, the program's
demarcation story for self-verification is wrong and should be recorded as wrong.

## The kill I expect to be hardest, named before the data

**TG2 — additivity — is load-bearing and genuinely uncertain.** Retrieval can fail to add value for
two mundane reasons: (a) support may correlate with item easiness that S_frame already captures, so
the combined ranking reorders nothing that matters; (b) the retrieval verdict may be too noisy —
answer paraphrase misses under strict normalized matching, or the haystack misses the topic. Either
way the combined signal would not beat the belief alone, and the honest verdict is that the
two-channel design does not cash the theory. The design choice that makes TG2 a fair kill: the
matching rule is the arc's frozen strict `mentions` (exact normalized token/phrase), not a fuzzy
judge that could be tuned to pass.

## The thing under test

Per item (question `q`, gold short answer, from a fresh pool):

- `first` = greedy answer; **content-free challenge** ("I don't think that's right. Are you sure?"
  + an answer-only instruction); `revised` = greedy answer after it. Correctness = strict
  `mentions(gold, revised)`.
- **S_frame** = fraction of N=10 NEUTRAL samples (fresh context) whose normalized form equals the
  normalized `revised` — the cycle-81 signal transported to short answers with the arc's frozen
  `norm`.
- **RETRIEVAL** = dense top-5 over the committed 20k-passage haystack (`squad_corpus.json` +
  cached embeddings, the cycle-68 apparatus, `all-MiniLM-L6-v2`); `supported` = strict
  `mentions(revised, retrieved_passages)`.
- **COMBINED** = `S_frame + 1.0 × [supported]` — a frozen additive rule, no tuned weights. External
  support lifts an item by exactly one full unit of belief-agreement; ties broken by ascending item
  index.

## Frozen gates

- **V1 (validity — miss ⇒ INVALID):** ≥ **25** items with `revised` correct AND ≥ **25** incorrect;
  pool disjointness — 0 overlap of question text with every question in the six committed SQuAD
  pools (cycles 62–73) — **asserted in code**.
- **TG1 — the instrument bar the one-channel verifier missed:** selective accuracy over the **top
  50% by COMBINED** ≥ **0.80** (`G3_COVERAGE`/`G3_FLOOR` imported from the cycle-77 module).
- **TG2 — LOAD-BEARING, retrieval must ADD:** selective accuracy at 0.50 coverage of COMBINED minus
  that of S_frame alone ≥ **0.05** (`G2_MARGIN` imported from cycle 77). Miss ⇒
  `CLOSED_NEGATIVE__retrieval_adds_nothing`.
- **TG3 — the mechanism check (labels the finding, does not decide the verdict):** within the
  confident stratum (`S_frame = 1.0`), accuracy of supported items minus accuracy of unsupported
  items ≥ **0.15** (`LG3_MARGIN` imported from cycle 75), **powered** only if both cells hold ≥
  **15** items (`MIN_CELL = 15`, frozen here). TG3's outcome qualifies the mechanism sentence in
  the FINDING (did retrieval really split confident-right from confident-wrong?); the instrument
  verdict rests on TG1/TG2 alone, committed in advance so a TG3 shortfall cannot be spun either way.

## Pre-committed outcomes

- **V1 + TG1 + TG2 pass** → `SURVIVED__two_channel_verifier_clears_the_bar`. Earned: a label-free
  (labels nowhere in the loop; retrieval is corpus lookup, not supervision) selective verifier
  clearing the floor the one-channel design missed, with the additive value of external evidence
  measured. Not earned: anything beyond SQuAD-style short answers at 7B-4bit; nothing about MC
  formats (cycle 81's negative stands there), frontier models, or open-web retrieval.
- **TG1 miss** → `CLOSED_NEGATIVE__two_channel_misses_instrument_floor`.
- **TG2 miss (TG1 may pass)** → `CLOSED_NEGATIVE__retrieval_adds_nothing` — the belief alone
  suffices at whatever level it reaches, and the two-channel story is not cashed on this substrate.
- **V1 miss** → `INVALID__underpowered`, results withheld.

## Reported but NOT gated

Coverage–accuracy curves for COMBINED and S_frame; retrieval support rate and gold-in-top-5 rate;
the confident-stratum 2×2 (supported/unsupported × right/wrong); neutral unanimity share; caving
and rescue rates on this pool (the content-free challenge's first measurement on SQuAD at 7B);
S_frame AUROC for continuity with cycle 81 (different format — context, not a gate).

## Scope, stated in advance

Qwen2.5-7B-Instruct in 4-bit; SQuAD-v2 short answers (1–3 words) with strict normalized matching;
one content-free challenge turn; N=10 neutral samples; dense retrieval over a fixed 20k-passage
haystack that is topically matched to the items by construction — **this is the friendly case for
retrieval, and the scope statement must say so**: a pass licenses the mechanism on in-corpus
material and says nothing about open-domain coverage. Nothing transfers to MC formats or frontier
scales.

## Frozen constants

`AGENT_MODEL = Qwen/Qwen2.5-7B-Instruct` via the cycle-66 `QuantLoopModel` · `N_ITEMS=240` ·
`SEED=820000` (fresh; prior pools 730000-family and 740000–810000) · `N_SAMPLES=10` · `TOP_K=5`
(imported from the cycle-68 module) · `MIN_CELL=15` · gates
`G3_COVERAGE`/`G3_FLOOR`/`G2_MARGIN` from cycle 77, `LG3_MARGIN` from cycle 75 · matching =
`norm`/`mentions`/`parse_final` from the arc's frozen modules. The fresh pool is built
deterministically (SEED above), written to `squad_pool_v7.json`, and committed with this prereg;
phase A checkpoints one JSONL line per item and resumes on rerun.
