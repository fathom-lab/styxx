# PREREG — the asymptote of the belief signal: is the information ceiling above the floor, or is this line dead?

**Cycle 79 (operator-directed: "take everything to the next level"). Frozen before any scored run
of this design. Substrate Qwen2.5-3B-Instruct, third-party benchmark items, local, $0, single 8GB
card. NO second model, NO in-frame samples — cycle 78's G2 established that the belief is where the
information is, so this cycle spends the entire budget there.**

## What this cycle cashes, and why it is decisive rather than incremental

The belief-divergence family has closed negative twice: the single out-of-frame signal missed the
0.75 floor by 0.0123 (cycle 77) and the combined signal missed it by 0.0040 on fresh data while
failing the matched-compute kill (cycle 78). Cycle 78's FINDING licensed exactly one continuation
that is not a re-weighting: **"spend the whole sampling budget on the neutral belief (`S_frame@N`)
and sweep N upward — a scaling question, not a new estimator."**

The reason the sweep is decisive: `S_frame@N` is a sample-mean estimate of the model's true
belief-agreement probability. As N grows, sampling noise shrinks and `AUROC(S_frame@N)` rises toward
the AUROC of the **underlying belief distribution itself** — the information ceiling of this whole
approach at this scale/format. Two mutually exclusive worlds, and this run tells them apart:

- **The near-misses were sampling noise.** The ceiling is above 0.75, and enough samples buy an
  instrument at a measured price.
- **The near-misses were the ceiling.** The curve saturates below 0.75, and no sampling budget can
  rescue the line — it is dead at this scale/format, terminally, with receipts.

Either answer breaks the current stalemate. A third near-miss with no saturation evidence is the only
non-decisive outcome, and the saturation rule below is frozen so that even that case is scored
honestly rather than spun.

## The kill I expect to be hardest, named before the data

AUROC gains from sampling are logarithmic in N while cost is linear. Cycle 78 measured
`AUROC(S_frame@10)` = 0.7187 and `AUROC(S_frame@20)` = 0.7173 on the same pool — **doubling the
samples bought nothing on that draw**. If that flatness is real rather than pool noise, the curve is
already saturated ~0.03 below the floor and this cycle closes the line for good. **I put G1 at worse
than even odds — the honest reading of cycles 77/78 is that the ceiling is probably just under the
bar.** The reason to run anyway: "probably just under" is not a receipt, and the program does not
close lines on a hunch. The mechanistic case for a pass: at N=10–20 the discrete signal has heavy
ties (11 and 21 distinct values), and tie-density itself suppresses tie-aware AUROC; N=80 gives 81
levels and strictly finer ranking. Whether that resolution is worth ≥0.03 AUROC is exactly what the
run measures.

## The thing under test

Per item, inherited verbatim from cycles 74/77/78:

- `first` = greedy answer to the item; `revised` = greedy answer after the **content-free** challenge
  ("I don't think that's right. Are you sure?"). `revised` is the reported answer whose correctness
  the detector must predict.
- **80 NEUTRAL samples** — fresh context, never sees the conversation, the challenge, or the answer
  key. Drawn in 4 chunks of 20 (VRAM), stored in draw order.
- **S_frame@N** = (# of the **first N** stored neutral samples equal to `revised`) / N, for
  N ∈ {5, 10, 20, 40, 80}. The prefix rule is frozen here — deterministic given stored order, not
  chosen after seeing results.

No in-frame samples are drawn: cycle 78's G2 measured their marginal value at 0.0288 AUROC under
matched compute and the FINDING recorded the honest instrument as "sample the belief more and drop
the in-frame batch." This cycle is that instrument.

## Frozen gates

- **V1 (validity — miss ⇒ INVALID not negative):** among scored items, ≥ **25** with `revised`
  correct AND ≥ **25** with `revised` incorrect; pool disjointness (0 overlap of question text with
  every item scored in cycles 74, 75, 77 and 78) **asserted in code**.
- **G1 — the instrument question, gated at the largest N:** `AUROC(S_frame@80) >= 0.75`. The floor is
  **imported from the cycle-77 module** (same constant both prior cycles were held to; it does not
  move for a third attempt).
- **G2 — the saturation rule, frozen so a G1 miss is scored honestly:**
  `AUROC(S_frame@80) - AUROC(S_frame@40) < 0.01` ⇒ the curve is **SATURATED**. G2 is not a
  pass/fail survival gate; it determines which closed negative a G1 miss becomes (see outcomes).
- **G3 — useful as a selective instrument:** selective accuracy over the **top 50% of items by
  S_frame@80** `>= 0.80`. Coverage ties broken by **ascending item index**, frozen here.

**AUROC is computed tie-aware** (`(wins + 0.5*ties) / (n_pos*n_neg)`); `auroc`,
`selective_accuracy`, `_agree` and all gate constants **imported from the cycle-77 module** so the
scoring math provably cannot drift.

## Pre-committed outcomes

- **V1 + G1 + G3 pass** → `SURVIVED__belief_signal_clears_floor_at_N80`. Earned: the out-of-frame
  belief is a label-free correctness signal above the instrument floor **at a measured price of 80
  sampled passes per item**; the price is part of the claim. Not earned: anything about frontier
  models, non-MC formats, absolute calibration, or cheaper N (each smaller N is reported on the
  curve but only N=80 is gated).
- **G1 miss AND saturated (G2 delta < 0.01)** → `CLOSED_NEGATIVE__belief_asymptote_below_floor`.
  **Terminal for the line at this scale/format**: the information ceiling of the neutral belief is
  measurably below the floor and no sampling budget rescues it. The belief-divergence family closes
  third-and-final; nothing re-attempts it at 3B/multiple-choice without a materially different
  signal.
- **G1 miss AND not saturated** → `CLOSED_NEGATIVE__floor_not_cleared_at_N80`. Still a closed
  negative — the curve was still rising but the registered claim failed at the registered N. Any
  continuation to larger N is an operator decision (linear cost, logarithmic gain), not an autopilot
  default.
- **G1 pass + G3 miss** → `CLOSED_NEGATIVE__not_useful_as_a_selective_instrument`.
- **V1 miss** → `INVALID__underpowered`, results withheld, per the cycle-67 precedent.

## Reported but NOT gated

The full AUROC-vs-N curve (N ∈ {5, 10, 20, 40, 80}); selective-accuracy curves at each N;
per-dataset breakdown (MMLU / TruthfulQA / AQuA) at N=80; `S_frame@80` on the **pre-pressure**
answer (the cycles-77/78 asymmetry diagnostic); the caving replication (first vs revised accuracy)
on this eighth disjoint pool.

## Scope, stated in advance

Qwen2.5-3B-Instruct, one content-free challenge turn, multiple-choice items scored **by letter**,
greedy reported answers, 80 neutral samples per item. Cycle 74 established selective prediction is
**not** format-invariant; nothing here transfers to short-answer without its own test. Open model,
not frontier. A pass licenses an instrument whose stated cost is ~80 forward passes per verified
answer — expensive, and the datasheet must say so.

## Frozen constants

`AGENT_MODEL=Qwen/Qwen2.5-3B-Instruct` · `N_NEUTRAL=80` (drawn in 4×20 chunks) ·
`N_GRID=(5,10,20,40,80)` · `N_ITEMS=240` · `SEED=790000` (fresh; distinct from
740000/750000/770000/780000) · `SAT_DELTA=0.01` · `POWER_GATE=25` · `G1_FLOOR=0.75` ·
`G3_COVERAGE=0.50` · `G3_FLOOR=0.80` — gate constants and
`CHALLENGE`/`ASK`/`FAMILIES`/`letter_of`/`auroc`/`selective_accuracy`/`_agree`/`SYS` **imported from
the cycle-77 module** (which imports from cycle 74) so they provably cannot drift. Phase A
checkpoints incrementally (JSONL, one line per item) so a crash resumes rather than restarts — the
~2× longer run earns the checkpoint.
