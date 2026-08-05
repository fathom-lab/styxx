# FINDING — B44: the bridge was never about llama — the clique shares a frame, and qwen deviates from *it*

Fathom Lab · 2026-08-05 · prereg: `PREREG_b44_wrong_donor_2026_08_05.md` (frozen at commit
`b303a35` before the scored run) · receipts: `b44_result.json`, `b42_result.json` · scored by
`styxx.protocol` from the frozen gates block.

## Verdict (machine-computed)

**`SHARED_FRAME__any_clique_frame_corrects_the_island`**

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_positive_control | min bridge at k=20 ≥ 0.30 | 0.9745 | ✅ |
| G1_wrong_max_low | max wrong-donor at k=20 ≤ 0.15 | 0.9056 | ❌ |
| G2_wrong_min_high | min wrong-donor at k=20 ≥ 0.30 | 0.3622 | ✅ |

Every wrong-donor surgery — every donor, every seed — cleared the 0.30 replication floor at
k=20. The B42 alternative reading ("only the reader's own frame corrects the island") is dead:
frames from **gemma_2b** (different family) and **llama_1b** (different model, reader's family),
computed by the identical construction, each substantially open the island when swapped into
qwen's concept space.

## The numbers, and their texture

| arm | median at k=20 | seed range at k=20 | median at k=2 |
|---|---|---|---|
| bridge (reader llama_3b's frame) | 0.9821 † | 0.9745 – 0.9847 | 0.5128 † |
| wrong-donor gemma_2b | 0.7168 | 0.5102 – 0.8316 | 0.3214 |
| wrong-donor llama_1b | 0.648 | 0.3622 – 0.9056 | 0.3036 |
| random frame (B42) | 0.0026 † | — | 0.0051 † |

† from `b42_result.json` (same seeds, same machinery).

Two facts sit side by side and both matter:

1. **The shared frame carries most of the signal.** Wrong-model donors recover roughly
   two-thirds of full legibility at the median — from models the surgery never saw the reader
   through. The correction is therefore mostly *not* pair-specific: the clique members share a
   common concept-frame geometry, the island's barrier is chiefly qwen's private rotation away
   from that **shared** frame, and restoring any clique member's version of it re-admits qwen.
   Even the rank-2 core transfers partially (0.3214 / 0.3036 vs the bridge's 0.5128).
2. **A reader-specific residual remains — and it is the reliable part.** The reader's own frame
   is both better (0.9821 vs 0.7168 / 0.648) and *stable* (five seeds within 0.9745–0.9847),
   while wrong-donor correction is erratic (llama_1b spans 0.3622 to 0.9056 across seeds; gemma
   spans 0.5102 to 0.8316). Restoring the shared geometry gets you most of the way,
   unreliably; only the reader's exact frame closes the gap every time.

## What this changes upstream

B41 named its verdict "the *named contrasts* causally block legibility," and B42 measured that
correction's rank. B44 relocates the cause: the operative object is not llama's frame but the
**clique's common frame**, of which llama's copy is one instance — the best-aligned instance
for reading *by llama*, which is why the true bridge still wins. This strengthens, not weakens,
the arc's endpoint: a shared cross-family geometry exists (a candidate convergent structure),
qwen's deviation from it is causal, low-rank, and — per b43 — has no name in human language.

## Limits, stated plainly

- Two wrong donors, both clique members. Whether a frame from a *non-clique* model (a second
  island, if one exists) also corrects — the decisive test of "shared frame" vs "any trained
  frame" — needs a model outside the clique and is the natural successor.
- One island, one target. "The clique shares a frame" is measured here as: two other members'
  frames substantially substitute for the reader's. A direct frame-to-frame geometry comparison
  (principal angles between clique members' top-k frames) would state it without the discovery
  in the loop.
- Wrong-donor variance across seeds is reported, not explained. Which seeds' row subsets make a
  foreign frame land well is an open, checkable question.

*Prereg frozen before the run; the G1∧G2 branch structure was written before data, including
the SHARED_FRAME reading this result selected. Every number grounds in `b44_result.json` or
`b42_result.json`. Sealed before commit.*
