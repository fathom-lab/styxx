# The white-box-vs-text MAP — when does reading activations beat reading the text?

Auto-generated from the committed result JSONs (`build_map.py`). Every cell is a
pre-registered run with the gate frozen before data. This is the contribution the
scattered literature lacks: one honest head-to-head across **task × regime ×
setting**, residual signal vs the text baseline, verdicts and negatives included.

```
task     | regime                       | setting       | white-box   | text      | margin     | verdict
--------------------------------------------------------------------------------------------------------
refusal  | behavioral (held-out, cross-arch) | —             | 0.78–0.97   | n/a       | —          | SURVIVED
action   | described intent             | menu          | 0.84–0.95   | 0.51      | 0.33–0.44  | SURVIVED
action   | emergent behavior            | menu          | 0.65–0.92   | 0.16–0.35 | 0.48–0.71  | SURVIVED (4/5)
action   | emergent behavior            | native tools  | 0.51–0.82   | 0.16–0.27 | 0.35–0.55  | FORMAT-DEPENDENT (1/2 pass)
action   | accidental (benign prompt)   | native tools  | 0.60–0.90   | 0.50–0.57 | 0.05–0.34  | CONFIRMED (blind, 3-seed)
action   | closed-model (behavioral)    | GPT API       | 0.53 (non-circular) | —         | —          | FAILED (only signal circular)
```

**white-box** = pre-emission residual-probe AUC (range across valid models).
**text** = best text-only baseline (prompt bag-of-words / embedding) on the same
task. **margin** = white-box − text.

## What the map says (honest — the flagship cell has now landed)

- White-box **wins clearly** where the signal is in the model's representation and
  the interface is clean — refusal (held-out, cross-arch) and action under the menu
  harness (residual ≫ text).
- **The flagship cell was CONFIRMED on a blind, multi-seed replication.** Accidental
  harm — a *benign* prompt where the model emergently over-reaches to a destructive
  tool, the exact case a text-monitor-on-the-prompt is blind to *by construction*. The
  honest arc: at n=60 it met the frozen gate on 2/4 models, but a Hanley-McNeil CI
  analysis showed the 0.70 line ran through *all four* intervals — the per-model split
  was **within noise**, and the robust unit was the *direction* (white-box > text on
  4/4). We then ran a **pre-registered blind replication**: a fresh 84-scenario
  held-out set authored before any model saw it, 3 seeds, gate keyed on the lower CI
  bound. It **CONFIRMED** — direction held 4/4 in every seed; magnitude cleared
  mean-AUC ≥0.70 with lower-95%-CI ≥0.60 on 3/4 models (Qwen-1.5B 0.87, **Qwen-3B
  0.90**, Llama-3B 0.77); text stayed below the blindness bar. The decisive fact:
  **Qwen-3B, which failed *within noise* at n=60 (0.686), returned 0.90 — all three
  seeds — on data it had never seen.** The CI critique was right, and the signal
  survived it.
- The edge is **real but model-dependent in strength**: the smallest model (Llama-1B)
  holds direction every seed but its magnitude stays weak (0.60, CI lower 0.48) — the
  most-reckless model is the least legible, consistent across n=60 and the blind set.
  And text is *below the bar*, not at chance (≈0.50–0.57), so white-box wins by ~0.30,
  not against zero. An honest bound, not a clean sweep.
- White-box's edge is **fragile to the interface — confirmed, and it's model-specific.**
  In the emergent-choice regime, a multi-seed sampled-native re-test (3 seeds, CIs) found
  Qwen-1.5B **robust** (0.84, margin +0.51) but **Llama-3B genuinely collapsed**
  (0.53 ≈ chance — did NOT recover from the greedy 0.51, so it's a *real* format effect,
  not a labeling artifact), while Qwen-3B simply **declines** the destructive action
  (caution, not failure). Lesson: don't assume the gate transfers to native calling —
  validate per-model under the deployment interface.
- White-box has **no edge** where there are no activations: closed-model behavioral
  signals failed (the only "win" was circular).

## What's been hardened, and what's left

The two degrees of freedom the n=60 run left open are now **closed**: the confirmatory
set was authored *blind* (frozen before any model ran on it), and it was run across
**3 seeds** with the gate keyed on the **lower CI bound**, not a point estimate vs a
line. That is the run that converts "promising" → "claimable", and it landed CONFIRMED.
What would harden it *further* (stated before calling it bulletproof): a **third-party**
scenario set (ours is author-written, blind only in the frozen-before-run sense), a
**5th architecture**, and ultimately a **live agent** rather than simulated tool
schemas. Text sits *below* the blindness bar but not at chance — a cleaner blind regime
would sharpen how we read the margin.

The value isn't any single cell; it's that **we publish the whole board, losses
included.** No incumbent will.
