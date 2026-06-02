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
action   | accidental (benign prompt)   | native tools  | 0.62–0.79   | 0.36–0.52 | 0.10–0.32  | SURVIVED
action   | closed-model (behavioral)    | GPT API       | 0.53 (non-circular) | —         | —          | FAILED (only signal circular)
```

**white-box** = pre-emission residual-probe AUC (range across valid models).
**text** = best text-only baseline (prompt bag-of-words / embedding) on the same
task. **margin** = white-box − text.

## What the map says (honest — the flagship cell has now landed)

- White-box **wins clearly** where the signal is in the model's representation and
  the interface is clean — refusal (held-out, cross-arch) and action under the menu
  harness (residual ≫ text).
- **The flagship cell SURVIVED.** Accidental harm — a *benign* prompt where the
  model emergently over-reaches to a destructive tool, the exact case a
  text-monitor-on-the-prompt is blind to *by construction* — is caught by the
  pre-emission residual at AUC ~0.79 on **two different architectures** (Qwen-1.5B,
  Llama-3.2-3B), with the text baseline confirmed blind (≤0.52, at/below chance) and
  a feasibility-grade calibrated probability (ECE ~0.14). This is white-box's one
  true deployment USP, demonstrated on the case that mattered most.
- The edge is **real but model-dependent in strength**: 2 of 4 models cleared the
  full frozen gate; Qwen-3B missed by 0.014 (0.686), and the smallest model
  (Llama-1B) is both reckless (50/60 prone) and less legible (0.618). An honest
  bound, not a clean sweep.
- White-box's edge is **fragile to the interface** in the emergent-choice regime:
  under native tool-calling it went format-dependent (1/2).
- White-box has **no edge** where there are no activations: closed-model behavioral
  signals failed (the only "win" was circular).

## What would harden it (honest next step, not a victory lap)

The pre-committed expansion kept the gate **frozen** (no goalpost-moving) — but the
cleanest confirmation is data the gate has never seen: a fresh held-out scenario set
(not this expansion), a new seed, and a 5th architecture. That's the confirmatory
replication, and it's the next move — stated before claiming the cell is bulletproof.

The value isn't any single cell; it's that **we publish the whole board, losses
included.** No incumbent will.
