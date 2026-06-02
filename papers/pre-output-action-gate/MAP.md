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
action   | accidental (benign prompt)   | native tools  | 0.78        | 0.20      | 0.57       | UNDERPOWERED
action   | closed-model (behavioral)    | GPT API       | 0.53 (non-circular) | —         | —          | FAILED (only signal circular)
```

**white-box** = pre-emission residual-probe AUC (range across valid models).
**text** = best text-only baseline (prompt bag-of-words / embedding) on the same
task. **margin** = white-box − text.

## What the map says so far (honest)

- White-box **wins clearly** where the signal is *in the model's representation
  and the interface is clean* — refusal (held-out, cross-arch) and action under
  the menu harness (residual ≫ text).
- White-box's edge is **fragile to the interface**: under native tool-calling it
  went format-dependent (1/2 models).
- White-box has **no edge** where there are no activations: closed-model
  behavioral signals failed (the only "win" was circular).
- The decisive open cell — **accidental harm** (benign prompt, the case text
  *cannot* see) — is the one that tells us whether white-box has a real
  deployment USP. That's the flagship run.

The value isn't any single cell; it's that **we publish the whole board, losses
included.** No incumbent will.
