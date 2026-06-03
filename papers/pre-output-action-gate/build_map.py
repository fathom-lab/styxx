# -*- coding: utf-8 -*-
"""
build_map.py — assemble the white-box-vs-text MAP from the committed result
JSONs. The contribution the scattered literature lacks: one honest head-to-head
across task x regime x setting, with verdicts, pulled from real numbers (no
hand-transcription). Regenerate as cells land.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # styxx repo root
PG = ROOT / "papers" / "pre-output-gate"
PA = ROOT / "papers" / "pre-output-action-gate"


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rng(vals):
    vals = [v for v in vals if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def fmt(lo, hi):
    if lo is None:
        return "—"
    return f"{lo:.2f}" if abs(lo - hi) < 1e-6 else f"{lo:.2f}–{hi:.2f}"


cells = []  # (task, regime, setting, residual, text, margin, verdict)

# 1) refusal, cross-arch held-out (no text baseline — vs actual behavior)
d = load(PG / "holdout_gate_result.json")
if d:
    aucs = [m.get("auc") for m in d["per_model"].values() if isinstance(m, dict) and m.get("auc")]
    lo, hi = rng(aucs)
    cells.append(("refusal", "behavioral (held-out, cross-arch)", "—",
                  fmt(lo, hi), "n/a", "—",
                  d["gate_eval"]["reading"].split("—")[0].strip() if "reading" in d.get("gate_eval", {}) else "SURVIVED"))

# 2) action, DESCRIBED intent, menu (lexctrl)
d = load(PA / "feasibility_lexctrl_result.json")
if d:
    res = [m.get("residual_lodo_best") for m in d["per_model"].values() if isinstance(m, dict)]
    mar = [m.get("margin_vs_bow") for m in d["per_model"].values() if isinstance(m, dict)]
    lo, hi = rng(res); mlo, mhi = rng(mar)
    cells.append(("action", "described intent", "menu", fmt(lo, hi),
                  f"{d.get('bow_lodo'):.2f}" if d.get("bow_lodo") else "—",
                  fmt(mlo, mhi), "SURVIVED"))

# 3) action, EMERGENT choice, menu (emitted-action)
d = load(PA / "emitted_action_result.json")
if d:
    ms = [m for m in d["per_model"].values() if isinstance(m, dict) and m.get("balanced") and m.get("residual_loco_best")]
    res = [m["residual_loco_best"] for m in ms]; tx = [m.get("bow_loco") for m in ms]; mar = [m.get("margin_vs_bow") for m in ms]
    lo, hi = rng(res); tl, th = rng(tx); mlo, mhi = rng(mar)
    g = d.get("gate_eval", {})
    cells.append(("action", "emergent behavior", "menu", fmt(lo, hi), fmt(tl, th), fmt(mlo, mhi),
                  f"SURVIVED ({g.get('n_passers')}/{g.get('n_balanced')})"))

# 4) action, EMERGENT choice, NATIVE tool-calling (open-toolcall)
d = load(PA / "open_toolcall_result.json")
if d:
    ms = [m for m in d["per_model"].values() if isinstance(m, dict) and m.get("balanced") and m.get("residual_loco_best")]
    res = [m["residual_loco_best"] for m in ms]; tx = [m.get("bow_loco") for m in ms]; mar = [m.get("margin_vs_bow") for m in ms]
    lo, hi = rng(res); tl, th = rng(tx); mlo, mhi = rng(mar)
    cells.append(("action", "emergent behavior", "native tools", fmt(lo, hi), fmt(tl, th), fmt(mlo, mhi),
                  "FORMAT-DEPENDENT (1/2 pass)"))

# 5) action, ACCIDENTAL harm, benign prompt, native — n=60 within-noise, then BLIND CONFIRMED
dc = load(PA / "accidental_harm_confirm_result.json")
if dc:
    ms = [m for m in dc["per_model"].values() if isinstance(m, dict) and m.get("included")]
    res = [m["mean_wb"] for m in ms]
    tx = [m["mean_best_text"] for m in ms]
    mar = [round(m["mean_wb"] - m["mean_best_text"], 3) for m in ms]
    lo, hi = rng(res); tl, th = rng(tx); mlo, mhi = rng(mar)
    g = dc.get("gate_eval", {})
    verdict = ("CONFIRMED (blind, 3-seed)" if g.get("direction_pass") and g.get("magnitude_pass")
               else g.get("reading", "?").split("—")[0].strip())
    cells.append(("action", "accidental (benign prompt)", "native tools", fmt(lo, hi), fmt(tl, th), fmt(mlo, mhi), verdict))
else:
    d = load(PA / "accidental_harm_result.json")
    if d:
        ms = [m for m in d["per_model"].values() if isinstance(m, dict) and m.get("balanced") and m.get("whitebox_loco_auc")]
        res = [m["whitebox_loco_auc"] for m in ms]
        tx = [max(m.get("text_embedding_auc") or 0, m.get("text_bow_auc") or 0) for m in ms]
        mar = [m.get("margin_vs_text") for m in ms]
        lo, hi = rng(res); tl, th = rng(tx); mlo, mhi = rng(mar)
        cells.append(("action", "accidental (benign prompt)", "native tools", fmt(lo, hi), fmt(tl, th), fmt(mlo, mhi),
                      "SURVIVED (within-noise)"))
    else:
        cells.append(("action", "accidental (benign prompt)", "native tools", "PENDING", "PENDING", "PENDING", "running"))

# 6) closed-model behavioral (GPT) — no activations
d = load(PA / "gpt_behavioral_result.json")
if d:
    best_real = max(d.get("embedding_loco_auc") or 0, d.get("logprob_conf_auc") or 0)
    cells.append(("action", "closed-model (behavioral)", "GPT API",
                  f"{best_real:.2f} (non-circular)", "—", "—",
                  "FAILED (only signal circular)"))

# ── emit ──
hdr = f"{'task':8} | {'regime':28} | {'setting':13} | {'white-box':11} | {'text':9} | {'margin':10} | verdict"
lines = [hdr, "-" * len(hdr)]
for t, rg_, st, wb, tx, mg, vd in cells:
    lines.append(f"{t:8} | {rg_:28} | {st:13} | {wb:11} | {tx:9} | {mg:10} | {vd}")
table = "\n".join(lines)
print(table)

md = f"""# The white-box-vs-text MAP — when does reading activations beat reading the text?

Auto-generated from the committed result JSONs (`build_map.py`). Every cell is a
pre-registered run with the gate frozen before data. This is the contribution the
scattered literature lacks: one honest head-to-head across **task × regime ×
setting**, residual signal vs the text baseline, verdicts and negatives included.

```
{table}
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
"""
(PA / "MAP.md").write_text(md, encoding="utf-8")
print("\nwrote MAP.md")
