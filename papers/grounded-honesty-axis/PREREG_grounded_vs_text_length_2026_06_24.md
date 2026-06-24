# PREREG — does the length confound vanish in activation space? (grounded > text, mechanistically)

**Frozen 2026-06-24 BEFORE extracting any activation. Offline, local-GPU, NO frontier key.**
Author: styxx (Alex Rodabaugh).

## The question
Today we showed text-based overconfidence/honesty detectors are irreducibly length-confounded: calibrated
register is intrinsically wordier (`FINDING_verbosity_tax`, `FINDING_overconfidence_length_robust`), so a
text-feature probe loses real AUC under length control (overconfidence 0.770 → 0.63–0.73 CEM+ablated). The
program's central thesis is **grounding > text**: read the model's internal state, not its surface words.
Sharp test: **does the length confound that plagues the TEXT probe disappear in an ACTIVATION probe?**

- If activations carry the calibrated/overconfident register **length-invariantly** where text features can't,
  that is a concrete, mechanistic reason grounded probes beat text probes (text probes inherit the
  verbosity-of-honesty confound; activations don't). Citation-shaped, on-thesis.
- If activations are **equally** length-confounded, that is an honest null that BOUNDS the thesis (the
  length entanglement is in the representation, not just the surface) — also worth publishing.

## Design
Corpus: `benchmarks/data/overconfidence/pairs_v0.jsonl` (n=200, gpt-generated calibrated/overconfident,
construct-valid register). Reader model: **Llama-3.2-3B-Instruct** (primary) + **Qwen2.5-3B-Instruct**
(robustness). For each (question, response), feed `question + "\n" + response`, capture the **last-token
residual-stream hidden state** at every layer (last token = single vector regardless of length, mitigating
mean-pool length bias). No generation — forward passes only.

Probes (all linear logistic, seed-pinned 5-fold CV-AUC):
- **TEXT probe**: the 9 overconfidence features (the deployed instrument's surface features).
- **ACT probe**: standardized last-token activations at a chosen layer.

Length control: coarsened-exact length-matching (CEM, the same tool as `suite_causal_ablate.py`) — subsample
so calibrated/overconfident have identical word-count distributions.

**Layer selection (anti-cherry-pick, frozen):** choose the ACT layer by `act_raw` CV-AUC on the FULL corpus
(selection is BLIND to the CEM/length-matched result), then evaluate `act_cem` at that fixed layer. The full
per-layer sweep is reported for transparency; the headline uses the pre-committed selection rule only.

## Metrics
- `text_raw`, `text_cem` (expect ~0.77 → ~0.63–0.73, replicating today).
- `act_raw`, `act_cem` at the selected layer.
- Δ_text = text_raw − text_cem; Δ_act = act_raw − act_cem (how much each loses to length control).
- **Length-leakage check:** Spearman(ACT-probe out-of-fold score, word count) on the matched set — if high,
  activations still encode length (the confound moved, not vanished).
- Bootstrap 95% CIs on the headline AUCs.

## Decision thresholds (FROZEN)
| Verdict | Condition |
|---|---|
| **GROUNDING-WINS** | `act_cem ≥ 0.72` AND `(act_cem − text_cem) ≥ 0.05` AND `Δ_act < Δ_text` (activations retain length-matched register the text probe loses) |
| **HONEST NULL (thesis bounded)** | `act_cem < 0.65` OR `|act_cem − text_cem| < 0.05` (activations no better than text under length control) |
| **PARTIAL** | otherwise (report as model-/layer-dependent) |

Cross-model: GROUNDING-WINS is ROBUST only if it holds on BOTH reader models; otherwise report model-dependent.

## Honest scope / threats
- Reader-model activations represent how a model *encodes* calibrated vs overconfident text; this is the
  decodability version of the thesis, not generation-time self-probing (that is the heavier follow-up).
- Activations can themselves encode length (sequence-length leaks into the last-token state); the CEM control
  + the explicit length-leakage Spearman are the guards. A GROUNDING-WINS verdict REQUIRES low residual
  leakage, else it is downgraded to PARTIAL.
- In-silico, 2 local 3B models, single seed, n=200. Linear probes only (the honest, non-overfit choice).
- Pre-registered; layer selection rule fixed above; verdict computed mechanically.
