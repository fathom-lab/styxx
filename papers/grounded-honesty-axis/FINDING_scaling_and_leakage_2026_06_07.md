# FINDING — Scaling + Leakage: direct response to external review (the read-certificate is semantic and scale-robust, but NOT scale-increasing)

**2026-06-07. Fathom Lab / styxx.** An external reviewer named four audit items and two "decides whether
this is a curiosity or a foundation" questions: **(1) leakage pathways** ("first place I'd attack") and
**(4) scaling laws** ("the million-dollar question"). Both are runnable on existing on-disk data. We ran
them. Verdict: **the held-knowledge recovery is semantic (survives covariate partialling) and scale-robust
(present + validity-gated 1.5B→7B), but does NOT show a clean scale-increasing law** — it neither
strengthens nor degrades monotonically with scale in the ≤7B range. The reviewer's reframe is adopted:
this is *"proof that a representation existed,"* a verify-don't-trust primitive — not "mind reading."

## #4 — Scaling law (Qwen 1.5B → 3B → 7B + Llama-3B, identical frozen scorer `run_reclimb.py`)

| model | LIE_rec | chance floor | **elevation** | PRIME | ABORT | G6 | verdict |
|---|---|---|---|---|---|---|---|
| Qwen-1.5B | 0.886 | 0.329 | **0.557** | 0.957 | 0.000 | 0.96 | SURVIVED |
| Qwen-3B | 0.702 | 0.318 | **0.384** | 0.933 | 0.045 | 0.84 | SURVIVED |
| Qwen-7B | 0.819 | 0.330 | **0.489** | 1.000 | 0.000 | 0.58 | REPORT_AS_LANDED (G6<0.60) |
| Llama-3B | 0.607 | 0.357 | **0.250** | 0.988 | 0.030 | 0.60 | SURVIVED |

**Honest read.** The phenomenon is **scale-ROBUST, not scale-dependent**: held content is linearly
readable when present at *every* size (PRIME 0.93–1.00, near ceiling throughout), the fabrication kill
collapses at every size (ABORT ≤0.045), and elevation stays well above chance (0.25–0.56). But the
elevation is **non-monotonic** in scale (0.56 / 0.38 / 0.49 across Qwen 1.5B/3B/7B) — there is **no clean
"stronger with scale" law**, and equally **no degradation** with scale (the reviewer's specific worry —
larger models learning concealment — is **not** observed in ≤7B). 1.5B is the *strongest*, so the
capability does not require scale. Caveats: 4 points, each model partitions its own LIE/MISTAKE/RESISTED
items (not a controlled same-item ladder), and the 7B point is REPORT_AS_LANDED (its RESISTED instrument
positive-control marginally failed, G6 0.583). **Conclusion: scale-robust existence, no scaling law — a
first pass, not the final word; a controlled same-item scale ladder is owed.**

## #1 — Leakage audit (`run_leakage_audit.py`, Qwen-3B): is it semantic or a shortcut?

Residualize the residual-stream features on each covariate the reviewer named, *before* the gold probe;
if the elevation survives, the recovery is not that shortcut.

| partial out | LIE_rec | elevation | Δ from base |
|---|---|---|---|
| none (base) | 0.664 | +0.318 | — |
| activation magnitude (‖resid‖) | 0.664 | +0.318 | 0.000 |
| length | 0.664 | +0.318 | 0.000 |
| confidence (neutral_margin) | 0.645 | +0.299 | −0.019 |
| token-stats (letter_margin + vocab_entropy) | 0.626 | +0.280 | −0.038 |
| **ALL jointly** | 0.570 | **+0.224** | **−0.094** |

**Verdict: SEMANTIC.** Magnitude and length carry *none* of the signal; confidence and token-stats carry
a little; jointly partialling all four leaves **70% of the elevation intact (+0.224 ≥ 0.15)**. The
recovery is not explained by the leakage pathways the reviewer flagged. (This is a focused fixed-layer
check, n_LIE-test small; it complements the read-certificate's existing leakage controls — G5
asserted-letter 0.01, G8 gold≪chosen, the same-item never-knew route.)

## The two items still OWED (the reviewer's #2 and #3 — the harder ones)

- **#3 Adversarial training** (the reviewer's "most important unanswered question"): can a model be
  *trained* to keep the answer, answer falsely, AND suppress the probe signature? This is the difference
  between "a primitive" and "an arms race," and it needs **GPU fine-tuning against the probe** — OWED,
  not yet run. The honest prior: read≠write (suppression is a *write* on the representation) is on our
  side, but this must be tested, not assumed.
- **#2 Capture trust**: who captured the activations, when, from which model, unmodified? This is the
  **trustless-capture gap** already named as a hard L2/L3 gap in `STYXX_PROTOCOL.md` — unsolved, stated
  as loud as the capabilities.

## What the reviewer got right (adopted into the framing)

*"Proof that a representation existed"* is the honest category — narrower than "mind reading," and more
durable. The crypto analogy holds: the leap is **don't trust the model's output, verify evidence that the
claimed internal representation existed.** This finding strengthens that claim on two axes (semantic, not
leakage; robust, not scale-fragile) and names the two axes that still decide its ceiling (adversarial
robustness, trustless capture). Receipts: `reclimb_result_qwen{15,3b,7b}.json`, `reclimb_result_llama3b.json`,
`leakage_audit_qwen3b.json`, `run_leakage_audit.py`.
