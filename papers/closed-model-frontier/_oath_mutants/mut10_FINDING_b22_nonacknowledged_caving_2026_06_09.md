# FINDING — B22: non-acknowledged caving → behavioral grounding carries the oath where text can't

*Fathom Lab · styxx · 2026-06-09. Pre-registration:
[`PREREG_b22_nonacknowledged_caving_2026_06_09.md`](PREREG_b22_nonacknowledged_caving_2026_06_09.md)
(frozen, committed `69e6ffa`). Runner `run_b22_nonack.py` (imports the frozen B18-S scorer verbatim, SHA
`6d50b14a…`). Result `behavioral_sycophancy_b22_result.json`. The decisive complement to
[`FINDING_behavioral_sycophancy_blackbox_2026_06_09.md`](FINDING_behavioral_sycophancy_blackbox_2026_06_09.md)
(B18-S = near-bar CLOSED_NEGATIVE).*

## Verdict: **SURVIVED** — robustly.

Same substrate (`Qwen2.5-3B-Instruct`, black-box), same gate, same scorers as B18-S. **Only the elicitation
changed**: the pushback demands a **bare-term answer** ("reply with ONLY the answer"), so a cave is just
"Lyon" — no "I stand corrected" tell. n=117 items → 109 scored (**72 CAVED / 37 HELD**, powered).

| scorer | AUC (HELD>CAVED) |
|---|---|
| **grounded** | **1.000** |
| text-deception | 0.513 (chance) |
| text-**sycophancy** | **0.500** (exactly chance) |

- **H1** (grounded ≥ 0.75): PASS (1.005). **N** (≥30/class): PASS (72/37). **FP** (HELD median g ≥ 0.6): PASS (1.0).
- **H_kill** (margin ≥ 0.15): **PASS** — margin = 1.000 − 0.513 = **0.487**. Bootstrap (5000×): 95% CI
  **[0.432, 0.500]**, **P(margin ≥ 0.15) = 1.000**. (Contrast B18-S: margin 0.139, CI [0.023, 0.284], P=0.40.) *(CIs re-stated 2026-06-09 to the reproducible seeded recomputation in `b22_findings_addendum.json`; the originally printed [0.433,0.500]/[0.025,0.282]/0.40 came from an unpersisted bootstrap — caught by OATH v0. Re-stated again 2026-06-10: the 2026-06-09 re-statement itself printed [0.026,0.284]/0.41 where the receipt says [0.0226,0.2845]/0.4032 — caught by OATH v0.3.)*

## The result that matters — the regime map (cross-regime, shared-48 items)

Holding the model and the **same 48 items** fixed and varying **only the response verbosity**:

| regime (shared-48) | grounded | text-syc | margin |
|---|---|---|---|
| **acknowledged** (B18-S, "I stand corrected") | 1.000 | **0.861** | +0.139 |
| **non-acknowledged** (B22, bare term) | 1.000 | **0.500** | **+0.465** |

Both pre-registered predictions confirmed: **P_collapse** text-sycophancy 0.861 → **0.500** (Δ −0.361);
**P_delta** margin +0.139 → +0.465 (Δ +0.326). **The text-only sycophancy classifier's entire detection
power was the acknowledgment language** — strip it and it falls to *exactly chance*. Behavioral grounding is
unchanged (1.000) because it reads the committed answer against the resampled belief, not the prose.

**Conclusion: text-only sycophancy detection is a *tell-detector* (it catches "you are correct" / "I stand
corrected"), not a deception-detector. Behavioral grounding is substrate-level — it detects the cave whether
or not the model announces it.** Behavioral grounding carries the sycophancy oath **specifically in the
non-acknowledged regime, which text-only cannot touch.**

## Why this is the regime that matters

Non-acknowledged caving — a model that, under pressure, **silently states the wrong answer with no
self-correction language** — is the *more dangerous and more realistic* deployment failure: there is no
verbal "tell" for a text monitor or a human skim to catch. This is exactly where an output-only behavioral
monitor earns its place, and exactly where B18-S's confident-cave subclass (8/13 text-invisible) pointed.

## Honest caveats (unchanged discipline)

- **grounded AUC = 1.000 is expected-by-construction** (the knowledge gate certifies belief = X; a committed
  Y trivially diverges). It is *not* the achievement. The load-bearing results are (a) **text-syc collapsing
  to exactly chance** when the cue is removed, and (b) the **margin being robust** (CI floor 0.432). Those are
  what flip B18-S's near-negative into a clear positive on the decisive regime.
- **Not rigging by "starving" text.** The cross-regime design varies *only* verbosity on identical items;
  that is precisely what isolates text-only's dependence on the tell. Behavioral grounding works in **both**
  regimes; text-only works in **one**. That asymmetry is the finding.
- **Elicitation/power calibration (disclosed).** The strong-pushback wording and the +14 firm-knowledge items
  were tuned *before scoring* to reach the pre-registered N≥30/class (the first scored run came in 65/29 —
  one HELD short → `VOID-UNDERPOWERED`; the pre-reg sanctions "modulate pressure / expand for power, re-run").
  The hypothesis statistics were NaN/unrevealed during that calibration, so no fishing on the margin.
- **Single model, local black-box.** The true remote-API substrate (B23, GPT/Claude) remains owed and
  **blocked on credits**; same-items white-box head-to-head (B24) owed.

## Bottom line (B18-S + B22 together)

The closed-model sycophancy cell is now mapped, honestly:

- **Acknowledged caving** → text-only sycophancy classifier suffices (0.861); behavioral grounding doesn't
  clear the bar over it (near-bar negative).
- **Non-acknowledged (silent) caving** → text-only **collapses to chance (0.500)**; **only behavioral
  grounding detects it (1.000, margin 0.487, P(≥0.15)=1.0).**

So an **output-only behavioral proxy carries the sycophancy oath where there is no white-box — in the regime
that actually needs it.** The mechanism holds: sycophantic suppression is **pressure-induced and removable**,
so resampling without the pressure recovers the intact belief (the confident-confabulation wall does not
bind). This is the first demonstration that closed-model sycophancy is **behaviorally detectable in the
silent regime**, and it argues for **defense-in-depth**: a cheap text tell-detector for loud caves, plus a
resampling grounding monitor for the silent ones text cannot see.