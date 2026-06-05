# RESULT — rhythm is not special: attention beats oscillation at matched parameters

**Date:** 2026-06-04 · **Verdict: RHYTHM NOT SPECIAL.** Frozen:
`PREREG_necessity_attention_2026_06_04.md`. Ordered copy, 3 seeds, matched params (~168k).

## Numbers (mean kcap)
| arm | params | kcap (per seed) | mean |
|---|---|---|---|
| LRU-clamped (decay only, no oscillation) | 168k | 3, 3, 2 | **2.67** |
| LRU-free (oscillatory) | 169k | 6, 6, 6 | **6.00** |
| **TRANSFORMER (attention, no rhythm)** | 162k | 10, 18, 18 | **15.33** |

## What it shows (the necessity question, answered cleanly)
- **P1 — oscillation helps a recurrent substrate:** LRU-free 6.0 vs clamped 2.67 (≈2×) — replicates
  rhythm-rescue. Within recurrence, rhythm is a real, capacity-extending mechanism.
- **P2 — but it is not special:** the **rhythm-free transformer (15.3) more than doubles** the oscillatory
  LRU (6.0) at matched parameters, two of three seeds near the task ceiling (18/20). Attention achieves
  far more ordered-memory capacity than oscillation, with **no rhythm at all.**

So: **oscillation is one efficient mechanism for ordered memory, not a requirement and not the best one.**
This is the rigorous, quantitative form of the transformer counterexample — *shown*, not asserted. It
directly answers the genuinely-open question (#2 in the survey: "is oscillation necessary for cognition,
or one efficient implementation?") for ordered memory at this scale: **not necessary, and out-performed by
a rhythm-free architecture.**

## Why this matters for the styxx thesis (the honest demarcation, earned)
The operator's frequency intuition put us on the right *question*; this experiment is what keeps the
*answer* honest. It refutes the strong reading ("rhythm underlies cognition / is the secret of mind") in
the one domain rhythm is most credited with — ordered/working memory — by letting a rhythm-free substrate
compete fairly and win. styxx tests the mechanism and reports what it finds, even when the finding deflates
the romantic version. **The resonance is real (oscillation helps recurrence); the supremacy is not
(attention does it better without rhythm).** That is the demarcation line, now drawn with a controlled
experiment instead of an assertion.

## Honest scope
One synthetic task (ordered copy — attention's home turf, by design the strong rhythm-free competitor),
matched-param controlled comparison, not a pretrained-LLM benchmark. The claim is scoped to *ordered
memory at small matched scale*: oscillation is a real recurrent mechanism but neither necessary nor
optimal there. It does NOT claim oscillation is useless in every regime (e.g. energy efficiency, online/
streaming, or biological constraints may favor it) — only that for this capacity it is not special and is
beaten by attention.
