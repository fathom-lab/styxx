# FINDING (settling) — template probe directions are orthogonal to the real truth axis, robustly; the axis is recoverable only from NATURAL data

**2026-06-24. Pre-registered `PREREG_truth_axis_settling_2026_06_24.md` (`f6a1e04`). Offline, local-GPU, NO
frontier key.** Settles the open question from `FINDING_truth_direction_deflation` (was the OOD-transfer
failure "construct too narrow" or robust?). Reproduce: `python scripts/build_wide_truthset.py`;
`python scripts/build_ood_naturals.py`; `python scripts/truth_axis_settling.py`.

## Design (the powered, de-confounded version)
- **Wide construct:** 14 domains, 260 cyclic-derangement minimal pairs (false BY CONSTRUCTION). Silence
  re-verified: adversary-fair BoW leave-one-domain-out **0.501** (exactly chance).
- **Large OOD natural test:** 70 curated misconceptions(false)/surprising(true) → **multi-agent fact-checked**
  (3 independent verifiers, union drop) → **9 contested items dropped** (Mpemba, Eiffel ">15cm", honey,
  goldfish-stunting, blue-whale-largest, hummingbirds, etc.) → **61 verified** natural statements.
- Fit on the wide construct, test OOD; permutation null (1000x), cosine to the in-OOD-internal direction.

## Result — widening the construct does NOT recover the axis (both models)
| | Qwen (L19) | Llama (L14) |
|---|---|---|
| wide-construct dir → OOD (fair AUC) | 0.554 | 0.667 |
| **permutation-null p** (vs random construct directions) | **0.634** | **0.243** |
| **cosine(wide dir, real OOD-internal dir)** | **−0.05** | **+0.14** |
| truth axis EXISTS? (OOD-internal leave-one-out) | **0.878** | **0.941** |
| BoW construct→OOD (text silence floor) | 0.543 | 0.543 |

- **No significant transfer on either model** (p=0.63 / 0.24): the wide-construct direction predicts natural
  truth no better than a RANDOM direction. (The single-shuffle control looked like 0.64–0.68 "partial" for
  Llama; the proper 1000x permutation null shows the fair-AUC floor at n=61 is ~0.6 and 0.667 is inside it —
  noise, not transfer.)
- **Still orthogonal** to the real truth axis (cosine ≈ 0), exactly as in the narrow 6-template construct.
- **The truth axis robustly exists** and is linearly recoverable — but from NATURAL data (OOD-internal LOO
  0.88/0.94), NOT from the template construct.

## Conclusion (the settled, definitive claim)
**Template-based probe constructs yield directions ORTHOGONAL to the model's real concept axis and transfer
to natural statements no better than chance — and this is ROBUST to construct width (6→14 domains, n=25→61,
permutation-tested).** A 0.98 in-construct AUC reflects template surface structure, not the concept. The
model's real truth axis is present and recoverable, but ONLY by fitting on the natural distribution itself.
The fix for a failing probe is NOT "add more template domains"; it is "fit on natural, non-templated data and
validate by OOD transfer + orthogonality to the natural-data direction."

## Why this matters (the contribution, now rigorous)
AI-safety/interpretability increasingly validates oversight probes (deception, truthfulness, harmfulness) on
constructed statement sets. This shows a constructed probe can be **fully validated by every standard check —
high accuracy, cross-domain generalization, a verified text-silence gate — and still be orthogonal to the
concept**, with the failure invisible until you test natural OOD transfer AND cosine to the natural-data
direction. The required validation battery: natural held-out transfer (permutation-tested) + orthogonality
check. Demonstrated robustly, dual-model, permutation-controlled, with a multi-agent-verified ground-truth set.

## Honest scope
2 local 3B models; wide construct = 14 TEMPLATE domains (a frontier version would use thousands of natural
statements + frontier models — needs scale/key); OOD n=61 (kills the n=25 caveat) fact-verified. Single seed,
linear probes. Verdict mechanical against the frozen prereg: ROBUST-FAILURE (Qwen) / PARTIAL-but-non-
significant (Llama) → jointly, the transfer failure is robust. 10th rigorous self-check of the day; here the
proper permutation null prevented over-reading Llama's 0.667.
