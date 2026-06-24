# FINDING — overconfidence's length signal is PARTLY INTRINSIC; the offline rebuild is an honest NULL

**2026-06-24. Pre-registered `PREREG_overconfidence_length_robust_2026_06_24.md` (`a2708e8`, frozen BEFORE
generating). Offline, local-GPU, NO frontier key.** Reproduce:
`python scripts/overconfidence_length_robust.py --generate --model {qwen,gemma}` then `--model {qwen,gemma}`.

## What was attempted and the frozen result
The suite causal audit named **overconfidence the one causally length-confounded instrument**
(`FINDING_suite_causal_length_2026_06_24.md`). The owed fix (blocked yesterday on a frontier key) was a
corpus-level length-matched rebuild. Attempted offline with local generators under the frozen ship/no-ship
bars. **Result: HONEST NULL across both clean generators** (Qwen2.5-3B and gemma-2-2b-it; Phi-3.5 errored on
a transformers version mismatch). Neither could produce a corpus that is simultaneously construct-valid AND
length-matched — and the reason is the finding.

| generator | calib vs overconf words | d_len | hedge std-diff | refit full / no-length CV-AUC | gate |
|---|---|---|---|---|---|
| gpt-4o-mini (original) | 76.6 / 65.3 (1.17×) | −0.61 | — | 0.770 / — | (confounded baseline) |
| Qwen2.5-3B @ ~55w ask | 30.4 / 26.2 (1.16×) | −0.47 | −0.30 | 0.656 / 0.597 | FAIL (register partial + length unmatched) |
| gemma-2-2b-it @ ~55w ask | 54.4 / 47.0 (1.16×) | −0.74 | −0.52 | 0.767 / 0.677 | FAIL (length unmatched) |

## The actual finding: calibration carries a ~16% verbosity tax
Across **three model families and three very different absolute lengths** (76w / 54w / 30w), calibrated
answers are a near-invariant **1.16–1.17× longer** than overconfident answers — even when the model is
explicitly instructed to write equal length. Hedging *costs words*: stating a caveat ("where the evidence is
partial…", "widely but not universally accepted…") is intrinsically more verbose than a flat assertion
("There are seven continents."). **For overconfidence, response length is PARTLY a construct-intrinsic
feature, not a purely spurious confound** — the same category as loop (where length is fully intrinsic),
milder. You cannot length-match a calibrated/overconfident corpus without fighting the construct, which is
exactly why all three generators produced longer calibrated answers and the rebuild returns null.

## What this corrects and what it means
- **Corrects my own suite finding's wording.** Calling overconfidence "length-confounded" overstated it. More
  precisely: its discrimination is **partly length-mediated, and that length is partly intrinsic to calibrated
  register.** The length-free register floor is ~0.68 (Qwen 0.60 / gemma 0.68, refit dropping length); the
  deployed 0.770 adds ~0.09 of length, of which an unknown fraction is the legitimate verbosity-of-calibration.
- **The naive "fix" would be wrong.** Force-ablating the length features (→ ~0.68–0.72) deletes partly-real
  signal and was correctly NOT shipped (yesterday's instinct, now explained).
- **Decision: NO new weights. Ship a CAVEAT, not a v0.3.** overconfidence_v0 keeps its weights; the honest
  disclosure is: "uses response length as a partial cue; ~0.09 AUC of the 0.770 headline is length, which is
  itself partly the legitimate verbosity of calibrated register and partly spurious — interpret short-vs-long
  scores with care." A clean spurious-vs-intrinsic split needs a frontier-key regen with HARD length control
  (and may remain construct-limited).

## Honest scope (this is a null, reported as one)
- Two clean generators agree on the null for the SAME structural reason (length↔register entanglement) →
  robust. Phi-3.5 was a tooling error, not evidence.
- The ~16% calibration-verbosity ratio is the one positive, replicable result (3 families, n=200 each); it is
  a micro-finding about epistemic register, not a headline guardrail result.
- A pre-registered fix returning a clean null, reported without dressing it as a ship, is the intended
  discipline. No instrument was degraded; understanding improved; a wrong fix was prevented.
