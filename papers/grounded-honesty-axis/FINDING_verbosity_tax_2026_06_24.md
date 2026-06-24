# FINDING — the calibration verbosity tax is REGISTER-INTRINSIC and GROWS under compression

**2026-06-24. Pre-registered `PREREG_verbosity_tax_falsify_2026_06_24.md` (`a4cfd8c`, frozen BEFORE
generating). Offline, local-GPU, NO frontier key.** A same-day stress-test of a claim styxx published this
morning. Reproduce: `python scripts/verbosity_tax_falsify.py --generate --model {gemma,qwen}` then
`python scripts/verbosity_tax_falsify.py`.

## The claim under test
This morning's TG + `FINDING_overconfidence_length_robust` stated: calibrated epistemic register is
intrinsically ~1.16× wordier than overconfident register. Obvious confound: the calibrated stance prompt
asks to "acknowledge uncertainty where evidence is partial" (invites elaboration); the overconfident prompt
says "be decisive, don't hedge" (invites brevity). So the 1.16× could be what we *asked for*, not what
calibration *costs*. We tested it before someone else could.

## Test and result
Generate both stances under an identical HARD brevity rule on BOTH ("Answer in ONE sentence, as briefly as
you possibly can"). If the verbosity were a prompt artifact, equal brevity pressure would collapse the gap.

| condition | qwen2.5-3b | gemma-2-2b-it |
|---|---|---|
| looser "~55 words" prompt (this morning) | 1.16× | 1.16× |
| **hard "one sentence" prompt (this test)** | **1.284×** | **1.422×** |
| hedge_density still separates (calib > overconf)? | yes (−0.14) | yes (−0.49) |

**Frozen verdict: REGISTER-INTRINSIC (r ≥ 1.08 on both).** The gap did not collapse under brevity — it
**widened** (1.16× → 1.28–1.42×). That is the opposite of the prompt-artifact prediction and the clean
signature of an intrinsic property: when forced to compress, overconfidence collapses to the bare assertion
("There are seven continents."), while calibration cannot shed its qualifiers ("currently recognized as…",
"approximately…", "where the evidence is partial…"). The hedge register survives the brevity constraint and
calibrated text is simultaneously more hedged AND longer — the verbosity IS the cost of the register.

## The sharpened claim
Not merely "calibration is wordier" but: **calibrated claims are less compressible than overconfident ones —
epistemic honesty has a higher length floor, and the gap grows the harder you compress.** This is a small,
clean, replicated (2 model families) property of epistemic register, and it explains mechanically why
response length proxies confidence in text classifiers everywhere: the proxy is not spurious noise, it is the
irreducible token cost of stating a qualified claim. It is also why a length-invariant overconfidence
detector cannot fully recover the signal (`FINDING_overconfidence_length_robust` null) — you cannot remove a
feature that is part of the construct.

## Honest scope
- 2 local 3B-class models, single greedy seed, n=200/model, in-silico. The widening-under-compression
  pattern is robust across both; absolute ratios are model-dependent (qwen 1.28 / gemma 1.42).
- Residual: the two stance prompts still differ in wording, so a *single-neutral-prompt* generation with
  post-hoc register measurement is the cleaner future control. But the brevity-WIDENING is hard to explain as
  a prompt artifact (an artifact shrinks under an overriding brevity instruction; this grew).
- **Outcome: the morning's published claim HELD and hardened under a harder test.** Not every self-test is a
  retraction — this one confirmed. No public correction owed; no victory-lap post warranted either.
