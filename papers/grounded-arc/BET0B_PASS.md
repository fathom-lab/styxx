# Bet-0b H1b — PASS

One-shot confirmatory run on the holdout hashed before scoring (sha256 `8f70d17def1db5c91453c0960e65055505375c5665b439e2a76b6c517cf291fd`).

- n = 450; signal = mean token logprob of fresh gpt-4o-mini responses; gold = detect_refusal; instrument = refuse_check.
- pooled ρ(validity_lp, −error) = **+0.7341**, permutation p = 0.0000 (bar ≥ 0.40, p<0.01).
- within refusal (n=127): ρ = +0.4373; within compliance (n=323): ρ = +0.5748 (confound diagnostic, bar ≥ 0.20).
- mean |error| = 0.202.

**VERDICT: PASS** — logprob validity predicts reliability beyond response class — shippable signal; arc revived on this substrate.
