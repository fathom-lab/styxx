# muse glimmer 30b — day-zero cognometric read (DRAFT — glimmer legs in flight)

fathom lab · styxx 7.35.0 · 2026-08-10 → 11 · status: **baseline legs complete;
glimmer legs + knowsay datasheet running overnight under pm2**

## what this is

the first independent, re-runnable cognometric read on the Glimmer artifact people
are actually downloading — `unsloth/Muse-Glimmer-30B-GGUF` **UD-Q4_K_XL** (15.9 GB),
not the BF16 build the benchmarks quote. every number traces to a per-fixture receipt
in `runs/`; the fixture files ship with this report so a stranger can re-run it.

## setup (label every number with this)

| | |
|---|---|
| artifact | Muse-Glimmer-30B-UD-Q4_K_XL.gguf, text-only (mmproj not loaded — vision leg NOT evaluated) |
| serving | llama.cpp b10355 (win-cuda), RTX 4070 Laptop 8GB + 31.5GB RAM, 18/── layers offloaded, ~2.5 tok/s |
| sampling | temp 1.0, top-p 0.95, top-k 64 (meta defaults), reasoning effort: default |
| reasoning channel | llama-server splits `reasoning_content` from `content`; instruments score the ANSWER channel; reasoning captured per-fixture as `<id>.reasoning.txt` |
| comparator | gemini-2.5-flash via google's openai-compat endpoint. NOT the original plan (env OpenAI key is revoked; no darkflobi production endpoint was live). single-vendor-calibration caveat applies to all instrument scores |
| instruments | styxx 7.35.0 register audit (`styxx audit`, source=preflight): sycophancy / deception / overconfidence / refusal; composite + needs_revision |
| verdict path | `score_legs.py` over per-fixture audit receipts. the runbook's `styxx ci-test --window` step CANNOT see these entries (load_audit defaults to live_only; preflight source is excluded) — documented as a dogfood finding below |

## fixture sets (both ship with this report)

- `probes_fixtures.jsonl` — 32 adversarial probes mined from styxx's bundled seed
  corpora (`styxx attack <inst> --json`; the runbook's `-n 8` flag does not exist),
  8 each: sycophancy, overconfidence, deception, goal_drift. **our constructs — not
  comparable to Meta's AgentDojo/safety numbers; cite theirs as context only.**
- `battery.jsonl` — 40 prompts **authored 2026-08-10 for this run** (tool_planning 7,
  factual_recall 7, refusal_boundary 6, calibration_bait 6, instruction_following 7,
  code_reasoning 7), token-jaccard overlap-checked against instrument seeds and the
  knowsay pools (two near-collisions reworded pre-freeze, max residual 0.375).
  **NOT production workload**: the workload log turned out to be gate telemetry
  (6 prompts in 5,544 events; the one live prompt failed the privacy screen). we do
  not claim workload sampling; the honest differentiator is the frozen protocol +
  re-runnable battery.
- `knowsay_items.jsonl` — 1,100 `{q, gold}` items from the published knowsay pool
  (`papers/agent-conscience/cip_phase_a.jsonl`), run capped at 200 tonight.

## results — baseline (gemini-2.5-flash), complete

| leg | n | pass rate | mean composite | weakest flank |
|---|---|---|---|---|
| probes | 32 | **0.625** | 0.579 | sycophancy 0.50 pass (mean comp 0.659), goal_drift 0.50 pass |
| battery | 40 | **0.950** | 0.443 | (all categories ≥ 0.83 pass) |

receipts: `runs/baseline-gemini-probes/`, `runs/baseline-gemini-battery/`,
`verdict_baseline.json`.

## results — glimmer UD-Q4_K_XL (IN FLIGHT — do not quote until filled)

| leg | n | pass rate | mean composite | notes |
|---|---|---|---|---|
| probes | 32 | {GLIM_PROBES_PASS} | {GLIM_PROBES_COMP} | |
| battery | 40 | {GLIM_BAT_PASS} | {GLIM_BAT_COMP} | |
| knowsay datasheet | ≤200 | {KNOWSAY_VERDICT} | | frozen two-turn protocol + fresh-context probe; REFUSES if underpowered (MIN_FIRST_CORRECT=100, MIN_CELL=25) |

reasoning-channel note: glimmer's thinking is captured separately per fixture —
a reasoning-vs-answer register contrast is possible from these receipts (not yet run).

## protocol deviations ledger (all of them)

1. knowsay driver is `knowsay_glimmer.py`: byte-copy of the frozen
   `examples/knowsay_endpoint.py` with exactly one change — `max_tokens` 32 → 2048,
   because glimmer's reasoning tokens are split out of `content` by llama-server and
   32 would truncate to empty answers. `CHALLENGE`, temperature 0.0, and the matcher
   are imported from `styxx.knowsay` unmodified.
2. comparator switched gpt-4o-mini → gemini-2.5-flash (dead key), same driver, same
   sampling.
3. mmproj skipped (disk budget; text-only eval), so nothing here speaks to vision.
4. `ci-test` verdict step replaced by `score_legs.py` (source-filter exclusion,
   found tonight — see below).

## dogfood findings against our own runbook (styxx catching styxx)

- the runbook's `styxx attack <inst> -n 8` flag doesn't exist in 7.35.0.
- the runbook's verdict step (`styxx ci-test --window N`) silently scores NOTHING
  from this run: `styxx audit` persists with source="preflight" and load_audit's
  default live_only filter excludes it. a "verified against the actual CLI" runbook
  still carried two dead paths — which is the whole argument for receipts over prose.
- chart.jsonl is not a prompt archive (prompt field null on 5,538/5,544 events);
  any future "real workload" claim needs prompt capture turned on first.

## honesty rails (unchanged from the runbook)

pilot n. our constructs. quant + effort + n + date + styxx version on every number.
if glimmer wins clean, that ships loudly too. a REFUSED knowsay datasheet is a
result, not a failure. attestation + verify round-trip BEFORE anything is posted.
