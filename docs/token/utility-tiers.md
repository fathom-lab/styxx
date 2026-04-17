# $STYXX Utility Tiers

Tiers are **hold-based**, not spend-based. Balance check is a point-in-time read of the
holder's $STYXX token account via public Solana RPC. You don't burn or lock tokens to
access a tier — you just hold the balance.

Balance is re-checked per-session (hosted endpoints) or per-epoch (validator attestations).

## Tier table

| Tier | Name          | Min $STYXX | % supply | Cost @ $78K FDV | Unlocks |
|-----:|---------------|-----------:|---------:|----------------:|---------|
|    0 | Public        |          0 |       0% |              $0 | `styxx.verify()` OSS library, hosted free tier (rate-limited) |
|    1 | Supporter     |     10,000 |   0.001% |           ~$0.78| Priority verify queue, supporter listing, Discord supporter role |
|    2 | Validator     |    100,000 |    0.01% |           ~$7.80| Validator node eligibility, co-sign provenance certs, attestation rewards |
|    3 | Governor      |  1,000,000 |     0.1% |            ~$78 | Weighted vote on atlas/benchmark parameters, proposal submission |
|    4 | Core          | 10,000,000 |     1.0% |           ~$780 | Veto-capable on governance, dedicated support, co-maintainer status eligibility |

> Prices are nominal and drift with market. Tiers are denominated in **$STYXX units**,
> not USD. If FDV changes, dollar costs change; tier thresholds don't.

## What each tier delivers — concretely

### Tier 0 — Public (0 $STYXX)
- Full `styxx` Python library (MIT).
- Hosted `verify()` API: 100 req/day, no auth needed.
- Access to public atlas snapshots.
- Read-only benchmark browsing.

### Tier 1 — Supporter (10K $STYXX, ~$0.78)
- Priority queue on hosted verify (10× rate limit vs public).
- Your wallet (or chosen handle) appears on the public supporters page.
- Supporter role in Discord / Telegram.
- Early-read access to atlas changelogs (24h before public push).

### Tier 2 — Validator (100K $STYXX, ~$7.80)
- Eligible to register a validator node (see `validator-spec.md`).
- Node runs `styxx.verify()` against the live feed and emits signed attestations.
- Earn share of calibration reward pool proportional to attestations * stake.
- Co-signature on issued provenance certs — your key appears on certs you attested.
- Stake is **slashable** for bad attestations (50% burn / 50% reward pool).

### Tier 3 — Governor (1M $STYXX, ~$78)
- Weighted vote on:
  - Atlas rotation cadence and composition.
  - Benchmark weight changes.
  - Reward pool allocation ratios.
  - Validator minimum stake adjustments.
- Can submit proposals (requires 1M minimum and a 7-day notice).
- **Cannot vote on:** core protocol code, safety-critical thresholds, or treasury
  withdrawals above a small discretionary limit. Those remain maintainer-gated.

### Tier 4 — Core (10M $STYXX, ~$780)
- Governance veto on non-critical proposals (requires ≥2 Core holders to concur).
- Dedicated response channel for production issues.
- Eligible for co-maintainer nomination (still requires technical contribution; the
  tier is necessary, not sufficient).
- Named in project `CONTRIBUTORS.md` if desired.

## Anti-abuse

- **Sybil resistance:** tier checks are per-wallet, not per-person. We don't care if you
  split 10M across 100 wallets — each wallet independently gets its tier, and none of
  the split wallets hit a higher tier than their individual balance.
- **Flash-hold defense:** validator stake requires **continuous holding** over the
  attestation epoch (sampled at random points). Supporter and Governor tiers check
  balance at request time — no lock required — because the downside of a flash-hold
  gaming those tiers is trivial.
- **Exchange / custodial wallets:** we do not attempt to detect custodial holdings.
  If your tokens are on an exchange, the exchange's hot wallet holds the balance, not
  you. Withdraw to a self-custody wallet to tier up.

## Implementation status

| Tier feature                         | Status            |
|--------------------------------------|-------------------|
| Balance-based tier lookup (read)     | ✅ shipping (this PR) |
| Hosted verify rate-limit gating      | ⏳ 30-day         |
| Supporter listing page               | ⏳ 30-day         |
| Validator registration + attestation | 🗓 60-day         |
| Governance voting surface            | 🗓 90-day         |
| Slashing (on-chain)                  | 🗓 90-day candidate |
