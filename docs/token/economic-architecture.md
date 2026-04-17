# $STYXX Economic Architecture

> Design document. This describes intended mechanics. On-chain implementation of anything
> beyond the fixed supply itself is staged — see 30/60/90 roadmap at bottom.

## 1. Supply

- **Fixed supply:** 1,000,000,000 $STYXX (1B), set at pump.fun launch.
- **Mint authority:** revoked (pump.fun standard bonding curve).
- **Freeze authority:** revoked.
- **Distribution at launch:** 100% via bonding curve — no team allocation, no insider
  pre-mint, no vesting cliffs. This is a constraint, not a feature: pump.fun gives us
  fair launch and denies us a treasury. We work around it.

### Reference sizing (at $78K FDV, ~$190K ATH FDV)

| Holding | % supply | Cost @ $78K FDV | Cost @ $190K FDV |
|--------:|---------:|----------------:|-----------------:|
|     10K |   0.001% |          ~$0.78 |           ~$1.90 |
|    100K |    0.01% |           ~$7.8 |            ~$19  |
|      1M |     0.1% |            ~$78 |           ~$190  |
|     10M |     1.0% |           ~$780 |          ~$1,900 |
|    100M |      10% |          ~$7.8K |           ~$19K  |

Tiers are priced against *current* reality. If price 10x's, tiers are reviewed.

## 2. Demand sources (sinks)

These are why someone buys $STYXX instead of ignoring it.

1. **Validator staking.** Operators must lock $STYXX to run a validator node that co-signs
   provenance certificates. Minimum stake = 100K $STYXX (see `validator-spec.md`). Stake
   is slashable on bad attestations.
2. **Calibration reward claims.** Contributors who submit accepted trajectory data receive
   $STYXX from the reward pool. This drives *distribution*, not direct demand — but it
   creates a reason for researchers to interact with the token.
3. **Premium verify.** Hosted `verify()` endpoints above free-tier rate limits require
   holding (not spending) a minimum balance. Balance-gated, not burn-gated: users aren't
   penalized, they're filtered.
4. **Benchmark submission fees.** Entry into the public Styxx benchmark leaderboard costs
   a small $STYXX deposit, refunded on honest submission, forfeit on detected gaming.
5. **Governance weight.** Holders above 1M $STYXX get weighted votes on non-critical
   parameters (atlas rotation, benchmark weights). Critical code changes remain dev-gated.
6. **Gated access.** Atlas early-access, supporter listings, priority support queue.

## 3. Supply sources

Because supply is fixed and 100% was distributed at launch, ongoing emissions come from
**holdings the protocol itself accumulates**, not from minting:

- **Benchmark forfeit pool.** Deposits from gaming attempts flow to the reward pool.
- **Slashed validator stake.** 50% burned (sent to a provably-inaccessible address),
  50% to the reward pool.
- **Donated float.** Any $STYXX the team or community buys with protocol revenue (USD
  from enterprise licenses) flows into the reward pool.

We do **not** mint. We cannot mint. Don't promise emissions we can't deliver.

## 4. Treasury structure

Because pump.fun launches disallow a pre-mint, there is **no protocol treasury at T=0**.
Treasury accumulates over time from:

- Protocol revenue (enterprise licenses, SaaS verify) → market-bought into $STYXX → routed
  to a multisig "reward pool wallet."
- Benchmark forfeit and slash flows.

**Reward pool wallet:** Solana multisig (Squads v4 or equivalent). Signers: core contributors
only at launch. Signer rotation moves to holder vote once the governance surface ships.

Treasury transparency: monthly on-chain snapshot published to `darkflobi.com/styxx/treasury`
(not live yet — 60-day target).

## 5. Emission schedule

**None.** Supply is fixed. Distribution from reward pools is demand-paced, not time-paced:

- Calibration rewards: paid per accepted submission, capped at monthly quota.
- Validator rewards: paid per attestation epoch, funded from reward pool, not from mint.
- If reward pool empties, rewards pause until replenished. No IOUs.

## 6. Why this doesn't collapse

The worry with any fixed-supply utility token is that once it's distributed, there's no
recurring demand. Our answer:

- Demand is **hold-gated** (tier unlocks) not **spend-gated** → buyers don't face
  a "spend your bag" disincentive.
- Validator stake creates a **continuous withdrawal** from float (~scale with network
  usage).
- Slashing creates a **deflationary pressure** (50% of slash burned).
- Reward pool is **bounded by protocol revenue**, not by emissions — meaning rewards
  scale with actual usage, not inflation.

## 7. 30 / 60 / 90 day roadmap

**30 days — shippable now:**
- Read-only tier lookup (`styxx token tier <wallet>`). ✅ in-progress.
- Documented utility tiers with off-chain honor-system gating on hosted endpoints.
- Reward pool multisig wallet created, documented, zero balance.

**60 days — design + testnet:**
- Validator attestation schema, signed with holder keypair, aggregated off-chain.
- Calibration reward claim flow (off-chain signed claim, manual batch payout from multisig).
- Treasury dashboard.

**90 days — candidate:**
- On-chain slashing via custom program OR continued off-chain slashing with transparency log.
- Holder-weighted off-chain governance votes on atlas parameters (Snapshot-style).

Nothing past 90 days is promised. Anything we can't ship in 90 days we don't put in the doc.

## 8. Anti-goals

- No yield farming, no LP incentive programs, no "staking APY."
- No bridged wrapped $STYXX on other chains until there is a concrete reason.
- No airdrops to unrelated communities.
- No token-gated access to the **open-source library**. The OSS code stays free.
