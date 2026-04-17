# $STYXX Token — Overview

$STYXX is the utility token for the Styxx provenance and calibration network. It exists to
coordinate a decentralized validator set, compensate contributors who submit high-quality
trajectory calibration data, and gate premium / governance surfaces of the protocol.

**Contract (Solana):** `Dxw3u4KxN32KpSdHSq4TkwjfMPJTPeosa22JXN15pump`
**Launch venue:** pump.fun (bonded)
**Supply:** fixed, 1,000,000,000 $STYXX (pump.fun standard, no mint authority)

## What it is not

- Not a security. Not an investment contract. Not a promise of yield.
- Not required to use `styxx.verify()`. The core library is and will remain open source.
- Not a governance token for anything that doesn't yet exist. We don't ship DAOs on vibes.

## What it does, today

- **Rate-limit bypass** on hosted verify endpoints above the free tier.
- **Validator eligibility**: nodes that co-sign provenance certificates must stake $STYXX.
- **Calibration reward pool**: contributors who submit reproducible trajectory data against
  the Styxx benchmark earn from a reward pool denominated in $STYXX.
- **Gated surfaces**: supporter listings, priority verify queues, early access to Atlas
  updates.

## What it is designed to do (90-day horizon)

- Light-weight off-chain governance on non-critical parameters (benchmark weights, atlas
  rotation cadence).
- Slashable validator attestations — see `validator-spec.md`.

Anything beyond that lives in roadmap, not in this doc.

## Files in this folder

- [`economic-architecture.md`](economic-architecture.md) — sinks, sources, treasury, why demand.
- [`utility-tiers.md`](utility-tiers.md) — concrete holder tiers and what each unlocks.
- [`validator-spec.md`](validator-spec.md) — validator node design.
- [`calibration-rewards.md`](calibration-rewards.md) — contributor reward mechanics.

## For enterprise readers

The Styxx protocol is usable without ever touching $STYXX. Enterprise deployments can run
their own validator set, pin their own atlas, and ignore the token entirely. The token
coordinates the *public* network — it does not gate the *technology*.
