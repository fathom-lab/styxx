# Styxx Validator Specification

> **Non-consensus validator.** There is no Styxx L1. Validators do **not** produce blocks.
> They independently run `styxx.verify()` on a shared feed of verification requests and
> emit signed attestations. Aggregation and dispute resolution happen off-chain, with
> slashing settled against staked $STYXX.

## 1. Role

A validator's job is to **independently reproduce** verification of provenance certificates
and attest to the result. The network runs several validators concurrently; a cert is
considered *network-attested* when ≥ N of M registered validators sign the same result.

## 2. Requirements

- **Stake:** ≥ 100,000 $STYXX (Tier 2), held in the validator's registered wallet.
- **Uptime:** ≥ 90% over a rolling 7-day window (measured by heartbeat pings).
- **Reproducibility:** validator must run a pinned version of `styxx` and a pinned atlas
  snapshot. Version drift is detectable via the attestation payload hash.
- **Solana keypair:** used only for signing attestations (ed25519). Never holds tokens
  directly — the *stake* wallet is separate, referenced by address.

## 3. Architecture

```
   +-----------------------+            +------------------------+
   |  Verification Feed    |  ----->    |  Validator Node N      |
   |  (public queue of     |            |   - styxx.verify()     |
   |   cert requests)      |            |   - sign attestation   |
   +-----------------------+            +------------------------+
              |                                     |
              |                                     v
              |                          +------------------------+
              |                          |   Attestation Log      |
              +------------------------->|   (append-only,        |
                                         |    publicly auditable) |
                                         +------------------------+
                                                     |
                                                     v
                                         +------------------------+
                                         |   Aggregator           |
                                         |   - ≥N of M agree?     |
                                         |   - dispute detection  |
                                         +------------------------+
```

The aggregator is initially a single publicly-operated service run by the core team. It
emits **network-attested certs** and publishes discrepancies.

## 4. Attestation payload

Each attestation is a signed JSON object:

```json
{
  "schema": "styxx.attestation/v1",
  "request_id": "sha256(...)",
  "cert_hash": "sha256(...)",
  "styxx_version": "3.3.1",
  "atlas_snapshot": "atlas-2026-04-14",
  "verdict": "pass" | "fail" | "abstain",
  "metrics": { "d_axis": 0.535, "auc": 0.663, ... },
  "validator_pubkey": "<ed25519 pubkey base58>",
  "stake_wallet": "<solana address>",
  "timestamp": "2026-04-17T10:00:00Z",
  "signature": "<ed25519 signature base58>"
}
```

The signature covers the SHA-256 of the canonical JSON (keys sorted, no whitespace) of
all fields except `signature` itself.

## 5. Registration flow

1. Operator acquires ≥100K $STYXX in a self-custody wallet.
2. Operator generates an ed25519 attestation keypair (separate from stake wallet).
3. Operator submits a registration to the validator registry:
   - `stake_wallet` (address)
   - `attestation_pubkey`
   - `endpoint` (https URL for heartbeat + feed pull)
   - `operator_contact` (email / handle)
   - A signed challenge proving control of both keys.
4. Registry verifies stake balance via public Solana RPC, adds validator to active set.
5. Balance re-checked each epoch. If stake drops below 100K, validator is moved to
   `inactive` and stops receiving rewards; attestations past that point are discarded.

## 6. Rewards

- Paid per epoch (proposed: 24h).
- Per-validator reward = `base_share * reputation * uptime_factor`.
- `base_share` = epoch reward pool / active validator count.
- `reputation` = exponentially-weighted moving average of attestation agreement rate.
- Funded from the calibration reward pool (see `calibration-rewards.md`).
- **Not funded by mint.** If pool is empty, rewards pause.

## 7. Slashing

Conditions:

1. **Equivocation:** validator signs two conflicting attestations for the same `request_id`.
   → Full stake slash (100%).
2. **Demonstrable bad attestation:** a cert the validator signed is later shown (via
   re-run at pinned version + atlas) to have been produced by a non-matching input.
   → 25% stake slash per incident, cumulative up to 100%.
3. **Sustained disagreement:** validator's attestation disagreement rate vs majority
   exceeds 20% over an epoch.
   → 5% stake slash + temporary suspension pending review.

**Slash settlement (pre-90-day, off-chain):**
- Slash is *scored* off-chain and published to a transparency log.
- Reward distributions to the offending validator are zeroed until stake is topped back
  up to minimum **plus** slash amount.
- The slashed portion is tracked as a debt against the validator's reward stream.

**Slash settlement (post-90-day, candidate):**
- Custom Solana program holds validator stake in an escrow PDA.
- Slash is executed on-chain: 50% burn (send to `1nc1nerator11111111111111111111111111111111`),
  50% to reward pool multisig.

## 8. Dispute resolution

- Any holder ≥ 1M $STYXX (Tier 3) can file a dispute against an attestation within 7 days.
- Core maintainers + 3 randomly-selected Tier 3 holders re-run `verify()` at the pinned
  version/atlas. Majority result decides.
- Frivolous disputes: 10K $STYXX deposit forfeit on >3 consecutive rejected disputes from
  the same filer.

## 9. Reference implementation hooks

- `styxx/token.py :: get_tier(wallet) -> Tier` — used at registration.
- `styxx/token.py :: verify_holder_signature(pubkey, message, signature)` — used for
  attestation verification.
- `styxx/verify.py` — the actual verification logic; unchanged by this spec.
- `styxx.attestation` (future module) — sign / verify / canonicalize attestation payloads.

## 10. Non-goals

- No L1, no P2P gossip, no Byzantine fault tolerance protocol. The aggregator is a trusted
  service at launch; decentralization of the aggregator is a separate, later design.
- No smart-contract validator in the MEV sense. This isn't that.
- No delegation / liquid staking at launch. Operators run their own stake.
