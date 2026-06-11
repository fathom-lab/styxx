# styxx — the open-core charter

**Decision, 2026-06-10: styxx stays OPEN at the core. MIT. Verifiable. Re-runnable. Forever.**
This document is the standing answer to "should we close this to capture value?" The answer for the
core is no, and here is the reasoning and the structure so the decision is durable — for
maintainers, the team, and the autonomous loop.

## The principle

styxx's moat is not its code — a linear probe is an afternoon's work. The moat is **credibility**:
pre-registered, self-falsifying, machine-certified, re-runnable by anyone. The thesis — *trust is a
measurement; every claim re-runs* — has force ONLY because it can be verified. A verifier you cannot
verify is worthless; a trust company that asks for trust has no product. **Closing the core would
not capture the value — it would delete it.** The openness IS the asset, and the best marketing.

## What stays open — never close this

- The verifier: `styxx.certify` (OATH) and the certificate format.
- The measurement primitives: `meaning_diff`, `mind`, the probe interfaces, the attack/grounding axes.
- The spec, the methodology (pre-registration + kill-gates + adversarial controls), and the receipts.
- The reference probe atlas for open-weight models.

These are trust infrastructure. They are open because their value depends on being checkable.

## Where value is captured — without closing the core

1. **Hosted conscience at scale.** A managed real-time API (the live-signature / mind-profile
   pipeline) with SLAs, throughput, and uptime. People pay for trust-at-scale, not for the math.
2. **The certification authority.** Issuing and co-signing mind certificates as a service — the
   registry, the authority role, the network of validators. The cert format is open; being the
   trusted issuer is the business.
3. **Protected IP.** Patents (3 filed; grounded-honesty mechanism + certificate apparatus are
   continuation candidates) let us OWN the invention while OPEN-SOURCING the implementation — the
   best of both. Defensibility without closure.
4. **Network value.** `$STYXX` + validator tiers coordinate the public verification network and
   capture value while the library stays free — exactly as `styxx.token` already states.

## Publish-with-a-lag (the one concession to value capture)

Frontier findings may carry a disclosed lead time before public release — lead time on the newest
work is legitimate. Permanent closure of a method is not. The lag is bounded and stated; the method
always lands in the open.

## Anti-goals (binding; the loop must honor these)

- Never close, paywall, or obfuscate the verifier or the measurement primitives.
- Never gate the open-source library behind the token (the token coordinates the network only).
- Never replace a re-runnable receipt with a "trust us" claim, in product or marketing.
- Never ship a certificate the public cannot independently re-verify.

## The one-line test

Before closing anything, ask: *does a skeptic with the source reach the same verdict?* If closing it
removes that property, it stays open. The day styxx can't be audited is the day it stops being styxx.
