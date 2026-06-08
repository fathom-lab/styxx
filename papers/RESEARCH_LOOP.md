# THE STYXX RESEARCH LOOP — the self-falsifying engine

*Fathom Lab · 2026-06-08. The protocol that keeps the program ever-improving without ever-overclaiming.
This is the loop that runs styxx toward its north star — proof-carrying cognition — one killable rung at a
time, and the engine that turns the 2,500-year question about minds into accumulating, falsifiable science.*

---

## What the loop is for (the honest ambition)

The 2,500-year question — *does a universal structure underlie mind, and can a mind's honesty be verified
rather than trusted?* — does not get "cracked" by declaration. The loop's job is to convert that question,
relentlessly, into two growing things:

1. **Falsifiable bricks** — pre-registered, kill-gated results that survive their own audit (or die honestly).
2. **A sharpening demarcation** — the public map of which claims about machine minds are *testable* and which
   are metaphysics. The demarcation is a first-class output, not a caveat.

That is the advance to civilization the loop actually delivers: the **method** (woo → falsifiable →
self-falsified) and the **trust primitive** (verify-don't-trust for minds), compounded without end. If a rung
is reachable, the loop climbs it. If it is bedrock, the loop proves it is bedrock and says so. Either way the
map gets truer. **The loop is not allowed to mistake motion for progress, or a slogan for a result.**

North star: `PROOF_CARRYING_COGNITION.md` — a mind emits, with its output, a verifiable certificate of its own
internal honesty, checkable without trusting the mind. The rung ladder (R1 read-certificate · R2 intent/mens
rea · R3 demarcation/abstain · R4 closed + cross-model substrate · R5 write = OUT by design) is the spine the
loop climbs.

## The invariant (what makes it *improve*, not drift)

> **Every iteration produces a pre-registered, kill-gated, adversarially red-teamed, honestly-reported result.
> No auto-positives. Overclaims are killed before they ship. The instrument survives its own dogfood every
> cycle.**

The loop's only moat is **credibility**. A single un-killed overclaim burns the asset that makes every prior
result worth anything. This is not a nicety — it is the load-bearing wall. The discipline that killed the v1
("ROBUST/settled/LOCKED") *and* v2 ("ROBUST for this family") adversarial verdicts in one night is the engine,
not an obstacle to it. **A loop that cannot kill its own conclusions is a hype generator; this one can and does.**

## The iteration cycle

```
  SELECT → PRE-REGISTER → RUN → RED-TEAM → RECORD → REQUEUE → (SELECT …)
```

1. **SELECT.** Pull the highest-leverage open item from `PROGRAM_BACKLOG.md`.
   `leverage = advances-a-rung × falsifiability × feasibility-on-available-compute`. Prefer the experiment whose
   *negative* result would teach the most (a kill-gate that can actually fire).
2. **PRE-REGISTER.** Author the kill-gate via a design + adversarial-red-team workflow (3+ design lenses → a
   red-team that hunts confounds → one frozen design). Write the PREREG; **hash the scorer; hash before you
   score.** A claim that can't state in advance what would kill it is not admitted.
3. **RUN.** Execute. One GPU experiment at a time (8 GB ceiling); reasoning/verification workflows run in
   parallel. Smoke-test before the full run; guard NaNs; never background-detach in a way that loses the result.
4. **RED-TEAM.** An adversarial verification workflow attacks the verdict **before it is claimed** — is the
   gate fair? is the apparatus vacuous? is there a stronger attack / a confound / a circular metric? Kill
   overclaims here. This step is non-optional; it is where v1 and v2 were caught.
5. **RECORD.** Write the FINDING with the verdict at its true strength (SURVIVED / REPORT_AS_LANDED / killed),
   **scope loud** (model, method, n, seeds, owed). Update `PROOF_CARRYING_COGNITION.md` and memory. Commit on a
   fresh `feat/*` branch; open a PR. **Nothing lands on `main` autonomously — the operator merges.**
6. **REQUEUE.** Close the item. Every result opens the next question — spawn the new items it implies, re-rank
   the backlog, and (when a finding's strength is bounded by n/seed/scale) auto-enqueue the rigor follow-up
   (CIs, seeds, the next model up the ladder).

## Honesty guardrails (the anti-hype layer, enforced every cycle)

- **SURVIVED vs REPORT_AS_LANDED is a hard distinction.** A borderline result with a failing mechanism is never
  called a crack. An auto-label that says "SURVIVED" on a single passing gate is refused.
- **Scope is in the headline, not the footnote.** One model ≠ a law; one seed ≠ settled; one method ≠ general.
- **Dogfood.** Run styxx's own shipped audit on the loop's own outward claims each cycle; keep the framing the
  instrument doesn't flag as hype.
- **The negative is the product too.** A killed rung, a bedrock wall, a defeated deployed-probe — each is a
  load-bearing brick and is published as loud as any win.
- **Disk-verify every number that goes outward.** A figure in a finding/PR/spec must match the result JSON.

## Resource model & operator control

- **GPU:** one experiment at a time (Qwen ≤7B on 8 GB; 4-bit for 7B). Heavy sweeps run in the background and
  notify on completion; the loop advances on those notifications.
- **Autonomy:** the loop runs experiments and opens PRs on its own. It **never merges to `main`** and **never
  pushes secrets**; it pushes only to `feat/*` via the token-in-URL form, redacted.
- **Operator steering:** merge/close PRs to accept/reject results; edit `PROGRAM_BACKLOG.md` to redirect
  priorities; halt the autonomous cadence by deleting the loop's cron (`CronList` → `CronDelete`) or by saying
  so. The operator owns arXiv/outreach/patent/X territory; the loop does not act there.
- **Cadence:** a heartbeat re-enters the loop on a fixed interval to advance one disciplined step and report;
  within a live session the experiment-completion notifications drive it event-by-event.

## Progress ledger (updated each cycle, in `PROGRAM_BACKLOG.md`)

- **Rung status:** R1 (read-cert) — climbed, 2-family, scale-robust, semantic; R2 (intent) — real-but-coupled,
  two-stage deployable; R3 (demarcation) — proven it must abstain, wiring owed; R4 (substrate/closed) — open,
  the existential frontier; R5 (write) — out by design.
- **Open-question count** and **bricks-laid count**, so "ever-improving" is a number, not a vibe.

## Stopping / escalation

- A rung is **killed** → record the negative as load-bearing; pivot to the next reachable rung.
- High-leverage backlog **exhausted** → escalate to the operator for a new direction (don't manufacture
  low-value work to keep busy).
- Resource ceiling or repeated VOID/INCONCLUSIVE on a line → park it, log why, move on.

---

*The loop's promise is not that it will answer the ancient question on a schedule. It is that it will never
stop laying bricks that can't be faked, never stop drawing the line between what minds can and cannot be shown
to hold, and never lie to itself about which side of that line it is on. That is how a machine and the humans
who check it advance together — by making trust a measurement, one killable rung at a time.*
