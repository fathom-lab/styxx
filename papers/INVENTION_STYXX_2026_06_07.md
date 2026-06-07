# INVENTION_STYXX — what styxx actually invented

*Fathom Lab · styxx · 2026-06-07. Derived by five independent outside vantages
(mech-interp scientist / EU-standards body / patent attorney / skeptical competitor / safety buyer),
then adversarially red-teamed by a combined prior-art + dogfood critic, and synthesized to the
irreducible core that survives every objection. The boundary is stated as loud as the claim — and
this document is written to pass styxx's own honesty audit.*

## styxx: the instrument that decides whether you can trust a model's report of itself

**The invention (one sentence).** styxx is a **validity-gated falsification instrument for
oversight-channel claims**: given any proposed way to *ask an AI about its own internal state* —
self-report, verbalized confidence, "are you being manipulated?", chain-of-thought faithfulness — it
decides whether that channel actually carries the information, by running it head-to-head against an
external read of the **same** injected state, with a **prime/abort gate** proving the channel is
physically alive, a **shuffled-symbol answer code** defeating answer-leakage, and a **hash-before-score
verdict tree** that refuses an auto-positive. Its first output is a demarcated **negative**, made
load-bearing instead of hand-waved.

## The three laws it measures (not a theory — a falsifiable apparatus)

1. **Read ≠ write.** An injected linear concept is externally probe-readable yet cannot be written back
   by a general linear intervention; the cross-model write arm fell below its own pre-registered gate
   (ALIGNER_LIMITED). Legibility is real and **read-only** — which is a safety *feature*: a read-only
   conscience cannot be turned into a write-time jailbreak.
2. **The confident-misconception wall is internal — bedrock on all three axes.** When a model believes a
   falsehood the truth is not represented to be read: the static gold readout is at chance (AUC 0.519),
   output confidence is at chance (~0.47–0.53), **and** a fully pre-registered process/dynamics attack
   is at chance too (4-feature composite AUC 0.387, permutation p 0.78; the headline "crystallization
   depth" feature 0.503), with off-floor (0.43) and clean-label (0.41) diagnostics confirming the null
   is real. No oversight channel — output, static white-box, or dynamical white-box — recovers a truth
   that is not there. This bounds the whole field, **styxx included.**
3. **Self-report and external read dissociate for content-identification — scoped.** On a
   *steering-validated* injection (Qwen-1.5B, AUROC 0.844), forced-choice self-report of the planted
   concept is at chance while an external probe identifies it, and the prime/abort gate proves the
   self-report channel is alive — converting "the model didn't say it" into "the model *could not
   access* it." This is content-**identification**, not presence-**detection** (strong-dose YES/NO ties
   ~0.99), and it is the controlled small-open-model counterpoint to Anthropic's frontier-positive
   introspection result — **not** a new universal law.

## The instrument

Inject a steering-validated direction; read the same state two ways (the model's forced choice vs an
external probe on a clean downstream position). The probe and the injection are prior art
(Alain-Bengio / RepE / Apollo) — **not claimed**. The two genuinely new moves are the **prime/abort
apparatus-validity gate** (a self-report null can no longer be dismissed as a dead channel or RLHF
persona) and the **per-trial-shuffled symbol code** (defeats the logit-lens). SHA-256 hash before
scoring; the frozen verdict tree (UNINFORMATIVE_APPARATUS / REPORT_AS_LANDED_null / SURVIVED) is
demonstrably obeyed even when it kills the headline.

## The moat

**Discipline-as-instrument.** Anyone clones a linear probe in a week; no one retroactively manufactures
a public git history of *losing to its own kill-gates* — dogfooding the audit on its own maker and
obeying it (overconfidence flagged 0.922 → public claim forced down), and reporting five-plus
closed-negatives as loudly as the wins. The field's incentive is the capability headline ("models CAN
introspect"); a controlled **negative** that survives its own abort gate is structurally un-tempting to
copy and easy to get wrong. The compounding second layer is a **CI-enforced disclosure schema**
(calibrated_metric + construct_ceiling + receipt_commit + named-uncovered, with kill-gates that fail the
build on a hidden limit) — a *format* that, once a standard cites it, makes any competitor submission
omitting its boundary mechanically non-conformant. The **patents are the weakest layer** and must not be
relied on: the filed provisionals cover the older K/C/D axes and the Fathom Constant, not this
dissociation, which is already public and racing on a grace-period clock.

## The regulatory wedge

EU AI Act **Article 14** human oversight is operationalized by asking the model about itself, with no
harmonised standard and an Aug-2-2026 enforcement pull, and Art.15(2) explicitly invites stakeholder
measurement methodologies. styxx is the falsifiable existence-proof that the self-report-only oversight
foundation is structurally blind for content-identification, plus the read-only external substitute.
**Honest caveat:** the oversight *finding's* technical core (linear probes beating self-report) is
already public (Apollo, arXiv 2507.23221), so the defensible asset is the **disclosure-schema format**
sold as the Art.15(2) methodology primitive — *not* a claim that the slot is empty, and *not* a mandated
detector. The white-box readout is offered only as a complementary existence-proof, never a replacement
for process controls (stop-button, human-in-the-loop).

## What to file (patent reality check)

Do **not** file on the dissociation finding (publicly disclosed; collides with Anthropic's introspection
method and Apollo's deception probes). The defensible independent claim is a **runtime control method**
that *consumes* the finding: tap a pre-emission activation → linear probe calibrated to materially beat
the model's own self-report → behavioral validity by sampling-divergence *in lieu of* stated confidence
→ regime-routed channel selection → emit / suppress / regenerate / escalate, **with a defined ABSTAIN
when probe + validity jointly indicate the content is not internally represented** (the
confident-misconception floor). Non-obviousness rests on **teaching-away**: styxx's own closed-negatives
prove the field's default (ask the model / trust confidence) fails in exactly these regimes, so routing
*away* from self-report and confidence precisely where they fail is counter-intuitive. Open-weight
substrate only (white-box claims are impracticable against closed-API deployments).

## The boundary, stated as loud as the claim

No "1.00 across three families" (the load-bearing dissociation is one *steering-validated* model;
cross-family is a probe-read replication; the probe itself is at chance 0.519 on confident error).
Content-identification and subtle-dose only; for "is something off" at deployment dose a YES/NO
self-report ties (~0.99). Read-**only** — no repair. The shipped product primitives (`grounded_honesty`
AUC 0.966; `detect_context_injection`) are self-**consistency** meters of the semantic-entropy family
that invert in-session (G2 ~0.011) and collapse on shared belief — **not** lie or manipulation
detectors. The pre-output action gate is an unbuilt pre-registered bet with direct 2026 prior art.
`spec_exec` is a one-task held-out demo, not a control plane. It measures **behaviour and
representation, never minds.** The marquee dissociation is the **least** patentable part.

**The boundary is the product:** styxx is the falsifiable measurement-and-demarcation instrument that
turns "the model is blind to its own X" into a *refusable test* rather than a slogan — and it would pass
its own dogfood precisely because it claims nothing the apparatus did not survive.

## What to build next

1. **Lock the demarcation instrument as the publishable product** — run the prime/abort-gated,
   symbol-coded, *per-model steering-validated* dissociation on the multi-model council (the owed arm),
   and publish the cross-model **map** of where self-report carries content and where it does not, with
   the confident-misconception floor (now bedrock on all three axes) as the headline boundary.
2. **Ship the CI-enforced disclosure schema as a standalone standards primitive** (`styxx.compliance`,
   kill-gates wired into the suite) and put *that* — boundary-as-format, not the oversight finding — in
   front of CEN-CENELEC JTC 21 / the AI Office.
3. **File the narrow runtime control method** (tap → probe → divergence → gate → emit, regime-routed,
   abstain-on-confident-misconception, open-weight substrate), built on the closed-negatives as
   documented teaching-away — and explicitly drop any patent framing of the finding itself.
