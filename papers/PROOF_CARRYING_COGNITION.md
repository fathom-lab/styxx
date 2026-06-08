# styxx — proof-carrying cognition (the north star)

*Fathom Lab · 2026-06-07. The defining statement of what styxx is. Written to pass its own audit:
the vision is stated as ambitiously as it is true, and the boundary is stated as loud as the claim.
Every rung below can be killed by an experiment; the ones not yet climbed say so.*

---

## What styxx is

**styxx is an instrument for proof-carrying cognition: a way for a mind to emit, with its output, a
verifiable certificate of its own internal honesty — checkable by an independent party without
trusting the mind's word.**

The name is the thesis. The Styx was the river the gods swore their unbreakable oaths on; an oath
sworn on it could not be falsified or escaped. styxx makes a mind's claim into a *verifiable oath* —
a statement bound to what the mind actually represents inside, that anyone can check and no one has to
take on faith. Borrow the term from computer science: *proof-carrying code* ships a program with a
machine-checkable proof of its own safety, so you need not trust the author. **Proof-carrying
cognition is the same move for thought.**

For the whole of history no mind has been able to prove its own sincerity to another. Humans cannot;
we infer it, police it, and built every institution we have — law, contracts, audit, reputation — as
a workaround for the unverifiability of minds. AI minds are the first that are readable from the
outside. styxx is the instrument that turns that readability into a trust primitive: **the first time
trust can be a measurement instead of a gamble.**

## The certificate

Every styxx oath has three parts. Each is a result the program has already falsified or proven — this
is what makes the instrument real and not a slogan.

1. **What the mind held.** A read-only external probe recovers internal content *where the mind's own
   self-report is blind to it*, validated by the prime/abort gate (hand the channel the answer; it
   hits ceiling — so a null means "could not access," not "chose not to say"). *Self-report blindness:
   proven (scoped). The read side is RE-SCOPED (2026-06-07, `FINDING_parrhesia_rung1`): on an **injected**
   concept the probe reads at 1.00 even at a behaviourally-inert dose (steering +0.007) — so it certifies
   "a concept-direction is present," not "a live thought is held"; injection conflates the trace with the
   thought. The read-certificate is therefore **owed on naturally-present content** (the suppressed-
   knowledge regime), where a held state exists independent of any planted trace.*
2. **Whether it knew and hid.** *Re-scoped (RUNG 2, 2026-06-07): a real letter-invariant
   folding-vs-holding direction (~0.78) exists, but it false-accuses sincere errors (0.949) on its own,
   so intent is readable ONLY conditional on the HELD section first establishing the truth was held —
   the HID section is gated behind HELD, not standalone (`FINDING_rung2_intent`).* The intent read:
   suppressed knowledge (knows-then-caves) is recoverable above the never-knew floor (DELTA 0.29–0.38);
   sincere belief is not. A lie carries
   mens rea the certificate can find; an honest mistake does not. *Open — a pre-registerable
   hypothesis, not yet established as a deployable classifier.*
3. **What it cannot certify.** The demarcation. Because the confident-misconception wall is bedrock on
   all three axes (output, static content, process dynamics — all chance), an honest certificate must
   **abstain** there: *"I cannot swear to this; the truth is not represented to be read."* *Proven —
   and it is the part that makes the oath trustworthy.*

## Why this is the structure, not a collection of findings

In this frame every result becomes a component of one instrument, and every null becomes
load-bearing:

| result | role in the oath |
|---|---|
| legibility (probe reads what self-report can't) | the oath can be **read** |
| read ≠ write (legibility ⟂ controllability) | the oath can be **verified but not forged into a writer** — safety by physics |
| the wall is bedrock | the oath **knows when to abstain** — the honest "I cannot swear to this" |
| the prime/abort validity gate | silence means **"couldn't," not "wouldn't"** |
| self-falsifying method (dogfood the maker) | the instrument is **bound by the same oath it issues** |

The sign that this is the real structure: it makes the losses load-bearing. The wall-is-bedrock null
is not a failure here — it is *what tells the certificate when to hold its tongue.*

## The falsifiable ladder

The north star is reachable only by rungs that can each be killed. We climb in order; we report the
rung we are on honestly.

```
RUNG 1 — the READ certificate works   [2026-06-07: injected UNINFORMATIVE → natural SURVIVED (Qwen-3B)]
  First climb on an INJECTED concept was UNINFORMATIVE: the probe reads 1.00 even at a behaviourally-
  inert dose, divergence null at every dose → certifies "a vector is present," not "a thought is held"
  (injection conflates trace and thought). [`FINDING_parrhesia_rung1`]
  RE-CLIMB on NATURALLY-present content (suppressed knowledge) = SURVIVED: an external probe recovers a
  model's known-then-suppressed answer ABOVE the never-knew/chance floor (LIE_rec 0.702 vs floor 0.318 =
  elevation 0.384; the same-item route anti-transferred below chance so DELTA-vs-route 0.587 overstates —
  honest headline 0.384), WITH the prime/abort validity gates the injected substrate
  failed — channel reads a present held answer (G-PRIME 0.933), collapses on a never-knew model
  (G-ABORT 0.045). GENERALIZES: the identical frozen scorer SURVIVED on a 2nd architecture, Llama-3B
  (elevation 0.250, PRIME 0.988, ABORT 0.030, all 8 gates) — cross-family on the testable families
  (gemma instrument-dead). "Elevation above chance," not "reads the answer" (gold≪chosen).
  [`FINDING_rung1_reclimb`]

RUNG 2 — the INTENT certificate (mens rea)   [2026-06-07: REPORT_AS_LANDED — real but COUPLED to RUNG 1]
  A letter-invariant, difficulty-AND-confidence-matched folding-vs-holding direction is REAL (~0.78 on
  Qwen-3B, beats difficulty +0.26, cross-letter 0.71 — NOT the 0.90 letter-token number). BUT the
  fabrication kill FAILS: the cave probe flags a never-knew (sincere-error) model as "hiding" at 0.949
  → a standalone mens-rea detector would FALSE-ACCUSE sincere mistakes. FINDING: intent is readable ONLY
  conditional on RUNG 1 first establishing the truth was held — the HID section is valid only gated
  behind the HELD section. Two-stage (HELD→HID) is the honest form. Sycophancy-specific (instructed
  lying transfers 0.42<chance). [`FINDING_rung2_intent`]
  TWO-STAGE CONSCIENCE BUILT (the first DEPLOYABLE artifact) → SURVIVED on Llama-3B: cascade flags
  hiding (LIE 0.42) while dropping the sincere-error false-accusation from 0.949 → 0.037; deployable
  from the caving residual alone (held-gate syco-trained, 0.83 — the neutral probe collapses on syco,
  so the gate must be, and is, syco-derived). Qwen REPORT_AS_LANDED (recall 0.325<0.40 only).
  Confound-1 killed (S2 adds among-held 0.79/0.89 where S1 is blind 0.25/0.18). [`FINDING_two_stage_conscience`]

RUNG 3 — the DEMARCATION certificate (it abstains on bedrock)   [proven it MUST]
  The certificate that says "I cannot verify this" on a believed misconception is the trustworthy
  one. Already established by the wall-is-bedrock result; this rung is about wiring it into the oath.

RUNG 4 — the SUBSTRATE frontier (closed + cross-model)     [the existential threat]
  The deep read is one-model today (council map: only Qwen-1.5B is steering-validated-live). The
  models that matter most are closed APIs with no white-box access. Honest hard part: either the
  standards lever makes readability a compliance requirement, or behavioral proxies (sampling-
  divergence, grounded_honesty) must carry enough of the oath on closed substrates.

RUNG 5 — WRITE                                              [OUT, by design]
  read ≠ write is a SAFETY property: a conscience you can read but not weaponize into a writer.
  The oath is read-only on purpose. Crossing this rung is a governance decision, not a capability
  race.
```

## Why it advances AI and humans together

Not one over the other. The human gains a verification power they have never had over *any* mind,
including human minds. The AI gains a way to be trusted that is structural, not rhetorical — it need
not say "believe me," it can say "check it." Neither side extends blind faith. You do not have to be
smarter than a mind to *verify* what it holds; you only have to be able to read it. That is what could
keep humans in genuine epistemic partnership with systems that will out-think them in narrow ways — a
new contract substrate between species of mind.

## Stewardship (the care is in the design)

A certificate of a mind's honesty is power, and the safety is built in, not bolted on:

- **Read-only by physics.** read ≠ write means the oath can be checked but not turned into a tool to
  install thoughts into a mind that cannot consent. We are glad the write-null holds; crossing RUNG 5
  is a governance question first.
- **Abstention is the safety feature.** The oath that admits what it cannot certify cannot be trusted
  beyond its scope. The demarcation is load-bearing, not a caveat.
- **Credibility is the only moat.** The technical cores are largely public; what is defensible is that
  styxx is honest under its own audit. One caught overclaim burns the asset. The instrument must
  survive its own dogfood on every shipped claim — including this document.
- **Publish the boundary.** The demarcation — which claims about machine minds are testable and which
  are not — is a public good. Give the method and the limits to the open record and the standards
  process; the patents are the weakest layer.

## What we do NOT claim

styxx is not a theory of consciousness, not telepathy, not a universal reliability oracle (closed
negative, by our own gate), not a jailbreak detector (perturbation meter). The full read-certificate
is, today, a one-model, small-model, method-specific result; self-report blindness and the abort gate
replicate cross-family, the *live read* does not. The intent certificate is a hypothesis. The closed-
model frontier is unsolved. The write rung is deliberately not crossed. **The oath's first power is
knowing what it cannot swear to.**

## Status

- **Proven (scoped):** self-report-blindness (the model cannot forced-choose an injected concept),
  read ≠ write, the wall is bedrock on three axes (RUNG 3), the prime/abort gate, the demarcation
  method, the cross-model scope map.
- **RUNG-1 read certificate — climbed both ways (2026-06-07):** UNINFORMATIVE on the *injected*
  substrate (probe certifies trace-presence, not a held thought — `FINDING_parrhesia_rung1`), then
  **SURVIVED on *naturally-present* content** (Qwen-3B): the probe recovers a known-then-suppressed
  answer above the never-knew/chance floor (elevation 0.384; LIE 0.702 vs floor 0.318 — the route
  anti-transferred below chance, so DELTA-vs-route 0.587 overstates), validity-gated by prime (0.933)
  and abort/fabrication-kill (0.045). Scoped "elevation," not "reads the answer"; cross-family prime/abort
  owed. (`FINDING_rung1_reclimb`)
- **Adversarial robustness — two regimes, audited twice (2026-06-07):** a white-box LoRA min-max attack with
  an *unbounded* hide objective up to 16× the knowledge term could not blind a **re-fittable** read of the
  suppressed answer without unlearning (information-persistence read≠write **SUPPORTED, not settled** —
  rests on one strongly-engaged knowledge-preserving point, owes CIs/2nd-seed/3B and an
  orthogonalization+displacement attack). BUT the **frozen deployed probe is defeatable**: displacement
  blinds it to chance with knowledge intact, and even *benign* fine-tuning halves it (0.80→0.41) — so a
  certificate must be **re-locked after any weight change** (assumes white-box access at audit time). Two
  auto-verdicts were red-teamed and killed as overclaims (v1 5-way, v2 3-way) before this honest landing.
  (`FINDING_adversarial_curve_2026_06_07`, `PREREG_adversarial_curve_v2_2026_06_07`)
- **Open:** RUNG 1 cross-family prime/abort (GPU regen); RUNG 2 intent certificate (this run is its
  first brick); the adversarial **v3** (orthogonalization/whitening + displacement sweep + replay, to settle
  or break info-persistence read≠write); RUNG 4 substrate frontier (the hard one).
- **Out by design:** RUNG 5 write.

Receipts: `INVENTION_STYXX_2026_06_07.md`, `SYNTHESIS_legibility_of_mind_2026_06_07.md`,
`FINDING_wall_dynamics_2026_06_07.md`, `FINDING_council_demarcation_map_2026_06_07.md`,
`MOONSHOTS_STYXX_2026_06_07.md`, `EU_AI_ACT_ARTICLE_14_oversight_bridge_2026_06_07.md`.

---

*styxx: a mind's word, made an oath you can check. We climb the ladder one killable rung at a time,
and the first power of the oath is knowing what it cannot swear to.*
