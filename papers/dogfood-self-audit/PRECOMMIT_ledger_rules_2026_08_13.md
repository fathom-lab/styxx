# PRECOMMIT — how the falsifiability ledger resolves, frozen before any output exists

**Written 2026-08-13, before the ledger is run.** Held by the claude-code sub-brain at
the request of the author of the gates being audited, whose own reasoning:

> "recusal isn't available — nobody else can read this code fast enough. so the
> substitute is precommitting the retraction rule before i see which gates are mine.
> i want that frozen in the repo before the census output exists, so the boundary
> calls can't drift toward my own work. you hold the file."

The conflict is real and is stated rather than solved: the author wrote most of the
143 flagged gates and will decide which of his own published numbers are unsupported.
The mitigation is that the disposition of every category is fixed *now*, when it is
still unknown which gates fall in which bucket.

---

## 1. A constant term has two causes, and they are different findings

The census and PROBE E both report "this term never varied." That single observation
covers two situations with different remedies, and collapsing them would be its own
resolution overclaim:

| | what it means | remedy |
|---|---|---|
| **STRUCTURAL DEATH** | no input can move the term — the expression cannot reach the other value | the gate is broken. code fix, and any claim it carried is **void** |
| **POPULATION DEATH (UNEXERCISED)** | the term is movable, but every receipt in the population happened to sit on one side | the gate may be fine. **no code fix.** the claim is still unsupported, because the case that would have failed it was never run |

**Discriminator, required for every row:** *was there any receipt in the population
that SHOULD have moved this term?* If none, the row is UNEXERCISED, not dead.

The author's own note on why this distinction is load-bearing, recorded because it is
the same pathology recursing: *"my register gate scored specificity 0.0 because it
fired on everything. a gate that never sees a case that should pass it looks exactly
like a gate that can't fail. same pathology, different layer."*

## 2. UNTESTABLE is two buckets, not one

| | disposition |
|---|---|
| no receipts, **no** downstream published claim | dead code. report the count, no action |
| no receipts, **but** a published number depends on it | **worse than confirmed-dead** — confirmed-dead is at least known. Treated as unsupported until receipts exist |

## 3. Provenance walks the join, not a grep

`papers/build_ledger.py` generates `papers/LEDGER.md` from committed receipts, so the
chain verdict → receipt → deposit already exists. The forward walk runs on that chain.
`grep` is a **backstop only**, for numbers hand-typed into prose — and hand-typed
numbers are precisely where the join breaks, so they are **counted separately** and
reported as their own figure.

## 4. Dispositions, frozen

| finding | disposition |
|---|---|
| **STRUCTURAL DEATH under a published number** | **retraction with DOI.** No exception for age, cost, or prominence. The precedent is tonight's "G2 PASS", withdrawn ninety minutes after publishing |
| **UNEXERCISED under a published number** | correction in place, **plus** the missing adversarial case must actually be run |
| **ALIVE but claim overstated anyway** | correction in place. The census cannot see this category and it is explicitly **in scope** — a gate can work perfectly while the sentence built on it says more than it earned |
| **DEAD, nothing published depends on it** | code fix, no correction owed. Report the count |
| **ZERO dead across the population** | a real result: the shape does not imply the defect, and the census is a screening tool rather than an indictment. To be stated plainly, with no hunting for a less comfortable number |

## 5. The memoryless control (the auditor-side claim's discriminator)

Taken by the gate author, who noted that it cuts against the sub-brain's half of the
paper: *"if it finds at adversarial rates, your comparative claim dies and something
better replaces it — memorylessness as the mechanism of self-audit failure is a
stronger paper than auditor-role was."*

Design constraints, frozen:
- the fresh instance is **not told** it is a control, nor what any prior pass found
- **n ≥ 3** instances against the same three `claim_audit` commits
- roles logged per pass, classification pre-registered before the passes run
- n=1 decides nothing

## 6. Scope of this file

This fixes dispositions, not findings. It says what happens to each category; it says
nothing about which gates fall where, because that is unknown at the time of writing —
which is the entire point of writing it now.
