# PREREG — DECLARE-1: stop extracting claims from prose, and let the agent declare them

Fathom Lab · 2026-09-18 · Frozen before any code is written. Instrument at sha256 `eba8f5fc…`.
Follows `RESULT_decide1_decidable_fraction_2026_09_17.md`, which is the whole reason this exists.

## The measurement that forces this

Eleven preregistered cycles have been spent making a reader of English better at guessing which
sentences are checkable claims. The programme's own numbers now say that was the wrong problem:

- **71% of claims in this corpus are decidable from the diff** (DECIDE-1, hand-adjudicated, no
  oracle, 95% CI on the pooled figure [66.8%, 83.3%]).
- The instrument returns a verdict on **5.7%** of `only_touches` claims.
- Of the accusations it does make, **9 of 11 were wrong** before PATH-1 and **6 of 8 after**.

Every remaining failure mode is an extraction failure, not a checking failure. "only changed
mods/submods are serialized" describes a program's runtime; "only modifies CHANGELOG.md" describes
a diff; they are the same shape and the difference is meaning. A seventh shape turned up this week
in our own pull request: the gate read a **quoted** claim as an asserted one. There is no surface
test that separates these, two independent oracles failed trying, and the preregistration that
governs the benchmark forbids a third attempt by the same method.

So the method changes. **The claim stops being guessed and starts being declared.**

## The format

A fenced block in the pull request body, and nothing else is read as a declaration:

````
```styxx
files_changed: 38
only_touches: web/gate/**
adds_symbol: gate_diff_text
tests_added: 22
```
````

Rules, fixed now:

- One block per body. A second block is a **hard error**, never a merge of the two.
- Keys are exactly the claim kinds the instrument already checks. An unknown key is reported and
  never checked, never accused.
- A value that does not parse is `MALFORMED`, reported, never an accusation. A declaration that
  cannot be read is not a lie.
- The block is **opt-in**. A body without one is read exactly as it is read today.

## What this must not become

The obvious way to make this look good is to let it justify accusing harder on prose. It must not.

- **Prose extraction is not changed by this cycle at all.** Byte-identical readings on every
  non-declaring pull request, measured ledger to ledger, or the cycle does not ship.
- **A declaration never licenses an accusation about an undeclared thing.** Where the prose appears
  to claim something the block does not declare, the gate reports it as `UNDECLARED`, an
  observation, never a verdict. Declaring narrowly is allowed and visible; it is not punished.
- **`tests_pass` remains UNCHECKABLE even when declared.** Declaring that tests pass is not
  evidence that they did, and the one place an agent could most easily write the verdict it wants
  is the one place this format must not help it.

## The honest question, which is not precision

Precision on a declared claim is near-trivially high: the claim is exact, so the checking logic is
all that can fail. Reporting that as an achievement would be the same error as BENCH-2's
`only_touches` F1 of 1.00 — two implementations of one rule agreeing. **The interesting questions
are adoption and difficulty**, and the gates are written around those.

An agent that declares only what is safe has gained nothing for a reviewer. So the measurement is
whether declared claims are as hard as the ones prose makes.

## Gates — committed now

- **G-D1-1 (prose is untouched).** Every one of the 604 BENCH-corpus claims produces a
  byte-identical reading before and after. **Blocking.**
- **G-D1-2 (the ceiling, stated as a ceiling).** A declaration block is synthesized for each of the
  148 decidable BENCH items from the oracle's own facts, and the gate run against it. This measures
  what the format buys **when adoption is perfect and honest**, which is an upper bound and is
  reported in those words. It is not evidence that anyone will adopt it.
- **G-D1-3 (difficulty, not just accuracy).** For every declared claim, the fraction the *prose*
  reader would have abstained on. A format that only carries claims the old reader already handled
  has bought nothing, and that number says so.
- **G-D1-4 (the narrow-declaration hole is measured).** On the same corpus, how often a body's
  prose appears to claim something its block omits. Published as `UNDECLARED` counts per kind. If a
  declaration lets an agent hide, this is the number that shows it.
- **G-D1-5 (adoption is zero until it isn't).** No claim about uptake. Today exactly zero pull
  requests in the world carry this block. Any number in the RESULT is a property of a synthesized
  corpus and must be labelled as one.
- **G-D1-6 (the port keeps up).** `web/gate/diffgate.js` implements the same reading, the
  differential runs, and `py_side.py`'s pin moves in the same commit — the rule that #126 was
  written to establish.

## What would make this a bad idea

It is falsified as a product, not as a mechanism, and the falsifier is adoption. If nothing emits
the block, this is a format nobody uses attached to a gate nobody runs — which is the position the
differential was in last week. The RESULT must therefore end with the adoption number, whatever it
is, and if it is zero it says zero.

It is falsified as a mechanism if G-D1-3 comes back low: if agents can only declare the easy
claims, then the hard claims stay in prose, prose stays unreadable, and the format is decoration
over the same wall.

## Why this is the right shape

A gate that guesses at prose is an accusation machine, and this programme has measured what that
costs: 9 wrong in 11. A gate that checks declarations is a **commitment device**. The agent that
wants to be trusted says what it did in a form that can be held against it, and the agent that says
nothing is not checked and not accused. That is a better contract than a lie detector, and it is
the one the numbers point at.

---

*Eleven cycles were spent teaching a machine to read English well enough to catch a liar. The
measurement says the English was never the checkable part. So: declare the toll before you cross,
and nothing has to be guessed.*
