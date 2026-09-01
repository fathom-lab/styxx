# RESULT — V14: the named false accusations were removed, and precision did not follow

Fathom Lab · 2026-09-01 · Prereg: `PREREG_v14_repair_2026_08_31.md`, frozen before the
repairs existed and before the held-out split was read. Receipts: `v14_gates.json`,
`v14_adjudication.json`, `v14_packet.json`, `v14_answers.json`, `v14_panel_raw.json`,
`v14_residual.json`, `v14_key_digest.txt`.

**Two gates passed, the one that mattered failed, and the accusation stays disabled.**

## What passed

- **G-S1 (subset invariant) — PASS.** Across all 71,016 eligible pull requests, **zero paths
  gain an accusation**. Both repairs are accusation-removing only, as preregistered.
- **G-S2 (cumulative recovery) — PASS at 0.6975** against the 0.6667 bar that V13 alone
  failed at 0.3462. Corpus-wide, path accusations fall from 4,427 to 1,344.

The two repairs were designed against the residual rather than guessed: the surviving
accusations were grouped by the gate's own reason and by sentence shape, with the shape
labels written before any counting. Half fell into named shapes; reading the rest exposed
containment repaired for the wrong verbs, and bare names — `asmcrypto.js`, `ethers.js` —
that are packages rather than files.

## What failed

**G-S3 (held-out precision) — FAIL. Observed 0.16 against a floor of 0.95.**

A fresh blind panel, new sample, new sealed decoys, key digest committed publicly before any
item was judged. The panel called **30 of 30 decoys correctly** and was unanimous on the
large majority of items, so the figure is a fact about the instrument rather than about the
judging. Of one hundred sampled held-out accusations, sixteen were upheld.

## The finding, stated carefully

The preregistration anticipated this outcome and named what it would mean: *the gap between
the false accusations we can name and the precision a stranger measures becomes the finding.*
That gap is now measured, and it is wide. **Removing the large majority of the false accusations we
could characterise did not move precision to anywhere near the floor.**

The reason is arithmetic and worth stating plainly. The repairs remove accusations; they
cannot add any. Precision after repair therefore depends entirely on whether the removed
accusations were disproportionately the false ones — and on held-out prose they were not
disproportionate enough. The classes we could name are not the classes that dominate the
error.

**One comparison this paper explicitly refuses to make.** The earlier 0.23 figure was
measured over a sample drawn from all eligible pull requests; this 0.16 is held-out only.
They are different populations and charting them as a trend would be dishonest, so no claim
is made here that precision fell, rose, or held. What is established is the level: after two
repair cycles, precision on prose the instrument's authors have never read is **0.16**, and
the floor is 0.95. The matched held-out baseline was never measured, and measuring it would
take another panel; that omission is named rather than papered over.

## What this decides

The preregistration committed to a rule for exactly this: three cycles of mechanical repair
falling short is evidence about the approach rather than an invitation to a fourth attempt.
**This lab is not repairing this class again.** The accusing verdict for path claims stays
disabled in shipped code, and the four `xfail(strict=True)` markers guarding the catches it
gave up stay where they are.

What survives is narrower and honest: the gate still checks counts, symbols and prefix
claims, and it still reports what it read and what it never read. The path-claim class is
retired as an accuser and kept as an observer.

## The boundary this leaves standing

An instrument that reads open-ended prose with a closed template set has now been measured
three times and failed three times, and the third failure came *after* its named defects
were fixed. That is the strongest evidence this lab has produced that the limit is not in
the templates. The successors already preregistered — the worklog, and the three-artifact
reconciliation it makes possible — do not read prose to decide anything, which is the whole
reason they were designed that way.

## A defect in this measurement, disclosed

Three of the 130 items were adjudicated by two seats rather than three; one batch returned
short. Their majorities are unaffected by the missing seat, and no item was decided by a
single seat, but the protocol said three and three did not happen everywhere.

---

*We found the defects, named them, fixed them, and proved the fix removed the bulk of
the accusations we understood. A panel that had never seen this prose then scored the result
at 0.16. The repairs were real; the theory behind them was too small.*
