# SPEC — a manifest/0.1 has no rung, as its own decision already says (v0.1)

**Frozen 2026-09-06, before the code.** One rule, R1, and the guards that must fail before it is
written. Found by the eight-dimension adversarial audit (`wf_9466dcba-f49`, dimensions `manifest`
and `digest` — two reviewers reached it independently), and re-derived here before this spec.

## The verifier contradicts a decision it prints inside its own receipt

`DECISIONS["rung"]` travels in `verifier.decisions`, inside the digested core. It says, verbatim:

> a manifest/0.2 declares rung L1 … or L2 …; L3 (signed, verified) is reserved and refused; **a
> manifest/0.1, or a 0.2 with no rung, resolves at rung `undeclared`, never at L2**; an unknown or
> reserved rung string makes every rN span UNRESOLVED rung_unknown — the verifier declining to see a
> manifest that claims what it cannot check, never an accusation against the author

`Manifest.core()` honours it — `rung` is added to the digested body only when
`spec == "sworn/manifest/0.2"` (sworn.py:735). `from_dict` does not: it reads `rung` for **any**
spec (:767), and `rung_status()` (:690) consults it with no reference to the spec. So a
`manifest/0.1` resolves at whatever rung it declares, which the receipt's own text says never
happens. `sworn_verify.js` mirrors the same shape exactly — read at l.784, digested only for 0.2 at
l.795, `rungStatus()` at l.807 with no spec check — so **both implementations contradict the
decision in the same way**, and the differential harness cannot see it.

## Because `rung` is outside a 0.1 manifest's digest, this launders a verdict

Measured at `f6179c6e`, on a document whose sentence asserts `0.99` against a receipt of `0.42` —
a plainly false sentence:

```
0.1  rung=None -> SWORN-FAILED  rungs={"undeclared":1}  intact=True   manifest_digest=a328205796f6
0.1  rung=L9   -> SWORN-HELD    rungs={"unresolved":1}  intact=True   manifest_digest=a328205796f6
0.2  rung=None -> SWORN-FAILED  rungs={"undeclared":1}  intact=True   manifest_digest=2cdc2c8d5630
0.2  rung=L9   -> SWORN-HELD    rungs={"unresolved":1}  intact=False   (digest moved)
```

Appending one key to a `manifest/0.1` turns a false document from **SWORN-FAILED into SWORN-HELD**
with a **byte-identical `manifest_digest`** and `intact()` still true. The 0.2 row is the control
and shows the intended behaviour: there `rung` is digested, the appended key breaks the digest, and
the manifest reports `intact=False`. **0.1 is the only channel where the tamper is invisible.**

## What is NOT wrong, and is not changed here

The second half of the composition is **documented design and stays exactly as it is**:

- an unknown rung making every rN span UNRESOLVED is the decision quoted above — "the verifier
  declining to see a manifest that claims what it cannot check, never an accusation";
- a document with UNRESOLVED spans and no FAILED ones being SWORN-HELD is also documented, tested,
  and a proposal to change it was **retracted** in PR #74 after measuring its blast radius.

Neither is touched. An earlier draft of this finding called the whole composition "verdict
laundering"; that overstated it. What makes the 0.1 case a defect is not that an unknown rung
declines — it is that **a 0.1 manifest is honoured as declaring a rung at all**, against the
verifier's own printed rule, through a field its digest does not cover.

## R1 — the spec gates the rung, on both sides

`rung_status()` returns `("undeclared", None)` whenever `spec != "sworn/manifest/0.2"`, before any
other test. `Manifest.rung` continues to be stored as given — a manifest that loads must never
crash the verifier — but it is not consulted for a 0.1.

This mirrors `Manifest.core()` exactly, so the field a 0.1 manifest digests and the field it is
judged by become the same field: none.

`sworn_verify.js` takes the identical change in the same commit.

## What moves

- **Nothing committed.** Of 13 committed manifests, 1 is spec 0.1 and **0** declare a rung.
- A `0.1` manifest declaring `L1`/`L2` moves from `rungs {"L1"|"L2"}` to `rungs {"undeclared"}` —
  the documented outcome.
- A `0.1` manifest declaring an unknown rung moves from every rN `UNRESOLVED rung_unknown` to
  **adjudicated**. This is the repair: a false span that was hidden becomes FAILED.
- The conformance set will move, because every receipt embeds `verifier.sworn_sha256`. Per the
  ERRATA in `SPEC_numeric_sign_is_not_dropped_v01_2026_09_06.md`, the bar is not an unchanged
  `set_sha256` but: the same vector count, and **the multiset of expected outcomes unchanged unless
  a vector genuinely exercises a 0.1 rung**. If one does, it is a real verdict change, it is named,
  and the set is re-recorded in its own commit.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| R-G1 | a `0.1` declaring `L2` resolves `undeclared` | red: resolves `L2` | green |
| R-G2 | a false document under a `0.1` with an unknown rung is SWORN-FAILED, not SWORN-HELD | red: SWORN-HELD | green |
| R-G3 | the tamper is no longer free: appending a rung to a `0.1` cannot change `rungs` at all | red | green |
| R-G4 | a `0.2` declaring `L1`/`L2` still resolves at that rung | green throughout — catches over-reach |
| R-G5 | a `0.2` with an appended rung still reports `intact=False` | green throughout |
| R-G6 | Python and the JS verifier agree by core digest on every (spec, rung) pair | green with both wrong, **red if only one side is fixed** — the parity gate |

R-G2 is the guard that must be seen red. R-G6 must be demonstrated red by reverting one side, as in
the sign repair, or it is decoration.

## What this does not claim

That the rung means anything. Nothing verifies a rung; the verifier prints what the manifest
declares, and `DECISIONS["provenance"]` says so. This makes a `0.1` stop declaring one, which is
what the format already said it did.
