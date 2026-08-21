# RESULT — the screen found nothing outside our own code, and it was never able to

**Every one of the 8 external candidates was refuted, 24/24, unanimously.
The screen's recall on its own training corpus is 10%.
The preregistered test is `INVALID` and licenses no claim about how common this
defect class is.**

Prereg: `PREREG_flattering_external_2026_08_21.md`, frozen and pushed at commit
`4272d44` before the scan ran. Raw: `out_flattering_external.json`,
`out_flattering_adjudication.json`.

Per **G4** — "if BENIGN ≥ 80%, that is a finding about the detector, and it goes
in the title." BENIGN is 100%. It is in the title.

---

## what was asked

Every SILENT-PASS result this project has published came from auditing **its own
repository**. A defect class found only in the code that named it is
indistinguishable from a house style. So: does the flattering-default pattern —
an explicit emptiness guard whose fallback is the reassuring constant — occur in
Python written by other people?

## what happened

**19,632 files** across **91 third-party packages** (1 unparseable).

| | |
|---|---:|
| TIER-A (polarity evidence, claimed) | **8** |
| TIER-B (structural only, never claimed) | 662 |
| packages with ≥ 1 TIER-A | 7 / 91 |
| **adjudicated GENUINE** | **0** |
| adjudicated BENIGN | 8 (24/24 refutations, all at high confidence) |

**G2 tripped before anything was claimed:** 8 < 15 → `INVALID__UNDERPOWERED`. A
proportion cannot be estimated from eight.

The corrected count is smaller still. `torch/_numpy/testing/utils.py` is a
**verbatim vendored copy** of `numpy/testing/_private/utils.py` (and the torch
copy has zero callers anywhere — dead code), and the two scipy hits are the
**byte-identical expression** at lines 102 and 226 of one function. **8 flagged
nodes → 6 distinct code sites → 0 confirmed.**

## the objection that invalidates the run

Raised by the adversarial adjudication, not by me:

> *A screen with zero recall and a defect-free corpus produce byte-identical
> output, and this run cannot distinguish them.*

**That is this project's own defect class, committed by this project's own
experiment.** An instrument that never fired, reported as a clean corpus. There
was no positive control. `0 of 8` was uninterpretable the moment it was printed,
and I printed it anyway.

So the control was run afterwards — the frozen screen against **real pre-fix
source for all 20 corpus cases**, extracted from git at `<fix_commit>~1`
(`scripts/flattering_positive_control.py`):

**2 of 20. Recall 10%.**

Caught: SP-2026-0001 (`witness`, `"OK"`), SP-2026-0012 (`temperature`,
`confabulation_ratio`). Missed: the other eighteen, **including four of the five
SP-6 cases the screen was written for.**

That 10% is an **upper bound on recall anywhere**, because these are the cases
the rules were derived from. Against an instrument with ≤10% recall and a 100%
false-alarm rate on the one corpus where it fired, `0 of 8` carries no
information in either direction.

## why it failed — eight named flaws, three of which each kill all six

From the adjudication. Every one is a check the screen does not perform.

**C1 — no consumer-liveness analysis (kills 6/6).** The single highest-yield
analysis omitted. `inspect_ai`'s `_validated` has four references, all inside its
own accessor and **zero external readers** — the caller my claim described *does
not exist*. Both numpy call sites discard the return value. scipy's
`stop_criteria` declares `constr_violation` and never dereferences it.

**C2 — a determinate zero read as an absent measurement (fires 8/8).**
`HAS_REFCOUNT` is an import-time constant. `in_fit` is bound to a literal at
every call site. `t.args` is generic arity — zero means *maximally* determined.
`len(b)` is a fixed problem dimension. None of them is a runtime sample count.
The discriminating question the screen never asks: **can this expression vary
across executions at this program point?**

**C3 — return-channel misidentification (kills 5/8).** `_assert_valid_refcount`'s
real codomain is `{True, None}` plus `AssertionError`. **Success is falsy `None`;
the skip value is truthy `True`.** The polarity is not merely conflated, it is
*inverted*, and the screen never enumerated the return set.

**C4 — polarity from name morphology (fires 6/8), and this is the load-bearing
one.** In sklearn, `True` is the *expensive conservative* branch and the flagged
`None` is the cheap default — **the screen flagged the strictly weaker,
side-effect-free branch as the flattering one.**

**C5 — the fallback identity was never evaluated (kills 4/8).**
`np.linalg.norm(np.array([]), np.inf)` returns `0.0`. **Both branches of the
scipy ternary are numerically identical** — it is a legacy compatibility shim,
not a fallback. A ternary whose branches provably agree cannot conflate anything.

**C6 — inbound argument read as outbound measurement.** `force_writeable=…`
travels *into* `check_array`. The defect class requires an outbound measurement a
downstream reader misinterprets; an inbound value is categorically ineligible and
should have been excluded at pattern-match time.

**C7 — cross-function attribution without dispatch resolution.** Two
`stop_criteria` implementations exist; the one that thresholds the value is
reachable only when `len(b) ≥ 1`, so **the flagged branch is unreachable on the
only path where the value is read.**

**C8 — fabricated specifics in the claim narrative, and this one is mine.** My
hit description named **PyPy**. `HAS_REFCOUNT` excludes only Pyston; numpy carries
a *separate* `IS_PYPY` two lines above, and modern PyPy exposes `sys.getrefcount`.
I also cited `analysis/beta/_dataframe/columns.py`, a path that **does not exist** —
`analysis/beta/` is a deprecation shim. I pattern-matched "refcount guard → PyPy"
without reading the constant above the guard, in a claim written to be checked.

Independently measured, from outside the frozen detector
(`scripts/flattering_flaw_audit.py`): **87% of TIER-B and 62.5% of TIER-A rest on
a bare name in the test position** — `if not HAS_REFCOUNT:` — which the screen
reads as an empty container and which is, in real code, a boolean flag.

## the part that is actually a result

> **The filter selects for the construct where the pattern is benign, and
> deselects for the construct where it is dangerous.**

The screen requires polarity vocabulary. In numerics, compilers and ML plumbing,
the words that pass the filter — `valid`, `assert`, `check`, `has_` — are exactly
the words those libraries use for **one-sided predicates and assertion helpers**,
which is precisely the construct where an optimistic empty-case return is the
*mathematically correct* answer: vacuous truth, arity zero, the supremum of an
empty set. The words that would mark a genuine measurement channel — `score`,
`confidence`, `risk`, `entropy`, `coverage` — are nearly absent, because these
libraries name their measurements `mean`, `sum`, `norm`, `b`, `rc`.

All eight survivors were predicates, assertion helpers, or bare numeric
magnitudes. **Not one was a reported metric.** The screen never surfaced a single
instance of its own target *kind of object*, let alone a defective one.

And the corpus was wrong for the question. This defect class lives in evaluation,
monitoring, gating, telemetry and safety code. A scientific-Python
`site-packages` contains almost none of that. **Scanning 19,632 files of numerics
is scanning the wrong organ for the pathogen.**

**No sentence of the form "this defect class is rare in third-party Python" is
supported by this run, and none appears in this project's writeups.**

## what it did find — in our own code, twice

The same screen, pointed at styxx, produced 8 TIER-A hits. Six were false alarms
of the classes above. **Two were live defects that 46 fixes on 2026-08-19 had
missed**, and neither was in the corpus:

**`three_axis/regen_scorer.py` — `_entropy_topk` returned `0.0` nats for no
alternatives.** 0.0 nats is not neutral; it is maximum certainty. The real damage
was one level up: `mean_entropy_topk_nats` is guarded by
`sum(Hs)/len(Hs) if Hs else float("nan")`, which is correct — but a provider
returning `logprob` without `top_logprobs` made every call return a real-looking
zero, so `Hs` filled, the guard saw a non-empty list, and **the honest refusal
never fired.**

**`diffgate.py` — `only_touches` verified a scope claim against an unreadable
diff.** `outside = [p for p in status if …]` is empty when `status` is empty, so
`"VERIFIED" if not outside` returned **VERIFIED** for the input
`"Sorry, I could not produce a diff."` The module whose entire purpose is
refusing to take an agent's word took the agent's word, by way of a vacuous
truth. Both now fixed, with tests; `DiffGate` grew a `measured` channel, because
`PASS`/`FAIL` cannot carry *"this gate did not run."*

## the instrument card, and what ships

| | |
|---|---|
| recall (own corpus, real pre-fix source) | **2/20 — 10%**, an upper bound |
| precision, external | **0/8** |
| precision, in-repo | 2/8 |
| dominant false-positive cause | bare name in test position — 87% of TIER-B |

**`styxx.flattering` is therefore NOT exported from `styxx/__init__.py` and gets
no console script.** It ships as a research script. A screen with 10% recall and
a 100% external false-alarm rate is not a product surface, and putting it behind
`import styxx` would advertise a capability the measurements do not support.

**`styxx/flattering.py` is left byte-identical to `4272d44`.** Prereg G3 forbids
editing it after the scan, and that includes not writing these numbers into its
docstring — an unedited file keeps the run checkable by anyone who clones it. Any
future edit ships with a re-run.

## what this licenses next, and under what conditions

The adjudication named the missing analysis without being asked to design one:
**C1, consumer liveness — "does any reader exist, and does it decide on this
value?"** — is the check that kills all six candidates, and **C6** states the
type constraint directly: the defect requires an *outbound* measurement that a
*downstream reader* misinterprets.

That is the same conclusion `styxx.contract` reached by failing in the other
direction (`RESULT_contract_sp6_2026_08_21.md`: 3/3 boundary-visible, 0/2
interior). Two instruments, two independent failures, one shape. It is written up
in `SYNTHESIS_the_edge_2026_08_21.md`, and it is a **hypothesis generated by
failed runs** — it licenses a new preregistration, on a corpus of evaluation and
safety code rather than numerics, with a **positive control declared in advance**.
It licenses nothing today.

## a caveat pointing the other way

The adjudicator refuted 24 of 24 with no known-genuine control of its own — the
same two-sided-admissibility problem, one level up. Five load-bearing factual
claims were independently spot-verified (numpy's empty inf-norm returns `0.0`;
`HAS_REFCOUNT` excludes Pyston not PyPy; `_validated` has zero external readers;
torch's copy has zero callers; the two scipy hits are the identical expression),
and all five held. So the unanimity is not an adjudicator skipping the work. It
is also not independently calibrated.

## cost

One scan (19,632 files, ~3 min), 24 adjudication agents, one positive control.
No API spend beyond the adjudication. No GPU.
