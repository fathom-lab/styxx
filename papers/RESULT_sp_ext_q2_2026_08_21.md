# RESULT — SP-EXT Q2: 3 of 40, a 7.5% accept rate, and the corpus falsifying our own screen

**Accept rate 7.5%, below G2's 20% floor. By the frozen rule that goes in the
title: on this evidence the harvest queries are close to noise.** Three cases
survive, in three repositories, and every other gate points at caution too.

Prereg: `PREREG_sp_external_corpus_2026_08_21.md` (`38b8428`) + sampling
amendment `AMENDMENT_sp_ext_q2_sampling_2026_08_21.md` (`87b51fa`), both frozen
before the candidates they govern were read. Raw: `out_sp_ext_q2.json`,
`out_sp_ext_q2_sample.json`, `out_sp_ext_q2_verdicts.json`.

---

## the run

Q2 standalone, finally executable on full-blob clones, examined **2,914 commits**
by pickaxe across all 14 repositories and returned **140 candidates** under the
verbatim frozen regexes — **17.5× the 8** that the message-intersection produced.
A random **40**, `seed = 20260821`, drawn before inspection and adjudicated
3 lenses each, 120 agents, 0 errors.

| | |
|---|---:|
| Q2 pool | 140 |
| sampled and adjudicated | **40** |
| **accepted** | **3** |
| rejected | 37 |
| **accept rate** | **7.5%** |
| left UNADJUDICATED | **100** |

## the gates

- **G1 YIELD** — 3 < 12. **No claim about how common this defect class is**, here
  or anywhere. The threshold was deliberately not rescaled for the smaller
  denominator; it was set for a corpus.
- **G2 ACCEPT RATE** — 7.5%, **below the 20% floor, so it goes in the title**.
  The queries pull in far more than they find. (For context and not as an
  escape: the 95% interval on 3/40 runs roughly 2%–20%.)
- **G3 SPREAD** — 3 repositories: `giskard`, `inspect_ai`, `trulens`. Still fewer
  than 4. **NARROW. No cross-project claim.**
- **G5 RECALL** — unknown by construction. **SP-EXT is a lower bound and is never
  quoted as a rate.**

## the new case

**SPX-2026-0003 — `truera/trulens`, `Dummy.__instancecheck__`**

```python
def __instancecheck__(self, __instance: Any) -> bool:
    return True

def __subclasscheck__(self, __subclass: type) -> bool:
    return True
```

`Dummy` is the placeholder installed when an **optional dependency fails to
import**. With these methods, `isinstance(anything, MissingOptionalClass)`
returned **True — for any object.**

Every guard of the form `if isinstance(x, SomeOptionalClass):` took its branch as
though the object really were that type. `True` is precisely what a genuine type
match returns, so **a missing dependency read as a satisfied type check.** The
pre-fix tree carries 291 `isinstance` call sites; any naming a Dummy-substituted
symbol matched unconditionally.

Fixed to `return False` — failing closed — with a docstring warning added:
*"While dummies can be used as types, they return false to all `isinstance`."*
Accepted 2-of-3.

## the consistency check nobody planned

The frozen draw happened to include all three commits from the earlier
intersection run. The amendment forbade swapping them out, which turned an
accident into a replication:

| commit | earlier run | this run |
|---|---|---|
| `dd75e974ee` giskard | ACCEPT 0/3 | **ACCEPT 0/3** |
| `acd139cc75` inspect_ai | ACCEPT 0/3 | **ACCEPT 1/3** |
| `34beafda81` inspect_ai | **accepted 2-of-3, overturned by hand** | **rejected 3/3** |

The third row is the one that matters. That candidate was accepted by the
protocol and I overturned it by reading the source: an unparseable target made the
math scorer emit `INCORRECT`, which fails *closed*. Afterwards I added one clause
to the R2 lens — *a value in the alarming direction is also a reject* — and on the
re-run the same candidate is **unanimously rejected**.

**The manual overturn and the protocol fix agree, on the case that motivated
both.** That is the closest thing to a positive control this adjudication has, and
it arrived by accident because a sampling rule forbade convenience.

## what the corpus did to our own instrument

`styxx.flattering`, scored against SP-EXT through the committed harness:

> **recall 0 of 3.**

Its full card is now: 10% recall on the internal corpus it was built from, 0 of 8
precision on third-party code, **0 of 3 on real external cases.** The instrument
is weak, and only an external corpus could have said so — our own corpus is the
one its rules were derived from.

`flattering` is frozen by its own preregistration and is not being edited to
catch these. The misses are pinned in `tests/test_external_scoring.py`.

## what is NOT claimed

- **Nothing about prevalence.** Three cases. G1 and G5 both forbid a rate.
- **Nothing about the 100 undrawn candidates**, in either direction. They are
  unadjudicated, which is neither rejected nor absent.
- **Nothing about the 37 rejections being clean code.** They failed *this*
  inclusion rule.
- **Subtype labels are not stable and should not be trusted.** `giskard` was
  labelled SP-2 unanimously in the first run and SP-1 unanimously in the second,
  on identical source. The **accept/reject decision replicated; the taxonomy label
  did not.** Anyone using SP-EXT should treat subtypes as a weak annotation and
  the verdict as the datum.

## what it establishes

Three silent-pass defects, in three independent projects that build evaluation and
observability infrastructure, each anchored to a fix its own maintainers shipped:

- a **pass rate of 100% from zero evaluations**, uploaded to a hosted dashboard as
  a genuine success rate, with the flattering default written into the docstring
  as the contract;
- a **screen scale factor of 1.0** returned when the probe could not run, where
  1.0 is also the correct answer for every non-HiDPI display;
- **`isinstance` returning True against a module that failed to import.**

All three are the same shape: *the value a failed measurement returns is a value a
successful measurement could also return.*

Three cases is not a prevalence estimate. It is an existence proof in the field,
and this project did not have one this morning.
