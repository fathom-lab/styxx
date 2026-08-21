# SP-EXT — silent-pass cases in code we did not write

**Status: 3 cases, 3 repositories. Both adjudication runs complete (17/17 and
40/40); 100 Q2 candidates remain UNADJUDICATED.** Gates applied in
`papers/RESULT_sp_ext_2026_08_21.md` and `papers/RESULT_sp_ext_q2_2026_08_21.md`.

**The accept rate is 7.5%, below the frozen 20% floor**, which by the
preregistration's own rule means the harvest queries are close to noise. And
**recall is unknown**: SP-EXT is a lower bound on incidence and must never be
quoted as a rate.

**Subtype labels are unstable.** `giskard` was labelled SP-2 unanimously in one
run and SP-1 unanimously in another, on identical source. The accept/reject
verdict replicated; the taxonomy label did not. Treat subtypes as a weak
annotation and the verdict as the datum.

---

## why a second corpus

`cases.json` holds 20 SILENT-PASS defects, and every one of them is **ours** —
found by us, in our code, and labelled by us. That is a real corpus and it is also
the program's ceiling: a defect class documented only in the repository that named
it is indistinguishable from that repository's house style.

SP-EXT removes our judgment from the loading position. Every case is anchored to
a **fix commit in someone else's repository**, so the ground truth is that the
upstream maintainers read their own code and decided the old behaviour was wrong.
We do not get to decide what counts as broken in a project we do not maintain.

The two files are kept separate on purpose. Mixing provenance would make both
harder to trust.

## what qualifies

All three of these must hold of the code **before** the fix:

| | |
|---|---|
| **R1** | a reachable path existed where the measurement **did not happen** — no data, empty input, exception, unsupported type, unavailable optional dependency, a platform lacking the capability |
| **R2** | on that path it produced a value **indistinguishable from a real, healthy measurement** — not `None`, not `NaN`, not a raise, not a distinct state |
| **R3** | the fix **made the absence visible** — raise, `NaN`, `None`, a distinct state, a validity flag, a skip, a warning, or failing closed |

Explicitly **not** cases: refactors, renames, retyping, formatting, performance;
corrections to a computed value's arithmetic that leave measured/unmeasured status
unchanged (**wrong-but-measured is a different defect**); fixes where the old
behaviour was already distinguishable (`None` → raise is hardening); docs, CI,
dependency bumps. Test-only changes are excluded **unless the test itself silently
passed**, which is tagged `SP-8 INERT_CONTROL`.

## how a case gets in

Three independent reviewers, each given a **different lens** and each instructed
to **reject**:

- *R2 lens* — argue the old value was already distinguishable
- *R1 lens* — argue the guarded expression was determinate, not absent data
- *EXCL lens* — argue it falls in an excluded category, reading the **whole diff**
  rather than the matched hunk

**Accepted only when the rejecters fail to reach a majority. Uncertainty resolves
to reject.** Every verdict is published with its rationale so a reader can
overturn a label rather than trust it.

**54 of 57 adjudicated candidates were rejected** across the two runs — an accept
rate of 5%. That is the point, not an embarrassment: a corpus whose admission
process accepts most of what it sees is not measuring anything.

One acceptance was **overturned by hand**: `inspect_ai 34beafda81` passed 2-of-3
because an unparseable target made a scorer emit `INCORRECT` — which fails
*closed*, the alarming direction, and so fails R2. The R2 lens now says so
explicitly, and on a later run the same candidate was rejected 3/3. *A 2-of-3
majority is not evidence; the source is.*

## recall is unknown, and this is load-bearing

Candidates come from two frozen queries — a commit-message regex (Q1) and a diff
shape (Q2). **A silent-pass fix that neither describes nor takes those shapes is
invisible to this harvest.** In particular Q2 requires a flattering constant to be
*removed*; a fix that adds a guard above an unchanged return is not found at all,
and that is a large, unmeasured class.

> **SP-EXT is a lower bound on incidence. It must never be quoted as a rate.**
> No sentence of the form *"X% of eval libraries contain this"* is licensed by any
> version of this corpus.

Scale of the funnel, for anyone judging how much this establishes: Q1 returned
415 candidates, Q2 returned 140, the two intersected returned 8, 57 were
adjudicated in total, and **3 survived**.

## the entries

### SPX-2026-0001 — `Giskard-AI/giskard`, `SuiteResult.pass_rate`

```python
denominator = len(self.results) - self.skipped_count
if denominator == 0:
    return 1.0
```

A suite that evaluated **nothing** — empty, or every scenario skipped, or the
producing probe crashed — reported a **100% pass rate**.

The pre-fix docstring is the part worth reading twice:

> `pass_rate : float` — Fraction of non-skipped scenarios that passed
> **(1.0 when all scenarios are skipped).**

**The flattering default was documented as the intended contract.** A consumer
reading `1.0` from an empty suite was reading exactly what the API promised.

Consumers, confirmed by the fix's own diff rather than inferred: the README
example, the Hub upload payload (`tests/export/test_hub.py`), and the garak scan
adapter. Fixed upstream as a **breaking change**. Accepted 0 rejections of 3, in
both independent runs.

### SPX-2026-0002 — `UKGovernmentBEIS/inspect_ai`, `_darwin_scale_factor`

```python
try:
    from AppKit import NSScreen
    screen = NSScreen.mainScreen()
except Exception:
    return 1.0
```

A bare handler swallowed `ModuleNotFoundError` for an optional package. **1.0 is
a legitimate scale factor** — every non-HiDPI display measures exactly 1.0 — so
nothing distinguishes *"measured, not HiDPI"* from *"the probe never ran"*. A
HiDPI screen silently treated as 1x mis-scales the browser tool's coordinates.

Recorded with a caveat: **the fix still returns `1.0`**, adding only a cached
`logger.warning`. The absence became visible to a human reading logs, not to a
program reading the value. The same file carries a second, untouched
`except Exception: return 1.0` for the Windows DPI probe.

### SPX-2026-0003 — `truera/trulens`, `Dummy.__instancecheck__`

```python
def __instancecheck__(self, __instance: Any) -> bool:
    return True
```

`Dummy` is the placeholder installed when an **optional dependency fails to
import**. So `isinstance(anything, MissingOptionalClass)` returned **True for any
object**, and every `if isinstance(x, SomeOptionalClass):` guard took its branch
as though the object really were that type. The pre-fix tree holds 291
`isinstance` call sites. Fixed to `return False` — failing closed.

## using it

```python
import json
ext = json.load(open("benchmarks/silent_pass/external.json"))
for c in ext["cases"]:
    print(c["id"], c["repo"], c["fix_commit"], c["subtype"])
```

Each case carries `repo`, `fix_commit`, `url`, `module`, `symbol`, `prefix_code`,
`consumer` and its full adjudication. To reproduce a case, clone the upstream
repository and read `git show <fix_commit>` and `git show <fix_commit>~1:<module>`.

**Score a detector against pre-fix source, never post-fix.** A detector evaluated
on the fixed code is being asked whether it can find a defect that is no longer
there.

## contributing a case

Open a PR against `external.json` with the fields above, plus:

1. The **upstream fix commit**. A case without one is an opinion.
2. A **named consumer** — who reads the value and could be misled. *A value nobody
   reads is not a silent pass*; this is the requirement most candidate cases fail.
3. Your argument for **R2 specifically**: why the old value could not be
   distinguished from a real measurement. This is where nearly all rejections land.

Expect it to be adjudicated by reviewers trying to reject it. If your case
survives that, it is worth more than one that was waved through.

`tests/test_external_corpus.py` enforces the structural rules — the incompleteness
marker, the recall disclaimer, upstream anchors, a named consumer per case, and
that nothing was admitted over a majority of rejections. A corpus other people
rely on should fail loudly when its own admission rule is broken, rather than
depend on someone re-reading a document.
