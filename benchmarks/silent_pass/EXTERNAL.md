# SP-EXT — silent-pass cases in code we did not write

**Status: INCOMPLETE. 7 of 17 candidates adjudicated, 1 accepted. No gate has been
applied.** See `papers/RESULT_sp_ext_interim_2026_08_21.md`. This document
describes what SP-EXT is and how to use or extend it; it does not describe a
finished corpus, because there isn't one yet.

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

Six of the seven adjudicated candidates were rejected **3/3**. That rate is the
point, not an embarrassment: a corpus whose admission process accepts most of what
it sees is not measuring anything.

## recall is unknown, and this is load-bearing

Candidates come from a frozen commit-message regex intersected with a frozen diff
shape. **A silent-pass fix described some other way is invisible to this harvest.**

> **SP-EXT is a lower bound on incidence. It must never be quoted as a rate.**
> No sentence of the form *"X% of eval libraries contain this"* is licensed by any
> version of this corpus.

The harvest is also currently a lower bound *on that lower bound*: the
preregistration's second query could not execute (see the interim result), so only
the intersection of the two queries ran.

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
adapter. Fixed upstream as a breaking change. Subtype **SP-2 SENTINEL_DEFAULT**,
agreed 3/3.

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
