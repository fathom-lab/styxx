# RESULT — the edge screen never reached the external run: 0 of 20 at the GO/NO-GO

**Prereg G0: catch ≥ 8 of 20, or no external run and the number gets published.
It caught 0. No external repository was fetched. The preregistration is
terminated.**

Prereg: `PREREG_edges_2026_08_21.md`, frozen and pushed at `6723be3` **before a
line of `styxx/edges.py` was written.** Control:
`scripts/edges_positive_control.py`. Diagnostic: `scripts/edges_funnel.py`.

---

## what the gate was for

`styxx.flattering` scored 10% recall on this same corpus *while being fitted to
it*, and its external `0 of 8` was uninterpretable as a direct consequence — a
screen with zero recall and a defect-free corpus produce byte-identical output.
So G0 was written to stop exactly that from happening twice:

> If the edge screen catches fewer than 8 of the 20 corpus cases, it is **not run
> against external code at all**.

**The gate did its job.** Six public repositories were named in the prereg and
never cloned. This is what a kill criterion is supposed to feel like from the
inside.

## the instrument works; the specification was too narrow

The screen is not broken. Against hand-written cases it is 3 for 3 on true
positives and 0 for 7 on the negative controls (`tests/test_edges.py`), where
each negative control is a **named false-positive class** from the flattering
adjudication — NaN defended, no consumer, boolean flag misread as a container,
no contrast, non-deciding consumer, rebound local, both branches quiet.

Against 227 real files it flagged **nothing**, and the funnel says precisely why:

| stage | count |
|---|---:|
| functions defined | 1417 |
| producers (absence → constant) | 95 |
| …that also have a computed return | 71 |
| **decisions on a producer's value** | **19** |
| killed by CONTRAST (req. 5) | 8 |
| killed by POLARITY (req. 4) | 12 |
| flagged | **0** |

**The binding constraint is stage one, not the five requirements.** Requirements
3, 4 and 5 are not what emptied the result — there was almost nothing arriving
for them to filter. Nineteen producer→decision pairs in a 227-file package.

The reason is topology, and it was measured afterwards
(`scripts/corpus_topology.py`) rather than assumed:

| how the bad value exits its producer | cases |
|---|---:|
| scalar (returned bare, or a local then returned) | 9 — 45% |
| **object-field** (`keyword=` into a constructor, read later as `r.field`) | 6 — 30% |
| **dict-key** (`d["valid"] = …`, read later by key) | 3 — 15% |
| module-level / unclassified | 2 — 10% |

That table is the secondary finding. The primary one is harder:

> **`styxx.edges` requires the producer call and the decision to sit in the same
> function. By the corpus's own recorded consumers, that holds for 0 of 20
> cases.**

Every consumer on record is somewhere else — *"ForecastGate compares risk_level
against a threshold"*, *"context-injection defence at the caller layer"*,
*"ProtocolEnvelope.validate"*, *"calibrate, learned_classifier, antipatterns,
weather, feedback"*. Not one is the producing function.

**So the G0 floor of 8 was unachievable by construction.** The specification I
froze excluded 100% of the target before a single file was scanned. The screen
did not fail to find the defects; it was incapable of finding any of them, and
0/20 was determined the moment the design was fixed.

## the actual finding, and it is about my process

I declared this blindness **in advance**, in the module's own docstring:

> *"Deliberately shallow, and each limit is a declared blindness rather than an
> approximation: no tuple unpacking, no attribute targets, no cross-function
> propagation, no flow through dataclass fields."*

And then I preregistered a gate against a corpus **whose defects live almost
entirely on the far side of that exact hop.**

> **Declaring a limitation is not the same as measuring whether it is fatal.**
>
> And worse: **I set a numeric target that my own frozen specification made
> impossible, and a five-minute measurement of the corpus would have shown it.**

The two diagnostics that made this obvious — `edges_funnel.py` and
`corpus_topology.py` — took minutes and could both have been run before the
preregistration was frozen. A limitation written into a docstring reads like
diligence; it discharges nothing. This is the same shape as the flattering run's
missing positive control, one level earlier in the process — there I failed to
check whether the instrument could fire at all, here I failed to check whether it
could fire *on the topology I was aiming it at*.

There is a specific correction to make to my own doctrine. This project's rule has
been **"a leg that cannot fail must not gate."** Today produced its mirror:

> **A gate that cannot pass measures nothing either.** `G0 ≥ 8 of 20` looked
> stringent and was in fact vacuous — it could only ever return the answer it
> returned. Both failure modes are the same error, which is a gate whose outcome
> was fixed before the data arrived.

That is the ninth time in this program that a tool or an experiment committed a
version of the defect class it hunts, and it is the second today.

## what does not follow

**Nothing about H1.** The thesis in `SYNTHESIS_the_edge_2026_08_21.md` — that
SILENT-PASS is a property of an edge — is **untested by this run**. `styxx.edges`
is one operationalization of it, and an intra-procedural one at that.
`INVALID`/terminated is not a null and must not be cited as evidence against the
reframe, any more than the RDM-reliability `INVALID ×2` was evidence against
representational reliability.

**Nothing about G7.** The comparison against `contract` and `flattering` on a
shared external corpus never ran, so the falsifier the prereg named is still
outstanding.

## what ships

`styxx/edges.py` stays in the tree, **unexported from `styxx/__init__.py`, with no
console script** — the same treatment `flattering` got, for the same reason: an
instrument that flags nothing on real code is not a product surface. Its
docstring carries this number.

Acceptance tests stay, and the seven negative controls stay asserted, because
they encode the adjudication's findings and any future widening that breaks them
is a regression into the instrument this one was built to replace.

## the honest next step, and its precondition

The obvious extension is to follow a value across the hop it cannot cross — into
a dataclass field or dict key, and out again at a reader elsewhere. That is a
**different mechanism**, not a widened rule, so it would be a **new
preregistration and attempt 2**, with the count visible.

But it is not being written yet, because that would repeat today's mistake at a
higher level: guessing at the fix rather than measuring the target. **The
precondition is a measurement of the corpus's actual topology** — for each of the
20 known defects, where does the value go between being produced and being
decided on? Same function, dataclass field, dict, module global, or never decided
on at all? That measurement is cheap, it is about known-true instances, and it
determines whether field-flow is the right mechanism or merely the obvious one.

Until that exists, any v2 is a hopeful guess wearing a preregistration.

## cost

One instrument, 14 acceptance tests, one positive control, one funnel diagnostic.
No external fetches. No API spend. No GPU.
