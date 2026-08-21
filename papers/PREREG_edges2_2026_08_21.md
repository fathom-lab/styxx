# PREREG — edge screen, attempt 2: follow the value across one hop

> ## STATUS: **NOT FROZEN. NOT RUN.** Held at the attainability check.
>
> The attainability section below was written before the check was run, and the
> check **failed: 0 of 3 hops demonstrated** on real pre-fix source
> (`scripts/edges2_attainability.py`). No consumer was found that decides on
> `TruthMap.confabulation_ratio`, on `d["valid"]`, or on the result of
> `detect_context_injection` — in the pre-fix tree or the current one.
> `confabulation_ratio` is written, put in an f-string, and serialized. It is
> never branched on.
>
> A follow-up census (`scripts/corpus_consumer_census.py`) reports 15/20
> "DECIDED", **and that number is not usable**: it traces a token *name* across
> the repository, so `if gate == "pass"` counts as a consumer of whatever
> produced `gate`. That is C7 — cross-function attribution by name without
> dispatch resolution — committed inside the measurement written to avoid it.
> It is an upper bound; the attainability check is a lower bound of 0.
>
> **So the G0 floor of 10/20 cannot be shown answerable, and by this document's
> own rule it is therefore not frozen.** The instrument is not written. No
> repository has been fetched.
>
> What has to happen first: resolve the interval between 0 and 15 by tracing
> actual value paths for a sample of cases, not names. Only then can a floor be
> set that is neither vacuous nor unreachable.

**Attempt 2 of the edge thesis. Attempt 1 terminated at its own GO/NO-GO with
0 of 20 (`RESULT_edges_2026_08_21.md`), having never touched external code. The
count is stated here so it stays visible.**

Written before `styxx/edges2.py` exists and before any external repository is
fetched.

---

## why this is a new mechanism and not a widened rule

Attempt 1 required the producer call and the decision to sit in **the same
function**. Measured afterwards (`scripts/corpus_topology.py`), that condition
holds for **0 of the 20 corpus cases** — every recorded consumer is somewhere
else: `ForecastGate`, *"the caller layer"*, `ProtocolEnvelope.validate`,
*"calibrate, learned_classifier, antipatterns, weather, feedback"*.

The change here is not a loosened threshold. It is **inter-procedural
propagation across one hop**, which attempt 1 did not have in any form. Rerunning
attempt 1 with different constants would still score 0.

## the measured shape of the target, from attempt 1's post-mortem

| how the value exits its producer | cases | reachable by the design below |
|---|---:|:--:|
| scalar — returned bare or via a local | 9 | yes |
| object-field — `keyword=` into a constructor, read later as `r.field` | 6 | yes |
| dict-key — `d["valid"] = …`, read later by key | 3 | yes |
| module-level / unclassified | 2 | no |

**18 of 20 are in-principle reachable.** That number is why the gate below is
set where it is, and this table is the reason the gate is not vacuous.

## the mechanism

Producer detection is unchanged from attempt 1 (absence → constant, with bare
names disambiguated by container use). What is new is the hop:

1. **return hop** — `F` returns the constant; a caller binds `x = F(...)` and
   decides on `x`. *(One assignment, intra-procedural in the caller.)*
2. **field hop** — the constant is passed as `keyword=` into a constructor; a
   reader elsewhere decides on `obj.<keyword>`. Matched by **attribute name**,
   with the class name recorded and reported.
3. **key hop** — the constant is written to a dict under a literal key; a reader
   elsewhere decides on `d["<key>"]`.

Requirements 3 (indistinguishable — NaN/None/Measured are defended), 4 (polarity
from the **consumer's** branch structure, never from names) and 5 (contrast)
carry over unchanged and remain asserted in `tests/test_edges.py`.

**Hops 2 and 3 match by name, not by type, and that is a stated weakness.** Two
classes with a `confidence` field are indistinguishable to this screen. The
false-positive cost is real and is exactly what G5's adjudication exists to
price; the alternative — a type inference pass — is out of scope and saying so
is cheaper than pretending otherwise.

## ATTAINABILITY, and this section exists because attempt 1 lacked it

> **A gate that cannot pass measures nothing either.** Attempt 1's floor of
> `≥ 8 of 20` looked stringent and was unachievable by construction: `0/20` was
> determined the moment the design was frozen, before any data existed. That is
> the same error as a leg that cannot fail, with the sign reversed.

So, **before this preregistration is frozen**, one worked case of each hop is
demonstrated traversable by the specified mechanism, recorded in
`scripts/edges2_attainability.py`, and its output pasted into this document. If
any hop cannot be demonstrated, the gate below is lowered **before freezing**, in
the open, with the reason stated — not after seeing a result.

**No gate in this project is frozen again without an attainability check.**

## gates, frozen

**G0 — GO/NO-GO.** ≥ **10 of 20** corpus cases, against real pre-fix source at
`<fix_commit>~1`. That is 56% of the 18 the design can in principle reach.
Below 10 → no external run, the number is published, and **the edge thesis is
recorded as unsupported by two attempts** rather than pursued into a third.
This is in-sample and an **upper** bound on recall.

**G1 — PRIMARY, precision.** ≥ **30%** of adjudicated external findings GENUINE.

**G2 — RESOLUTION.** Intra-package call resolution < 25% → `INVALID__BLIND`.
Both the intra-package and raw figures are published.

**G2b — HOP ACCOUNTING, new.** Findings are reported **broken down by hop**, and
the false-positive rate is reported **per hop**. If one hop supplies most
findings and most refutations, that hop is named in the title.

**G3 — POWER.** < 15 external findings → `INVALID__UNDERPOWERED`, not a null.

**G4 — TWO-SIDED.** BENIGN ≥ 80% goes in the title. High precision achieved by
barely firing puts the firing rate in the title instead.

**G5 — ADJUDICATION.** Three independent reviewers per finding, each prompted to
**refute**, distinct lenses, uncertainty resolving **against** the hypothesis.
Every verdict published with rationale.

**G6 — ANTI-TUNING.** `styxx/edges2.py` frozen at a commit recorded in the
RESULT before any external fetch. Any later edit voids the run.

**G7 — COMPARISON, the falsifier.** `contract` and `flattering` run over the same
external corpus. H1 requires the edge screen to beat **both** on genuine
findings. If a node screen matches it, node analysis was merely underpowered,
"edge" is a distinction without a difference, and
`SYNTHESIS_the_edge_2026_08_21.md` is **retracted**.

## corpus, named before fetching

`EleutherAI/lm-evaluation-harness` · `explodinggradients/ragas` ·
`confident-ai/deepeval` · `NVIDIA/garak` · `truera/trulens` ·
`Giskard-AI/giskard` · local `inspect_ai`. Pinned by commit SHA in the RESULT.
Tests excluded. No repository added or dropped after any output is seen.

## stopping rule, and it is terminal

**There is no attempt 3.** If G0 fails again, the edge thesis has had two
mechanisms and two preregistrations and has produced nothing measurable, and the
honest record is that `SYNTHESIS_the_edge_2026_08_21.md` is a hypothesis this
project could not operationalize. Running mechanisms until one clears is how a
program fabricates a finding, and this one has documented itself approaching that
line twice today already.
