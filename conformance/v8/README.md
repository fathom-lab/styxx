# The styxx.v8 conformance set

Built to `styxx/v8/INTERFACES_layer2.md` section 10, under the lab's calibration rule: **an
agreement number is worth nothing without a measurement of the instrument's detection power.**
Every count lives in `index.json` and `mutation_coverage.json`; none is written here.

## What is here

| file | what it is |
|---|---|
| `index.json` | one digest over everything (`set_sha256`), the family files by sha256, the blob store by sha256, the entrypoint table, the honest remainder, and provenance outside the digest |
| `vectors/<family>.json` | one file per family, vectors sorted by id |
| `blobs.json` | every value a vector names, keyed by sha256, base64 of its RFC 8785 canonical bytes |
| `recorder.py` | the pytest plugin that turns every call into a `styxx.v8` entrypoint into a record while the sources run |
| `gen_vectors.py` | runs the sources under the recorder, folds the records into vectors, replays every one, classifies every drop, refuses a moved core unless the operator retires it by name with a reason, writes the set |
| `replay.py` | replays one vector through `styxx.v8`; the reference for what a second implementation does, entrypoint by entrypoint |
| `mutation_catalogue.json` | localised edits to the implementation, each with the behaviour it breaks, plus semantics-preserving controls |
| `mutation_coverage.py` | applies each edit to a scratch copy, monkeypatches it over the real module, replays the committed set, and reports what the set saw |
| `mutation_coverage.json` | the run: the gates, the per-region breakdown, and the miss list in full |

The generated files are never hand-edited. `tests/test_v8_conformance.py` replays the committed set
with nothing skipped, and breaks the set and the implementation on purpose to show that the replay
would notice.

## What a vector is

One recorded call into one `styxx.v8` entrypoint, its arguments as data and the outcome the
implementation produced, addressed by
`id = sha256(canonical_bytes({"family": family, "inputs": inputs}))`.

The families and what they hold:

* **cert** — a cert's bytes, and what `cert.check` said about them.
* **merkle** — leaf hashes, and the RFC 6962 root, inclusion proof, consistency proof or
  verification they produce.
* **sth** — an STH, a proof and a log public key, and the `(ok, reason)` a log verifier returns.
* **distance** — two sides of an Appendix B channel, and the number.
* **floor** — the runs of a noise plan and a channel, and the floor block.
* **decide** — a distance, a floor, a confirmation and the coverage and skew flags, and the
  section 5.2 verdict.
* **exit** — two fingerprint certs and a resolver, and the verdict, the exit code and a digest over
  the result body `verify --diff` produced.

A `cert.check` outcome pins the reason KINDS a cert produces, never a reason's tail: the tail
carries interpreter text and file paths, and a second implementation owes the taxonomy, not the
prose. A call that raises is an outcome too, and pins the exception type alone.

`verify.diff`'s resolver is the one argument that is not data. It is carried as its kind — no
resolver, a resolver that answers ids, or an argument the dispatch refuses — plus, for the second
of those, the id-to-cert answers the call actually received.

Numbers cross the set through RFC 8785, which renders `1.0` as `1`, so both sides are compared by
canonical bytes rather than by Python equality. No entrypoint in the set distinguishes the two.

## How to consume it

1. Read `index.json`; recompute `set_sha256 = sha256(canonical_bytes(index minus set_sha256 minus
   provenance))` and compare.
2. For each family file, compare its bytes' sha256 to `index.families[name].sha256`; the same for
   `blobs.json`.
3. For each vector, resolve every blob it names, check each hashes to its key, and re-derive `id`
   from its own family and inputs.
4. Call the entrypoint the vector names on the arguments it carries, and compare the outcome to
   `expect` by canonical bytes.

`python conformance/v8/replay.py` does steps 3 and 4 through `styxx.v8` and prints pass and fail
counts per family; `--id <hex>` replays one vector and prints both sides.

## How to regenerate it

```
python conformance/v8/gen_vectors.py            # regenerate in place; refuses a moved core
python conformance/v8/gen_vectors.py --check    # regenerate in memory; exit 1 if set_sha256 differs
```

The generator loads the committed set before it writes. A vector already in the set whose outcome
differs from what the run produced is a moved core: the generator prints its id, both outcomes and
the tests that produced it, and exits without writing anything. A moved core is a finding about
`styxx.v8`, never a reason to rewrite the set. The set digest changes only in a commit that says
why.

**Every drop is classified, because "dropped" was one word for two events.** An id the run no
longer produces is replayed against the tree that dropped it:

* it still reproduces → **input churn**. The sources stopped making that call and the address left
  with it; nothing about `styxx.v8` moved. An ordinary retirement, no permission asked.
* it does not reproduce → **a behaviour change wearing a drop's clothes**, and it is refused
  exactly like a moved core. The address changed only because the repair that moved the answer
  also touched the fixture the address is computed from. This is not hypothetical: on 2026-09-09
  five of the six answers that moved left through this door, and a generator that printed one word
  for both would have lost all five in a list of drops.
* it cannot be replayed at all → **undiagnosed**, and it is refused rather than assumed benign.

### Retiring a moved core, with a reason

A behaviour change can be deliberate — a repair closes an attack, and a vector that pinned the
acceptance the attack used is now a record of a defect. Deciding that is the operator's, so it is
resolved in the tool rather than by hand or by an override:

```
python conformance/v8/gen_vectors.py \
  --retire <id> --reason "ENV-ABSENT: subject.environment became required ..." \
  --retire <id> --reason "A-NORUNS: body.runs became required on a noise plan ..."
```

The shape of the path is what keeps it from becoming an override:

* **The ids are named.** There is no flag meaning "retire whatever moved". An operator who cannot
  name the address has not looked at it.
* **A reason is required, per id.** `--retire` and `--reason` are positional pairs; an id with no
  reason, a reason with no id, or an id given twice is refused before the sources are even run.
* **Only an address that actually moved can be retired.** Naming an input-churn drop is refused —
  it needs no reason and cannot carry one — as is naming an address that did not move at all. So
  the reason recorded beside a retired core is always a reason about `styxx.v8`, never about a
  test that was rewritten.
* **A moved core nobody named is still a refusal**, with the same message and the same exit code
  as before this path existed.
* **The retirement is recorded, not just permitted.** `index.provenance.retired.with_reason`
  carries the id, the old outcome, the new one, the sources that produced it, the diagnosis and
  the operator's reason. Ordinary retirements sit in `retired.input_churn` in the same block,
  carrying the address alone: the two are never merged. Both halves are ledgers — every run
  carries the previous one forward, or the record would be erased by the next regeneration — and
  an address the sources start producing again leaves `input_churn`, while nothing leaves
  `with_reason`. The whole block is provenance, outside `set_sha256`: recording a retirement does
  not change the identity of the set.

The limit, stated rather than papered over: the tool checks that a reason is **present**, never
that it is **true**. It can prove which address moved, from what to what, and that a human named it
before it was written; it cannot tell a repair from a regression, and a wrong reason recorded here
is a wrong reason recorded durably. What it removes is the silent case — an answer that changed
with nobody's name on it — and that is all it removes.

The generator has no clock and reads no git. The set is a function of the sources and of the
implementation, so two runs on two machines produce the same bytes and `--check` is a real
comparison. It refuses to record from a run in which a source test failed: the vectors are the
outcomes the tests chose, and a failing run has none to offer.

## What moved on 2026-09-09, and why

Two rules were repaired in `styxx.v8` on 2026-09-09, both closing an attack an adversary had
demonstrated:

* **(a) A challenge must declare what produced its distances.** `schema/challenge.json` now
  requires `body.subject` and `body.recipe_core`. Attack: C3 of
  `papers/v8/challenge_and_attack_2026_09_09` — every challenge the run produced carried
  `subject: {}` and `recipe: {}`, and a reader could not tell from the cert what was run.
* **(b) A ref role is named once.** `cert.duplicate_roles` refuses a repeated role except
  `result`, `robustness` and `run`. Attack: C-DUPREF, same paper — duplicate `own` refs on a
  challenge were resolved last-wins by the log while a reader walking `refs` in order saw the
  earlier one.

Five committed vectors stopped reproducing under the repaired rules. All five are `cert.check`,
all five were addresses the set no longer produces once the sources build certs the repaired rules
accept, and each is named here with what answers for it now:

| retired | pinned | repair | answered now by |
|---|---|---|---|
| `41a682e2…4cb5cfbc` | `ok: true` on a challenge with no `body.subject` and no `body.recipe_core` | (a) | `7b803597…05a3b29e`, `ok: true` |
| `b8777ac4…ab637ffe` | `ok: true` on the same shape, built by `verify.challenge_body` | (a) | `0166328d…8f5549ac`, `ok: true` |
| `4bbbb3cb…6a8eafb0` | `reason_kinds: ["schema"]` where the repaired schema reports three | (a) | `b3c888c1…d5eb628a`, `reason_kinds: ["schema"]` |
| `6e1391de…3663fe281b` | `reason_kinds: ["schema"]` where the repaired schema reports three | (a) | `63c69306…620d492d`, `reason_kinds: ["schema"]` |
| `a6f0718b…6e6228dd` | `ok: true` on a fingerprint naming the `battery` role twice | (b) | `24e41a00…56a26891`, `ok: true` — an address the set already held |

All five carried an outcome that is now wrong. Three of them pinned an
ACCEPTANCE of a cert the repaired rules refuse, which is the finding written up in
`papers/v8/conformance_pinned_a_defect_2026_09_09.md`: a vector that pins `ok: true` freezes
whatever the implementation accepted on the day it was recorded, and fails when that acceptance is
withdrawn.

`a6f0718b…` is a fifth case and a different one. Its source built the fingerprint's battery ref
twice, so removing the duplicate makes the call identical to the plain fingerprint round trip; the
address `24e41a00…` already existed and now names that source too. The set is one address smaller
for it, and nothing about a single `battery` ref went unrecorded.

The rest of the ids that moved in this regeneration moved because a source now calls with different
arguments, not because an answer changed. The test that separates the two: replay the retired
vector against the tree that retired it. 84 of the 89 retired ids still reproduce, and the five
above are the ones that do not. `papers/v8/conformance_pinned_a_defect_2026_09_09.md` carries the
full 89-id ledger.

## The regeneration attempted later on 2026-09-09, and why it did not write

Four further repairs landed in `styxx.v8` the same day and the committed set stopped replaying:
six vectors of it. The regeneration was attempted, the diagnostic below was run first, and the
generator **refused**. Nothing under `conformance/v8/` was rewritten. What follows is the record of
what the run found, so the next attempt starts from it rather than from the terminal output.

The tree it describes, by the digests `index.provenance` would have carried:

```
cert.py 35faded729989da1   distances.py fcf747fb4e71d201   floor.py c40a8b80a928938d
jcs.py  85f25506275c2fc3   keys.py      ee3e8994c7a1bef5   log.py   4ef37b11a387611d
merkle.py 0266425b70d9c496 verify.py    5db73c15d2fe15c5
```

The §5.5 disclosure (`floor.baseline_gap`, `Log.baseline_gap`, `verify`'s `baseline_gap` block)
landed between the first run of the diagnostic and this one. The split below was run twice, once
before it and once after, and is byte-identical both times: the disclosure adds no vector, retires
none and moves none, because no source builds a fingerprint whose replaced baseline a resolver can
reach. Only the digests above moved.

### The split, run before anything was regenerated

152 committed addresses this run no longer produces; 648 new addresses; **1 moved core**. Every
retired id was replayed against the tree that retired it, which is the test that separates a
rewritten source from a repaired defect:

* **147 still reproduce** — the committed answer is still this tree's answer for that input, and
  the id retired because a source now calls with different arguments. Input churn, nothing owed.
  By entrypoint: 105 `floor.pairwise`, 40 `cert.check`, 5 `log.verify_sth`, 1 `merkle.root`,
  1 `verify.diff`, minus the five below.
* **5 do not reproduce** — the answer moved. Each is named here with the repair that moved it, and
  no sixth cause was found:

| retired | pinned | now | repair |
|---|---|---|---|
| `4d7cf5c4…3834479f` | `ok: true` on an alias-subject fingerprint | `schema[fingerprint]: subject: 'environment' is a required property` | ENV-ABSENT |
| `631c0d24…8f9eb75c` | `ok: true`, the same shape from `test_v8_verify.py` | the same reason | ENV-ABSENT |
| `745af6cc…c8244eb` | `reason_kinds: ["schema"]` on a sealed prereg with no commitment | two reasons: the commitment, plus `body: 'runs' is a required property` | A-NORUNS |
| `7f980b9f…800aea43` | `ok: true` on `{"kind": "noise-plan"}` | `body: 'runs' is a required property` | A-NORUNS |
| `bcc66e04…a3d3dbde` | `verify.diff` exit 3, `mismatch`, on that same prereg as B | exit 4, `invalid` — `_gate_certs` refuses B before any comparison | A-NORUNS, transitively |

**ENV-ABSENT**: `schema/fingerprint.json` now requires `subject.environment`, because a fingerprint
is the record of a run and a run happened somewhere. An alias subject carried no environment block,
so its floor's `not_covered` was empty and §5.4 read that as covering every environment there is.

**A-NORUNS**: `schema/prereg.json` now requires `runs` on kind `noise-plan`. Enforcing the minimum
only when the key happened to be present made the whole of it optional, and a plan minted with
`runs` deleted fixed no R at all, which reopened the hand-picked-subset attack.

### The blocker: one moved core, and it is the mirror case

```
b0cd1d4eb73ba0de04ecfc7e7a46afc9802e861addc385040dbed890e7e9c3d6  cert.check
  was      {"ok": true,  "reason_kinds": [],         "type": "prereg"}
  this run {"ok": false, "reason_kinds": ["schema"], "type": "prereg"}   # body: 'runs' is required
  the source that recorded it   test_v8_log.py::test_a_plan_that_fixes_no_r_is_not_checked_for_completeness
  the source that records it now test_v8_log.py::test_a_noise_plan_that_names_no_r_does_not_check_out
```

The input is a noise-plan prereg with no `runs`. The committed vector pins that `styxx.v8` ACCEPTS
it — which is the A-NORUNS hole itself, recorded, addressed and replayed. A-NORUNS closed the hole,
so the answer at that address moved, and the generator refuses to write on a moved core.

This is the case `papers/v8/conformance_pinned_a_defect_2026_09_09.md` described this morning and
said had not yet fired:

> That is right when the committed outcome was right. When the committed outcome was the defect,
> the same rule stands in the way of the repair, and the rule cannot tell the two cases apart,
> because the information that would tell them apart is not in the set.

It has now fired. The five ids in the table above retired as DROPS, because their sources changed
what they call with, so their addresses changed and the refusal never saw them. This one did not:
the test that recorded the acceptance was deleted by the repair, and its replacement calls
`cert.check` on the same bytes, so the address survived and only the answer moved.

Nothing here works around it. The set was not rewritten, no vector was hand-edited, no exception
list was added, and no schema was weakened. **Retiring an acceptance vector because the acceptance
was the defect is a decision about what the set was for, not a predicate over bytes, and the
generator has no way to be told it — so it is owed to the operator, and one decision unblocks the
whole regeneration**: with `b0cd1d4e…` retired there are no other moved cores, and the rest is 152
drops and 648 additions.

### Two consequences the run demonstrated

* **`mutation_coverage.py` is blocked behind the same staleness.** It replays the committed set
  unmutated first and refuses when that does not reproduce, because a detection rate measured over
  a set that already fails is a number about the baseline. Its own words: *"REFUSED: the committed
  set does not replay; nothing below would mean anything."* The miss list therefore did not move
  and could not be re-measured. It stands as `mutation_coverage.json` recorded it, and that
  receipt now describes a tree that no longer exists.
* **The drop notice is still weaker than the refusal, and the split still has to be run by hand.**
  147 to 5 here, 84 to 5 this morning. Consequence 3 of the finding paper is unchanged and owed.

## The regeneration that did write, later on 2026-09-09

The decision the section above says is owed was taken: the six answers that moved were retired by
name, each with the repair that moved it, through the new `--retire`/`--reason` path. The set was
regenerated and `conformance/v8/` was rewritten by the generator. Nothing was hand-edited, no
exception list was added, no schema was weakened, and the refusal is untouched — a run of
`gen_vectors.py` with no flags still refuses on a moved core, and the same run refuses if even one
of the six is left unnamed.

**The split this run measured**, with every drop replayed against the tree that dropped it:

* 648 addresses added.
* 149 dropped and still reproducing — input churn. By entrypoint: `floor.pairwise` 105,
  `cert.check` 36, `log.verify_sth` 5, `log.verify_inclusion` 1, `merkle.inclusion_proof` 1,
  `merkle.root` 1.
* 5 dropped and no longer reproducing, plus 1 moved core — **the six**, and no seventh cause.

Two of those churn drops are new since the split recorded above, which counted 105/36/5/1 and no
`log.verify_inclusion` or `merkle.inclusion_proof`. `styxx/v8/log.py` moved in commit `27a6529e`
between the two runs; both new drops still reproduce, so no answer moved with them. The earlier run
built two vectors more than this one for the same reason.

**What was retired, and why.** The full rows, with both outcomes and the sources, are in
`index.provenance.retired.with_reason`:

| retired | kind | reason recorded |
|---|---|---|
| `4d7cf5c4…3834479f` | behaviour-change | ENV-ABSENT — `subject.environment` became required; the retired answer accepted a fingerprint with the key absent, which turned the environment guard off by omission |
| `631c0d24…8f9eb75c` | behaviour-change | ENV-ABSENT — the same shape, recorded from `test_v8_verify.py` |
| `7f980b9f…800aea43` | behaviour-change | A-NORUNS — `body.runs` became required on a noise plan; the retired answer accepted a plan with the key absent, which let an issuer hand-pick which of its own runs composed a floor |
| `b0cd1d4e…e7e9c3d6` | **moved-core** | A-NORUNS — the same acceptance at an address that survived the repair, which is why this one blocked the generator and the other five did not |
| `745af6cc…c8244eb` | behaviour-change | A-NORUNS — the prereg now fails schema twice, for the violation the test injects and for the absent key; the single-reason outcome is retired and the cert is refused still |
| `bcc66e04…a3d3dbde` | behaviour-change | A-NORUNS reaching `verify.diff` — the B-side cert is invalid rather than merely not comparable, so the run reports exit 4 where it reported exit 3 |

Five of the six retired an **acceptance**. That is the prediction in
`papers/v8/conformance_pinned_a_defect_2026_09_09.md` holding: a vector that pins `ok: true`
freezes whatever the implementation accepted on the day it was recorded, and every later rule can
falsify it.

Both consequences the blocked run demonstrated are now closed. The drop notice is no longer weaker
than the refusal — the split is the generator's, not a hand-run script — and `mutation_coverage.py`
replays a set that reproduces again.

### The miss list did not change

The coverage was re-measured against the regenerated set and the tree as it stands. **The same five
mutants are missed, by name**: `merkle-leaf-prefix-dropped`, `keys-small-order-accepted`,
`cert-small-order-key-accepted`, `cert-noncanonical-point-accepted`, `log-duplicate-id-accepted`.
Nothing became catchable and nothing stopped being catchable; the per-region table is identical and
the detection rate is unchanged to four decimal places. What moved is how many vectors notice a
caught mutant — `floor-alpha-off-by-one` went from 296 witnesses to 773 — which is the direction
that matters: **the 648 vectors added are redundant with respect to this catalogue.** They deepen
the places the set already reached and open none of the places it does not. The key-validation gap
in particular (a small-order or non-canonical issuer key) is exactly as open as it was.

That is worth stating plainly because a set growing by half its size and a detection rate that does
not move is the same finding the lab keeps meeting: an agreement number counts vectors, and vectors
are not coverage.

### A deadlock in the receipt guard, found and narrowed

`mutation_coverage.py` refused to write while `mutation_coverage.json` was tracked by git, on the
rule that a receipt is history. `tests/test_v8_conformance.py` asserts the opposite — that the
receipt describes the modules in the tree today — so after any change to a mutable module the test
demanded a fresh receipt and the tool refused to produce one. Nothing could satisfy both.

The guard now refuses only what it was written for: a re-run over the same set and the same modules,
where writing today's number over the one that was vouched for pins a measurement instead of
testing it. When `set_sha256` or a mutable module has moved, the committed receipt already fails its
own tests, so the run measures a different object — it says which digests moved before it writes,
and prints how the miss list differs from the receipt it replaces. The replacement is never silent
and the previous bytes are in git.

## What passing means, and does not

Agreement on these vectors makes an implementation agree with `styxx.v8` on the inputs five test
files chose. It does not make it correct, and it does not mean section 10's contract is covered:
the tests were written by the builder, the weakest attacker there is, and the vectors carry only
what those tests happened to reach.

`index.unvectored` is the honest remainder, and every entry carries a note saying why it is a gap.
The two kinds of gap are kept apart on purpose. Some are unreachable by construction — `verify
--diff` runs no confirmation step, so section 5.2 cannot reach `drift` from that entrypoint at all.
Some are simply not driven: `verify.diff` can return `invalid`, and every test in the sources that
reaches that exit code drives `verify --ref` instead, which section 10 does not record. The second
kind is a real hole and is labelled as one. `cert.check`'s numeric and not-an-object reasons are a
third case: the sources reach them, and the record cannot carry the argument, because a cert
holding a NaN has no canonical bytes. Those calls are listed one by one under
`index.unvectored.skipped` with the test that made them.

Calls from the property tests are not recorded, and every property test is named. A hypothesis
example is a function of the hypothesis version and its database, so a vector recorded from one
would move the set digest on a dependency upgrade with no change to `styxx.v8`.

## What the set can actually see

`mutation_coverage.json` is the answer, and its miss list is the deliverable. Each catalogued edit
breaks one behaviour in one module; the set is replayed against the result; **caught** means an
implementation that got that behaviour wrong would fail this set, and **missed** means it would
pass. The controls are semantics-preserving edits, and a caught control voids the whole run — the
set would be detecting that a file was touched rather than that an answer changed.

Three findings from the run are worth reading before the rate:

* **The set never hashes a leaf.** Section 10 makes leaf hashes an *input* to the merkle family, so
  no vector calls `merkle.leaf_hash`, and dropping RFC 6962's `0x00` leaf prefix changes nothing
  the set looks at. A port could get the leaf prefix wrong and pass everything here.
* **The small-order issuer key rule is implemented twice.** `keys.decode_public` validates the
  point, and `cert.public_key_reason` then runs its own copy over the same bytes. Breaking either
  copy alone moves no outcome, because the other still rejects the key under the same reason
  prefix — and the vectors record reason kinds rather than reason texts, so they could not tell the
  two rejections apart even if one did. A redundancy is invisible to a set of examples by
  construction; only the mutation run says so.
* **An append-time refusal has no entrypoint.** The log family verifies STHs and proofs; nothing in
  the set calls `Log.append`, so accepting a duplicate id is not something these vectors can see.

The rate is a property of the set and of this catalogue together. A catalogue that avoided the thin
places would report a higher one and mean less, so the thin places are in it.

## What is owed

`floor.baseline_gap` and `Log.baseline_gap` (§5.5's disclosure, added 2026-09-09) are **unvectored
by construction**: neither is in `ENTRYPOINTS`, and `tests/test_v8_baseline_gap.py` is not in
`SOURCES`, so no vector reaches either. `verify.diff` sets `result_body["baseline_gap"]` only when
the cert being compared replaced a baseline the resolver can reach, and no source builds that, so
the digest over the result body is unchanged for all fourteen `exit` vectors — the disclosure is
invisible to this set. Adding a source changes what the set is a function of and is its own
decision; until it is made, an implementation can agree on every vector here and print no gap
at all.

`verify.ref` as a recorded entrypoint, which would close the `invalid` and `unavailable` exit codes
and reach the runner's own refusal; an `append` family, which would reach the refusal matrix the log
enforces on the way in; a `cert.check` outcome that pins reason texts as well as kinds, which is
what it would take to see a layered rule break; a second implementation, which is the only thing
that turns any of this from a regression suite into a conformance set; and vectors recorded from a
`styxx.v8` this lab did not write, which is the only thing that would turn the builder's own
choices into a measurement.
