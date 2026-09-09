# The constraint census: this lab's own published verdict is a first claim, and its census is zero

2026-09-09. Module: `styxx/v8/constraint.py`. Receipts: `census.py`, `output.txt`, `census.json`
in this directory. Tests: `tests/test_v8_constraint.py`.

## Why this quantity is owed

`papers/v8/THE_BOUNDARY_2026_09_09.md` opened with a partition of surviving defects: a class
reachable by a check over logged bytes, and a class "reachable by nothing" — fields whose forged
value is byte-indistinguishable from the honest one. That second class had four members. All four
were reclassified as reachable over the course of one day, and each of the four errors was the
same error: the argument examined one certificate instead of the log the certificate sits in.
Another logged cert already carried information a forged value would contradict, and nothing
compared them.

What is left is not a list of unreachable fields. It is one unreachable act: **the first claim
about anything**. A subject nobody has measured has no prior logged cert to contradict, so every
consistency check on it is vacuous whatever it reports. Consistency accrues; it cannot be
bootstrapped.

That makes one number owed beside every verdict — *how much prior logged material could have
contradicted this claim* — and it was computed nowhere. `styxx/v8/constraint.py` computes it.

> **NARROWED THE SAME DAY.** A sixth adversarial pass showed the residue is smaller than *the first
> claim about anything* and larger in a way that matters more: a cross-certificate comparison
> constrains a claim only at the factor levels the log already holds, the issuer declares the
> levels, and the space of levels is unbounded. So the unreachable case is **a first claim in any
> (subject, factor level) cell**, and a forger widens a floor simply by declaring a level nobody has
> measured — demonstrated on these bytes, flipping this lab's own published verdict from
> `exceeds_floor` to `same`. The idea that a growing record closes in on a liar does not survive it:
> the record grows and so does the empty space in it. See `THE_BOUNDARY_2026_09_09.md`, fifth
> correction.

## What the module does

`census(cert, prior)` returns, per cross-cert predicate, the count of prior entries that predicate
could have run against. It is a census, not a check. It never says a cert is right or wrong; it
says how much a check would have had to work with. The five predicates are the ones this system
has or should have:

| predicate | what it asks | status |
|---|---|---|
| `floor_agreement` | across entries sharing a `weights_sha256` and a battery, the floor's distances and the per-run channel values must be consistent | THE_BOUNDARY member 1 |
| `snapshot_agreement` | one `(hf_repo, revision)` names one set of Appendix A.2 content hashes, and one set names one `(hf_repo, revision)` | THE_BOUNDARY member 3, named there as not implemented |
| `schedule` | the runs' factor assignments are the logged noise plan's schedule, in order | THE_BOUNDARY member 2; the plan carries no schedule field today |
| `determinism` | a run at a batch level this subject has already been logged at must reproduce the recorded channel values | falls out of the same bytes as member 1 |
| `issuer_history` | how many entries this issuer key signed before | constrains a party only against itself |

Four columns are reported per predicate. `matched` is prior entries in the predicate's scope.
`own` is the subset that is the cert's own material — its `refs`, its `recipe.battery`, its floor's
`plan` and `runs`. `available` is `matched - own`. `usable` is the available entries that actually
carry the fields the predicate reads. `independent` is the usable entries signed by a **different
issuer key**.

**Own material is excluded, and that exclusion is the whole result.** A floor is not corroborated
by the four runs it consumes; those runs are its input. `Log.previous_comparable` does not make
this exclusion, and THE_BOUNDARY records what that costs on this very log: the lookup for the
canonical fingerprint returns entry 5, which is that canonical's own floor run 4.

## The numbers, read from `papers/v8/first_verdict_2026_09_09/log`

Seven entries. One issuer key, `ed25519:Lm5ELynReA3UwDWWIfW90W6JrD18Y3VWtwCijoZuUzA`, read from
`log/keys/issuers.json`. Full transcript in `output.txt`; machine-readable in `census.json`.

| idx | type | prior | own material | constraining | independent | verdict |
|---:|---|---:|---:|---:|---:|---|
| 0 | battery | 0 | 0 | **0** | 0 | unconstrained |
| 1 | prereg | 1 | 1 | **0** | 0 | unconstrained |
| 2 | fingerprint | 2 | 2 | **0** | 0 | unconstrained |
| 3 | fingerprint | 3 | 2 | 1 | 0 | constrained |
| 4 | fingerprint | 4 | 2 | 2 | 0 | constrained |
| 5 | fingerprint | 5 | 2 | 3 | 0 | constrained |
| 6 | fingerprint (the floor) | 6 | 6 | **0** | 0 | unconstrained |

Constraining entries in index order: `[0, 0, 0, 1, 2, 3, 0]`.

## The headline number

**Entry 6 is the verdict this artifact exists to publish** — the canonical fingerprint and the
noise floor of `first_verdict_2026_09_09`. Six entries precede it. Its `refs` name six ids, read
from `log/entries/000000/00000006.json`: the battery (entry 0), the noise plan (entry 1), and the
four floor runs (entries 2, 3, 4, 5).

**All six prior entries are its own material. Not one prior entry is left for any predicate to
compare it against. The census is 0 on all five.**

Every cross-cert predicate returns vacuous on this lab's flagship certificate. Not "agrees", not
"disagrees" — nothing to compare. The predicates that emptied class two this morning, run against
the verdict published this morning, reach it on nothing.

The shape is worth reading rather than only the endpoint. Entries 3, 4 and 5 do not name one
another, so each is constrained by the runs logged below it and the count rises 1, 2, 3. Then the
certificate that aggregates all of them consumes every one of them and the count returns to 0.
Consistency accrued across four runs, and the claim built on top of it started over.

`independent` is 0 at every index, because this log has one issuer key. On these bytes a party is
compared only against itself anywhere in the log, which is the weakest form of the quantity and is
why the column is reported separately rather than folded into the total.

## The schedule predicate, measured rather than argued

The member-2 finding is now a number the census prints rather than a paragraph. The plan on file
(entry 1) fixes the factors, their levels and the run count, and carries none of `schedule`,
`assignments`, `runs_spec` or `reference`. So `plans_on_file = 1`, `plans_carrying_a_schedule = 0`,
and the census names the reason in `blocked_by` on every entry that rests on that plan rather than
counting the plan as constraint. The count of run schedules that plan admits — 57,600 — is
reported by `papers/v8/class_two_empty_2026_09_09/member2_demo.py`; it was not re-derived here.

## What this does not reach

* **It is not evidence of correctness.** A large census counts prior entries in scope; it says
  nothing about whether they agree. A single issuer who has been consistently wrong from the first
  entry yields a growing census and constrains nothing a stranger cares about. That is what the
  `independent` column exists to expose, and on this log it is 0 everywhere.
* **It does not reach suppression.** A party who commits to a favourable schedule before running
  is contradicted by nothing, so a schedule predicate would pass. That residue is inherited from
  THE_BOUNDARY's member-3 caveat unchanged.
* **It verifies nothing.** `read_log_entries` reads bytes and parses JSON. It checks no signature,
  no cert id, no Merkle root, no tree head. The census is a disclosure computed alongside
  verification, never a substitute for it. No signature on the published log was checked in
  producing any number in this document.
* **The predicate list is this module's own claim about what predicates exist.** Two of the five
  are implemented in `log.py` today and three are not; a predicate nobody has written constrains
  nothing regardless of what the census counts as available for it.

## Tests

`tests/test_v8_constraint.py`, 33 tests, all passing (`python -m pytest tests/test_v8_constraint.py`,
run 2026-09-09). The suite pins both halves: that the number is real when history exists — a
sibling run on the same subject and battery that a cert does not name is counted — and that it is
zero when the history is the cert's own input, asserted directly against the published bytes in
`test_the_published_floor_is_a_first_claim`. The published profile `[0, 0, 0, 1, 2, 3, 0]` is
pinned, as is `plans_carrying_a_schedule = 0`.

The full v8 suite was not run: four other agents were editing `log.py`, `cert.py`, `verify.py`,
`cli.py` and `conformance/v8/` while this was written, and a suite result taken mid-edit would
have been a number about their working tree rather than about this module.

## What is owed

1. **Nothing prints this beside a verdict yet.** `verify` and the `cli` verdict path should carry
   `constraint.disclosure_line(...)` on their output. Both files are owned by other agents in this
   worktree and were not touched.
2. **The noise plan should carry its schedule.** Until it does, the member-2 predicate has nothing
   to compare and the census will keep reporting `plans_carrying_a_schedule = 0` — which is the
   honest report, and also a standing defect.
3. **`Log.previous_comparable` should exclude the appending cert's own floor runs**, matching the
   exclusion this module makes. It is in `log.py` and was not touched.
4. **A second issuer is the only thing that moves the `independent` column off 0.** No amount of
   further logging by this key will do it. That is the same conclusion THE_BOUNDARY reaches about
   challenges, arrived at from a different direction and now with a number attached.
