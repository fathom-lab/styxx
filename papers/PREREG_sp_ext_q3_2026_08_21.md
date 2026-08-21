# PREREG — SP-EXT Q3: the guard that was never written, finally written

**Frozen before the query was implemented and before any Q3 candidate was seen.**

Parent: `PREREG_sp_external_corpus_2026_08_21.md` (`38b8428`).

---

## the gap this closes, named in a published result before it was addressed

`RESULT_sp_ext_q2_2026_08_21.md` states the limitation plainly:

> **Q2 requires a flattering constant to be REMOVED.** A fix that adds a guard
> *above an unchanged return* matches neither query, and is invisible to this
> entire harvest.

That class is not marginal. It is arguably the **canonical** silent-pass fix: the
value was always wrong on the empty path, and the repair is to stop reaching it —
`if not samples: raise` added at the top, with the original `return 0.0` left
exactly where it was, still correct for the non-empty case.

Q1 finds it only if the author described it a particular way. Q2 cannot find it at
all, because nothing flattering is removed. **This is a query for the complement.**

## Q3, frozen

A commit qualifies as a Q3 candidate when **all three** hold:

1. **A REFUSAL GUARD IS ADDED.** The diff adds lines matching an absence test —
   `if not X`, `if len(X) == 0`, `if X is None`, `if not X:` — followed within 3
   lines by a refusal: `raise`, `return None`, `return float("nan")`, `math.nan`,
   `warn`, `warning`, `skip`.
2. **THE FUNCTION IT LANDS IN ALREADY RETURNED A FLATTERING CONSTANT.** The
   **pre-fix** body of the enclosing function contains a line matching
   `return\s+(0|0\.0|True|1\.0|"(pass|ok|valid|healthy)")`.
3. **THAT CONSTANT IS NOT REMOVED BY THE COMMIT.** No removed line in the diff
   matches the pattern in (2).

Requirement 3 makes Q3 the **strict complement of Q2**: a commit satisfying both
is a Q2 candidate and is excluded here, so the two pools do not overlap and the
yields are additive.

Requirement 2 is what keeps this from being *"every commit that added
validation"*, which is an enormous and mostly irrelevant class. The guard has to
have been added to a function that **was** returning something flattering.

## what I expect to go wrong, said in advance

Adding validation is one of the most common things a commit does. Even with
requirement 2, **precision is likely to be worse than Q2's 7.5%**, because a
function can hold a `return True` for reasons unrelated to the guard being added.
That prediction is recorded here so that a low accept rate is a confirmed
expectation rather than a disappointment reinterpreted afterwards.

**If Q3's accept rate comes in below Q2's, that is the finding**, and it says the
harvest cannot reach this class either — not that the class is rare.

## gates

G1, G2, G3, G5 from the parent prereg apply unchanged, computed over Q3's pool and
reported separately from Q1's and Q2's. Additionally:

**G6 — POOL SIZE.** If Q3 returns more than 200 candidates, a random sample of 100
is drawn with `seed = 20260821`, disclosed, and the undrawn remainder is reported
as UNADJUDICATED — never as rejected, never as absent.

**G7 — COMPLEMENT CHECK.** The overlap between the Q2 and Q3 pools is computed and
published. It should be zero by construction; **if it is not, requirement 3 is
implemented wrongly and the run is void**, because a query that double-counts
inflates every rate derived from it.

## adjudication

Unchanged: three reviewers per candidate, distinct lenses, each prompted to
REJECT, uncertainty resolving to REJECT, every verdict published with its
rationale, `module_path` and `defect_line` taken from git output the reviewer
actually ran.

The R2 lens carries the clause added after an overturn earlier today: **a value in
the alarming direction is also a reject** — failing closed is not a silent pass.

## what Q3 still cannot reach

A silent-pass fix that neither removes a flattering constant nor adds a
recognisable guard — one that restructures the function, or moves the computation
behind a new abstraction — is invisible to Q1, Q2 and Q3 alike.

**SP-EXT remains a lower bound on incidence under every query, and is never quoted
as a rate.**
