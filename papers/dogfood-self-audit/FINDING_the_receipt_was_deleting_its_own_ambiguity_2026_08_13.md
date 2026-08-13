# FINDING — the auditor was discarding 62% of the receipt, then reporting the survivors as provenance

Fathom Lab · 2026-08-13 · third pass · receipts: `context_resolution_validation.json`,
`match_ambiguity.json` · fix in `styxx/claim_audit.py`, pinned by
`tests/test_claim_audit_chance_floor.py`.

## How this was found

The previous commit shipped ambiguity *disclosure* and closed with a stated limit: disclosure is
not resolution, and path-aware grounding remains undone. This pass went after that limit.

Built a resolver that scores each candidate source path against the words surrounding the claim.
Then — before believing it — built a ground-truth fixture: a receipt with deliberately colliding
values across semantically distinct cells, and sentences whose intended path is known by
construction.

**The validation failed immediately, and not in the way I expected.** Every case reported exactly
**one** candidate, against a receipt built specifically to make collisions. Accuracy equalled the
dict-order baseline exactly: `RESOLVER_IS_DECORATION`.

The resolver was fine. The receipt loader was not.

## The defect

`_flatten` recorded `value -> path` with `setdefault`. **Value-keyed, first-writer-wins.** Every
repeated value in a receipt was silently discarded, keeping only whichever path was visited
first.

On the real c6 receipts: **425 numeric leaves, 163 retained. 262 leaves — 62% — were invisible to
the auditor.** Rates repeat constantly (0.000 and 1.000 above all), so the losses were
concentrated exactly where claims cluster.

The consequence is worse than lost coverage. When a claim grounded to a path, that path was
frequently an *arbitrary survivor of a collision that had already been deleted*. The tool could
not report ambiguity because the evidence of ambiguity was destroyed upstream of the check.
**Value-keyed dedupe made every receipt look unambiguous by construction.**

This is the same failure family as the two before it, at a lower layer: the previous pass
measured 20% ambiguity and I reported that as the honest number. It was an undercount produced by
a defect I had not yet found. **A measurement is only as honest as the data structure underneath
it, and I had audited the judge without auditing the loader.**

## What the numbers actually are

| quantity | before | after |
|---|---|---|
| receipt paths visible to the auditor | 163 | **426** |
| grounded claims matching >1 path (my own prereg) | 14/64 | **46/64** |
| of those, resolved by surrounding text | — | 26 |
| still arbitrary, and now labelled so | — | **20** |

Two thirds of my "grounded" claims were never uniquely pinned. That was true yesterday; it is
merely *visible* now.

## Does context resolution earn its keep?

Ground truth by construction, measured against a dict-order baseline:

| fixture | dict-order | context | lift |
|---|---|---|---|
| EASY — sentences reuse the receipt's key names | 0.417 | 1.000 | +0.583 |
| **HARD — paraphrased prose, key names absent** | **0.500** | **0.750** | **+0.250** |

**The HARD number is the one to quote.** The EASY fixture scoring 1.000 was pre-registered as a
sign the fixture is too easy — sentences reusing exact key names is a rigged test — and it is kept
only as a floor.

The two HARD failures are the important part: both were labelled **`arbitrary`**, not resolved
incorrectly. The resolver requires a strict win over the runner-up, so when the words carry no
disambiguating signal it declines rather than inventing provenance. **A resolver that fails loudly
is worth more than one that guesses quietly.**

## Shipped

- `_flatten` retains **every** path per value (`value -> [paths]`).
- `_candidates` returns all colliding paths across the whole receipt.
- `_resolve_by_context` — token overlap between the claim's ±90-character context and candidate
  path names, with a stoplist so ubiquitous words ("cells", "result", "value") cannot carry a
  match, and a strict-win requirement.
- `ClaimNumber.context` / `.resolved_by` / `.context_score`;
  `GroundingReport.n_context_resolved` / `.n_arbitrary`.
- `summary()` now states the split: *"46/64 grounded claims match more than one source path — 26
  resolved by surrounding text, 20 still arbitrary (dict order)."*
- 3 new tests: duplicate-value retention (regression for this defect), resolver correctness plus
  its refusal case, and an accounting identity (`n_ambiguous == n_context_resolved + n_arbitrary`)
  so no claim can hide in an unlabelled third bucket.

## Stated limits

- **0.750 on paraphrased prose is not 1.000.** A quarter of ambiguous claims still resolve
  arbitrarily on realistic text, and the tool says so per-claim rather than in aggregate only.
- **Token overlap is not understanding.** It matches key *names*; a receipt with opaque keys
  (`c1`, `x7`) gives the resolver nothing, and it will correctly fall back to `arbitrary`.
- **A uniquely-pinned match can still be the wrong cell** if the receipt genuinely holds that
  value once and the author meant something else entirely. Grounding remains a value check with
  a provenance hint, not a semantic proof.
- The HARD fixture is 8 cases, authored by me. Small, and subject to the P1 lesson: an author's
  own battery encodes the author's blind spots on both sides.

*Three passes on the same tool. The first found the headline was uninterpretable, the second found
a false positive inside it, and the third found that the data the first two ran on was 62%
missing. Each honest number I published was refuted by the next layer down — which is the argument
for continuing to look after the result already looks good.*
