# RECON — structure separates where vocabulary did not

Fathom Lab · 2026-08-28 · **RECON. No preregistration was frozen and none should be yet.**
Receipt: `oath_structural_obligation_census.json`. Follows
`RECON_obligation_repair_is_not_lexical_2026_08_27.md`.

Yesterday's reconnaissance killed the cheap repair: widening the trigger vocabulary catches one
missed claim in eighty-five when the word list is fitted on one set of documents and scored on
another. Its conclusion was that a trigger word list *is* a marker standing in for "this sentence
asserts a measurement", and that widening a marker cannot repair a predicate whose defect is that
it is a marker.

This is the alternative, scored. **It works**, with the reservations below, and the census was
built so that a null result would have been just as visible.

## The candidates, written before the data was consulted

Each rule comes from a general argument about how a measurement differs *typographically* from a
configuration value — not from looking at which tokens the panel called claims. Two deliberately
**lexical** rules are included as a control on the census's own thesis: small fixed English word
lists, written a priori. If structure generalises and vocabulary does not, those two should fail
while the structural ones hold. If everything fails together, the story is about held-out
generalisation and not about structure at all.

## What separates

Ground truth is the majority verdict of three blind seats over every abstained token in both
2026-08-27 panels: 85 missed claims against 127 correct abstentions. **The base rate of claims
among abstained tokens is `0.4009`** — that is what a rule has to beat to carry information.

| rule | recall | precision | fold 0 | fold 1 |
|---|---|---|---|---|
| `precision_and_outside_code` | 0.3765 | **0.8** | 0.6 | 0.8667 |
| `precision_2dp` | 0.3765 | **0.7805** | 0.5455 | 0.8667 |
| `table_not_first_cell` | 0.5412 | **0.6765** | 0.75 | 0.6667 |
| `outside_code_span` | 0.9882 | 0.4308 | 0.3387 | 0.4737 |
| `markdown_emphasis` | 0.0471 | 1.0 | — | 1.0 |
| **`LEXICAL_reporting_verb`** | 0.0235 | **0.4** | 0.0 | 1.0 |
| **`LEXICAL_comparative`** | 0.0588 | 0.5556 | 0.5 | 0.6 |
| null: obligate everything | 1.0 | 0.4009 | — | — |

Read the two lexical rows first, because they are the control. `reporting_verb` lands at `0.4` —
**the base rate exactly**, which is what carrying no information looks like — and its two folds
are `0.0` and `1.0`, which is what n=2 looks like. `comparative` is barely above base rate on nine
tokens. The a-priori word lists fail, as the previous RECON predicted they would.

The structural rules do not. Decimal precision reaches `0.78` and, conjoined with "not inside a
code span", `0.80` — twice the base rate. Table position reaches `0.68` at the highest usable
recall of any rule, and it is the most **fold-stable** thing here: `0.75` and `0.6667`, which is
the shape of a rule that is reading something real rather than something local.

**These rules carry no fitted parameters.** That is the structural difference from a word list and
the reason the folds mean what they appear to mean: there is no training half, so nothing can be
memorised. The two folds are a stability check, not a train/test split, and they are reported as
such.

## What it would do at scale

Projected over all `5909` abstained tokens in both corpora, multiplying the corpus-wide fire count
by the measured precision:

| rule | fires | share of abstentions | ≈ claims recovered | ≈ non-claims obligated |
|---|---|---|---|---|
| `precision_and_outside_code` | 1120 | 0.1895 | 896 | 224 |
| `precision_2dp` | 1167 | 0.1975 | 911 | 256 |
| `table_not_first_cell` | 1727 | 0.2923 | 1168 | 559 |

These are order-of-magnitude estimates, not measurements. They multiply a corpus-wide count by a
precision measured over the 212 addressable adjudicated tokens, and the fold-to-fold swing is
their honest error bar. The
adjudicated abstentions were drawn as uniform samples of the abstained pools, which is what makes
the projection defensible at all; that it pools two corpora with different composition is a
weakness of it.

**`markdown_emphasis` must not be read from that table.** Its `1.0` precision rests on four
catches and zero costs, and the projection of "101 claims, 0 non-claims" is arithmetic on a
sample too small to carry it. It is reported because suppressing it would be worse, and it is
named here so nobody quotes it.

## What this does not show, and it is the important part

**Nothing here measures what these rules do to tokens the verifier currently obligates.** The
census scores them only against adjudicated *abstentions* — tokens already declined. As an added
trigger a rule can only reach that population, so the accounting is sound for the repair shape
proposed; but every previous widening in this lane died on interaction effects, and "no
interaction is possible" is an argument, not a measurement.

**Precision of `0.80` means one newly obligated token in five is not a claim.** At the projected
scale that is roughly 224 new obligations on non-claims, each of which becomes either a false
accusation or — worse, per the internal RESULT — a false verification. A repair that halves the
miss rate while adding hundreds of false attestations may not be a repair.

**Recall is modest.** The best usable rule recovers about half the misses, the most precise about
two in five. Even a successful cycle here leaves most of the coverage gap open.

**The ground truth is three correlated LLM seats over 212 tokens.** Every number above inherits
that ceiling, and a human re-adjudication of the retained packets remains the largest outstanding
check on all of it.

## What is owed

This licenses a **preregistration**, not a clause. What such a cycle would have to freeze before
it runs:

1. a bar on the *cost* side, not just the benefit side, since that is where every previous
   widening died;
2. a `styxx-discriminates` check against the null rule **with `share_of_control` read**, because a
   rule that does nothing also has a low cost;
3. a held-out split by document, stated before the numbers are seen;
4. and a measurement of what the shipped verifier does to the newly obligated population, not a
   projection of it.

The candidate to beat is `precision_and_outside_code`, on the numbers above. It is not proposed
here, and this document freezes nothing.

---

*Two decimal places and a backtick carry more information about whether a sentence is making a
claim than every measurement word this laboratory could think of.*
