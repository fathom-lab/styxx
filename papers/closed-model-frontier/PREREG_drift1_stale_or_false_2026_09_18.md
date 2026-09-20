# PREREG — DRIFT-1: when the gate calls a file count false, was it false when it was written?

Fathom Lab · 2026-09-18 · Frozen before any verdict is computed. Instrument at sha256 `9b620e00…`
and **not modified by this run** — DRIFT-1 measures the corpus, not the instrument. Follows
`RESULT_decide1_decidable_fraction_2026_09_17.md` (the instrument is silent on most of what a human
can settle) and `RESULT_scope1_ABANDONED_2026_09_18.md` (a repair whose only footprint was its own
derivation set).

## The question

BENCH-2's dataset carries 116 `files_changed_count` claims. **47 of them — 40.5% — disagree with the
final diff.** The instrument, and the oracle, and every write-up we have published, treat that
disagreement as one thing: the pull request says a number and the diff says another.

There is a second explanation none of our work has separated from the first:

> The number was **true when it was written**, and the branch moved afterwards.

An agent opens a pull request, writes "8 files changed", and then pushes three more commits in
response to review. The body is now wrong. It was never a lie. If that is where a large share of
our accusations come from, then the instrument's verdict is wrong *in kind* — not in value — and the
right output is "stale as of commit 3", not `[LIE]`.

This preregistration is the measurement that separates the two. It is not a repair, and no change to
`styxx/diffgate.py` is authorised by it.

## Method

Fully local. No network call is required to produce any number in this run.

- **Population.** All 116 `files_changed_count` rows of `bench2_dataset.jsonl`, each carrying a
  stated integer and a `diff_sha256`. The 568 diffs were re-fetched from source on 2026-09-18 and
  566 of 568 matched their published hash (`RESULT_scope1_ABANDONED_2026_09_18.md`).
- **Commit order.** `pr_commits.parquet` from the AIDev corpus (CC-BY-4.0, Zenodo
  10.5281/zenodo.16919272), taken in stored row order.
- **File sets per commit.** `pr_commit_details.parquet`, joined on `(pr_id, sha)`.
- **Prefix file set** `F(k)` = the union of `filename` over commits `0..k` in that order.
- **Classification** of a stated count `N` for a pull request with `m` commits:
  - **MATCHES_FINAL** — `N == |F(m-1)|`.
  - **STALE** — `N != |F(m-1)|` and `N == |F(k)|` for some `k < m-1`. The smallest such `k` is
    recorded.
  - **NEITHER** — `N` matches no prefix.

## The thing that could make this meaningless, stated first

**A pull request body can be edited after it is opened, and the corpus carries no edit history.** A
body that matches an early prefix is *consistent with* having been written early; it is not proof
that it was. A body could equally have been written late and simply be wrong by a number that
happens to coincide with an earlier state.

So STALE is a statement about consistency with an earlier state, not about when the text was typed,
and the result must use that wording throughout. Any sentence in the write-up claiming to know when
a body was written is a defect in the write-up.

A coincidence check is therefore part of the measurement rather than an afterthought: for every
STALE item, the number of *distinct* prefix counts is recorded, because `N` matching one of two
possible values is much weaker evidence than `N` matching one of nine.

## Predictions, committed now

1. **Of the 47 claims that disagree with the final diff, at least 40% classify as STALE.** Below
   that, the drift hypothesis does not explain our accusations and the write-up says so in its own
   headline.
2. **MATCHES_FINAL will be within ±3 of 69**, the count BENCH-2's oracle labelled SUPPORTED.
   A larger gap means the reconstruction disagrees with the cached diffs and is not measuring what
   it claims to.
3. **At least 5 items classify as NEITHER.** A three-way split with an empty third bucket would
   suggest the classifier is absorbing everything into two, which is a defect rather than a finding.

## Gates

- **G-DR1-1 (blocking, the order is real).** Stored commit order must reproduce GitHub's
  `/pulls/{n}/commits` order **sha for sha** on 10 pull requests drawn with seed 20260918 before
  any classification is run. Two were checked by hand before this freeze
  (`dynaconf/dynaconf#1330`, `polarsource/polar#7298`, four commits each, exact). Any mismatch in
  the ten and prefixes are meaningless: the run is INVALID and nothing is published from it but the
  mismatch.
- **G-DR1-2 (blocking, the reconstruction agrees with the diffs).** For ≥90% of pull requests,
  `|F(m-1)|` must equal the file count of the locally cached diff whose sha256 matches the published
  row. Pull requests failing this are **excluded and listed individually with both numbers**, not
  reinterpreted; the expected cause is a file added and later reverted, which appears in the union
  of commits and not in `base...head`. Below 90%, the run is INVALID.
- **G-DR1-3 (no near-matches).** STALE requires exact integer equality at a strictly earlier prefix.
  `N` within one of a prefix count is NEITHER. This gate exists because loosening it later would
  manufacture the finding.
- **G-DR1-4 (coincidence reported).** Every STALE item publishes the number of distinct prefix
  counts available to it. The result reports the share of STALE items whose pull request had only
  two distinct prefix counts, where a match is least informative.
- **G-DR1-5 (power).** Wilson intervals on every proportion. If the interval on the STALE share
  spans the 40% prediction, the result says in its own headline that the run does not settle it.
- **G-DR1-6 (the instrument is untouched).** `styxx/diffgate.py` is byte-identical before and after.
  DRIFT-1 ships a measurement and a paper. Any repair it motivates is a separate preregistration
  with its own held-out evidence, and — per SCOPE-1 — one whose footprint extends beyond the items
  it was derived from.

## What a passing DRIFT-1 does not license

It does not license calling the instrument correct. An accusation that says `[LIE]` about a body
that was true when written is still a wrong output, and measuring *why* it is wrong does not repair
it. It also says nothing about `only_touches`, where precision is 0.25 and coverage is 5.4%; file
counts are the one claim kind where extraction is reliable, which is exactly why the drift question
can be asked here and not there.

---

*Every accusation this instrument has ever made assumed the diff it is reading is the diff the
author was describing. Nobody has checked that assumption, including us, and it is cheap to check.*

---

## AMENDMENT A — appended 2026-09-18, before any claim was classified

G-DR1-1 was run first, as written, on the ten pull requests drawn with seed 20260918. **It fails as
written, on one item, and the failure is informative rather than fatal.** Nothing was classified
before this was appended, and the pre-amendment hash `9e5ac4e2…` stands beside the amended one.

**What the ten showed.** Stored corpus order is a prefix of GitHub's live order, sha for sha, on
**10 of 10**. Nine are exact. One is not:

| pull request | commits in corpus | commits live today | relationship |
|---|---|---|---|
| `microsoft/FluidFramework#25610` | 30 | **66** | corpus is the first 30, in order |

The corpus is a **snapshot**. That pull request gained thirty-six commits after AIDev read it. Order
is sound — which is the property prefixes need and the only property this gate was ever really
about — but "the last commit in the corpus" is not "the last commit of the pull request".

**A1 — G-DR1-1 is restated.** The gate now reads: *stored commit order must be a prefix of GitHub's
order, sha for sha, on all ten.* Measured: **10 of 10, PASS.** The count of exact matches (9) and of
strict prefixes (1) is published either way. The original wording asked for something stricter than
the measurement needs and would have voided a run whose ordering is demonstrably correct; the
weaker property is stated here rather than applied silently, and it was fixed before, not after,
seeing any classification.

**A2 — three reference points, not two.** Because of A1 the run now distinguishes:

- `|F(m-1)|` — the file set at the corpus's last commit;
- the file count of the **diff fetched from source on 2026-09-18**, which is the pull request as it
  stands today;
- the integer the body states.

`MATCHES_FINAL` is renamed **`MATCHES_CORPUS_HEAD`** throughout, because that is what it measures.
Where the cached diff and the corpus head disagree, the pull request moved after the snapshot, and
that count is reported separately rather than folded into either bucket. G-DR1-2's exclusion rule is
unchanged and now carries this second cause alongside revert-shrinkage; both are listed per item.

**A3 — the prediction that depended on the old name.** Prediction 2 said `MATCHES_FINAL` would land
within ±3 of the oracle's 69 `SUPPORTED`. It now applies to whichever of the two reference points
the oracle itself used — the fetched diff — and is unchanged in value. If the corpus head and the
fetched diff disagree on enough pull requests for this to be ambiguous, that disagreement is the
headline and the drift classification is reported beneath it.
