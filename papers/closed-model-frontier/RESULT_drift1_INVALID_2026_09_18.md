# RESULT — DRIFT-1 is INVALID: the union of a pull request's commits is not its diff

Fathom Lab · 2026-09-18 · Prereg `PREREG_drift1_stale_or_false_2026_09_18.md`, sha256 `9e5ac4e2…`
at freeze, `1f5829c7…` after Amendment A, both appended before any claim was classified. Receipts:
`drift1_classify.py`, `drift1_classification.json`, `drift1_order_sample.json`,
`drift1_corpus_order.json`. **The instrument is unchanged at sha256 `9b620e00…` and was not run.**

**G-DR1-2 failed. The reconstruction agrees with the fetched diff on 74 of 111 pull requests —
66.7% [57.5%, 74.7%] — against a preregistered floor of 90%.** DRIFT-1 is void as preregistered.

## The question it was going to answer

40.5% of the corpus's file-count claims — 47 of 116 — disagree with the diff. Every write-up we have
published treats that as one thing. DRIFT-1 asked whether some of it is a second thing: a number
that was **true when it was written**, on a branch that moved afterwards. If so, `[LIE]` is the wrong
word and "stale as of commit 3" is the right one.

## Why it is void

The method reconstructed each pull request's file set as the **union of `filename` over its commits
in order**, from AIDev's `pr_commit_details`. G-DR1-2 checked that union against the pull request's
actual diff, re-fetched from source and hash-verified. On a third of the population they disagree.

| | |
|---|---|
| pull requests in the population | **111** |
| union equals the fetched diff | **74** — 66.7% [57.5%, 74.7%] |
| union **larger** than the diff | **31** |
| union **smaller** than the diff | **7** |
| median (diff − union) | **−3 files** |

The direction is the tell. In 31 of 37 disagreements the union is **larger**, which is what happens
when a file is touched and then reverted, or when the branch is rebased: the file appears in a
commit and not in `base...head`. The seven that run the other way are pull requests that grew after
the corpus snapshot was taken — `onetimesecret/onetimesecret#1538` goes 64 → 645,
`microsoft/typespec#8139` goes 300 → 1090.

The prereg named revert-shrinkage as the expected cause and allowed 10% for it. **It is 28%.** That
is not a tolerance being exceeded by a little; it is the method being wrong about what a pull
request's diff is.

**So: the union of a pull request's per-commit file lists is not the pull request's diff, and on this
corpus it is wrong about one in three.** Anyone reaching for `pr_commit_details` to reconstruct
file scope — we did, this morning — should know that before they publish a number on it.

## The number the run produced, and why it is not a finding

On the 74 pull requests that survived, of the 22 whose stated count disagrees with the head,
**11 — 50.0%** matched an earlier commit prefix exactly. Per claim rather than per pull request:
13 of 25, **52.0%**, 95% Wilson **[33.5%, 70.0%]**. Three of the 13 rest on a pull request with only
two distinct prefix counts, where a match is nearly meaningless.

**This number cannot be reported as a finding and is not one.** Its denominator is precisely the
subset where the union equals the diff — pull requests with no reverts and no post-snapshot growth
— which is a biased sample of exactly the wrong kind for a question about branches moving. And the
interval spans the preregistered 40% prediction, so even taken at face value it would not have
settled the question (G-DR1-5).

It is printed here because the prereg says every number gets published whatever it is, and because
hiding a 52% that happens to support our hypothesis would be worse than printing it with the reason
it proves nothing.

What the surviving items look like, for whoever runs this properly:

| pull request | stated | at head | true at commit | distinct prefix counts |
|---|---|---|---|---|
| `Azure/azure-sdk-for-python#41548` | 7 | 9 | 1 of 3 | 4 |
| `Azure/azure-mcp#576` | 8 | 9 | 1 of 3 | 3 |
| `reflex-dev/reflex-web#1512` | 76 | 79 | 0 of 2 | 3 |
| `artsy/force#15294` | 1 | 2 | 0 of 2 | 2 |

## What did hold

**G-DR1-1 passed, and it is worth having.** Stored corpus order is a prefix of GitHub's live commit
order, **sha for sha, on 10 of 10** pull requests drawn with seed 20260918 before anything was
classified. Nine are exact; one — `microsoft/FluidFramework#25610` — has 30 commits in the corpus
and **66 live today**, in order. AIDev's row order is chronological and can be relied on. The corpus
is a snapshot and cannot.

That check was written to protect a method that turned out to be broken for an unrelated reason,
and it is the only part of this run anyone should reuse.

## What would actually answer the question

Not the union. The file set at commit *k* is `base...sha_k`, which has to come from the forge — the
compare endpoint, or a clone — because it is the thing reverts and rebases are defined against.
That is one request per commit per pull request, which is why it was not the first thing tried, and
it is now the only honest way to ask this question. A sample of 40 pull requests would do it.

DRIFT-1 is not re-run by the same method. Whatever runs next gets its own preregistration, and its
G-DR1-2 equivalent has to pass before any classification is computed rather than after.

## Deviations

Amendment A was appended before any claim was classified, restating G-DR1-1 from "reproduces
GitHub's order" to "is a prefix of GitHub's order" after the FluidFramework snapshot gap appeared,
and renaming `MATCHES_FINAL` to `MATCHES_CORPUS_HEAD` because that is what it measures. Both hashes
are published. No gate was relaxed after a classification was seen; G-DR1-2 was not touched, and it
is the one that failed.

---

*We set out to show that our instrument calls people liars when it means their branch moved, and the
first thing we measured was our own reconstruction being wrong about a third of the pull requests it
read. The hypothesis is still open. The method is not.*
