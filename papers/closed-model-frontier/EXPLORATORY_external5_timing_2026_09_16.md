# EXPLORATORY — when were the 70 upheld descriptions written? (not preregistered)

Fathom Lab · 2026-09-16 · An addendum to `RESULT_external5_survivors_at_source_2026_09_16.md`,
run after it and labelled for what it is: a look at the timing caveat with the two timestamps the
public record offers. Receipt: `exploratory_e5_timing.json` (counts and per-item classes; no PR
named). Nothing here changes the RESULT's numbers.

## Method, and its two holes

For the 89 reached items: the pull request's opening time (from its page) and the author date of
each of its commits (from the `.patch` endpoint, which the container can reach). An item sits on
"commits after opening" when its PR's last commit was authored more than a minute after the PR
opened. For the 52 upheld file counts, the files first touched by post-opening commits were
counted from the same `.patch`; an item is **stale_exact** when the live count minus those files
equals the described count.

The holes, stated before the numbers: author dates survive rebases, so "after opening" is a lower
bound; and the REST API does not expose when a description was last edited, so a description
written or rewritten after the last commit is invisible here. Several agents open the pull request
first and push every commit afterwards (all files "after opening", none before), and write the
description last — for those the opening time says nothing about the description at all.

## What the two timestamps say

- 61 of the 70 upheld items, and 17 of the 19 overturned, sit on PRs that took commits after
  opening (5 and 1 unknown). Upheld and overturned alike, at the same rate: this is the base rate
  of agent pull requests — median 1.7 hours from opening to the last commit, quartiles 0.3 / 1.7 /
  20.0 — not a property of the accusations.
- Of the 52 upheld file counts: **6 stale_exact** (the described count is exactly the live count
  minus the files later commits introduced); **29** where later commits added files and the live
  count exceeds the described one (consistent with staleness, not proof of it); **6** where no file
  was added after opening (staleness cannot explain them); 8 other; 3 unknown.

## What that settles, and what it does not

It does not settle the withholding question EXTERNAL-5 pre-committed. Six counts were exactly
overtaken by later commits; six could not have been; for the forty in between the public record
cannot say when the sentence was written. A door that re-gates on every push (CI, the hooks) sees
description and diff at the same moment, and a mismatch there is real whatever its history; the
corpus reading has no such moment. The decision is about the word the gate prints for a count
that no longer matches, and timestamps will not make it. It is left where the RESULT left it: a
repair prereg, owed, whose first line has to be that sentence.
