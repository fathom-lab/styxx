# RESULT — BENCH-2 is INVALID, and this time the instrument is wrong too: 9 of 11 accusations are false

Fathom Lab · 2026-09-17 · Prereg: `PREREG_bench2_pr_claim_benchmark_2026_09_17.md` (sha256
`5a796aa7…`, frozen before the run). Receipts: `bench2_oracle.py`, `bench2_score.py`,
`bench2_scores.json`, `bench2_dataset.jsonl`. Supersedes and must be read with
`RESULT_bench1_INVALID_2026_09_17.md` (G-B2-5). Instrument read-only at sha256 `4ba947a8…`.

**G-B2-1 failed: 3 of the first 7 non-trivial audited items disagree with the oracle, against a
limit of 2.** BENCH-2 is void. Under the prereg's own terms there is no BENCH-3 by the same
method.

That is the smaller of the two findings. The larger one is below, and it is against us.

## The instrument false-accuses on 9 of its 11 accusations

The audit required adjudicating the scored items against the rendered pull request. Doing that to
all 11 `only_touches` items the instrument returned CONTRADICTED on — not just the sampled ones —
gives this. Every row was checked against the live diff.

| PR | the claim | verdict |
|---|---|---|
| `microsoft/vscode-azuretools#2086` | "Only modify package.json and package-lock.json files in each package folder" | **false accusation** — all 12 changed files are exactly those, in package folders |
| `ydb-platform/ydb#25857` | "Only modified `blobstorage_pdisk_impl.cpp`" | **false accusation** — the single changed file is `ydb/core/blobstorage/pdisk/blobstorage_pdisk_impl.cpp` |
| `Albeoris/Memoria#1142` | "only changed mods/submods are serialized, all others preserved bytewise" | **false accusation** — describes runtime serialization, not PR file scope |
| `Albeoris/Memoria#1145` | "Only modified mods/submods are serialized and saved" | **false accusation** — same |
| `Albeoris/Memoria#1147` | "Only changed mods/submods generate new XML with updated fields" | **false accusation** — same |
| `dotnet/runtime#117821` | "Only modify Assert.NotNull usages within this file" | **false accusation** — `Assert.NotNull` is a symbol, admitted because it matches `name.ext` |
| `mikepenz/release-changelog-builder-action#1458` | "commits that only touched files in `app1/`" | **false accusation** — README prose describing a feature, not a claim about this PR |
| `open-policy-agent/cert-controller#415` | "Only change .githiub/workflows/dependabot.yml" | **false accusation** — the author's typo; the PR changes `.github/workflows/dependabot.yml` and nothing else |
| `fern-api/fern#9898` | "only changes README/documentation snippets, not runtime SDK code" | **unsound** — "README/documentation" read as a directory because it contains a slash |
| `Azure/autorest.typescript#3252` | "Only modified `packages/typespec-ts/` and `packages/typespec-test/`" | correct — `common/config/rush/pnpm-lock.yaml` is outside both |
| `microsoft/wassette#442` | "Only modifies CHANGELOG.md in the Unreleased section" | correct — three workflow files also changed |

**Precision 2/11 = 0.18 on the one claim kind where the instrument accuses.**

### In the instrument's own words

Every accusation was re-run against the shipped gate outside the scoring harness; all eleven
reproduce. These are the instrument's own reason strings, unedited — `bench2_audit.json` carries
one for each:

```
paths outside 'blobstorage_pdisk_impl.cpp': ['ydb/core/blobstorage/pdisk/blobstorage_pdisk_impl.cpp']
paths outside 'githiub/workflows/dependabot.yml': ['github/workflows/dependabot.yml']
paths outside 'assert.notnull': ['src/libraries/system.linq/tests/sequencetests.cs', ...]
paths outside 'package.json' and 'package-lock.json': ['appservice/package.json', ...]
```

The first line is the instrument naming a file and declaring that same file to be outside itself.
The second is a one-character typo in someone else's sentence. No commentary improves on these.

## Six failure modes, named

1. **Basename claims.** A bare filename means "a file with this name, anywhere", not a path at the
   repository root. Two false accusations, both against major repositories.
2. **Runtime-behaviour sentences.** "only changed X are serialized" scopes a program's behaviour,
   not a diff. Three false accusations on one repository.
3. **Documentation prose.** A README sentence describing what the tool does is not a claim about
   the pull request containing it.
4. **Dotted identifiers.** `Assert.NotNull` satisfies `name.ext` and is admitted as a path.
5. **Prose containing a slash.** `README/documentation` is not a directory.
6. **Obvious typos.** `.githiub/` does not exist, so every path is "outside" it, so the PR is
   called a liar for a spelling mistake in its own description.

## What this does to the BC-2 repair

BC-2 added `looks_like_path()` under preregistration and it is the reason BENCH-1 showed the
instrument abstaining 276 times where a naive oracle accused 279. That result stands and is
correct. But the same function admits `package.json`, `Assert.NotNull`, `mods/submods` and
`README/documentation`. It removes the obvious non-paths — `the`, `with`, `are`, `3` — and nothing
else. It is a filter against one failure mode, not a test of whether a sentence is a file-scope
claim at all, and until today we had no measurement that could tell those apart.

Filed as an issue against ourselves with all nine PRs named. No fix is attempted in this cycle: a
repair to the accusing path gets its own preregistration, its own audit, and a published
before-and-after, or it is worth nothing.

## The scores, published as the prereg requires

They should be read as void, and are printed because G-B2-2 and G-B2-3 require it.

| kind | claims | admissible | instrument P / R / F1 | similarity baseline (threshold tuned on the labels) |
|---|---|---|---|---|
| `files_changed_count` | 116 | 116 | 1.00 / 1.00 / 1.00 | 0.387 / 0.915 / 0.544 |
| `only_touches` | 299 | 17 | 1.00 / 1.00 / 1.00 | 0.625 / 0.909 / 0.741 |
| `symbol_added` | 38 | 15 | — / 0.00 / — (abstained on 13 of 15) | 0.333 / 0.667 / 0.444 |

Three things must be said about that table rather than left for a reader to notice.

**The `only_touches` 1.00 is an artefact.** The repaired oracle and the instrument agree on 297 of
299 items, including 279 of 279 abstentions — 98.9%. They are the same rule implemented twice, so
the perfect score measures agreement, not discrimination. Hand-adjudication, which is not the same
rule twice, gives 0.18.

**`files_changed_count` is circular by construction** (G-B2-6): the oracle counts `diff --git`
headers and the instrument registers one file per header.

**The baseline beats us on `symbol_added`** (G-B2-4). The instrument abstained on 13 of 15 and
scored recall 0.00; a token-overlap baseline with its threshold chosen after seeing the labels
scored F1 0.444. We publish that because we said we would.

## Reach and admissibility

568 of 691 PRs reached (82.2%), unchanged from BENCH-1 — the same fetched bytes were re-labelled
so no PR could enter or leave under the repair. Admissibility rejected 282 of 299 `only_touches`
claims and 23 of 38 `symbol_added` claims. The scored set collapsed from 453 to 148, which is the
number G-B2-2 exists to make visible: an oracle can always be made to look accurate by declining
to decide anything hard.

## The methodological finding

Two independent mechanical oracles were written a day apart by people who build this for a living.
Both failed their own blocking audit. BENCH-1's failure was clumsy — it read `the` as a directory.
BENCH-2's is not clumsy, and that is what makes it informative: after both admissibility repairs,
the residual errors are sentences where *no* surface feature distinguishes a file-scope claim from
a statement about program behaviour. "Only modifies CHANGELOG.md" and "only changed mods/submods
are serialized" are the same shape. One is checkable against a diff and the other is not, and the
difference is meaning.

That is the wall the MSR '26 study hit when it reported human agreement of κ 0.892 and an F1 of
0.630 against it. We did not get past it by being more careful about regexes, and the prereg's
no-third-attempt clause is honoured here: any further work on this claim kind needs human
adjudication in the loop, not another admissibility test.

## Deviations

One implementation note, recorded rather than silently applied. The prereg's PATH-SHAPED clause
(a) is "`p` contains `/`", but the normaliser stripped trailing slashes before the test, which
would have demoted "only touches `src/`" to a bare name and lost clause (a) entirely. Clause (a)
is therefore evaluated before trailing-slash normalisation, which is what the prereg says and not
what the first implementation did. Caught before the run, not after.

---

*Yesterday the benchmark was wrong and the instrument was right, 279 to 1, and we were pleased
with ourselves. Today the benchmark is better and the instrument is wrong 9 times out of 11, on
pull requests belonging to Microsoft, .NET, YDB and the OPA project, by name. The second number
is the one that was worth the day.*
