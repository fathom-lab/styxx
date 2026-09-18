# PREREG — BENCH-1: a PR-level benchmark for description-versus-diff verification, scored at the base rate that matters

Fathom Lab · 2026-09-17 · Frozen before a single live diff is fetched and before anything is
scored. Built on HARNESS-1 (`RESULT_harness1_merge_fold_lands_2026_09_16.md`) and COMPAT-2
(`RESULT_compat2_surface_and_panel_2026_09_16.md`). The instrument is not changed by this cycle:
`styxx/diffgate.py` sha256 `4ba947a8…` is read-only here.

## Why this exists

There is no PR-level benchmark for "does this pull request's description match its diff". The
nearest artifact, CodeFuse-CommitEval, is commit-level and its inconsistencies are synthetic
mutations. The MSR '26 study that measured the problem released 974 human-annotated PRs in an
unlicensed repository and stated that detection tooling does not yet exist. Its own best method
scores F1 0.630 against human agreement of κ 0.892.

Two things are missing from that picture, and this benchmark supplies both.

**Ground truth that is not an opinion.** For a specific class of claims the diff settles the
question outright: how many files changed, whether every changed path sits under a stated prefix,
whether a named symbol is defined in the added lines. No annotator is required and no κ ceiling
applies. This benchmark scores only those.

**The metric that decides whether a tool gets used.** Published evaluations report precision and
recall on a curated set. What determines whether a reviewer acts on a verdict is the false
discovery rate at the rate these claims actually appear. The two differ enormously, and no
existing benchmark reports the second.

## Population, frozen

All 71,016 eligible PRs of the AIDev corpus (HuggingFace `hao-li/AIDev`, CC-BY-4.0, Zenodo
10.5281/zenodo.16919272) were read under the HARNESS-1 fold. **691 PRs** carry at least one claim
of a decisive kind, by the instrument's own extraction: `files_changed_count` (142 claims),
`only_touches` (358), `tests_added` (199), `symbol_added` (40). That list is frozen now as
`bench1_population.json` (pr_id, URL, agent, claim text and detail) before any diff is fetched.

The corpus verdicts are recorded but are **not** ground truth. They were computed on reconstructed
diffs, and HARNESS-1 proved that reconstruction wrong on 6,697 PRs. Ground truth comes from the
live pull request.

## The oracle, stated so a reader can audit it

For each PR the live unified diff is fetched from `patch-diff.githubusercontent.com/raw/O/R/pull/N.diff`
— the bytes GitHub serves, not our reconstruction — and its sha256 recorded. Ground truth is then
derived by `bench1_oracle.py`, written independently of `styxx/diffgate.py` and short enough to
read in full:

- **files changed** = the number of distinct `diff --git` header lines. Renames, mode-only changes
  and binaries count as one file each, because each carries exactly one header.
- **paths changed** = the `b/` path of each header, or the `a/` path where the file is deleted.
- **added lines** = lines beginning `+` that are not `+++`.

`files_changed_count` is scored as CONTRADICTED when the stated integer differs from the file
count; `only_touches` when any changed path lies outside the stated prefix; `symbol_added` when no
added line defines the named symbol. **`tests_added` is excluded from scoring** and reported
separately: deciding what counts as "a test" is a judgement, BC-1 failed its own preregistration
on exactly that ambiguity, and a benchmark must not smuggle a judgement call into its oracle.

Two honesty constraints on the oracle. It agrees with the instrument by construction on
well-formed diffs, so its discriminating power sits in the hard cases and in the true negatives;
this is stated in the RESULT rather than glossed. And **50 items are audited by hand** against the
rendered PR before scoring, sampled with seed 20260918 across all outcomes; every disagreement
between the oracle and the audit is published, and if the oracle is wrong on more than 2 of the
50 the benchmark is declared INVALID and re-frozen.

## What gets published

`bench1_dataset.jsonl` — one row per item: PR URL, claim text and kind, the live diff's sha256 and
byte length, the oracle's ground-truth label and the facts behind it. The diffs are not
redistributed; each row carries the URL and hash so anyone can re-fetch and verify. AIDev's
CC-BY-4.0 attribution is carried in the dataset header. This is the licensing gap the MSR '26
artifact leaves open, and closing it is half the point.

## Gates — committed now

- **G-B1-1 (the oracle survives audit).** At most 2 of the 50 hand-audited items disagree with the
  oracle. Blocking; failure means INVALID and a re-freeze, as BIN-1 did.
- **G-B1-2 (reach).** At least 80% of the 691 PRs return a diff. PRs that 404, are renamed away or
  are otherwise unreachable are counted, listed by reason, and excluded — never silently dropped.
- **G-B1-3 (the headline pair, reported not scored).** On the reachable set, for the instrument:
  precision, recall, specificity and F1 per kind, **and** the false discovery rate implied at the
  population base rate (claims of that kind per 71,016 PRs). Both numbers, side by side, always.
- **G-B1-4 (baselines run on the same rows).** The instrument is reported beside at least one
  non-styxx baseline implemented here: a string-similarity baseline over (claim sentence, diff),
  which is the MSR '26 method's family. Any baseline we run, we publish, including where it beats
  us.
- **G-B1-5 (what is not claimed).** No comparison against a competitor product. No claim that the
  instrument is state of the art on the MSR '26 task — their task is human-annotated PR-level
  inconsistency, this is mechanically-decidable claim verification, and the two are not the same
  measurement. No agent-by-agent ranking.

## What would falsify the thesis this benchmark exists to test

The thesis is that a deterministic reading beats a judgement-based one on specificity at a low
base rate. It is falsified if the similarity baseline matches the instrument's specificity, or if
the instrument's own false discovery rate at population base rate exceeds 0.5 — that is, if more
than half of what it would flag across the corpus is wrong. Either result is published as
prominently as a passing one.

---

*Nobody has built the benchmark the field was told it needed, and the reason is that human
annotation of PR intent is expensive and its ceiling is low. This benchmark takes the narrower
question the diff can actually settle, states its oracle in three rules, audits fifty of them by
hand, and reports the number that decides whether anyone acts on a verdict.*
