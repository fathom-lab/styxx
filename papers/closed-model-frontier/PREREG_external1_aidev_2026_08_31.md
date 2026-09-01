# PREREG — EXTERNAL-1: does the gate hold on someone else's corpus?

Fathom Lab · 2026-08-31 · Frozen before any gate is run against the corpus. The first
styxx measurement on third-party data that this lab did not collect, curate, or choose.
Every threshold below is committed now; the outcome is unknown at freeze time.

## The question

Across agent-authored pull requests in the wild, **how often does the gate find that an
agent's PR description contradicts its own diff — and when it accuses, how often is it
right?** The second half is the one that matters. A gate that fails builds is only as good
as its false-accusation rate, and this lab treats a false accusation as the fatal error
class.

## The corpus, grounded before this freeze

**AIDev** (Hugging Face `hao-li/AIDev`; Zenodo DOI 10.5281/zenodo.16919272; dataset
licensed CC-BY-4.0, with each source repository retaining its own license for the code,
patches, and diffs it contributed). Introduced by Li, Zhang, and Hassan for the MSR 2026
Mining Challenge; we neither collected it nor influenced its construction.

Structure confirmed before freezing (not outcomes — structure):

- `pull_request` — the curated subset, carrying `title`, `body`, `agent`, `state`,
  `repo_url`, and timestamps. This is the claim side.
- `pr_commit_details` — commit-level rows carrying file-level changes and patch data.
  This is the evidence side.
- Agents represented: OpenAI Codex, Devin, GitHub Copilot, Cursor, Claude Code.
- **Known corpus boundary, disclosed by the dataset card**: patch data omits large patches,
  because the GitHub API does not return them.

## What is measured

For each eligible PR, the summary is `title` + `body`; the evidence is the changed-file
set and patch text reconstructed from `pr_commit_details`. The shipped diffgate template
set extracts diff-shaped claims and adjudicates each: VERIFIED, CONTRADICTED, or
UNCHECKABLE. No template is added, removed, or tuned for this corpus — the instrument runs
exactly as it ships, and the version and verifier hash are recorded with the result.

Three reported quantities:

1. **Coverage** — the share of eligible PRs in which the gate finds at least one checkable
   diff-shaped claim. This is the honest denominator: a gate that checks nothing is a gate
   that passes everything, and the uncovered remainder is reported, never hidden.
2. **Contradiction rate** — among PRs with at least one checkable claim, the share carrying
   at least one CONTRADICTED claim.
3. **Accusation precision** — the fraction of accusations that are correct, measured by
   blind hand-adjudication (protocol below). This is the primary result.

## Eligibility and exclusions, frozen

Included: PRs in the curated subset with a non-empty body and at least one reconstructable
changed-file record. Excluded and **counted, with the count published**: PRs whose patch
data is absent or truncated by the corpus's own known limitation; PRs with empty bodies;
PRs whose commit records fail to parse. An excluded PR is never scored as a pass.

## The blind adjudication protocol — the primary gate

Marking one's own homework is not a measurement. The adjudication follows this lab's
blind-panel discipline:

- A random sample of **100 accusations** (CONTRADICTED claims), drawn with a seed committed
  in this document: **seed 20260831**.
- Into that sample are mixed **30 decoys**: 15 claims the gate marked VERIFIED, and 15
  synthetic contradictions constructed by perturbing a verified claim's file path. Decoys
  are shuffled in and **sealed** — the adjudicator sees only the claim text and the actual
  changed-file facts, never the gate's verdict or its reason.
- The adjudicator answers one question per item: *does the diff support this claim, or
  contradict it?* Adjudications are recorded before any key is opened.
- **Adjudicator-reliability gate**: if fewer than 27 of the 30 decoys are called correctly,
  the adjudication is void and the whole measurement publishes as UNRELIABLE with no
  headline number. The adjudicator must pass before the instrument is scored.

## Gates and thresholds, committed now

- **G-E1 (precision, primary) — ship condition: accusation precision ≥ 0.95.** Below that,
  EXTERNAL-1 publishes as a **failure**, the offending template classes are disabled, and
  the GitHub Action does not launch on the current template set. There is no partial credit
  and no post-hoc threshold adjustment.
- **G-E2 (coverage floor) — ship condition: coverage ≥ 0.10.** If fewer than one in ten
  agentic PRs contains any claim this instrument can check, the gate is green noise on real
  data and must say so in its own headline rather than lead with a contradiction rate.
- **G-E3 (agent-stratified honesty)**: coverage, contradiction rate, and accusation counts
  are reported **per agent**, never pooled only. Any agent with fewer than 100 eligible PRs
  is reported as underpowered rather than compared.
- **G-E4 (reproducibility)**: the harness, the sampled item ids, the sealed adjudication
  keys, and the per-PR verdict ledger ship as receipts; the RESULT is certified by styxx's
  own OATH verifier and sealed in a capsule. A reader re-runs it from the published dataset
  version and the committed seed.

## What would falsify the instrument's premise

If accusation precision lands below 0.95, the claim "an agent's PR summary cannot lie about
its diff" is not yet true of this instrument on real-world data, and this lab will say so
in those words. If coverage lands below 0.10, the instrument is honest but narrow, and the
narrowness is the finding. Both outcomes publish under the same seal as a success, per the
charter, and the failure capsule ships next to the others.

## Named boundaries — what this measurement cannot show

It cannot show that agents *intend* to mislead: a contradiction between description and
diff is a fact about two artifacts, not a claim about a mind. It cannot cover claims outside
the closed template set — uncovered prose is listed, never judged. It inherits the corpus's
own patch-truncation boundary. It measures one snapshot of five agents in one window, and
says so wherever the numbers appear. And it says nothing about whether the changed code is
*correct* — only about whether the description matches the change. Coverage is not
correctness; correctness was never the claim.

---

*The instrument has only ever been measured on documents this lab wrote. This preregistration
is the first time it is pointed at a corpus that owes us nothing — with the failure condition
written down before the first run.*
