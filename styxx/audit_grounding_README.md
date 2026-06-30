# styxx.audit_grounding — is this claim backed by its data?

## The problem it addresses
AI now produces claims — papers, eval reports, dashboards, model self-reports — faster than anyone can check
them. The load-bearing question is the oldest one in science: does every number trace to a result, and does the
language stay within what the data actually shows? Almost no one checks this mechanically, and it is exactly
where errors and overclaims hide.

## What it does
`audit_grounding(text, sources)` — **deterministic, no LLM, no web:**
- extracts every statistical number from the text (RSA, CIs, p-values, %, fold-changes)
- classifies each **GROUNDED** (present in the data) / **DERIVED** (a %/ratio of two source values) /
  **UNSOURCED** (no backing found)
- runs a heuristic **overclaim linter** (`detect_overclaims`): flags priority/"first", equivalence asserted from
  a failed rejection, causal language from correlational evidence, hype, and "survives/robust" with no CI nearby
- renders a shareable report card (`report.render_html()`)

```python
from styxx import audit_grounding
rep = audit_grounding(paper_text, ["result_a.json", "result_b.json"])
rep.verdict        # "ALL GROUNDED" or "UNSOURCED: N/total"   (exit-code / CI-gate friendly)
rep.unsourced      # the numbers with no backing — review these
rep.overclaims     # language that may reach past the data
```
CLI: `python -m styxx.claim_audit <claim.md> <result_a.json> <result_b.json> …` (non-zero exit if unsourced).

## Why deterministic matters
The grounding layer has no model in the loop to hallucinate and no API to call — a number is backed or it isn't,
and it runs in CI. That is the part you can trust to gate a merge. The overclaim linter is **heuristic**: it
raises flags for a human to review, not verdicts.

## Honest scope (what it is *not*)
- It checks whether numbers trace to the sources you give it — **not** whether those sources are themselves
  correct. That is a different question (see `audit_confound` / `validate_against_ground_truth`).
- The overclaim linter is pattern-based; it will miss some and over-flag others. It is a linter, not a judge.
- It does not replace review. It automates the cheap, mechanical part so review can spend its attention on
  judgment.

## The proof
Dogfooded on its authors' own preprint: it grounded **119 of 132** numbers, flagged the 12 that live in scripts
or prose rather than committed result files (including one stat the authors had caught only by hand), and the
overclaim linter flagged the authors' **own** equivalence phrasing for revision. The tool turned on its makers
and found real things — which is the only kind of demonstration worth trusting.

— fathom-lab / styxx
