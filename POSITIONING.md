# POSITIONING — what styxx is, in the product's own numbers

Fathom Lab · 2026-08-30. Every number below is bound to a committed receipt, and this document
carries its own OATH certificate (`POSITIONING.certificate.json`) issued by the verifier it
describes. A positioning claim that cannot survive the product is not a positioning claim this
repository is allowed to make.

## The thesis

**Verification correctness is not verification coverage.** A green checkmark that will not say
what it declined to examine is a half-truth, and the industry ships that half-truth as a product
category. styxx makes the boundary itself a software object: every certificate now states, in
machine-readable form, what the verifier was *obligated* to check, what it *volunteered*, what it
*refused*, and what it *never read*.

That is the identity: **epistemic boundaries as software objects.** Not a hallucination detector,
not a fact-checker, not an LLM judge. An attestation layer whose first loyalty is to its own
limits.

## The ladder

1. **diffgate** — verifies what an AI agent *says it changed* against what actually changed.
   The commercial wedge: agent adoption is bottlenecked on trust in agent reports.
2. **OATH** — verifies what a document *says was measured* against the receipts of the
   measurement. The research engine: certificates with a `verdict`, a ledger, and an
   `epistemics_summary` stating the boundary.
3. **The attestation layer** — policy over both: orgs gate merges, releases, and claims on
   certificates that confess their own coverage.

## The numbers (each one a receipt, not a slogan)

- **The unobligated oath, counted.** Across 192 internal certificates re-issued live:
  5951 verified tokens, of which 3458 were *volunteered* — an unobligated-oath rate of
  **0.5811**. The weakest cell (value match alone, path never compared) holds 2023 tokens,
  share 0.3399. Receipt: `papers/closed-model-frontier/oath_unobligated_oath_census.json`.
- **Abroad it is worse.** Over 82 external repositories: 575 verified, 414 volunteered —
  rate **0.72**, with the weakest share at 0.36. Receipt:
  `papers/closed-model-frontier/oath_external_epistemics_census.json`.
- **Obligation predicts claimhood.** Joining blind-panel verdicts to the obligation tag:
  obligated oaths land on real claims at 0.8472 internally and 0.7826 externally; volunteered
  external oaths collapse to **0.3654**. The boundary object is not bookkeeping — it predicts
  where the instrument's word is good. Receipt:
  `papers/closed-model-frontier/oath_obligation_claimhood_join.json`.
- **The gate audited its own author.** All 54 commits on the branch that built this system,
  gated against their own diffs: 6 claims extracted, 3 verified, 3 contradicted (all three
  hand-adjudicated as the catalogued mention-vs-use defect), and **2732 sentences never read**.
  Receipt: `papers/closed-model-frontier/agent_branch_attestation.json`.

## What this prices

The pipeline — extract, obligate, bind, certify, expose the boundary — runs end to end, on
documents and on agent branches. The open problems are named and measured, not hidden: **claim
extraction from agent prose** (the never-read band above) and **the obligation predicate for
documents** (the volunteered majority above). Lexical repairs to both are measured dead in this
corpus; the structural direction is licensed and unfrozen. Anyone claiming these problems are
solved is selling the half-truth this instrument exists to reject.

## What styxx will not say

It will not call a document true. It will not call an agent honest. It certifies a narrower,
harder thing: *these tokens matched these receipts, this was the obligation surface, this is
what was never examined* — and it prints the volunteered split in the terminal on every run,
because a verified count without its boundary is exactly the failure mode being sold elsewhere.

---

*The moat is not the verifier. The moat is that every number above survives the verifier.*
