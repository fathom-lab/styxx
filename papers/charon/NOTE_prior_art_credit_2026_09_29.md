# Note — charon's neighbours, with dates

2026-09-29. This note sits beside `SPEC_charon_v01_2026_09_02.md` and
`RESULT_charon_v01_ships_2026_09_02.md` and edits neither (the RESULT is a sworn document, and the
log is history). Neither makes a priority claim. Neither credits the neighbours below, and the
spec's third property, SKEW versus DRIFT, was separated earlier by someone else. This note says so.

**Where the facts come from.** The landscape synthesis of 2026-09-28 and its critic pass (single
agents through a summarising web fetcher and the GitHub API, no hashed bytes, no frozen procedure,
no human review). Only neighbours from those fresh sweeps are named here; the chained-log lineage
the sand survey passes coded is deliberately left out, as the landscape's handling note requires.
charon landed on `main` on 2026-09-02.

| neighbour | what it does | its date | against charon | where |
|---|---|---|---|---|
| **Proof-Carrying Agent Actions**, Zexun Wang | Section 8.3 names four replay failure classes and separates "policy snapshot drift" from "materially drifted risk result": a checker that moved, told apart from a result that moved. | arXiv 2026-06-02 | three months before charon's SKEW / DRIFT split | https://arxiv.org/html/2606.04104v1 |
| **Microsoft agent-governance-toolkit receipts**, PR #1519 (Prashan Sapkota) | Ed25519 receipts over RFC 8785 JCS, chained by parent hash, bound to Cedar decisions, with an offline verifier (which checks signatures and chain, not decisions) and SLSA output. | opened and merged 2026-04-27 | four months before; signed, which charon is not | https://github.com/microsoft/agent-governance-toolkit/pull/1519 |
| **Swarm Orchestrator**, Brad Kinnard (ISC; self-reported dates) | A hash-chained audit ledger (v10.0.0). `rederive.mjs` in every run bundle re-derives each verdict from the bundle and names what it cannot; verdicts signed with DSSE / in-toto (v14.0.0). | ledger 2026-05-23; re-derivation evidence dated 2026-09-02, released 2026-09-07 | the ledger before charon; re-derivation from recorded bytes shown on the day charon landed | https://github.com/moonrunnerkc/swarm-orchestrator |
| **Agentic Witnessing**, Antony Rowstron (ARIA) | A TEE-hosted auditor answers questions about private data; a hash-chained, signed transcript; True / False / Unsure / Error. | arXiv 2026-04-27 | before; a hardware root, which charon lacks | https://arxiv.org/abs/2604.24203 |
| **Sello (Notarized Agents)**, Juan Figuera | The receiving service signs a COSE receipt per action; names set completeness as an open limit. | arXiv 2026-05-30 | before; counterparty signatures, which charon lacks | https://arxiv.org/abs/2606.04193 |
| **RFC 9942 (COSE Receipts), Rekor v2, the C2SP witness protocol** | Witnessed, append-only logs with standard receipts and consistency proofs. | dates not recorded by the landscape | charon is a single-writer chain with no witness, no consistency proof and no signature | https://www.rfc-editor.org/rfc/rfc9942 |

**Reading from today.** Hash chaining, re-deriving recorded verdicts from bytes, and telling a
moved checker from a moved result are all earlier than charon and not the lab's. What the landscape
did not find among the neighbours it read is a receipt-set size and a digest of every verifier module
on each line (`styxx/charon.py`, `_MODULES` and `_CERTIFIES`); that is bounded by those sources and
is not a survey result. On evidence strength charon is weaker than most rows above: one writer,
nothing signed, nothing witnessed.

`styxx/charon.py` is not edited to carry this credit, because its bytes are digested into every
charon line it writes (`_MODULES` names `styxx.charon` for every kind), and a docstring edit would
read as SKEW against the committed log. The lab-wide record is `../NOTE_prior_art_credit_2026_09_29.md`.
