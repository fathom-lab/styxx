# PROTOCOL — systematic prior-art survey for the OATH lane

Fathom Lab · 2026-08-26 · **Frozen before the systematic pass runs.** Queries, inclusion rules,
extraction fields and the pre-committed outcome table below do not move. A survey whose search
terms are chosen after seeing what they return is a literature review of the reviewer's priors.

## Disclosed contamination, stated first

**A pilot already happened, and this protocol is therefore not blind.** On this same date one
reader spent an afternoon searching and found four sources, recorded in
`RECON_oath_prior_art_2026_08_26.md`: `statcheck`, `arXiVeri`, *Proof-Carrying Numbers*
(arXiv 2509.06902) and *Agent-Native Research Artifacts* (arXiv 2604.24658).

Those four are **already known and are excluded from the novelty accounting below** — they cannot
be rediscovered and counted as evidence that the search works. The systematic pass exists to find
what the pilot missed, and its outcome is scored on what it adds, not on what it confirms. This is
the same shape as the v0.10 census preceding the v0.11 prereg: pilot first, freeze second, and the
pilot's own findings do not get to vote in the frozen measurement.

## The question

Who else has built machinery that binds a stated numeric claim to an external artifact and lets a
third party mechanically check the binding without trusting the claimant — and which parts of the
OATH lane's construction are, on the record, unoccupied?

Three sub-questions, scored separately because they have different neighbourhoods:

- **Q1 — the instrument.** Automated checking of numeric claims in documents against data.
- **Q2 — the certificate.** A persisted, re-checkable artifact binding claim, evidence and
  checker identity, such that a *later* failure is a detectable signal (drift).
- **Q3 — the discipline.** Preregistration enforced as a runtime precondition; a published
  negatives record; an instrument audited against its own standard.

## Sources and query families, frozen

Search surfaces: general web search, arXiv, ACL Anthology, CRAN/PyPI/GitHub, and the OECD/OSF
methods literature. Query families, executed verbatim, at most lightly re-spelled where a surface
requires it:

**Q1 family**
1. `automatic verification numeric claims scientific paper against data`
2. `check reported numbers in paper match released results file tool`
3. `claim evidence binding scientific document machine checkable`
4. `LLM numeric hallucination verification against source data protocol`
5. `reproducibility checking tool extracts numbers from manuscript`

**Q2 family**
6. `verifiable certificate binds claim to dataset hash provenance`
7. `attestation document claims data integrity re-verification over time`
8. `detect when a published result no longer reproduces automated drift`
9. `proof-carrying code analogy claims evidence machine-checkable proof`

**Q3 family**
10. `preregistration enforced automatically analysis blocked until registered`
11. `machine-readable preregistration kill gate automated scoring`
12. `publishing null results ratio laboratory record automated ledger`
13. `tool audits its own output self-application evaluation instrument`

## Inclusion and exclusion, frozen

**Include** a source iff ALL hold:
- it describes a SYSTEM, TOOL, PROTOCOL or DEPLOYED PRACTICE (not solely a position paper or a
  dataset with no checking machinery);
- it involves checking a stated claim or number against something external to the statement;
- it was **actually fetched** by the agent recording it, and the fetched content supports the
  extraction.

**Exclude**: general fact-checking of natural-language claims against a text corpus with no
numeric/artifact binding; plagiarism detection; pure benchmark datasets with no verifier;
citation-graph tooling; and anything the recorder could not open.

## The hallucination bar — the leg most likely to fail

A prior-art survey conducted by language models is worthless if any citation is invented, and
this is the failure mode this protocol is most exposed to. Therefore:

- Every recorded source carries a URL that the recording agent fetched.
- **Every source is independently RE-FETCHED by a second agent that did not record it**, which
  confirms the title, the venue/date, and that the content supports the extracted description.
- A source that fails re-fetch is dropped and **counted**. The dropped count is published.
- Agents are instructed that "could not verify" is a correct and expected answer, and that
  recording a source from memory is the one unrecoverable error.

## Extraction fields, frozen

For each included source: title; authors if stated; venue and date; URL fetched; which
sub-question it bears on; what it checks; what artifact it produces; whether that artifact is
re-checkable later; whether it requires cooperation from the claimant; and one sentence on how it
differs from OATH.

## Pre-committed outcome table

- **Any source found that does all three of Q1, Q2 and Q3** → `OATH_NOT_NOVEL`. The lane's
  novelty claim is withdrawn in full and the north star is rewritten to a positioning claim.
- **Q2 occupied** (a persisted certificate whose later failure is the signal) →
  `CERTIFICATE_PRIOR_ART`. The drift argument is retired as a novelty claim and kept only as an
  engineering property.
- **Q3 occupied** (preregistration as an enforced runtime precondition) →
  `DISCIPLINE_PRIOR_ART`. The strongest surviving claim from this lab goes with it, and what
  remains is scale of application rather than mechanism.
- **Q1 occupied and Q2/Q3 not** → `NARROWED_TO_DISCIPLINE`. Expected. The instrument is
  positioned as one of several, and the contribution is stated as the contract, the drift
  property, and the external-reach negative.
- **≥3 new qualifying sources not in the pilot** → the pilot is declared insufficient and any
  outward claim waits for a human-reviewed survey.
- **0 new qualifying sources across all 13 queries** → `SEARCH_UNDERPOWERED`, not `NOVEL`. A null
  from thirteen queries run by language models is evidence about the search, not about the world,
  and may not be reported as an absence of prior art.

That last row is the one that matters. **This protocol cannot return "styxx is novel."** The most
it can return is "these queries, on this date, found these things" — and the honest reading of a
null is that the search was too weak, not that the field is empty.

## Out of scope

Priority disputes; any claim about who was first; patentability; the SILENT-PASS lane, which has
its own survey in `RECON_landscape_2026_08_21.md`; and the interpretability rungs (R1/R2), whose
neighbourhood is a different literature and needs its own protocol.

---

*Frozen on commit. A survey that can only confirm its author is not a survey.*
