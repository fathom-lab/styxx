# PROTOCOL — pricing one sentence about the sand: the neighbours of the plate, the ferry log, the checksum, the seal and the bounty, frozen before any fetch

Fathom Lab · 2026-09-13 · **A frozen procedure, not a result.** Committed before the first source
is fetched; the commit id of this file is the receipt that it was. Successor in shape to
`papers/sworn/PROTOCOL_sworn_prior_art_2026_09_02.md`, which priced one sentence about sworn output
and is not edited. It prices exactly one sentence — the one the lab is being asked to say about the
sand, and has so far refused to say without this procedure. The survey that runs it is
`SURVEY_sand_neighbours_<date>.md`; its receipt is `sand_prior_art_survey.json`, beside it.

## The sentence being priced, verbatim

> We know of no lab that binds every published number to bytes at a commit, re-derives every
> verdict from those bytes into a chained log, gives every receipt a face a stranger reads without
> json, fingerprints a model's behavior on hashed canaries against a measured null floor under a
> preregistration sealed on a public chain before the run, and pays a standing bounty against its
> own verifier — at once.

The sentence concedes in advance that every clause has neighbours: hash visualisation is 1999,
timestamping is 1991, representational similarity is 2008, drift measurement by log-probabilities is
standard, chained transparency logs are 2013. Only the conjunction is claimed unoccupied, and that
claim was written from memory. This procedure replaces memory with a fetch, and it may retire the
sentence entirely.

## The clauses

| id | clause | what it asserts, operationally |
|---|---|---|
| C1 | *binds every published number to bytes at a commit* | already priced: `SURVEY_sworn_neighbours_2026_09_05.md` found the sworn clauses OCCUPIED with neighbours named; C1 inherits that status and is not re-priced here |
| C2 | *re-derives every verdict from those bytes into a chained log* | an append-only, hash-chained log whose every line is a verdict re-computed from the receipt bytes by an independent verifier, head pinned outside the log (charon) |
| C3 | *gives every receipt a face a stranger reads without json* | a deterministic picture of a receipt digest, rendered from the digest alone, with a stated mapping version, used as a reading aid for reproduction |
| C4a | *fingerprints a model's behavior on hashed canaries against a measured null floor* | teacher-forced log-probabilities on a fixed, hashed item set; a comparison graded against an in-situ same-weights floor; a verdict with an interval |
| C4b | *under a preregistration sealed on a public chain before the run* | the hash of a frozen preregistration anchored in a public ledger, the anchor re-verifiable by a stranger, before any data |
| C5 | *pays a standing bounty against its own verifier* | a published, standing offer to pay a stranger whose record shows the lab's own verifier disagreeing with the lab, adjudicated by the verifier |
| C6 | the conjunction | one lab does C1 through C5 together, in public, on its own claims |

## The pricing rule, per clause and per source

Each READ source receives, for each clause it might touch, exactly one of:

- **RETIRES** — the source does the same thing for the same object (a lab's own published claims,
  or a model's behavior, as the clause states). The clause is then false as a novelty claim.
- **OCCUPIES** — the source does the same thing for a narrower or different object (host keys
  instead of receipts; a training checkpoint instead of a served model; a step of a build instead
  of a claim; a prize for preregistering instead of for breaking a verifier). The clause survives
  only with the neighbour named beside it.
- **SILENT** — the source does not address the clause.

A SKIMMED source may only be recorded as SILENT or as OCCUPIES *from abstract*; it may not RETIRE.
An UNFETCHABLE source is recorded as UNCHECKABLE and every clause the list says it *might* occupy
is marked **UNPRICED**. UNPRICED is never read as free.

Clause status after all sources are scored: RETIRED if any READ source RETIRES it; else UNPRICED
if any candidate that might occupy it was UNFETCHABLE; else OCCUPIED if any source OCCUPIES it;
else FREE. C6 is RETIRED if one READ source RETIRES or OCCUPIES every one of C2–C5 at once (one
lab, one practice); otherwise C6 takes the weakest status among C1–C5 in the order
RETIRED > UNPRICED > OCCUPIED > FREE and survives as the conjunction of the surviving clauses only.

## The sentence rule

1. If C4a or C5 is RETIRED the sentence is RETIRED entirely: those two are what the sand is *for*.
2. A RETIRED clause is deleted from the sentence. An UNPRICED clause is deleted. What could not be
   checked is not claimed.
3. An OCCUPIED clause survives with its nearest neighbour named in the same sentence.
4. The word "first" is not licensed by any outcome of this procedure. The licensed form is
   "we know of no …", and only for the surviving conjunction.

## The source list, closed at this commit

Named by title, author and year; located by those, and a source that cannot be located from them
is UNFETCHABLE, not replaced. Nothing is added after this commit; a lead met while fetching is
recorded under `leads_not_scored` and not scored.

| id | source | might occupy |
|---|---|---|
| S01 | Perrig & Song, *Hash Visualization: a New Technique to Improve Real-World Security*, CrypTEC 1999 | C3 |
| S02 | OpenSSH 5.1 release notes (2008), `VisualHostKey` / random art | C3 |
| S03 | Don Park, *Identicon*, 2007 (the blog post that named them) | C3 |
| S04 | Haber & Stornetta, *How to Time-Stamp a Digital Document*, Journal of Cryptology 1991 | C4b |
| S05 | Peter Todd, OpenTimestamps (2016), project documentation | C4b |
| S06 | Proof of Existence (Aráoz, 2013), project page | C4b |
| S07 | Laurie, Langley, Kasper, *Certificate Transparency*, RFC 6962 (2013) | C2 |
| S08 | Sigstore Rekor transparency log, project documentation (2021–) | C2 |
| S09 | Chambers & Tzavella, *The past, present and future of Registered Reports*, Nature Human Behaviour 2022 | C4b |
| S10 | Center for Open Science, the Preregistration Challenge (2015–2018), programme page | C4b, C5 |
| S11 | Chen, Zaharia & Zou, *How Is ChatGPT's Behavior Changing over Time?*, arXiv 2307.09009 (2023) | C4a |
| S12 | Dutta et al., *Accuracy Is Not All You Need*, arXiv 2407.09141 (2024) | C4a |
| S13 | Xu et al., *Instructional Fingerprinting of Large Language Models*, NAACL 2024 (arXiv 2401.12255) | C4a |
| S14 | Zhang et al., *REEF: Representation Encoding Fingerprints for Large Language Models*, arXiv 2410.14273 (2024) | C4a |
| S15 | Kriegeskorte, Mur & Bandettini, *Representational Similarity Analysis*, Frontiers in Systems Neuroscience 2008 | C4a |
| S16 | Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, 2025 | C4a |
| S17 | Hochlehnert et al., *A Sober Look at Progress in Language Model Reasoning*, COLM 2025 (arXiv 2504.07086) | C4a |
| S18 | Pineau et al., *Improving Reproducibility in Machine Learning Research*, JMLR 2021 (arXiv 2003.12206) — the ML Reproducibility Challenge | C5 |
| S19 | Jia et al., *Proof-of-Learning: Definitions and Practice*, IEEE S&P 2021 | C4a, C4b |
| S20 | Immunefi, bug-bounty programme documentation for smart-contract verifiers (the largest standing verifier bounty programme) | C5 |

Twenty sources, closed. Two are programme pages rather than papers (S10, S20) because the clause
they might occupy is a practice, not a result.

## What the survey must record, per source

The URL as fetched, the fetch time, the sha256 of the bytes saved from the page, READ / SKIMMED /
UNFETCHABLE, and for each clause in the "might occupy" column the verdict with one sentence of
reason quoting the source. Per clause: the status, the retiring and occupying sources, the
nearest neighbour. The conjunction's status. The surviving sentence, verbatim, or the word
RETIRED. All of it in `sand_prior_art_survey.json`; the SURVEY document swears to it.

## What this procedure does not do

It does not search beyond the list, so a neighbour the list missed is a neighbour the survey
missed; the leads column exists to say so. It does not price "revolutionary", "novel" or "first":
those words are outside the licensed form regardless of outcome. It does not price the sand's
value, only its neighbours.
