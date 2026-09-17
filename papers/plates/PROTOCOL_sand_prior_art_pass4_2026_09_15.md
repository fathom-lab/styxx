# PROTOCOL — pass 4 of the sand survey: the terms defined before reading, the nearest neighbours counted instead of flagged, and a frozen search beside the leads

Fathom Lab · 2026-09-15 · **A frozen procedure, not a result.** Committed, and pushed, before the first query
is sent and before the first source is fetched; the commit id of this file is the receipt that it was.
Successor to `PROTOCOL_sand_prior_art_2026_09_13.md` (frozen at `9f5fd42c`) and
`PROTOCOL_sand_prior_art_pass3_2026_09_14.md` (frozen at `54b066ff`), neither edited. It exists because pass 3
and its correction and erratum showed three weaknesses of the earlier passes, each written down there:

1. **Two terms were undefined.** "An interval on the comparison" and "a floor measured on the same weights"
   were coded differently by different readers, and the clause's margin depended on the reading
   (`ERRATUM_sand_neighbours_pass3_correction_2026_09_14.md`).
2. **Nearness was a reader's flag.** The licensed sentence named the sources readers called nearer; by count
   of elements the nearest source under every reading (Xu et al.) was not named.
3. **The list only followed citations.** Every source in passes 1 to 3 was either chosen from memory or cited
   by a source already read. A neighbour nobody cited — in particular recent work that asks whether an API
   serves the model it claims — could not enter the survey at all.

This pass defines the terms, counts nearness, and adds a frozen search. It may retire the fingerprint clause,
and with it (by sentence rule 1) the sentence.

## What this pass inherits and cannot undo

C3 (*a face for every receipt*) and C4b (*sealed on a public chain*) are RETIRED (pass 2) and stay retired. C1
is OCCUPIED on the authority of `papers/sworn/SURVEY_sworn_neighbours_2026_09_05.md` and is not re-priced. C2,
C4a and C5 are OCCUPIED after pass 3 and its erratum; this pass can move any of them to RETIRED or UNPRICED,
or name nearer neighbours, and cannot move any to FREE. The pricing rule (RETIRES / OCCUPIES / SILENT), the
status rule and the sentence rule of `PROTOCOL_sand_prior_art_2026_09_13.md` hold, amended only as below.

## The fingerprint clause's elements, defined before any fetch

C4a: *fingerprints a model's behaviour on hashed canaries against a measured null floor*. A reader codes each
element **true only when the source does it**, not discusses it, and gives a verbatim quote for every element
coded true.

- **E1 self-comparison.** The comparison is between a model and (i) the same model at another time, (ii) a
  quantized, compressed, pruned or otherwise differently-served version of the same trained weights, or (iii)
  the model an API or deployment claims to serve, represented by a reference copy of those weights.
  Comparisons between different models — which model is this, which family, was it stolen or fine-tuned from
  mine — are not a self-comparison, whatever the method.
- **E2 fixed item set.** Both sides of the comparison are given the same inputs, fixed before the comparison.
  Inputs sampled afresh for each side, or evolved during the comparison, are not a fixed item set.
- **E3 a floor measured on the same weights and used to grade.** The source measures the spread between
  repeated evaluations of identical weights — the same checkpoint, or an API version the provider identifies
  as the same — in the setup used for the comparison or for one of its sides, and the comparison's verdict
  depends on that spread (a threshold set from it, a null distribution drawn from it, a test against it). A
  spread across different seeds, retrainings or independently trained models is not this floor. A spread
  measured and not used to grade the comparison is not this floor. Identity inferred from stable scores is not
  identical weights.
- **E4 an interval on the comparison.** For the compared quantity the source reports either a confidence or
  credible interval, or a hypothesis test whose null is "no difference" at a stated level (an alpha, a
  critical value, a reported p-value). A spread of repeated runs of one side, or a threshold with no stated
  level, is not an interval on the comparison.
- **E5 log-probabilities (the thing).** The compared quantity is computed from the model's token
  log-probabilities, probabilities or logits on given continuations — teacher-forced scoring, or log-probs an
  API returns — not from sampled text, extracted answers or hard labels alone.
- **E6 hashed item set (descriptive).** The item set is published with a digest a stranger can check. Coded
  and reported; not required for RETIRES, because no source in passes 1 to 3 was coded for it and a clause
  should not survive on an element nobody looked for.

A source **RETIRES** C4a when it carries E1 to E5 for the same object: a served model's behaviour against its
own earlier or differently-served self. A RETIRES verdict stands only when **two further readers**, reading
the same full text independently and blind to the first reader's coding, each code E1 to E5 true. If either
does not, the source is recorded **DISPUTED**, and a DISPUTED source makes C4a **UNPRICED** under the inherited
status rule. Under sentence rule 2 an UNPRICED clause is deleted from the sentence; this protocol adds that a
sentence without its fingerprint clause is not the sand's sentence, and the lab licenses no sentence until a
later pass settles the dispute.

## Nearness, counted

For C4a, a source's **distance** is the number of E1 to E5 it does not carry. After all readings, the fingerprint
clause's parenthesis names, for each of E1 to E5, the listed source (from any pass) at the smallest distance
whose only missing element is that one, if such a source exists — at most five names, each with the element
that separates it from the clause. Readers' "nearer" flags are recorded and do not decide anything.

## The list, closed by rule

**Part A — the leads.** The nine unscored leads of pass 3 that are not already sources:

| id | source |
|---|---|
| A01 | Cao, Jia & Gong, *IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary*, AsiaCCS 2021 |
| A02 | Lukas, Zhang & Kerschbaum, *Deep Neural Network Fingerprinting by Conferrable Adversarial Examples*, ICLR 2021 |
| A03 | Finlayson, Ren & Swayamdipta, *Logits of API-Protected LLMs Leak Proprietary Information*, COLM 2024 |
| A04 | Russinovich & Salem, *Hey, That's My Model! Introducing Chain & Hash, an LLM Fingerprinting Technique*, arXiv 2407.10887 (2024) |
| A05 | Carlini, Paleka, Dvijotham et al., *Stealing Part of a Production Language Model*, ICML 2024 |
| A06 | Aiyappa, An, Kwak & Ahn, *Can We Trust the Evaluation on ChatGPT?*, arXiv 2303.12767 (2023) |
| A07 | Hooker, Moorosi, Clark, Bengio & Denton, *Characterising Bias in Compressed Models*, arXiv 2010.03058 (2020) |
| A08 | Ouyang, Zhang, Harman & Wang, *An Empirical Study of the Non-Determinism of ChatGPT in Code Generation*, TOSEM 34(2), 2025 |
| A09 | Song, Wang, Li & Lin, *The Good, the Bad, and the Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism*, arXiv 2407.10457 (2024) |

**Part B — the search.** Twelve queries, frozen here, each sent once to each of three engines, taking the first
ten results per query per engine, on the date the search runs:

| q | clause | query |
|---|---|---|
| Q01 | C4a | model equality testing which model is an API serving |
| Q02 | C4a | auditing model substitution in LLM APIs |
| Q03 | C4a | detecting changes in a deployed language model API over time |
| Q04 | C4a | detecting quantization of a served language model from its outputs |
| Q05 | C4a | log probability fingerprint to verify language model identity |
| Q06 | C4a | statistical test whether two language models are the same |
| Q07 | C4a | nondeterminism of LLM inference with identical weights and evaluation variance |
| Q08 | C4a | behavioral drift monitoring of large language models with a fixed prompt set |
| Q09 | C2 | tamper-evident hash chained log of machine learning evaluation results |
| Q10 | C2 | transparency log of model evaluations or AI claims |
| Q11 | C5 | bounty for reproducing or refuting published machine learning results |
| Q12 | C5 | paid bug bounty for errors in scientific research or verification tools |

Engines: **(1)** the arXiv API (`export.arxiv.org/api/query`, `search_query=all:` with the query's words joined
by `AND`, sorted by relevance); **(2)** the Semantic Scholar Graph API (`/graph/v1/paper/search`); **(3)** a
general web search. Every result is recorded with the engine, the query, its rank, its title, authors, year,
link and, where the engine returns one, its abstract.

**Inclusion rule**, applied to title and abstract alone by screeners who do not read further: a result enters
the list when it (a) compares a model's outputs with the same model's outputs at another time, under another
serving (quantized, compressed, a claimed API model), or tests whether an API serves a claimed model; or (b)
measures repeated-run variance of identical weights for use in comparing models or versions; or (c) keeps an
append-only or hash-chained log of evaluation results or verdicts; or (d) offers payment for reproducing or
refuting published results or for finding errors in a verifier. A result is excluded when it is already a
source in passes 1 to 3, when it is only about identifying which of different models produced text
(authorship, watermark detection, family attribution), or when it is not a paper, a report, or the page of a
standing programme. Two screeners apply the rule independently to every result; a result enters when either
includes it.

**Cap.** Included results are ranked by the number of (engine, query) pairs that returned them, ties broken by
the later year, then by title. The first **twenty-five** enter the list as B01 to B25. Everything below the cap
is recorded with its rank and stays a lead. No source is added after the cap is applied, by anyone, for any
reason; a source a reader meets while reading is a lead of the next pass.

## What the survey must record

For every query result: engine, query, rank, what was returned, both screeners' decisions with one sentence of
reason. For every listed source: the URL fetched, the time, the sha256 of the bytes, READ / SKIMMED /
UNFETCHABLE, the characters of extracted text, and for each clause it might touch the verdict, the object in one
sentence, E1 to E6 with a quote for each coded true, and one sentence of reason. For every RETIRES: the two blind
confirmations. Per clause: status, retiring, disputed and occupying sources, and the nearest source per element.
The conjunction's status and the surviving sentence, or RETIRED, or UNLICENSED (a disputed retirement). All of it
in `sand_prior_art_survey_pass4_<date>.json`, built mechanically by a committed script from the committed search
record, screen record, fetch record and reader returns; the reader prompts are committed beside the returns; the
extracted full texts are not committed (other people's work) and their sha256 are. The SURVEY swears to the
record.

## What this procedure does not do

It does not claim the search is exhaustive: three engines, twelve queries, ten results each, on one date. It does
not re-open C3 or C4b. It does not price "first", "novel" or "revolutionary", which no outcome licenses. It does
not price the sand's value, only its neighbours. It does not let anyone — the lab, a reader, a screener — add a
source after the cap.
