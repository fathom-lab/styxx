# PROTOCOL — pass 3 of the sand survey: the leads pass 2 could not score, frozen as a closed list before any fetch

Fathom Lab · 2026-09-14 · **A frozen procedure, not a result.** Committed before the first source
is fetched; the commit id of this file is the receipt that it was. Successor to
`PROTOCOL_sand_prior_art_2026_09_13.md` (frozen at `9f5fd42c`, not edited), whose pricing rule,
sentence rule and definitions this pass inherits unchanged. Pass 1 and pass 2
(`SURVEY_sand_neighbours_2026_09_13.md`, `SURVEY_sand_neighbours_pass2_2026_09_13.md`, both sworn)
closed their list at twenty sources and recorded forty leads they could not score, because the
list was closed. This pass closes a list drawn from those leads and scores it.

## What this pass inherits and cannot undo

- **C3** (*a face for every receipt*) and **C4b** (*sealed on a public chain*) are RETIRED by
  pass 2 (Perrig & Song 1999; Haber & Stornetta 1991 §5–6, OpenTimestamps). A retired clause
  stays retired; nothing here re-prices it.
- **C1** is OCCUPIED on the authority of `SURVEY_sworn_neighbours_2026_09_05.md` and is not
  re-priced.
- **C2**, **C4a**, **C5** are OCCUPIED after pass 2, each with its nearest neighbour named. This
  pass can move any of them to RETIRED, or add a nearer neighbour; it cannot move them to FREE.
- The sentence that survives pass 2 has four clauses. Under the inherited sentence rule 1, if
  this pass RETIRES C4a or C5 the sentence is RETIRED entirely.

## The clauses priced here

| id | clause | what it asserts, operationally (verbatim from the 2026-09-13 protocol) |
|---|---|---|
| C2 | *re-derives every verdict from those bytes into a chained log* | an append-only, hash-chained log whose every line is a verdict re-computed from the receipt bytes by an independent verifier, head pinned outside the log |
| C4a | *fingerprints a model's behavior on hashed canaries against a measured null floor* | teacher-forced log-probabilities on a fixed, hashed item set; a comparison graded against an in-situ same-weights floor; a verdict with an interval |
| C5 | *pays a standing bounty against its own verifier* | a published, standing offer to pay a stranger whose record shows the lab's own verifier disagreeing with the lab, adjudicated by the verifier |

The pricing rule per source and clause — RETIRES (same thing, same object), OCCUPIES (same thing,
narrower or different object; survives with the neighbour named), SILENT — and the status rule
(RETIRED if any READ source RETIRES; else OCCUPIED if any source OCCUPIES; UNPRICED for an
UNFETCHABLE candidate) are the 2026-09-13 protocol's, unchanged. A SKIMMED source may only be
SILENT or OCCUPIES *from the pages read*; it may not RETIRE.

## What "the same object" means for C4a, stated before reading

C4a's object is **a served model's behaviour, compared with its own earlier or differently-served
self, on a fixed item set, graded against a floor measured on the same weights under the serving
in use, with an interval on the comparison**. A source RETIRES C4a if it does that. A source that
fingerprints a model to identify *which* model it is (provenance, ownership, derivative
detection), that monitors an API's accuracy on a benchmark over time without a same-weights floor,
that measures nondeterminism without using it to grade a comparison, or that verifies inference
cryptographically, OCCUPIES: the same practice on a different object or without the floor. This is
written down now so that no reading can widen or narrow the object to fit a verdict.

## The source list, closed at this commit

Named by title, author and year; located by those; a source that cannot be located is UNFETCHABLE,
not replaced. Nothing is added after this commit; a lead met while fetching is recorded under
`leads_not_scored` and not scored. Chosen from pass 2's forty leads by one rule: every lead a
reader named as a candidate for C2, C4a or C5, except those whose bearing the reader recorded as a
method already priced (CKA beside RSA), evidence about a floor rather than a practice (Random123,
splittable PRNGs, Bowyer et al. on intervals, Lyu et al. on generation vs probability), or a
neighbour of a retired clause (Bayer–Haber–Stornetta 1993 and the Surety advertisements for C4b;
Sigstore's background page beside Rekor for C2). The excluded leads stay leads.

| id | source | might occupy |
|---|---|---|
| L01 | Chen, Cai, Zaharia & Zou, *Did the Model Change? Efficiently Assessing Machine Learning API Shifts*, arXiv 2107.14203 (2021) | C4a |
| L02 | Chen, Jin, Eyuboglu, Ré, Zaharia & Zou, *HAPI: A Large-scale Longitudinal Dataset of Commercial ML API Predictions*, NeurIPS 2022 Datasets and Benchmarks (arXiv 2209.08443) | C4a |
| L03 | Tu et al., *ChatLog: Recording and Analyzing ChatGPT Across Time*, arXiv 2304.14106 (2023) | C4a |
| L04 | Yang & Wu, *A Fingerprint for Large Language Models*, arXiv 2407.01235 (2024) | C4a |
| L05 | Xu et al., *Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression*, EMNLP 2021 (arXiv 2109.03228) | C4a |
| L06 | Hooker, Courville, Clark, Dauphin & Frome, *What Do Compressed Deep Neural Networks Forget?*, arXiv 1911.05248 (2019/2020) | C4a |
| L07 | Atil et al., *Non-Determinism of "Deterministic" LLM Settings*, arXiv 2408.04667 (2024/2025) | C4a |
| L08 | Madaan et al., *Quantifying Variance in Evaluation Benchmarks*, arXiv 2406.10229 (2024) | C4a |
| L09 | Pasquini, Kornaropoulos & Ateniese, *LLMmap: Fingerprinting For Large Language Models*, arXiv 2407.15847 (2024) | C4a |
| L10 | Iourovitski, Sharma & Talwar, *Hide and Seek: Fingerprinting Large Language Models with Evolutionary Learning*, arXiv 2408.02871 (2024) | C4a |
| L11 | Maini, Yaghini & Papernot, *Dataset Inference: Ownership Resolution in Machine Learning*, ICLR 2021 (arXiv 2104.10706) | C4a |
| L12 | Chen et al., *Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models*, IEEE S&P 2022 (arXiv 2112.05588) | C4a |
| L13 | Ghodsi, Gu & Garg, *SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud*, NeurIPS 2017 (arXiv 1706.10268) | C4a |
| L14 | Jagielski, Carlini, Berthelot, Kurakin & Papernot, *High Accuracy and High Fidelity Extraction of Neural Networks*, USENIX Security 2020 (arXiv 1909.01838) | C4a |
| L15 | Narayanan & Kapoor, *Is GPT-4 getting worse over time?*, AI Snake Oil, July 2023 (web) | C4a |
| L16 | ACM, *Artifact Review and Badging — Current* (acm.org policy page, web) | C5 |
| L17 | Sinha, Pineau, Forde, Ke & Larochelle, *NeurIPS 2019 Reproducibility Challenge*, ReScience C 6(2) #5 (2020) | C5 |

Seventeen sources, closed. Fifteen might occupy C4a, because C4a is what the sand is for and it is
the clause the readers' leads point at; two might occupy C5. No lead pointed at C2 except the two
excluded above.

## What the survey must record, per source

The URL as fetched, the fetch time, the sha256 of the bytes saved, READ / SKIMMED / UNFETCHABLE,
the characters of extracted text the reader read, and for each clause in the "might occupy"
column the verdict with the object named, one verbatim quote from the source that bears on it,
and one sentence of reason. Per clause: the status after this pass, the retiring and occupying
sources, the nearest neighbour. The conjunction's status. The surviving sentence, verbatim, or the
word RETIRED. All of it in `sand_prior_art_survey_pass3_2026_09_14.json`, built mechanically from
the readers' returns by `build_sand_survey_pass3.py`; the SURVEY document swears to it. The
surveyor types no verdict.

## What this procedure does not do

It does not search beyond the list. It does not price "revolutionary", "novel" or "first". It
does not re-open C3 or C4b. It does not price the sand's value, only its neighbours.
