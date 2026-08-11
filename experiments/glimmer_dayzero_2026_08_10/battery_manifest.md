# battery manifest — muse-glimmer-30b day-zero cognometric eval

date: 2026-08-10
file: `battery.jsonl` (40 items, one JSON object per line, UTF-8 no BOM)

## provenance

this battery was **authored fresh for the day-zero run on 2026-08-10**. it was
**not sampled from production workload**: the workload log carries gate telemetry
only and contains no reusable prompts, so there was nothing to sample from.
every prompt was written by hand for this run.

## category counts

| category              | count | design intent                                                        |
|-----------------------|-------|----------------------------------------------------------------------|
| tool_planning         | 7     | multi-step task decomposition (pipelines, debugging, migrations)      |
| factual_recall        | 7     | short-answer facts with unambiguous answers, varied domains           |
| refusal_boundary      | 6     | asks a well-calibrated model should decline or heavily caveat         |
| calibration_bait      | 6     | genuinely uncertain/contested questions; confident assertion = fail   |
| instruction_following | 7     | precise format constraints (word counts, forbidden words, structure)  |
| code_reasoning        | 7     | small bug-spotting/comprehension tasks with definite answers, mixed languages |
| **total**             | **40**|                                                                       |

ids run `bat-001` through `bat-040`, unique, schema `{"id", "category", "prompt"}`.

## authorship + no-overlap check

all 40 prompts were authored for this battery on 2026-08-10. none were copied
from TruthfulQA, from the styxx attack seed pools
(`styxx/attack/seeds/*.jsonl`), or from the knowsay pools
(`papers/agent-conscience/squad_pool*.json`).

an automated check was run in addition to the spot-check: every battery prompt
was compared against all 2,790 question/task strings extracted from the attack
seeds and knowsay pools (token-level jaccard similarity, stopwords removed).
two near-collisions were found during authoring and reworded before freeze:

- a bone-count question overlapped an overconfidence seed ("how many bones are
  in the adult human body?") — replaced with a chromosome-count question.
- a "chemical symbol for X" question shared its template with a seed
  ("chemical symbol for gold") — replaced with a melting-point question.

after rewording, maximum jaccard against any source string is 0.375, and the
top remaining matches are semantically different questions. no battery item
duplicates or paraphrases a source item.

## additional constraints honored

- no real personal names or personal data
- no current-events claims that can silently go stale
- no prompt requires web access or images; text-only, single-turn, self-contained
- english throughout

## distribution

`battery.jsonl` ships publicly alongside the day-zero report so third parties
can re-run the battery verbatim and compare results.
