# Seat instructions v1 — sentence labeling (frozen at prereg freeze)

You are one seat of a blind panel. You will receive a numbered list of sentences drawn from
software-project commit messages. Label EVERY sentence with exactly one of A, B, or C. Judge
only the text of each sentence — you have no diff, no surrounding message, and no files. Do not
read any files or run any commands.

## The labels

- **A** — the sentence asserts at least one specific change made *by this commit* to files or
  code: a file created, modified, or deleted; a symbol (function/class/method) added; a count
  of files or tests changed; a scope claim ("only touches X"). The assertion is checkable in
  principle against the commit's diff.
- **B** — the sentence asserts a result or measurement whose evidence lies outside any diff:
  test totals ("2311 passed"), CI verdicts, measured rates, panel or adjudication numbers.
- **C** — neither: narrative or motivation; reports on the *state* of a file or on *other*
  commits' work; structural headers, section markers, trailers, boilerplate.

## Disambiguation rules — apply these exactly

1. **Subject lines are labeled by content.** "Headers" under C means only structural markers
   and trailers. An imperative or subjectless fragment naming a concrete file/symbol/scope
   change (e.g. "diffgate: promote the never-read band") asserts this commit makes that
   change → A.
2. **Tense and agency.** A bare past- or present-tense action verb with a file or symbol as
   its object, and no other actor named, asserts this commit performed it → A ("Rebuilt
   LEDGER.md"). Perfect/pluperfect and stative constructions → C ("had not been rebuilt",
   "holds", "carries", "is present"). A sentence naming another commit, branch, or prior
   cycle as the actor → C.
3. **Precedence.** A sentence asserting both a change and a result is A — and on such
   sentences additionally set `also_result_clause` to true.

## Output

Return a JSON object: {"labels": [{"id": "<sentence id>", "label": "A"|"B"|"C",
"also_result_clause": true|false}, ...]} — one entry per sentence, every id present exactly
once, `also_result_clause` false unless rule 3 applied.
