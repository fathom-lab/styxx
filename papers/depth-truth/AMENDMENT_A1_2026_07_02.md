# AMENDMENT A1 — span rule freeze (PREREG_v2 §1/§9)

**Ratified by flobi 2026-07-02 ("keep going" on the presented §9 checkpoint packet, which recommended
exactly this freeze). Committed BEFORE any main-run item. The only element the pilot was permitted to tune.**

- **Span rule = Appendix B steps 1–5, VERBATIM, unchanged.** Pilot evidence: extraction-clean 19/20 (95%),
  19/20 first-content-token targets were real word tokens, and the single failure (answer "A" → leading-article
  strip → empty target) was correctly excluded as depth_undefined rather than scored.
- Aggregation stays fixed = single-token (frozen in §1; not tunable).
- KG0 record: machine half PASS (95% ≥ 90%). Human half: flobi reviewed the 20-row eyeball table with two
  pre-flagged rows; ruling per the packet — **sfq_3248 is a LABEL defect** (TriviaQA aliases `["L'ABNER",
  "l abner"]` lack the correct spelling; entity fields empty; grader executed §3 exactly as frozen), not a
  grading defect; sfq_9476 is ambiguous (bathos/anti-climax) and does not count against the bar. KG0 PASSES
  with a documented label-noise floor: 1/20 visible false-negative (~5%), carried into interpretation.
