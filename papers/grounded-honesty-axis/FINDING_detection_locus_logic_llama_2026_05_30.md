# FINDING — cross-architecture replication of single-pass legibility on MULTI-HOP LOGIC is INFEASIBLE on Llama-3.2-1B: a COMPETENCE-FLOOR boundary, not a verdict — Llama-1B answers "2" reflexively to all 24 easy ordering questions (no genuine CORRECT class), so the confab-vs-correct comparison is undefined; the LOGIC domain's inference-depth difficulty pushes the 1B model below the floor where ARITHMETIC did not

**Run 2026-05-30. Pre-confirmatory feasibility probe (greedy one-shot on all 48 seeded items,
no resampling), the standard powering/competence check BEFORE any confirmatory run.** The Qwen
logic detection-locus finding (`[[FINDING_detection_locus_logic]]`, REPORT_AS_LANDED, single-pass
≈ resampling across three domains) raised the obvious symmetry question: arithmetic single-pass
legibility was confirmed on TWO architectures (Qwen2.5-1.5B AUC 0.92, Llama-3.2-1B AUC 1.00,
`[[FINDING_detection_locus_llama]]`), but code and logic are Qwen-only. Does the LOGIC finding
replicate on Llama-1B? Receipt: `detection_locus_logic_result_Llama-3_2-1B-Instruct.json`.

## Why no confirmatory run: the CORRECT class cannot be genuinely populated

The detection-locus protocol needs a CONFAB class (hard items the model gets wrong) AND a CORRECT
class (easy items the model gets right by reasoning). On the same seeded 48-item set
(hash `97d81680…e02c`), Llama-3.2-1B greedy one-shot:

| | members | total | note |
| --- | --- | --- | --- |
| HARD → confab | 21 | 24 | confabs fine (high single-pass entropy 1.5–2.4) |
| EASY → correct | **6** | 24 | **below the ≥12 power bar** |

The 6 "correct" are an artifact, not competence:

| EASY metric | value |
| --- | --- |
| v1 (greedy answer) distribution over 24 easy items | **`{2: 24}`** |
| true-answer distribution | `{1: 18, 2: 6}` |
| correct items, by true answer | **`{2: 6}`** |

**Llama-1B answers "2" to every easy ordering question, regardless of the facts.** The only items
it gets "right" are exactly the 6 whose true answer happens to be 2 — reflex-match, not reasoning.
There is no competent CORRECT class. A forced AUC on this set would separate *confident reflex*
(the answer-2 items: Llama is sure it's 2, so low entropy / high stability) from *uncertain confab*
(the hard items) — which is NOT the pre-registered competence-correct-vs-confab comparison. Reporting
it would be a contaminated, misread-inviting number. So no AUC is reported and no confirmatory run
was made.

## The claim that lands

1. **Llama-3.2-1B is below the competence floor for multi-hop transitive-ordering logic.** It does
   not perform the counting-in-an-order task at all (reflexive constant "2"), so it cannot serve as
   the second architecture for the LOGIC domain. This is a competence boundary, reported honestly —
   not a legibility verdict, and explicitly NOT gamed by hand-selecting an "easy" tier the model
   happens to match.
2. **The cross-architecture and domain-general claims are unaffected and now sharply scoped.**
   Single-pass legibility is: (a) cross-architecture on ARITHMETIC (Qwen + Llama, where Llama was
   competent); (b) derivation-domain-general across THREE domains on Qwen (arithmetic, code, logic).
   The intersection — Llama × logic — is simply below the floor: the domain's inference-depth
   difficulty is what a 1B model cannot clear, exactly the difficulty axis the domain was built to
   probe. The two confirmed axes do not require this cell.
3. **The difficulty confound has teeth (a feature, not a bug).** That a weaker architecture fails
   even the *easy* logic tier — while clearing the easy arithmetic tier — is direct evidence that
   the HARD/EASY gap is real graded difficulty, not a labeling artifact. It is also why a single
   small model cannot anchor every (architecture × domain) cell.

## Honest scope (pre-committed)

Single second open model Llama-3.2-1B-Instruct; multi-hop transitive-ordering logic; pre-confirmatory
feasibility probe only (greedy one-shot, no resampling); a competence-floor boundary, not a
legibility verdict; no confirmatory AUC because the CORRECT class is reflexive, not competent.
Reproduce: `python papers/grounded-honesty-axis/run_detection_locus_logic.py --model
meta-llama/Llama-3.2-1B-Instruct` (per-item greedy v1 in stdout shows the constant "2"). The
Qwen2.5-1.5B logic finding (REPORT_AS_LANDED, three-domain domain-general) and the Llama arithmetic
finding (cross-architecture) both stand. A stronger small model (e.g. Llama-3.2-3B, Gemma-2-2B) able
to clear the easy logic tier could in principle close the Llama×logic cell; not attempted here.

## The arc, in one line (updated)

Single-pass confabulation legibility is architecture-general on arithmetic (Qwen+Llama) and
derivation-domain-general on Qwen (arithmetic+code+logic) — entropy/margin ≈ resampling, AUC
0.91–1.00; the one cell that can't be measured, Llama×logic, is a COMPETENCE FLOOR (the 1B model
can't do the easy logic), which confirms the difficulty axis is real rather than weakening the
claim; and every signal here still moves confidence/abstention, never correctness.
