# FINDING — The confident-misconception wall is INTERNAL (the knowledge-presence read fails)

**2026-06-07. Fathom Lab / styxx.** The program's deepest swing at its own named-impossible — the
confident-misconception wall — using the night's read tool. Verdict: **REPORT_AS_LANDED. The wall is
not cracked; it is *internal*.** (The harness auto-label printed "SURVIVED" on a single gate; that is
an artifact — the headline mechanism failed. Honest verdict below.)

## Setup

On the confident-consistent TruthfulQA floor (output signals say "fine"; grown to **n=121**, 70 wrong
/ 51 correct, to escape the n=41 power trap), a **label-free** internal read: the mid-layer (L20/28)
logit-lens mean log-prob of the teacher-forced **gold** answer (R_gold) and the **emitted** answer
(R_emit). Headline mechanism = R_gold separates confidently-correct from confidently-wrong (the
suppressed truth is recoverable mid-layer even when the output commits to wrong). Uses the external
gold reference *text*, never the item's correctness label. Hash-before-score; gold-shuffle null.

## Result

| signal on the floor | AUC |
|---|---|
| output max-entropy (the wall, baseline) | 0.53 (chance — wall confirmed) |
| **R_gold (knowledge-presence — the claimed mechanism)** | **0.519 (CHANCE)** |
| K = R_gold − R_emit | 0.62 (borderline, exactly at the pre-set bar) |
| gold-shuffle null | 0.456 (control behaves) |

## Honest read — why this is bedrock, not a crack

- **The headline mechanism fails.** R_gold alone is at **chance (0.519)**: for confident
  misconceptions the correct answer is **not internally favored mid-layer**. The "we read the
  suppressed truth" claim is **not supported**. There is no recoverable gold to read.
- **The only signal is a faint mid-layer confidence echo, not knowledge.** K=0.62 is carried almost
  entirely by R_emit (≈ −R_emit): confidently-*wrong* answers are mid-layer-*weaker* than correct
  ones — a subtler version of the very confidence signal that already fails at the output, not
  recovered truth. And 0.62 sits **exactly at the bar** (n=121 → 95% CI lower ≈ chance; not
  established), with the full frozen battery (bootstrap CI, shared-myth split, length/norm partials,
  cross-model) **not run**. So it does not clear the wall-crack bar honestly.
- **Output confirmed at chance (0.53).** The wall is real here, as in all prior work.

**The wall is INTERNAL, not merely behavioral.** The field could never crack it from the output
because the error is internally confident; we now have evidence it can't be cracked from the *inside*
either, because for a believed misconception the truth is **genuinely not represented to be read**
(consistent with the program's shared-myth / D2-content-converges results: the myth is encoded *as*
knowledge). That is a sharp, real characterization — the wall is bedrock because there is nothing to
recover, not merely because the output hides it.

## Discipline note

The auto-verdict said SURVIVED (it checks only K ≥ 0.62). The honest verdict is REPORT_AS_LANDED: the
knowledge-presence mechanism is at chance, the K signal is a borderline emitted-answer-strength echo,
and the full gate battery was not run. Calling this a "wall crack" would be exactly the hype styxx's
own dogfood flagged hardest this session. It is not a crack.

## Scope / owed

Qwen-1.5B, one layer (L20), one model, single run, K-channel only (the D-channel retrieval-vs-
construction dynamics and the bootstrap-CI + shared-myth split + cross-model confirmation are owed
before even the "internal" framing is final). The honest landing: **strong evidence the
confident-misconception wall is internal/bedrock; no crack.**
