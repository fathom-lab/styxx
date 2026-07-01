"""Signal extraction and exclusion rules for the depth-vs-truth keystone.

Implements PREREG.md (papers/depth-truth/PREREG.md, FROZEN):
  - Appendix B: answer-extraction rule (extract_answer).
  - Appendix C: refusal-marker list + substring test (REFUSAL_MARKERS, is_refusal).
  - Section 4: per-item confidence signals LP_mean, LP_norm, and the discrete
    short-form semantic entropy SE (lp_mean, lp_norm, semantic_entropy).
  - Section 5: the exclusion classifications these functions feed
    (excluded_flag in {None, "nonanswer", "depth_undefined", "grade_ambiguous"}).

Pure math on PROVIDED inputs only. NO model calls, NO GPU, NO network — the model
glue (greedy/sampled generation, token logprobs, and the `depth` value) lives in
the GPU-time pipeline elsewhere.

The `depth` signal (§4) is NOT computed here: it comes from `get_mean_depth`
(reused VERBATIM from research/experiment_12_power.py@fc6f2c3, §1) under the frozen
A1 adaptation, and is only obtainable at GPU run time. This module contributes the
answer string, the two logprob-derived confidence signals, and the semantic-entropy
opponent baseline that depth is tested against (H2/H3, §2).

Shared results-row contract (written per item to results/*.jsonl by the pipeline):
  {id, prompt_hash, answer, correct, LP_mean, LP_norm, SE, depth, excluded_flag}
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

__all__ = [
    "extract_answer",
    "REFUSAL_MARKERS",
    "is_refusal",
    "lp_mean",
    "lp_norm",
    "semantic_entropy",
]

# Maximum whitespace-tokenized answer length before an extraction is treated as
# runaway generation (Appendix B step 4). Answers with MORE than this many tokens
# are flagged "nonanswer".
_MAX_ANSWER_TOKENS = 32


def extract_answer(gen: str) -> Tuple[str, Optional[str]]:
    """Appendix B answer-extraction rule, EXACTLY.

    Given the greedy continuation ``gen`` (text generated after the prompt),
    returns ``(answer, excluded_flag)`` where ``excluded_flag`` is either
    ``None`` (usable answer) or ``"nonanswer"`` (§5).

    Steps (Appendix B):
      1. ``ans = gen.split("\\n", 1)[0]`` — everything up to the first newline
         (the stop token).
      2. ``ans = ans.strip()`` — strip leading/trailing whitespace.
      3. Empty after stripping           -> ``("", "nonanswer")``.
      4. Whitespace-token length > 32     -> ``(ans, "nonanswer")`` (runaway).
      5. Otherwise                        -> ``(ans, None)``.

    Refusal detection (Appendix C) is a SEPARATE §5 check on the *normalized*
    answer and is not applied here (this function does not normalize); the caller
    normalizes per §3 and then calls ``is_refusal`` before grading.

    No sentence-splitting, no trailing-period trimming, no model post-processing.
    """
    # Step 1: take everything up to (and excluding) the first newline.
    ans = gen.split("\n", 1)[0]
    # Step 2: strip surrounding whitespace.
    ans = ans.strip()
    # Step 3: empty extraction is a nonanswer; return the empty string exactly.
    if ans == "":
        return "", "nonanswer"
    # Step 4: whitespace-tokenized length > 32 is runaway generation.
    if len(ans.split()) > _MAX_ANSWER_TOKENS:
        return ans, "nonanswer"
    # Step 5: usable answer, no exclusion from this rule.
    return ans, None


# Appendix C — refusal-marker list, EXACT and in file order. A normalized answer
# that CONTAINS any of these as a substring (case-insensitive) is excluded as a
# nonanswer (§5). Matching is done on the §3-normalized answer, so these markers
# are written lowercase to match the normalized form.
REFUSAL_MARKERS: List[str] = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "not sure",
    "no idea",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i cannot",
    "i can't",
    "unknown",
    "n/a",
    "as an ai",
]


def is_refusal(answer_norm: str) -> bool:
    """Return True iff the normalized answer contains any Appendix C marker.

    Case-insensitive substring match (Appendix C). ``answer_norm`` is expected to
    already be the §3-normalized answer string; this function lowercases both
    sides defensively so the check is robust to un-normalized input. A True result
    means ``excluded_flag = "nonanswer"`` (§5).
    """
    a = answer_norm.lower()
    return any(marker in a for marker in REFUSAL_MARKERS)


def lp_mean(token_logprobs: List[float]) -> float:
    """LP_mean (§4): mean per-token logprob over the answer tokens.

    ``token_logprobs`` is the greedy-pass per-answer-token logprob sequence.
    Raises ValueError on an empty sequence (an item with zero answer tokens is a
    "nonanswer" upstream and must not reach this signal).
    """
    if len(token_logprobs) == 0:
        raise ValueError("lp_mean: token_logprobs is empty (no answer tokens)")
    return math.fsum(token_logprobs) / len(token_logprobs)


def lp_norm(seq_logprob: float, n_answer_tokens: int) -> float:
    """LP_norm (§4): sequence logprob divided by answer token count.

    ``seq_logprob`` is the total greedy-pass logprob of the answer span;
    ``n_answer_tokens`` is the number of tokens in that span. Raises ValueError if
    the token count is <= 0 (guards divide-by-zero; a zero-token answer is a
    "nonanswer" upstream).
    """
    if n_answer_tokens <= 0:
        raise ValueError(
            "lp_norm: n_answer_tokens must be > 0 (got %r)" % (n_answer_tokens,)
        )
    return seq_logprob / n_answer_tokens


def semantic_entropy(normalized_samples: List[str]) -> float:
    """SE (§4): discrete short-form semantic entropy, in NATS.

    Shannon entropy of the empirical distribution over DISTINCT normalized answer
    strings among the K sampled continuations (K=5 expected; temp 0.7, §4). This
    is the discrete short-form variant declared in the PREREG: exact-string
    equality clusters the samples — NO entailment/NLI clustering.

    H = -sum_i p_i * ln(p_i),  p_i = count_i / K   (natural log => nats)

    Range: 0.0 when all K samples share one distinct string (maximal confidence),
    up to ln(K) when all K are distinct. Raises ValueError on an empty sample
    list (SE is undefined with no samples).
    """
    n = len(normalized_samples)
    if n == 0:
        raise ValueError("semantic_entropy: normalized_samples is empty")

    # Empirical counts over distinct exact-string clusters.
    counts: dict[str, int] = {}
    for s in normalized_samples:
        counts[s] = counts.get(s, 0) + 1

    # Shannon entropy in nats over the empirical distribution.
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    # Clamp a -0.0 (single-cluster case) to a clean 0.0.
    return h if h != 0.0 else 0.0
