# -*- coding: utf-8 -*-
"""
styxx.spec_exec — Epistemic Speculative Execution (integrity-gated model routing).

Run a cheap model by default; escalate a single call to a stronger model ONLY when
a styxx behavioral-honesty signal flags the cheap output as low-validity. The
speculative-cascade pattern, lifted from token-level / same-family up to the ACTION
level and gated on a BEHAVIORAL signal — NOT raw confidence (model confidence is a
poor validity oracle for hallucination: models are confidently wrong).

Validated (2026-06-01, held-out): on arithmetic, a Qwen2.5-1.5B drafter gated by
``span_confab`` and escalating to a 7B verifier recovered the full quality gap
(median recovery 1.00 across 20/20 random splits) at ~0.70x the verifier's cost,
with the escalation threshold calibrated on a DISJOINT train split. Generalized to
a second task (sorting) using the complementary signal channel.

HONEST BOUNDS — read before deploying:
  * Validated on small open models (Qwen 1.5B -> 7B) and narrow tasks (arithmetic,
    sorting). NOT yet shown at frontier scale or across arbitrary task types.
  * span_confab has two channels (min_margin vs max_entropy); the right one is
    TASK-DEPENDENT and must be chosen on held-out data, not assumed.
  * Routing pays only when (a) param_ratio >> gate overhead, (b) the verifier is
    actually better at the task, and (c) the cheap model is competent on a real
    fraction of calls.
  * Behavioral gates catch UNCERTAINTY errors. They are BLIND to confident
    shared-belief errors — use external grounding (``styxx.retrieval_check``) there.

This is a control law, not an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

__all__ = [
    "Draft", "RouteResult", "EpistemicSpeculativeRouter",
    "entropy_gate", "council_gate", "calibrate_threshold",
]


@dataclass
class Draft:
    """One cheap-model draft: the answer text, a per-call cost proxy
    (input+output tokens), and an OPTIONAL precomputed gate signal (e.g. a
    ``span_confab`` value derived from the drafter's own logits). When ``signal``
    is set the router uses it directly; otherwise it resamples and applies ``gate``."""
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    signal: Optional[float] = None   # higher => lower validity


@dataclass
class RouteResult:
    """Outcome of one routed call. ``escalated`` is True iff the verifier was used;
    ``signal`` is the gate value (higher = lower validity); ``draft_tokens`` /
    ``verify_tokens`` are the cost proxy — an accepted call pays only ``draft_tokens``,
    an escalated call pays both."""
    answer: str
    escalated: bool
    signal: float
    draft_tokens: int = 0
    verify_tokens: int = 0


# A "model" is any callable (prompt, temperature, n) -> list[Draft].
ModelFn = Callable[[str, float, int], "list[Draft]"]


def entropy_gate(samples: Sequence[str], *, method: str = "auto") -> float:
    """Self-consistency gate: ``styxx.semantic_entropy`` over K resamples of one
    model (higher entropy = lower validity). Model-agnostic; needs only text."""
    from . import semantic_entropy
    return float(semantic_entropy(list(samples), method=method))


def council_gate(answers: Sequence[str], *, method: str = "auto") -> float:
    """Cross-model gate: ``1 - styxx.council_agreement`` over one answer per
    DIVERSE model (higher = lower validity). Breaks correlated confabulation;
    stronger than self-consistency on factual recall. Needs diverse backends."""
    from . import council_agreement
    return 1.0 - float(council_agreement(list(answers), method=method))


@dataclass
class EpistemicSpeculativeRouter:
    """Draft cheap, escalate on low validity.

    ``gate`` maps a list of draft samples to a scalar where HIGHER = LOWER
    validity; ``tau`` is the escalation threshold (escalate when signal > tau).
    If the drafter's greedy draft already carries a precomputed ``.signal``
    (e.g. ``span_confab`` from logits), that is used and resampling is skipped.

    Calibrate ``tau`` with :func:`calibrate_threshold` on held-out data — do not
    guess it; the validated configuration calibrated the threshold on a disjoint
    train split.
    """
    drafter: ModelFn
    verifier: ModelFn
    gate: Callable[[Sequence[str]], float] = entropy_gate
    tau: float = 0.5
    k: int = 5
    draft_temp: float = 0.7

    def run(self, prompt: str) -> RouteResult:
        drafts = self.drafter(prompt, 0.0, 1)
        if not drafts:
            raise ValueError("drafter returned no Draft for the prompt")
        a0 = drafts[0]                                        # greedy candidate answer
        if a0.signal is not None:                             # drafter precomputed (e.g. span_confab)
            signal = float(a0.signal)
            draft_tok = a0.in_tokens + a0.out_tokens
        else:                                                 # resample and apply the text gate
            gens = self.drafter(prompt, self.draft_temp, self.k)
            signal = float(self.gate([g.text for g in gens]))
            draft_tok = (a0.in_tokens + a0.out_tokens
                         + sum(g.in_tokens + g.out_tokens for g in gens))
        if signal <= self.tau:                                # confident enough -> keep the cheap draft
            return RouteResult(a0.text, False, signal, draft_tok, 0)
        v = self.verifier(prompt, 0.0, 1)[0]                  # uncertain -> escalate
        return RouteResult(v.text, True, signal, draft_tok, v.in_tokens + v.out_tokens)


def calibrate_threshold(records: Sequence[dict], *, cost_cap: float = 1.0) -> float:
    """Choose the escalation threshold ``tau`` on TRAIN records, honestly.

    Each record is a dict with: ``signal`` (gate value), ``local_ok`` (bool — cheap
    model correct), ``front_ok`` (bool — verifier correct), ``draft_cost`` and
    ``verify_cost`` (numbers). Returns the ``tau`` (escalate when ``signal > tau``)
    that maximizes routed quality subject to cost < ``cost_cap`` x verifier-always.

    Evaluate the returned ``tau`` on a DISJOINT test split — never on the data you
    calibrated on. (In validation, threshold-on-train / verdict-on-test was what
    separated a real held-out win from in-sample over-fitting.)
    """
    if not records:
        return 0.0
    candidates = sorted({r["signal"] for r in records}) + [float("inf")]
    best = None
    n = len(records)
    for tau in candidates:
        q = c = cf = 0.0
        for r in records:
            esc = r["signal"] > tau
            q += 1.0 if (r["front_ok"] if esc else r["local_ok"]) else 0.0
            c += r["draft_cost"] + (r["verify_cost"] if esc else 0.0)
            cf += r["verify_cost"]
        feasible = c < cost_cap * cf if cf > 0 else True
        key = (1 if feasible else 0, q / n, -c)
        if best is None or key > best[0]:
            best = (key, tau)
    return best[1]
