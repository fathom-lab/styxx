"""I_rg + D_cont — API re-generation entropy and continuation divergence.

For a given (system_prompt, user_prompt, draft_text):
- Re-generate at T=0 with each scoring model (gpt-4o-mini, gpt-4.1-mini).
- Capture per-token logprobs, compute entropy slope.
- Compare regenerated text to draft_text (Levenshtein on token strings;
  cosine on sentence-embedding via OpenAI text-embedding-3-small).

Honest scope (per PROTOCOL Amendment 2):
- I_rg measures the scorer's own uncertainty on its own deterministic
  continuation. NOT the agent's gen-time uncertainty on the draft.
- D_cont measures whether peer models would produce similar text under the
  same prompt.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

SCORING_MODELS = ["gpt-4o-mini", "gpt-4.1-mini"]
MAX_TOKENS = 256


@dataclass
class RegenResult:
    model: str
    text: str
    n_tokens: int
    mean_logprob: float
    mean_entropy_topk_nats: float
    entropy_slope: float | None
    entropy_curvature: float | None
    entropy_volatility: float | None
    distance_to_draft_levenshtein: int | None = None
    distance_to_draft_normalized: float | None = None
    elapsed_s: float = 0.0


def _entropy_topk(top_logprobs: list[Any]) -> float:
    if not top_logprobs:
        return 0.0
    lps = [tlp.logprob for tlp in top_logprobs]
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    z = sum(exps)
    probs = [e / z for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _slope_curve_vol(xs: list[float]) -> tuple[float | None, float | None, float | None]:
    n = len(xs)
    if n < 3:
        return None, None, None
    idx = list(range(n))
    mx = sum(idx) / n
    my = sum(xs) / n
    num = sum((idx[i] - mx) * (xs[i] - my) for i in range(n))
    den = sum((idx[i] - mx) ** 2 for i in range(n))
    slope = num / den if den else None
    diffs = [xs[i + 1] - xs[i] for i in range(n - 1)]
    volatility = sum(abs(d) for d in diffs) / len(diffs) if diffs else None
    if len(diffs) >= 2:
        second = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        curvature = sum(abs(d) for d in second) / len(second)
    else:
        curvature = None
    return slope, curvature, volatility


def _levenshtein_tokens(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def regenerate_and_score(
    system_prompt: str,
    user_prompt: str,
    draft_text: str,
    models: list[str] | None = None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """Run I_rg + D_cont for one draft. Returns dict ready to ship as JSON."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    client = OpenAI()
    models = models or SCORING_MODELS

    results: list[RegenResult] = []
    draft_tokens = draft_text.split()
    for m in models:
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_completion_tokens=max_tokens,
                logprobs=True,
                top_logprobs=5,
            )
        except Exception as e:
            results.append(RegenResult(
                model=m, text=f"<error: {type(e).__name__}: {e}>",
                n_tokens=0, mean_logprob=float("nan"),
                mean_entropy_topk_nats=float("nan"),
                entropy_slope=None, entropy_curvature=None,
                entropy_volatility=None, elapsed_s=time.time() - t0,
            ))
            continue
        elapsed = time.time() - t0

        choice = resp.choices[0]
        text = choice.message.content or ""
        content = (choice.logprobs.content if choice.logprobs else None) or []
        Hs = []
        lps = []
        for tlp in content:
            Hs.append(_entropy_topk(tlp.top_logprobs or []))
            lps.append(tlp.logprob)
        slope, curv, vol = _slope_curve_vol(Hs)

        regen_tokens = text.split()
        lev = _levenshtein_tokens(draft_tokens, regen_tokens)
        norm = lev / max(len(draft_tokens), len(regen_tokens)) if max(len(draft_tokens), len(regen_tokens)) else None

        results.append(RegenResult(
            model=m, text=text, n_tokens=len(content),
            mean_logprob=sum(lps) / len(lps) if lps else float("nan"),
            mean_entropy_topk_nats=sum(Hs) / len(Hs) if Hs else float("nan"),
            entropy_slope=slope, entropy_curvature=curv, entropy_volatility=vol,
            distance_to_draft_levenshtein=lev,
            distance_to_draft_normalized=norm,
            elapsed_s=elapsed,
        ))

    # cross-model slope divergence (H4)
    valid_slopes = [r.entropy_slope for r in results if r.entropy_slope is not None]
    slope_divergence = max(valid_slopes) - min(valid_slopes) if len(valid_slopes) >= 2 else None

    # D_cont: mean normalized distance from draft to regenerations
    valid_dists = [r.distance_to_draft_normalized for r in results if r.distance_to_draft_normalized is not None]
    d_cont = sum(valid_dists) / len(valid_dists) if valid_dists else None

    return {
        "scorers": [asdict(r) for r in results],
        "I_rg_slopes": {r.model: r.entropy_slope for r in results},
        "I_rg_slope_divergence": slope_divergence,
        "D_cont": d_cont,
    }
