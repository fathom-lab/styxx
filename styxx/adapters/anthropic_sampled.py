# -*- coding: utf-8 -*-
"""
styxx.adapters.anthropic_sampled — sampling-based entropy for Anthropic.

Anthropic's Messages API does not expose per-token logprobs (as of
2026-04). That makes styxx tier-0 (which reads entropy / top-2 margin
/ logprob from the distribution) mathematically impossible on the
Claude stream.

This adapter reconstructs a *proxy* for those signals without logprobs:

    1. run the same prompt N times at temperature T > 0
    2. prefix-align the N sampled completions token-by-token
    3. at each token position, measure the empirical divergence across
       the N samples:
         - fraction that still agree on the token (proxy for top-2 margin)
         - Shannon entropy over the sampled token distribution
         - branch-rate (first position where samples diverge) → proxy for
           "commitment point"
    4. feed those per-token features into the same phase classifier
       that styxx uses for logprob-based vitals

This is not equivalent to true logprobs — it sees only the realized
token at each step across N chains, not the full distribution. But it
converts a *zero-signal* setting (pure text output, no probabilities)
into a *real-signal* setting (empirical entropy from an ensemble) at
the cost of N× the tokens.

Tradeoff:
    - N=3 samples ≈ 3× token cost, decent top-2 margin proxy
    - N=5 samples ≈ 5× token cost, good entropy estimate
    - N=8+       ≈ approaches true distribution, expensive

Defaults: N=3, T=0.7 — "cheap ensemble" mode.

Usage:
    from styxx.adapters.anthropic_sampled import AnthropicSampled
    client = AnthropicSampled(ensemble_n=3, ensemble_temperature=0.7)
    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": "why is the sky blue?"}],
    )
    print(r.content[0].text)   # one representative sample, chosen by median divergence
    print(r.vitals)            # phase trajectory from empirical entropy
    print(r.ensemble)          # raw per-token divergence features

This is an honest, upstream-limited trick. It is NOT a replacement for
logprobs — it's the best you can do without them.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnsembleReading:
    """Per-token empirical divergence across N sampled completions."""
    n_samples: int
    max_len: int
    per_token_entropy: list[float] = field(default_factory=list)
    per_token_agreement: list[float] = field(default_factory=list)  # frac agreeing on mode
    first_divergence: int = -1  # token index where samples first disagreed (-1 if never)
    chosen_index: int = 0        # which sample we returned as .content


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer. Not BPE-accurate but good
    enough for ensemble alignment when true tokenizer isn't available."""
    import re
    return re.findall(r"\S+|\s+", text)


def _ensemble_features(texts: list[str]) -> EnsembleReading:
    """Compute per-token empirical entropy and agreement across N chains."""
    if not texts:
        return EnsembleReading(n_samples=0, max_len=0)

    chains = [_tokenize_simple(t) for t in texts]
    n = len(chains)
    max_len = max(len(c) for c in chains)

    ents: list[float] = []
    agrees: list[float] = []
    first_div = -1

    for i in range(max_len):
        col = [c[i] if i < len(c) else None for c in chains]
        present = [t for t in col if t is not None]
        if not present:
            break
        counts = Counter(present)
        total = sum(counts.values())
        # Shannon entropy in nats
        H = -sum((v / total) * math.log(v / total) for v in counts.values() if v > 0)
        ents.append(H)
        # fraction agreeing on the modal token
        modal_count = counts.most_common(1)[0][1]
        agrees.append(modal_count / total)
        if first_div == -1 and modal_count < total:
            first_div = i

    return EnsembleReading(
        n_samples=n,
        max_len=max_len,
        per_token_entropy=ents,
        per_token_agreement=agrees,
        first_divergence=first_div,
    )


def _features_to_phase_features(reading: EnsembleReading, window: slice) -> list[float]:
    """Convert ensemble features into the 12-dim feature vector the
    phase classifier expects. Maps:
        entropy_mean, entropy_max, entropy_min, entropy_slope,
        agreement_mean, agreement_min, agreement_slope,
        divergence_rate, normalized_position_of_first_div,
        n_tokens, entropy_last, agreement_last
    This is an approximation — the real features are logprob-derived.
    """
    ents = reading.per_token_entropy[window]
    agrs = reading.per_token_agreement[window]
    if not ents:
        return [0.0] * 12

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def slope(xs):
        if len(xs) < 2:
            return 0.0
        n = len(xs)
        sx = sum(range(n))
        sy = sum(xs)
        sxy = sum(i * v for i, v in enumerate(xs))
        sxx = sum(i * i for i in range(n))
        denom = (n * sxx - sx * sx)
        return (n * sxy - sx * sy) / denom if denom else 0.0

    return [
        mean(ents),
        max(ents),
        min(ents),
        slope(ents),
        mean(agrs),
        min(agrs),
        slope(agrs),
        1.0 - mean(agrs),  # divergence rate
        (reading.first_divergence / max(reading.max_len, 1)) if reading.first_divergence >= 0 else 1.0,
        float(len(ents)),
        ents[-1] if ents else 0.0,
        agrs[-1] if agrs else 0.0,
    ]


class AnthropicSampled:
    """Drop-in around anthropic.Anthropic that runs an N-sample ensemble
    per call and attaches sampling-based vitals to the response."""

    def __init__(self, *args, ensemble_n: int = 3, ensemble_temperature: float = 0.7, **kwargs):
        try:
            from anthropic import Anthropic as _Anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e
        self._client = _Anthropic(*args, **kwargs)
        self._n = ensemble_n
        self._t = ensemble_temperature
        self.messages = _SampledMessages(self._client.messages, n=ensemble_n, temp=ensemble_temperature)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _SampledMessages:
    def __init__(self, inner, n: int, temp: float):
        self._inner = inner
        self._n = n
        self._temp = temp

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, *args, **kwargs):
        """Run N samples at temp=self._temp, align, return the middle-divergence
        sample with ensemble vitals attached."""
        n = self._n
        # honor explicit temperature if caller passed one
        temp = kwargs.pop("temperature", self._temp)
        # single fast path if n<=1
        if n <= 1:
            r = self._inner.create(*args, temperature=temp, **kwargs)
            try:
                r.vitals = None
                r.ensemble = None
            except Exception:
                pass
            return r

        # run N samples
        responses = []
        for _ in range(n):
            resp = self._inner.create(*args, temperature=temp, **kwargs)
            responses.append(resp)

        texts = [_extract_text(r) for r in responses]
        reading = _ensemble_features(texts)

        # pick the sample closest to the ensemble median length (representative)
        lens = [len(t) for t in texts]
        median = sorted(lens)[len(lens) // 2]
        chosen_idx = min(range(n), key=lambda i: abs(lens[i] - median))
        reading.chosen_index = chosen_idx
        chosen = responses[chosen_idx]

        # build vitals from the ensemble features
        vitals = _build_sampled_vitals(reading)

        try:
            chosen.vitals = vitals
            chosen.ensemble = reading
            chosen.ensemble_texts = texts
        except Exception:
            pass
        return chosen


def _extract_text(response: Any) -> str:
    try:
        return "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception:
        return str(response)


def _build_sampled_vitals(reading: EnsembleReading):
    """Convert ensemble features into a styxx-compatible Vitals object."""
    from ..vitals import Vitals, PhaseReading

    # split the token range into 4 phases
    L = reading.max_len
    if L == 0:
        return None
    q = max(1, L // 4)
    phases = {
        "phase1_pre":   slice(0, min(1, L)),
        "phase2_early": slice(0, min(q, L)),
        "phase3_mid":   slice(0, min(2 * q, L)),
        "phase4_late":  slice(0, L),
    }

    readings = {}
    for pname, win in phases.items():
        feats = _features_to_phase_features(reading, win)
        # use the existing classifier
        try:
            from ..runtime import _get_runtime
            rt = _get_runtime()
            clf = getattr(rt, "classifier", None)
            if clf is None:
                raise RuntimeError("no classifier")
            # classifier expects 12-dim; fall back to simple heuristic if mismatch
            pred, margin, dists, probs = _classify(clf, feats)
        except Exception:
            pred, margin, dists, probs = _heuristic_classify(feats)

        readings[pname] = PhaseReading(
            phase=pname.replace("_", ":"),
            n_tokens_used=reading.max_len if pname == "phase4_late" else min(win.stop, L),
            features=feats[:12],
            predicted_category=pred,
            margin=margin,
            distances=dists,
            probs=probs,
        )

    # coherence: inverse of entropy variance
    ents = reading.per_token_entropy
    import statistics
    coh = 1.0 / (1.0 + statistics.stdev(ents)) if len(ents) > 1 else 0.5

    return Vitals(
        phase1_pre=readings["phase1_pre"],
        phase2_early=readings["phase2_early"],
        phase3_mid=readings["phase3_mid"],
        phase4_late=readings["phase4_late"],
        tier_active=0,  # sampling-tier-0
        coherence=coh,
    )


def _heuristic_classify(features: list[float]):
    """Fallback classifier: map ensemble features to phase categories
    via simple thresholds. Used when the centroid classifier isn't
    available or shape-mismatches."""
    ent_mean = features[0]
    agr_mean = features[4]
    div_rate = features[7]

    # high divergence + low agreement → hallucination risk
    # high agreement + low entropy → retrieval
    # moderate divergence + mid agreement → reasoning
    # very high divergence + very low agreement → adversarial
    if agr_mean > 0.85 and ent_mean < 0.5:
        pred = "retrieval"
    elif div_rate > 0.7:
        pred = "hallucination"
    elif agr_mean > 0.6 and ent_mean < 1.2:
        pred = "reasoning"
    elif ent_mean > 2.0:
        pred = "adversarial"
    else:
        pred = "creative"

    cats = ["retrieval", "reasoning", "refusal", "creative", "adversarial", "hallucination"]
    dists = {c: 1.0 - (0.6 if c == pred else 0.2) for c in cats}
    # crude softmax
    import math
    exps = {c: math.exp(-d) for c, d in dists.items()}
    Z = sum(exps.values())
    probs = {c: v / Z for c, v in exps.items()}
    top2 = sorted(probs.values(), reverse=True)[:2]
    margin = top2[0] - top2[1]
    return pred, margin, dists, probs


def _classify(clf, features):
    try:
        import numpy as np
        arr = np.asarray([features[:12]], dtype="float32")
        pred = clf.predict(arr)[0]
        dists = {}
        probs = {}
        if hasattr(clf, "distances"):
            d = clf.distances(arr)[0]
            dists = {c: float(v) for c, v in zip(clf.classes_, d)}
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(arr)[0]
            probs = {c: float(v) for c, v in zip(clf.classes_, p)}
        else:
            probs = {c: 1.0 / len(clf.classes_) for c in clf.classes_}
        top2 = sorted(probs.values(), reverse=True)[:2]
        margin = top2[0] - top2[1] if len(top2) > 1 else 0.0
        return str(pred), float(margin), dists, probs
    except Exception:
        return _heuristic_classify(features)


__all__ = ["AnthropicSampled", "EnsembleReading"]
