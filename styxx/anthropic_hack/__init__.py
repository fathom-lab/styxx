# -*- coding: utf-8 -*-
"""
styxx.anthropic_hack — three complementary approaches to recovering a
cognitive-state signal on Anthropic's Messages API, which does not
expose per-token logprobs.

Modules:
    text_features : cheap text-level heuristic classifier
    consensus     : N-sample ensemble → empirical entropy trajectory
    companion     : local open-weight proxy model (Llama-3.2-1B or fallback)

None of these are a true substitute for tier-0 logprob vitals. Each is
labelled clearly in the resulting Vitals so downstream code can tell
what kind of signal it's looking at.
"""
from . import text_features, consensus, companion  # noqa: F401

__all__ = ["text_features", "consensus", "companion"]
