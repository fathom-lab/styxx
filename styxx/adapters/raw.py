# -*- coding: utf-8 -*-
"""
styxx.adapters.raw — the no-SDK adapter.

For users who already have logprob trajectories in hand. You pass
entropy, logprob, and top2_margin arrays; styxx returns a full
Vitals object with phase readings, classification, and optional
ASCII card rendering.

This is the most portable adapter — it has zero SDK dependencies
and works with any LLM whose logprobs you can capture yourself.

Usage:
    from styxx import Raw
    styxx = Raw()
    vitals = styxx.read(
        entropy=[2.1, 1.9, 1.8, ...],
        logprob=[-0.4, -0.3, -0.5, ...],
        top2_margin=[0.5, 0.4, 0.6, ...],
    )
    print(vitals.summary)
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core import StyxxRuntime
from ..vitals import Vitals


class RawAdapter:
    """Direct logprob-trajectory input. No SDK required.

    This is the adapter used by `styxx scan <file>` and by unit
    tests that want determinism without network calls.
    """

    def __init__(self, runtime: Optional[StyxxRuntime] = None):
        self.runtime = runtime or StyxxRuntime()

    def read(
        self,
        entropy: Sequence[float],
        logprob: Sequence[float],
        top2_margin: Sequence[float],
    ) -> Vitals:
        """Compute vitals for a pre-captured logprob trajectory.

        All three sequences must have the same length. The runtime
        will compute every phase whose token window fits in the
        trajectory — phases whose cutoff exceeds len(entropy) are
        left as None in the returned Vitals.
        """
        if len(entropy) != len(logprob) or len(entropy) != len(top2_margin):
            raise ValueError(
                f"raw adapter requires aligned trajectories; "
                f"got entropy={len(entropy)}, logprob={len(logprob)}, "
                f"top2_margin={len(top2_margin)}"
            )
        return self.runtime.run_on_trajectories(
            entropy=entropy, logprob=logprob, top2_margin=top2_margin,
        )
