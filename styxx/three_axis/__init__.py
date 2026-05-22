"""styxx.three_axis — send-time three-axis cognometric gate.

Strictly additive. Gated behind STYXX_THREE_AXIS=1. No effect on default
pipeline. See papers/three-axis-sendtime-gate/PROTOCOL.md for preregistered
hypotheses and method.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("STYXX_THREE_AXIS", "0") == "1"


__all__ = ["enabled"]
