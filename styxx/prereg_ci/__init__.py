# -*- coding: utf-8 -*-
"""styxx.prereg_ci: preregistration as CI (prereg-ci).

A PREREG seals one machine-readable decision; on every pull request the verdict of every receipt
is re-derived from the sealed bytes and read against the verdict the receipt and its RESULT
record. ``core`` is the pure half: ``rescore`` re-derives a verdict from a PREREG's gates fence
and a receipt, ``compare`` reads a recorded verdict and digest against it. The modules that read
git, run a locked scorer and print the report are added beside it, step by step, and none of them
is re-exported from ``styxx``.

``VERSION`` names the design's version of the tool, not styxx's (``styxx._version``).
"""
from __future__ import annotations

from styxx.prereg_ci.core import compare, rescore

VERSION = "v0"

__all__ = ["compare", "rescore"]
