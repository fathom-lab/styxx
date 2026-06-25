# -*- coding: utf-8 -*-
"""The rigor gate, as a BLOCKING test — "the lab that doesn't overclaim", by construction.

Fails the build if any committed result JSON asserts a strong positive verdict ("robust / significant /
real / proven / confirmed / undeniable / generalizes / established") WITHOUT attached uncertainty
quantification (CI / bootstrap / permutation-p) or an explicit disclosure (corrigendum / hedged verdict).

History (2026-06-24/25): two overclaims — genmatch_xvendor 'RESIDUAL ROBUST' (no CIs) and crossfamily's
post-hoc power floor — had shipped to the public record and were caught only by a hand-run adversarial pass.
This gate turns that vigilance into infrastructure: claim a win, show error bars. See scripts/rigor_gate.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rigor_gate import SCAN_DIRS, scan_file  # noqa: E402


def test_no_overclaims_in_committed_result_jsons():
    flagged = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.json")):
            if "result" in f.name.lower():
                fl = scan_file(f)
                if fl:
                    flagged.append((str(f.relative_to(ROOT)), fl))
    assert not flagged, (
        "rigor-gate: strong-claim verdict(s) without uncertainty quantification or disclosure:\n"
        + "\n".join(f"  {f}\n    {fl}" for f, fl in flagged)
        + "\n  Fix: attach a CI / bootstrap / permutation-p, hedge the verdict, or add a corrigendum."
    )
