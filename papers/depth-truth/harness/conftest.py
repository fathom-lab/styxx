"""Test config for the depth-truth harness.

PREREG §2 fixes the real bootstrap at 10,000 resamples, and the H2/H3 paired bootstrap REFITS the logistic
models on every resample — correct, but ~20k sklearn fits per call, which makes the synthetic unit tests time
out. The tests only verify the LOGIC (perfect predictor -> AUROC 1.0, noise -> CI includes 0.5, redundant depth
-> no false additivity), not the tail resolution, so we shrink the resample count for tests only. Production —
the pilot/main runbook — imports analysis untouched and uses the frozen 10,000 default.
"""
import pytest

import analysis


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    # h1 reads _N_BOOT at call time (bootstrap_auroc_ci(..., n=_N_BOOT)); h2/h3 loop range(_N_BOOT).
    # Patching the module global makes all three fast. Seed stays 7, so results remain deterministic.
    monkeypatch.setattr(analysis, "_N_BOOT", 500, raising=True)
