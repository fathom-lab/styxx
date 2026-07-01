# -*- coding: utf-8 -*-
"""Import-time regression guards (perf-and-weight, Phase 1).

`import styxx` used to cost ~2.9s because two flagship auditors (audit_confound / validate_probe) imported
scikit-learn EAGERLY at top level, dragging in sklearn + scipy + pandas. That import is now lazy (deferred to
the first CALL of those functions). These tests lock the win in:

  * the DETERMINISTIC guard — a cold `import styxx` must not pull sklearn / scipy / pandas — is timing-independent
    and catches the exact regression forever;
  * the wall-time budgets back it up with generous CI margin.

Measured (this machine): import styxx 2.9s -> 0.49s; `styxx --help` 2.97s -> 0.70s.
"""
import subprocess
import sys
import time


def _subproc(args):
    t0 = time.perf_counter()
    r = subprocess.run(args, capture_output=True, text=True)
    return time.perf_counter() - t0, r


def test_import_styxx_does_not_pull_heavy_stack():
    """The load-bearing guard: a bare `import styxx` must stay off the sklearn/scipy/pandas graph."""
    code = (
        "import sys, styxx\n"
        "for lib in ('sklearn', 'scipy', 'pandas'):\n"
        "    assert lib not in sys.modules, f'{lib} was imported at `import styxx` time (perf regression)'\n"
    )
    _, r = _subproc([sys.executable, "-c", code])
    assert r.returncode == 0, r.stderr


def test_import_styxx_under_budget():
    dt, r = _subproc([sys.executable, "-c", "import styxx"])
    assert r.returncode == 0, r.stderr
    # generous CI margin; the real regression (eager sklearn) is ~2.9s, comfortably above this.
    assert dt < 2.0, f"import styxx took {dt:.2f}s (budget 2.0s) — a heavy import likely crept back in"


def test_styxx_help_under_budget():
    dt, r = _subproc([sys.executable, "-m", "styxx", "--help"])
    assert r.returncode == 0, r.stderr
    assert dt < 3.0, f"styxx --help took {dt:.2f}s (budget 3.0s)"
