"""``python -m styxx.v8`` -- the entry point, and nothing else.

The whole command surface is :mod:`styxx.v8.cli` (contract section 9).  This module exists so
that ``python -m styxx.v8 <verb> ...`` runs it, and it deliberately holds no logic of its own:
every decision -- which verbs exist, what the exit code is, what lands on stdout -- belongs to
``cli.run``/``cli.main`` where a test can drive it in-process without a subprocess.

The v8 surface lives ONLY here (GATED S11-01: recommendation (a) -- namespace v8 under
``python -m styxx.v8`` and leave the 7.x top-level ``styxx`` verbs alone until 8.1; the operator
may reverse).  ``styxx/cli.py`` is not imported, wrapped or edited by this package.

``main`` returns the code rather than raising ``SystemExit``, so the only ``sys.exit`` in the
package is the one below, on the ``__main__`` line.
"""
from __future__ import annotations

import sys

from .cli import main

# GATED S11-01: recommendation implemented; operator may reverse.

if __name__ == "__main__":  # pragma: no cover -- exercised by tests/test_v8_cli.py as a subprocess
    sys.exit(main())
