"""Single source of truth for the styxx version.

This module is intentionally trivial — it contains ONLY the version literal and
no imports — so that:

  * setuptools reads it at build time via ``[tool.setuptools.dynamic]``
    (``version = {attr = "styxx._version.__version__"}``). Because the value is a
    plain literal, setuptools AST-parses it *without importing the package*, so
    the heavy ``styxx`` import graph never runs during a build; and

  * ``styxx/__init__.py`` can import it on a bare source checkout with no install,
    making ``styxx.__version__`` reflect the *code that is actually running*
    rather than whatever happens to be in the installed distribution metadata.

The version stamped into every attestation / vitals receipt is derived from here.
A divergence between this literal and the installed-distribution metadata signals
a stale install and is surfaced via ``styxx.__version_mismatch__``.

To cut a release, bump THIS literal. ``pyproject.toml`` reads it dynamically, so
the two can never drift.
"""

__version__ = "7.24.2"
