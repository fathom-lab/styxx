# -*- coding: utf-8 -*-
"""Guard: the README never advertises a `styxx.X` attribute that doesn't resolve.

7.17.2 fixed `styxx.mind` (headlined at README.md but raising AttributeError because
the submodule was never imported into the package namespace). This test pins that
class of overclaim shut: every `styxx.<name>` the README references must resolve as
an attribute or an importable submodule, and every name in `__all__` must resolve.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import styxx

# `styxx.<name>` tokens that are NOT Python attribute references (URLs, domains).
# Anything not in here MUST resolve, so a future styxx.org-style addition is the
# only thing that needs to be allow-listed — a real missing API still fails.
_NON_API_TOKENS = {"org", "dev"}

_README = Path(__file__).resolve().parent.parent / "README.md"
# Match `styxx.<name>` but NOT when it sits in a URL path or filename
# (e.g. `pypi/v/styxx.svg`) — the negative lookbehind drops a preceding `/`.
_TOKEN_RE = re.compile(r"(?<![\w/])styxx\.([A-Za-z_][A-Za-z0-9_]*)")


def _resolves(name: str) -> bool:
    if hasattr(styxx, name):
        return True
    # submodule that exists but isn't imported into the namespace is still a
    # valid `import styxx.<name>` reference
    return importlib.util.find_spec(f"styxx.{name}") is not None


def test_readme_styxx_attributes_resolve():
    text = _README.read_text(encoding="utf-8")
    referenced = sorted(set(_TOKEN_RE.findall(text)) - _NON_API_TOKENS)
    unresolved = [n for n in referenced if not _resolves(n)]
    assert not unresolved, (
        f"README references styxx.{{{', '.join(unresolved)}}} which do not resolve "
        f"as an attribute or submodule — either expose them in styxx/__init__.py or "
        f"correct the README (this is the styxx.mind regression guard)."
    )


def test_all_exports_resolve():
    missing = [n for n in styxx.__all__ if not hasattr(styxx, n)]
    assert not missing, f"__all__ names that do not resolve: {missing}"


def test_pyproject_summary_within_pypi_limit():
    """PyPI rejects an upload whose Summary (pyproject description) exceeds 512
    chars (core-metadata rule). Guard it locally so a description edit can't fail
    the publish workflow after the tag is already pushed. Regex-extracted (no
    tomllib) so it runs on the full 3.9+ support range."""
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^description = "(.*)"$', pyproject, re.MULTILINE)
    assert m, "could not find single-line description in pyproject.toml"
    desc = m.group(1)
    assert len(desc) <= 512, f"pyproject description is {len(desc)} chars; PyPI caps Summary at 512"
