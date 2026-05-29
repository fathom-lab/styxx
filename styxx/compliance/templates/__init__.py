# -*- coding: utf-8 -*-
"""styxx.compliance.templates — paste-and-customize EU AI Act conformity declaration templates.

The templates are markdown files in this package; they are shipped as package data
and accessible at runtime via :func:`load_template`.

Available templates (v0.1, 2026-05-29):

- ``accuracy_declaration``        — Article 15.1(a) instructions-of-use accuracy declaration
- ``robustness_statement``        — Article 15.3 technical-redundancy / fail-safe statement
- ``boundary_statement``          — the seven EU AI Act requirements styxx does NOT cover
- ``sycophancy_disclosure``       — restrained-tech FPR 0.30 construct-ceiling disclosure
- ``injection_resistance_disclosure`` — load-bearing SECURITY MODEL operational requirement

USAGE:

    from styxx.compliance.templates import load_template, list_templates

    print(list_templates())
    # ['accuracy_declaration', 'boundary_statement', ...]

    md = load_template("accuracy_declaration")
    # markdown string ready to write to disk or paste into a doc

The templates ship with the styxx wheel as package data; they are versioned alongside
the calibrated AUC numbers they cite. Bumping styxx without re-deriving the cited
numbers against the deployed version may produce a stale declaration. See
``templates/README.md`` for the version-bump operational guidance.

Not legal advice. Independent legal review required for any production declaration.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["load_template", "list_templates", "TEMPLATE_NAMES"]


TEMPLATE_NAMES: tuple[str, ...] = (
    "accuracy_declaration",
    "robustness_statement",
    "boundary_statement",
    "sycophancy_disclosure",
    "injection_resistance_disclosure",
)


def list_templates() -> tuple[str, ...]:
    """Return the names of all templates shipped at this styxx version.

    The names are stable across patch versions but new templates may be added
    at minor-version bumps (e.g., a few-shot-lie injection-resistance disclosure
    in a future styxx 7.8.x). Check the README at this version for the active set.
    """
    return TEMPLATE_NAMES


def load_template(name: str) -> str:
    """Load a template by name and return the markdown string.

    Args:
        name: one of the names in :data:`TEMPLATE_NAMES`.

    Returns:
        The template markdown as a UTF-8 string. Customize the ``<...>``
        placeholders in the output before writing the declaration; preserve
        the construct-ceiling and boundary disclosures (kill-gates A2 and A3).

    Raises:
        FileNotFoundError: if ``name`` is not a known template.
    """
    if name not in TEMPLATE_NAMES:
        raise FileNotFoundError(
            f"unknown template {name!r}; available: {', '.join(TEMPLATE_NAMES)}"
        )
    # importlib.resources gives us robust access to package data whether the
    # package is installed as an editable, a wheel, or a zipped egg.
    pkg = resources.files(__name__)
    md = (pkg / f"{name}.md").read_text(encoding="utf-8")
    return md
