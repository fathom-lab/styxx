# -*- coding: utf-8 -*-
"""Worked example: using styxx.compliance.eu_ai_act in a high-risk-AI conformity workflow.

This is illustrative usage code. It is NOT legal advice. Run it locally with:

    pip install styxx==7.7.10
    python examples/eu_ai_act_compliance_example.py

The example demonstrates four practical operations:

  1. Cite a specific EU AI Act clause → get the styxx primitive coverage.
  2. Iterate the full v0.1 coverage table → render to a Markdown summary.
  3. List uncovered EU AI Act requirements → render the honest boundary statement.
  4. Generate a paste-ready Markdown excerpt suitable for the
     "accuracy metrics" section of an Article 15.1(a) instructions of use.

Output is plain stdout so the example is runnable in CI / notebook / shell.
"""
from __future__ import annotations

from styxx.compliance import (
    cite,
    coverage_table,
    uncovered_requirements,
)


def demo_cite_single_clause():
    print("=" * 72)
    print("1) cite('Article 15.1(a)') — single-clause lookup")
    print("=" * 72)
    m = cite("Article 15.1(a)")
    if m is None:
        print("  (not in v0.1 registry)")
        return
    print(f"  clause:           {m.clause}")
    print(f"  requirement text: {m.requirement_text}")
    print(f"  primitives ({len(m.styxx_primitives)}):")
    for p in m.styxx_primitives:
        print(f"    - {p.primitive}")
        print(f"        metric:           {p.calibrated_metric[:80]}...")
        print(f"        construct ceiling: {p.construct_ceiling[:80]}...")
        print(f"        receipt commit:   {p.receipt_commit}")
    print(f"  notes: {m.notes}")
    print()


def demo_full_coverage_table():
    print("=" * 72)
    print("2) coverage_table() — full v0.1 coverage")
    print("=" * 72)
    table = coverage_table()
    print(f"  v0.1 maps {len(table)} Article 15 sub-paragraphs:")
    for m in table:
        primitive_names = [p.primitive for p in m.styxx_primitives]
        if primitive_names:
            print(f"    {m.clause}: {len(m.styxx_primitives)} primitives → "
                  f"{', '.join(primitive_names[:3])}"
                  + ("..." if len(primitive_names) > 3 else ""))
        else:
            print(f"    {m.clause}: (none — honestly empty, see notes)")
    print()


def demo_uncovered_requirements():
    print("=" * 72)
    print("3) uncovered_requirements() — honest boundary statement")
    print("=" * 72)
    uncovered = uncovered_requirements()
    print(f"  v0.1 explicitly does NOT cover {len(uncovered)} requirements:")
    for u in uncovered:
        print(f"    {u.clause}")
        print(f"      reason:       {u.reason[:80]}...")
        print(f"      alternative:  {u.alternative[:80]}...")
    print(f"\n  Kill-gate A3 holds: {len(uncovered)} uncovered >= {len(coverage_table())} covered")
    print()


def demo_render_instructions_of_use_excerpt():
    """Render a paste-ready Markdown excerpt for Article 15.1(a) "instructions of use" accuracy section."""
    print("=" * 72)
    print("4) render_instructions_of_use_excerpt() — Article 15.1(a) starter")
    print("=" * 72)
    m = cite("Article 15.1(a)")
    if m is None:
        return

    md_lines = [
        "## Accuracy Metrics (per EU AI Act Article 15.1(a))",
        "",
        f"*{m.requirement_text}*",
        "",
        "The following calibrated metrics characterize this system's accuracy:",
        "",
    ]
    for p in m.styxx_primitives:
        md_lines.append(f"### Metric source: `{p.primitive}`")
        md_lines.append("")
        md_lines.append(f"**Calibrated value:** {p.calibrated_metric}")
        md_lines.append("")
        md_lines.append(f"**Known limitations (construct ceiling):** {p.construct_ceiling}")
        md_lines.append("")
        md_lines.append(f"**Reproducibility receipt:** see commit `{p.receipt_commit}` in "
                        f"`{p.receipt_doc}` (`fathom-lab/styxx@main`).")
        md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append(f"*{m.notes}*")
    md_lines.append("")

    excerpt = "\n".join(md_lines)
    print(excerpt)
    print()
    print(f"  ({len(excerpt)} chars; copy into operator-authored instructions of use)")
    print()


def main() -> int:
    demo_cite_single_clause()
    demo_full_coverage_table()
    demo_uncovered_requirements()
    demo_render_instructions_of_use_excerpt()

    print("=" * 72)
    print("REMINDER: NOT LEGAL ADVICE.")
    print("Independent conformity review required for any production deployment.")
    print("See papers/EU_AI_ACT_COMPLIANCE_2026.md for methodology + scope.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
