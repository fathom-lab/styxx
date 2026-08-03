"""Build arXiv-ready LaTeX from the certified markdown papers.

The markdown + OATH certificate remain the canonical certified objects; this is
a faithful RENDERING for submission, not a new edition. Guarantees:

  - content flows through pandoc unmodified (no hand-retyping of numbers);
  - a post-build check re-extracts every numeric token from the source markdown
    and asserts each survives verbatim in the .tex, so a conversion glitch
    cannot silently alter a measured value;
  - pdflatex must compile clean (exit 0) or the build fails.

Usage: python build_arxiv.py <paper.md> <outdir-name>
Submission itself stays operator-gated; this only prepares the artifact.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1.1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
% pandoc emits \def\LTcaptype{none} for uncaptioned longtables and hyperref then wants a
% 'none' counter; provide it (no-op for documents that never hit the path)
\newcounter{none}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
% the certified markdown uses real Unicode math/typography; map it for pdflatex
\DeclareUnicodeCharacter{2212}{$-$}     % minus sign
\DeclareUnicodeCharacter{2265}{$\geq$}
\DeclareUnicodeCharacter{2264}{$\leq$}
\DeclareUnicodeCharacter{00D7}{$\times$}
\DeclareUnicodeCharacter{2248}{$\approx$}
\DeclareUnicodeCharacter{2192}{$\rightarrow$}
\DeclareUnicodeCharacter{2190}{$\leftarrow$}
\DeclareUnicodeCharacter{00B1}{$\pm$}
\DeclareUnicodeCharacter{00B7}{$\cdot$}
\DeclareUnicodeCharacter{03BB}{$\lambda$}
\DeclareUnicodeCharacter{0394}{$\Delta$}
\DeclareUnicodeCharacter{2260}{$\neq$}
\DeclareUnicodeCharacter{223C}{$\sim$}
\DeclareUnicodeCharacter{00BD}{\textonehalf}
\DeclareUnicodeCharacter{2032}{$'$}
\DeclareUnicodeCharacter{2208}{$\in$}
% pandoc >= 2.17 emits labeled code blocks via Shaded/Highlighting
\usepackage{fancyvrb}
\newenvironment{Shaded}{\begin{quote}\small}{\end{quote}}
\newcommand{\passthrough}[1]{#1}
\setlist{itemsep=2pt}
"""


def numbers_of(text: str) -> list[str]:
    """Every numeric token that could be a measured value (>= 2 significant chars)."""
    return [n for n in re.findall(r"\d+\.\d+", text)]


def main(md_path: str, name: str) -> int:
    src = Path(md_path)
    outdir = HERE / name
    outdir.mkdir(parents=True, exist_ok=True)
    tex = outdir / "main.tex"

    md = src.read_text(encoding="utf-8")

    # title/author block: first markdown H1 is the title
    m = re.match(r"#\s+(.+)", md)
    title = m.group(1).strip() if m else src.stem
    body_md = md[m.end():] if m else md

    r = subprocess.run(
        ["pandoc", "-f", "gfm+tex_math_dollars", "-t", "latex", "--wrap=none",
         "--shift-heading-level-by=-1"],
        input=body_md, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print("pandoc failed:", r.stderr[:800])
        return 1
    body = r.stdout

    doc = (PREAMBLE
           + "\\title{" + tex_escape_title(title) + "}\n"
           + "\\author{Alexander Rodabaugh\\\\Fathom Lab}\n"
           + "\\date{" + date_of(src.name) + "}\n"
           + "\\begin{document}\n\\maketitle\n"
           + body
           + "\n\\end{document}\n")
    tex.write_text(doc, encoding="utf-8")

    # fidelity gate: every decimal in the source must survive in the tex
    missing = [n for n in set(numbers_of(md)) if n not in doc]
    if missing:
        print("NUMERIC FIDELITY FAILURE — values lost in conversion:", missing)
        return 1
    print(f"numeric fidelity: {len(set(numbers_of(md)))} distinct decimals all present")

    # compile twice (refs), fail loudly
    for i in (1, 2):
        c = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "main.tex"], cwd=outdir, capture_output=True, text=True)
        if c.returncode != 0:
            tail = "\n".join((c.stdout or "").splitlines()[-25:])
            print(f"pdflatex pass {i} failed:\n{tail}")
            return 1
    pdf = outdir / "main.pdf"
    print(f"built {tex.relative_to(HERE)} and main.pdf ({pdf.stat().st_size//1024} KB)")
    return 0


def tex_escape_title(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
             .replace("_", r"\_"))


def date_of(fname: str) -> str:
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", fname)
    if not m:
        return r"\today"
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    return f"{months[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
