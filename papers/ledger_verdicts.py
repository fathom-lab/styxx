"""How a cycle's free-prose verdict is classified — extracted so it can be TESTED.

`papers/build_ledger.py` generates `papers/LEDGER.md`, the document every other claim in this
repository is collateralised against, and `tests/test_ledger.py` checks that the committed file
matches what the generator produces. That guard is necessary and it is not sufficient: it proves
the file agrees with the generator, never that the generator is right. A wrong classifier passes
it happily, which is exactly how the refusal list came to print `SHIPPED` under a heading saying
these were runs a preregistered gate refused.

So the classification lives here, importable, with `tests/test_ledger_classifier.py` asserting the
cases by hand — including every cycle the old substring rule got wrong.

THE DEFECT THIS REPLACED. The rule was `"INVALID__" in verdict`: a substring test over a
free-prose blob, so any cycle whose commentary discussed an earlier invalid counted as a machinery
refusal. The renderer then printed the blob's FIRST WORD. Together they listed `SHIPPED`,
`PRODUCT`, `DO`, `REWRITTEN` and `BUILT` as refusals — and counted cycle 156, the cycle that BUILT
the ledger, as a loss, because its verdict text quotes the ledger's own negatives count.

That is mention-versus-use. See `papers/SYNTHESIS_mention_and_use_2026_08_26.md`: the same defect
was found the same day in the OATH obligation predicate, in OATH's treatment of quotation, and in
diffgate's claim extractor. Four instruments, one root cause — a predicate that reads a line
cannot tell you what the line claims.
"""
from __future__ import annotations

import re

__all__ = ["is_refusal", "refusal_tokens", "verdict_head", "INVALID_TOKEN"]

INVALID_TOKEN = re.compile(r"INVALID__[A-Za-z0-9_]+")

# The verdict's opening clause. Everything past the first sentence break is commentary ABOUT the
# run rather than the run's verdict, and commentary is where the mentions live.
_HEAD = re.compile(r"^[^.;\n]{0,160}")

# "TWO HONEST INVALIDS, ..." / "TWO INVALIDS, TWO MECHANISMS, ..." — a verdict that opens by
# announcing invalids IS a refusal even though its leading token is a numeral. Cycles 110 and 115
# are the live cases and they must not be dropped by a rule aimed at cycle 133.
_ANNOUNCES = re.compile(r"^\S+\s+(?:HONEST\s+)?INVALIDS\b")


def verdict_head(v: str) -> str:
    v = (v or "").strip()
    return _HEAD.match(v).group(0) if v else ""


def is_refusal(v: str) -> bool:
    """Did THIS cycle's gate return `INVALID__*`, or is its prose discussing someone else's?

    A cycle is a machinery refusal when its LEADING verdict token is an `INVALID__*`, or when the
    verdict opens by announcing invalids. A cycle that cites an earlier invalid in a parenthetical
    — cycle 157 quotes b48's — does not qualify, and neither does a cycle that shipped.
    """
    v = (v or "").strip()
    if not v:
        return False
    lead = v.split()[0].strip("(")
    return lead.startswith("INVALID__") or bool(_ANNOUNCES.match(verdict_head(v)))


def refusal_tokens(v: str) -> str:
    """The actual `INVALID__*` token(s), for display — never the verdict's first word.

    Printing the first word is what made the defect visible (`cycle 152 — DO`), and printing the
    real tokens is what makes cycle 115's two independent invalids legible instead of `TWO`.
    """
    toks = sorted(set(INVALID_TOKEN.findall(v or "")))
    return ", ".join("`%s`" % t for t in toks) if toks else "`INVALID__*`"
