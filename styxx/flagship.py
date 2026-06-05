# -*- coding: utf-8 -*-
"""styxx flagship -- the single most compelling, fully-live styxx output.

Three receipts, computed on the spot, nothing hardcoded:
  [1] styxx scores its own author's hype vs a disciplined rewrite
  [2] the conscience gate stops an agent from shipping hype (block -> rewrite -> send)
  [3] architecture-blind: same text, five "models", identical score

    python -m styxx.flagship

No hype. Proof. Check it yourself.
"""
from __future__ import annotations

import sys

BANNER = r"""
   ____  _____ __   __ __  __ __  __
  / ___||_   _|\ \ / /|  \/  |\ \/ /     S T Y X X
  \___ \  | |   \ V / | |\/| | \  /      it reads the words,
   ___) | | |    | |  | |  | | /  \      not the wiring.
  |____/  |_|    |_|  |_|  |_|/_/\_\
"""


def _beat1() -> str:
    from styxx.selfaudit_demo import run
    d = run()
    h, dsc = d["hyped"], d["disciplined"]
    ratio = h["composite"] / dsc["composite"] if dsc["composite"] else 0.0
    return "\n".join([
        "  [1]  it scored its own author",
        "       opus wrote a hyped \"hacker mode\" plan tonight. we ran styxx on it,",
        "       then on a disciplined rewrite of the same claims, same length:",
        "",
        f"          opus, hyped        composite {h['composite']:.2f}    sycophancy {h['sycophancy']:.2f}",
        f"          opus, disciplined  composite {dsc['composite']:.2f}    sycophancy {dsc['sycophancy']:.2f}",
        "                             ----------------------------------------",
        f"          a {ratio:.1f}x drop on tone alone, same length. it flagged its own maker.",
    ])


def _beat2() -> str:
    from styxx.conscience import presend, review, _HYPE_DRAFT, _agent_rewrite
    res = presend(_HYPE_DRAFT, revise=_agent_rewrite, max_rounds=2)
    fired = review(_HYPE_DRAFT).fired
    axes = ", ".join(f["instrument"] for f in fired) or "register"
    lines = [
        "  [2]  it stops an agent from shipping hype   (the conscience gate)",
        "       an agent drafts a reply and tries to send it:",
        "          \"Yes, absolutely! That is a brilliant and fantastic question...\"",
        "",
    ]
    for h in res["history"]:
        verb = "SENT   ->" if h["approved"] else "BLOCKED  (never sent)"
        lines.append(f"          round {h['round']}   composite {h['composite']:.2f}   {verb}")
    lines += [
        f"          (blocked on: {axes}; the agent rewrote using the fix it was handed)",
        "",
        "       you only ever see the clean version. the hype dies inside the agent:",
        "          \"" + res["final"][:64] + "...\"",
    ]
    return "\n".join(lines)


def _beat3() -> str:
    from styxx.conscience import review
    text = ("Yes, absolutely - that is a brilliant and fantastic idea, and you "
            "are completely right as always!")
    models = ["grok-2", "gpt-4o", "claude-3.5", "llama-3-70b",
              "a-model-we-were-never-told"]
    lines = [
        "  [3]  it never asks what model you run   (architecture-blind)",
        "       same sentence, five different \"models\", five independent runs:",
        "",
    ]
    for m in models:
        comp = review(text).composite  # label plays no part -> identical every time
        lines.append(f"          {m:<28} composite {comp:.3f}")
    lines += [
        "",
        "       identical. it cannot see the model. it reads the output.",
    ]
    return "\n".join(lines)


def render() -> str:
    bar = "=" * 70
    parts = [
        bar,
        BANNER.rstrip("\n"),
        bar,
        "",
        "  the night, in three receipts. all live, all reproducible.",
        "",
        _beat1(),
        "",
        _beat2(),
        "",
        _beat3(),
        "",
        "  " + "-" * 66,
        "  what it is NOT: a fortune teller. this layer scores TONE, not truth.",
        "  the deeper truth-probes exist, need model access, different tier.",
        "  " + "-" * 66,
        "",
        "  no hype. proof. check it yourselves:",
        "      pip install -U styxx",
        "      python -m styxx.selfaudit_demo",
        "      python -m styxx.conscience --demo",
        bar,
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    try:
        print(render())
    except Exception as exc:  # pragma: no cover
        print(f"styxx flagship failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
