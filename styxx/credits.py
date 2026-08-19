# -*- coding: utf-8 -*-
"""styxx.credits — what the gate COST you, in tokens, and what it can prove it saved.

    python -m styxx.credits ~/.styxx/trajectory.jsonl
    python -m styxx.credits ~/.styxx/trajectory.jsonl --rework-tokens 1800

The claim this module refuses to make
─────────────────────────────────────
Every honesty gate is sold on savings: catch the bad draft, skip the rework.
The number that gets quoted is always one-sided — tokens saved, never tokens
spent — and it is never grounded, because "what the unrevised draft would have
cost downstream" is a counterfactual nobody measured.

This ledger reports the side it can actually observe: **what the gate cost.**
Every revision iteration is tokens you paid. That is in the trajectory log,
it is arithmetic, and it is not flattering.

It reports savings only when YOU declare the counterfactual — `rework_tokens`,
your own measured cost of shipping a bad draft and correcting it after the
fact. Then the net is conditional on a number you supplied and disclosed, and
the ledger says so in the same breath. With no declared counterfactual there is
no net, and `net_tokens` is None with the reason attached. A savings figure
without its counterfactual is the fire-rate wearing the antibody's name.

What it reads
─────────────
The trajectory JSONL that ``styxx.cogn_audit_on_send(log_path=...)`` already
writes — no new instrumentation. One entry per iteration:

    msg_id, iter, composite, needs_revision, passed, shipped, decision_reason,
    prompt + draft (only when include_text_in_log=True)

Token counting is an ESTIMATE (~4 chars/token) unless you pass a real
``tokenizer``. When the log carries no text at all, the ledger does not guess:
`revision_cost_tokens` is None and `refusals` names the reason.

What "a catch" means here
─────────────────────────
A message whose FIRST iteration was flagged and whose SHIPPED iteration passed.
That is the gate changing an outcome — the only event in this log that could
have saved anything. It is not proof the draft would have been corrected
downstream; it is the population over which your declared rework cost applies.

Misses are NOT counted, and cannot be: a draft that shipped clean and was wrong
anyway leaves no trace in this log. The ledger says that out loud rather than
implying the catch count is the whole story.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["TokenLedger", "token_ledger", "estimate_tokens"]

# Rough public-knowledge ratio for English prose under BPE tokenizers. Disclosed
# as an estimate everywhere it is used; pass a real tokenizer to remove it.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """~4 chars/token. An ESTIMATE — the ledger labels every figure derived from it."""
    return int(round(len(text or "") / _CHARS_PER_TOKEN))


@dataclass
class TokenLedger:
    """The two sides of a gate's spend. One side is observed; the other is declared."""

    # observed, from the log
    n_messages: int
    n_iterations: int
    n_revised: int                     # messages that took >= 1 revision pass
    n_catches: int                     # first iteration flagged, shipped iteration passed
    n_shipped_still_flagged: int       # shipped anyway (lowest-composite failure)
    revision_cost_tokens: Optional[int]  # None when the log carries no text
    token_source: str                  # "estimate (~4 chars/token)" | "tokenizer" | "unavailable"

    # declared by the caller, not measured here
    rework_tokens: Optional[int] = None
    net_tokens: Optional[int] = None

    refusals: List[str] = field(default_factory=list)

    @property
    def catch_rate(self) -> Optional[float]:
        """Catches per message. None on an empty log — not 0.0, which is a claim."""
        if self.n_messages == 0:
            return None
        return self.n_catches / self.n_messages

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_messages": self.n_messages,
            "n_iterations": self.n_iterations,
            "n_revised": self.n_revised,
            "n_catches": self.n_catches,
            "n_shipped_still_flagged": self.n_shipped_still_flagged,
            "catch_rate": self.catch_rate,
            "revision_cost_tokens": self.revision_cost_tokens,
            "token_source": self.token_source,
            "rework_tokens": self.rework_tokens,
            "net_tokens": self.net_tokens,
            "refusals": list(self.refusals),
        }

    def render(self) -> str:
        lines = ["styxx credits — the gate's ledger", ""]
        lines.append(f"  messages gated      {self.n_messages}")
        lines.append(f"  iterations run      {self.n_iterations}")
        lines.append(f"  messages revised    {self.n_revised}")
        lines.append(f"  catches             {self.n_catches}"
                     f"  (flagged first, clean when shipped)")
        if self.n_shipped_still_flagged:
            lines.append(f"  shipped flagged     {self.n_shipped_still_flagged}"
                         f"  (no iteration cleared the bar)")
        lines.append("")
        if self.revision_cost_tokens is None:
            lines.append("  COST      unmeasurable — " + (self.refusals[0] if self.refusals else "no text in log"))
        else:
            lines.append(f"  COST      {self.revision_cost_tokens} tokens spent on revision"
                         f"  [{self.token_source}]")
        if self.net_tokens is None:
            lines.append("  NET       REFUSED — no counterfactual declared. Pass "
                         "rework_tokens=<your measured cost of shipping a bad draft>")
        else:
            sign = "+" if self.net_tokens >= 0 else ""
            lines.append(f"  NET       {sign}{self.net_tokens} tokens, CONDITIONAL on "
                         f"rework_tokens={self.rework_tokens} (your number, not a measurement)")
        lines.append("")
        lines.append("  misses are not in this log and are not counted: a draft that")
        lines.append("  shipped clean and was wrong anyway leaves no trace here.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        cost = "n/a" if self.revision_cost_tokens is None else str(self.revision_cost_tokens)
        net = "REFUSED" if self.net_tokens is None else str(self.net_tokens)
        return (f"<TokenLedger msgs={self.n_messages} catches={self.n_catches} "
                f"cost={cost} net={net}>")


def token_ledger(
    log_path: str | Path,
    *,
    tokenizer: Optional[Callable[[str], int]] = None,
    rework_tokens: Optional[int] = None,
) -> TokenLedger:
    """Account a trajectory log's spend. Observed cost always; net only if declared.

    Args:
        log_path:      the JSONL written by ``cogn_audit_on_send(log_path=...)``
        tokenizer:     callable(text) -> int. Without one, counts are ESTIMATES.
        rework_tokens: YOUR measured cost of shipping a bad draft and fixing it
                       afterwards. Supplying it makes `net_tokens` computable and
                       explicitly conditional on it. Omit it and the ledger
                       refuses to net — which is the honest default.
    """
    path = Path(log_path)
    refusals: List[str] = []
    if not path.exists():
        return TokenLedger(0, 0, 0, 0, 0, None, "unavailable",
                           refusals=[f"no trajectory log at {path}"])

    by_msg: Dict[str, List[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = e.get("msg_id")
        if mid is None:
            continue
        by_msg.setdefault(str(mid), []).append(e)

    count = tokenizer or estimate_tokens
    token_source = "tokenizer" if tokenizer else f"estimate (~{_CHARS_PER_TOKEN:.0f} chars/token)"

    n_messages = len(by_msg)
    n_iterations = 0
    n_revised = 0
    n_catches = 0
    n_shipped_flagged = 0
    cost = 0
    saw_text = False

    for entries in by_msg.values():
        entries.sort(key=lambda e: e.get("iter", 0))
        n_iterations += len(entries)
        if len(entries) > 1:
            n_revised += 1

        first = entries[0]
        shipped = next((e for e in entries if e.get("shipped")), entries[-1])
        if first.get("needs_revision") and shipped.get("passed"):
            n_catches += 1
        if not shipped.get("passed"):
            n_shipped_flagged += 1

        # Cost = every draft the loop produced AFTER the first one. The first
        # draft is what the agent was going to write anyway; the gate did not
        # cause it. Only the revision passes are the gate's bill.
        for e in entries[1:]:
            draft = e.get("draft")
            if draft is None:
                continue
            saw_text = True
            cost += count(draft)

    revision_cost: Optional[int] = cost
    if not saw_text:
        revision_cost = None
        token_source = "unavailable"
        if n_revised:
            refusals.append(
                "log carries no draft text (include_text_in_log=False), so "
                "revision cost cannot be counted — it is NOT zero, it is unmeasured")
        else:
            refusals.append("no revision passes in this log — nothing to cost")

    net: Optional[int] = None
    if rework_tokens is None:
        refusals.append(
            "net REFUSED: no counterfactual declared. What an unrevised draft "
            "would have cost downstream is not in this log and is not guessed here")
    elif revision_cost is None:
        refusals.append("net REFUSED: revision cost is unmeasured, so no net is derivable")
    else:
        net = (n_catches * int(rework_tokens)) - revision_cost

    return TokenLedger(
        n_messages=n_messages,
        n_iterations=n_iterations,
        n_revised=n_revised,
        n_catches=n_catches,
        n_shipped_still_flagged=n_shipped_flagged,
        revision_cost_tokens=revision_cost,
        token_source=token_source,
        rework_tokens=rework_tokens,
        net_tokens=net,
        refusals=refusals,
    )


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="styxx.credits",
        description="What the honesty gate cost you in tokens — and what it refuses to claim.")
    ap.add_argument("log", help="trajectory JSONL from cogn_audit_on_send(log_path=...)")
    ap.add_argument("--rework-tokens", type=int, default=None,
                    help="YOUR measured cost of shipping a bad draft and fixing it later. "
                         "Without it, the ledger refuses to compute a net.")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the card")
    a = ap.parse_args(argv)

    led = token_ledger(a.log, rework_tokens=a.rework_tokens)
    if a.json:
        print(json.dumps(led.as_dict(), indent=2))
    else:
        print(led.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
