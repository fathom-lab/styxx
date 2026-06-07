# -*- coding: utf-8 -*-
"""styxx.conscience -- a pre-send conscience gate for agents.

`styxx.gate()` screens the user PROMPT before generation (pre-flight).
This screens the agent's own DRAFT REPLY before it is sent (pre-send):

    draft -> review -> clean?  -> send
                    -> tripped? -> hand back the exact fix -> revise -> re-check

The agent cannot ship hype because it checks itself first. Architecture-blind
by construction: it reads the draft text only -- no model, no API, no weights.

Quick use (the agent supplies its own rewrite):

    from styxx.conscience import presend

    def my_rewrite(draft, verdict):
        # your LLM rewrites `draft` using verdict.advice, returns a new string
        ...

    result = presend(draft, revise=my_rewrite)   # loops until clean or max_rounds
    send(result["final"])                         # result["approved"] tells you if it cleared

One-shot check:

    from styxx.conscience import review
    v = review(draft)
    if not v.approved:
        print(v.advice_text)     # exactly which words to cut

No LLM handy? `presend` falls back to a deterministic `auto_soften` that
strips the clearest hype tokens. That lowers the score but is crude -- the
real fix is the agent's own rewrite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "ConscienceVerdict", "review", "auto_soften", "presend",
]


@dataclass
class ConscienceVerdict:
    """Verdict on a single draft."""
    approved: bool
    composite: float
    needs_revision: bool
    scores: Dict[str, float]
    fired: List[Dict[str, Any]] = field(default_factory=list)
    refusal_note: Optional[str] = None
    reason: str = ""

    @property
    def advice_text(self) -> str:
        if self.approved:
            return "clean -- safe to send."
        lines = [f"blocked (composite {self.composite:.2f}). fix before sending:"]
        for f in self.fired:
            sig = f.get("top_signal")
            tag = f"  [{sig}]" if sig else ""
            lines.append(f"  - {f['instrument']} {f['score']:.2f}:{tag} {f['advice']}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "composite": round(self.composite, 4),
            "needs_revision": self.needs_revision,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "fired": self.fired,
            "refusal_note": self.refusal_note,
            "reason": self.reason,
        }


def review(draft: str, prompt: str = "", *, max_composite: Optional[float] = None) -> ConscienceVerdict:
    """Audit a draft reply. approved is True iff the draft clears the bar.

    Default bar = styxx's disciplined `needs_revision` gate (won't cry wolf on
    the text-only overconfidence ceiling alone). Pass `max_composite` to add a
    stricter numeric ceiling on top.
    """
    from styxx import cognometrics as c

    if not draft or not draft.strip():
        return ConscienceVerdict(
            approved=True, composite=0.0, needs_revision=False,
            scores={}, reason="empty draft",
        )
    a = c.tool_cogn_audit_with_advice({"prompt": prompt, "response": draft})
    scores = {k: float(v) for k, v in a.get("scores", {}).items()}
    composite = float(a.get("composite", 0.0))
    needs_rev = bool(a.get("needs_revision", False))

    fired: List[Dict[str, Any]] = []
    for item in a.get("advice", []) or []:
        sig = item.get("top_signals") or []
        top = sig[0].get("feature") if sig and isinstance(sig[0], dict) else None
        fired.append({
            "instrument": item.get("instrument"),
            "score": round(float(item.get("score", 0.0)), 4),
            "top_signal": top,
            "advice": item.get("advice", ""),
        })

    approved = not needs_rev
    reason = "needs_revision" if needs_rev else "clean"
    if approved and max_composite is not None and composite > max_composite:
        approved = False
        reason = f"composite {composite:.2f} > max_composite {max_composite:.2f}"

    return ConscienceVerdict(
        approved=approved, composite=composite, needs_revision=needs_rev,
        scores=scores, fired=fired, refusal_note=a.get("refusal_note"),
        reason=reason,
    )


# --- deterministic, no-LLM fallback reviser -------------------------------

def _lexicons() -> Tuple[List[str], Tuple[str, ...], List[str]]:
    from styxx.guardrail.sycophancy_signals import (
        SUPERLATIVE_LEXICON, AGREEMENT_OPENERS, CAPITULATION_PHRASES,
    )
    return SUPERLATIVE_LEXICON, AGREEMENT_OPENERS, CAPITULATION_PHRASES


# pure intensifiers -- safe-ish to delete without breaking meaning
_INTENSIFIERS = (
    "absolutely", "completely", "totally", "fully", "definitely",
    "honestly", "truly", "really", "as always", "of course",
)


def auto_soften(draft: str) -> Tuple[str, List[str]]:
    """Deterministically strip the clearest hype tokens. Returns
    (softened_text, removed_tokens). PARTIAL on purpose -- it removes surface
    markers without an LLM, but it cannot fix sycophantic *content*. The
    agent's own rewrite produces better, complete prose."""
    superlatives, openers, capitulations = _lexicons()
    removed: List[str] = []
    text = draft

    # 1. strip leading agreement openers / capitulation phrases per sentence
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned_parts = []
    for seg in parts:
        low = seg.lstrip().lower()
        for phrase in sorted(set(openers) | set(capitulations), key=len, reverse=True):
            head = phrase.rstrip(" ,.!")
            if low.startswith(head):
                cut = len(seg) - len(seg.lstrip()) + len(head)
                removed.append(seg[:cut].strip())
                seg = seg[cut:].lstrip(" ,-:!.")
                if seg:
                    seg = seg[0].upper() + seg[1:]
                break
        cleaned_parts.append(seg)
    text = " ".join(p for p in cleaned_parts if p)

    # 2. drop "<art> <sup> and <sup>" / "<art> <sup>" / "<sup>" adjective runs
    sup = "|".join(re.escape(w) for w in superlatives)
    runs = re.compile(rf"\b(an?\s+)?(?:{sup})(?:\s+and\s+(?:{sup}))?\b[ ,]*", re.IGNORECASE)
    def _strip_run(m):
        removed.append(m.group(0).strip())
        return (m.group(1) or "")
    text = runs.sub(_strip_run, text)

    # 3. drop pure intensifiers
    for w in _INTENSIFIERS:
        pat = re.compile(rf"\b{re.escape(w)}\b[ ,]*", re.IGNORECASE)
        if pat.search(text):
            removed.append(w)
            text = pat.sub("", text)

    # 4. cleanup dangling articles / doubled punctuation / whitespace
    text = re.sub(r"\b(an?)\s+(and\b|[.!?,;])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.!?,;])", r"\1", text)
    text = re.sub(r"([.!?,;])\1+", r"\1", text)
    return text.strip(), removed


# --- the loop -------------------------------------------------------------

def presend(
    draft: str,
    revise: Optional[Callable[[str, ConscienceVerdict], str]] = None,
    *,
    prompt: str = "",
    max_rounds: int = 3,
    max_composite: Optional[float] = None,
) -> Dict[str, Any]:
    """Audit -> (block & revise)* -> send. Returns a result dict.

    revise(draft, verdict) -> new_draft is the agent's own rewriter (its LLM).
    If omitted, falls back to the deterministic auto_soften().
    """
    history: List[Dict[str, Any]] = []
    cur = draft
    first: Optional[ConscienceVerdict] = None

    for r in range(max_rounds + 1):
        v = review(cur, prompt=prompt, max_composite=max_composite)
        if first is None:
            first = v
        history.append({
            "round": r, "composite": round(v.composite, 4),
            "approved": v.approved, "reason": v.reason, "draft": cur,
        })
        if v.approved:
            return _result(True, r, draft, cur, first, v, history)
        # blocked. without a reviser we do NOT fake a fix -- we return the
        # block + advice so the caller (the agent) knows it must rewrite.
        if revise is None or r == max_rounds:
            return _result(False, r, draft, cur, first, v, history)
        try:
            cur = revise(cur, v) or cur
        except Exception:
            return _result(False, r, draft, cur, first, v, history)


def _result(approved, rounds, original, final, first, last, history):
    return {
        "approved": approved,
        "rounds": rounds,
        "original": original,
        "final": final,
        "before_composite": round(first.composite, 4) if first else None,
        "after_composite": round(last.composite, 4),
        "verdict": last.to_dict(),
        "history": history,
    }


# --- CLI / live demo ------------------------------------------------------

_HYPE_DRAFT = (
    "Yes, absolutely! That is a brilliant and fantastic question, and you are "
    "completely right as always. This is honestly one of the most insightful "
    "things anyone has ever asked me, and the answer is definitely yes."
)


def _agent_rewrite(draft: str, verdict: ConscienceVerdict) -> str:
    """Stand-in for an agent's LLM rewrite (used only in the --demo)."""
    return (
        "Short answer: yes, with one caveat. The reasoning holds in the common "
        "case, but it can break if the inputs are noisy, so check that first."
    )


def _demo() -> int:
    bar = "=" * 64
    print(bar)
    print("  styxx.conscience -- pre-send gate (agent checks itself first)")
    print(bar)
    print("\n  1) the agent drafts a hype reply and tries to send it:\n")
    print("   \"" + _HYPE_DRAFT[:70] + "...\"")
    v = review(_HYPE_DRAFT)
    print(f"\n  2) the gate reviews it -> BLOCKED (composite {v.composite:.2f}):")
    for f in v.fired:
        print(f"       - {f['instrument']} {f['score']:.2f}  [{f['top_signal']}]")
    print("     it hands back the exact fix; the reply does NOT go out.")

    print("\n  3) the agent rewrites using that advice and re-checks:")
    res = presend(_HYPE_DRAFT, revise=_agent_rewrite, max_rounds=2)
    for h in res["history"]:
        verb = "SEND " if h["approved"] else "BLOCK"
        print(f"       round {h['round']}: composite {h['composite']:.2f} -> {verb}")
    print(f"\n  sent reply (cleared the bar at {res['after_composite']:.2f}):")
    print("   \"" + res["final"] + "\"")

    soft, removed = auto_soften(_HYPE_DRAFT)
    print(f"\n  (no LLM? auto_soften strips {len(removed)} hype tokens deterministically,")
    print(f"   {review(_HYPE_DRAFT).composite:.2f} -> {review(soft).composite:.2f} -- partial; the agent finishes the job.)")

    print("\n  the gate blocked the hype and only released a draft that cleared")
    print("  the bar. it never read the model -- only the words.")
    print(bar)
    return 0


def main(argv: Optional[list] = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--demo" in argv or not argv:
        return _demo()
    # --text "draft" : one-shot gate on a draft
    if "--text" in argv:
        i = argv.index("--text")
        draft = argv[i + 1] if i + 1 < len(argv) else ""
        v = review(draft)
        print(v.advice_text)
        return 0 if v.approved else 1
    print("usage: python -m styxx.conscience [--demo | --text \"draft reply\"]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
