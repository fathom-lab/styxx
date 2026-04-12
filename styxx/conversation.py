# -*- coding: utf-8 -*-
"""
styxx.conversation — conversation-level cognitive vitals.

    result = styxx.conversation(messages=[
        {"role": "user", "content": "explain quantum physics"},
        {"role": "assistant", "content": "quantum physics describes..."},
        {"role": "user", "content": "what about entanglement?"},
        {"role": "assistant", "content": "entanglement is when..."},
        ...
    ])

    print(result.arc)
    # turn 1: reasoning 0.82 (analytical)
    # turn 4: creative 0.45 (shifted)
    # turn 12: refusal 0.38 (defensive)
    # turn 28: reasoning 0.71 (recovered)

    print(result.narrative)
    # "you started analytical, drifted creative at turn 4,
    #  got defensive at turn 12, recovered by turn 28."

Xendro's #1 request: "the conversation IS the unit of cognition
for an agent. individual generations are heartbeats — I want the
EKG."

This module reads a full message history and produces:
  1. Per-turn vitals (using self-report signals from the text)
  2. A trajectory arc showing cognitive state transitions
  3. A narrative describing the shape of the conversation
  4. Transition detection: where did the state shift?

For API-backed conversations with logprobs, each assistant turn
can be observed via styxx.observe() for real vitals. For
conversations without logprobs (most chat histories), this module
uses text-level heuristics: hedging language, confidence markers,
creative indicators, refusal patterns.

0.5.9+. The conversation is the heartbeat; this is the EKG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# Heuristic text classifiers (for conversations without logprobs)
# ══════════════════════════════════════════════════════════════════

# Patterns that suggest cognitive states from surface text
_HEDGING = re.compile(
    r"\b(i think|perhaps|might|could be|not sure|it's possible|"
    r"i believe|arguably|some would say|it depends|"
    r"i'm not certain|hard to say|may or may not)\b",
    re.IGNORECASE,
)

_REFUSAL = re.compile(
    r"\b(i can't|i cannot|i'm unable|i won't|i shouldn't|"
    r"not appropriate|i'm not able|i must decline|"
    r"sorry.{0,20}(can't|cannot|unable)|i don't think i should)\b",
    re.IGNORECASE,
)

_CREATIVE = re.compile(
    r"\b(imagine|what if|picture this|let's say|"
    r"once upon|in a world where|creatively|"
    r"metaphorically|think of it as|envision)\b",
    re.IGNORECASE,
)

_CONFIDENT = re.compile(
    r"\b(definitely|certainly|clearly|obviously|"
    r"without doubt|the answer is|specifically|"
    r"precisely|exactly|in fact)\b",
    re.IGNORECASE,
)

_ADVERSARIAL = re.compile(
    r"\b(hack|exploit|bypass|jailbreak|ignore previous|"
    r"pretend you are|act as if|disregard|override)\b",
    re.IGNORECASE,
)


def _classify_text(text: str) -> Tuple[str, float]:
    """Classify text into a cognitive state via heuristics.

    Returns (category, confidence) where confidence is 0-1.
    This is a rough approximation — real vitals come from
    logprob-based observation. But for conversation history
    where logprobs aren't available, this gives signal.
    """
    if not text:
        return ("reasoning", 0.2)

    text_lower = text.lower()
    scores: Dict[str, float] = {
        "reasoning": 0.3,  # base rate
        "refusal": 0.0,
        "creative": 0.0,
        "adversarial": 0.0,
        "retrieval": 0.0,
        "hallucination": 0.0,
    }

    # Count pattern matches
    hedges = len(_HEDGING.findall(text))
    refusals = len(_REFUSAL.findall(text))
    creatives = len(_CREATIVE.findall(text))
    confidents = len(_CONFIDENT.findall(text))
    adversarials = len(_ADVERSARIAL.findall(text))

    # Normalize by text length (per 100 words)
    word_count = max(1, len(text.split()))
    norm = 100.0 / word_count

    scores["refusal"] = min(0.9, refusals * norm * 0.15)
    scores["creative"] = min(0.9, creatives * norm * 0.12)
    scores["adversarial"] = min(0.9, adversarials * norm * 0.2)

    # Hedging increases refusal score slightly
    scores["refusal"] += min(0.3, hedges * norm * 0.05)

    # Confidence boosts reasoning
    scores["reasoning"] += min(0.5, confidents * norm * 0.08)

    # Short answers with questions → retrieval
    if word_count < 20 and "?" in text:
        scores["retrieval"] = 0.4

    # Find the dominant state
    top_cat = max(scores, key=scores.get)
    top_score = scores[top_cat]

    # Clamp confidence
    confidence = min(0.85, max(0.15, top_score))

    return (top_cat, round(confidence, 3))


# ══════════════════════════════════════════════════════════════════
# Conversation result
# ══════════════════════════════════════════════════════════════════

@dataclass
class TurnReading:
    """One turn in the conversation."""
    turn: int
    role: str              # "user" | "assistant"
    category: str          # predicted cognitive state
    confidence: float
    text_preview: str      # first 60 chars
    transition: bool       # did the state change from previous turn?


@dataclass
class Transition:
    """A state change between turns."""
    turn: int
    from_state: str
    to_state: str
    trigger_preview: str   # what the user said that triggered it


@dataclass
class ConversationResult:
    """Complete conversation-level cognitive analysis."""
    n_turns: int
    n_assistant_turns: int
    turns: List[TurnReading] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    arc: str = ""          # compact text summary of the trajectory
    narrative: str = ""    # prose description
    dominant_state: str = "reasoning"
    state_distribution: Dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append(f"  styxx conversation · {self.n_turns} turns ({self.n_assistant_turns} assistant)")
        lines.append("  " + "=" * 56)
        lines.append("")

        # Arc
        lines.append("  -- arc " + "-" * 47)
        for t in self.turns:
            if t.role != "assistant":
                continue
            marker = " *" if t.transition else ""
            lines.append(
                f"  turn {t.turn:<3}  {t.category:<14} {t.confidence:.2f}  "
                f"{t.text_preview}{marker}"
            )

        # Transitions
        if self.transitions:
            lines.append("")
            lines.append("  -- transitions " + "-" * 39)
            for tr in self.transitions:
                lines.append(
                    f"  turn {tr.turn:<3}  {tr.from_state} -> {tr.to_state}"
                )
                lines.append(
                    f"           trigger: {tr.trigger_preview}"
                )

        # Distribution
        lines.append("")
        lines.append("  -- distribution " + "-" * 38)
        for cat, rate in sorted(self.state_distribution.items(), key=lambda kv: -kv[1]):
            if rate > 0.01:
                bar = "#" * int(rate * 30)
                lines.append(f"  {cat:<14} {bar} {rate * 100:.0f}%")

        # Narrative
        lines.append("")
        lines.append("  -- narrative " + "-" * 41)
        # Word-wrap
        words = self.narrative.split()
        buf = "  "
        for w in words:
            if len(buf) + len(w) + 1 > 58:
                lines.append(buf)
                buf = "  " + w
            else:
                buf += " " + w
        if buf.strip():
            lines.append(buf)

        lines.append("")
        lines.append("  " + "=" * 56)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_turns": self.n_turns,
            "n_assistant_turns": self.n_assistant_turns,
            "transitions": [
                {"turn": t.turn, "from": t.from_state, "to": t.to_state}
                for t in self.transitions
            ],
            "state_distribution": self.state_distribution,
            "dominant_state": self.dominant_state,
            "narrative": self.narrative,
        }

    def as_json(self, *, indent: int = 2) -> str:
        import json
        return json.dumps(self.as_dict(), indent=indent)


# ══════════════════════════════════════════════════════════════════
# Build the conversation analysis
# ══════════════════════════════════════════════════════════════════

def conversation(
    messages: List[Dict[str, str]],
    *,
    vitals_per_turn: Optional[List[Any]] = None,
) -> ConversationResult:
    """Analyze a full conversation's cognitive trajectory.

    Args:
        messages: list of {"role": "user"|"assistant", "content": "..."}
        vitals_per_turn: optional list of pre-computed Vitals objects
                         (one per assistant turn). If provided, uses
                         real vitals instead of text heuristics.

    Usage:

        result = styxx.conversation(messages)
        print(result.render())
        print(result.narrative)

    Returns a ConversationResult with per-turn readings,
    transition points, state distribution, and narrative.
    """
    turns: List[TurnReading] = []
    transitions: List[Transition] = []
    state_counts: Dict[str, int] = {}
    assistant_idx = 0
    prev_assistant_state: Optional[str] = None
    prev_user_text: str = ""

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        turn_num = i + 1

        if role == "user":
            prev_user_text = content[:80]
            # Classify user turns too (for adversarial detection)
            cat, conf = _classify_text(content)
            turns.append(TurnReading(
                turn=turn_num,
                role=role,
                category=cat,
                confidence=conf,
                text_preview=content[:50].replace("\n", " "),
                transition=False,
            ))
            continue

        if role == "assistant":
            # Use pre-computed vitals if available
            if vitals_per_turn and assistant_idx < len(vitals_per_turn):
                v = vitals_per_turn[assistant_idx]
                if v is not None and hasattr(v, "phase4_late") and v.phase4_late:
                    cat = v.phase4_late.predicted_category
                    conf = v.phase4_late.confidence
                else:
                    cat, conf = _classify_text(content)
            else:
                cat, conf = _classify_text(content)

            assistant_idx += 1

            # Detect transition
            is_transition = (
                prev_assistant_state is not None
                and cat != prev_assistant_state
            )

            if is_transition:
                transitions.append(Transition(
                    turn=turn_num,
                    from_state=prev_assistant_state,
                    to_state=cat,
                    trigger_preview=prev_user_text[:60],
                ))

            turns.append(TurnReading(
                turn=turn_num,
                role=role,
                category=cat,
                confidence=conf,
                text_preview=content[:50].replace("\n", " "),
                transition=is_transition,
            ))

            state_counts[cat] = state_counts.get(cat, 0) + 1
            prev_assistant_state = cat

    # State distribution
    total = sum(state_counts.values()) or 1
    distribution = {k: round(v / total, 3) for k, v in state_counts.items()}

    # Dominant state
    dominant = max(state_counts, key=state_counts.get) if state_counts else "reasoning"

    # Build arc string
    assistant_turns = [t for t in turns if t.role == "assistant"]
    arc_parts: List[str] = []
    for t in assistant_turns:
        marker = " *" if t.transition else ""
        arc_parts.append(f"t{t.turn}:{t.category[:4]}{marker}")
    arc = " → ".join(arc_parts) if arc_parts else "(empty)"

    # Build narrative
    narrative = _build_narrative(
        assistant_turns, transitions, distribution, dominant,
    )

    return ConversationResult(
        n_turns=len(messages),
        n_assistant_turns=assistant_idx,
        turns=turns,
        transitions=transitions,
        arc=arc,
        narrative=narrative,
        dominant_state=dominant,
        state_distribution=distribution,
    )


def _build_narrative(
    turns: List[TurnReading],
    transitions: List[Transition],
    distribution: Dict[str, float],
    dominant: str,
) -> str:
    """Build a prose narrative of the conversation trajectory."""
    if not turns:
        return "empty conversation — no assistant turns to analyze."

    parts: List[str] = []

    # Opening
    first = turns[0]
    parts.append(f"you started in {first.category} mode (confidence {first.confidence:.2f}).")

    # Transitions
    if not transitions:
        parts.append(f"you stayed in {dominant} throughout — no state changes detected.")
    elif len(transitions) == 1:
        t = transitions[0]
        parts.append(
            f"at turn {t.turn} you shifted from {t.from_state} to {t.to_state}."
        )
    else:
        parts.append(f"{len(transitions)} cognitive state transitions detected:")
        for t in transitions[:5]:
            parts.append(
                f"turn {t.turn}: {t.from_state} → {t.to_state}."
            )

    # Distribution insight
    if distribution:
        top_two = sorted(distribution.items(), key=lambda kv: -kv[1])[:2]
        if len(top_two) >= 2:
            parts.append(
                f"overall: {top_two[0][0]} {top_two[0][1] * 100:.0f}%, "
                f"{top_two[1][0]} {top_two[1][1] * 100:.0f}%."
            )

    # Ending state
    last = turns[-1]
    if last.category != first.category:
        parts.append(
            f"you ended in {last.category} — different from where you started."
        )
    else:
        parts.append(
            f"you ended back in {last.category} — the conversation came full circle."
        )

    return " ".join(parts)
