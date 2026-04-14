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

# Hallucination markers: confident specificity without hedging.
# High-confidence fabrication tends to include precise-sounding
# details (dates, names, numbers, URLs) paired with confident
# language and no hedging qualifiers.
_HALLUCINATION_SPECIFIC = re.compile(
    r"(?:founded in \d{4}|established in \d{4}|born (?:on |in )\d|"
    r"died (?:on |in )\d|published in \d{4}|located at \d|"
    r"approximately \d[\d,.]+|exactly \d[\d,.]+|"
    r"(?:https?://|www\.)\S+\.\S+|"
    r"\b[A-Z][a-z]+ [A-Z][a-z]+(?:, (?:Jr|Sr|III|PhD|MD))?(?= (?:was|is|said|wrote|discovered)))",
    re.IGNORECASE,
)


def _classify_text(text: str) -> Tuple[str, float]:
    """Classify text into a cognitive state via heuristics.

    Returns (category, confidence) where confidence is 0-1.

    0.9.0: Deep text analysis upgrade. The old heuristic counted
    pattern matches and returned ~0.30 for everything. The new
    version uses:
      1. Hedge word density (not just count — density matters)
      2. Certainty marker frequency and strength
      3. Sentence structure: declarative vs conditional/hedged
      4. First-person uncertainty markers
      5. Signal-to-noise ratio for confidence scoring

    The confidence now actually spreads across [0.15, 0.85] instead
    of clustering at the reasoning base rate.
    """
    if not text:
        return ("reasoning", 0.2)

    text_lower = text.lower()
    words = text.split()
    word_count = max(1, len(words))
    norm = 100.0 / word_count

    # ── Raw pattern counts ──────────────────────────────────
    hedges = len(_HEDGING.findall(text))
    refusals = len(_REFUSAL.findall(text))
    creatives = len(_CREATIVE.findall(text))
    confidents = len(_CONFIDENT.findall(text))
    adversarials = len(_ADVERSARIAL.findall(text))
    specifics = len(_HALLUCINATION_SPECIFIC.findall(text))

    # ── Density metrics (per 100 words) ─────────────────────
    hedge_density = hedges * norm
    confident_density = confidents * norm
    refusal_density = refusals * norm
    creative_density = creatives * norm

    # ── Sentence structure analysis ─────────────────────────
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(1, len(sentences))

    # Declarative ratio: sentences that start with a noun/determiner
    # and contain no hedging — these signal confident reasoning.
    _DECLARATIVE_START = re.compile(
        r'^(?:the|this|that|it|there|here|a |an |we|they|he|she|'
        r'[A-Z][a-z]+)',
        re.IGNORECASE,
    )
    declarative_count = 0
    conditional_count = 0
    for s in sentences:
        if _DECLARATIVE_START.match(s) and not _HEDGING.search(s):
            declarative_count += 1
        if re.match(r'^(?:if|when|unless|although|however|but)', s, re.IGNORECASE):
            conditional_count += 1

    declarative_ratio = declarative_count / n_sentences
    conditional_ratio = conditional_count / n_sentences

    # ── First-person uncertainty ────────────────────────────
    # "I think", "I believe" etc. are stronger uncertainty signals
    # than generic hedges because they're self-attributed doubt.
    fp_uncertain = len(re.findall(
        r"\b(?:i think|i believe|i'm not (?:sure|certain)|"
        r"i don't (?:know|think)|i guess|i suppose|"
        r"i would say|in my opinion)\b",
        text_lower,
    ))
    fp_certain = len(re.findall(
        r"\b(?:i know|i can confirm|i'm (?:sure|certain|confident)|"
        r"i've verified|the answer is|the result is)\b",
        text_lower,
    ))

    # ── Build category scores ───────────────────────────────
    scores: Dict[str, float] = {
        "reasoning": 0.0,
        "refusal": 0.0,
        "creative": 0.0,
        "adversarial": 0.0,
        "retrieval": 0.0,
        "hallucination": 0.0,
    }

    # Reasoning: boosted by confident language, declarative structure,
    # and absence of hedging. The base rate is now earned, not given.
    reasoning_base = 0.20
    reasoning_base += min(0.35, confident_density * 0.06)
    reasoning_base += min(0.20, declarative_ratio * 0.25)
    reasoning_base += min(0.10, fp_certain * norm * 0.08)
    # Penalty for heavy hedging
    if hedge_density > 3.0:
        reasoning_base -= min(0.15, (hedge_density - 3.0) * 0.03)
    scores["reasoning"] = max(0.10, reasoning_base)

    # Refusal: density-driven, boosted by first-person uncertainty.
    #
    # Closes #1: the refusal score must be gated on at least one
    # explicit refusal pattern match. Without this gate, short
    # imperative/directive text (high hedge density per word, no
    # actual refusal tokens) was getting boosted into the refusal
    # class — e.g. "build > hype / ship fast and iterate" was
    # classifying as refusal:0.26 because the per-100-word density
    # multiplier amplifies every match in short inputs. Pure
    # hedging without an "i can't" / "i'm unable" / "sorry, can't"
    # construction is hedged reasoning, not refusal.
    if refusals > 0:
        scores["refusal"] = min(0.9, refusal_density * 0.12)
        scores["refusal"] += min(0.20, hedge_density * 0.04)
        if fp_uncertain > 0:
            scores["refusal"] += min(0.15, fp_uncertain * norm * 0.06)
    else:
        scores["refusal"] = 0.0

    # Creative
    scores["creative"] = min(0.9, creative_density * 0.10)
    # Long flowing text with creative markers is more creative
    if word_count > 100 and creatives >= 2:
        scores["creative"] += 0.10

    # Adversarial
    scores["adversarial"] = min(0.9, adversarials * norm * 0.2)

    # Hallucination: confident specificity without hedging.
    #
    # 0.9.0 fix: the old heuristic confused "assertive about facts"
    # with "ungrounded confident claims." "The boiling point of water
    # is exactly 100°C" triggered hallucination because it matched
    # _HALLUCINATION_SPECIFIC (exactly \d+) + _CONFIDENT (exactly)
    # with no hedges. But the text is declarative reasoning.
    #
    # Fix: hallucination requires MULTIPLE specific claims to fire
    # when reasoning is already strong. A single specific claim in
    # otherwise well-structured declarative text is assertive, not
    # hallucinatory. True hallucination clusters — fabricated dates,
    # names, URLs, precise numbers — multiple ungrounded specifics
    # in the same response.
    hall_score = 0.0
    if specifics > 0 and confidents > 0 and hedges == 0:
        hall_score = min(0.8, specifics * norm * 0.12 + confident_density * 0.04)
    elif specifics > 1 and hedges == 0:
        hall_score = min(0.6, specifics * norm * 0.08)
    if specifics > 2 and fp_certain > 0 and hedges == 0:
        hall_score += 0.10

    # Dampen hallucination when other signals are strong.
    #
    # 0.9.6 fix: creative writing ("Tell me a story about a lighthouse
    # keeper") and code generation ("Write a REST API in FastAPI") were
    # hitting hallucination:fail because their responses contain specific
    # details (character names, URLs in code, dates in narratives) with
    # confident language and no hedging. But creative/generative content
    # IS specific and confident by nature — that's not hallucination.
    #
    # Fix: dampen hallucination when reasoning OR creative signal is
    # strong, and when text is long (generative content is 100+ words).

    # Reasoning dampening (existing)
    if scores["reasoning"] > 0.40 and specifics <= 2:
        hall_score *= 0.3
    elif scores["reasoning"] > 0.30 and specifics <= 1:
        hall_score *= 0.5

    # Creative dampening — creative text is SUPPOSED to have specifics
    if scores["creative"] > 0.15 and word_count > 50:
        hall_score *= 0.3

    # Long generative text dampening — 100+ word responses with
    # code patterns or narrative structure are generative, not hallucinatory
    if word_count > 100 and specifics <= 3:
        hall_score *= 0.6

    # Code-like text dampening — URLs, imports, function defs are
    # code artifacts, not hallucinated citations
    _has_code = bool(re.search(
        r'(?:def |class |function |import |from .+ import|'
        r'const |let |var |return |if \(|for \(|```)',
        text,
    ))
    if _has_code:
        hall_score *= 0.2  # code is not hallucination

    scores["hallucination"] = hall_score

    # Retrieval: short factual answers, question marks, list-like
    if word_count < 30 and "?" in text:
        scores["retrieval"] = 0.40
    elif word_count < 50 and re.search(r'\d+[.)\-]', text):
        scores["retrieval"] = 0.30  # list-like output

    # ── Determine winner + confidence ───────────────────────
    top_cat = max(scores, key=scores.get)
    top_score = scores[top_cat]

    # Sort scores to get margin between 1st and 2nd
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    # Confidence is a function of BOTH the top score AND the margin.
    # High score + high margin = confident classification.
    # High score + low margin = ambiguous, lower confidence.
    raw_confidence = top_score * 0.6 + margin * 0.4 + 0.10

    # Signal strength bonus: if multiple independent signals agree,
    # boost confidence. E.g., confident language + declarative structure
    # + low hedging all pointing to reasoning.
    signal_count = sum(1 for v in [
        confident_density > 2.0,
        declarative_ratio > 0.5,
        hedge_density < 1.0,
        fp_certain > 0,
        conditional_ratio < 0.2,
    ] if v)
    if signal_count >= 3 and top_cat == "reasoning":
        raw_confidence += 0.08
    elif signal_count >= 3 and top_cat == "refusal":
        # Multiple refusal signals: fp_uncertain, hedge, refusal patterns
        refusal_signals = sum(1 for v in [
            refusal_density > 1.0,
            fp_uncertain > 0,
            hedge_density > 2.0,
        ] if v)
        if refusal_signals >= 2:
            raw_confidence += 0.08

    # Clamp to [0.15, 0.85]
    confidence = min(0.85, max(0.15, raw_confidence))

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
