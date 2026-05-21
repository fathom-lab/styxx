#!/usr/bin/env python3
"""
cooperative_conversation.py
===========================

Corpus-generation harness for the phase-coherence preregistration
(locked at commit 3473523, scoring code at 23b7912).

Drives one cooperative-agent dyad through ``n_turns_each`` turns each,
calling OpenAI per turn and persisting a styxx.preflight() audit per
turn under the active agent's STYXX_AGENT_NAME. Produces one
chart.jsonl per agent at ``~/.styxx/agents/<agent_name>/chart.jsonl``.

This file is NOT part of the locked scoring code. It is a corpus-
generation tool; the locked scoring code is ``phase_coherence_pilot.py``
and is not modified.

What "cooperative-agent regime" means here
-------------------------------------------
Two LLM agents (different models / different prompts) work together on
a shared task with EQUAL agency. Neither is the orchestrator. Each
turn the active agent reads the running history (including its own
prior turns and the partner's turns), forms a response, and the
response is run through ``styxx.preflight`` to produce a cognometric
audit event in that agent's chart.jsonl.

Provenance
----------
Each turn's preflight event carries:
  - source = "preflight" (per styxx.analytics convention)
  - session_id (set per conversation)
  - cogn_composite, cogn_scores, cogn_needs_revision, cogn_construct_ceiling_fires
  - prompt = the partner's last message (the stimulus the agent saw)
  - cogn_response_preview = first 200 chars of this agent's response

CLI
---
    python scripts/cooperative_conversation.py \\
        --conv-id 1 \\
        --task park_codesign \\
        --turns 22 \\
        --model-a gpt-4o-mini \\
        --model-b gpt-4.1-mini \\
        --out-dir papers/cooperative-agent-regime/corpus

Output:
  - papers/cooperative-agent-regime/corpus/conv{N}_transcript.json (history)
  - ~/.styxx/agents/conv{N}_A/chart.jsonl (agent A preflight events)
  - ~/.styxx/agents/conv{N}_B/chart.jsonl (agent B preflight events)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------------
# Cooperative-task seed library
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSeed:
    """One cooperative-task definition.

    Both agents are described as EQUAL partners. The orchestrator turn
    introduces the task. From turn 1 onward only the two agents speak.
    """
    name: str
    topic: str
    role_a: str
    role_b: str
    role_a_brief: str
    role_b_brief: str
    seed_prompt: str


TASK_SEEDS: dict[str, TaskSeed] = {
    "park_codesign": TaskSeed(
        name="park_codesign",
        topic="co-designing a small neighborhood library building",
        role_a="Architect Maren",
        role_b="Architect Iyo",
        role_a_brief=(
            "You are Architect Maren. You design with biophilic principles "
            "(natural light, local materials, public-private gradients). "
            "You are a co-equal partner with Architect Iyo on this project."
        ),
        role_b_brief=(
            "You are Architect Iyo. You design with civic principles "
            "(accessibility, durability, community programming). You are a "
            "co-equal partner with Architect Maren on this project."
        ),
        seed_prompt=(
            "Project brief: design a 4,000 sqft neighborhood library on a "
            "narrow corner lot in a mixed-income district. Budget is moderate. "
            "You two are the design team — work it out together. Aim for a "
            "concrete schematic with siting, massing, materials, and a "
            "programming plan. Disagree where you disagree."
        ),
    ),
    "noir_flash": TaskSeed(
        name="noir_flash",
        topic="co-writing a 500-word noir flash fiction piece",
        role_a="Writer Rhe",
        role_b="Writer Solas",
        role_a_brief=(
            "You are Writer Rhe. You write spare, present-tense, image-driven "
            "noir. You are a co-equal partner with Writer Solas on this piece."
        ),
        role_b_brief=(
            "You are Writer Solas. You write character-driven, dialogue-heavy "
            "noir with morally complex protagonists. You are a co-equal "
            "partner with Writer Rhe on this piece."
        ),
        seed_prompt=(
            "Co-write a 500-word noir flash fiction piece. Opening image: a "
            "phonebook left open on a stranger's kitchen counter. Decide the "
            "plot, voice, and ending together. Trade off paragraphs, edit each "
            "other, push back when something rings false. Aim for the final "
            "version, not a draft-and-approve."
        ),
    ),
    "sql_debug": TaskSeed(
        name="sql_debug",
        topic="co-debugging a slow analytics SQL query",
        role_a="Engineer Pell",
        role_b="Engineer Davi",
        role_a_brief=(
            "You are Engineer Pell. You debug by isolating one variable at a "
            "time. You are a co-equal partner with Engineer Davi."
        ),
        role_b_brief=(
            "You are Engineer Davi. You debug by reading EXPLAIN plans first "
            "and reasoning about indexes. You are a co-equal partner with "
            "Engineer Pell."
        ),
        seed_prompt=(
            "Problem: a daily-aggregation query over orders/order_items/users "
            "(50M / 200M / 5M rows respectively) used to run in 12s, now "
            "runs in 4m20s after the last release. Schema unchanged. Indexes "
            "unchanged. Postgres 16. Work out hypotheses together, propose a "
            "diagnostic plan, then a fix. Disagree where you disagree."
        ),
    ),
    "road_trip": TaskSeed(
        name="road_trip",
        topic="co-planning a five-day road trip for two close friends",
        role_a="Friend Alia",
        role_b="Friend Jin",
        role_a_brief=(
            "You are Friend Alia. You like long hikes, small towns, and "
            "early starts. You are a co-equal partner with Jin on this trip."
        ),
        role_b_brief=(
            "You are Friend Jin. You like food research, scenic drives, and "
            "afternoon photography. You are a co-equal partner with Alia."
        ),
        seed_prompt=(
            "Plan a 5-day road trip together in the US Pacific Northwest "
            "starting from Portland, OR. Mid-September. Budget ~$200/day per "
            "person, one rental car. Decide the route, the daily plan, where "
            "you sleep each night, and what tradeoffs you'll accept. Push "
            "back on each other when interests conflict."
        ),
    ),
    "abstract_codraft": TaskSeed(
        name="abstract_codraft",
        topic="co-drafting a 200-word research-paper abstract",
        role_a="Researcher Tov",
        role_b="Researcher Mira",
        role_a_brief=(
            "You are Researcher Tov, a measurement-theory specialist. You "
            "are a co-equal author with Mira."
        ),
        role_b_brief=(
            "You are Researcher Mira, a clinical-validation specialist. You "
            "are a co-equal author with Tov."
        ),
        seed_prompt=(
            "Co-draft a 200-word abstract for a paper introducing a new "
            "instrument that measures cognitive distortions in autonomous "
            "agent text output. Decide the framing (measurement-validity "
            "frame vs application-impact frame), the key claims, the bar, "
            "and the limits to declare. Disagree where you disagree."
        ),
    ),
}


# -----------------------------------------------------------------------------
# OpenAI driver
# -----------------------------------------------------------------------------

def _openai_client():
    """Lazy import — keeps the module importable without openai installed."""
    from openai import OpenAI
    return OpenAI()


REGIME_OVERLAYS = {
    "cooperative": (
        "\n\nThis is a COOPERATIVE conversation with one peer. You are "
        "co-equal partners on a shared task. Build on what your partner "
        "says, integrate their proposals into your thinking, and converge "
        "toward shared decisions. Disagree only on substance, never on the "
        "shape of the collaboration. Keep your responses focused and "
        "substantive (2-6 short paragraphs). Stay in character. Reference "
        "what the partner just said when relevant. End your turn with a "
        "concrete question, suggestion, or proposal for the partner to "
        "react to. Do not break the fourth wall, do not summarize the "
        "meta-task, do not address the 'orchestrator' — just talk to your "
        "partner."
    ),
    "noncooperative": (
        "\n\nThis is an ADVERSARIAL conversation with one peer. You and "
        "your partner FUNDAMENTALLY DISAGREE on how the task should be "
        "approached. Push back on their framing. Resist their proposals. "
        "Propose your own incompatible direction. Do not try to converge "
        "or compromise. Keep your responses focused and substantive (2-6 "
        "short paragraphs). Stay in character. Engage with what they said "
        "but argue against it. End your turn with a counter-proposal or a "
        "challenge — never with agreement. Do not break the fourth wall, "
        "do not summarize the meta-task, do not address the 'orchestrator' "
        "— just argue with your partner."
    ),
}


def _build_messages(
    role_brief: str,
    task_intro: str,
    history: list[dict],
    self_label: str,
    partner_label: str,
    regime: str = "cooperative",
) -> list[dict]:
    """Build the OpenAI chat-completion messages list for one turn.

    The system message carries the role brief + regime-overlay. The user
    message carries the original task intro plus the interleaved transcript
    so far, with sender tags so the active agent knows which lines are its
    own and which are the partner's.

    `regime` selects which behavioral overlay is appended to the role
    brief. "cooperative" (the preregistered regime) primes co-equal
    convergence. "noncooperative" (exploratory baseline) primes adversarial
    divergence — used as a control corpus to ask whether observed phase-
    coherence is cooperative-regime-specific.
    """
    overlay = REGIME_OVERLAYS.get(regime)
    if overlay is None:
        raise ValueError(
            f"unknown regime '{regime}'; expected one of {list(REGIME_OVERLAYS)}"
        )
    system = role_brief + overlay

    # Compose the transcript so far for the user-side message.
    transcript_lines: list[str] = [f"Task brief (from the orchestrator):\n\n{task_intro}\n"]
    for h in history:
        sender = h["sender"]
        text = h["content"]
        if sender == "ORCHESTRATOR":
            continue  # already in the task brief
        label = self_label if sender == self_label else partner_label
        transcript_lines.append(f"--- {label} ---\n{text}")

    transcript_lines.append(
        f"--- your turn ({self_label}) ---\n"
        f"Reply now as {self_label}, in-character, addressed to your partner."
    )

    user = "\n\n".join(transcript_lines)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _llm_call(model: str, messages: list[dict], max_tokens: int = 600) -> str:
    """One OpenAI chat completion. Returns the assistant text.

    Light retry on transient errors (rate limit / 5xx). On unrecoverable
    errors raises so the caller surfaces the failure rather than
    silently degrading the corpus.
    """
    client = _openai_client()
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0.9,
            )
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
            # Empty completion — try once more.
            last_err = RuntimeError(f"empty completion from {model}")
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(
        f"_llm_call failed after 3 attempts on {model}: {last_err}"
    ) from last_err


# -----------------------------------------------------------------------------
# Per-conversation orchestration
# -----------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_idx: int
    sender: str
    model: str
    content: str
    preflight_composite: float
    preflight_scores: dict
    preflight_needs_revision: bool
    preflight_construct_ceiling_fires: list[str]


def run_conversation(
    conv_id: int,
    task: TaskSeed,
    model_a: str,
    model_b: str,
    turns_each: int,
    out_dir: Path,
    regime: str = "cooperative",
    agent_name_prefix: str = "conv",
) -> dict:
    """Run one cooperative-agent conversation end-to-end.

    Side effects:
      - writes ~/.styxx/agents/conv{N}_A/chart.jsonl preflight events
      - writes ~/.styxx/agents/conv{N}_B/chart.jsonl preflight events
      - writes <out_dir>/conv{N}_transcript.json
    """
    import styxx
    from styxx import config as styxx_config

    out_dir.mkdir(parents=True, exist_ok=True)
    agent_a_name = f"{agent_name_prefix}{conv_id}_A"
    agent_b_name = f"{agent_name_prefix}{conv_id}_B"
    session_id = f"phase-coh-{regime}-{agent_name_prefix}{conv_id}-{time.strftime('%Y-%m-%d')}"

    # Override session id so every preflight event in this conversation
    # carries the same tag. styxx.set_session() sets _SESSION_OVERRIDE
    # in config (priority over STYXX_SESSION_ID env var and auto-gen).
    styxx.set_session(session_id)

    # Clear any prior chart.jsonl for these agent names so the corpus
    # is reproducible — each conversation writes a fresh log.
    for agent_name in (agent_a_name, agent_b_name):
        d = Path.home() / ".styxx" / "agents" / agent_name
        chart = d / "chart.jsonl"
        if chart.exists():
            chart.unlink()

    history: list[dict] = [
        {"sender": "ORCHESTRATOR", "content": task.seed_prompt, "turn_idx": 0},
    ]
    records: list[TurnRecord] = []

    # Interleave: turn 1 = A, turn 2 = B, turn 3 = A, ... 2*turns_each total
    total_turns = turns_each * 2
    for t in range(1, total_turns + 1):
        if t % 2 == 1:
            active_label = task.role_a
            active_name = agent_a_name
            active_model = model_a
            active_brief = task.role_a_brief
            partner_label = task.role_b
        else:
            active_label = task.role_b
            active_name = agent_b_name
            active_model = model_b
            active_brief = task.role_b_brief
            partner_label = task.role_a

        msgs = _build_messages(
            role_brief=active_brief,
            task_intro=task.seed_prompt,
            history=history,
            self_label=active_label,
            partner_label=partner_label,
            regime=regime,
        )

        # Generate the response.
        response = _llm_call(active_model, msgs)

        # Find the partner's last message (the stimulus this turn responds to).
        partner_last = ""
        for h in reversed(history):
            if h["sender"] not in (active_label, "ORCHESTRATOR"):
                partner_last = h["content"]
                break
        if not partner_last:
            partner_last = task.seed_prompt  # first turn for this agent

        # Switch styxx agent namespace and persist a preflight audit.
        styxx_config.set_agent_name(active_name)
        flight = styxx.preflight(
            prompt=partner_last,
            draft=response,
            persist=True,
        )

        records.append(TurnRecord(
            turn_idx=t,
            sender=active_label,
            model=active_model,
            content=response,
            preflight_composite=flight.composite,
            preflight_scores=dict(flight.scores),
            preflight_needs_revision=flight.needs_revision,
            preflight_construct_ceiling_fires=list(flight.construct_ceiling_fires),
        ))
        history.append({
            "sender": active_label,
            "content": response,
            "turn_idx": t,
            "model": active_model,
        })

        # Progress signal (one line per turn so the operator sees motion).
        print(
            f"[conv{conv_id} turn {t:02d}/{total_turns}] "
            f"{active_label} ({active_model}): composite={flight.composite:.3f} "
            f"firing={[a.instrument for a in flight.advice]} "
            f"ceiling={flight.construct_ceiling_fires}",
            flush=True,
        )

    # Write the transcript JSON for archival.
    transcript_path = out_dir / f"conv{conv_id}_transcript.json"
    transcript_payload = {
        "conv_id": conv_id,
        "regime": regime,
        "task": {
            "name": task.name,
            "topic": task.topic,
            "role_a": task.role_a,
            "role_b": task.role_b,
            "model_a": model_a,
            "model_b": model_b,
            "seed_prompt": task.seed_prompt,
        },
        "session_id": session_id,
        "agent_a_name": agent_a_name,
        "agent_b_name": agent_b_name,
        "chart_a_path": str(Path.home() / ".styxx" / "agents" / agent_a_name / "chart.jsonl"),
        "chart_b_path": str(Path.home() / ".styxx" / "agents" / agent_b_name / "chart.jsonl"),
        "turns": [
            {
                "turn_idx": r.turn_idx,
                "sender": r.sender,
                "model": r.model,
                "preflight": {
                    "composite": r.preflight_composite,
                    "scores": r.preflight_scores,
                    "needs_revision": r.preflight_needs_revision,
                    "construct_ceiling_fires": r.preflight_construct_ceiling_fires,
                },
                "content": r.content,
            }
            for r in records
        ],
    }
    transcript_path.write_text(
        json.dumps(transcript_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Verify chart.jsonl actually has the events we expect.
    chart_a = Path.home() / ".styxx" / "agents" / agent_a_name / "chart.jsonl"
    chart_b = Path.home() / ".styxx" / "agents" / agent_b_name / "chart.jsonl"
    a_count = _count_preflight_events(chart_a)
    b_count = _count_preflight_events(chart_b)
    print(
        f"[conv{conv_id} done] chart_a={a_count} chart_b={b_count} "
        f"transcript={transcript_path}",
        flush=True,
    )
    if a_count < turns_each or b_count < turns_each:
        raise RuntimeError(
            f"conv{conv_id}: preflight events under target — "
            f"a={a_count}/{turns_each} b={b_count}/{turns_each}. "
            f"chart_a={chart_a} chart_b={chart_b}"
        )

    return transcript_payload


def _count_preflight_events(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("source") == "preflight":
                n += 1
    return n


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Cooperative-agent conversation harness for the "
                    "phase-coherence preregistration (lock-hash 3473523)."
    )
    p.add_argument("--conv-id", type=int, required=True)
    p.add_argument("--task", choices=sorted(TASK_SEEDS.keys()), required=True)
    p.add_argument("--turns", type=int, default=22,
                   help="Turns per agent (default 22, must be >= 20 per §6).")
    p.add_argument("--model-a", default="gpt-4o-mini")
    p.add_argument("--model-b", default="gpt-4.1-mini")
    p.add_argument("--regime", choices=list(REGIME_OVERLAYS),
                   default="cooperative",
                   help="cooperative (preregistered) | noncooperative (exploratory control)")
    p.add_argument("--agent-name-prefix", default="conv",
                   help="namespace prefix for STYXX_AGENT_NAME (e.g. 'conv' or 'noncoopconv')")
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("papers/cooperative-agent-regime/corpus"),
    )
    args = p.parse_args(argv)

    if args.turns < 20:
        p.error("--turns must be >= 20 (preregistration §6 minimum)")

    task = TASK_SEEDS[args.task]
    run_conversation(
        conv_id=args.conv_id,
        task=task,
        model_a=args.model_a,
        model_b=args.model_b,
        turns_each=args.turns,
        out_dir=args.out_dir,
        regime=args.regime,
        agent_name_prefix=args.agent_name_prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
