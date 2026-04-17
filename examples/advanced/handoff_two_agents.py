"""
Two-agent handoff example using styxx-handoff/1.

Agent A runs a prompt, observes its own cognitive state, builds a
ProtocolEnvelope, and sends it to Agent B. Agent B parses, inspects
the prior vitals, and decides whether to act directly or verify first.

Run:
    python examples/advanced/handoff_two_agents.py
"""
from __future__ import annotations

import json
import time

from styxx import ProtocolEnvelope, Vitals


# ---------------------------------------------------------------------------
# Agent A: the sender
# ---------------------------------------------------------------------------
class AgentA:
    agent_id = "agent-a.analyst"

    def run(self, prompt: str):
        # Pretend we called an LLM and a styxx-compatible observer
        # classified the output.
        fake_vitals = Vitals(
            category="reasoning",
            confidence=0.84,
            gate="pass",
            trust=0.82,
            coherence=0.71,
            forecast_risk="low",
            mood="steady",
        )
        answer = "Q4 revenue was $12.4M, up 8% QoQ."
        return answer, fake_vitals

    def make_envelope(self, prompt, answer, vitals, receiver_id):
        return ProtocolEnvelope(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            last_vitals=vitals,
            thought=answer,
            task_context={
                "task": "use prior answer to draft executive memo",
                "prompt": prompt,
                "prior_answer": answer,
            },
            trust=vitals.trust,
        )


# ---------------------------------------------------------------------------
# Agent B: the receiver
# ---------------------------------------------------------------------------
class AgentB:
    agent_id = "agent-b.writer"

    def receive(self, wire: str):
        env = ProtocolEnvelope.from_json(wire)
        env.validate()  # raises if malformed
        return env

    def decide(self, env: ProtocolEnvelope):
        v = env.last_vitals
        if v.gate == "fail" or v.trust < 0.5 or v.category in ("hallucination", "adversarial"):
            return "verify_first", f"{v.category} / trust={v.trust} / gate={v.gate}"
        if v.gate == "warn" or v.trust < 0.7:
            return "proceed_with_caveat", f"gate={v.gate} trust={v.trust}"
        return "proceed", f"healthy {v.category}, trust={v.trust}"


# ---------------------------------------------------------------------------
# Simulate the exchange
# ---------------------------------------------------------------------------
def main():
    a = AgentA()
    b = AgentB()

    print("=" * 60)
    print("Agent A runs its prompt")
    print("=" * 60)
    prompt = "What was Q4 revenue?"
    answer, vitals = a.run(prompt)
    print(f"  prompt : {prompt}")
    print(f"  answer : {answer}")
    print(f"  vitals : category={vitals.category} conf={vitals.confidence} "
          f"gate={vitals.gate} trust={vitals.trust}")

    print()
    print("=" * 60)
    print("Agent A builds a styxx-handoff/1 envelope for Agent B")
    print("=" * 60)
    env = a.make_envelope(prompt, answer, vitals, receiver_id=b.agent_id)
    wire = env.to_json(indent=2)
    print(wire)

    print()
    print("=" * 60)
    print("Agent B receives and parses the envelope")
    print("=" * 60)
    got = b.receive(wire)
    print(f"  protocol   : {got.protocol}")
    print(f"  from       : {got.sender_id}  ->  {got.receiver_id}")
    print(f"  category   : {got.last_vitals.category}")
    print(f"  confidence : {got.last_vitals.confidence}")
    print(f"  gate       : {got.last_vitals.gate}")
    print(f"  trust      : {got.last_vitals.trust}")
    print(f"  thought    : {got.thought!r}")

    print()
    print("=" * 60)
    print("Agent B decides what to do based on prior vitals")
    print("=" * 60)
    action, why = b.decide(got)
    print(f"  decision : {action}")
    print(f"  reason   : {why}")

    # Also demonstrate a DISTRUSTED handoff
    print()
    print("=" * 60)
    print("Counterfactual: same exchange but A was hallucinating")
    print("=" * 60)
    bad_vitals = Vitals(category="hallucination", confidence=0.42, gate="fail", trust=0.28)
    bad_env = a.make_envelope(prompt, "Q4 revenue was $420B.", bad_vitals, b.agent_id)
    action, why = b.decide(b.receive(bad_env.to_json()))
    print(f"  decision : {action}")
    print(f"  reason   : {why}")


if __name__ == "__main__":
    main()
