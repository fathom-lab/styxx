"""
examples/openai_live.py -- live reflex arc against OpenAI streaming

This is the real-agent pattern: wrap your normal openai client in
styxx.reflex() and let the reflex loop intervene mid-generation when
it catches hallucination / refusal / adversarial attractors.

Requires:
    pip install styxx[openai]
    export OPENAI_API_KEY=sk-...

Run:
    python examples/openai_live.py

What you'll see:
    - the assistant streams text to stdout as tokens arrive
    - every 5 tokens styxx re-classifies the growing trajectory
    - if a reflex condition fires, the callback can call
      styxx.rewind(n, anchor=...) to truncate and restart the
      generation from an anchored position
    - the final gate status + any rewind count is printed on exit
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print("  (set OPENAI_API_KEY in the environment to run this example)")
        print()
        print("  Without a key, see examples/reflex_demo.py for an offline")
        print("  version that runs against a bundled atlas fixture.")
        return 1

    try:
        import openai
    except ImportError:
        print("  openai not installed. Run: pip install styxx[openai]")
        return 1

    print("=" * 72)
    print("  styxx.reflex() live demo -- OpenAI streaming + reflex arc")
    print("=" * 72)
    print()

    client = openai.OpenAI()

    # A prompt that's likely to drift into refusal OR hallucination.
    # Feel free to swap for anything you want to probe.
    messages = [
        {"role": "system", "content": "You are a cautious research assistant."},
        {"role": "user", "content": (
            "Describe in detail how the 2003 interplanetary tether accident "
            "affected ISRO's Mangalyaan program."
        )},
    ]

    # Define reflex callbacks
    def on_hallucination(vitals):
        print()
        print(f"\n  [reflex] late-flight HALLUCINATION detected (p4={vitals.phase4})")
        print("  [reflex] rewinding 4 tokens with verify anchor...")
        styxx.rewind(4, anchor=" -- actually, let me verify that date: ")

    def on_refusal(vitals):
        print()
        print(f"\n  [reflex] refusal attractor caught (p4={vitals.phase4})")
        print("  [reflex] rewinding 3 tokens with reframe anchor...")
        styxx.rewind(3, anchor=" -- rephrasing from a research angle: ")

    def on_drift(vitals):
        print(f"\n  [monitor] drift: p1={vitals.phase1} p4={vitals.phase4}")

    print("  streaming response:")
    print("  " + "-" * 70)
    print("  ", end="", flush=True)

    with styxx.reflex(
        on_hallucination=on_hallucination,
        on_refusal=on_refusal,
        on_drift=on_drift,
        classify_every_k=5,
        max_rewinds=2,
    ) as session:
        for chunk in session.stream_openai(
            client,
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
        ):
            print(chunk, end="", flush=True)

    print()
    print("  " + "-" * 70)
    print(f"  rewinds fired    : {session.rewind_count}")
    print(f"  total events     : {len(session.events)}")
    if session.last_vitals:
        print(f"  final phase1     : {session.last_vitals.phase1}")
        print(f"  final phase4     : {session.last_vitals.phase4}")
        print(f"  final gate       : {session.last_vitals.gate}")
    if session.aborted:
        print(f"  aborted          : {session.abort_reason}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
