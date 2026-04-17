"""
examples/quickstart.py -- the styxx hello-world.

This is the first example to run. It shows the core styxx value
proposition in a single screen of code:

    1. Swap `from openai import OpenAI` for `from styxx import OpenAI`.
    2. Call chat.completions.create() exactly like you always have.
    3. Read `response.vitals` for a free cognitive vitals card on every call.

If OPENAI_API_KEY is not set we fall back to a bundled trajectory demo
so you can see the vitals card immediately, no credentials required.

Run:

    python examples/quickstart.py
"""

import os
import sys
from pathlib import Path

# Resolve the local package when running from a source checkout. Not
# needed once styxx is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from styxx import OpenAI


def live_demo() -> None:
    """Real OpenAI call -- requires OPENAI_API_KEY."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
    )
    print("\nmodel response:")
    print(f"  {response.choices[0].message.content}\n")
    print(response.vitals.summary)


def offline_demo() -> None:
    """Fallback demo using a bundled logprob trajectory -- no key required."""
    import styxx
    from styxx.cli import _get_demo_trajectory

    print("OPENAI_API_KEY not set -- running offline trajectory demo instead.")
    print("To see the live path, set OPENAI_API_KEY and re-run.\n")

    entropy, logprob, top2, _prompt = _get_demo_trajectory("reasoning")
    vitals = styxx.Raw().read(entropy=entropy, logprob=logprob, top2_margin=top2)
    print(vitals.summary)


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        live_demo()
    else:
        offline_demo()
