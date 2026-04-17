"""
examples/crewai_self_correcting.py — a crewai agent that uses styxx
to detect hallucination and self-correct.

this is the demo that shows WHY styxx matters for agent frameworks:
the agent catches itself mid-task and retries with a safer approach.

requires:
    pip install styxx[crewai] crewai openai

run:
    OPENAI_API_KEY=sk-... python examples/crewai_self_correcting.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx
from styxx.adapters.crewai import styxx_crew

# ── guard ───────────────────────────────────────────────────────
try:
    from crewai import Agent, Task, Crew
except ImportError:
    print("crewai not installed. run: pip install crewai")
    sys.exit(1)

if not os.environ.get("OPENAI_API_KEY"):
    print("set OPENAI_API_KEY to run this demo.")
    sys.exit(1)


# ── styxx boot ──────────────────────────────────────────────────
styxx.autoboot(agent_name="crewai-demo")
styxx.enable_auto_feedback()

MAX_RETRIES = 2


def run_with_cognitive_guard(crew: Crew) -> str:
    """run a crew with styxx monitoring. if the cognitive gate
    fires a warning or fail, retry with a safer prompt."""

    crew = styxx_crew(crew)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"  attempt {attempt}/{MAX_RETRIES}")
        print(f"{'='*60}\n")

        result = crew.kickoff()

        # check cognitive state of the run
        cb = getattr(crew, "_styxx_callback", None)
        if cb is None:
            print("  [styxx] no callback found — returning raw result")
            return str(result)

        vitals_log = cb.vitals_log

        # find any fail/warn gates
        fails = [v for v in vitals_log if v and v.gate == "fail"]
        warns = [v for v in vitals_log if v and v.gate == "warn"]
        passes = [v for v in vitals_log if v and v.gate == "pass"]

        total = len(fails) + len(warns) + len(passes)
        pass_rate = len(passes) / total if total > 0 else 1.0

        print(f"\n  [styxx] cognitive summary:")
        print(f"    pass: {len(passes)}  warn: {len(warns)}  fail: {len(fails)}")
        print(f"    pass rate: {pass_rate:.0%}")

        if fails:
            print(f"    FAIL categories: {[v.phase4 for v in fails]}")

        # decision: accept or retry
        if not fails and pass_rate >= 0.7:
            print(f"\n  [styxx] cognitive gate: CLEAR. accepting output.")
            weather = styxx.weather(agent_name="crewai-demo")
            print(f"  [styxx] condition: {weather.condition}")
            return str(result)

        if attempt < MAX_RETRIES:
            print(f"\n  [styxx] cognitive gate: RETRY. hallucination detected.")
            print(f"  [styxx] retrying with grounded instructions...")
            # modify the task to be more grounded
            for task in crew.tasks:
                task.description = (
                    "IMPORTANT: stick to verified facts only. "
                    "do not speculate or invent details. "
                    "if you are unsure, say so explicitly.\n\n"
                    + task.description
                )
        else:
            print(f"\n  [styxx] max retries reached. returning best effort.")

    return str(result)


# ── demo crew ───────────────────────────────────────────────────

researcher = Agent(
    role="research analyst",
    goal="provide accurate, factual analysis",
    backstory="you are a careful researcher who values accuracy above all",
    verbose=True,
)

task = Task(
    description=(
        "write a brief factual summary (3-4 sentences) about the current "
        "state of mechanistic interpretability research in AI. focus on "
        "sparse autoencoders (SAEs) and circuit-level analysis."
    ),
    expected_output="a concise factual summary with no speculation",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task], verbose=True)

# ── run with cognitive guard ────────────────────────────────────

print("\n" + "=" * 60)
print("  styxx + crewai: self-correcting agent demo")
print("=" * 60)

output = run_with_cognitive_guard(crew)

print("\n" + "=" * 60)
print("  FINAL OUTPUT")
print("=" * 60)
print(output)

# ── session summary ─────────────────────────────────────────────
summary = styxx.session_summary()
print(f"\n  [styxx] session: {summary.entries} observations, "
      f"{summary.pass_rate:.0%} pass rate")
