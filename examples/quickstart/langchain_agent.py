"""
LangChain agent + styxx — drop-in cognitive observability.

Run:
    pip install -U styxx langchain langchain-openai
    export OPENAI_API_KEY=sk-...
    python langchain_agent.py

Output:
    A standard LangChain tool-using agent runs your task. styxx wraps
    every LLM call inside, then renders a flamegraph showing where the
    agent's cognition shifted (planning → execution → answer) and
    flagging any drift, refusal, or confabulation along the way.

Spec: https://doi.org/10.5281/zenodo.19746215
"""

import styxx
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ── 1. Define some tools the agent can call ────────────────────────
@tool
def lookup_weather(city: str) -> str:
    """Return the current weather for a given city."""
    # Stub — replace with a real API in production.
    fake = {"Paris": "13°C, light rain", "Tokyo": "21°C, clear",
            "New York": "8°C, overcast"}
    return fake.get(city, f"Weather data unavailable for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"calculation error: {e}"


# ── 2. Build the agent (standard LangChain) ────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [lookup_weather, calculate]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# ── 3. Wrap the agent invocation with @styxx.profile ───────────────
@styxx.profile
def run_agent(question: str):
    return executor.invoke({"input": question})


# ── 4. Use it ──────────────────────────────────────────────────────
if __name__ == "__main__":
    result, profile = run_agent(
        "What's the weather in Paris, and if it's under 15°C, "
        "calculate 17 * 23 for me."
    )

    print("Agent answer:")
    print(f"  {result['output']}\n")

    print(profile.summary)
    print()

    # Save flamegraph
    out = "agent_run.html"
    profile.to_html(out)
    print(f"Flamegraph written to: {out}")

    # Export to LangSmith if you want
    # ls_trace = profile.to_langsmith()
    # langsmith_client.create_run(**ls_trace)
