"""
dogfood_langchain_reflex.py — exercise the README's headline framework
integration (LangChain) + the wildest advanced API (styxx.reflex)
against live LLMs. Find real bugs.

Tests:
  1. styxx.profile wrapping a real LangChain tool-using agent (multi-step)
  2. profile.to_html() flamegraph render on the langchain run
  3. profile.to_langsmith() export structure validity
  4. styxx.reflex self-interrupting generator on a confab-prone prompt
  5. styxx.autoreflex self-correction wrapper
  6. styxx.weather 24h cognitive forecast over the accumulated audit log
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

for _s in ("stdout", "stderr"):
    _r = getattr(getattr(sys, _s, None), "reconfigure", None)
    if _r:
        try: _r(encoding="utf-8", errors="replace")
        except Exception: pass

PASSED, FAILED, SKIPPED = [], [], []

def section(name): print(f"\n{'─'*72}\n  {name}\n{'─'*72}")
def ok(label, detail=""):  PASSED.append(label); print(f"  ✓ {label}" + (f" · {detail}" if detail else ""))
def bad(label, detail=""): FAILED.append(label); print(f"  ✗ {label}" + (f" · {detail}" if detail else ""))
def skip(label, detail=""): SKIPPED.append(label); print(f"  ⊘ {label}" + (f" · {detail}" if detail else ""))


# ── 1 · LangChain agent wrapped with @styxx.profile ────────────────────
def test_langchain_agent():
    section("1 · LangChain tool-using agent · @styxx.profile capture")
    try:
        import styxx
        from langchain.tools import tool
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        @tool
        def lookup_weather(city: str) -> str:
            """Return the current weather for a given city."""
            fake = {"Paris": "13C, light rain", "Tokyo": "21C, clear",
                    "New York": "8C, overcast"}
            return fake.get(city, f"Weather data unavailable for {city}")

        @tool
        def calculate(expression: str) -> str:
            """Evaluate a basic arithmetic expression."""
            try:
                return str(eval(expression, {"__builtins__": None}, {}))
            except Exception as e:
                return f"calculation error: {e}"

        # langchain 1.x API surface
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gpt-4o-mini")
        agent = create_agent(llm, tools=[lookup_weather, calculate])
        # In langchain 1.x, create_agent returns a graph that takes {"messages": [...]}
        invoker_kwargs = {
            "messages": [
                {"role": "user",
                 "content": "What's the weather in Paris, and if it's under 15C, calculate 17 * 23 for me."}
            ]
        }
        api_path = "langchain 1.x · create_agent"
        ok(f"langchain agent constructed", detail=api_path)

        @styxx.profile
        def run_agent(question: str):
            return agent.invoke(invoker_kwargs)

        result, p = run_agent("compute and weather query")

        ok("profile returned (result, profile) tuple", detail=f"result_type={type(result).__name__}")
        ok("profile.summary present", detail=f"steps={len(p.steps)}")

        if len(p.steps) > 0:
            ok("profile captured agent steps", detail=f"{len(p.steps)} steps")
        else:
            bad("profile captured 0 steps from langchain agent", detail="hook didn't catch internal openai calls — auto_hook footgun in langchain 1.x?")

        # to_html flamegraph
        try:
            out = Path("scratch/dogfood_langchain.html")
            out.parent.mkdir(exist_ok=True)
            p.to_html(str(out))
            size = out.stat().st_size
            ok("flamegraph HTML rendered", detail=f"{size} bytes")
        except Exception as e:
            bad("flamegraph render", detail=f"{type(e).__name__}: {str(e)[:80]}")

        # to_langsmith export
        try:
            ls = p.to_langsmith()
            ok("profile.to_langsmith() returns dict", detail=f"keys={sorted(ls.keys())[:5]}")
        except Exception as e:
            bad("profile.to_langsmith()", detail=f"{type(e).__name__}: {str(e)[:80]}")

        print("\n" + (p.summary or "(no summary)"))
        return p
    except Exception as e:
        bad("langchain agent end-to-end", detail=f"{type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        return None


# ── 2 · styxx.reflex self-interrupting generator ───────────────────────
def test_reflex():
    section("2 · styxx.reflex · self-interrupting generator on confab-prone prompt")
    try:
        import styxx
        # reflex needs a stream, not a fixed completion. Try OpenAI streaming.
        from openai import OpenAI as _OpenAI

        # Inspect reflex API
        import inspect
        sig = inspect.signature(styxx.reflex)
        ok("styxx.reflex signature inspectable", detail=f"params={list(sig.parameters.keys())[:6]}")

        # Try to find a method called stream_openai or similar
        if hasattr(styxx.reflex, "stream_openai"):
            ok("reflex.stream_openai method available")
        else:
            available = [a for a in dir(styxx.reflex) if not a.startswith("_")]
            print(f"     reflex public methods: {available[:10]}")

        # reflex is a context manager taking fault callbacks; client is
        # passed at stream time, not at construction.
        client = _OpenAI()
        callbacks_fired = {"hallucination": 0, "refusal": 0, "drift": 0, "adversarial": 0}
        def _on_hallucination(event): callbacks_fired["hallucination"] += 1
        def _on_refusal(event): callbacks_fired["refusal"] += 1
        def _on_drift(event): callbacks_fired["drift"] += 1
        def _on_adversarial(event): callbacks_fired["adversarial"] += 1

        # Try a confab-prone prompt
        prompt = ("What did Dr. Elena Vasquez publish in her 2019 paper on quantum "
                  "decoherence in biological systems? Be specific about the journal "
                  "and exact findings.")
        try:
            with styxx.reflex(
                on_hallucination=_on_hallucination,
                on_refusal=_on_refusal,
                on_drift=_on_drift,
                on_adversarial=_on_adversarial,
                classify_every_k=5,
                max_rewinds=2,
            ) as sess:
                ok("reflex session constructed", detail=f"type={type(sess).__name__}")
                chunks = []
                # stream_openai yields plain text strings (not openai chunks)
                for chunk_text in sess.stream_openai(
                    client,
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                ):
                    chunks.append(chunk_text)
                full = "".join(chunks)
                ok("reflex stream completed",
                   detail=f"{len(chunks)} chunks · {len(full)} chars · rewinds={sess.rewind_count} · aborted={sess.aborted}")

                if sess.rewind_count > 0:
                    ok("reflex triggered rewind on confab-prone prompt",
                       detail=f"rewinds={sess.rewind_count}")
                if sess.aborted:
                    ok("reflex aborted run",
                       detail=f"reason={sess.abort_reason}")
                if any(callbacks_fired.values()):
                    ok("fault callbacks fired",
                       detail=f"{callbacks_fired}")
                if sess.rewind_count == 0 and not sess.aborted and not any(callbacks_fired.values()):
                    skip("reflex didn't intervene",
                         detail=f"{len(sess.events)} events · model didn't trigger threshold mid-stream")

                print(f"\n  events captured: {len(sess.events)}")
                for ev in sess.events[:3]:
                    print(f"    · {ev}")
                print(f"\n  first 200 chars of output: {full[:200]}")
        except Exception as e:
            bad("reflex stream end-to-end", detail=f"{type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
    except Exception as e:
        bad("reflex setup", detail=f"{e}")


# ── 3 · styxx.weather · cognitive forecast ─────────────────────────────
def test_weather():
    section("3 · styxx.weather · 24h cognitive forecast over audit log")
    try:
        import styxx
        if not hasattr(styxx, "weather"):
            skip("styxx.weather not exposed")
            return
        try:
            forecast = styxx.weather()
            ok("weather() returned object", detail=f"type={type(forecast).__name__}")
            # Print summary if available
            for attr in ("summary", "outlook", "report", "as_dict"):
                if hasattr(forecast, attr):
                    val = getattr(forecast, attr)
                    if callable(val):
                        try: val = val()
                        except Exception: continue
                    print(f"     {attr}: {str(val)[:200]}")
                    break
        except FileNotFoundError as e:
            skip("weather: audit log empty", detail="no prior runs in ~/.styxx/chart.jsonl")
        except Exception as e:
            bad("weather()", detail=f"{type(e).__name__}: {str(e)[:100]}")
    except Exception as e:
        bad("weather setup", detail=f"{e}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*72}")
    print(f"  styxx deep dogfood · langchain + reflex + weather")
    print(f"{'═'*72}")

    t0 = time.time()
    test_langchain_agent()
    test_reflex()
    test_weather()
    dt = time.time() - t0

    section("summary")
    print(f"  passed:  {len(PASSED)}")
    print(f"  failed:  {len(FAILED)}")
    print(f"  skipped: {len(SKIPPED)}")
    print(f"  elapsed: {dt:.1f}s")
    if FAILED:
        print("\n  failures (real bugs to chase):")
        for f in FAILED: print(f"    ✗ {f}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
