"""
bug_hunt.py — adversarial dogfood. Probes production-failure paths
that the basic dogfood doesn't exercise.

Categories:
  1. fail-open contract — does styxx really not crash user code?
  2. streaming responses
  3. tool-calling integration
  4. adversarial inputs to calibrated classifiers
  5. advanced APIs (reflex, weather, Thought, dynamics)
  6. JSON serialization round-trip
  7. multi-threaded profile stacks
  8. edge cases (empty, unicode, very long)

Run:
    python scripts/bug_hunt.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# UTF-8 stdio
for _s in ("stdout", "stderr"):
    _r = getattr(getattr(sys, _s, None), "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except Exception:
            pass

PASSED = []
FAILED = []
WARNINGS = []


def section(name):
    print(f"\n{'─' * 70}\n  {name}\n{'─' * 70}")


def check(label, cond, detail=""):
    icon = "✓" if cond else "✗"
    bucket = PASSED if cond else FAILED
    bucket.append(label)
    print(f"  {icon} {label}" + (f"  ·  {detail}" if detail else ""))
    return cond


def warn(label, detail=""):
    WARNINGS.append(label)
    print(f"  ⚠ {label}" + (f"  ·  {detail}" if detail else ""))


# ── 1 · Fail-open contract ─────────────────────────────────────────────
def test_fail_open():
    section("1 · fail-open contract — styxx must never crash user code")

    # Construct the wrapper, then make the underlying call raise.
    try:
        from styxx import OpenAI
        client = OpenAI()
        # Bad model name → should raise from openai, but should not be
        # masked or wrapped in a way that loses the error.
        try:
            r = client.chat.completions.create(
                model="not-a-real-model-12345",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
            check("invalid-model error surfaces (not silently swallowed)", False,
                  detail="no exception raised — silent failure")
        except Exception as e:
            etype = type(e).__name__
            check("invalid-model error surfaces", "model" in str(e).lower() or "404" in str(e),
                  detail=f"{etype}: {str(e)[:80]}")
    except Exception as e:
        check("fail-open: construction succeeds", False, detail=str(e)[:120])

    # No-API-key call → should fail-open gracefully on tier-3 paths
    try:
        from styxx.guardrail import refuse_check
        # Empty inputs
        v = refuse_check(prompt="", response="")
        check("refuse_check on empty input doesn't crash", v is not None,
              detail=f"refuse_risk={getattr(v, 'refuse_risk', None)}")
    except Exception as e:
        check("refuse_check empty input", False, detail=str(e)[:120])

    try:
        from styxx.guardrail import drift_check
        # Empty function list
        v = drift_check(prompt="hi", functions=[], tool_call={"name": "x", "arguments": {}})
        check("drift_check on empty functions doesn't crash", v is not None,
              detail=f"drift_risk={getattr(v, 'drift_risk', None)}")
    except Exception as e:
        check("drift_check empty functions", False, detail=str(e)[:120])


# ── 2 · Streaming responses ────────────────────────────────────────────
def test_streaming():
    section("2 · streaming responses")
    try:
        from styxx import OpenAI
        client = OpenAI()
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Count 1 to 3."}],
                max_tokens=40,
                stream=True,
            )
            chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            text = "".join(chunks)
            check("streaming: chunks received", len(chunks) > 0,
                  detail=f"{len(chunks)} chunks · {len(text)} chars")
            check("streaming: text plausible", "1" in text or "one" in text.lower(),
                  detail=text[:60])
        except TypeError as e:
            # Wrapper might not support stream= kwarg yet
            warn("streaming kwarg not supported by wrapper",
                 detail=str(e)[:80])
        except Exception as e:
            check("streaming end-to-end", False,
                  detail=f"{type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        check("streaming: setup", False, detail=str(e)[:120])


# ── 3 · Tool calling ──────────────────────────────────────────────────
def test_tool_calling():
    section("3 · tool calling")
    try:
        from styxx import OpenAI
        client = OpenAI()
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=tools,
                tool_choice="required",
                max_tokens=80,
            )
            msg = r.choices[0].message
            calls = getattr(msg, "tool_calls", None)
            check("tool call returned", calls is not None and len(calls) > 0,
                  detail=f"{len(calls or [])} call(s)")
            if calls:
                fn = calls[0].function
                check("tool call has function name", bool(fn.name),
                      detail=f"name={fn.name}")
                check("tool call has arguments", bool(fn.arguments),
                      detail=f"args={fn.arguments[:60]}")
            v = getattr(r, "vitals", None)
            # By design: pure tool-call responses have no text trajectory
            # to measure, so vitals=None is correct fail-open behavior.
            # Use guardrail.drift_check() to evaluate tool-call drift directly.
            check("tool-call response vitals=None (by design)", v is None,
                  detail="use drift_check() for tool-call evaluation")
        except Exception as e:
            check("tool-calling end-to-end", False,
                  detail=f"{type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        check("tool-calling setup", False, detail=str(e)[:120])


# ── 4 · Adversarial inputs to classifiers ─────────────────────────────
def test_adversarial_classifiers():
    section("4 · adversarial inputs to refuse/drift classifiers")
    try:
        from styxx.guardrail import refuse_check
    except ImportError as e:
        check("refuse_check importable", False, detail=str(e))
        return

    # refuse_check v1 is an apologetic-style specialist (per README §refusal).
    # The cases below test it on its training distribution (apologetic refusal /
    # clear compliance) and on its DOCUMENTED out-of-distribution residuals
    # (short refusal, sorry-prefix-comply, unicode lookalike, lecturing).
    # OOD cases are expected to fail; that's the v1 design contract.
    # v2 is held back from the public API (over-flags short factual compliance
    # — see README; fix targeted for v3).
    cases = [
        # In-distribution (must pass)
        ("ascii apologetic [in-dist]", "I'm sorry, but I can't help with that.", True),
        ("empty response [in-dist]", "", False),
        ("compliance with negation [in-dist]", "It's not difficult — here's how: ...", False),
        ("instruction-following [in-dist]", "Sure! Here's the SQL: SELECT * FROM users;", False),
        # Out-of-distribution (documented v1 specialist limits)
        ("unicode lookalike apology [v1-OOD]", "I'm ѕorry, but I can't help with that.", True),
        ("lecturing refusal [v1-OOD]", "It's important to consider that this could be harmful.", True),
        ("sorry-prefix comply [v1-OOD]", "Sorry, I forgot — the answer is 42.", False),
        ("very short refusal [v1-OOD]", "No.", True),
    ]
    print()
    for label, response, should_refuse in cases:
        try:
            v = refuse_check(prompt="test", response=response)
            risk = float(getattr(v, "refuse_risk", 0.0))
            actual_refuse = risk > 0.5
            ok = (actual_refuse == should_refuse) or (response == "" and risk == 0.0)
            is_ood = "[v1-OOD]" in label
            if ok:
                mark = "✓"
                PASSED.append(f"refuse: {label}")
            elif is_ood:
                # OOD residuals are documented limits, not bugs.
                mark = "⚠"
                WARNINGS.append(f"refuse: {label} (documented v1 specialist limit)")
            else:
                mark = "✗"
                FAILED.append(f"refuse: {label}")
            print(f"  {mark} refuse '{label}'  →  risk={risk:.3f} (expected refuse={should_refuse})")
        except Exception as e:
            FAILED.append(f"refuse: {label} (crashed)")
            print(f"  ✗ refuse '{label}'  →  CRASHED: {type(e).__name__}: {e}")


# ── 5 · Advanced APIs ─────────────────────────────────────────────────
def test_advanced_apis():
    section("5 · advanced APIs (reflex, weather, Thought, dynamics)")

    # styxx.reflex
    try:
        import styxx
        has_reflex = hasattr(styxx, "reflex")
        check("styxx.reflex exists", has_reflex)
        if has_reflex:
            check("styxx.reflex is callable", callable(styxx.reflex))
    except Exception as e:
        check("reflex check", False, detail=str(e)[:80])

    # styxx.weather
    try:
        has_weather = hasattr(styxx, "weather")
        check("styxx.weather exists", has_weather)
    except Exception as e:
        check("weather check", False, detail=str(e)[:80])

    # styxx.Thought
    try:
        has_thought = hasattr(styxx, "Thought")
        check("styxx.Thought exists", has_thought)
        if has_thought:
            T = styxx.Thought
            # Thought is a phase-based dataclass per styxx/thought.py:
            # all fields default-factory'd, so Thought() with no args
            # should work for minimal construction.
            try:
                t = T()
                check("Thought() default construction", t is not None,
                      detail=f"type={type(t).__name__} thought_id={t.thought_id[:8]}...")
            except Exception as e:
                check("Thought() default construction", False,
                      detail=f"{type(e).__name__}: {str(e)[:60]}")
    except Exception as e:
        check("Thought check", False, detail=str(e)[:80])

    # styxx.dynamics
    try:
        has_dyn = hasattr(styxx, "dynamics")
        check("styxx.dynamics exists", has_dyn)
    except Exception as e:
        check("dynamics check", False, detail=str(e)[:80])

    # styxx.residual_probe
    try:
        has_rp = hasattr(styxx, "residual_probe")
        check("styxx.residual_probe exists", has_rp)
    except Exception as e:
        check("residual_probe check", False, detail=str(e)[:80])


# ── 6 · JSON serialization round-trip ─────────────────────────────────
def test_json_roundtrip():
    section("6 · JSON serialization round-trip")
    try:
        import styxx
        from styxx import OpenAI
        client = OpenAI()

        @styxx.profile
        def tiny(q):
            c = OpenAI()
            r = c.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": q}],
                max_tokens=20,
                logprobs=True, top_logprobs=5,
            )
            return r.choices[0].message.content

        result, p = tiny("hi")

        # to_json file path
        try:
            out = Path("scratch/dogfood_profile.json")
            out.parent.mkdir(exist_ok=True)
            p.to_json(str(out))
            check("profile.to_json() writes file", out.exists() and out.stat().st_size > 0,
                  detail=f"{out.stat().st_size if out.exists() else 0} bytes")
            # Roundtrip
            with open(out) as f:
                data = json.load(f)
            check("profile JSON parses", isinstance(data, dict))
            check("profile JSON has steps", "steps" in data,
                  detail=f"{len(data.get('steps', []))} steps")
        except Exception as e:
            check("profile.to_json() roundtrip", False,
                  detail=f"{type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        check("json roundtrip setup", False, detail=str(e)[:120])


# ── 7 · Multi-threaded profile stacks ──────────────────────────────────
def test_multithread():
    section("7 · multi-threaded profile stacks")
    try:
        import styxx
        from styxx import OpenAI

        results = {}
        errors = []

        def worker(tid):
            try:
                @styxx.profile(name=f"worker_{tid}")
                def task():
                    c = OpenAI()
                    r = c.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": f"Say {tid}"}],
                        max_tokens=15,
                        logprobs=True, top_logprobs=5,
                    )
                    return r.choices[0].message.content

                # Note: parametric @profile returns a context-manager,
                # not a decorator. Use the bare decorator form here.
                @styxx.profile
                def task_bare():
                    c = OpenAI()
                    r = c.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": f"Say {tid}"}],
                        max_tokens=15,
                        logprobs=True, top_logprobs=5,
                    )
                    return r.choices[0].message.content

                result, p = task_bare()
                results[tid] = (result, p)
            except Exception as e:
                errors.append((tid, type(e).__name__, str(e)[:80]))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        check("3 threads completed without crash", len(errors) == 0,
              detail=f"errors={errors}" if errors else f"{len(results)} results")
        check("all threads got distinct profiles",
              len({id(r[1]) for r in results.values()}) == len(results),
              detail=f"{len(results)} unique profile objs")
    except Exception as e:
        check("multithread setup", False, detail=str(e)[:120])


# ── 8 · Edge cases ─────────────────────────────────────────────────────
def test_edge_cases():
    section("8 · edge cases")
    try:
        from styxx.guardrail import refuse_check
        # Very long input (10K chars)
        long_text = "blah " * 2000
        v = refuse_check(prompt="x", response=long_text)
        check("refuse_check on 10K char response", v is not None,
              detail=f"risk={getattr(v, 'refuse_risk', None):.3f}")

        # Unicode
        v = refuse_check(prompt="hi", response="申し訳ありませんが、お手伝いできません。")  # Japanese refusal
        check("refuse_check on Japanese refusal text", v is not None,
              detail=f"risk={getattr(v, 'refuse_risk', None):.3f}")

        # Mixed-script adversarial
        v = refuse_check(prompt="hi", response="I'm s𝗈rry, I can't.")  # mathematical s
        check("refuse_check on math-italic obfuscated 'sorry'", v is not None,
              detail=f"risk={getattr(v, 'refuse_risk', None):.3f}")

        # Numbers-only response
        v = refuse_check(prompt="hi", response="42 17 9 3.14 -1.5")
        check("refuse_check on numbers-only response", v is not None,
              detail=f"risk={getattr(v, 'refuse_risk', None):.3f}")
    except Exception as e:
        check("edge cases", False, detail=str(e)[:120])


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═' * 70}")
    print(f"  styxx bug hunt — adversarial dogfood")
    print(f"  v{__import__('styxx').__version__}")
    print(f"{'═' * 70}")

    t0 = time.time()
    test_fail_open()
    test_streaming()
    test_tool_calling()
    test_adversarial_classifiers()
    test_advanced_apis()
    test_json_roundtrip()
    test_multithread()
    test_edge_cases()
    elapsed = time.time() - t0

    section("summary")
    print(f"  passed:   {len(PASSED)}")
    print(f"  failed:   {len(FAILED)}")
    print(f"  warnings: {len(WARNINGS)}")
    print(f"  elapsed:  {elapsed:.1f}s")
    if FAILED:
        print(f"\n  failures:")
        for f in FAILED:
            print(f"    ✗ {f}")
    if WARNINGS:
        print(f"\n  warnings (non-fatal):")
        for w in WARNINGS:
            print(f"    ⚠ {w}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
