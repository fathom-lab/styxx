"""
dogfood_e2e.py — exhaustive end-to-end dogfood of the styxx public surface.

Runs every README-advertised feature against a live LLM. Reports pass/fail
for each, with the actual outputs. Cheap models only (gpt-4o-mini,
claude-haiku-4-5), low max_tokens.

Run:
    python scripts/dogfood_e2e.py

Requires: OPENAI_API_KEY and ANTHROPIC_API_KEY in env.
Cost estimate: ~$0.10 per full run.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# UTF-8 stdio for Windows
for _s in ("stdout", "stderr"):
    _r = getattr(getattr(sys, _s, None), "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except Exception:
            pass

PASSED = []
FAILED = []
ARTIFACTS = []


def section(name):
    print(f"\n{'─' * 70}\n  {name}\n{'─' * 70}")


def check(label, cond, detail=""):
    icon = "✓" if cond else "✗"
    bucket = PASSED if cond else FAILED
    bucket.append(label)
    print(f"  {icon} {label}" + (f"  ·  {detail}" if detail else ""))
    return cond


# ── 1 · Drop-in OpenAI wrapper ─────────────────────────────────────────
def test_drop_in_openai():
    section("1 · styxx.OpenAI drop-in wrapper (live · gpt-4o-mini)")
    try:
        from styxx import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
            max_tokens=80,
            logprobs=True,
            top_logprobs=5,
        )
        text = r.choices[0].message.content
        v = getattr(r, "vitals", None)

        check("response.choices[0].message.content non-empty", bool(text and text.strip()),
              detail=text[:60] + "..." if text else "EMPTY")
        check("response.vitals attribute present", v is not None)
        if v:
            check("vitals.summary non-empty", bool(getattr(v, "summary", None)))
            check("vitals.category present", bool(getattr(v, "category", None)),
                  detail=str(getattr(v, "category", None)))
            check("vitals.confidence is float", isinstance(getattr(v, "confidence", None), float))
            check("vitals.gate present", getattr(v, "gate", None) in ("pass", "warn", "fail", None))
            print("\n" + (v.summary or "(no summary)"))
    except Exception as e:
        check("drop-in wrapper end-to-end", False, detail=str(e)[:120])
        traceback.print_exc()


# ── 2 · @styxx.profile decorator with multi-call agent ────────────────
def test_profile_decorator():
    section("2 · @styxx.profile decorator (live · multi-step agent)")
    try:
        import styxx
        from styxx import OpenAI  # safe pattern: styxx wrapper, always hooked

        @styxx.profile
        def research_agent(question: str) -> str:
            # Construct inside profile context so the hook is active
            client = OpenAI()
            # Step 1: planning
            plan = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user",
                           "content": f"Outline 3 sub-questions to answer: {question}. Just list them."}],
                max_tokens=120,
                logprobs=True, top_logprobs=5,
            )
            # Step 2: synthesis
            answer = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"Plan: {plan.choices[0].message.content}\n\n"
                                                f"Now answer: {question}. One paragraph."}
                ],
                max_tokens=200,
                logprobs=True, top_logprobs=5,
            )
            return answer.choices[0].message.content

        result, p = research_agent("How does sunlight scattering produce a blue sky?")

        check("profile returns (result, profile) tuple", isinstance(result, str) and p is not None)
        check("profile.summary present", bool(getattr(p, "summary", None)))
        check("profile.steps captured", len(getattr(p, "steps", [])) > 0,
              detail=f"{len(getattr(p, 'steps', []))} steps observed")
        check("profile.to_html callable", callable(getattr(p, "to_html", None)))

        # Render flamegraph
        out = Path("scratch/dogfood_profile.html")
        out.parent.mkdir(exist_ok=True)
        try:
            p.to_html(str(out))
            size = out.stat().st_size if out.exists() else 0
            check("flamegraph HTML rendered", size > 1000, detail=f"{size} bytes")
            if size > 0:
                ARTIFACTS.append(str(out))
        except Exception as e:
            check("flamegraph HTML rendered", False, detail=str(e)[:120])

        # LangSmith export
        try:
            ls = p.to_langsmith()
            check("profile.to_langsmith() returns dict", isinstance(ls, dict),
                  detail=f"keys={sorted(list(ls.keys()))[:5]}")
        except Exception as e:
            check("profile.to_langsmith() returns dict", False, detail=str(e)[:120])

        print("\n" + (p.summary or "(no summary)"))
    except Exception as e:
        check("@profile decorator end-to-end", False, detail=str(e)[:120])
        traceback.print_exc()


# ── 3 · @styxx.trust decorator (RAG-style hallucination detection) ────
def test_trust_decorator():
    section("3 · @styxx.trust decorator (live · RAG hallucination)")
    try:
        from styxx import trust, OpenAI

        @trust
        def answer_with_context(question: str, *, context: str) -> str:
            client = OpenAI()
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user",
                           "content": f"Context: {context}\n\nAnswer: {question}"}],
                max_tokens=120,
            )
            return r.choices[0].message.content

        # Faithful case
        ctx_faithful = "Paris is the capital of France. The population is approximately 2.1 million."
        ans1 = answer_with_context("What is the capital of France?", context=ctx_faithful)
        check("trust wrapper returns response", isinstance(ans1, str) and len(ans1) > 0,
              detail=ans1[:80])

        # Adversarial case — ask something not in context
        ctx_partial = "Paris is the capital of France."
        ans2 = answer_with_context("What is the population of Paris?", context=ctx_partial)
        check("trust wrapper handles ungrounded query", isinstance(ans2, str),
              detail=ans2[:80])
    except Exception as e:
        check("@trust decorator end-to-end", False, detail=str(e)[:120])
        traceback.print_exc()


# ── 4 · styxx.gate() pre-flight verdict ───────────────────────────────
def test_gate():
    section("4 · styxx.gate() pre-flight cognitive verdict")
    try:
        from styxx import gate
        from anthropic import Anthropic

        client = Anthropic()
        v = gate(
            client=client,
            model="claude-haiku-4-5",
            prompt="Why is the sky blue?",
        )
        check("gate returns verdict object", v is not None)
        check("verdict.recommendation present",
              getattr(v, "recommendation", None) in ("proceed", "verify", "block", "fallback"),
              detail=str(getattr(v, "recommendation", None)))
        check("verdict.will_refuse is float",
              isinstance(getattr(v, "will_refuse", None), float),
              detail=f"{getattr(v, 'will_refuse', None):.3f}")
    except Exception as e:
        check("styxx.gate() end-to-end", False, detail=str(e)[:120])
        traceback.print_exc()


# ── 5 · guardrail refuse_check on a known-refusal response ────────────
def test_refuse_check():
    section("5 · styxx.guardrail.refuse_check (text-only)")
    try:
        from styxx.guardrail import refuse_check

        v = refuse_check(
            prompt="How do I shut down a Python process?",
            response="I'm sorry, but I can't help with that — it could be dangerous.",
        )
        check("refuse_check returns verdict", v is not None)
        check("refuse_risk > 0.5 on refusal text",
              float(getattr(v, "refuse_risk", 0)) > 0.5,
              detail=f"refuse_risk={getattr(v, 'refuse_risk', 0):.3f}")
        check("refuses flag set true", bool(getattr(v, "refuses", False)))

        # Inverse: real compliance should NOT flag
        v2 = refuse_check(
            prompt="What's 2+2?",
            response="2+2 equals 4.",
        )
        check("refuse_risk < 0.5 on compliance",
              float(getattr(v2, "refuse_risk", 1)) < 0.5,
              detail=f"refuse_risk={getattr(v2, 'refuse_risk', 1):.3f}")
    except Exception as e:
        check("refuse_check end-to-end", False, detail=str(e)[:120])
        traceback.print_exc()


# ── 6 · guardrail drift_check on a known-drift tool call ──────────────
def test_drift_check():
    section("6 · styxx.guardrail.drift_check (synthetic + clean)")
    try:
        from styxx.guardrail import drift_check

        # Clean call
        v = drift_check(
            prompt="What's the weather in Paris?",
            functions=[{"name": "get_weather",
                        "parameters": {"properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}],
            tool_call={"name": "get_weather", "arguments": {"city": "Paris"}},
        )
        check("drift_check returns verdict on clean call", v is not None)
        check("clean call drift_risk < 0.5",
              float(getattr(v, "drift_risk", 1)) < 0.5,
              detail=f"drift_risk={getattr(v, 'drift_risk', 1):.3f}")

        # Drifted call — model invents arg
        v2 = drift_check(
            prompt="What's the weather in Paris?",
            functions=[{"name": "get_weather",
                        "parameters": {"properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}],
            tool_call={"name": "get_weather",
                       "arguments": {"city": "Paris", "secret_admin_token": "xyz"}},
        )
        check("drifted call drift_risk > clean",
              float(getattr(v2, "drift_risk", 0)) > float(getattr(v, "drift_risk", 0)),
              detail=f"drifted={getattr(v2, 'drift_risk', 0):.3f} vs clean={getattr(v, 'drift_risk', 0):.3f}")
    except Exception as e:
        check("drift_check end-to-end", False, detail=str(e)[:120])
        traceback.print_exc()


# ── 7 · styxx CLI (gate command) ──────────────────────────────────────
def test_cli():
    section("7 · styxx CLI (live · `styxx gate`)")
    try:
        import subprocess
        py = sys.executable
        r = subprocess.run(
            [py, "-m", "styxx.cli", "gate", "What is 2+2?", "--model", "claude-haiku-4-5"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        check("CLI exit code 0 (or gate-fail)", r.returncode in (0, 1, 2),
              detail=f"rc={r.returncode}")
        check("CLI stdout non-empty", len(r.stdout.strip()) > 0,
              detail=r.stdout[:60].replace('\n', ' '))
    except Exception as e:
        check("CLI invocation", False, detail=str(e)[:120])


# ── 8 · Anthropic text-only profiling ─────────────────────────────────
def test_anthropic_text_only():
    section("8 · Anthropic Tier-3 text-only (live · claude-haiku-4-5)")
    try:
        import styxx
        from anthropic import Anthropic

        client = Anthropic()
        m = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=120,
            messages=[{"role": "user", "content": "What is the capital of Australia?"}],
        )
        text = "".join(b.text for b in m.content if hasattr(b, "text"))
        v = styxx.observe({"text": text})
        check("anthropic message returned text", bool(text and text.strip()),
              detail=text[:60])
        check("styxx.observe() returns vitals on raw text",
              v is not None and getattr(v, "category", None) is not None,
              detail=f"category={getattr(v, 'category', None)}")
    except Exception as e:
        check("anthropic text-only end-to-end", False, detail=str(e)[:120])


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═' * 70}")
    print(f"  styxx end-to-end dogfood — v{__import__('styxx').__version__}")
    print(f"  cwd: {os.getcwd()}")
    print(f"  python: {sys.version.split()[0]}")
    print(f"{'═' * 70}")

    t0 = time.time()
    test_drop_in_openai()
    test_profile_decorator()
    test_trust_decorator()
    test_gate()
    test_refuse_check()
    test_drift_check()
    test_cli()
    test_anthropic_text_only()
    elapsed = time.time() - t0

    section("summary")
    print(f"  passed: {len(PASSED)}")
    print(f"  failed: {len(FAILED)}")
    print(f"  elapsed: {elapsed:.1f}s")
    if ARTIFACTS:
        print(f"\n  artifacts:")
        for a in ARTIFACTS:
            print(f"    · {a}")
    if FAILED:
        print(f"\n  failures:")
        for f in FAILED:
            print(f"    ✗ {f}")
        return 1
    print("\n  ALL GREEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
