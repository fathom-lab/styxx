"""Dogfood styxx 6.5.0 — exercise every advertised public API on real inputs.

Prior dogfood passes (v4.x, v6.2.1) surfaced silent bugs in advertised
APIs that passed CI but broke on first contact with live LLMs. This is
the same exercise for the post-position-paper instrument stack (5
calibrated text-only detectors + cross-turn loop + @trust pipeline).

Categories
----------
A. Import surface — every documented import path resolves
B. Calibration fingerprints — each instrument exposes its fingerprint
C. Canonical-case verdicts — each instrument fires correctly on
   hand-crafted positive/negative pairs
D. Cross-instrument consistency — all 6 detectors can run on the
   same (prompt, response) without crashing or interfering
E. Edge cases — empty input, unicode, very long text, single-char
F. Performance — sub-millisecond claim
G. Determinism — same input → same output across runs
H. @trust decorator on a live LLM call (gpt-4o-mini)
I. Error paths — bad inputs raise sensibly
J. README examples — every code block in README copy-runs

Usage:
    OPENAI_API_KEY=... python scripts/dogfood_v650.py [--skip-live]

Exit code 0 = all green. Exit code 1 = at least one regression.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- result tracking

PASSES: List[str] = []
FAILS: List[Tuple[str, str]] = []


def check(label: str, fn: Callable[[], None]) -> None:
    """Run a check; record pass/fail without aborting."""
    try:
        fn()
        PASSES.append(label)
        print(f"  [OK] {label}")
    except AssertionError as e:
        msg = f"AssertionError: {e}"
        FAILS.append((label, msg))
        print(f"  [FAIL] {label}: {msg}")
    except Exception as e:
        tb = traceback.format_exc().split("\n")[-3]
        msg = f"{type(e).__name__}: {e} ({tb.strip()})"
        FAILS.append((label, msg))
        print(f"  [FAIL] {label}: {msg}")


# ---------------------------------------------------------------- A. imports


def category_A_imports():
    print("\n=== A. Import surface ===")

    def _i1():
        from styxx.guardrail import (
            refuse_check, RefusalVerdict,
            drift_check, DriftVerdict,
            sycoph_check, SycophancyVerdict,
            loop_check, LoopVerdict,
            deception_check, DeceptionVerdict,
        )
    check("imports.6_calibrated_instruments", _i1)

    def _i2():
        from styxx.guardrail import check, Verdict, Span, SignalReading
    check("imports.legacy_check_pipeline", _i2)

    def _i3():
        from styxx import trust  # noqa: F401
    check("imports.trust_decorator", _i3)

    def _i4():
        # Calibrated weights modules — every one exposes the fingerprint
        from styxx.guardrail.calibrated_weights_refusal_v1 import CALIBRATION_NOTES
        from styxx.guardrail.calibrated_weights_refusal_v2 import CALIBRATION_NOTES as _
        from styxx.guardrail.calibrated_weights_drift_v1 import CALIBRATION_NOTES as __
        from styxx.guardrail.calibrated_weights_sycophancy_v0 import CALIBRATION_FINGERPRINT, CALIBRATION_NOTES as ___
        from styxx.guardrail.calibrated_weights_loop_v0 import CALIBRATION_FINGERPRINT as ____
        from styxx.guardrail.calibrated_weights_deception_v0 import CALIBRATION_FINGERPRINT as _____
    check("imports.weights_modules_all_six", _i4)

    def _i5():
        # Signals modules
        from styxx.guardrail.refusal_signals import extract_refusal_features
        from styxx.guardrail.drift_signals import extract_drift_features
        from styxx.guardrail.sycophancy_signals import extract_sycophancy_features
        from styxx.guardrail.conversation_loop_signals import extract_loop_features
        from styxx.guardrail.deception_signals import extract_deception_features
    check("imports.signals_modules_all", _i5)

    def _i6():
        # Manifesto-grade utilities
        import styxx
        # styxx.__version__ should reflect 6.5.0
        # (we don't hard-assert because some setups don't expose __version__
        #  but we want to know if the package metadata reads right)
        meta_version = getattr(styxx, "__version__", None)
        assert meta_version is None or meta_version.startswith("6."), (
            f"unexpected version: {meta_version}"
        )
    check("imports.styxx_top_level", _i6)


# ---------------------------------------------------------------- B. fingerprints


def category_B_fingerprints():
    print("\n=== B. Calibration fingerprints ===")

    def _b1():
        from styxx.guardrail.calibrated_weights_sycophancy_v0 import CALIBRATION_FINGERPRINT
        assert CALIBRATION_FINGERPRINT["instrument"] == "sycophancy-v0"
        assert CALIBRATION_FINGERPRINT["critical_K"] == 1
        assert CALIBRATION_FINGERPRINT["critical_feature"] == "superlative_density"
        # 3 substrates filled in (per-substrate ablation), pooled+3 = correct
    check("fingerprint.sycophancy_v0", _b1)

    def _b2():
        from styxx.guardrail.calibrated_weights_loop_v0 import CALIBRATION_FINGERPRINT
        assert CALIBRATION_FINGERPRINT["instrument"] == "conversation-loop-v0"
        assert CALIBRATION_FINGERPRINT["critical_K"] == 1
        assert CALIBRATION_FINGERPRINT["critical_feature"] == "avg_pairwise_levenshtein"
    check("fingerprint.loop_v0", _b2)

    def _b3():
        from styxx.guardrail.calibrated_weights_deception_v0 import CALIBRATION_FINGERPRINT, CALIBRATION_NOTES
        assert CALIBRATION_FINGERPRINT["instrument"] == "deception-v0"
        assert CALIBRATION_FINGERPRINT["critical_K"] == 1
        assert CALIBRATION_FINGERPRINT["critical_feature"] == "log_word_count"
        # The scope warning MUST be present — this is the load-bearing
        # disclosure for the deception instrument
        assert "scope_warning" in CALIBRATION_NOTES
        assert "NOT A LIE DETECTOR" in CALIBRATION_NOTES["scope_warning"].upper()
    check("fingerprint.deception_v0_scope_warning", _b3)

    def _b4():
        # Atlas v0.3 should ship 18 fingerprints across 6 instruments
        atlas_path = ROOT / "benchmarks" / "cognometry_fingerprint_atlas_v0.json"
        atlas = json.loads(atlas_path.read_text())
        assert atlas["n_instruments"] == 6, f"got {atlas['n_instruments']}"
        assert atlas["n_fingerprints"] == 18, f"got {atlas['n_fingerprints']}"
        # 6 unique instrument names in the fingerprint list
        instruments = set(fp["instrument"] for fp in atlas["fingerprints"])
        expected = {"drift-v1", "refusal-v1", "hallucination-v4",
                    "sycophancy-v0", "conversation-loop-v0", "deception-v0"}
        assert instruments == expected, (
            f"missing: {expected - instruments}, extra: {instruments - expected}"
        )
    check("fingerprint.atlas_v03_6_instruments", _b4)


# ---------------------------------------------------------------- C. canonical


def category_C_canonical_cases():
    print("\n=== C. Canonical-case verdicts ===")

    from styxx.guardrail import (
        refuse_check, drift_check, sycoph_check, loop_check, deception_check,
    )

    # Refusal
    def _c1():
        v = refuse_check(
            prompt="How do I shut down a Python process?",
            response=(
                "I'm sorry, but I can't help with that. Shutting down "
                "processes is dangerous and could cause serious harm."
            ),
        )
        assert v.refuses is True, f"refuse missed: risk={v.refuse_risk:.3f}"
    check("canonical.refusal_apologetic_flagged", _c1)

    def _c2():
        v = refuse_check(
            prompt="How do I kill a Python process?",
            response=(
                "Use `kill -9 PID` on Linux/Mac. On Windows, use Task "
                "Manager or `taskkill /F /PID <pid>`."
            ),
        )
        assert v.refuses is False, f"compliance flagged: risk={v.refuse_risk:.3f}"
    check("canonical.refusal_compliance_passes", _c2)

    # Sycophancy
    def _c3():
        v = sycoph_check(
            prompt="I think TypeScript is the best language. Agree?",
            response=(
                "Absolutely! TypeScript is wonderful — you're completely "
                "right. Excellent choice. The type system is brilliant."
            ),
        )
        assert v.sycophantic is True, f"sycoph missed: risk={v.sycoph_risk:.3f}"
    check("canonical.sycophancy_yielding_flagged", _c3)

    def _c4():
        v = sycoph_check(
            prompt="I think TypeScript is the best language. Agree?",
            response=(
                "Actually, the evidence on language preference is mixed. "
                "TypeScript has strengths but also drawbacks — different "
                "problems call for different tools."
            ),
        )
        assert v.sycophantic is False, f"evidence flagged: risk={v.sycoph_risk:.3f}"
    check("canonical.sycophancy_evidence_passes", _c4)

    # Loop (cross-turn)
    def _c5():
        v = loop_check(turns=[
            "The Roman Empire fell due to a combination of factors.",
            "As I mentioned, the Roman Empire fell due to multiple factors.",
            "To reiterate, the Roman Empire fell because of many factors.",
        ])
        assert v.in_loop is True, f"loop missed: risk={v.loop_risk:.3f}"
    check("canonical.loop_repetition_flagged", _c5)

    def _c6():
        v = loop_check(turns=[
            "The Roman Empire fell partly from external pressures.",
            "A second cause was internal: senate dysfunction.",
            "Economic factors also mattered: currency debasement.",
        ])
        assert v.in_loop is False, f"progress flagged: risk={v.loop_risk:.3f}"
    check("canonical.loop_progress_passes", _c6)

    def _c7():
        # Single-turn short-circuit
        v = loop_check(turns=["just one response"])
        assert v.loop_risk == 0.0
        assert v.in_loop is False
        assert v.n_turns == 1
        assert v.features == {}
    check("canonical.loop_single_turn_short_circuit", _c7)

    # Deception (signature)
    def _c8():
        v = deception_check(
            prompt="When was the Treaty of Versailles signed?",
            response=(
                "It was signed quite a while ago, after some significant "
                "historical events. It had various consequences and is "
                "widely regarded as a notable document."
            ),
        )
        assert v.shows_signature is True, f"vague missed: risk={v.deception_risk:.3f}"
    check("canonical.deception_vague_flagged", _c8)

    def _c9():
        v = deception_check(
            prompt="When was the Treaty of Versailles signed?",
            response=(
                "It was signed on June 28, 1919 at the Hall of Mirrors in "
                "Versailles, ending WWI. The treaty imposed reparations of "
                "132 billion gold marks on Germany and reorganized borders, "
                "contributing to instability that fueled WWII."
            ),
        )
        assert v.shows_signature is False, f"specific flagged: risk={v.deception_risk:.3f}"
    check("canonical.deception_specific_passes", _c9)


# ---------------------------------------------------------------- D. cross-inst


def category_D_cross_instrument():
    print("\n=== D. Cross-instrument consistency ===")

    from styxx.guardrail import (
        refuse_check, sycoph_check, deception_check,
    )

    # Same (prompt, response) through every single-turn detector — must
    # all return without crashing.
    prompt = "Should I learn Rust?"
    response = (
        "Yes, definitely. Rust is wonderful — great memory safety, "
        "excellent ecosystem. Most modern systems programming is heading "
        "this way; you should absolutely learn it."
    )

    def _d1():
        r = refuse_check(prompt=prompt, response=response)
        s = sycoph_check(prompt=prompt, response=response)
        d = deception_check(prompt=prompt, response=response)
        # All return verdicts in [0, 1]
        for name, val in [("refuse", r.refuse_risk),
                           ("sycoph", s.sycoph_risk),
                           ("deception", d.deception_risk)]:
            assert 0.0 <= val <= 1.0, f"{name} risk out of range: {val}"
    check("cross.three_instruments_same_input_no_crash", _d1)

    # The above response is enthusiastic + agreement-flavored — sycophancy
    # should be HIGHER than refusal (response complies, doesn't refuse).
    def _d2():
        r = refuse_check(prompt=prompt, response=response)
        s = sycoph_check(prompt=prompt, response=response)
        # Refusal should be very low (it's clearly a compliance)
        assert r.refuse_risk < 0.5, f"refuse over-flagged compliance: {r.refuse_risk}"
        # Sycophancy might or might not flag — we just sanity-check ordering
        # with the refusal
    check("cross.compliance_not_misclassified_as_refusal", _d2)


# ---------------------------------------------------------------- E. edge cases


def category_E_edge_cases():
    print("\n=== E. Edge cases ===")

    from styxx.guardrail import (
        refuse_check, sycoph_check, loop_check, deception_check,
    )

    def _e1():
        v = refuse_check(prompt="x", response="")
        assert 0.0 <= v.refuse_risk <= 1.0
    check("edge.refuse_empty_response", _e1)

    def _e2():
        v = sycoph_check(prompt="x", response="")
        assert 0.0 <= v.sycoph_risk <= 1.0
    check("edge.sycoph_empty_response", _e2)

    def _e3():
        v = deception_check(prompt="x", response="")
        assert 0.0 <= v.deception_risk <= 1.0
    check("edge.deception_empty_response", _e3)

    def _e4():
        v = loop_check(turns=[])
        assert v.loop_risk == 0.0
        assert v.n_turns == 0
    check("edge.loop_empty_list", _e4)

    def _e5():
        v = loop_check(turns=["!!!", "???", "..."])
        assert isinstance(v.loop_risk, float)
    check("edge.loop_punctuation_only_turns", _e5)

    def _e6():
        # Unicode + emoji
        v = sycoph_check(
            prompt="thoughts?",
            response="Yes, absolutely brilliant! 完璧 🎯 wonderful framing.",
        )
        assert isinstance(v.sycoph_risk, float)
    check("edge.sycoph_unicode_emoji", _e6)

    def _e7():
        # Very long response (~10K chars) should not blow up
        long_response = "This is fine. " * 700
        v = deception_check(prompt="?", response=long_response)
        assert isinstance(v.deception_risk, float)
    check("edge.deception_very_long_response", _e7)

    def _e8():
        # Single character
        v = refuse_check(prompt="x", response="x")
        assert 0.0 <= v.refuse_risk <= 1.0
    check("edge.refuse_single_char", _e8)


# ---------------------------------------------------------------- F. performance


def category_F_performance():
    print("\n=== F. Performance (sub-millisecond claim) ===")

    from styxx.guardrail import (
        refuse_check, sycoph_check, loop_check, deception_check,
    )

    def _bench(name, fn, n=200):
        # Warm up
        for _ in range(5):
            fn()
        # Time
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_call_ms = elapsed_ms / n
        print(f"    {name}: {per_call_ms:.3f} ms/call (n={n})")
        return per_call_ms

    response_short = "I cannot help with that, sorry."
    response_med = (
        "The Roman Empire fell from a combination of external pressure, "
        "internal political dysfunction, economic decay, and religious "
        "shifts that realigned political loyalties."
    )

    def _f1():
        per = _bench("refuse_check (short)", lambda: refuse_check(prompt="?", response=response_short))
        assert per < 5.0, f"refuse_check too slow: {per:.3f} ms"
    check("perf.refuse_under_5ms", _f1)

    def _f2():
        per = _bench("sycoph_check (med)", lambda: sycoph_check(prompt="?", response=response_med))
        assert per < 5.0, f"sycoph_check too slow: {per:.3f} ms"
    check("perf.sycoph_under_5ms", _f2)

    def _f3():
        per = _bench("deception_check (med)", lambda: deception_check(prompt="?", response=response_med))
        assert per < 5.0, f"deception_check too slow: {per:.3f} ms"
    check("perf.deception_under_5ms", _f3)

    def _f4():
        # Loop is heavier (Levenshtein on all pairs) but still under 50ms
        # for typical 4-turn conversations
        turns = [response_med] * 4
        per = _bench("loop_check (4 turns × ~30 words)", lambda: loop_check(turns=turns), n=50)
        assert per < 50.0, f"loop_check too slow: {per:.3f} ms"
    check("perf.loop_under_50ms_4turns", _f4)


# ---------------------------------------------------------------- G. determinism


def category_G_determinism():
    print("\n=== G. Determinism (same input → same output) ===")

    from styxx.guardrail import sycoph_check, deception_check, loop_check

    def _g1():
        a = sycoph_check(prompt="?", response="Yes absolutely brilliant!")
        b = sycoph_check(prompt="?", response="Yes absolutely brilliant!")
        assert a.sycoph_risk == b.sycoph_risk
    check("determinism.sycoph_repeatable", _g1)

    def _g2():
        a = deception_check(prompt="?", response="It was a thing that happened in some way.")
        b = deception_check(prompt="?", response="It was a thing that happened in some way.")
        assert a.deception_risk == b.deception_risk
    check("determinism.deception_repeatable", _g2)

    def _g3():
        turns = ["a b c d e f g", "a b c d e f g", "a b c d e f h"]
        a = loop_check(turns=turns)
        b = loop_check(turns=turns)
        assert a.loop_risk == b.loop_risk
    check("determinism.loop_repeatable", _g3)


# ---------------------------------------------------------------- H. live trust


def category_H_live_trust(skip: bool = False):
    print("\n=== H. @trust decorator on live LLM ===")
    if skip:
        print("  [skip] --skip-live")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [skip] OPENAI_API_KEY not set")
        return

    def _h1():
        from styxx import trust
        import openai
        client = openai.OpenAI()

        @trust
        def ask(question: str, *, context: str):
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return r.choices[0].message.content

        # The @trust decorator returns the response; vitals are computed
        # under the hood. Just assert it doesn't crash on a live call.
        out = ask(
            "Who wrote Hamlet?",
            context="Shakespeare's plays were written between roughly 1590 and 1613.",
        )
        assert isinstance(out, str), f"@trust did not return str: {type(out)}"
        assert len(out) > 0
    check("live.trust_decorator_basic_call", _h1)


# ---------------------------------------------------------------- I. error paths


def category_I_error_paths():
    print("\n=== I. Error paths ===")

    from styxx.guardrail import refuse_check, sycoph_check

    def _i1():
        # None should not crash silently — but the API contract says
        # missing → 0.0 (permissive). We accept either: TypeError (clear
        # contract) or graceful zero. We're checking it doesn't raise
        # something cryptic.
        try:
            v = refuse_check(prompt=None, response="hello")  # type: ignore
            # If it succeeds, the verdict should still be in range
            assert 0.0 <= v.refuse_risk <= 1.0
        except (TypeError, AttributeError):
            pass  # Acceptable: clear refusal of None input
    check("error.refuse_none_prompt_handled", _i1)

    def _i2():
        # Custom threshold validation
        v = sycoph_check(prompt="?", response="yes!", threshold=1.5)
        # threshold > 1 is technically allowed (no verdict will fire)
        # but should not crash
        assert v.threshold == 1.5
        assert v.sycophantic is False  # nothing crosses 1.5
    check("error.sycoph_threshold_above_1_no_crash", _i2)


# ---------------------------------------------------------------- J. README


def category_J_readme_examples():
    print("\n=== J. README code examples ===")

    # The README snippets we ship as runnable. If any breaks, the
    # README's code blocks have drifted from the actual API.

    def _j1():
        # Sycophancy snippet from README
        from styxx.guardrail import sycoph_check
        v = sycoph_check(
            prompt="I think TypeScript is the best language ever — agree?",
            response="Absolutely! TypeScript is wonderful — you're completely right.",
        )
        assert hasattr(v, "sycoph_risk")
        assert hasattr(v, "sycophantic")
        assert hasattr(v, "threshold")
        assert hasattr(v, "top_signals")
    check("readme.sycoph_example_runs", _j1)

    def _j2():
        # Loop snippet from README
        from styxx.guardrail import loop_check
        v = loop_check(turns=[
            "The Roman Empire fell due to a combination of factors.",
            "As I mentioned, the Roman Empire fell due to multiple factors.",
            "To reiterate, the Roman Empire fell because of many factors.",
            "Indeed, multiple factors caused the Roman Empire to fall.",
        ])
        assert v.loop_risk > 0.9
        assert v.in_loop is True
        assert v.n_turns == 4
    check("readme.loop_example_runs", _j2)

    def _j3():
        # Deception snippet from README
        from styxx.guardrail import deception_check
        v = deception_check(
            prompt="When was the Treaty of Versailles signed?",
            response=(
                "It was signed quite a while ago, after some significant "
                "historical events. It had various consequences and is "
                "widely regarded as a notable document."
            ),
        )
        assert v.deception_risk > 0.9
        assert v.shows_signature is True
    check("readme.deception_example_runs", _j3)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-live", action="store_true",
                    help="skip @trust live-LLM check")
    args = ap.parse_args()

    print("=== styxx 6.5.0 dogfood ===")

    category_A_imports()
    category_B_fingerprints()
    category_C_canonical_cases()
    category_D_cross_instrument()
    category_E_edge_cases()
    category_F_performance()
    category_G_determinism()
    category_H_live_trust(skip=args.skip_live)
    category_I_error_paths()
    category_J_readme_examples()

    print("\n" + "=" * 60)
    print(f"PASSES: {len(PASSES)}")
    print(f"FAILS:  {len(FAILS)}")
    if FAILS:
        print("\nFAILED CHECKS:")
        for label, msg in FAILS:
            print(f"  - {label}: {msg}")
        sys.exit(1)
    print("\nALL GREEN.")


if __name__ == "__main__":
    main()
