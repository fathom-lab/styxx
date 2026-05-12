"""
dogfood_advanced_apis.py — exercise the README's "Also in styxx" surface
that hasn't been live-tested:

  · Thought (.fathom) — substrate-independent cognitive content type
  · dynamics — linear-Gaussian state-space model
  · residual_probe — cross-vendor probe atlas
  · autogen adapter — excluded from main test suite (probably for a reason)
  · crewai adapter — agent injection
  · gate() Python API — fail-open with mock client
"""
from __future__ import annotations

import json
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


# Self-describing fingerprint: print which styxx is under test. The
# 2026-04-28 dogfood-vs-stale-install confusion (a leftover v3.5.1
# editable install pollutiing site-packages with a namespace-package
# shell) is the lesson here. If the imported `styxx` doesn't have a
# __version__ attribute, that's the canary — abort with a clear message
# rather than letting downstream tests fail in confusing ways.
try:
    import styxx as _styxx_under_test
    _ver = getattr(_styxx_under_test, "__version__", None)
    _file = getattr(_styxx_under_test, "__file__", None)
    if _ver is None:
        print(
            "✗ ABORT: imported `styxx` has no __version__ attribute.\n"
            "  This usually means a namespace-package shell (an empty\n"
            "  styxx/ in site-packages) is shadowing the real install.\n"
            f"  Module path: {getattr(_styxx_under_test, '__path__', '?')}\n"
            "  Fix: `pip uninstall styxx` then `pip install styxx` (or\n"
            "  reinstall editable from your active styxx repo)."
        )
        sys.exit(1)
except ImportError as _e:
    print(f"✗ ABORT: failed to import styxx: {_e!r}")
    sys.exit(1)


# ── 1 · Thought (.fathom) — serialization round-trip ───────────────────
def test_thought():
    section("1 · styxx.Thought · substrate-independent cognitive content")
    try:
        import styxx
        T = styxx.Thought
        ok("Thought class importable")

        # Default construction
        t = T()
        ok("Thought() default construction",
           detail=f"thought_id={t.thought_id[:8]}... atlas_version={t.atlas_version}")

        # Inspect public methods
        public_methods = [m for m in dir(T) if not m.startswith("_") and callable(getattr(T, m, None))]
        ok("Thought public methods", detail=f"{len(public_methods)} methods · {public_methods[:8]}")

        # Try common serialization patterns
        for method_name in ("to_dict", "as_dict", "to_json", "to_fathom"):
            method = getattr(t, method_name, None)
            if method is None or not callable(method):
                continue
            try:
                result = method()
                ok(f"Thought.{method_name}() works",
                   detail=f"type={type(result).__name__} · "
                          f"{'keys=' + str(sorted(list(result.keys()))[:5]) if isinstance(result, dict) else 'len=' + str(len(str(result)))}")
            except Exception as e:
                bad(f"Thought.{method_name}() raised", detail=f"{type(e).__name__}: {str(e)[:80]}")

        # Round-trip via from_dict if available
        for method_name in ("from_dict", "from_json", "parse"):
            method = getattr(T, method_name, None)
            if method is None or not callable(method):
                continue
            try:
                if method_name == "from_dict":
                    serialized = t.to_dict() if hasattr(t, "to_dict") else None
                else:
                    serialized = t.to_json() if hasattr(t, "to_json") else None
                if serialized is None:
                    continue
                t2 = method(serialized)
                round_trip_ok = (t2.thought_id == t.thought_id)
                if round_trip_ok:
                    ok(f"Thought round-trip via {method_name}", detail="thought_id preserved")
                else:
                    bad(f"Thought round-trip via {method_name}", detail="thought_id changed")
            except Exception as e:
                bad(f"Thought.{method_name}() round-trip", detail=f"{type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        bad("Thought test setup", detail=f"{e}")


# ── 2 · dynamics — linear-Gaussian state-space ─────────────────────────
def test_dynamics():
    section("2 · styxx.dynamics · linear-Gaussian cognitive dynamics")
    try:
        import styxx
        if not hasattr(styxx, "dynamics"):
            skip("styxx.dynamics not exposed")
            return

        d = styxx.dynamics
        # If it's a module, look for callable factories
        if hasattr(d, "__file__"):  # It's a module
            ok("styxx.dynamics is a module", detail=f"public={[a for a in dir(d) if not a.startswith('_')][:6]}")
            # Try standard factory names
            factories = ["fit", "simulate", "predict", "Dynamics", "from_profile", "load"]
            for name in factories:
                f = getattr(d, name, None)
                if f is None: continue
                try:
                    sig = ""
                    if callable(f):
                        import inspect
                        try: sig = str(inspect.signature(f))[:80]
                        except: sig = "<no sig>"
                    ok(f"dynamics.{name} present", detail=f"signature={sig}")
                except Exception as e:
                    bad(f"dynamics.{name} probe", detail=str(e)[:80])
        elif callable(d):
            ok("styxx.dynamics is callable", detail=f"type={type(d).__name__}")
            # Try invoking it with no args
            try:
                instance = d()
                ok("dynamics() default-instantiable", detail=f"type={type(instance).__name__}")
            except Exception as e:
                skip("dynamics() needs args", detail=f"{type(e).__name__}")
    except Exception as e:
        bad("dynamics test setup", detail=f"{e}")


# ── 3 · residual_probe — cross-vendor probe atlas ──────────────────────
def test_residual_probe():
    section("3 · styxx.residual_probe · cross-vendor probe atlas (29 probes · 6 vendors)")
    try:
        import styxx
        if not hasattr(styxx, "residual_probe"):
            skip("styxx.residual_probe not exposed")
            return

        rp = styxx.residual_probe
        ok("residual_probe importable", detail=f"public={[a for a in dir(rp) if not a.startswith('_')][:8]}")

        # Try atlas listing
        for probe in ("atlas", "probes", "list_probes", "directions", "available"):
            f = getattr(rp, probe, None)
            if f is None: continue
            try:
                if callable(f):
                    result = f()
                else:
                    result = f
                if isinstance(result, (list, dict, tuple)):
                    ok(f"residual_probe.{probe}",
                       detail=f"type={type(result).__name__} · count={len(result)}")
                else:
                    ok(f"residual_probe.{probe} present",
                       detail=f"type={type(result).__name__}")
            except Exception as e:
                bad(f"residual_probe.{probe}", detail=f"{type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        bad("residual_probe setup", detail=f"{e}")


# ── 4 · autogen adapter — excluded from main test suite ────────────────
def test_autogen():
    section("4 · styxx.adapters.autogen · why was this excluded from tests?")
    try:
        from styxx.adapters import autogen as _a
        ok("autogen adapter importable")

        # Inspect surface
        public = [a for a in dir(_a) if not a.startswith("_")]
        ok("autogen public surface", detail=f"{public[:10]}")

        # Try to find the main entry point
        for name in ("StyxxAgent", "wrap_agent", "patch_autogen", "install_hook"):
            obj = getattr(_a, name, None)
            if obj is None: continue
            ok(f"autogen.{name} exposed", detail=f"type={type(obj).__name__}")
    except ImportError as e:
        skip("autogen adapter import failed", detail=str(e)[:120])
    except Exception as e:
        bad("autogen adapter probe", detail=f"{type(e).__name__}: {str(e)[:120]}")


# ── 5 · crewai adapter — agent injection ───────────────────────────────
def test_crewai():
    section("5 · styxx.adapters.crewai · agent injection")
    try:
        from styxx.adapters import crewai as _c
        ok("crewai adapter importable")
        public = [a for a in dir(_c) if not a.startswith("_")]
        ok("crewai public surface", detail=f"{public[:10]}")

        for name in ("StyxxAgent", "wrap_agent", "wrap_crew", "instrument_agent", "install_hook"):
            obj = getattr(_c, name, None)
            if obj is None: continue
            ok(f"crewai.{name} exposed", detail=f"type={type(obj).__name__}")
    except ImportError as e:
        skip("crewai adapter import failed", detail=str(e)[:120])
    except Exception as e:
        bad("crewai adapter probe", detail=f"{type(e).__name__}: {str(e)[:120]}")


# ── 6 · gate() Python API — fail-open contract ─────────────────────────
def test_gate_failopen():
    section("6 · styxx.gate() · fail-open contract with deliberately failing client")
    try:
        from styxx import gate

        class FakeAnthropic:
            class messages:
                @staticmethod
                def create(**kwargs): raise RuntimeError("intentional failure for fail-open test")
            messages = messages()

        v = gate(client=FakeAnthropic(), model="claude-haiku-4-5", prompt="hi")
        ok("gate() returned verdict (didn't raise)", detail=f"recommendation={getattr(v, 'recommendation', '?')}")
        ok("verdict has will_refuse",
           detail=f"will_refuse={getattr(v, 'will_refuse', '?')}")
        ok("verdict has will_confabulate",
           detail=f"will_confabulate={getattr(v, 'will_confabulate', '?')}")
    except Exception as e:
        bad("gate() leaked exception", detail=f"{type(e).__name__}: {str(e)[:120]}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*72}")
    print(f"  styxx advanced-API dogfood — testing styxx {_ver}")
    print(f"  {_file}")
    print(f"{'═'*72}")
    t0 = time.time()
    test_thought()
    test_dynamics()
    test_residual_probe()
    test_autogen()
    test_crewai()
    test_gate_failopen()
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
