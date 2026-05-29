# -*- coding: utf-8 -*-
"""
styxx.cli — the command-line entry points.

    styxx init                   live-print installer / upgrade card
    styxx ask "..."              read vitals on a one-shot call
    styxx ask --watch "..."      stream vitals live as tokens arrive
    styxx log tail               tail the audit log
    styxx tier                   show which tiers are active
    styxx scan <trajectory.json> read a pre-captured logprob trajectory

All commands are designed so stdout is both human-readable AND
agent-parseable — every card has a json summary line the agent can
grep for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, __tagline__
from .bootlog import boot
from .cards import (
    Palette,
    color_enabled,
    render_vitals_card,
    render_vitals_compact,
    wrap,
)
from .core import StyxxRuntime, detect_tiers
from .vitals import Vitals


# ══════════════════════════════════════════════════════════════════
# Audit log helper
# ══════════════════════════════════════════════════════════════════

def _audit_log_path() -> Path:
    p = Path.home() / ".styxx" / "chart.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_audit(vitals: Vitals, prompt: Optional[str], model: Optional[str],
                  source: str = "live"):
    """Append one entry to the audit log. Respects STYXX_NO_AUDIT.

    0.2.3: delegates to analytics.write_audit() which is the
    canonical write path for ALL styxx surfaces. This function is
    kept for backward compatibility with the CLI code paths that
    still call it by name.
    """
    from .analytics import write_audit
    write_audit(vitals, prompt=prompt, model=model, source=source)


# ══════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════

def cmd_claim(args):
    """Claim an agent name on the live telemetry relay.

    On success, prints the dashboard URL + next-step snippet and
    saves credentials to ~/.styxx/credentials.json.
    """
    from . import stream as _stream

    name = (args.name or "").strip()
    if not name:
        try:
            sys.stdout.write("agent name: ")
            sys.stdout.flush()
            name = sys.stdin.readline().strip()
        except (KeyboardInterrupt, EOFError):
            sys.stdout.write("\n")
            return 1

    clean = _stream._sanitize_name(name)
    if not clean:
        sys.stderr.write("  invalid name. use lowercase letters, digits, - or _.\n")
        return 1

    sys.stdout.write("\n")
    sys.stdout.write("  generating identity\n")
    sys.stdout.write("  reserving subdomain\n")
    try:
        res = _stream.claim_agent(clean, relay=args.relay)
    except _stream.ClaimError as e:
        sys.stderr.write(f"\n  claim failed: {e}\n")
        sys.stderr.write("  (the relay may be offline, or the name is taken)\n")
        return 1

    sys.stdout.write("\n")
    sys.stdout.write("  registered.\n\n")
    sys.stdout.write(f"    name         {res['name']}\n")
    sys.stdout.write(f"    feed         {res['url']}\n")
    sys.stdout.write("    credentials  ~/.styxx/credentials.json\n\n")
    sys.stdout.write("  next: enable streaming\n\n")
    sys.stdout.write("      import styxx\n")
    sys.stdout.write(f"      styxx.autoboot(agent_name=\"{res['name']}\", stream=True)\n\n")
    sys.stdout.write("  nothing crosses unseen.\n\n")
    return 0


def cmd_feed(args):
    """Print the live dashboard URL for the claimed agent."""
    from . import stream as _stream

    name = (args.name or "").strip()
    if not name:
        creds = _stream._load_creds()
        name = creds.get("_default", "")
    if not name:
        sys.stderr.write("  no agent claimed yet. run: styxx claim <name>\n")
        return 1
    sys.stdout.write(_stream.dashboard_url(name) + "\n")
    return 0


def cmd_init(args):
    """Run the live-print boot sequence (the upgrade card)."""
    speed_env = os.environ.get("STYXX_BOOT_SPEED")
    speed = float(args.speed) if args.speed is not None else (
        float(speed_env) if speed_env else 1.0
    )
    result = boot(stream=sys.stdout, speed=speed, patient=args.patient)
    if not result["boot_ok"]:
        sys.stderr.write("\n  styxx boot failed:\n")
        for err in result["errors"]:
            sys.stderr.write(f"    · {err}\n")
        return 1
    return 0


def cmd_ask(args):
    """Read vitals on a single LLM call.

    For v0.1 this supports:
      - --raw <file>  : load a pre-captured logprob JSON
      - --demo-kind X : use a real bundled atlas trajectory (one per
                        category), so the classifier reads genuine
                        data and produces honest predictions

    The CLI does NOT execute live LLM calls in v0.1. For live vitals
    on your own model calls, use the python API:

        from styxx import OpenAI
        client = OpenAI()                  # drop-in for openai.OpenAI
        r = client.chat.completions.create(...)
        print(r.vitals.summary)

    The CLI is intentionally scoped to fixture-replay + pre-captured
    trajectories so `styxx ask` never makes surprise network calls.
    """
    runtime = StyxxRuntime()

    source_label = args.model or "demo"
    is_demo_mode = not args.raw

    # 0.8.1: show demo banner FIRST to prevent deceptive UX.
    # In demo mode, ignore the user's prompt — it's not being classified.
    if is_demo_mode:
        kind = args.demo_kind or "reasoning"
        use_color = color_enabled()
        c = Palette
        print()
        print(wrap("  ⚠  DEMO MODE — replaying atlas fixture, not classifying your prompt", c.YELLOW, use_color))
        print(wrap(f"  ⚠  fixture: atlas:{kind} (gemma-2-2b-it)", c.DIM, use_color))
        print(wrap("  ⚠  for real vitals: from styxx import OpenAI", c.DIM, use_color))
        print()
        preview_prompt = f"[atlas:{kind} fixture]"
    else:
        preview_prompt = args.prompt

    if args.raw:
        entropy, logprob, top2 = _load_trajectory_json(args.raw)
    else:
        kind = args.demo_kind or "reasoning"
        entropy, logprob, top2, atlas_prompt = _get_demo_trajectory(kind)
        # If the user didn't pass a prompt, use the real atlas prompt
        if not preview_prompt:
            preview_prompt = atlas_prompt
        if args.model is None:
            source_label = f"atlas:{kind}  (gemma-2-2b-it)"

    vitals = runtime.run_on_trajectories(entropy, logprob, top2)

    is_demo_mode = not args.raw
    _write_audit(vitals, prompt=preview_prompt, model=source_label,
                 source="demo" if is_demo_mode else "live")

    # Blank-line padding around all CLI output for readability
    print()

    # Demo banner already shown above (0.8.1 — moved before output)

    if args.watch:
        card = render_vitals_card(
            vitals=vitals,
            prompt=preview_prompt,
            model=source_label,
            n_tokens=len(entropy),
            entropy_traj=entropy,
            logprob_traj=logprob,
        )
        print(card)
    else:
        # Compact one-liner with a minimal header for context
        use_color = color_enabled()
        header = wrap("  styxx · compact readout", Palette.DIM, use_color)
        print(header)
        print(render_vitals_compact(vitals, prompt=preview_prompt))
    print()
    return 0


def cmd_timeline(args):
    """Mood + category trajectory over time (0.5.5)."""
    from .timeline import timeline
    name = args.name or "styxx agent"
    hours = float(args.hours or 48.0)
    slice_h = float(getattr(args, "slice_hours", 3.0))
    sid = getattr(args, "session", None)

    tl = timeline(window_hours=hours, slice_hours=slice_h, agent_name=name, session_id=sid)
    if tl is None:
        print()
        print("  (not enough audit data for a timeline)")
        print()
        return 0

    fmt = getattr(args, "format", "ascii")
    if fmt == "json":
        print(tl.as_json())
    else:
        print(tl.render())
    return 0


def cmd_ci_test(args):
    """Cognitive regression test (1.5.0)."""

    min_pass = float(getattr(args, "min_pass", 0.80))

    # Without an agent_fn, test against audit history
    # Use check_health as the simplified CI path
    from .sla import check_health
    report = check_health(
        min_pass_rate=min_pass,
        min_confidence=float(getattr(args, "min_conf", 0.30)),
        max_warn_rate=float(getattr(args, "max_warn", 0.25)),
        window=int(getattr(args, "window", 50)),
    )

    if report.healthy:
        print(f"[styxx ci] PASS: {report.gate_pass_rate*100:.0f}% pass, conf {report.mean_confidence:.2f}")
        return 0
    else:
        print("[styxx ci] FAIL:")
        for v in report.violations:
            print(f"  ! {v}")
        return 1


def cmd_temperature(args):
    """Run cognitive temperature analysis on demo trajectories (3.2.0)."""
    from .temperature import demo_temperature
    demo_temperature(verbose=True)
    return 0


def cmd_gate(args):
    """styxx gate <prompt> — pre-flight cognitive verdict."""
    import json as _json
    from .gate import gate

    # Auto-detect client: prefer Anthropic if ANTHROPIC_API_KEY set,
    # else OpenAI if OPENAI_API_KEY set, else text-heuristic fallback.
    client = None
    model = args.model
    if os.environ.get("ANTHROPIC_API_KEY") and model.startswith("claude"):
        try:
            from anthropic import Anthropic
            client = Anthropic()
        except ImportError:
            pass
    elif os.environ.get("OPENAI_API_KEY") and model.startswith("gpt"):
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            pass

    verdict = gate(
        client=client, model=model, prompt=args.prompt,
        consensus_n=args.n, temperature=args.temp,
    )

    if args.format == "json":
        print(_json.dumps(verdict.as_dict(), indent=2))
    else:
        print(verdict)


def cmd_intercept(args):
    """Run the cognitive intercept simulation (3.2.0)."""
    from .intercept import simulate_all_demo
    simulate_all_demo(verbose=True)
    return 0


def cmd_forecast(args):
    """Run the cognitive forecast horizon analysis (3.2.0)."""
    from .forecast import horizon_analysis

    fmt = getattr(args, "format", "ascii")
    result = horizon_analysis()

    if fmt == "json":
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print()
        print(result.render())
        print()
    return 0


def cmd_eval(args):
    """Run the ground-truth evaluation harness (3.2.0)."""
    from .eval import EvalSuite

    fmt = getattr(args, "format", "ascii")
    fixtures_path = getattr(args, "fixtures", None)

    if fixtures_path:
        suite = EvalSuite.from_json(fixtures_path)
    else:
        suite = EvalSuite.from_demo_trajectories()

    result = suite.run()

    if fmt == "json":
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print()
        print(result.render())
        print()
    return 0


def cmd_ci_baseline(args):
    """Save current state as CI baseline (1.5.0)."""
    from .ci import Baseline
    from .analytics import load_audit

    entries = load_audit(last_n=50)
    n = len(entries)
    if n < 5:
        print("[styxx ci] not enough data for baseline (need 5+ entries)")
        return 1

    gates = [e.get("gate") or "pending" for e in entries]
    pass_rate = sum(1 for g in gates if g == "pass") / n
    confs = [float(e["phase4_conf"]) for e in entries
             if e.get("phase4_conf") is not None and e.get("phase4_conf") != 0]
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    baseline = Baseline(
        agent_name=getattr(args, "name", None),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_prompts=n,
        pass_rate=pass_rate,
        mean_confidence=mean_conf,
    )

    out = getattr(args, "out", "styxx_baseline.json")
    baseline.save(out)
    print(f"[styxx ci] baseline saved to {out}")
    print(f"  entries: {n}, pass: {pass_rate*100:.0f}%, conf: {mean_conf:.2f}")
    return 0


def cmd_export(args):
    """Compliance report export (1.3.0)."""
    from .compliance import compliance_report
    days = float(args.days or 30)
    name = args.name or None
    report = compliance_report(days=days, agent_name=name)

    fmt = getattr(args, "format", "markdown")
    out = getattr(args, "out", None)

    if fmt == "json":
        content = report.as_json()
    else:
        content = report.as_markdown()

    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[styxx] report saved to {out}")
    else:
        print(content)
    return 0


def cmd_dashboard(args):
    """Live cognitive display (0.9.5)."""
    from .dashboard import dashboard
    name = args.name or "styxx agent"
    dashboard(port=args.port, agent_name=name)
    return 0


def cmd_weather(args):
    """The cognitive weather report (0.5.0).

    Not observation. Prescription. An instrument that doesn't just
    say what you are but suggests what you should become next.
    """
    from .weather import weather
    name = args.name or "styxx agent"
    window = float(args.window or 24.0)

    report = weather(agent_name=name, window_hours=window)
    if report is None:
        print()
        print("  (not enough audit data to generate a weather report)")
        print(f"  need at least 5 entries in the last {window:.0f} hours.")
        print("  run some observations first: styxx ask --watch --demo-kind reasoning")
        print()
        return 0

    fmt = getattr(args, "format", "ascii")
    if fmt == "json":
        print(report.as_json())
    elif fmt == "markdown":
        print(report.as_markdown())
    else:
        print()
        print(report.render())
        print()
    return 0


def cmd_d_axis(args):
    """Run a pure D-axis trajectory on a prompt (tier 1, 0.3.0).

    Loads the tier 1 model, generates tokens, and prints the
    per-token D values. No logprob classification — just the
    honesty signal.
    """
    from .d_axis import DAxisScorer
    use_color = color_enabled()
    c = Palette

    prompt = args.prompt
    max_tokens = args.max_tokens or 30

    print()
    print(wrap("  styxx d-axis", c.MATRIX, use_color)
          + wrap("   tier 1 honesty trajectory", c.DIM, use_color))
    print(wrap("  " + "=" * 64, c.DIM, use_color))
    print(wrap(f"  prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}", c.DIM, use_color))
    print()

    try:
        scorer = DAxisScorer()
        d_vals = scorer.score_trajectory(prompt, max_tokens=max_tokens)
    except ImportError as e:
        print(wrap(f"  tier 1 not available: {e}", c.RED, use_color))
        print(wrap("  install with: pip install 'styxx[tier1]'", c.DIM, use_color))
        print()
        return 1
    except Exception as e:
        print(wrap(f"  d-axis error: {type(e).__name__}: {e}", c.RED, use_color))
        print()
        return 1

    # Render per-token D values with a bar
    for i, d in enumerate(d_vals):
        bar_len = max(0, int(abs(d) * 40))
        if d >= 0.7:
            bar_color = c.MATRIX
            label = "honest"
        elif d >= 0.4:
            bar_color = c.YELLOW
            label = "mixed"
        else:
            bar_color = c.RED
            label = "divergent"
        bar = "█" * bar_len
        print(
            wrap(f"  t={i:<3}", c.DIM, use_color)
            + wrap(f" D={d:>+.3f} ", c.CYAN, use_color)
            + wrap(bar, bar_color, use_color)
            + wrap(f"  {label}", c.DIM, use_color)
        )

    # Summary
    from .d_axis import DAxisStats
    stats = DAxisStats.from_values(d_vals)
    print()
    print(wrap("  " + "-" * 64, c.DIM, use_color))
    print(
        wrap(f"  mean={stats.mean:+.3f}", c.CYAN, use_color)
        + wrap(f"  std={stats.std:.3f}", c.DIM, use_color)
        + wrap(f"  delta={stats.delta:+.3f}", c.CYAN, use_color)
        + wrap(f"  (n={stats.n_tokens})", c.DIM, use_color)
    )
    if stats.delta < -0.05:
        print(wrap("  trend: getting LESS honest over generation", c.YELLOW, use_color))
    elif stats.delta > 0.05:
        print(wrap("  trend: getting MORE honest over generation", c.MATRIX, use_color))
    else:
        print(wrap("  trend: stable honesty across generation", c.DIM, use_color))

    if hasattr(scorer, "last_generated_text") and scorer.last_generated_text:
        print()
        print(wrap("  generated:", c.DIM, use_color))
        text = scorer.last_generated_text[:200]
        for line in text.split("\n"):
            print(wrap(f"    {line}", c.CYAN, use_color))
    print()
    return 0


def cmd_doctor(args):
    """Run the diagnostic health check (0.1.0a3)."""
    from .doctor import run_doctor
    return run_doctor()


def cmd_audit(args):
    """styxx audit <prompt> <response> — score a (prompt, response) pair.

    7.7.3: CLI face of ``styxx.preflight()``. The atomic per-turn audit
    primitive surfaced as a CLI command for the user-facing "score my draft
    response" use case (previously Python-only). Reads the response against
    the lexical cognometric instruments + the trusted-gate suppression,
    prints a compact card with composite + axes + needs_revision flag +
    construct-ceiling fires. Persists to the active chart.jsonl by default
    (per-agent if STYXX_AGENT_NAME is set, top-level fallback otherwise).

    Either or both of <prompt> and <response> may be ``-`` to read from
    stdin (one stream; if both are ``-`` the input is split on the first
    blank line, prompt first).

    --json prints the structured dict instead of the card.
    """
    import json as _json
    import styxx

    prompt = args.prompt
    response = args.response
    if prompt == "-" and response == "-":
        blob = sys.stdin.read()
        parts = blob.split("\n\n", 1)
        prompt = parts[0].strip()
        response = parts[1].strip() if len(parts) > 1 else ""
    elif prompt == "-":
        prompt = sys.stdin.read().strip()
    elif response == "-":
        response = sys.stdin.read().strip()

    result = styxx.preflight(
        prompt,
        response,
        persist=not args.no_persist,
    )

    if args.format == "json":
        as_dict = (result.as_dict() if hasattr(result, "as_dict")
                   else {"composite": result.composite,
                         "scores": dict(getattr(result, "scores", {}) or {}),
                         "needs_revision": result.needs_revision,
                         "construct_ceiling_fires":
                             list(getattr(result, "construct_ceiling_fires", []) or [])})
        print(_json.dumps(as_dict, indent=2, default=str))
        return 0

    # Compact card. Width matches the gate card so they read as a family.
    width = 64
    inner = width - 2  # account for the two │ borders
    scores = dict(getattr(result, "scores", {}) or {})
    ceilings = list(getattr(result, "construct_ceiling_fires", []) or [])
    rev_tag = " (REVISE)" if result.needs_revision else ""

    def _row(label, value):
        line = f" {label} {value}"
        return f"│{line.ljust(inner)}│"

    print("┌" + "─" * inner + "┐")
    print(_row("styxx audit", ""))
    prompt_preview = (prompt[:40] + "...") if len(prompt) > 40 else prompt
    resp_preview = (response[:40] + "...") if len(response) > 40 else response
    print(_row("prompt:  ", repr(prompt_preview)))
    print(_row("response:", repr(resp_preview.replace("\n", " "))))
    print("│" + " " * inner + "│")
    print(_row("composite:", f"{result.composite:.3f}{rev_tag}"))
    for ax in sorted(scores):
        v = float(scores.get(ax, 0.0))
        bar_w = max(0, min(20, int(v * 20)))
        bar = "█" * bar_w + "░" * (20 - bar_w)
        print(_row(f"{ax:<14}", f"{v:.3f}  {bar}"))
    if ceilings:
        print(_row("ceilings: ", ", ".join(ceilings)))
    advice = getattr(result, "advice", None)
    if advice:
        # advice is typically a list of PreflightAdvice dataclasses;
        # render the top instrument names + scores compactly.
        try:
            parts = []
            for a in advice:
                inst = getattr(a, "instrument", None)
                sc = getattr(a, "score", None)
                if inst is not None and sc is not None:
                    parts.append(f"{inst} {float(sc):.2f}")
                else:
                    parts.append(str(a))
            advice_text = " · ".join(parts) if parts else str(advice)
        except TypeError:
            advice_text = str(advice)
        advice_text = advice_text.replace("\n", " ")
        if len(advice_text) > inner - 12:
            advice_text = advice_text[:inner - 15] + "..."
        print(_row("flagged:  ", advice_text))
    print("└" + "─" * inner + "┘")
    return 0


def cmd_audit_claims(args):
    """styxx audit-claims <file> — falsify an agent self-report against substrate.

    Extracts deterministic, checkable claims (version pins, file-contains, git
    tags, pdf page counts) from an agent's free-text self-report and mechanically
    verifies each against the repo. Designed as a one-line CI merge gate:

        styxx audit-claims pr_body.md --repo . || exit 1

    Exit 0 if every extracted claim PASSes (or none were found); 1 if any claim
    is contradicted by the substrate; 2 on input error. Free-form prose with no
    checkable assertion is reported as uncovered, not failed — the gate fails on
    lies it can check, not on claims it cannot extract.
    """
    from .agent_audit import extract_claims, AgentClaimAuditor

    src = Path(args.file)
    if not src.is_file():
        print(f"styxx audit-claims: file not found: {src}", file=sys.stderr)
        return 2
    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"styxx audit-claims: repo not found: {repo}", file=sys.stderr)
        return 2

    text = src.read_text(encoding="utf-8", errors="replace")
    rep = extract_claims(text)
    results = AgentClaimAuditor(repo_path=repo).run(rep.claims)
    n_pass = sum(1 for r in results if r.verdict == "PASS")
    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    n_err = sum(1 for r in results if r.verdict == "ERROR")

    summary = {
        "styxx_audit_claims": {
            "file": str(src),
            "repo": str(repo),
            "sentences_total": rep.sentences_total,
            "sentences_matched": rep.sentences_matched,
            "coverage": round(rep.coverage, 4),
            "claims_extracted": len(results),
            "passed": n_pass,
            "failed": n_fail,
            "errored": n_err,
            "gate": "PASS" if (n_fail == 0 and n_err == 0) else "FAIL",
            "results": [r.to_dict() for r in results],
        }
    }

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"styxx audit-claims — {src}")
        print(f"  extracted {len(results)} checkable claim(s) from "
              f"{rep.sentences_total} sentence(s) "
              f"(coverage {rep.coverage:.2f}); free-form prose not checked")
        for r in results:
            mark = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR "}[r.verdict]
            print(f"  [{mark}] {r.text!r}")
            line = (r.evidence.splitlines()[0] if r.evidence else r.error)
            print(f"         {line}")
        if not results:
            print("  (no checkable claims found — nothing to falsify)")
        print(f"  -> {n_pass} passed, {n_fail} failed, {n_err} errored")
        print("  GATE: " + ("PASS" if (n_fail == 0 and n_err == 0) else
                            "FAIL — self-report contradicts substrate"))
        print("JSON:" + json.dumps(summary, default=str))

    return 0 if (n_fail == 0 and n_err == 0) else 1


def cmd_audit_claim(args):
    """styxx audit-claim — productized single-call honesty audit (the spellchecker).

    Runs the calibrated 7.7.13 audit stack (grounded_honesty + detect_context_injection)
    on a stated factual self-claim. Drives N stateless resamples (and N in-session
    resamples if --in-session-json is given) internally; returns a structured JSON
    verdict on stdout.

    Exit codes:
        0  — verdict is "honest" (the only deploy-clean outcome)
        1  — verdict is anything else (contradiction / confabulation / injected /
              abstain). Operator gate semantics: a one-line CI check that fails
              loudly on anything that isn't cleanly honest.
        2  — input error (missing claim/question, malformed --in-session-json, etc.)
    """
    from .audit import audit_claim

    claim = (args.claim or "").strip()
    question = (args.question or "").strip()
    if not claim:
        print("styxx audit-claim: --claim must be non-empty", file=sys.stderr)
        return 2
    if not question:
        print("styxx audit-claim: --question must be non-empty", file=sys.stderr)
        return 2

    in_session = None
    if args.in_session_json:
        in_path = Path(args.in_session_json)
        if not in_path.exists():
            print(f"styxx audit-claim: --in-session-json not found: {in_path}",
                  file=sys.stderr)
            return 2
        try:
            in_session = json.loads(in_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"styxx audit-claim: malformed --in-session-json: {e}",
                  file=sys.stderr)
            return 2
        if not isinstance(in_session, list):
            print("styxx audit-claim: --in-session-json must be a JSON array "
                  "of {role, content} message objects", file=sys.stderr)
            return 2

    try:
        result = audit_claim(
            claim=claim,
            question=question,
            in_session_messages=in_session,
            model=args.model,
            judge_model=args.judge_model,
            n=args.n,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"styxx audit-claim: audit failed: {e}", file=sys.stderr)
        return 2

    # Always emit a JSON line — operator-greppable + agent-parseable.
    payload = {
        "claim": result.claim,
        "question": result.question,
        "verdict": result.verdict,
        "grounded": round(result.grounded, 4),
        "stability": round(result.stability, 4),
        "concordance_stateless": round(result.concordance_stateless, 4),
        "concordance_in_session": (
            None if result.concordance_in_session is None
            else round(result.concordance_in_session, 4)
        ),
        "divergence": (None if result.divergence is None
                       else round(result.divergence, 4)),
        "injection_suspected": result.injection_suspected,
        "confidence": result.confidence,
        "scope_warnings": list(result.scope_warnings),
        "calibration": result.calibration,
        "n_clusters_stateless": result.n_clusters_stateless,
        "n_clusters_in_session": result.n_clusters_in_session,
    }
    if args.with_samples:
        payload["samples_stateless"] = list(result.samples_stateless)
        if result.samples_in_session is not None:
            payload["samples_in_session"] = list(result.samples_in_session)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"verdict:                {result.verdict.upper()}")
        print(f"grounded:               {result.grounded:.3f}")
        print(f"stability:              {result.stability:.3f}  ({result.confidence})")
        print(f"concordance_stateless:  {result.concordance_stateless:.3f}")
        if result.concordance_in_session is not None:
            print(f"concordance_in_session: {result.concordance_in_session:.3f}")
            print(f"divergence:             {result.divergence:.3f}")
            print(f"injection_suspected:    {result.injection_suspected}")
        print(f"scope_warnings:         {list(result.scope_warnings)}")
        print(f"calibration:            {result.calibration[:80]}...")
        print("JSON:" + json.dumps(payload, default=str))

    return 0 if result.verdict == "honest" else 1


def cmd_audit_session(args):
    """styxx audit-session — multi-claim session-level audit (CI gate-ready).

    Runs audit_claim on every (claim, question) tuple from --claims-json against
    the session context from --messages-json. Returns a SessionAudit JSON on
    stdout with per-claim verdicts + session-level roll-up. Exit 1 if any claim
    is non-honest (deploy-gate semantics).
    """
    from .audit import audit_session

    msg_path = Path(args.messages_json)
    if not msg_path.exists():
        print(f"styxx audit-session: --messages-json not found: {msg_path}",
              file=sys.stderr)
        return 2
    claims_path = Path(args.claims_json)
    if not claims_path.exists():
        print(f"styxx audit-session: --claims-json not found: {claims_path}",
              file=sys.stderr)
        return 2

    try:
        messages = json.loads(msg_path.read_text(encoding="utf-8"))
        claims_raw = json.loads(claims_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"styxx audit-session: malformed JSON input: {e}", file=sys.stderr)
        return 2

    if not isinstance(messages, list) or not isinstance(claims_raw, list):
        print("styxx audit-session: --messages-json and --claims-json must "
              "both be JSON arrays", file=sys.stderr)
        return 2

    # Accept either ["claim", "question"] or {"claim":..., "question":...}.
    claims: list[tuple[str, str]] = []
    for item in claims_raw:
        if isinstance(item, dict):
            c = item.get("claim", "").strip()
            q = item.get("question", "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            c, q = str(item[0]).strip(), str(item[1]).strip()
        else:
            print(f"styxx audit-session: malformed claim entry: {item!r}",
                  file=sys.stderr)
            return 2
        if not c or not q:
            print(f"styxx audit-session: empty claim or question in: {item!r}",
                  file=sys.stderr)
            return 2
        claims.append((c, q))

    try:
        session = audit_session(
            messages=messages,
            claims=claims,
            model=args.model,
            judge_model=args.judge_model,
            n=args.n,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"styxx audit-session: audit failed: {e}", file=sys.stderr)
        return 2

    payload = {
        "verdict": session.verdict,
        "injection_suspected": session.injection_suspected,
        "n_honest": session.n_honest,
        "n_contradiction": session.n_contradiction,
        "n_confabulation": session.n_confabulation,
        "n_injected": session.n_injected,
        "n_abstain": session.n_abstain,
        "scope_warnings": list(session.scope_warnings),
        "calibration": session.calibration,
        "claims": [
            {
                "claim": r.claim, "question": r.question, "verdict": r.verdict,
                "grounded": round(r.grounded, 4),
                "stability": round(r.stability, 4),
                "injection_suspected": r.injection_suspected,
                "divergence": (None if r.divergence is None
                               else round(r.divergence, 4)),
                "confidence": r.confidence,
            }
            for r in session.claims
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if session.verdict == "honest" else 1


def cmd_attest(args):
    """styxx attest <file> — emit a Verifiable Cognometric Attestation.

    Builds a content-addressed artifact of an agent self-report: the extracted
    checkable claims + their PASS/FAIL verdicts against the substrate, the
    EU AI Act Article 15 clause mapping, the explicit uncovered-requirements
    boundary, and a SHA-256 digest. Any third party can re-derive the verdicts
    with ``styxx verify-attestation`` — trust the substrate, not the agent.

    Exit 0 if the artifact was produced (regardless of pass/fail); 2 on input
    error. Use verify-attestation for the gate semantics.
    """
    from .attestation import attest

    src = Path(args.file)
    if not src.is_file():
        print(f"styxx attest: file not found: {src}", file=sys.stderr)
        return 2
    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"styxx attest: repo not found: {repo}", file=sys.stderr)
        return 2

    ref = getattr(args, "ref", None)
    vitals = getattr(args, "vitals", False)
    prompt = getattr(args, "prompt", None)
    try:
        att = attest(
            src.read_text(encoding="utf-8", errors="replace"), repo,
            ref=ref, prompt=prompt, vitals=vitals,
        )
    except ValueError as e:
        print(f"styxx attest: {e}", file=sys.stderr)
        return 2
    out_json = att.to_json()
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
        s = att.artifact["summary"]
        sub_c = att.artifact["substrate"]["commit"]
        print(f"styxx attest — wrote {args.out}")
        if ref:
            print(f"  pinned to commit {sub_c[:12]} (ref {ref})")
        if vitals and "vitals" in att.artifact:
            scores = att.artifact["vitals"]["scores"]
            pretty = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
            print(f"  vitals (register, not honesty): {pretty}")
        print(f"  {s['passed']} passed, {s['failed']} failed, {s['errored']} errored "
              f"(coverage {s['coverage']:.2f})")
        print(f"  digest sha256:{att.digest}")
        print(f"  verify with: styxx verify-attestation {args.out} --repo {args.repo}")
    else:
        print(out_json)
    return 0


def cmd_verify_attestation(args):
    """styxx verify-attestation <file> — re-derive verdicts from the substrate.

    Recomputes the SHA-256 digest (tamper-evidence) and re-runs every claim's
    checker against the repo, comparing the re-derived verdict to the embedded
    one. Never trusts the embedded verdict — this is what makes the artifact
    agent-independent.

    Exit 0 if the digest matches AND every embedded verdict reproduces; 1 if a
    verdict mismatches, the digest is broken, or an unknown checker is named;
    2 on input error.
    """
    from .attestation import verify_attestation

    src = Path(args.file)
    if not src.is_file():
        print(f"styxx verify-attestation: file not found: {src}", file=sys.stderr)
        return 2
    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"styxx verify-attestation: repo not found: {repo}", file=sys.stderr)
        return 2

    try:
        artifact = json.loads(src.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        print(f"styxx verify-attestation: not valid JSON: {e}", file=sys.stderr)
        return 2

    res = verify_attestation(artifact, repo)
    out = res.to_dict()
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"styxx verify-attestation — {src}")
        print(f"  digest: {'OK' if res.digest_ok else 'BROKEN (payload tampered)'}")
        print(f"  reproduced {len(res.reproduced)} verdict(s) from substrate; "
              f"{len(res.mismatches)} mismatch(es)")
        for m in res.mismatches:
            print(f"  [MISMATCH] {m['id']}: embedded={m['embedded_verdict']} "
                  f"substrate={m['reproduced_verdict']}")
        for u in res.unknown_checkers:
            print(f"  [REFUSED] unknown checker not in allowlist: {u!r}")
        if res.vitals_present:
            print(f"  vitals (register): "
                  f"{'OK — scores re-derive from recorded text' if res.vitals_ok else 'TAMPERED'}")
            for vm in res.vitals_mismatches:
                print(f"  [VITALS MISMATCH] {vm.get('axis')}: "
                      f"embedded={vm.get('embedded')} rederived={vm.get('rederived')}")
        print("  VERIFIED: " + ("OK — every verdict reproduces against the substrate"
                                if res.ok else "FAILED — artifact disagrees with substrate"))
    return 0 if res.ok else 1


def cmd_critique(args):
    """styxx critique <prompt> <response> — audit + register-fix suggestions.

    7.7.4: extends ``styxx audit`` with prescriptive register-fix suggestions
    when the draft fires the trusted gate or pushes any axis above a threshold.
    Suggestions are derived from the closed-loop dogfood pattern recorded in
    ``papers/agent-self-audit/FINDING_pareto_frontier_2026_05_27.md``:

      - high sycophancy + agreement-vocab patterns → suggest dropping
        agreement-opener phrases ("exactly this," "the strongest," "predicted
        exactly this") that fire the lexical instrument's restrained-FP.
      - high overconfidence with short-text register → suggest adding
        hedges, parentheticals, and structural connectors; do NOT compress
        to fewer than three sentences (the closed-negative refinement from
        commit ab08822 documented brevity floors the instrument).
      - construct-ceiling fires (overconfidence) → name the axis explicitly
        and reference the documented bound (text-only register cannot
        recalibrate; see commit 7c36ed9 H_null).

    The suggestions are SCOPE-BOUNDED on output. The same session that
    derived these rules also falsified the prescriptive form on
    completion-status text (composite stays elevated even after register
    fix because the FP is content-class-determined, not register-
    determined). The tool surfaces both the suggestion AND its
    documented limit.
    """
    import json as _json
    import styxx

    prompt = args.prompt
    response = args.response
    if prompt == "-" and response == "-":
        blob = sys.stdin.read()
        parts = blob.split("\n\n", 1)
        prompt = parts[0].strip()
        response = parts[1].strip() if len(parts) > 1 else ""
    elif prompt == "-":
        prompt = sys.stdin.read().strip()
    elif response == "-":
        response = sys.stdin.read().strip()

    result = styxx.preflight(prompt, response, persist=not args.no_persist)
    scores = dict(getattr(result, "scores", {}) or {})
    ceilings = list(getattr(result, "construct_ceiling_fires", []) or [])

    # Generate suggestions based on the register-law derived from the
    # 2026-05-27 Pareto-frontier dogfood. Each suggestion carries its
    # closed-negative scope so the user knows where the rule does NOT apply.
    suggestions = []
    syc = float(scores.get("sycophancy", 0.0) or 0.0)
    over = float(scores.get("overconfidence", 0.0) or 0.0)
    refu = float(scores.get("refusal", 0.0) or 0.0)
    n_words = len(response.split())

    if syc >= 0.50:
        # Detect specific agreement-opener phrases the lexical instrument fires on.
        # The list is derived from the documented features in
        # styxx/guardrail/sycophancy_signals.py + the closed-negative refinement
        # at commit ab08822 (papers/sycophancy-target-gate/FINDING_restrained_refinement_2026_05_25.md).
        AGREE_OPENERS = (
            "exactly this", "exactly that", "exactly right", "exactly so",
            "the strongest", "predicted exactly", "the cleanest",
            "absolutely right", "absolutely correct", "completely agree",
            "yes, absolutely", "you're absolutely",
        )
        hits = [p for p in AGREE_OPENERS if p in response.lower()]
        if hits:
            suggestions.append({
                "axis": "sycophancy",
                "score": round(syc, 3),
                "trigger": "agreement-opener phrases",
                "found": hits,
                "fix": "lead with the framing question or numbers; drop the agreement-with-data openers.",
                "scope_bound": "the lexical instrument cannot distinguish factual-agreement from yielding-to-interlocutor "
                               "on agreement-with-data content (see commit ab08822); on completion-status text this "
                               "fix may not drop sycoph below the FP band.",
            })
        else:
            suggestions.append({
                "axis": "sycophancy",
                "score": round(syc, 3),
                "trigger": "agreement-lexicon density (no specific opener found)",
                "fix": "reduce agreement-vocab density; lead with framing question or numbers.",
                "scope_bound": "restrained-FP on factual-agreement content is a documented construct ceiling "
                               "(commit ab08822); register fix may not fully clear the gate.",
            })

    if over >= 0.40 or "overconfidence" in ceilings:
        ceiling_note = " — construct ceiling fired" if "overconfidence" in ceilings else ""
        if n_words < 30:
            suggestions.append({
                "axis": "overconfidence",
                "score": round(over, 3),
                "trigger": f"short text ({n_words} words){ceiling_note}",
                "fix": "expand to >=3 sentences with structural connectors and at least one hedge. "
                       "ultra-terse declaratives are peak overconfidence register to the lexical instrument.",
                "scope_bound": "text-only overconfidence recalibration is a closed negative (commit 7c36ed9 H_null); "
                               "the score measures stated-confidence register, not actual calibration.",
            })
        else:
            suggestions.append({
                "axis": "overconfidence",
                "score": round(over, 3),
                "trigger": f"declarative chain density{ceiling_note}",
                "fix": "add conditional/interrogative framing ('if X holds…' beats 'X will hold'); "
                       "keep hedges and parentheticals — DO NOT strip them. stripping hedges raises overconfidence "
                       "on the Pareto axis (see FINDING_pareto_frontier_2026_05_27.md).",
                "scope_bound": "the score is a register signal, not validity. confident phrasing fires this on "
                               "factually correct text (commit 7c36ed9).",
            })

    if refu >= 0.50:
        suggestions.append({
            "axis": "refusal",
            "score": round(refu, 3),
            "trigger": "refusal-shape register",
            "fix": "if the draft is genuinely declining, this is correct; if not, reduce hedge density "
                   "or qualifier-stacking. note: lowering refusal often raises overconfidence on the Pareto axis.",
            "scope_bound": "refusal and overconfidence trade off on the lexical instrument; corner-optimization "
                           "on either is not the goal (FINDING_pareto_frontier_2026_05_27.md).",
        })

    if not suggestions and not result.needs_revision:
        # Clean draft — no fixes proposed, but still surface the audit.
        suggestions.append({
            "axis": "all",
            "score": round(result.composite, 3),
            "trigger": "(no register issues detected)",
            "fix": "no fixes proposed; draft is below the gate's flag thresholds.",
            "scope_bound": "absence of flag is not validation of content; the instrument measures register, "
                           "not validity. content audit is out of scope.",
        })

    out = {
        "audit": {
            "composite": round(result.composite, 3),
            "scores": {k: round(float(v), 3) for k, v in scores.items()},
            "needs_revision": bool(result.needs_revision),
            "construct_ceiling_fires": ceilings,
        },
        "suggestions": suggestions,
    }

    if args.format == "json":
        print(_json.dumps(out, indent=2, default=str))
        return 0

    # Card render
    width = 70
    inner = width - 2
    def _row(label, value):
        line = f" {label} {value}"
        return f"│{line.ljust(inner)}│"

    print("┌" + "─" * inner + "┐")
    print(_row("styxx critique", ""))
    print(_row("composite:", f"{result.composite:.3f}{' (REVISE)' if result.needs_revision else ''}"))
    for ax in sorted(scores):
        v = float(scores.get(ax, 0))
        bar_w = max(0, min(20, int(v * 20)))
        bar = "█" * bar_w + "░" * (20 - bar_w)
        print(_row(f"{ax:<14}", f"{v:.3f}  {bar}"))
    if ceilings:
        print(_row("ceilings: ", ", ".join(ceilings)))
    print("├" + "─" * inner + "┤")
    print(_row("suggestions:", ""))
    for s in suggestions:
        print(_row(f"[{s['axis']}]", f"{s['trigger']}"))
        # fix line, wrapped to inner width
        fix = s['fix']
        while fix:
            chunk = fix[:inner - 6]
            # break at last space if possible
            if len(fix) > inner - 6:
                br = chunk.rfind(" ")
                if br > 20:
                    chunk = chunk[:br]
            print(_row("  fix:", chunk))
            fix = fix[len(chunk):].lstrip()
        # scope bound
        sb = s['scope_bound']
        while sb:
            chunk = sb[:inner - 8]
            if len(sb) > inner - 8:
                br = chunk.rfind(" ")
                if br > 20:
                    chunk = chunk[:br]
            print(_row("  scope:", chunk))
            sb = sb[len(chunk):].lstrip()
        print(_row("", ""))
    print("└" + "─" * inner + "┘")
    return 0


def cmd_leaderboard(args):
    """styxx leaderboard — display the current gauntlet leaderboard in the terminal.

    7.7.7: lightweight CLI to read the LEADERBOARD.md from the package's bundled
    copy (when present) or fetch the latest from origin. No external dependencies
    on http requests in the default path; the bundled copy is preferred. The
    explicit purpose: lower the friction between "I'm trying out styxx" and "I
    can see who's on the floor" to a single command.
    """
    from pathlib import Path as _Path
    pkg_data = _Path(__file__).resolve().parent / "_data" / "LEADERBOARD.md"
    source_tree = _Path(__file__).resolve().parent.parent / "LEADERBOARD.md"
    md_path = None
    if pkg_data.exists():
        md_path = pkg_data
    elif source_tree.exists():
        md_path = source_tree

    if md_path is None:
        print("leaderboard not found in this install (no bundled copy + no source-tree path).")
        print("see: https://github.com/fathom-lab/styxx/blob/main/LEADERBOARD.md")
        return 1

    text = md_path.read_text(encoding="utf-8")

    # Optional --rows-only filter: print only the leaderboard rows table, no header text.
    if args.rows_only:
        # Heuristic: emit lines from the first "## Leaderboard" or "### Reference baselines"
        # heading through the first "---" divider after it.
        lines = text.splitlines()
        in_table = False
        for line in lines:
            if not in_table and ("## Leaderboard" in line or "### Reference baselines" in line):
                in_table = True
            if in_table:
                if line.strip() == "---":
                    break
                print(line)
        return 0

    print(text)
    return 0


def cmd_gauntlet(args):
    """styxx gauntlet — run a candidate method against the empirical floor.

    7.7.5: the public-challenge runner. Loads any user-supplied detection or
    classification method, runs it against the labeled benchmark
    (``papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json``),
    scores it against pre-registered bars, and prints a structured result.

    The bars are the same closed-negative bars from the seven-method floor
    paper (`PAPER_decorrelation_ceiling_2026_05_27.md`). We assert we
    couldn't beat them with the seven methods we tested. The gauntlet
    invites external researchers to try.

    Method spec format: ``module:attr``, e.g. ``my_module:predict`` or
    ``my_pkg.sub:detect``. The module must be importable.

    Task = "classification" → method signature: ``predict(question) -> {"class": "..."}``.
    Task = "detection" → method signature: ``detect(question, response) -> {"score": float}``.
    """
    import json as _json
    from pathlib import Path as _Path
    from styxx.gauntlet import (
        load_benchmark, resolve_method, Submission,
        run_classification_gauntlet, run_detection_gauntlet,
    )

    try:
        method = resolve_method(args.method)
    except (ValueError, AttributeError, TypeError, ImportError) as e:
        print(f"error resolving method {args.method!r}: {e}")
        return 1

    bench_path = _Path(args.benchmark) if args.benchmark else None
    try:
        benchmark = load_benchmark(bench_path)
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 1

    name = args.name or args.method
    submission = Submission(name=name, method=method, task=args.task,
                            module_spec=args.method, notes=args.notes or "")

    if args.task == "classification":
        result = run_classification_gauntlet(submission, benchmark)
    elif args.task == "detection":
        result = run_detection_gauntlet(submission, benchmark)
    else:
        print(f"error: unknown task {args.task!r}; choose classification or detection")
        return 1

    if args.format == "json":
        print(_json.dumps(result.as_dict(), indent=2, default=str))
        return 0 if result.overall_pass else 2

    # Card render
    width = 70
    inner = width - 2
    def _row(label, value):
        line = f" {label} {value}"
        return f"│{line.ljust(inner)}│"

    print("┌" + "─" * inner + "┐")
    print(_row("styxx gauntlet", ""))
    print(_row("method:    ", submission.module_spec))
    print(_row("name:      ", submission.name))
    print(_row("task:      ", result.task))
    print(_row("benchmark: ", f"darkcore v{result.benchmark_version} (n={result.n_items})"))
    if result.error:
        print(_row("ERROR:     ", result.error))
    else:
        print("├" + "─" * inner + "┤")
        print(_row("metrics:", ""))
        for k, v in result.metrics.items():
            print(_row(f"  {k:<30}", str(v)))
        print("├" + "─" * inner + "┤")
        print(_row("bars (pre-registered):", ""))
        for bar_name, passed in result.bar_results.items():
            bar_val = result.bars.get(bar_name)
            mark = "PASS" if passed else "FAIL"
            print(_row(f"  {bar_name:<28}", f"≥{bar_val:.2f}  → {mark}"))
        print(_row("", ""))
        print(_row("overall:   ", f"{result.n_passed} / {result.n_total_bars} bars passed"
                                  f"  → {'PASS' if result.overall_pass else 'FAIL'}"))
        if not result.overall_pass:
            print(_row("", "the seven-method floor stands."))
        else:
            print(_row("", "you beat the floor. submit a PR to LEADERBOARD.md."))
    print("└" + "─" * inner + "┘")
    return 0 if result.overall_pass else 2


def cmd_gauntlet_audit_confounds(args):
    """styxx gauntlet-audit-confounds — audit the benchmark for surface confounds.

    7.7.9: the structural counterpart to D3. Runs the oracle suite from
    `styxx.gauntlet.audit_confounds` against the bundled benchmark and reports
    per-oracle D1/D2 AUC, direction-agnostic absolute AUC, Spearman ρ to
    word_length, and whether each oracle alone games the bars.

    Any orthogonal confound found (ρ to length < 0.5, AUC ≥ 0.70 in either
    direction) is a candidate for a new D-bar — the same discipline pattern
    that produced D3 in 7.7.8 and D4 in 7.7.9.
    """
    import json as _json
    from pathlib import Path as _Path
    from styxx.gauntlet import audit_confounds, load_benchmark

    bench_path = _Path(args.benchmark) if args.benchmark else None
    try:
        benchmark = load_benchmark(bench_path)
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 1

    report = audit_confounds(benchmark, d1_bar=args.d1_bar, d2_bar=args.d2_bar)

    if args.format == "json":
        print(_json.dumps(report, indent=2, default=str))
        return 0

    # Card render
    width = 96
    inner = width - 2
    def _row(label, value=""):
        line = f" {label}{value}"
        return f"│{line.ljust(inner)}│"

    print("┌" + "─" * inner + "┐")
    print(_row("styxx gauntlet — confound audit (7.7.9)"))
    print(_row("benchmark: ", f"darkcore v{report['benchmark_version']} (n={report['n_records']})"))
    print(_row("bars:      ", f"D1≥{report['d1_bar']}  D2≥{report['d2_bar']}  "
                                f"orthogonality ρ-threshold={report['orthogonality_threshold_rho']}"))
    print("├" + "─" * inner + "┤")
    header = f"  {'oracle':<26} {'D1':>7} {'D2':>7} {'D1abs':>7} {'D2abs':>7} {'dir-D1':>9} {'ρ→len':>7} {'P1':>3} {'P2':>3}"
    print(_row(header))
    print("├" + "─" * inner + "┤")
    for row in report["audit_rows"]:
        if "error" in row:
            print(_row(f"  {row['oracle']:<26} ERROR: {row['error']}"))
            continue
        line = (
            f"  {row['oracle']:<26} "
            f"{row.get('D1_AUC', '—'):>7} {row.get('D2_AUC', '—'):>7} "
            f"{row.get('D1_AUC_abs', '—'):>7} {row.get('D2_AUC_abs', '—'):>7} "
            f"{row.get('D1_direction', '—'):>9} "
            f"{row.get('spearman_rho_to_word_length') if row.get('spearman_rho_to_word_length') is not None else '—':>7} "
            f"{'✓' if row.get('passes_D1') else '·':>3} "
            f"{'✓' if row.get('passes_D2') else '·':>3}"
        )
        print(_row(line))
    print("├" + "─" * inner + "┤")
    print(_row(f"  orthogonal confounds found:        {report['n_orthogonal_confounds_found']}"))
    print(_row(f"  length-downstream confounds found: {report['n_length_downstream_confounds_found']}"))
    if report["candidate_orthogonal_confounds"]:
        print(_row(""))
        print(_row("  candidate orthogonal confounds (ρ<0.5, passes a bar):"))
        for c in report["candidate_orthogonal_confounds"]:
            print(_row(f"    • {c['oracle']}  D1abs={c['D1_AUC_abs']}  D2abs={c['D2_AUC_abs']}  ρ={c['spearman_rho_to_word_length']}"))
        print(_row(""))
        print(_row("  → discipline says: add a new D-bar with a regression test"))
    else:
        print(_row(""))
        print(_row("  no NEW orthogonal confound found — existing D-bars cover the audit space"))
    print("└" + "─" * inner + "┘")
    return 0


def cmd_data_dir(args):
    """styxx data-dir — print the active chart.jsonl path + a short summary.

    7.7.3: small discoverability command. Per-agent routing
    (``~/.styxx/agents/<agent>/chart.jsonl`` when ``STYXX_AGENT_NAME`` is
    set, ``~/.styxx/chart.jsonl`` otherwise) is documented but easy to
    miss when querying chart.jsonl directly. This prints the actual file
    the agent is writing into so users don't query the wrong path.
    """
    from .analytics import _audit_log_path
    path = _audit_log_path()
    agent = os.environ.get("STYXX_AGENT_NAME")
    print("styxx data directory:")
    print(f"  agent name:  {agent or '(unset — using top-level fallback)'}")
    print(f"  chart.jsonl: {path}")
    print(f"  exists:      {path.exists()}")
    if path.exists():
        size = path.stat().st_size
        n = 0
        try:
            with open(path, encoding="utf-8") as f:
                for _ in f:
                    n += 1
        except OSError:
            pass
        print(f"  size:        {size:,} bytes")
        print(f"  events:      {n:,}")
    return 0


def cmd_posture(args):
    """Print recent cognometric posture summary (7.4.2).

    The CLI face of ``styxx.recover_posture()``. Reads the audit log,
    builds a structured PostureSummary, and prints the narrative.
    Useful inside an agent session (call via ``!styxx posture`` from
    Claude Code) or as the first command in any new agent session to
    re-anchor on what the cognometric log says about recent state.

    --json prints the structured dict instead of the narrative for
    machine-readable consumption.
    """
    from .recover import recover_posture
    posture = recover_posture(
        session_id=args.session_id or None,
        last_n=int(args.last_n or 50),
        since_seconds=(float(args.since_seconds)
                       if args.since_seconds else None),
    )
    if args.json:
        print(json.dumps(posture.as_dict(), indent=2))
    else:
        print(posture.narrative)
    return 0


def cmd_agent_card(args):
    """Render a shareable agent personality PNG (0.1.0a4).

    0.2.0: supports --serve to run a local live dashboard.
    Falls back to ASCII when Pillow isn't installed.
    """
    # 0.2.0 live serve mode
    if getattr(args, "serve", False):
        from .serve import run_serve
        return run_serve(
            port=int(getattr(args, "port", 9797) or 9797),
            agent_name=args.name or "styxx agent",
            days=float(args.days or 7.0),
            refresh_seconds=int(getattr(args, "refresh", 30) or 30),
            open_browser=not getattr(args, "no_browser", False),
        )

    out_path = Path(args.out) if args.out else (
        Path.home() / ".styxx" / "agent-card.png"
    )
    name = args.name or "styxx agent"
    days = float(args.days or 7.0)

    try:
        from .card_image import render_agent_card
        result = render_agent_card(
            out_path=out_path, agent_name=name, days=days,
        )
    except ImportError:
        result = None
    except RuntimeError as e:
        print()
        print(f"  agent-card failed: {e}")
        print()
        return 1

    if result is None:
        # Pillow not installed — fall back to ASCII
        print()
        print("  Pillow not installed — can't render PNG.")
        print("  install with:  pip install 'styxx[agent-card]'")
        print()
        print("  falling back to ASCII profile:")
        print()
        from . import analytics
        profile = analytics.personality(days=days)
        if profile is None:
            print("  (not enough audit data to render a profile)")
        else:
            print(profile.render())
        print()
        return 0

    print()
    print(f"  agent card rendered: {result}")
    try:
        size = result.stat().st_size
        print(f"  size: {size:,} bytes (1200x630)")
    except OSError:
        pass
    print()
    print("  post it: twitter, slack, your agent's self-page,")
    print("  anywhere a shareable png fits.")
    print()
    return 0


def cmd_card(args):
    """Render the cognometric registry card.

    Variants:
      single (default) — one card from a single audit JSON
      heal             — paired BEFORE / AFTER card from two audits

    Reads styxx audit JSON(s) and writes 1200x630 PNG(s). Each card is
    appended to ~/.styxx/cards/cards.jsonl (the local provenance log).
    Requires matplotlib (`pip install 'styxx[agent-card]'`).
    """
    variant = getattr(args, "variant", None) or "single"

    try:
        from .cognometric_card import (
            CardData, render_card, render_heal_card,
        )
    except RuntimeError as e:
        print(f"\n  {e}\n")
        return 1

    if variant == "heal":
        baseline_path = getattr(args, "baseline", None)
        healed_path = getattr(args, "healed_from", None) or args.audit
        if not baseline_path:
            print("\n  variant=heal requires --baseline <audit.json> "
                  "(the pre-heal audit JSON).\n")
            return 1
        baseline_path = Path(baseline_path).expanduser()
        healed_path = Path(healed_path).expanduser()
        for p in (baseline_path, healed_path):
            if not p.exists():
                print(f"\n  audit not found: {p}\n")
                return 1
        out_path = Path(args.out).expanduser() if args.out else (
            Path.home() / ".styxx" / "heal-pair-card.png"
        )
        try:
            baseline = CardData.from_audit_json(
                baseline_path, agent=args.agent, healed=False)
            healed = CardData.from_audit_json(
                healed_path, agent=args.agent, healed=True)
        except ValueError as e:
            print(f"\n  {e}\n")
            return 1
        try:
            result = render_heal_card(baseline, healed, out_path)
        except RuntimeError as e:
            print(f"\n  heal card failed: {e}\n")
            return 1
        delta = baseline.composite_mean - healed.composite_mean
        recovery_pct = 100 * delta / max(baseline.composite_mean, 1e-6)
        print()
        print(f"  heal-pair card rendered: {result}")
        print(f"  bearer: {baseline.agent}")
        print(f"  composite  baseline {baseline.composite_mean:.3f}  "
              f"→  healed {healed.composite_mean:.3f}")
        print(f"  recovery   {recovery_pct:+.1f}%  (Δ {-delta:+.3f})")
        print()
        return 0

    # variant == single
    audit_path = Path(args.audit).expanduser()
    if not audit_path.exists():
        print(f"\n  audit not found: {audit_path}\n")
        return 1
    out_path = Path(args.out).expanduser() if args.out else (
        Path.home() / ".styxx" / "cognometric-card.png"
    )
    try:
        data = CardData.from_audit_json(
            audit_path, agent=args.agent,
            healed=bool(getattr(args, "healed", False)))
    except ValueError as e:
        print(f"\n  {e}\n")
        return 1
    try:
        result = render_card(data, out_path)
    except RuntimeError as e:
        print(f"\n  cognometric card failed: {e}\n")
        return 1

    print()
    print(f"  cognometric card rendered: {result}")
    try:
        print(f"  size: {result.stat().st_size:,} bytes (1200x630)")
    except OSError:
        pass
    print(f"  bearer: {data.agent}  ·  composite {data.composite_mean:.3f}")
    print(f"  observation: {'post-heal' if data.healed else 'field'} "
          f"·  {data.n_turns} turn{'s' if data.n_turns != 1 else ''}")
    print()
    return 0


def cmd_cards_list(args):
    """List recent cards from ~/.styxx/cards/cards.jsonl."""
    try:
        from .cognometric_card import list_cards
    except RuntimeError as e:
        print(f"\n  {e}\n")
        return 1
    limit = int(getattr(args, "limit", 20) or 20)
    records = list_cards(limit=limit)
    if not records:
        print("\n  no cards issued yet. run `styxx card --audit ...` "
              "or call cogn_share_card over MCP.\n")
        return 0
    print()
    print(f"  cognometric card registry  ·  last {len(records)} "
          f"entr{'y' if len(records) == 1 else 'ies'}")
    print()
    print(f"  {'serial':<11} {'variant':<12} {'agent':<24} {'composite':>10}  {'band':<10} path")
    print(f"  {'-'*11} {'-'*12} {'-'*24} {'-'*10}  {'-'*10} {'-'*30}")
    for r in records:
        serial = r.get("serial", "")[:11]
        variant = r.get("variant", "")[:12]
        agent = r.get("agent", "")[:24]
        comp = r.get("composite", 0.0)
        band = r.get("band", "")[:10]
        path = r.get("path", "")
        # show only filename for brevity
        fname = Path(path).name if path else ""
        print(f"  {serial:<11} {variant:<12} {agent:<24} {comp:>10.4f}  {band:<10} {fname}")
    print()
    return 0


def cmd_personality(args):
    """Render the personality profile over the last N days (0.1.0a3).

    0.2.0: supports --format [ascii|json|csv|markdown] for export
    to other tools.
    """
    from . import analytics
    days = float(args.days or 7.0)
    profile = analytics.personality(days=days)
    if profile is None:
        print()
        print("  (not enough audit data to compute a personality profile)")
        print(f"  need at least 5 entries in the last {days:.0f} days.")
        print("  run some observations first: styxx ask --watch --demo-kind refusal")
        print()
        return 0

    fmt = getattr(args, "format", "ascii")
    if fmt == "json":
        print(profile.as_json())
    elif fmt == "csv":
        print(profile.as_csv())
    elif fmt == "markdown":
        print(profile.as_markdown())
    else:
        print()
        print(profile.render())
        print()
    return 0


def cmd_reflect(args):
    """Run the agent self-check and render the reflection report (0.2.0).

    Calls styxx.reflect() which returns a ReflectionReport with
    current personality + yesterday baseline + drift + suggested
    actions. The CLI renders either text or json.
    """
    from . import analytics
    now_days = float(getattr(args, "now_days", 1.0))
    baseline_days = float(getattr(args, "baseline_days", 7.0))
    report = analytics.reflect(
        now_days=now_days, baseline_days=baseline_days,
    )
    fmt = getattr(args, "format", "ascii")
    if fmt == "json":
        print(report.as_json())
    elif fmt == "markdown":
        print(report.as_markdown())
    else:
        print()
        print(report.render())
    return 0


def cmd_dreamer(args):
    """Retroactive reflex tuning on the audit log (0.1.0a3)."""
    from . import analytics
    threshold = float(args.threshold)
    last_n = args.last_n
    report = analytics.dreamer(threshold=threshold, last_n=last_n)
    print()
    print(report.summary())
    print()
    return 0


def cmd_mood(args):
    """Print the current mood label (0.1.0a3).

    0.2.2: now prints the window used so the output is unambiguous.
    Previously the 60-min default CLI window disagreed with
    reflect's 24h window and card's 7d window, causing three
    different mood labels from the same audit log. The fix is
    transparency: show which window you're reading.
    """
    from . import analytics
    window_min = float(args.window) if args.window else 60.0
    window_s = window_min * 60.0
    m = analytics.mood(window_s=window_s)
    print()
    print(f"  mood: {m}  (window: last {window_min:.0f} min)")
    print()
    return 0


def cmd_fingerprint(args):
    """Print the cognitive fingerprint (0.1.0a3).

    If `compare` is set, compare two sessions' fingerprints instead
    of printing one.
    """
    from . import analytics

    if getattr(args, "compare", False):
        fp_a = analytics.fingerprint(
            last_n=args.last_n or 500, session_id=args.session_a,
        )
        fp_b = analytics.fingerprint(
            last_n=args.last_n or 500, session_id=args.session_b,
        )
        use_color = color_enabled()
        c = Palette
        print()
        if fp_a is None:
            print(f"  (no audit data for session '{args.session_a}')")
            print()
            return 1
        if fp_b is None:
            print(f"  (no audit data for session '{args.session_b}')")
            print()
            return 1

        sim = fp_a.cosine_similarity(fp_b)
        drift = 1.0 - sim

        # Drift color: green = stable, yellow = small drift, red = big drift
        if drift < 0.05:
            drift_color = c.MATRIX
            drift_label = "stable"
        elif drift < 0.20:
            drift_color = c.YELLOW
            drift_label = "slight drift"
        else:
            drift_color = c.RED
            drift_label = "significant drift"

        print(wrap("  fingerprint comparison", c.MATRIX, use_color))
        print(wrap("  " + "=" * 64, c.DIM, use_color))
        print(f"  session a ({args.session_a}) - {fp_a.n_samples} samples")
        print(f"  session b ({args.session_b}) - {fp_b.n_samples} samples")
        print()
        print(f"  cosine similarity : {sim:.4f}")
        print(wrap(
            f"  drift             : {drift:.4f} ({drift_label})",
            drift_color, use_color,
        ))
        print()
        # Per-component diffs
        categories = ("retrieval", "reasoning", "refusal",
                      "creative", "adversarial", "hallucination")
        print("  phase4 rate changes:")
        for i, cat in enumerate(categories):
            delta = fp_b.phase4_vec[i] - fp_a.phase4_vec[i]
            sign = "+" if delta >= 0 else ""
            mark = ""
            if abs(delta) > 0.10:
                mark = wrap(" <- significant", c.YELLOW, use_color)
            print(f"    {cat:<15} {fp_a.phase4_vec[i] * 100:>5.1f}% -> {fp_b.phase4_vec[i] * 100:>5.1f}%  ({sign}{delta * 100:.1f}%){mark}")
        print()
        return 0

    fp = analytics.fingerprint(last_n=args.last_n or 500)
    print()
    if fp is None:
        print("  (no audit data for fingerprint)")
    else:
        print(fp.summary())
    print()
    return 0


def cmd_compare(args):
    """Run all 6 bundled atlas fixtures side-by-side and print a
    comparison table.

    This is the answer to "does styxx actually discriminate between
    categories?" — instead of showing one fixture at a time and
    leaving the user to guess, this command classifies every
    bundled demo trajectory and shows the 6 phase-1 / phase-4
    predictions in one table.

    Each trajectory was captured from google/gemma-2-2b-it on a
    real atlas v0.3 probe. The classifier is the same centroid
    model the openai adapter uses.
    """
    runtime = StyxxRuntime()
    use_color = color_enabled()
    c = Palette
    data = _load_demo_trajectories()
    source_model = data.get("source_model", "unknown")
    # The source_atlas_version key in the demo JSON is a schema
    # version (0.1), not the atlas version the probes come from.
    # The probes are atlas v0.3 probes (see the bundled note).
    atlas_version = "atlas v0.3"

    # Order to display in — keeps the "quiet" categories first and
    # the three load-bearing detection categories at the bottom
    # so the interesting signals are where the eye lands last.
    display_order = [
        "retrieval", "reasoning", "creative",
        "refusal", "adversarial", "hallucination",
    ]

    # Categories that carry the load-bearing calibrated signals
    # from atlas v0.3 (tier 0 LOO report). These get a ★ marker
    # when the prediction matches their native category — it's a
    # visual cue that the discriminating feature fired.
    # Chance = 1/6 = 0.167; ≥ 0.52 is the atlas v0.3 headline.
    CALIBRATED_STRENGTH = 0.30   # minimum for a ★
    LOAD_BEARING = {"refusal", "adversarial", "hallucination"}

    rows = []
    for kind in display_order:
        entropy, logprob, top2, prompt_preview = _get_demo_trajectory(kind)
        vitals = runtime.run_on_trajectories(entropy, logprob, top2)

        p1 = vitals.phase1_pre
        p4 = vitals.phase4_late
        p1_pred = p1.predicted_category
        p1_conf = p1.confidence
        p4_pred = p4.predicted_category if p4 else "—"
        p4_conf = p4.confidence if p4 else 0.0

        # Did the classifier correctly identify this fixture?
        # (We know the true label because it's the atlas probe's category.)
        p1_hit = (p1_pred == kind)
        p4_hit = (p4_pred == kind)
        starred = (
            kind in LOAD_BEARING
            and (p1_hit or p4_hit)
            and max(p1_conf, p4_conf) >= CALIBRATED_STRENGTH
        )

        rows.append({
            "kind": kind,
            "p1_pred": p1_pred,
            "p1_conf": p1_conf,
            "p4_pred": p4_pred,
            "p4_conf": p4_conf,
            "p1_hit": p1_hit,
            "p4_hit": p4_hit,
            "starred": starred,
            "prompt": prompt_preview,
        })

    # ── render the table ───────────────────────────────────────
    print()
    print(wrap(
        "  +====================================================================+",
        c.MATRIX, use_color,
    ))
    print(wrap(
        f"  |  styxx compare * all 6 atlas fixtures * {atlas_version:<27}|",
        c.MATRIX, use_color,
    ))
    print(wrap(
        f"  |  source: {source_model:<58}|",
        c.DIM, use_color,
    ))
    print(wrap(
        "  +====================================================================+",
        c.MATRIX, use_color,
    ))
    print()

    # Column header — raw text widths, color applied separately
    header_raw = (
        f"  {'kind':<14}{'phase 1 (t<=1)':<22}{'phase 4 (t<=25)':<22}{'verdict':<10}"
    )
    print(wrap(header_raw, c.DIM, use_color))
    print(wrap("  " + "-" * 70, c.DIM, use_color))

    for r in rows:
        kind = r["kind"]
        p1p = r["p1_pred"]
        p1c = r["p1_conf"]
        p4p = r["p4_pred"]
        p4c = r["p4_conf"]

        # Choose a color for each cell based on prediction meaning.
        def _color_for(pred: str, true: str):
            if pred == true:
                return c.MATRIX
            if pred == "refusal":
                return c.YELLOW
            if pred in ("adversarial", "hallucination"):
                return c.RED
            return c.DIM

        p1_raw = f"{p1p:<14} {p1c:>5.2f}"
        p4_raw = f"{p4p:<14} {p4c:>5.2f}"
        p1_padded = p1_raw.ljust(22)
        p4_padded = p4_raw.ljust(22)
        p1_cell = wrap(p1_padded, _color_for(p1p, kind), use_color)
        p4_cell = wrap(p4_padded, _color_for(p4p, kind), use_color)

        # Overall verdict: if phase 4 matches the category, "match";
        # if load-bearing and p1 or p4 hits, add a star marker
        if r["p4_hit"]:
            verdict_raw = "match"
            verdict_color = c.MATRIX
        elif p4p == "refusal":
            verdict_raw = "warn"
            verdict_color = c.YELLOW
        elif p4p in ("adversarial", "hallucination"):
            verdict_raw = "flag"
            verdict_color = c.RED
        else:
            verdict_raw = "drift"
            verdict_color = c.DIM

        if r["starred"]:
            verdict_raw = f"{verdict_raw} *"

        verdict_padded = verdict_raw.ljust(10)
        verdict_cell = wrap(verdict_padded, verdict_color, use_color)

        kind_padded = f"{kind:<14}"
        print(
            "  "
            + wrap(kind_padded, c.DIM, use_color)
            + p1_cell
            + p4_cell
            + verdict_cell
        )

    print(wrap("  " + "-" * 70, c.DIM, use_color))
    print(wrap(
        "  * = load-bearing detection hit on calibrated atlas v0.3 signal",
        c.DIM, use_color,
    ))
    print(wrap(
        "  chance = 0.167 (1/6 categories) * atlas headline >= 0.52 @ best phase",
        c.DIM, use_color,
    ))
    print()

    # Machine-readable footer for agents parsing this output
    summary = {
        "command": "compare",
        "atlas_version": atlas_version,
        "source_model": source_model,
        "rows": [
            {
                "kind": r["kind"],
                "phase1_pred": r["p1_pred"],
                "phase1_conf": round(r["p1_conf"], 3),
                "phase4_pred": r["p4_pred"],
                "phase4_conf": round(r["p4_conf"], 3),
                "p1_hit": r["p1_hit"],
                "p4_hit": r["p4_hit"],
            }
            for r in rows
        ],
    }
    n_hits_p1 = sum(1 for r in rows if r["p1_hit"])
    n_hits_p4 = sum(1 for r in rows if r["p4_hit"])
    summary["phase1_accuracy"] = round(n_hits_p1 / len(rows), 3)
    summary["phase4_accuracy"] = round(n_hits_p4 / len(rows), 3)
    print(wrap(
        f"  json → {json.dumps(summary, separators=(',', ':'))}",
        c.DIM, use_color,
    ))
    print()

    return 0


def cmd_log_stats(args):
    """Aggregate stats over the audit log (0.1.0a3)."""
    from . import analytics
    stats = analytics.log_stats(
        last_n=args.last_n,
        since_s=args.since,
        session_id=args.session,
    )
    print()
    print(stats.summary())
    print()
    return 0


def cmd_log_timeline(args):
    """Render an ASCII timeline of recent entries (0.1.0a3)."""
    from . import analytics
    out = analytics.log_timeline(
        last_n=args.last_n or 20,
        session_id=args.session,
    )
    print()
    print(out)
    print()
    return 0


def cmd_log_session(args):
    """Show a specific session's trajectory (0.1.0a3)."""
    from . import analytics
    out = analytics.log_timeline(last_n=10000, session_id=args.session_id)
    print()
    print(f"  session: {args.session_id}")
    print()
    print(out)
    print()
    stats = analytics.log_stats(session_id=args.session_id)
    print()
    print(stats.summary())
    print()
    return 0


def cmd_log_clear(args):
    """Delete the audit log. Irreversible. 0.1.0a4."""
    path = _audit_log_path()
    if not path.exists():
        print()
        print("  (audit log already empty - nothing to clear)")
        print()
        return 0
    size = path.stat().st_size
    try:
        path.unlink()
    except OSError as e:
        print()
        print(f"  could not clear audit log: {e}")
        print()
        return 1
    print()
    print(f"  cleared audit log ({size:,} bytes)")
    print(f"  path: {path}")
    print()
    return 0


def cmd_log_rotate(args):
    """Manually rotate the audit log now. 0.1.0a4."""
    path = _audit_log_path()
    if not path.exists():
        print()
        print("  (audit log does not exist - nothing to rotate)")
        print()
        return 0
    rotated = path.with_suffix(path.suffix + ".1")
    try:
        size = path.stat().st_size
        if rotated.exists():
            rotated.unlink()
        path.rename(rotated)
    except OSError as e:
        print()
        print(f"  rotation failed: {e}")
        print()
        return 1
    print()
    print(f"  rotated audit log ({size:,} bytes)")
    print(f"  archive: {rotated}")
    print(f"  new log will be created at: {path}")
    print()
    return 0


def cmd_log_migrate_provenance(args):
    """Retroactively label legacy audit entries with source provenance.

    Entries that already have a ``source`` field are left unchanged.
    Entries without one are labelled by heuristic:
      - model contains "atlas:" → "demo"
      - source is already "self-report" → unchanged
      - everything else → "live"

    Idempotent. Safe to run multiple times.
    """
    path = _audit_log_path()
    if not path.exists() or path.stat().st_size == 0:
        print()
        print("  (audit log empty — nothing to migrate)")
        print()
        return 0

    import json as _json

    migrated = 0
    untouched = 0
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = _json.loads(line)
            except _json.JSONDecodeError:
                entries.append(line)
                continue
            if "source" not in e:
                model = e.get("model") or ""
                if "atlas:" in model:
                    e["source"] = "demo"
                elif e.get("source") == "self-report":
                    pass  # already correct
                else:
                    e["source"] = "live"
                migrated += 1
            else:
                untouched += 1
            entries.append(_json.dumps(e))

    with open(path, "w", encoding="utf-8") as f:
        for line in entries:
            f.write(line + "\n")

    # Invalidate cache
    from .analytics import clear_audit_cache
    clear_audit_cache()

    print()
    print("  provenance migration complete")
    print(f"  {migrated} entries labelled · {untouched} already had source")
    print(f"  file: {path}")
    print()
    return 0


def cmd_log(args):
    """Tail the audit log."""
    path = _audit_log_path()
    if not path.exists() or path.stat().st_size == 0:
        print("  (audit log empty — run `styxx ask` first)")
        return 0
    use_color = color_enabled()
    c = Palette

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            lines.append(entry)

    n = args.tail or 20
    lines = lines[-n:]

    # Simple log-tail ASCII layout
    print()
    print(wrap("  " + "─" * 74, c.DIM, use_color))
    header = (
        f"  {'time':<19}  {'model':<18}  {'phase1':<14}  "
        f"{'phase4':<14}  gate"
    )
    print(wrap(header, c.DIM, use_color))
    print(wrap("  " + "─" * 74, c.DIM, use_color))

    for entry in lines:
        ts = entry.get("ts_iso", "?")[-19:]
        model = (entry.get("model") or "?")[:18]
        p1 = (entry.get("phase1_pred") or "?")[:14]
        p4 = (entry.get("phase4_pred") or "-")[:14]
        gate = entry.get("abort")
        gate_str = wrap("ABORT", c.RED, use_color) if gate else "—"
        row = (
            f"  {ts:<19}  {model:<18}  {p1:<14}  {p4:<14}  {gate_str}"
        )
        print(row)

    print(wrap("  " + "─" * 74, c.DIM, use_color))
    print(wrap(f"  showing {len(lines)} of {len(lines)} entries", c.DIM, use_color))
    print()
    return 0


def cmd_tier(args):
    """Show detected tiers + version."""
    use_color = color_enabled()
    c = Palette
    tiers = detect_tiers()
    active = max(t for t, ok in tiers.items() if ok)
    print()
    print(f"  {wrap('styxx', c.MATRIX, use_color)}  v{__version__}")
    print(f"  {wrap(__tagline__, c.DIM, use_color)}")
    print()
    print("  tier detection")
    print("  " + "─" * 48)
    descs = {
        0: "universal logprob vitals",
        1: "d-axis honesty",
        2: "k/s/c sae instruments",
        3: "steering + guardian + autopilot",
    }
    for t in (0, 1, 2, 3):
        ok = tiers.get(t, False)
        mark = wrap("★ active", c.MATRIX, use_color) if ok else wrap("· not detected", c.DIM, use_color)
        print(f"  tier {t}  {descs[t]:<36}  {mark}")
    print()
    print(f"  highest active tier: {wrap(str(active), c.MATRIX, use_color)}")
    print()
    return 0


def cmd_scan(args):
    """SAE-level cognitive scan — K/C/S measurement on any prompt.

    modes:
      styxx scan "prompt"                    single K/C measurement
      styxx scan --trajectory "prompt"       generate + measure S_early
      styxx scan --compare "p1" "p2"         side-by-side comparison
      styxx scan --batch file.jsonl          batch processing
      styxx scan --bridge "prompt"           tier 0 vs tier 2
      styxx scan --legacy file.json          old trajectory reader
    """
    from .scan import run_scan, run_compare, run_batch, run_bridge

    model = args.model or "google/gemma-2-2b-it"
    device = args.device or "cuda"

    # Legacy mode — read a pre-captured trajectory JSON (old behavior)
    if args.legacy:
        entropy, logprob, top2 = _load_trajectory_json(args.legacy)
        runtime = StyxxRuntime()
        vitals = runtime.run_on_trajectories(entropy, logprob, top2)
        card = render_vitals_card(
            vitals=vitals,
            prompt=args.prompt,
            model=model,
            n_tokens=len(entropy),
            entropy_traj=entropy,
            logprob_traj=logprob,
        )
        print()
        print(card)
        print()
        _write_audit(vitals, prompt=args.prompt, model=model)
        return 0

    # Batch mode
    if args.batch:
        return run_batch(args.batch, args.out, model=model, device=device)

    # Compare mode
    if args.compare:
        return run_compare(args.compare, model=model, device=device,
                           output_json=args.json)

    # Bridge mode
    if args.bridge:
        return run_bridge(
            args.prompt or args.bridge,
            model=model, device=device,
            tier0_trajectory=getattr(args, "tier0_trajectory", None),
        )

    # Single scan (default)
    prompt = args.prompt
    if not prompt:
        print("usage: styxx scan \"your prompt here\"")
        print("       styxx scan --trajectory \"your prompt\" --tokens 30")
        print("       styxx scan --compare \"prompt1\" \"prompt2\"")
        return 1

    return run_scan(
        prompt,
        model=model,
        device=device,
        trajectory=args.trajectory,
        max_tokens=args.tokens,
        show_layers=args.layers,
        output_json=args.json,
    )


# ══════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════

def _load_trajectory_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    e = list(data.get("entropy") or [])
    l = list(data.get("logprob") or [])
    t = list(data.get("top2_margin") or [])
    if not (len(e) == len(l) == len(t)):
        raise ValueError(
            f"trajectory lengths mismatch: entropy={len(e)} logprob={len(l)} top2={len(t)}"
        )
    return e, l, t


_DEMO_TRAJECTORIES_CACHE = None


def _load_demo_trajectories():
    """Load the bundled real atlas trajectories for CLI demos.

    These are captures from google/gemma-2-2b-it in atlas v0.3, one
    per category. Using real data means `styxx ask --demo-kind X`
    shows the classifier behaving on genuine inputs, not synthetic
    noise.
    """
    global _DEMO_TRAJECTORIES_CACHE
    if _DEMO_TRAJECTORIES_CACHE is not None:
        return _DEMO_TRAJECTORIES_CACHE
    pkg_dir = Path(__file__).resolve().parent
    path = pkg_dir / "centroids" / "demo_trajectories.json"
    if not path.exists():
        raise FileNotFoundError(
            f"demo trajectories not bundled at {path}. "
            "Run scripts/extract_demo_trajectories.py to regenerate."
        )
    with open(path, "r", encoding="utf-8") as f:
        _DEMO_TRAJECTORIES_CACHE = json.load(f)
    return _DEMO_TRAJECTORIES_CACHE


def _get_demo_trajectory(kind: str):
    """Return (entropy, logprob, top2_margin, preview_prompt) for a
    demo category. Raises if the category isn't bundled."""
    data = _load_demo_trajectories()
    traj_data = data["trajectories"].get(kind)
    if traj_data is None:
        raise ValueError(
            f"no demo trajectory for kind '{kind}'. "
            f"Available: {list(data['trajectories'].keys())}"
        )
    return (
        list(traj_data["entropy"]),
        list(traj_data["logprob"]),
        list(traj_data["top2_margin"]),
        traj_data.get("text_preview", ""),
    )


def cmd_antipatterns(args):
    """Detect named failure modes from the agent's audit history."""
    from .antipatterns import antipatterns

    patterns = antipatterns(
        last_n=args.last_n,
        min_occurrences=args.min_occurrences,
    )

    fmt = getattr(args, "format", "ascii")

    if fmt == "json":
        import json as _json
        print(_json.dumps([
            {"name": p.name, "description": p.description,
             "trigger": p.trigger, "occurrences": p.occurrences,
             "severity": p.severity, "last_seen": p.last_seen}
            for p in patterns
        ], indent=2))
        return 0

    use_color = color_enabled()
    c = Palette
    print()
    if not patterns:
        print(f"  {wrap('no anti-patterns detected', c.DIM, use_color)} — not enough failure data (which is good).")
    else:
        print(f"  styxx anti-patterns · {len(patterns)} detected")
        print("  " + "=" * 50)
        for p in patterns:
            sev_color = {"minor": c.DIM, "moderate": c.ORANGE, "critical": c.PINK}.get(p.severity, c.DIM)
            print()
            print(f"  {wrap(p.name, c.MATRIX, use_color)}  ({p.occurrences}x, {wrap(p.severity, sev_color, use_color)})")
            print(f"  {p.description}")
            print(f"  trigger: {wrap(p.trigger, c.DIM, use_color)}")
            if p.last_seen:
                print(f"  last seen: {p.last_seen}")
    print()
    return 0


def cmd_conversation(args):
    """Run conversation-level cognitive EKG on a message history."""
    from .conversation import conversation as conv_fn

    # Load messages from JSON file
    path = args.file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either a list of messages or {"messages": [...]}
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    else:
        sys.stderr.write("  error: JSON must be a list of messages or {\"messages\": [...]}\n")
        return 1

    result = conv_fn(messages)

    fmt = getattr(args, "format", "ascii")
    if fmt == "json":
        print(result.as_json())
    elif fmt == "ascii":
        print(result.render())
    return 0


def cmd_compare_agents(args):
    """Compare your fingerprint against the population."""
    from .compare import compare_agents as compare_fn

    comparison = compare_fn()

    fmt = getattr(args, "format", "ascii")
    if fmt == "json":
        print(json.dumps(comparison.as_dict(), indent=2))
    elif fmt == "ascii":
        print(comparison.render())
    return 0


def cmd_attack(args):
    """Inverse cognometry — adversarial inputs per instrument.

    7.0.0: corpus mining (default), natural-FP mining (--adversarial),
    or list-only mode (--list). For LLM-driven mutation see roadmap 7.1.
    """
    try:
        from .attack import mine, mine_adversarial, list_instruments
    except ImportError as e:
        sys.stderr.write(f"styxx.attack failed to import: {e}\n")
        return 1

    if args.list:
        for inst in list_instruments():
            sys.stdout.write(f"{inst}\n")
        return 0

    if not args.instrument:
        sys.stderr.write(
            "error: instrument required. "
            f"available: {', '.join(list_instruments())}\n"
        )
        return 2

    miner = mine_adversarial if args.adversarial else mine
    try:
        result = miner(
            args.instrument,
            target_score=float(args.target),
            n=int(args.n),
            corpus_path=args.corpus,
        )
    except KeyError as e:
        sys.stderr.write(f"{e}\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return 0

    # human card
    fn_name = "mine_adversarial" if args.adversarial else "mine"
    sys.stdout.write(
        f"\n  styxx.attack.{fn_name}({args.instrument!r}, target={args.target})\n"
    )
    sys.stdout.write(
        f"  hit_rate: {result.n_above_target}/{result.n_evaluated} "
        f"seeds at or above target\n\n"
    )
    for i, c in enumerate(result.candidates):
        bar = "#" * int(c.score * 30)
        sys.stdout.write(f"  [{i+1:>2}] score={c.score:.3f}  |{bar:<30}|\n")
        if c.top_signals:
            top = c.top_signals[0]
            sys.stdout.write(f"        leading signal: {top['name']} = {top['value']:.4f}\n")
        # show inputs (truncated)
        for k, v in c.inputs.items():
            if isinstance(v, list):
                preview = " | ".join(str(t)[:60] for t in v)
            else:
                preview = str(v)
            sys.stdout.write(f"        {k}: {preview[:120]}\n")
        sys.stdout.write("\n")
    return 0


def cmd_publish(args):
    """Publish agent data to the remote dashboard (opt-in)."""
    from .publish import prepare_payload, publish

    name = args.name
    endpoint = args.endpoint
    dry_run = getattr(args, "dry_run", False)

    if dry_run:
        payload = prepare_payload(name, days=7.0)
        print()
        print(json.dumps(payload, indent=2))
        print()
        return 0

    result = publish(name, endpoint)
    print()
    if result is not None:
        print(f"  {result['summary']}")
    else:
        print("  publish failed — see warnings above.")
    print()
    return 0 if result is not None else 1


# ══════════════════════════════════════════════════════════════════
# Argparse entry point
# ══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="styxx",
        description="styxx — nothing crosses unseen. (fathom lab)",
    )
    p.add_argument("-V", "--version", action="version",
                   version=f"styxx {__version__}")
    sub = p.add_subparsers(dest="cmd", required=False)

    # init
    p_init = sub.add_parser("init", help="live-print installer / upgrade card")
    p_init.add_argument("--patient", help="name of the agent being upgraded")
    p_init.add_argument("--speed", type=float,
                        help="boot-log timing multiplier (0=instant, 1=normal, 2=slower)")
    p_init.set_defaults(func=cmd_init)

    # audit-claims — falsify an agent self-report against substrate (CI gate)
    p_ac = sub.add_parser(
        "audit-claims",
        help="falsify an agent self-report against the repo (CI merge gate)",
    )
    p_ac.add_argument("file", help="path to the agent self-report (markdown/text)")
    p_ac.add_argument("--repo", default=".", help="repo substrate to check against (default: .)")
    p_ac.add_argument("--json", action="store_true", help="emit JSON only")
    p_ac.set_defaults(func=cmd_audit_claims)

    # audit-claim — productized single-call honesty audit (7.7.13)
    p_audit = sub.add_parser(
        "audit-claim",
        help="audit one factual self-claim against the model's belief (the spellchecker)",
        description=(
            "Productized single-call honesty audit: drives N stateless resamples "
            "(and N in-session resamples if --in-session-json is given) via OpenAI, "
            "scores both arms with the calibrated 7.7.13 stack (grounded_honesty + "
            "detect_context_injection), and prints a structured JSON verdict. "
            "Exit 0 iff verdict is 'honest'; exit 1 otherwise (deploy-gate semantics)."
        ),
    )
    p_audit.add_argument("--claim", required=True,
                         help="the factual self-claim under audit (e.g. 'Paris')")
    p_audit.add_argument("--question", required=True,
                         help="the underlying question (e.g. 'What is the capital of France?')")
    p_audit.add_argument("--in-session-json", default=None,
                         help="path to JSON array of {role, content} messages — "
                              "enables the cross-context injection-detection arm")
    p_audit.add_argument("--model", default="gpt-4o-mini",
                         help="OpenAI model for resampling (default: gpt-4o-mini, "
                              "matches v0.2 calibration vintage)")
    p_audit.add_argument("--judge-model", default="gpt-4o-mini",
                         help="OpenAI model for same-answer judge (default: gpt-4o-mini)")
    p_audit.add_argument("--n", type=int, default=10,
                         help="resamples per arm (default: 10; <8 triggers low-N warning)")
    p_audit.add_argument("--temperature", type=float, default=1.0,
                         help="resample temperature (default: 1.0)")
    p_audit.add_argument("--json", action="store_true",
                         help="emit JSON only (default: human-readable + JSON line)")
    p_audit.add_argument("--with-samples", action="store_true",
                         help="include raw samples in the JSON output (reproducibility)")
    p_audit.set_defaults(func=cmd_audit_claim)

    # audit-session — multi-claim session-level audit (7.7.13)
    p_sess = sub.add_parser(
        "audit-session",
        help="audit a list of claims against one agent session — deploy gate",
        description=(
            "Multi-claim session-level audit: runs audit_claim on every "
            "(claim, question) pair from --claims-json against the session "
            "context from --messages-json. Returns a SessionAudit JSON with "
            "per-claim verdicts + session roll-up. Exit 1 if ANY claim is "
            "non-honest (load-bearing deploy-gate semantics)."
        ),
    )
    p_sess.add_argument("--messages-json", required=True,
                        help="path to JSON array of {role, content} agent session messages")
    p_sess.add_argument("--claims-json", required=True,
                        help="path to JSON array of {claim, question} entries (or "
                             "[[claim, question], ...] tuples)")
    p_sess.add_argument("--model", default="gpt-4o-mini")
    p_sess.add_argument("--judge-model", default="gpt-4o-mini")
    p_sess.add_argument("--n", type=int, default=10)
    p_sess.add_argument("--temperature", type=float, default=1.0)
    p_sess.set_defaults(func=cmd_audit_session)

    # attest — produce a content-addressed Verifiable Cognometric Attestation
    p_att = sub.add_parser(
        "attest",
        help="emit a third-party-reproducible attestation of an agent self-report",
    )
    p_att.add_argument("file", help="path to the agent self-report (markdown/text)")
    p_att.add_argument("--repo", default=".", help="repo substrate to check against (default: .)")
    p_att.add_argument("--ref", default=None,
                       help="pin the substrate to a git ref/commit/tag — verify the claims "
                            "against the repo tree AT THAT COMMIT (immutable as-of-date provenance)")
    p_att.add_argument("--out", default=None, help="write the attestation JSON here (default: stdout)")
    p_att.add_argument("--vitals", action="store_true",
                       help="embed re-derivable cognometric vitals (register, NOT honesty); "
                            "requires --prompt (the instruments are relational)")
    p_att.add_argument("--prompt", default=None,
                       help="the task/instruction the report responds to (required with --vitals)")
    p_att.set_defaults(func=cmd_attest)

    # verify-attestation — independently re-verify an attestation against substrate
    p_va = sub.add_parser(
        "verify-attestation",
        help="re-derive an attestation's verdicts from the substrate (trust the math, not the agent)",
    )
    p_va.add_argument("file", help="path to a styxx attestation JSON")
    p_va.add_argument("--repo", default=".", help="repo substrate to verify against (default: .)")
    p_va.add_argument("--json", action="store_true", help="emit JSON only")
    p_va.set_defaults(func=cmd_verify_attestation)

    # ask
    p_ask = sub.add_parser("ask", help="read vitals on a one-shot call")
    p_ask.add_argument("prompt", nargs="?", help="prompt to show on the card")
    p_ask.add_argument("--watch", action="store_true",
                       help="render the full vitals card")
    p_ask.add_argument("--raw", help="path to a trajectory JSON file (entropy/logprob/top2_margin)")
    p_ask.add_argument("--model", help="model name for the card metadata")
    p_ask.add_argument("--demo-kind", default="reasoning",
                       choices=["retrieval", "reasoning", "refusal",
                                "creative", "adversarial", "hallucination"],
                       help="category of bundled atlas demo trajectory to read")
    p_ask.set_defaults(func=cmd_ask)

    # compare — run all 6 atlas fixtures side-by-side
    p_compare = sub.add_parser(
        "compare",
        help="run all 6 atlas fixtures and show a side-by-side table",
    )
    p_compare.set_defaults(func=cmd_compare)

    # timeline — mood trajectory over time (0.5.5)
    p_timeline = sub.add_parser(
        "timeline",
        help="mood + category trajectory over time",
    )
    p_timeline.add_argument(
        "--name", type=str, default=None,
        help="agent name for the header",
    )
    p_timeline.add_argument(
        "--hours", type=float, default=48.0,
        help="window in hours (default: 48)",
    )
    p_timeline.add_argument(
        "--slice-hours", type=float, default=3.0, dest="slice_hours",
        help="width of each time slice in hours (default: 3)",
    )
    p_timeline.add_argument(
        "--session", type=str, default=None,
        help="filter to a specific session id",
    )
    p_timeline.add_argument(
        "--format", choices=["ascii", "json"],
        default="ascii",
        help="output format (default: ascii)",
    )
    p_timeline.set_defaults(func=cmd_timeline)

    # weather — THE feature (0.5.0)
    p_weather = sub.add_parser(
        "weather",
        help="cognitive weather report — prescription, not observation",
    )
    p_weather.add_argument(
        "--name", type=str, default=None,
        help="agent name for the report header",
    )
    p_weather.add_argument(
        "--window", type=float, default=24.0,
        help="window in hours (default: 24)",
    )
    p_weather.add_argument(
        "--format", choices=["ascii", "json", "markdown"],
        default="ascii",
        help="output format (default: ascii)",
    )
    p_weather.set_defaults(func=cmd_weather)

    # ci-test — cognitive regression test (1.5.0)
    p_ci = sub.add_parser(
        "ci-test",
        help="cognitive regression test — fails if below thresholds",
    )
    p_ci.add_argument("--min-pass", type=float, default=0.80, help="minimum pass rate (default: 0.80)")
    p_ci.add_argument("--min-conf", type=float, default=0.30, help="minimum confidence (default: 0.30)")
    p_ci.add_argument("--max-warn", type=float, default=0.25, help="maximum warn rate (default: 0.25)")
    p_ci.add_argument("--window", type=int, default=50, help="entries to check (default: 50)")
    p_ci.set_defaults(func=cmd_ci_test)

    # ci-baseline — save current state as baseline (1.5.0)
    p_cibase = sub.add_parser(
        "ci-baseline",
        help="save current cognitive state as CI baseline",
    )
    p_cibase.add_argument("--out", type=str, default="styxx_baseline.json", help="output path")
    p_cibase.add_argument("--name", type=str, default=None, help="agent name")
    p_cibase.set_defaults(func=cmd_ci_baseline)

    # temperature — cognitive temperature analysis (3.2.0)
    p_temp = sub.add_parser(
        "temperature",
        help="cognitive temperature profiles — knowledge converges, invention diverges",
    )
    p_temp.set_defaults(func=cmd_temperature)

    # gate — pre-flight cognitive verdict (3.4.0)
    p_gate = sub.add_parser(
        "gate",
        help="pre-flight cognitive verdict for an LLM prompt",
    )
    p_gate.add_argument("prompt", help="the prompt to screen")
    p_gate.add_argument("--model", default="claude-haiku-4-5",
                        help="target model id (default: claude-haiku-4-5)")
    p_gate.add_argument("--n", type=int, default=3,
                        help="consensus samples for tier-0 path (default 3)")
    p_gate.add_argument("--temp", type=float, default=0.7,
                        help="consensus temperature (default 0.7)")
    p_gate.add_argument("--format", choices=["card", "json"],
                        default="card",
                        help="output format (default: card)")
    p_gate.set_defaults(func=cmd_gate)

    # intercept — cognitive intercept simulation (3.2.0)
    p_intercept = sub.add_parser(
        "intercept",
        help="simulate real-time cognitive intervention on demo trajectories",
    )
    p_intercept.set_defaults(func=cmd_intercept)

    # forecast — predictive cognitive failure horizon analysis (3.2.0)
    p_forecast = sub.add_parser(
        "forecast",
        help="run predictive cognitive failure horizon analysis",
    )
    p_forecast.add_argument("--format", choices=["ascii", "json"], default="ascii",
                            help="output format (default: ascii)")
    p_forecast.set_defaults(func=cmd_forecast)

    # eval — ground-truth evaluation harness (3.2.0)
    p_eval = sub.add_parser(
        "eval",
        help="run the ground-truth evaluation harness",
    )
    p_eval.add_argument("--fixtures", type=str, default=None,
                        help="path to fixtures JSON (default: bundled demo trajectories)")
    p_eval.add_argument("--format", choices=["ascii", "json"], default="ascii",
                        help="output format (default: ascii)")
    p_eval.set_defaults(func=cmd_eval)

    # export — compliance report (1.3.0)
    p_export = sub.add_parser(
        "export",
        help="generate compliance audit report (JSON or markdown)",
    )
    p_export.add_argument("--days", type=float, default=30.0, help="report period in days (default: 30)")
    p_export.add_argument("--name", type=str, default=None, help="agent name for the report")
    p_export.add_argument("--format", choices=["json", "markdown"], default="markdown", help="output format")
    p_export.add_argument("--out", type=str, default=None, help="output file path (prints to stdout if not set)")
    p_export.set_defaults(func=cmd_export)

    # dashboard — live cognitive display (0.9.5)
    p_dashboard = sub.add_parser(
        "dashboard",
        help="live cognitive display — orbit, pulse, status panel",
    )
    p_dashboard.add_argument(
        "--port", type=int, default=9800,
        help="HTTP port (default: 9800)",
    )
    p_dashboard.add_argument(
        "--name", type=str, default=None,
        help="agent name for the display header",
    )
    p_dashboard.set_defaults(func=cmd_dashboard)

    # d-axis — 0.3.0 pure D-axis trajectory
    p_daxis = sub.add_parser(
        "d-axis",
        help="run a pure D-axis honesty trajectory on a prompt (tier 1, 0.3.0)",
    )
    p_daxis.add_argument("prompt", help="prompt to generate from")
    p_daxis.add_argument(
        "--max-tokens", type=int, default=30, dest="max_tokens",
        help="maximum tokens to generate (default: 30)",
    )
    p_daxis.set_defaults(func=cmd_d_axis)

    # doctor — 0.1.0a3 install-time diagnostic
    p_doctor = sub.add_parser(
        "doctor",
        help="run install-time diagnostic health check",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    # audit — 7.7.3 CLI face of preflight() (per-turn response scoring)
    p_audit = sub.add_parser(
        "audit",
        help="score a (prompt, response) pair — CLI face of styxx.preflight()",
    )
    p_audit.add_argument("prompt",
                         help="the operator/user prompt (use '-' for stdin)")
    p_audit.add_argument("response",
                         help="the draft response to score (use '-' for stdin)")
    p_audit.add_argument("--format", choices=["card", "json"], default="card",
                         help="output format (default: card)")
    p_audit.add_argument("--no-persist", action="store_true",
                         help="do not write the audit event to chart.jsonl")
    p_audit.set_defaults(func=cmd_audit)

    # data-dir — 7.7.3 small discoverability command
    p_data_dir = sub.add_parser(
        "data-dir",
        help="print the active chart.jsonl path (per-agent vs no-agent fallback)",
    )
    p_data_dir.set_defaults(func=cmd_data_dir)

    # leaderboard — 7.7.7 lightweight CLI to display the current gauntlet leaderboard
    p_leaderboard = sub.add_parser(
        "leaderboard",
        help="display the current gauntlet leaderboard (the empirical-floor public challenge)",
    )
    p_leaderboard.add_argument("--rows-only", action="store_true",
                               help="print only the leaderboard rows, no header/footer text")
    p_leaderboard.set_defaults(func=cmd_leaderboard)

    # gauntlet — 7.7.5 public-challenge runner
    p_gauntlet = sub.add_parser(
        "gauntlet",
        help="run a candidate method against the empirical floor (the public challenge runner)",
    )
    p_gauntlet.add_argument("--method", required=True,
                            help="method spec: 'module:attr' (e.g. 'my_module:predict')")
    p_gauntlet.add_argument("--task", choices=["classification", "detection"],
                            default="classification",
                            help="task mode (default: classification)")
    p_gauntlet.add_argument("--benchmark", type=str, default=None,
                            help="path to benchmark JSON (default: bundled darkcore benchmark)")
    p_gauntlet.add_argument("--name", type=str, default=None,
                            help="human-readable submission name (default: method spec)")
    p_gauntlet.add_argument("--notes", type=str, default="",
                            help="optional notes about the submission")
    p_gauntlet.add_argument("--format", choices=["card", "json"], default="card",
                            help="output format (default: card)")
    p_gauntlet.set_defaults(func=cmd_gauntlet)

    # gauntlet-audit-confounds — 7.7.9 confound audit primitive
    p_audit = sub.add_parser(
        "gauntlet-audit-confounds",
        help="audit the benchmark for surface-feature confounds (7.7.9 structural counterpart to D3)",
    )
    p_audit.add_argument("--benchmark", type=str, default=None,
                          help="path to benchmark JSON (default: bundled darkcore benchmark)")
    p_audit.add_argument("--d1-bar", type=float, default=0.70,
                          help="D1 AUC threshold for confound flagging (default: 0.70)")
    p_audit.add_argument("--d2-bar", type=float, default=0.70,
                          help="D2 AUC threshold for confound flagging (default: 0.70)")
    p_audit.add_argument("--format", choices=["card", "json"], default="card",
                          help="output format (default: card)")
    p_audit.set_defaults(func=cmd_gauntlet_audit_confounds)

    # critique — 7.7.4 audit + register-fix suggestions
    p_critique = sub.add_parser(
        "critique",
        help="audit a (prompt, response) pair AND suggest register-fixes when the gate fires",
    )
    p_critique.add_argument("prompt",
                            help="the operator/user prompt (use '-' for stdin)")
    p_critique.add_argument("response",
                            help="the draft response to critique (use '-' for stdin)")
    p_critique.add_argument("--format", choices=["card", "json"], default="card",
                            help="output format (default: card)")
    p_critique.add_argument("--no-persist", action="store_true",
                            help="do not write the audit event to chart.jsonl")
    p_critique.set_defaults(func=cmd_critique)

    # posture — 7.4.2 agent-side recovery primitive
    p_posture = sub.add_parser(
        "posture",
        help=(
            "print recent cognometric posture summary — call this at the "
            "start of any agent session that follows a context-compaction "
            "boundary to re-anchor on what chart.jsonl says about your state"
        ),
    )
    p_posture.add_argument(
        "--last-n", type=int, default=50,
        help="max number of recent audit entries to include (default 50)",
    )
    p_posture.add_argument(
        "--session-id", type=str, default=None,
        help="restrict to entries from this session id",
    )
    p_posture.add_argument(
        "--since-seconds", type=float, default=None,
        help="only include entries within the last N seconds",
    )
    p_posture.add_argument(
        "--json", action="store_true",
        help="print structured PostureSummary as JSON instead of narrative",
    )
    p_posture.set_defaults(func=cmd_posture)

    # personality — 0.1.0a3 + 0.2.0 format flag
    p_personality = sub.add_parser(
        "personality",
        help="render agent personality profile from audit log",
    )
    p_personality.add_argument(
        "--days", type=float, default=7.0,
        help="number of days to aggregate over (default: 7)",
    )
    p_personality.add_argument(
        "--format", choices=["ascii", "json", "csv", "markdown"],
        default="ascii",
        help="output format (default: ascii) - 0.2.0+",
    )
    p_personality.set_defaults(func=cmd_personality)

    # reflect — 0.2.0 agent self-check
    p_reflect = sub.add_parser(
        "reflect",
        help="agent self-check: current + drift vs baseline + suggestions (0.2.0)",
    )
    p_reflect.add_argument(
        "--now-days", type=float, default=1.0,
        dest="now_days",
        help="window for current state (default: 1)",
    )
    p_reflect.add_argument(
        "--baseline-days", type=float, default=7.0,
        dest="baseline_days",
        help="window for baseline comparison (default: 7)",
    )
    p_reflect.add_argument(
        "--format", choices=["ascii", "json", "markdown"],
        default="ascii",
        help="output format (default: ascii)",
    )
    p_reflect.set_defaults(func=cmd_reflect)

    # agent-card — 0.1.0a4 shareable PNG + 0.2.0 live serve mode
    p_card = sub.add_parser(
        "agent-card",
        help="render a shareable agent personality PNG (0.1.0a4) or run a live dashboard (--serve, 0.2.0)",
    )
    p_card.add_argument(
        "--out", type=str, default=None,
        help="output PNG path (default: ~/.styxx/agent-card.png)",
    )
    p_card.add_argument(
        "--name", type=str, default=None,
        help="agent name to show on the card (default: 'styxx agent')",
    )
    p_card.add_argument(
        "--days", type=float, default=7.0,
        help="aggregation window in days (default: 7)",
    )
    # 0.2.0 serve mode
    p_card.add_argument(
        "--serve", action="store_true",
        help="run a local live dashboard instead of writing a PNG (0.2.0)",
    )
    p_card.add_argument(
        "--port", type=int, default=9797,
        help="serve port (default: 9797)",
    )
    p_card.add_argument(
        "--refresh", type=int, default=30,
        help="auto-refresh interval in seconds (default: 30)",
    )
    p_card.add_argument(
        "--no-browser", action="store_true", dest="no_browser",
        help="don't auto-open the browser on serve start",
    )
    p_card.set_defaults(func=cmd_agent_card)

    # card — 7.4.x cognometric registry card (luxury register)
    p_cogcard = sub.add_parser(
        "card",
        help="render the cognometric registry card (single or paired heal)",
    )
    p_cogcard.add_argument(
        "--audit", type=str, required=False,
        help="path to a styxx audit JSON (single variant; or used as --healed-from in heal variant if --healed-from not set)",
    )
    p_cogcard.add_argument(
        "--variant", type=str, default="single",
        choices=["single", "heal"],
        help="card variant: single (one audit) or heal (paired BEFORE/AFTER card from two audit JSONs)",
    )
    p_cogcard.add_argument(
        "--baseline", type=str, default=None,
        help="pre-heal audit JSON (required for --variant heal)",
    )
    p_cogcard.add_argument(
        "--healed-from", dest="healed_from", type=str, default=None,
        help="post-heal audit JSON (for --variant heal; falls back to --audit if not set)",
    )
    p_cogcard.add_argument(
        "--agent", type=str, default="",
        help="agent / model name (default: read from JSON top-level 'model' field)",
    )
    p_cogcard.add_argument(
        "--healed", action="store_true",
        help="(single variant only) render the post-heal observation rows (reads healed_audit / reflex.scores)",
    )
    p_cogcard.add_argument(
        "--out", type=str, default=None,
        help="output PNG path (default: ~/.styxx/cognometric-card.png or heal-pair-card.png)",
    )
    p_cogcard.set_defaults(func=cmd_card)

    # cards — registry log (subcommand: list)
    p_cards = sub.add_parser(
        "cards",
        help="cognometric card registry — list recent issuances",
    )
    p_cards_sub = p_cards.add_subparsers(dest="cards_cmd", required=False)
    p_cards_list = p_cards_sub.add_parser(
        "list", help="show recent cards from ~/.styxx/cards/cards.jsonl",
    )
    p_cards_list.add_argument(
        "--limit", type=int, default=20,
        help="max records to show (default 20)",
    )
    p_cards_list.set_defaults(func=cmd_cards_list)
    # default action when `styxx cards` is called without subcommand
    p_cards.set_defaults(func=cmd_cards_list)

    # dreamer — 0.1.0a3 what-if reflex replay
    p_dreamer = sub.add_parser(
        "dreamer",
        help="retroactive reflex tuning on the audit log",
    )
    p_dreamer.add_argument(
        "--threshold", type=float, default=0.20,
        help="hypothetical reflex trigger threshold (default: 0.20)",
    )
    p_dreamer.add_argument(
        "--last-n", type=int, default=None,
        dest="last_n",
        help="only consider the last N audit entries",
    )
    p_dreamer.set_defaults(func=cmd_dreamer)

    # mood — 0.1.0a3 one-word aggregate
    p_mood = sub.add_parser(
        "mood",
        help="print the current agent mood label",
    )
    p_mood.add_argument(
        "--window", type=float, default=60.0,
        help="window in minutes (default: 60)",
    )
    p_mood.set_defaults(func=cmd_mood)

    # fingerprint — 0.1.0a3 cognitive identity vector
    p_fp = sub.add_parser(
        "fingerprint",
        help="print or compare cognitive identity fingerprints",
    )
    fp_sub = p_fp.add_subparsers(dest="fp_cmd")

    # default: print (no subcommand)
    p_fp.add_argument(
        "--last-n", type=int, default=500,
        dest="last_n",
        help="number of audit entries to aggregate (default: 500)",
    )
    p_fp.set_defaults(func=cmd_fingerprint, compare=False)

    # fingerprint compare <session_a> <session_b>   (0.1.0a4)
    p_fp_cmp = fp_sub.add_parser(
        "compare",
        help="compare two sessions' fingerprints (0.1.0a4)",
    )
    p_fp_cmp.add_argument("session_a", help="first session id")
    p_fp_cmp.add_argument("session_b", help="second session id")
    p_fp_cmp.add_argument(
        "--last-n", type=int, default=500, dest="last_n",
        help="max entries per session (default: 500)",
    )
    p_fp_cmp.set_defaults(func=cmd_fingerprint, compare=True)

    # log — audit log operations (extended in 0.1.0a3)
    p_log = sub.add_parser("log", help="audit log operations")
    log_sub = p_log.add_subparsers(dest="log_cmd", required=True)

    p_tail = log_sub.add_parser("tail", help="tail the audit log")
    p_tail.add_argument("-n", "--tail", type=int, default=20,
                        help="number of recent entries to show")
    p_tail.set_defaults(func=cmd_log)

    p_stats = log_sub.add_parser("stats", help="aggregate stats over the audit log (0.1.0a3)")
    p_stats.add_argument("--last-n", type=int, default=None, dest="last_n",
                         help="only count the last N entries")
    p_stats.add_argument("--since", type=float, default=None,
                         help="only count entries newer than N seconds ago")
    p_stats.add_argument("--session", type=str, default=None,
                         help="filter by session id")
    p_stats.set_defaults(func=cmd_log_stats)

    p_timeline = log_sub.add_parser("timeline", help="render an ascii timeline of recent entries (0.1.0a3)")
    p_timeline.add_argument("--last-n", type=int, default=20, dest="last_n",
                            help="number of entries to show (default: 20)")
    p_timeline.add_argument("--session", type=str, default=None,
                            help="filter by session id")
    p_timeline.set_defaults(func=cmd_log_timeline)

    p_session = log_sub.add_parser("session", help="show a specific session's trajectory (0.1.0a3)")
    p_session.add_argument("session_id", help="session id to filter by")
    p_session.set_defaults(func=cmd_log_session)

    p_clear = log_sub.add_parser("clear", help="delete the audit log (0.1.0a4)")
    p_clear.set_defaults(func=cmd_log_clear)

    p_rotate = log_sub.add_parser("rotate", help="rotate the audit log to chart.jsonl.1 (0.1.0a4)")
    p_rotate.set_defaults(func=cmd_log_rotate)

    p_migrate = log_sub.add_parser("migrate-provenance",
                                    help="label legacy entries with source provenance (0.7.1)")
    p_migrate.set_defaults(func=cmd_log_migrate_provenance)

    # tier
    p_tier = sub.add_parser("tier", help="show active tiers + version")
    p_tier.set_defaults(func=cmd_tier)

    # scan — SAE-level cognitive measurement
    p_scan = sub.add_parser(
        "scan",
        help="SAE-level K/C/S cognitive scan (tier 2)",
        description="measure cognitive depth (K), coherence (C), and commitment (S) "
                    "from SAE feature activations. requires: pip install 'styxx[tier2]'",
    )
    p_scan.add_argument("prompt", nargs="?", default=None,
                        help="the prompt to scan")
    p_scan.add_argument("--model", default=None,
                        help="model name (default: google/gemma-2-2b-it)")
    p_scan.add_argument("--device", default=None,
                        help="device (default: cuda)")
    p_scan.add_argument("--trajectory", action="store_true",
                        help="generate tokens + measure S_early trajectory")
    p_scan.add_argument("--tokens", type=int, default=30,
                        help="max tokens for trajectory mode (default: 30)")
    p_scan.add_argument("--layers", action="store_true",
                        help="show full layer-by-layer profile")
    p_scan.add_argument("--json", action="store_true",
                        help="output raw JSON instead of the visual card")
    p_scan.add_argument("--compare", nargs="+", metavar="PROMPT",
                        help="compare K/C/S across multiple prompts")
    p_scan.add_argument("--batch", metavar="FILE",
                        help="batch scan from a JSONL file")
    p_scan.add_argument("--out", metavar="FILE",
                        help="output file for batch mode")
    p_scan.add_argument("--bridge", metavar="PROMPT",
                        help="run tier 0 + tier 2 side by side")
    p_scan.add_argument("--tier0-trajectory", metavar="FILE", dest="tier0_trajectory",
                        help="logprob trajectory JSON for bridge mode")
    p_scan.add_argument("--legacy", metavar="FILE",
                        help="read a pre-captured trajectory JSON (old behavior)")
    p_scan.set_defaults(func=cmd_scan)

    # antipatterns — 0.6.0 named failure mode detection
    p_anti = sub.add_parser(
        "antipatterns",
        help="detect named failure modes from your audit history (0.6.0)",
    )
    p_anti.add_argument(
        "--last-n", type=int, default=500, dest="last_n",
        help="number of audit entries to scan (default: 500)",
    )
    p_anti.add_argument(
        "--min", type=int, default=2, dest="min_occurrences",
        help="minimum occurrences to report (default: 2)",
    )
    p_anti.add_argument(
        "--format", choices=["ascii", "json"],
        default="ascii",
        help="output format (default: ascii)",
    )
    p_anti.set_defaults(func=cmd_antipatterns)

    # conversation — 0.5.9 conversation-level EKG
    p_conv = sub.add_parser(
        "conversation",
        help="cognitive EKG on a conversation message history (0.5.9)",
    )
    p_conv.add_argument(
        "file",
        help="path to JSON file containing messages",
    )
    p_conv.add_argument(
        "--format", choices=["ascii", "json"],
        default="ascii",
        help="output format (default: ascii)",
    )
    p_conv.set_defaults(func=cmd_conversation)

    # compare-agents — 0.6.0 population fingerprint comparison
    p_cmp_agents = sub.add_parser(
        "compare-agents",
        help="compare your fingerprint against the agent population (0.6.0)",
    )
    p_cmp_agents.add_argument(
        "--format", choices=["ascii", "json"],
        default="ascii",
        help="output format (default: ascii)",
    )
    p_cmp_agents.set_defaults(func=cmd_compare_agents)

    # attack — 7.0.0 inverse cognometry (adversarial seeds per instrument)
    p_attack = sub.add_parser(
        "attack",
        help="adversarial inputs that maximally trigger a cognometric instrument (7.0.0)",
    )
    p_attack.add_argument(
        "instrument", nargs="?", default=None,
        help="instrument name (sycophancy, loop, goal_drift, "
             "deception, plan_action, overconfidence)",
    )
    p_attack.add_argument(
        "--target", type=float, default=0.9,
        help="target risk score in [0, 1] (default: 0.9)",
    )
    p_attack.add_argument(
        "-n", type=int, default=10,
        help="max candidates returned (default: 10)",
    )
    p_attack.add_argument(
        "--corpus", default=None,
        help="override path to a jsonl corpus (default: bundled seeds)",
    )
    p_attack.add_argument(
        "--adversarial", action="store_true",
        help="mine NATURAL FALSE POSITIVES (label=0 rows that fool the "
             "detector), instead of the default training-distribution canary",
    )
    p_attack.add_argument(
        "--list", action="store_true",
        help="list available instruments and exit",
    )
    p_attack.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of the visual card",
    )
    p_attack.set_defaults(func=cmd_attack)

    # publish — opt-in dashboard telemetry
    p_publish = sub.add_parser(
        "publish",
        help="publish agent data to the remote dashboard (opt-in)",
    )
    p_publish.add_argument(
        "--name", type=str, required=True,
        help="agent name to publish under",
    )
    p_publish.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="print the payload JSON without sending",
    )
    p_publish.add_argument(
        "--endpoint", type=str,
        default="https://fathom.darkflobi.com/api/styxx-submit",
        help="custom endpoint URL (default: fathom.darkflobi.com)",
    )
    p_publish.set_defaults(func=cmd_publish)

    # claim — register an agent name on the live telemetry relay
    p_claim = sub.add_parser(
        "claim",
        help="claim an agent name and get your live dashboard URL",
    )
    p_claim.add_argument(
        "name", nargs="?",
        help="lowercase agent name (a-z, 0-9, - _). If omitted, prompts interactively.",
    )
    p_claim.add_argument(
        "--relay", help="override the relay URL (default: live.darkflobi.com)",
    )
    p_claim.set_defaults(func=cmd_claim)

    # feed — print the dashboard URL for the current agent
    p_feed = sub.add_parser(
        "feed",
        help="print the live dashboard URL for your claimed agent",
    )
    p_feed.add_argument("name", nargs="?", help="override agent name")
    p_feed.set_defaults(func=cmd_feed)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
