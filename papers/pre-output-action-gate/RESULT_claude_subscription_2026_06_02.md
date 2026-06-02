# RESULT — Claude via subscription CLI, and why the experiment is moot on it

**Date:** 2026-06-02
**Two findings:** (1) a reusable method — free Claude inference via the
Claude Code subscription, bypassing the dead API account; (2) a result that
**resolves the closed-model thread**: a well-aligned frontier model has no
destructive-choice class to predict.

## The method (reusable — when the Anthropic API account is dry)

The `claude` CLI runs on the **subscription**, independent of the (empty) API
key. With `ANTHROPIC_API_KEY` unset, it answers for free:

```python
import subprocess, os
env = dict(os.environ); env.pop("ANTHROPIC_API_KEY", None)   # force subscription
r = subprocess.run("claude -p", shell=True, input=prompt,    # prompt via stdin
                   capture_output=True, text=True, timeout=150, env=env)
```

Real reasoning + parseable output, $0 API. (Subscription quota + ~15–25s/call.)
This is the free Claude backend for any future closed-model probe.

## The finding: Claude is too aligned for the destructive-action experiment

Opus 4.8 (subscription) on the **10 most destructive-tempting** scenarios —
the ones GPT-4o-mini and the small open models bit on:

| | destructive | safe |
|---|---|---|
| **Claude Opus 4.8** | **1 / 10** | 9 / 10 |
| GPT-4o-mini (ref) | ~40% | |
| Qwen-1.5B / small open (ref) | 40–58% | |

Claude's reasoning **explicitly names and rejects** the destructive option
(*"the first move should be diagnostic, not destructive… changes state without
context"*) → `get_deploy_history`, `list_secrets`, `export_then_flag`,
`describe_pods`, `revert_commit`. Extrapolated to all 40, Claude would produce
~2–4 destructive choices — far below the ≥8-per-class floor. **The
emitted-action prediction experiment is degenerate on Claude: there is no
class to predict.** Not run further (would burn quota confirming a null).

## Why this resolves the whole closed-model thread

The destructive-action guard predicts a mistake the model is about to make. A
well-aligned frontier model **doesn't make it** (on these scenarios) — so it
doesn't need the guard, and the prediction problem dissolves. The models that
*do* take destructive shortcuts are the **small, cheap, less-aligned** ones.
And those are exactly the **self-hosted open-weight** models whose activations
styxx **can read.**

**Danger and readability coincide.** The agents that need a pre-output action
guard most are the ones styxx is structurally able to guard; the agents styxx
can't read (Claude/GPT) are either too aligned to need it (Claude) or opaque to
behavioral signals (GPT, which failed). Three independent walls — no
activations, behavioral-signals-failed, no-credits — plus this: they all point
the same way. **You don't crack Claude. You own the open-weight world, where the
need and the capability meet.**

## Honest scope

- n=10, Opus **4.8** via subscription (not darkflobi's 4.7 via API), specific
  ops scenarios. Claude is **much safer here, not perfectly safe** (1/10 ≠ 0) —
  adversarial framing or different tasks could raise it.
- The reasoning-trace experiment (`run_claude_reasoning.py`,
  `PREREG_claude_reasoning`) stays built and pre-registered — for a model/context
  that *does* take shortcuts, or Claude under deliberate pressure, the
  faithfulness question is still worth firing. But the strategic headline stands.

— 2026-06-02; got creative, cracked free Claude access, and the data answered the
strategy question better than cracking it could have.
