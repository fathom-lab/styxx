---
name: cognometric-audit-on-send
description: "Pre-flight cognometric audit on every outbound draft. Picks the cleanest of an audit→revise trajectory before the message ships."
homepage: https://github.com/fathom-lab/styxx
metadata:
  {
    "clawdbot":
      {
        "emoji": "🜂",
        "events": ["message:outbound:pre-send"],
        "requires":
          {
            "config": ["workspace.dir"],
            "python": [">=3.10", "styxx>=7.4.2"]
          },
        "install":
          [
            {
              "id": "bundled",
              "kind": "bundled",
              "label": "Bundled with clawdbot"
            }
          ]
      }
  }
---

# cognometric-audit-on-send

Pre-flight every outbound message draft through `styxx.cogn_audit_on_send`
before the channel layer sends it. If the audit fires, the LLM is given the
firing instruments + advice and asked to revise; the loop iterates up to
`max_revise` times. The shipped draft is the **cleanest of the trajectory**,
not necessarily the last — matching the v3-vs-v4 selection rule that the
darkflobi reflex loop demonstrated on 2026-05-18 (msg_id 34706).

## Event

`message:outbound:pre-send`

> ⚠️ Status: **not yet emitted by clawdbot core.** This HOOK.md and
> `handler.js` are committed against the planned event contract. They are
> a no-op until the clawdbot send-path PR lands.

### Expected event payload

```ts
type OutboundPreSendEvent = {
  type: 'message:outbound';
  action: 'pre-send';
  sessionKey: string;
  channel: string;          // 'telegram' | 'signal' | 'discord' | ...
  prompt: string;           // the user prompt (or conversation context)
                            // the model was responding to
  draft: string;            // the model's generated draft, BEFORE sending
  msg_id?: string;          // upstream message id when available
};
```

### Expected handler return contract

```ts
type HookResult =
  | undefined                                // no change, ship draft as-is
  | { draft: string }                        // ship this text instead
  | { draft: string; abort: false }          // explicit version of above
  | { abort: true; reason?: string };        // do not send anything
```

## What runs

The handler shells out to a small Python bridge that wraps
`styxx.cogn_audit_on_send(...)`. The bridge:

1. reads `{prompt, draft, ...}` from stdin as JSON,
2. calls `styxx.cogn_audit_on_send(...)` with `persist_to_chart=True` so
   each iteration is written to `chart.jsonl` for `recover_posture()`
   continuity,
3. returns `{shipped_draft, trajectory, fired, n_iterations}` on stdout.

The decision rule (cleanest-of-trajectory, with the climbing-composite
fallback) lives **inside** `styxx.middleware.cogn_audit_on_send`. The
handler does not duplicate it. The handler's only job is wiring.

## llm_revise wiring

`styxx.cogn_audit_on_send` accepts `llm_revise: Optional[ReviseFn]`. When
the bridge has access to the same model that produced the original draft
(via clawdbot's session-LLM hook), it injects a callable that re-prompts
the model with the audit advice. When it doesn't, `llm_revise=None` puts
the middleware in **audit-only mode**: the trajectory is logged, the
original draft ships unchanged, but the firing record is preserved for
the register-corpus regression fixtures (see
`tests/test_register_fixtures.py`).

## Logging

Every iteration of the audit loop appends a record to
`memory/cognometric-trajectory.jsonl` in the workspace:

```jsonl
{"ts": "...", "msg_id": "...", "drafts": [...], "best_version": 3,
 "shipped_v": 3, "styxx_version": "7.4.2", "channel": "telegram"}
```

The daily extractor at `styxx/scripts/extract_register_corpus.py`
anonymizes this into `tests/fixtures/register_corpus.jsonl` for the
regression test corpus.

## Failure modes (audit-first)

- **Bridge subprocess crash** → handler returns `undefined`. Original
  draft ships. The crash is logged but never blocks a send.
- **Audit timeout** (default 10s wall clock) → handler returns
  `undefined`. Original draft ships. Timeout is logged.
- **Climbing-composite degradation guard fires** → middleware ships the
  pre-revise draft; trajectory log records the guard event.
- **No `message:outbound:pre-send` event source yet** → handler never
  runs. No-op. No regression risk to the existing send path.

This handler is **fail-safe by construction**: when in doubt, do nothing.
The original send path is the source of truth.

## See also

- `styxx.cogn_audit_on_send` — the primitive (`styxx/middleware.py`)
- `styxx.preflight` — the per-draft audit primitive (7.4.2)
- `styxx.recover_posture` — cross-session posture recovery (7.4.2)
- `tests/test_register_fixtures.py` — regression corpus from real
  in-production agent self-audits
